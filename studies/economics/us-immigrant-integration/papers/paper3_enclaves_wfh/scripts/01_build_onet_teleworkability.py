#!/usr/bin/env python3
"""
Paper 3 — Script 01: Construct & Validate Exact Dingel-Neiman (2020)
O*NET 31.0 Occupation-Level Teleworkability Index (`OCCSOC`).

Methodology:
  - Uses O*NET 31.0 `Work Activities.txt` (Importance scale, IM, 1-5) and `Occupation Data.txt`.
  - Following Dingel & Neiman (2020, JPubE), an occupation is classified as
    NON-teleworkable (dn_teleworkable = 0) if ANY of the 9 physical / outdoor /
    equipment-handling / mechanical-repair / in-person-care / direct-public
    Generalized Work Activities has Importance (IM) >= 4.0 (or >= threshold):
      4.A.3.a.1  Performing General Physical Activities
      4.A.3.a.2  Handling and Moving Objects
      4.A.3.a.3  Controlling Machines and Processes
      4.A.3.a.4  Operating Vehicles, Mechanized Devices, or Equipment
      4.A.1.b.2  Inspecting Equipment, Structures, or Materials
      4.A.3.b.4  Repairing and Maintaining Mechanical Equipment
      4.A.3.b.5  Repairing and Maintaining Electronic Equipment
      4.A.4.a.5  Assisting and Caring for Others
      4.A.4.a.8  Performing for or Working Directly with the Public
  - Also constructs a continuous standardized Teleworkability Score:
      telework_score = z(mean_cognitive_computer_IM) - z(mean_physical_inperson_IM)
    (standardized to mean 0, SD 1 across O*NET occupations).
  - Maps 8-digit O*NET-SOC codes (XX-XXXX.XX) -> 6-digit SOC (XXXXXX) and
    IPUMS OCCSOC wildcard codes (e.g., 1110XX, 15125X, 291141) via exact 6-digit,
    5-digit prefix, 4-digit prefix, and 3-digit prefix hierarchical matching.
  - Validates against realized 2021-2024 ACS WFH (`TRANWORK == 80`) rates!
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ONET_DIR = ROOT / "data" / "aux" / "onet"
ACS_PATH = ROOT / "data" / "ipums" / "raw" / "acs_2007_2024" / "usa_00012.csv.gz"
OUT_DIR = ROOT / "papers" / "paper3_enclaves_wfh" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Dingel-Neiman (2020) Work Activities disqualifiers (physical/equipment/public/care)
DISQUALIFIER_IDS = [
    "4.A.3.a.1",  # Performing General Physical Activities
    "4.A.3.a.2",  # Handling and Moving Objects
    "4.A.3.a.3",  # Controlling Machines and Processes
    "4.A.3.a.4",  # Operating Vehicles, Mechanized Devices, or Equipment
    "4.A.1.b.2",  # Inspecting Equipment, Structures, or Materials
    "4.A.3.b.4",  # Repairing and Maintaining Mechanical Equipment
    "4.A.3.b.5",  # Repairing and Maintaining Electronic Equipment
    "4.A.4.a.5",  # Assisting and Caring for Others
    "4.A.4.a.8",  # Performing for or Working Directly with the Public
]

# Cognitive / Computer / Information Work Activities (positive telework signal)
COGNITIVE_IDS = [
    "4.A.3.b.1",  # Working with Computers
    "4.A.2.a.2",  # Processing Information
    "4.A.2.a.4",  # Analyzing Data or Information
    "4.A.1.a.1",  # Getting Information
    "4.A.2.b.1",  # Making Decisions and Solving Problems
    "4.A.3.b.6",  # Documenting/Recording Information
    "4.A.4.c.1",  # Performing Administrative Activities
]


def build_onet_index() -> tuple[pd.DataFrame, pd.DataFrame]:
    wa = pd.read_csv(ONET_DIR / "Work Activities.txt", sep="\t")
    occ_meta = pd.read_csv(ONET_DIR / "Occupation Data.txt", sep="\t")[["O*NET-SOC Code", "Title"]]

    # Filter to Importance (IM) scale
    wa_im = wa[wa["Scale ID"] == "IM"].copy()
    wa_im["Data Value"] = pd.to_numeric(wa_im["Data Value"], errors="coerce")

    piv = wa_im.pivot_table(
        index="O*NET-SOC Code", columns="Element ID", values="Data Value", aggfunc="mean"
    ).reset_index()

    # Exact Dingel-Neiman disqualifier rule: max importance across physical/in-person activities >= 4.0
    piv["max_disq_im"] = piv[DISQUALIFIER_IDS].max(axis=1)
    piv["mean_disq_im"] = piv[DISQUALIFIER_IDS].mean(axis=1)
    piv["mean_cog_im"] = piv[COGNITIVE_IDS].mean(axis=1)
    piv["computer_im"] = piv["4.A.3.b.1"]

    # Binary Dingel-Neiman teleworkable indicator:
    # Cannot have any disqualifying physical/in-person activity >= 4.0, AND must use computers (>= 3.0)
    piv["dn_teleworkable"] = ((piv["max_disq_im"] < 4.0) & (piv["computer_im"] >= 3.0)).astype(float)

    # Continuous standardized teleworkability score
    z_cog = (piv["mean_cog_im"] - piv["mean_cog_im"].mean()) / piv["mean_cog_im"].std()
    z_disq = (piv["mean_disq_im"] - piv["mean_disq_im"].mean()) / piv["mean_disq_im"].std()
    raw_score = z_cog - z_disq
    piv["telework_score"] = (raw_score - raw_score.mean()) / raw_score.std()

    piv = piv.merge(occ_meta, on="O*NET-SOC Code", how="left")

    # Convert 8-digit O*NET-SOC (e.g. '15-1252.00') to 6-digit SOC ('151252')
    piv["soc6"] = piv["O*NET-SOC Code"].str.replace("-", "", regex=False).str.slice(0, 6)
    piv["soc5"] = piv["soc6"].str.slice(0, 5)
    piv["soc4"] = piv["soc6"].str.slice(0, 4)
    piv["soc3"] = piv["soc6"].str.slice(0, 3)
    piv["soc2"] = piv["soc6"].str.slice(0, 2)

    # Collapse 8-digit O*NET-SOC to 6-digit SOC
    soc6_df = (
        piv.groupby("soc6", as_index=False)
        .agg(
            dn_teleworkable=("dn_teleworkable", "mean"),
            telework_score=("telework_score", "mean"),
            max_disq_im=("max_disq_im", "mean"),
            computer_im=("computer_im", "mean"),
            title=("Title", "first"),
        )
    )
    return piv, soc6_df


def map_to_ipums_occsoc(piv: pd.DataFrame, soc6_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")

    # Pull all unique OCCSOC codes + realized 2021-2024 WFH rate + 2015-2019 WFH rate from ACS
    acs_occ = con.execute(f"""
        SELECT
            TRIM(CAST(OCCSOC AS VARCHAR)) AS occsoc,
            SUM(CASE WHEN YEAR BETWEEN 2021 AND 2024 THEN PERWT ELSE 0 END) AS wt_post,
            SUM(CASE WHEN YEAR BETWEEN 2021 AND 2024 AND TRANWORK = 80 THEN PERWT ELSE 0 END) /
                NULLIF(SUM(CASE WHEN YEAR BETWEEN 2021 AND 2024 AND TRANWORK > 0 THEN PERWT ELSE 0 END), 0) AS wfh_rate_post,
            SUM(CASE WHEN YEAR IN (2015, 2018, 2019) AND TRANWORK = 80 THEN PERWT ELSE 0 END) /
                NULLIF(SUM(CASE WHEN YEAR IN (2015, 2018, 2019) AND TRANWORK > 0 THEN PERWT ELSE 0 END), 0) AS wfh_rate_pre
        FROM read_csv_auto('{ACS_PATH}')
        WHERE AGE BETWEEN 18 AND 64 AND SCHOOL != 2 AND MET2013 > 0
          AND TRIM(CAST(OCCSOC AS VARCHAR)) NOT IN ('0', '', '000000')
        GROUP BY TRIM(CAST(OCCSOC AS VARCHAR))
    """).df()

    # Build prefix lookup tables from O*NET
    lu6 = soc6_df.set_index("soc6")[["dn_teleworkable", "telework_score", "title"]].to_dict("index")
    lu5 = piv.groupby("soc5")[["dn_teleworkable", "telework_score"]].mean().to_dict("index")
    lu4 = piv.groupby("soc4")[["dn_teleworkable", "telework_score"]].mean().to_dict("index")
    lu3 = piv.groupby("soc3")[["dn_teleworkable", "telework_score"]].mean().to_dict("index")
    lu2 = piv.groupby("soc2")[["dn_teleworkable", "telework_score"]].mean().to_dict("index")

    matched_rows = []
    match_levels = {"6-digit": 0, "5-digit": 0, "4-digit": 0, "3-digit": 0, "2-digit": 0, "unmatched": 0}
    for _, r in acs_occ.iterrows():
        code = str(r["occsoc"]).strip()
        clean = code.replace("-", "").replace(" ", "")
        # Strip trailing X/Y/0 wildcards for prefix matching
        digits_only = clean.rstrip("XY")
        val = None
        lvl = "unmatched"
        title = ""
        if len(clean) == 6 and clean in lu6:
            val = lu6[clean]
            title = val.get("title", "")
            lvl = "6-digit"
        elif len(digits_only) >= 5 and digits_only[:5] in lu5:
            val = lu5[digits_only[:5]]
            lvl = "5-digit"
        elif len(digits_only) >= 4 and digits_only[:4] in lu4:
            val = lu4[digits_only[:4]]
            lvl = "4-digit"
        elif len(digits_only) >= 3 and digits_only[:3] in lu3:
            val = lu3[digits_only[:3]]
            lvl = "3-digit"
        elif len(digits_only) >= 2 and digits_only[:2] in lu2:
            val = lu2[digits_only[:2]]
            lvl = "2-digit"

        match_levels[lvl] += 1
        matched_rows.append({
            "occsoc": code,
            "dn_teleworkable": float(val["dn_teleworkable"]) if val else np.nan,
            "dn_teleworkable_bin": float(val["dn_teleworkable"] >= 0.5) if val else np.nan,
            "telework_score": float(val["telework_score"]) if val else np.nan,
            "match_level": lvl,
            "title": title,
            "wt_post": float(r["wt_post"]),
            "wfh_rate_pre": float(r["wfh_rate_pre"]) if pd.notna(r["wfh_rate_pre"]) else np.nan,
            "wfh_rate_post": float(r["wfh_rate_post"]) if pd.notna(r["wfh_rate_post"]) else np.nan,
        })

    out_df = pd.DataFrame(matched_rows)
    out_df["wfh_jump"] = out_df["wfh_rate_post"] - out_df["wfh_rate_pre"]

    # Weighted validation stats
    val_sub = out_df.dropna(subset=["dn_teleworkable_bin", "wfh_rate_post", "wt_post"])
    val_sub = val_sub[val_sub["wt_post"] > 0]
    w = val_sub["wt_post"].to_numpy()

    def wcorr(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        mx = np.average(x, weights=weights)
        my = np.average(y, weights=weights)
        cov = np.average((x - mx) * (y - my), weights=weights)
        sx = np.sqrt(np.average((x - mx) ** 2, weights=weights))
        sy = np.sqrt(np.average((y - my) ** 2, weights=weights))
        return float(cov / (sx * sy))

    corr_bin_wfh = wcorr(val_sub["dn_teleworkable_bin"].to_numpy(), val_sub["wfh_rate_post"].to_numpy(), w)
    corr_score_wfh = wcorr(val_sub["telework_score"].to_numpy(), val_sub["wfh_rate_post"].to_numpy(), w)
    corr_score_jump = wcorr(
        val_sub.dropna(subset=["wfh_jump"])["telework_score"].to_numpy(),
        val_sub.dropna(subset=["wfh_jump"])["wfh_jump"].to_numpy(),
        val_sub.dropna(subset=["wfh_jump"])["wt_post"].to_numpy(),
    )

    tele_wfh_post = float(np.average(
        val_sub.loc[val_sub["dn_teleworkable_bin"] == 1, "wfh_rate_post"],
        weights=val_sub.loc[val_sub["dn_teleworkable_bin"] == 1, "wt_post"],
    ))
    nontele_wfh_post = float(np.average(
        val_sub.loc[val_sub["dn_teleworkable_bin"] == 0, "wfh_rate_post"],
        weights=val_sub.loc[val_sub["dn_teleworkable_bin"] == 0, "wt_post"],
    ))
    tele_share_emp = float(val_sub.loc[val_sub["dn_teleworkable_bin"] == 1, "wt_post"].sum() / val_sub["wt_post"].sum())

    # Top and bottom occupations by telework_score (among large occupations)
    large = out_df[out_df["wt_post"] > 500_000].dropna(subset=["telework_score"])
    top10 = large.sort_values("telework_score", ascending=False).head(10)[
        ["occsoc", "title", "dn_teleworkable_bin", "telework_score", "wfh_rate_pre", "wfh_rate_post"]
    ].to_dict(orient="records")
    bot10 = large.sort_values("telework_score", ascending=True).head(10)[
        ["occsoc", "title", "dn_teleworkable_bin", "telework_score", "wfh_rate_pre", "wfh_rate_post"]
    ].to_dict(orient="records")

    summary = {
        "n_onet_occupations": int(len(piv)),
        "n_acs_occsoc": int(len(out_df)),
        "match_levels": match_levels,
        "emp_weighted_teleworkable_share": tele_share_emp,
        "wfh_post_teleworkable": tele_wfh_post,
        "wfh_post_nonteleworkable": nontele_wfh_post,
        "wfh_ratio_tele_to_nontele": tele_wfh_post / max(nontele_wfh_post, 1e-6),
        "weighted_corr_bin_vs_wfh_post": corr_bin_wfh,
        "weighted_corr_score_vs_wfh_post": corr_score_wfh,
        "weighted_corr_score_vs_wfh_jump": corr_score_jump,
        "top10_occupations": top10,
        "bottom10_occupations": bot10,
    }
    return out_df, summary


def main() -> None:
    print("[01] Building exact Dingel-Neiman (2020) O*NET 31.0 Teleworkability Index...")
    piv, soc6_df = build_onet_index()
    print(f"   O*NET 8-digit occupations: {len(piv):,}, 6-digit SOCs: {len(soc6_df):,}")
    out_df, summary = map_to_ipums_occsoc(piv, soc6_df)

    out_df.to_parquet(OUT_DIR / "occsoc_teleworkability.parquet", index=False)
    out_df.to_csv(OUT_DIR / "occsoc_teleworkability.csv", index=False)
    (OUT_DIR / "onet_teleworkability_validation.json").write_text(json.dumps(summary, indent=2))

    print(f"   ACS OCCSOC codes mapped: {summary['n_acs_occsoc']:,} ({summary['match_levels']})")
    print(f"   Employment-weighted share teleworkable: {summary['emp_weighted_teleworkable_share']:.3f}")
    print(f"   Realized 2021-2024 WFH rate: Teleworkable = {summary['wfh_post_teleworkable']:.3f} vs. Non-Teleworkable = {summary['wfh_post_nonteleworkable']:.3f} ({summary['wfh_ratio_tele_to_nontele']:.2f}x)")
    print(f"   Weighted corr(TeleworkScore, WFH_2021_2024) = {summary['weighted_corr_score_vs_wfh_post']:.3f}, corr(Score, WFH_Jump) = {summary['weighted_corr_score_vs_wfh_jump']:.3f}")
    print("   Top 5 teleworkable large occupations:")
    for r in summary["top10_occupations"][:5]:
        print(f"      {r['occsoc']} | score={r['telework_score']:+.2f} | WFH: {r['wfh_rate_pre']:.2%} -> {r['wfh_rate_post']:.2%} | {r['title']}")
    print("   Bottom 5 teleworkable large occupations:")
    for r in summary["bottom10_occupations"][:5]:
        print(f"      {r['occsoc']} | score={r['telework_score']:+.2f} | WFH: {r['wfh_rate_pre']:.2%} -> {r['wfh_rate_post']:.2%} | {r['title']}")


if __name__ == "__main__":
    main()
