#!/usr/bin/env python3
"""
Paper 1 — Orthogonalized Occupation Task Exposure Construction
==============================================================
Builds clean occupation-level task exposures separating:
  1. LLM_Writing      : Written-English LLM exposure (O*NET Written Comprehension/
                        Expression/Documentation/Interpretation x OpenAI E1+0.5*E2),
                        both raw standardized and residualized against Coding/Math.
  2. LLM_Coding_Math  : Quantitative/Programming LLM exposure (O*NET Math Reasoning/
                        Number Facility/Computers/Data Analysis/Technical Drafting x
                        OpenAI E1+0.5*E2), both raw and residualized against Writing.
  3. FaceToFace_Social: Interpersonal/Persuasion/Oral-Leadership tasks (O*NET
                        Selling/Negotiating/Public/Directing/Coaching/Relationships/
                        Oral Expression) — non-LLM interpersonal control/placebo.

Outputs (in papers/paper1_genai_immigrants/results/):
  - tasks_occsoc.parquet     (keyed to 6-digit OCCSOC for ACS)
  - tasks_occsoc4.parquet    (4-digit SOC fallback for IPUMS wildcard OCCSOCs)
  - tasks_occ2010.parquet    (keyed to OCC2010 via ACS PERWT crosswalk)
  - tasks_occ2018.parquet    (keyed to 2018 Census OCC_2018 via official U.S. Census Bureau
                              2018 Occupation Code List crosswalk, census_2018_occ_crosswalk.xlsx,
                              Sept 26, 2019 release, sheet '2018 Census Occ Code List', for CPS ORG)
  - task_validation_summary.json (top/bottom occupations & correlation matrix)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
AUX_DIR = ROOT / "data" / "aux"
ONET_DIR = AUX_DIR / "onet"
ACS_FILE = ROOT / "data" / "ipums" / "raw" / "acs_2007_2024" / "usa_00012.csv.gz"
OUT_DIR = ROOT / "papers" / "paper1_genai_immigrants" / "results"
CROSSWALK_2018_FILE = OUT_DIR / "census_2018_occ_crosswalk.xlsx"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# O*NET Element IDs (Importance scale 'IM', native range 1..5 -> rescaled to [0, 1])
WRITTEN_ENG_IDS = [
    "1.A.1.a.2",  # Written Comprehension
    "1.A.1.a.4",  # Written Expression
    "4.A.3.b.6",  # Documenting/Recording Information
    "4.A.4.a.1",  # Interpreting the Meaning of Information for Others
]

CODING_MATH_IDS = [
    "1.A.1.c.1",  # Mathematical Reasoning
    "1.A.1.c.2",  # Number Facility
    "4.A.3.b.1",  # Working with Computers
    "4.A.2.a.4",  # Analyzing Data or Information
    "4.A.3.b.2",  # Drafting, Laying Out, and Specifying Technical Devices
]

FACE_TO_FACE_IDS = [
    "1.A.1.a.3",  # Oral Expression
    "4.A.4.a.4",  # Establishing and Maintaining Interpersonal Relationships
    "4.A.4.a.6",  # Selling or Influencing Others
    "4.A.4.a.7",  # Resolving Conflicts and Negotiating with Others
    "4.A.4.a.8",  # Performing for or Working Directly with the Public
    "4.A.4.b.4",  # Guiding, Directing, and Motivating Subordinates
    "4.A.4.b.5",  # Coaching and Developing Others
]


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)


def residualize(y: pd.Series, x: pd.Series) -> pd.Series:
    """OLS residual of y on [1, x], then standardized to mean 0, SD 1."""
    mask = y.notna() & x.notna()
    out = pd.Series(np.nan, index=y.index, dtype=float)
    X = np.column_stack([np.ones(mask.sum()), x[mask].to_numpy(dtype=float)])
    yv = y[mask].to_numpy(dtype=float)
    beta = np.linalg.lstsq(X, yv, rcond=None)[0]
    resid = yv - X @ beta
    out.loc[mask] = (resid - resid.mean()) / resid.std(ddof=0)
    return out


def main() -> None:
    print("Loading O*NET 31.0 Abilities & Work Activities...")
    ab = pd.read_csv(
        ONET_DIR / "Abilities.txt",
        sep="\t",
        usecols=["O*NET-SOC Code", "Element ID", "Scale ID", "Data Value"],
    )
    wa = pd.read_csv(
        ONET_DIR / "Work Activities.txt",
        sep="\t",
        usecols=["O*NET-SOC Code", "Element ID", "Scale ID", "Data Value"],
    )
    onet = pd.concat([ab, wa], ignore_index=True)
    onet = onet[onet["Scale ID"] == "IM"].copy()

    # Collapse 8-digit O*NET-SOC (e.g. 15-1252.00) to 6-digit SOC without hyphen (151252)
    onet["occsoc"] = onet["O*NET-SOC Code"].str.slice(0, 7).str.replace("-", "", regex=False)
    onet["val01"] = (onet["Data Value"] - 1.0) / 4.0  # [0, 1] scale

    all_ids = set(WRITTEN_ENG_IDS + CODING_MATH_IDS + FACE_TO_FACE_IDS)
    onet_sub = onet[onet["Element ID"].isin(all_ids)]

    piv = (
        onet_sub.groupby(["occsoc", "Element ID"], as_index=False)["val01"]
        .mean()
        .pivot(index="occsoc", columns="Element ID", values="val01")
    )

    onet_df = pd.DataFrame(index=piv.index)
    onet_df["written_eng_01"] = piv[WRITTEN_ENG_IDS].mean(axis=1)
    onet_df["coding_math_01"] = piv[CODING_MATH_IDS].mean(axis=1)
    onet_df["face_to_face_01"] = piv[FACE_TO_FACE_IDS].mean(axis=1)
    onet_df = onet_df.reset_index()

    # Merge OpenAI GPTs-are-GPTs (E1 = alpha, E1+0.5*E2 = beta) & Felten AIOE/LM
    base_ai = pd.read_parquet(AUX_DIR / "occsoc_ai_onet_tasks.parquet")
    ai_cols = [
        "occsoc",
        "occ_title",
        "soc_major_2d",
        "openai_dv_alpha",
        "openai_dv_beta",
        "openai_human_alpha",
        "openai_human_beta",
        "felten_aioe",
        "felten_aioe_lm",
    ]
    df = onet_df.merge(base_ai[ai_cols], on="occsoc", how="inner")

    # Require non-missing OpenAI beta & O*NET scores
    df = df.dropna(subset=["openai_dv_beta", "written_eng_01", "coding_math_01", "face_to_face_01"]).copy()

    # Raw domain-interacted LLM exposures (both factors in [0, 1], so product in [0, 1])
    df["raw_llm_writing"] = df["openai_dv_beta"] * df["written_eng_01"]
    df["raw_llm_coding_math"] = df["openai_dv_beta"] * df["coding_math_01"]
    df["raw_llm_writing_e1"] = df["openai_dv_alpha"] * df["written_eng_01"]
    df["raw_llm_coding_e1"] = df["openai_dv_alpha"] * df["coding_math_01"]

    # Also construct pure Felten-LM x Written English for robustness
    felten_01 = (df["felten_aioe_lm"] - df["felten_aioe_lm"].min()) / (
        df["felten_aioe_lm"].max() - df["felten_aioe_lm"].min()
    )
    df["raw_felten_writing"] = felten_01 * df["written_eng_01"]

    # Standardized (z-score) versions
    df["llm_openai_beta"] = zscore(df["openai_dv_beta"])
    df["llm_openai_alpha"] = zscore(df["openai_dv_alpha"])
    df["llm_human_beta"] = zscore(df["openai_human_beta"])
    df["llm_felten_lm"] = zscore(df["felten_aioe_lm"])
    df["llm_writing_raw"] = zscore(df["raw_llm_writing"])
    df["llm_coding_math_raw"] = zscore(df["raw_llm_coding_math"])
    df["face_to_face_social"] = zscore(df["face_to_face_01"])
    df["written_eng_only"] = zscore(df["written_eng_01"])
    df["coding_math_only"] = zscore(df["coding_math_01"])

    # Orthogonalized indices:
    #   llm_writing_orth     : LLM_Writing residualized on LLM_Coding_Math (captures pure writing/drafting LLM exposure)
    #   llm_coding_math_orth : LLM_Coding_Math residualized on LLM_Writing (captures pure coding/quant LLM exposure)
    #   face_to_face_orth    : FaceToFace_Social residualized on LLM_Writing & LLM_Coding_Math
    df["llm_writing_orth"] = residualize(df["raw_llm_writing"], df["raw_llm_coding_math"])
    df["llm_coding_math_orth"] = residualize(df["raw_llm_coding_math"], df["raw_llm_writing"])

    # Also residualize FaceToFace_Social against both LLM indices (for clean placebo check)
    X_both = np.column_stack([
        np.ones(len(df)),
        df["raw_llm_writing"].to_numpy(),
        df["raw_llm_coding_math"].to_numpy(),
    ])
    f2f_y = df["face_to_face_01"].to_numpy()
    b_f2f = np.linalg.lstsq(X_both, f2f_y, rcond=None)[0]
    r_f2f = f2f_y - X_both @ b_f2f
    df["face_to_face_orth"] = (r_f2f - r_f2f.mean()) / r_f2f.std(ddof=0)

    # Save 6-digit OCCSOC table
    df.to_parquet(OUT_DIR / "tasks_occsoc.parquet", index=False)

    # Save 4-digit SOC prefix fallback table (for IPUMS wildcard OCCSOCs like '1191XX')
    num_cols = [c for c in df.columns if c not in ("occsoc", "occ_title", "soc_major_2d")]
    df4 = df.copy()
    df4["occsoc4"] = df4["occsoc"].str.slice(0, 4)
    df4_agg = df4.groupby("occsoc4", as_index=False)[num_cols].mean()
    df4_agg.to_parquet(OUT_DIR / "tasks_occsoc4.parquet", index=False)

    # Build exact employment-weighted OCC2010 crosswalk from ACS 2015-2024
    print("Building employment-weighted OCC2010 task table from ACS 2015-2024...")
    con = duckdb.connect()
    con.register("t6", df)
    con.register("t4", df4_agg)

    agg_exprs = ", ".join(
        [f"SUM(COALESCE(t6.{c}, t4.{c}) * w.wt) / SUM(w.wt) AS {c}" for c in num_cols]
    )
    occ2010_df = con.execute(f"""
        WITH acs_occ AS (
            SELECT
                CAST(OCC2010 AS INTEGER) AS occ2010,
                REGEXP_REPLACE(UPPER(TRIM(CAST(OCCSOC AS VARCHAR))), '[-\\s]', '', 'g') AS occsoc,
                SUM(PERWT) AS wt
            FROM read_csv_auto('{ACS_FILE}', header=true, types={{'OCCSOC': 'VARCHAR'}})
            WHERE YEAR >= 2015 AND EMPSTAT = 1 AND OCC2010 > 0 AND OCC2010 < 9920
            GROUP BY 1, 2
        )
        SELECT
            w.occ2010,
            SUM(w.wt) AS acs_emp_wt,
            {agg_exprs}
        FROM acs_occ w
        LEFT JOIN t6 ON w.occsoc = t6.occsoc
        LEFT JOIN t4 ON SUBSTR(w.occsoc, 1, 4) = t4.occsoc4
        WHERE COALESCE(t6.llm_writing_orth, t4.llm_writing_orth) IS NOT NULL
        GROUP BY w.occ2010
        ORDER BY w.occ2010
    """).df()

    # Re-standardize OCC2010 indices to mean 0, SD 1 across OCC2010 codes (for unit comparability)
    for col in [
        "llm_openai_beta", "llm_openai_alpha", "llm_human_beta", "llm_felten_lm",
        "llm_writing_raw", "llm_coding_math_raw", "face_to_face_social",
        "llm_writing_orth", "llm_coding_math_orth", "face_to_face_orth",
    ]:
        occ2010_df[col] = zscore(occ2010_df[col])

    occ2010_df.to_parquet(OUT_DIR / "tasks_occ2010.parquet", index=False)

    # Build 2018 Census OCC_2018 task table from official Census 2018 Occupation Code List
    # (census_2018_occ_crosswalk.xlsx, sheet '2018 Census Occ Code List', Sept 26, 2019 release)
    print("Building 2018 Census OCC_2018 task table from official Census crosswalk...")
    xl_2018 = pd.read_excel(
        CROSSWALK_2018_FILE,
        sheet_name="2018 Census Occ Code List",
        header=None,
    )
    cw_rows = []
    for _, r in xl_2018.iterrows():
        c2 = str(r[2]).strip() if pd.notna(r[2]) else ""
        c3 = str(r[3]).strip() if pd.notna(r[3]) else ""
        c1 = str(r[1]).strip() if pd.notna(r[1]) else ""
        if re.fullmatch(r"\d{4}", c2) and c3 and c3 != "nan":
            cw_rows.append((int(c2), c1, c3))
    cw_df = pd.DataFrame(cw_rows, columns=["occ2018", "occ2018_title", "soc_raw"])

    soc6_lookup = df.set_index("occsoc")[num_cols]
    soc4_lookup = df4_agg.set_index("occsoc4")[num_cols]

    occ2018_rows = []
    for _, row in cw_df.iterrows():
        s = row["soc_raw"].replace("-", "").upper()
        if s in soc6_lookup.index:
            vals = soc6_lookup.loc[s]
        elif len(s) >= 4 and s[:4] in soc4_lookup.index:
            vals = soc4_lookup.loc[s[:4]]
        else:
            continue
        rec = {
            "occ2018": int(row["occ2018"]),
            "occ2018_title": row["occ2018_title"],
            "soc_raw": row["soc_raw"],
        }
        for c in num_cols:
            rec[c] = float(vals[c])
        occ2018_rows.append(rec)

    occ2018_df = pd.DataFrame(occ2018_rows)
    # Re-standardize z-score columns to mean 0, SD 1 across the 564 matched 2018 Census occupations
    # (matching the exact 10 standardized columns re-scaled in tasks_occ2010.parquet)
    for col in [
        "llm_openai_beta", "llm_openai_alpha", "llm_human_beta", "llm_felten_lm",
        "llm_writing_raw", "llm_coding_math_raw", "face_to_face_social",
        "llm_writing_orth", "llm_coding_math_orth", "face_to_face_orth",
    ]:
        occ2018_df[col] = zscore(occ2018_df[col])

    occ2018_df.to_parquet(OUT_DIR / "tasks_occ2018.parquet", index=False)

    # Validation summary: top/bottom 10 occupations & correlations
    corr_cols = [
        "llm_openai_beta",
        "llm_writing_raw",
        "llm_coding_math_raw",
        "llm_writing_orth",
        "llm_coding_math_orth",
        "face_to_face_social",
    ]
    corr_mat = df[corr_cols].corr().round(3)

    summary = {
        "n_occsoc": int(len(df)),
        "n_occsoc4": int(len(df4_agg)),
        "n_occ2010": int(len(occ2010_df)),
        "n_occ2018": int(len(occ2018_df)),
        "correlations": corr_mat.to_dict(),
        "top10_llm_writing_orth": df.nlargest(10, "llm_writing_orth")[
            ["occsoc", "occ_title", "llm_writing_orth", "llm_coding_math_orth", "openai_dv_beta"]
        ].to_dict(orient="records"),
        "top10_llm_coding_math_orth": df.nlargest(10, "llm_coding_math_orth")[
            ["occsoc", "occ_title", "llm_coding_math_orth", "llm_writing_orth", "openai_dv_beta"]
        ].to_dict(orient="records"),
        "top10_face_to_face_social": df.nlargest(10, "face_to_face_social")[
            ["occsoc", "occ_title", "face_to_face_social", "llm_writing_orth", "openai_dv_beta"]
        ].to_dict(orient="records"),
    }
    (OUT_DIR / "task_validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nSaved {len(df)} OCCSOC, {len(df4_agg)} OCCSOC4, {len(occ2010_df)} OCC2010 task records.")
    print("\n--- Correlation Matrix (6-digit OCCSOC) ---")
    print(corr_mat.to_string())
    print("\n--- Top 10 LLM_Writing (Orthogonalized) ---")
    print(df.nlargest(10, "llm_writing_orth")[["occsoc", "occ_title", "llm_writing_orth", "openai_dv_beta"]].to_string(index=False))
    print("\n--- Top 10 LLM_Coding_Math (Orthogonalized) ---")
    print(df.nlargest(10, "llm_coding_math_orth")[["occsoc", "occ_title", "llm_coding_math_orth", "openai_dv_beta"]].to_string(index=False))
    print("\n--- Top 10 FaceToFace_Social ---")
    print(df.nlargest(10, "face_to_face_social")[["occsoc", "occ_title", "face_to_face_social", "openai_dv_beta"]].to_string(index=False))


if __name__ == "__main__":
    main()
