#!/usr/bin/env python3
"""
Paper 3 — Script 04 (Contribution 3):
Origin -> Destination Migration Matrix: "Enclave Arbitrage" vs. "Enclave Abandonment"
(MIGMET131, 2015-2024).

Restricts to inter-MSA native co-ethnic movers (orig_met != dest_met), merges origin &
destination Zillow ZORI, ZHVI, and co-ethnic immigrant shares, and tests whether
post-2020 remote-capable movers leaving expensive gateway enclaves:
  - Abandoned enclaves (moved to low co-ethnic share metros), OR
  - Engaged in "Enclave Arbitrage" (moved to lower-ZORI / lower-ZHVI *secondary
    Sunbelt enclaves* like Houston, Dallas, Atlanta, Austin, Phoenix, Raleigh, Charlotte).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyfixest as pf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from essay3_analysis import build_ancestr_sql_case, build_bpl_sql_case, extract_pf_row, fwl_demean  # noqa: E402

ACS_PATH = ROOT / "data" / "ipums" / "raw" / "acs_2007_2024" / "usa_00012.csv.gz"
HOUSING_PARQUET = ROOT / "data" / "aux" / "clean_metro_housing.parquet"
TELE_PARQUET = ROOT / "papers" / "paper3_enclaves_wfh" / "results" / "occsoc_teleworkability.parquet"
OUT_DIR = ROOT / "papers" / "paper3_enclaves_wfh" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GATEWAY_MSAS_TOP5 = {
    35620: "New York",
    31080: "Los Angeles",
    41860: "San Francisco",
    16980: "Chicago",
    33100: "Miami",
}

GATEWAY_MSAS = {
    **GATEWAY_MSAS_TOP5,
    47900: "Washington DC",
    14460: "Boston",
    42660: "Seattle",
}

GATEWAY_MSAS_TOP12 = {
    **GATEWAY_MSAS,
    26420: "Houston",
    19100: "Dallas-Fort Worth",
    41740: "San Diego",
    41940: "San Jose",
}

WEBB6_WEIGHTS = np.array(
    [-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)],
    dtype=float,
)


def wild_cluster_bootstrap_webb(
    df: pd.DataFrame,
    y_col: str,
    treat_col: str,
    ctrl_cols: list[str],
    fe_cols: list[str],
    cluster_col: str,
    weight_col: str,
    n_boot: int = 9999,
    seed: int = 20261124,
) -> dict[str, float]:
    """
    Restricted Wild Cluster Bootstrap-t (Cameron, Gelbach & Miller 2008; Webb 2023)
    with Webb 6-point weights (recommended for G < 11 clusters, e.g., G = 8 gateways).
    Uses exact FWL-demeaned cluster-score algebra (Kline & Santos 2012 / Roodman et al. 2019).
    """
    needed = list(dict.fromkeys([y_col, treat_col, *ctrl_cols, *fe_cols, cluster_col, weight_col]))
    sub = df[needed].dropna().copy()
    n = len(sub)
    if n < 20:
        return {"wcb_pval": np.nan, "n_clusters": 0}

    w = np.clip(sub[weight_col].to_numpy(dtype=float), 1e-12, None)
    sw = np.sqrt(w)
    vars_all = [y_col, treat_col, *ctrl_cols]
    dem = fwl_demean(sub, vars_all, fe_cols, w)

    y = dem[:, 0] * sw
    x1 = dem[:, 1] * sw
    X2 = dem[:, 2:] * sw[:, None]

    # Partial out X2 from both y and x1 (exact FWL)
    if X2.shape[1] > 0:
        XtX2_inv = np.linalg.pinv(X2.T @ X2)
        y_tilde = y - X2 @ (XtX2_inv @ (X2.T @ y))
        x1_tilde = x1 - X2 @ (XtX2_inv @ (X2.T @ x1))
    else:
        y_tilde = y
        x1_tilde = x1

    xx = float(x1_tilde @ x1_tilde)
    if xx <= 1e-16:
        return {"wcb_pval": np.nan, "n_clusters": 0}

    # Unrestricted OLS estimate and cluster CRV1 t-stat
    beta_hat = float((x1_tilde @ y_tilde) / xx)
    e_unres = y_tilde - beta_hat * x1_tilde

    codes = pd.Categorical(sub[cluster_col]).codes
    G = int(codes.max()) + 1
    if G < 2:
        return {"wcb_pval": np.nan, "n_clusters": G}

    # Observed CRV1 t-stat
    scores_unres = np.bincount(codes, weights=x1_tilde * e_unres, minlength=G)
    dof_g = G / (G - 1.0)
    se_obs = np.sqrt(dof_g * float(np.sum(scores_unres ** 2))) / xx
    if se_obs <= 1e-16:
        return {"wcb_pval": np.nan, "n_clusters": G}
    t_obs = abs(beta_hat / se_obs)

    # Restricted residuals under H0: beta1 = 0 are simply y_tilde (since X2 & FEs already partialled out!)
    # Cluster-level restricted scores S_g and cluster Hessians H_g
    S_g = np.bincount(codes, weights=x1_tilde * y_tilde, minlength=G)
    H_g = np.bincount(codes, weights=x1_tilde * x1_tilde, minlength=G)

    rng = np.random.default_rng(seed)
    V = rng.choice(WEBB6_WEIGHTS, size=(n_boot, G), replace=True)

    # Vectorized bootstrap beta* and CRV1 SE* across all B=9,999 draws
    num_b = V @ S_g  # shape (B,)
    beta_b = num_b / xx
    # Cluster bootstrap scores: S*_bg = v_bg * S_g - beta*_b * H_g
    scores_b = V * S_g[None, :] - beta_b[:, None] * H_g[None, :]
    se_b = np.sqrt(np.maximum(dof_g * np.sum(scores_b ** 2, axis=1), 1e-32)) / xx
    t_b = np.abs(beta_b / se_b)

    wcb_pval = float((1 + np.sum(t_b >= t_obs)) / (n_boot + 1))
    return {"wcb_pval": wcb_pval, "n_clusters": G, "t_obs_fwl": float(t_obs)}

SUNBELT_SECONDARY_ENCLAVES = {
    26420: "Houston",
    19100: "Dallas-Fort Worth",
    12060: "Atlanta",
    12420: "Austin",
    38060: "Phoenix",
    39580: "Raleigh",
    16740: "Charlotte",
    45300: "Tampa",
    36740: "Orlando",
    29820: "Las Vegas",
    41700: "San Antonio",
    34980: "Nashville",
    19740: "Denver",
}


def run_module3() -> None:
    print("[04] Extracting inter-MSA native co-ethnic movers (MIGMET131, 2015-2024)...")
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")

    bpl_case = build_bpl_sql_case()
    anc1_case = build_ancestr_sql_case("ANCESTR1")
    anc2_case = build_ancestr_sql_case("ANCESTR2")

    con.execute(f"CREATE TEMP TABLE tele AS SELECT occsoc, dn_teleworkable_bin AS tele_bin, telework_score FROM read_parquet('{TELE_PARQUET}') WHERE dn_teleworkable_bin IS NOT NULL")

    # Compute MSA x eth co-ethnic shares (pre-COVID 2015-2019 average AND annual)
    con.execute(f"""
        CREATE TEMP VIEW acs_all AS
        SELECT
            a.YEAR,
            a.MET2013 AS dest_met,
            a.MIGMET131 AS orig_met,
            a.MIGRATE1,
            CASE WHEN a.BPL >= 100 THEN 1 ELSE 0 END AS is_immigrant,
            CASE
                WHEN a.BPL >= 100 THEN COALESCE({bpl_case}, {anc1_case})
                ELSE COALESCE({anc1_case}, {anc2_case})
            END AS eth,
            CAST(a.PERWT AS DOUBLE) AS PERWT,
            CAST(a.AGE AS DOUBLE) AS AGE,
            a.SEX,
            CASE WHEN a.EDUC BETWEEN 1 AND 11 THEN CAST(a.EDUC AS DOUBLE) ELSE NULL END AS EDUC,
            CASE WHEN a.TRANWORK = 80 THEN 1.0 ELSE 0.0 END AS is_wfh,
            t.tele_bin,
            t.telework_score
        FROM read_csv_auto('{ACS_PATH}') a
        LEFT JOIN tele t ON TRIM(CAST(a.OCCSOC AS VARCHAR)) = t.occsoc
        WHERE a.YEAR IN (2015, 2018, 2019, 2021, 2022, 2023, 2024)
          AND a.AGE BETWEEN 18 AND 64 AND a.SCHOOL != 2 AND a.MET2013 > 0
    """)

    # Co-ethnic immigrant share in each (eth, met2013) averaged over 2015-2019 (pre-determined)
    enclave_pre = con.execute("""
        WITH msa_p AS (
            SELECT dest_met AS met2013, SUM(PERWT) AS mp
            FROM acs_all WHERE YEAR IN (2015, 2018, 2019) GROUP BY dest_met
        ),
        imm_p AS (
            SELECT eth, dest_met AS met2013, SUM(PERWT) AS ip
            FROM acs_all WHERE YEAR IN (2015, 2018, 2019) AND is_immigrant = 1 AND eth IS NOT NULL
            GROUP BY eth, dest_met
        )
        SELECT i.eth, i.met2013, 100.0 * i.ip / m.mp AS coeth_share_pre
        FROM imm_p i JOIN msa_p m ON i.met2013 = m.met2013
    """).df()

    # Pull all inter-MSA native co-ethnic movers
    movers = con.execute("""
        SELECT
            YEAR,
            eth,
            orig_met,
            dest_met,
            PERWT,
            AGE,
            SEX,
            EDUC,
            is_wfh,
            tele_bin,
            telework_score
        FROM acs_all
        WHERE is_immigrant = 0
          AND eth IS NOT NULL
          AND MIGRATE1 IN (2, 3)
          AND orig_met >= 10000
          AND dest_met >= 10000
          AND orig_met != dest_met
          AND tele_bin IS NOT NULL
    """).df()

    print(f"   Inter-MSA co-ethnic native movers (2015-2024): {len(movers):,} rows (weighted={movers['PERWT'].sum():,.0f})")

    # Merge Zillow ZORI & ZHVI (2019 fixed baseline AND contemporaneous year)
    housing = pd.read_parquet(HOUSING_PARQUET)
    h19 = housing[housing["year"] == 2019][["met2013", "ln_zori", "ln_zhvi", "cbsa_label"]].drop_duplicates("met2013")
    h_yr = housing[["met2013", "year", "ln_zori", "ln_zhvi"]].rename(columns={"year": "YEAR"})

    movers = (
        movers
        .merge(h_yr.rename(columns={"met2013": "orig_met", "ln_zori": "orig_ln_zori", "ln_zhvi": "orig_ln_zhvi"}), on=["orig_met", "YEAR"], how="left")
        .merge(h_yr.rename(columns={"met2013": "dest_met", "ln_zori": "dest_ln_zori", "ln_zhvi": "dest_ln_zhvi"}), on=["dest_met", "YEAR"], how="left")
        .merge(h19[["met2013", "ln_zori", "ln_zhvi"]].rename(columns={"met2013": "orig_met", "ln_zori": "orig_ln_zori19", "ln_zhvi": "orig_ln_zhvi19"}), on="orig_met", how="left")
        .merge(h19[["met2013", "ln_zori", "ln_zhvi"]].rename(columns={"met2013": "dest_met", "ln_zori": "dest_ln_zori19", "ln_zhvi": "dest_ln_zhvi19"}), on="dest_met", how="left")
        .merge(enclave_pre.rename(columns={"met2013": "orig_met", "coeth_share_pre": "orig_share"}), on=["eth", "orig_met"], how="left")
        .merge(enclave_pre.rename(columns={"met2013": "dest_met", "coeth_share_pre": "dest_share"}), on=["eth", "dest_met"], how="left")
    )
    movers["orig_share"] = movers["orig_share"].fillna(0.0)
    movers["dest_share"] = movers["dest_share"].fillna(0.0)

    # Deltas (Destination - Origin)
    movers["d_ln_zori"] = movers["dest_ln_zori"] - movers["orig_ln_zori"]
    movers["d_ln_zhvi"] = movers["dest_ln_zhvi"] - movers["orig_ln_zhvi"]
    movers["d_ln_zori19"] = movers["dest_ln_zori19"] - movers["orig_ln_zori19"]
    movers["d_coeth_share"] = movers["dest_share"] - movers["orig_share"]
    movers["d_ln_coeth_share"] = np.log(movers["dest_share"] + 0.1) - np.log(movers["orig_share"] + 0.1)

    movers["post"] = (movers["YEAR"] >= 2021).astype(float)
    movers["from_gateway"] = movers["orig_met"].isin(GATEWAY_MSAS).astype(int)
    movers["to_gateway"] = movers["dest_met"].isin(GATEWAY_MSAS).astype(int)
    movers["to_sunbelt_enclave"] = movers["dest_met"].isin(SUNBELT_SECONDARY_ENCLAVES).astype(float)

    # Define mutually exclusive Enclave Arbitrage vs. Enclave Abandonment (using 2019 baseline ZORI)
    # Arbitrage: moved to >=10% cheaper metro (d_ln_zori19 < -0.10) WITH meaningful co-ethnic presence (dest_share >= 0.5 pp)
    # Abandonment: moved to >=10% cheaper metro (d_ln_zori19 < -0.10) WITH minimal co-ethnic presence (dest_share < 0.5 pp)
    movers["cheaper_metro"] = (movers["d_ln_zori19"] < -0.10).astype(float)
    movers["enclave_arbitrage"] = ((movers["d_ln_zori19"] < -0.10) & (movers["dest_share"] >= 0.5)).astype(float)
    movers["enclave_abandonment"] = ((movers["d_ln_zori19"] < -0.10) & (movers["dest_share"] < 0.5)).astype(float)
    movers["high_orig_enclave"] = (movers["orig_share"] >= 1.0).astype(int)

    movers["tele_post"] = movers["tele_bin"] * movers["post"]
    movers["wfh_post"] = movers["is_wfh"] * movers["post"]
    movers["eth_year"] = movers["eth"].astype(str) + "_" + movers["YEAR"].astype(str)

    # ------------------------------------------------------------------
    # Regressions: Teleworkable x Post2020 AND WFH x Post2020 on Mover Deltas
    # ------------------------------------------------------------------
    reg_rows = []
    outcomes = [
        ("d_ln_zori", "Delta_ln_ZORI"),
        ("d_ln_zhvi", "Delta_ln_ZHVI"),
        ("d_coeth_share", "Delta_CoEthShare_pp"),
        ("d_ln_coeth_share", "Delta_ln_CoEthShare"),
        ("to_sunbelt_enclave", "Prob_SunbeltSecondaryEnclave"),
        ("enclave_arbitrage", "Prob_EnclaveArbitrage"),
        ("enclave_abandonment", "Prob_EnclaveAbandonment"),
    ]

    for sample_tag, df_s in [
        ("AllMovers", movers),
        ("GatewayExits", movers[movers["from_gateway"] == 1]),
        ("HighEnclaveGatewayExits", movers[(movers["from_gateway"] == 1) & (movers["high_orig_enclave"] == 1)]),
    ]:
        for ycol, ylbl in outcomes:
            sub = df_s.dropna(subset=[ycol, "tele_bin", "post", "AGE", "EDUC", "PERWT"])
            # 1. Teleworkable x Post2020 (descriptive occupational teleworkability)
            m_t = pf.feols(
                f"{ycol} ~ tele_post + tele_bin + AGE + EDUC + SEX | eth_year + orig_met",
                data=sub, weights="PERWT", vcov={"CRV1": "orig_met"},
            )
            wcb_t = wild_cluster_bootstrap_webb(
                sub, ycol, "tele_post", ["tele_bin", "AGE", "EDUC", "SEX"],
                ["eth_year", "orig_met"], "orig_met", "PERWT", n_boot=9999,
            )
            reg_rows.append(extract_pf_row(
                m_t, "tele_post", f"{sample_tag}_ITT_TeleXPost_{ylbl}",
                {"sample": sample_tag, "treatment": "Teleworkable_x_Post",
                 "n_clusters": wcb_t["n_clusters"], "wcb_pval": wcb_t["wcb_pval"]},
            ))

            # 2. Realized WFH x Post2020
            m_w = pf.feols(
                f"{ycol} ~ wfh_post + is_wfh + AGE + EDUC + SEX | eth_year + orig_met",
                data=sub, weights="PERWT", vcov={"CRV1": "orig_met"},
            )
            wcb_w = wild_cluster_bootstrap_webb(
                sub, ycol, "wfh_post", ["is_wfh", "AGE", "EDUC", "SEX"],
                ["eth_year", "orig_met"], "orig_met", "PERWT", n_boot=9999,
            )
            reg_rows.append(extract_pf_row(
                m_w, "wfh_post", f"{sample_tag}_Realized_WFHXPost_{ylbl}",
                {"sample": sample_tag, "treatment": "WFH_x_Post",
                 "n_clusters": wcb_w["n_clusters"], "wcb_pval": wcb_w["wcb_pval"]},
            ))

    # ------------------------------------------------------------------
    # Pre-Specified Sensitivity Analysis Grid:
    # (a) Gateway Set (Top 5, Top 8, Top 12)
    # (b) Rent-Savings Threshold (5%, 10%, 15%)
    # (c) Co-Ethnic Destination Threshold (0.25 pp, 0.50 pp, 1.00 pp)
    # ------------------------------------------------------------------
    print("   [Sensitivity] Running pre-specified gateway / rent / co-ethnic threshold sensitivity grid with Webb WCB...")
    sens_rows = []
    gw_sets = [
        ("Top5", GATEWAY_MSAS_TOP5),
        ("Top8", GATEWAY_MSAS),
        ("Top12", GATEWAY_MSAS_TOP12),
    ]
    # Axis 1: Vary Gateway Set (holding rent=10%, coeth=0.50pp)
    # Axis 2: Vary Rent Threshold 5%, 10%, 15% (holding Top8, coeth=0.50pp)
    # Axis 3: Vary Co-Ethnic Threshold 0.25pp, 0.50pp, 1.00pp (holding Top8, rent=10%)
    grid_configs = [
        ("GatewaySet", "Top5", GATEWAY_MSAS_TOP5, 0.10, 0.50),
        ("GatewaySet", "Top8 (Baseline)", GATEWAY_MSAS, 0.10, 0.50),
        ("GatewaySet", "Top12", GATEWAY_MSAS_TOP12, 0.10, 0.50),
        ("RentThreshold", "5% (d_ln_zori < -0.05)", GATEWAY_MSAS, 0.05, 0.50),
        ("RentThreshold", "10% (Baseline)", GATEWAY_MSAS, 0.10, 0.50),
        ("RentThreshold", "15% (d_ln_zori < -0.15)", GATEWAY_MSAS, 0.15, 0.50),
        ("CoEthThreshold", "0.25 pp", GATEWAY_MSAS, 0.10, 0.25),
        ("CoEthThreshold", "0.50 pp (Baseline)", GATEWAY_MSAS, 0.10, 0.50),
        ("CoEthThreshold", "1.00 pp", GATEWAY_MSAS, 0.10, 1.00),
    ]
    for axis, cfg_lbl, gw_dict, rent_cut, eth_cut in grid_configs:
        df_g = movers[movers["orig_met"].isin(gw_dict)].copy()
        df_g["arb_tmp"] = ((df_g["d_ln_zori19"] < -rent_cut) & (df_g["dest_share"] >= eth_cut)).astype(float)
        df_g["abn_tmp"] = ((df_g["d_ln_zori19"] < -rent_cut) & (df_g["dest_share"] < eth_cut)).astype(float)
        for sample_tag, df_sub in [
            ("GatewayExits", df_g),
            ("HighEnclaveGatewayExits", df_g[df_g["high_orig_enclave"] == 1]),
        ]:
            for ycol, ylbl in [("arb_tmp", "Prob_EnclaveArbitrage"), ("abn_tmp", "Prob_EnclaveAbandonment")]:
                sub = df_sub.dropna(subset=[ycol, "tele_bin", "post", "AGE", "EDUC", "PERWT"])
                for tr_lbl, tr_var, base_var in [("Tele_x_Post", "tele_post", "tele_bin"), ("WFH_x_Post", "wfh_post", "is_wfh")]:
                    m_s = pf.feols(
                        f"{ycol} ~ {tr_var} + {base_var} + AGE + EDUC + SEX | eth_year + orig_met",
                        data=sub, weights="PERWT", vcov={"CRV1": "orig_met"},
                    )
                    wcb_s = wild_cluster_bootstrap_webb(
                        sub, ycol, tr_var, [base_var, "AGE", "EDUC", "SEX"],
                        ["eth_year", "orig_met"], "orig_met", "PERWT", n_boot=9999,
                    )
                    sens_rows.append(extract_pf_row(
                        m_s, tr_var, f"{axis}_{cfg_lbl}_{sample_tag}_{tr_lbl}_{ylbl}",
                        {
                            "axis": axis, "config": cfg_lbl, "sample": sample_tag,
                            "treatment": tr_lbl, "outcome": ylbl,
                            "n_clusters": wcb_s["n_clusters"], "wcb_pval": wcb_s["wcb_pval"],
                        },
                    ))
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(OUT_DIR / "module3_sensitivity_grid.csv", index=False)

    # ------------------------------------------------------------------
    # Descriptive Cell Means for Figure 3 (Pre 2015-19 vs Post 2021-24)
    # ------------------------------------------------------------------
    def wmean(g: pd.DataFrame, col: str) -> float:
        s = g.dropna(subset=[col, "PERWT"])
        return float(np.average(s[col], weights=s["PERWT"])) if len(s) > 0 else np.nan

    desc_rows = []
    for sample_tag, df_s in [
        ("AllMovers", movers),
        ("GatewayExits", movers[movers["from_gateway"] == 1]),
        ("HighEnclaveGatewayExits", movers[(movers["from_gateway"] == 1) & (movers["high_orig_enclave"] == 1)]),
    ]:
        for post_val, period_lbl in [(0.0, "Pre (2015-2019)"), (1.0, "Post (2021-2024)")]:
            for tele_val, tele_lbl in [(0.0, "Non-Teleworkable"), (1.0, "Teleworkable")]:
                g = df_s[(df_s["post"] == post_val) & (df_s["tele_bin"] == tele_val)]
                desc_rows.append({
                    "sample": sample_tag,
                    "period": period_lbl,
                    "group": tele_lbl,
                    "n_raw": int(len(g)),
                    "wt_pop": float(g["PERWT"].sum()),
                    "mean_d_ln_zori": wmean(g, "d_ln_zori"),
                    "mean_d_ln_zhvi": wmean(g, "d_ln_zhvi"),
                    "mean_d_coeth_share": wmean(g, "d_coeth_share"),
                    "mean_dest_share": wmean(g, "dest_share"),
                    "share_sunbelt": wmean(g, "to_sunbelt_enclave"),
                    "share_arbitrage": wmean(g, "enclave_arbitrage"),
                    "share_abandonment": wmean(g, "enclave_abandonment"),
                })

    # Top Gateway -> Destination flows post-2020 for teleworkable movers
    gw_post_tele = movers[(movers["from_gateway"] == 1) & (movers["post"] == 1) & (movers["tele_bin"] == 1)]
    top_dest = (
        gw_post_tele.groupby("dest_met", as_index=False)
        .agg(wt_movers=("PERWT", "sum"), mean_d_zori=("d_ln_zori", "mean"), mean_dest_share=("dest_share", "mean"))
        .merge(h19[["met2013", "cbsa_label"]], left_on="dest_met", right_on="met2013", how="left")
        .sort_values("wt_movers", ascending=False)
        .head(15)
    )

    reg_df = pd.DataFrame(reg_rows)
    desc_df = pd.DataFrame(desc_rows)
    reg_df.to_csv(OUT_DIR / "module3_enclave_arbitrage_regs.csv", index=False)
    desc_df.to_csv(OUT_DIR / "module3_enclave_arbitrage_means.csv", index=False)
    top_dest.to_csv(OUT_DIR / "module3_top_gateway_exit_destinations.csv", index=False)
    (OUT_DIR / "module3_summary.json").write_text(json.dumps({
        "n_movers": int(len(movers)),
        "n_gateway_exits": int((movers["from_gateway"] == 1).sum()),
        "regressions": reg_rows,
        "means": desc_rows,
        "top_destinations": top_dest.to_dict(orient="records"),
    }, indent=2, default=str))

    print("\n=== MODULE 3: ENCLAVE ARBITRAGE VS. ABANDONMENT (DiD on Inter-MSA Movers + Webb WCB) ===")
    for _, r in reg_df.iterrows():
        sig = "***" if r["pval"] < 0.01 else ("**" if r["pval"] < 0.05 else ("*" if r["pval"] < 0.10 else ""))
        print(
            f"  {r['spec']:54s}: coef={r['coef']:+.5f} ({r['se']:.5f}){sig:3s} | "
            f"p_asym={r['pval']:.3f} | p_wcb={r['wcb_pval']:.3f} (G={int(r['n_clusters'])}) | N={int(r['n']):,}"
        )

    print("\n=== GATEWAY EXITS DESCRIPTIVE MEANS ===")
    print(desc_df[desc_df["sample"] == "GatewayExits"][["period", "group", "mean_d_ln_zori", "mean_d_coeth_share", "share_sunbelt", "share_arbitrage", "share_abandonment"]].to_string(index=False))


if __name__ == "__main__":
    run_module3()
