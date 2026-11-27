#!/usr/bin/env python3
"""
Paper 3 — Script 03 (Contribution 2):
Intent-to-Treat Remote Work Natural Experiment:
Dingel-Neiman (2020) Teleworkability DiD & Dynamic Event Study (2015-2024, ref=2019).

Uses pre-determined O*NET 31.0 OCCSOC Teleworkability x Pre-COVID Origin Enclave Share x Year
to test how remote-work capability transformed:
  (a) True Origin-MSA Retention (MIGMET131 == MET2013),
  (b) Out-Migration from Top 8 High-Cost Gateway MSAs (NYC, LA, SF, CHI, MIA, DC, BOS, SEA),
  (c) Log Real Hourly Wages, and
  (d) Rent Burden (Annual Gross Rent / Wage Income).
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
from essay3_analysis import build_ancestr_sql_case, build_bpl_sql_case, extract_pf_row  # noqa: E402

ACS_PATH = ROOT / "data" / "ipums" / "raw" / "acs_2007_2024" / "usa_00012.csv.gz"
TELE_PARQUET = ROOT / "papers" / "paper3_enclaves_wfh" / "results" / "occsoc_teleworkability.parquet"
OUT_DIR = ROOT / "papers" / "paper3_enclaves_wfh" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Top 8 High-Cost Immigrant Gateway MSAs (MET2013 CBSA codes)
GATEWAY_MSAS = {
    35620: "New York",
    31080: "Los Angeles",
    41860: "San Francisco",
    16980: "Chicago",
    33100: "Miami",
    47900: "Washington DC",
    14460: "Boston",
    42660: "Seattle",
}
GATEWAY_SQL = "(" + ", ".join(str(k) for k in GATEWAY_MSAS) + ")"


def run_module2() -> None:
    print("[03] Streaming ACS 2015-2024 + O*NET Teleworkability for ITT DiD & Event Study...")
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")

    bpl_case = build_bpl_sql_case()
    anc1_case = build_ancestr_sql_case("ANCESTR1")
    anc2_case = build_ancestr_sql_case("ANCESTR2")

    con.execute(f"CREATE TEMP TABLE tele AS SELECT occsoc, dn_teleworkable_bin AS tele_bin, telework_score FROM read_parquet('{TELE_PARQUET}') WHERE dn_teleworkable_bin IS NOT NULL")

    # Build base 2015-2024 microdata view
    con.execute(f"""
        CREATE TEMP VIEW acs1524 AS
        SELECT
            a.YEAR,
            a.MET2013 AS dest_met,
            CASE
                WHEN a.MIGRATE1 = 1 THEN a.MET2013
                WHEN a.MIGRATE1 IN (2, 3) AND a.MIGMET131 >= 10000 THEN a.MIGMET131
                ELSE NULL
            END AS orig_met,
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
            t.telework_score,
            CASE
                WHEN a.INCWAGE > 0 AND a.INCWAGE < 999998 AND a.UHRSWORK > 0 AND a.WKSWORK2 > 0
                THEN LN((a.INCWAGE * a.CPI99) / (a.UHRSWORK *
                    CASE a.WKSWORK2 WHEN 1 THEN 7.0 WHEN 2 THEN 20.0 WHEN 3 THEN 33.0 WHEN 4 THEN 43.5 WHEN 5 THEN 48.5 WHEN 6 THEN 51.0 ELSE NULL END))
                ELSE NULL
            END AS ln_wage,
            CASE
                WHEN a.OWNERSHP = 2 AND a.RENTGRS > 0 AND a.INCWAGE >= 5000 AND a.INCWAGE < 999998
                THEN LEAST(GREATEST((a.RENTGRS * 12.0) / a.INCWAGE, 0.01), 1.0)
                ELSE NULL
            END AS rent_burden
        FROM read_csv_auto('{ACS_PATH}') a
        LEFT JOIN tele t ON TRIM(CAST(a.OCCSOC AS VARCHAR)) = t.occsoc
        WHERE a.YEAR IN (2015, 2018, 2019, 2021, 2022, 2023, 2024)
          AND a.AGE BETWEEN 18 AND 64 AND a.SCHOOL != 2 AND a.MET2013 > 0
    """)

    # Pre-COVID (2015-2019) average co-ethnic immigrant share in each MSA (100% pre-determined!)
    print("   [DuckDB] Computing pre-COVID (2015-2019) co-ethnic enclave shares & origin-metro telework exposure...")
    con.execute("""
        CREATE TEMP TABLE pre_enclave AS
        WITH msa_p AS (
            SELECT dest_met AS met2013, SUM(PERWT) AS mp
            FROM acs1524 WHERE YEAR IN (2015, 2018, 2019) GROUP BY dest_met
        ),
        imm_p AS (
            SELECT eth, dest_met AS met2013, SUM(PERWT) AS ip
            FROM acs1524 WHERE YEAR IN (2015, 2018, 2019) AND is_immigrant = 1 AND eth IS NOT NULL
            GROUP BY eth, dest_met
        )
        SELECT i.eth, i.met2013, 100.0 * i.ip / m.mp AS share_pre
        FROM imm_p i JOIN msa_p m ON i.met2013 = m.met2013
    """)

    # Genuinely pre-determined (2015-2019) origin-metro and origin-metro x ethnicity telework exposure
    # (immune to post-move occupation switching!)
    con.execute("""
        CREATE TEMP TABLE pre_metro_tele AS
        SELECT
            dest_met AS met2013,
            SUM(tele_bin * PERWT) / NULLIF(SUM(PERWT), 0) AS orig_metro_tele_pre
        FROM acs1524
        WHERE YEAR IN (2015, 2018, 2019) AND tele_bin IS NOT NULL
        GROUP BY dest_met
    """)
    con.execute("""
        CREATE TEMP TABLE pre_eth_metro_tele AS
        SELECT
            eth,
            dest_met AS met2013,
            SUM(tele_bin * PERWT) / NULLIF(SUM(PERWT), 0) AS orig_eth_tele_pre
        FROM acs1524
        WHERE YEAR IN (2015, 2018, 2019) AND is_immigrant = 0 AND eth IS NOT NULL AND tele_bin IS NOT NULL
        GROUP BY eth, dest_met
    """)

    # ------------------------------------------------------------------
    # Aggregate 1: Origin-MSA Retention & Gateway Out-Migration panel
    # Exact sufficient-statistic cells: (eth, orig_met, YEAR, tele_bin, SEX)
    # ------------------------------------------------------------------
    print("   [DuckDB] Aggregating origin-MSA retention & gateway out-migration panel (2015-2024)...")
    ret_df = con.execute(f"""
        SELECT
            n.eth,
            n.orig_met AS met2013,
            n.YEAR,
            n.tele_bin,
            n.SEX,
            CASE WHEN n.orig_met IN {GATEWAY_SQL} THEN 1 ELSE 0 END AS is_gateway,
            COALESCE(p.share_pre, 0.0) AS share_pre,
            pm.orig_metro_tele_pre,
            pe.orig_eth_tele_pre,
            SUM(CASE WHEN n.orig_met = n.dest_met THEN n.PERWT ELSE 0.0 END) / NULLIF(SUM(n.PERWT), 0) AS true_retention,
            SUM(CASE WHEN n.orig_met != n.dest_met THEN n.PERWT ELSE 0.0 END) / NULLIF(SUM(n.PERWT), 0) AS outmig_rate,
            SUM(n.is_wfh * n.PERWT) / NULLIF(SUM(n.PERWT), 0) AS wfh_rate,
            SUM(n.telework_score * n.PERWT) / NULLIF(SUM(n.PERWT), 0) AS tele_score,
            SUM(n.AGE * n.PERWT) / NULLIF(SUM(n.PERWT), 0) AS AGE,
            SUM(CASE WHEN n.EDUC IS NOT NULL THEN n.EDUC * n.PERWT END) / NULLIF(SUM(CASE WHEN n.EDUC IS NOT NULL THEN n.PERWT END), 0) AS EDUC,
            SUM(n.PERWT) AS PERWT,
            COUNT(*) AS raw_n
        FROM acs1524 n
        LEFT JOIN pre_enclave p ON n.eth = p.eth AND n.orig_met = p.met2013
        LEFT JOIN pre_metro_tele pm ON n.orig_met = pm.met2013
        LEFT JOIN pre_eth_metro_tele pe ON n.eth = pe.eth AND n.orig_met = pe.met2013
        WHERE n.is_immigrant = 0 AND n.eth IS NOT NULL AND n.orig_met IS NOT NULL AND n.tele_bin IS NOT NULL
        GROUP BY n.eth, n.orig_met, n.YEAR, n.tele_bin, n.SEX, COALESCE(p.share_pre, 0.0), pm.orig_metro_tele_pre, pe.orig_eth_tele_pre
    """).df()

    # ------------------------------------------------------------------
    # Aggregate 2: Destination-MSA Wages & Rent Burden panel
    # Exact sufficient-statistic cells: (eth, dest_met, YEAR, tele_bin, SEX)
    # ------------------------------------------------------------------
    print("   [DuckDB] Aggregating destination-MSA log wages & rent burden panel (2015-2024)...")
    wage_df = con.execute("""
        SELECT
            n.eth,
            n.dest_met AS met2013,
            n.YEAR,
            n.tele_bin,
            n.SEX,
            COALESCE(p.share_pre, 0.0) AS share_pre,
            SUM(CASE WHEN n.ln_wage IS NOT NULL THEN n.ln_wage * n.PERWT END) / NULLIF(SUM(CASE WHEN n.ln_wage IS NOT NULL THEN n.PERWT END), 0) AS ln_wage,
            SUM(CASE WHEN n.ln_wage IS NOT NULL THEN n.PERWT ELSE 0.0 END) AS wt_wage,
            SUM(CASE WHEN n.rent_burden IS NOT NULL THEN n.rent_burden * n.PERWT END) / NULLIF(SUM(CASE WHEN n.rent_burden IS NOT NULL THEN n.PERWT END), 0) AS rent_burden,
            SUM(CASE WHEN n.rent_burden IS NOT NULL THEN n.PERWT ELSE 0.0 END) AS wt_rb,
            SUM(n.AGE * n.PERWT) / NULLIF(SUM(n.PERWT), 0) AS AGE,
            SUM(CASE WHEN n.EDUC IS NOT NULL THEN n.EDUC * n.PERWT END) / NULLIF(SUM(CASE WHEN n.EDUC IS NOT NULL THEN n.PERWT END), 0) AS EDUC,
            SUM(n.PERWT) AS PERWT,
            COUNT(*) AS raw_n
        FROM acs1524 n
        LEFT JOIN pre_enclave p ON n.eth = p.eth AND n.dest_met = p.met2013
        WHERE n.is_immigrant = 0 AND n.eth IS NOT NULL AND n.tele_bin IS NOT NULL
        GROUP BY n.eth, n.dest_met, n.YEAR, n.tele_bin, n.SEX, COALESCE(p.share_pre, 0.0)
    """).df()

    print(f"   Total individual micro-obs represented: retention={int(ret_df['raw_n'].sum()):,}, wages/rent={int(wage_df['raw_n'].sum()):,}")

    def prep_df(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["post"] = (d["YEAR"] >= 2021).astype(float)
        d["eth_year"] = d["eth"].astype(str) + "_" + d["YEAR"].astype(str)
        d["met_year"] = d["met2013"].astype(str) + "_" + d["YEAR"].astype(str)
        d["eth_met"] = d["eth"].astype(str) + "_" + d["met2013"].astype(str)
        d["tele_year"] = d["tele_bin"].astype(int).astype(str) + "_" + d["YEAR"].astype(str)
        d["tele_share"] = d["tele_bin"] * d["share_pre"]
        d["tele_post"] = d["tele_bin"] * d["post"]
        d["share_post"] = d["share_pre"] * d["post"]
        d["ddd_tele_post_share"] = d["tele_bin"] * d["post"] * d["share_pre"]
        d["ddd_tele_post_gw"] = d["tele_bin"] * d["post"] * d.get("is_gateway", 0)
        # Event-study interaction dummies (ref = 2019)
        for yr in [2015, 2018, 2021, 2022, 2023, 2024]:
            iy = (d["YEAR"] == yr).astype(float)
            d[f"es_tele_share_{yr}"] = d["tele_bin"] * d["share_pre"] * iy
            d[f"es_share_{yr}"] = d["share_pre"] * iy
            d[f"es_tele_{yr}"] = d["tele_bin"] * iy
            if "is_gateway" in d.columns:
                d[f"es_tele_gw_{yr}"] = d["tele_bin"] * d["is_gateway"] * iy
        return d

    ret_df = prep_df(ret_df)
    wage_df = prep_df(wage_df)

    fe_base = "eth_year + met_year + eth_met + tele_year"
    es_years = [2015, 2018, 2021, 2022, 2023, 2024]
    es_tele_share_terms = " + ".join(f"es_tele_share_{yr}" for yr in es_years)
    es_share_terms = " + ".join(f"es_share_{yr}" for yr in es_years)

    did_rows = []
    es_rows = []

    # ------------------------------------------------------------------
    # 1. Pooled ITT DiD Regressions (2015-2024)
    # ------------------------------------------------------------------
    # (a) True Origin-MSA Retention — Full Sample (DDD: Teleworkable x Post x PreEnclaveShare)
    sub_ret = ret_df.dropna(subset=["true_retention", "share_pre", "AGE", "EDUC", "PERWT"])
    m_ret = pf.feols(
        f"true_retention ~ ddd_tele_post_share + tele_share + share_post + AGE + EDUC + SEX | {fe_base}",
        data=sub_ret, weights="PERWT", vcov={"CRV1": "met2013"},
    )
    did_rows.append(extract_pf_row(m_ret, "ddd_tele_post_share", "ITT_DiD_TrueRetention_AllMSAs", {"raw_n": int(sub_ret["raw_n"].sum())}))
    did_rows.append(extract_pf_row(m_ret, "tele_share", "ITT_DiD_TrueRetention_PreBase_TeleXShare", {"raw_n": int(sub_ret["raw_n"].sum())}))
    did_rows.append(extract_pf_row(m_ret, "share_post", "ITT_DiD_TrueRetention_NonTele_PostXShare", {"raw_n": int(sub_ret["raw_n"].sum())}))

    # (b) Gateway Out-Migration — All MSAs (Teleworkable x Post x GatewayMSA)
    m_gw_all = pf.feols(
        f"outmig_rate ~ ddd_tele_post_gw + tele_bin:is_gateway + AGE + EDUC + SEX | {fe_base}",
        data=sub_ret, weights="PERWT", vcov={"CRV1": "met2013"},
    )
    did_rows.append(extract_pf_row(m_gw_all, "ddd_tele_post_gw", "ITT_DiD_GatewayOutMig_TeleXPostXGateway", {"raw_n": int(sub_ret["raw_n"].sum())}))

    # (c) Gateway Out-Migration — Within Top 8 Gateways (Teleworkable x Post x PreEnclaveShare)
    sub_gw = sub_ret[sub_ret["is_gateway"] == 1].copy()
    m_gw_in = pf.feols(
        f"outmig_rate ~ ddd_tele_post_share + tele_share + share_post + AGE + EDUC + SEX | {fe_base}",
        data=sub_gw, weights="PERWT", vcov={"CRV1": "eth_met"},
    )
    did_rows.append(extract_pf_row(m_gw_in, "ddd_tele_post_share", "ITT_DiD_WithinGateways_OutMig_TeleXPostXShare", {"raw_n": int(sub_gw["raw_n"].sum())}))

    # (d) Log Real Hourly Wage (Teleworkable x Post x PreEnclaveShare)
    sub_w = wage_df.dropna(subset=["ln_wage", "share_pre", "AGE", "EDUC", "wt_wage"])
    sub_w = sub_w[sub_w["wt_wage"] > 0]
    m_w = pf.feols(
        f"ln_wage ~ ddd_tele_post_share + tele_share + share_post + AGE + EDUC + SEX | {fe_base}",
        data=sub_w, weights="wt_wage", vcov={"CRV1": "met2013"},
    )
    did_rows.append(extract_pf_row(m_w, "ddd_tele_post_share", "ITT_DiD_LogWage_TeleXPostXShare", {"raw_n": int(sub_w["raw_n"].sum())}))
    did_rows.append(extract_pf_row(m_w, "tele_share", "ITT_DiD_LogWage_PreBase_TeleXShare", {"raw_n": int(sub_w["raw_n"].sum())}))

    # (e) Rent Burden = Annual Rent / Wage Income (Teleworkable x Post x PreEnclaveShare)
    sub_rb = wage_df.dropna(subset=["rent_burden", "share_pre", "AGE", "EDUC", "wt_rb"])
    sub_rb = sub_rb[sub_rb["wt_rb"] > 0]
    m_rb = pf.feols(
        f"rent_burden ~ ddd_tele_post_share + tele_share + share_post + AGE + EDUC + SEX | {fe_base}",
        data=sub_rb, weights="wt_rb", vcov={"CRV1": "met2013"},
    )
    did_rows.append(extract_pf_row(m_rb, "ddd_tele_post_share", "ITT_DiD_RentBurden_TeleXPostXShare", {"raw_n": int(sub_rb["raw_n"].sum())}))
    did_rows.append(extract_pf_row(m_rb, "tele_share", "ITT_DiD_RentBurden_PreBase_TeleXShare", {"raw_n": int(sub_rb["raw_n"].sum())}))

    # (f) First-Stage Manipulation Check: Realized WFH rate on Teleworkable x Post
    m_fs = pf.feols(
        "wfh_rate ~ tele_post + tele_bin + AGE + EDUC + SEX | eth_year + met_year + eth_met",
        data=sub_ret, weights="PERWT", vcov={"CRV1": "met2013"},
    )
    did_rows.append(extract_pf_row(m_fs, "tele_post", "FirstStage_RealizedWFH_TeleXPost", {"raw_n": int(sub_ret["raw_n"].sum())}))

    # (g) Genuinely Pre-Determined Origin-Metro Pre-2019 Telework Exposure Checks
    # Immune to post-move occupation switching!
    sub_pre = sub_ret.dropna(subset=["orig_metro_tele_pre", "orig_eth_tele_pre"]).copy()
    sub_pre["pre_metro_post"] = sub_pre["orig_metro_tele_pre"] * sub_pre["post"]
    sub_pre["pre_metro_post_gw"] = sub_pre["orig_metro_tele_pre"] * sub_pre["post"] * sub_pre["is_gateway"]
    sub_pre["gw_post"] = sub_pre["is_gateway"] * sub_pre["post"]
    sub_pre["pre_eth_post"] = sub_pre["orig_eth_tele_pre"] * sub_pre["post"]
    sub_pre["pre_eth_post_gw"] = sub_pre["orig_eth_tele_pre"] * sub_pre["post"] * sub_pre["is_gateway"]
    sub_pre["pre_eth_post_share"] = sub_pre["orig_eth_tele_pre"] * sub_pre["post"] * sub_pre["share_pre"]

    # g1: Origin-Metro pre-2019 telework share x Post on Out-Migration (eth_year + eth_met FEs)
    m_pm_out = pf.feols(
        "outmig_rate ~ pre_metro_post + AGE + EDUC + SEX | eth_year + eth_met",
        data=sub_pre, weights="PERWT", vcov={"CRV1": "met2013"},
    )
    did_rows.append(extract_pf_row(m_pm_out, "pre_metro_post", "Pre2019_OrigMetroTele_Post_OutMig", {"raw_n": int(sub_pre["raw_n"].sum())}))

    # g2: Origin-Metro pre-2019 telework share x Post x Gateway on Out-Migration
    m_pm_gw = pf.feols(
        "outmig_rate ~ pre_metro_post_gw + pre_metro_post + gw_post + AGE + EDUC + SEX | eth_year + eth_met",
        data=sub_pre, weights="PERWT", vcov={"CRV1": "met2013"},
    )
    did_rows.append(extract_pf_row(m_pm_gw, "pre_metro_post_gw", "Pre2019_OrigMetroTele_PostXGateway_OutMig", {"raw_n": int(sub_pre["raw_n"].sum())}))

    # g3: Origin-Metro x Ethnicity pre-2019 telework share x Post x Gateway (with full eth_year + met_year + eth_met FEs!)
    m_pe_gw = pf.feols(
        "outmig_rate ~ pre_eth_post_gw + pre_eth_post + AGE + EDUC + SEX | eth_year + met_year + eth_met",
        data=sub_pre, weights="PERWT", vcov={"CRV1": "met2013"},
    )
    did_rows.append(extract_pf_row(m_pe_gw, "pre_eth_post_gw", "Pre2019_OrigEthMetroTele_PostXGateway_OutMig", {"raw_n": int(sub_pre["raw_n"].sum())}))

    # g4: Origin-Metro x Ethnicity pre-2019 telework share x Post x PreEnclaveShare on True Retention
    m_pe_ret = pf.feols(
        "true_retention ~ pre_eth_post_share + pre_eth_post + share_post + AGE + EDUC + SEX | eth_year + met_year + eth_met",
        data=sub_pre, weights="PERWT", vcov={"CRV1": "met2013"},
    )
    did_rows.append(extract_pf_row(m_pe_ret, "pre_eth_post_share", "Pre2019_OrigEthMetroTele_PostXShare_Retention", {"raw_n": int(sub_pre["raw_n"].sum())}))

    # ------------------------------------------------------------------
    # 2. Dynamic Event Studies (2015, 2018, [2019=0], 2021, 2022, 2023, 2024)
    # ------------------------------------------------------------------
    def extract_es(model, prefix: str, outcome_name: str) -> None:
        es_rows.append({"outcome": outcome_name, "year": 2019, "coef": 0.0, "se": 0.0, "pval": 1.0})
        for yr in es_years:
            v = f"{prefix}_{yr}"
            es_rows.append({
                "outcome": outcome_name,
                "year": yr,
                "coef": float(model.coef().get(v, np.nan)),
                "se": float(model.se().get(v, np.nan)),
                "pval": float(model.pvalue().get(v, np.nan)),
            })

    # ES 1: True Origin-MSA Retention (Teleworkable x EnclaveShare x Year)
    m_es_ret = pf.feols(
        f"true_retention ~ {es_tele_share_terms} + {es_share_terms} + tele_share + AGE + EDUC + SEX | {fe_base}",
        data=sub_ret, weights="PERWT", vcov={"CRV1": "met2013"},
    )
    extract_es(m_es_ret, "es_tele_share", "TrueRetention_TeleXShare")

    # ES 2: Gateway Out-Migration (Teleworkable x GatewayMSA x Year)
    es_gw_terms = " + ".join(f"es_tele_gw_{yr}" for yr in es_years)
    m_es_gw = pf.feols(
        f"outmig_rate ~ {es_gw_terms} + tele_bin:is_gateway + AGE + EDUC + SEX | {fe_base}",
        data=sub_ret, weights="PERWT", vcov={"CRV1": "met2013"},
    )
    extract_es(m_es_gw, "es_tele_gw", "GatewayOutMig_TeleXGateway")

    # ES 3: Log Real Hourly Wage (Teleworkable x EnclaveShare x Year)
    m_es_w = pf.feols(
        f"ln_wage ~ {es_tele_share_terms} + {es_share_terms} + tele_share + AGE + EDUC + SEX | {fe_base}",
        data=sub_w, weights="wt_wage", vcov={"CRV1": "met2013"},
    )
    extract_es(m_es_w, "es_tele_share", "LogWage_TeleXShare")

    # ES 4: Rent Burden (Teleworkable x EnclaveShare x Year)
    m_es_rb = pf.feols(
        f"rent_burden ~ {es_tele_share_terms} + {es_share_terms} + tele_share + AGE + EDUC + SEX | {fe_base}",
        data=sub_rb, weights="wt_rb", vcov={"CRV1": "met2013"},
    )
    extract_es(m_es_rb, "es_tele_share", "RentBurden_TeleXShare")

    did_df = pd.DataFrame(did_rows)
    es_df = pd.DataFrame(es_rows)
    did_df.to_csv(OUT_DIR / "module2_telework_did.csv", index=False)
    es_df.to_csv(OUT_DIR / "module2_telework_event_study.csv", index=False)
    (OUT_DIR / "module2_summary.json").write_text(json.dumps({
        "did": did_rows,
        "event_study": es_rows,
    }, indent=2, default=str))

    print("\n=== MODULE 2: TELEWORKABILITY ITT DiD (2015-2024) ===")
    for _, r in did_df.iterrows():
        sig = "***" if r["pval"] < 0.01 else ("**" if r["pval"] < 0.05 else ("*" if r["pval"] < 0.10 else ""))
        print(f"  {r['spec']:46s}: coef={r['coef']:+.5f} ({r['se']:.5f}){sig:3s} | t={r['tstat']:+.2f} | N_cells={int(r['n']):,} (raw={int(r['raw_n']):,})")

    print("\n=== MODULE 2: EVENT STUDY HIGHLIGHTS (ref=2019) ===")
    for outc, grp in es_df.groupby("outcome"):
        pts = ", ".join(f"{int(r['year'])}:{r['coef']:+.4f}" for _, r in grp.sort_values("year").iterrows())
        print(f"  {outc:28s}: {pts}")


if __name__ == "__main__":
    run_module2()
