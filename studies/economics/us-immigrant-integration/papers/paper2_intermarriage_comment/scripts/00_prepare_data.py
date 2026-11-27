#!/usr/bin/env python3
"""
Step 0: Prepare harmonized 1980-2024 IPUMS USA + O*NET/Autor-Dorn analysis frames for Paper 2.
Builds directly on verified IPUMS coding from scripts/essay2_analysis.py, adding:
  - Cohabiting partner nativity/co-ethnicity flags (MARST != 1 & (RELATED == 1114 | SPLOC > 0))
  - Spouse labor supply variables (UHRSWORK_SP, EMPSTAT_SP, sp_ft)
  - Clean Parquet caches in papers/paper2_intermarriage_comment/results/ for fast downstream runs.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import List

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
IPUMS_DIR = ROOT / "data" / "ipums" / "raw" / "essay2"
FULLPOP_1970_2000_PATH = ROOT / "data" / "ipums" / "raw" / "essay1_1970_2000" / "usa_00011.csv.gz"
FULLPOP_2007_2024_PATH = ROOT / "data" / "ipums" / "raw" / "acs_2007_2024" / "usa_00012.csv.gz"
AUX_DIR = ROOT / "data" / "aux"
PAPER_DIR = ROOT / "papers" / "paper2_intermarriage_comment"
RES_DIR = PAPER_DIR / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)

PERIOD_ORDER = [
    "1980",
    "1990",
    "2000",
    "2005-2007",
    "2008-2010",
    "2012-2015",
    "2017-2019",
    "2021-2024",
]
ENGLISH_SPEAKING_BPL = {150, 410, 411, 412, 413, 414, 700, 710}
US_TERRITORY_BPL = {100, 105, 110, 115, 120}


def load_ipums_data(data_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(data_dir / "*.csv.gz")))
    if not files:
        raise FileNotFoundError(f"No .csv.gz files found in {data_dir}")
    con = duckdb.connect()
    df = con.execute(
        "SELECT * FROM read_csv_auto(?, sample_size=-1, all_varchar=false)", [files]
    ).df()
    con.close()
    df.columns = [c.upper() for c in df.columns]
    num_cols = [
        "YEAR", "PERWT", "CPI99", "STATEFIP", "METAREA", "MET2013", "METRO",
        "SEX", "AGE", "MARST", "RACE", "HISPAN", "ANCESTR1", "BPL", "CITIZEN",
        "YRIMMIG", "YRSUSA1", "SPEAKENG", "SCHOOL", "EDUC", "EMPSTAT", "LABFORCE",
        "UHRSWORK", "WKSWORK1", "WKSWORK2", "INCWAGE", "OCC1990", "OCC2010",
        "NCHILD", "RELATE", "RELATED", "SPLOC",
        "BPL_SP", "RACE_SP", "HISPAN_SP", "ANCESTR1_SP", "EDUC_SP",
        "SPEAKENG_SP", "EMPSTAT_SP", "UHRSWORK_SP", "INCWAGE_SP", "CITIZEN_SP", "AGE_SP",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df


def assign_period(year: pd.Series) -> pd.Series:
    conds = [
        year == 1980,
        year == 1990,
        year == 2000,
        year.between(2005, 2007),
        year.between(2008, 2010),
        year.between(2012, 2015),
        year.between(2017, 2019),
        year.between(2021, 2024),
    ]
    return pd.Series(np.select(conds, PERIOD_ORDER, default="OTHER"), index=year.index)


def clean_occsoc(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
        .astype(str)
        .str.strip()
        .str.replace("-", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.upper()
    )


def merge_occupation_tasks(df: pd.DataFrame) -> pd.DataFrame:
    onet_path = AUX_DIR / "occsoc_ai_onet_tasks.parquet"
    if onet_path.exists() and "OCCSOC" in df.columns:
        onet = pd.read_parquet(onet_path)
        task_cols = [
            "onet_social_skill",
            "onet_english_comm",
            "onet_english_written",
            "onet_english_oral",
            "onet_math_analytical",
            "openai_dv_beta_std",
            "felten_aioe_lm",
        ]
        keep_cols = ["occsoc"] + [c for c in task_cols if c in onet.columns]
        onet_exact = onet[keep_cols].drop_duplicates("occsoc").copy()
        onet_exact["occsoc4"] = onet_exact["occsoc"].str[:4]
        onet_4d = onet_exact.groupby("occsoc4", as_index=False)[
            [c for c in task_cols if c in onet.columns]
        ].mean()

        df["occsoc_clean"] = clean_occsoc(df["OCCSOC"])
        df["occsoc4"] = df["occsoc_clean"].str[:4]
        df = df.merge(
            onet_exact.drop(columns=["occsoc4"]),
            left_on="occsoc_clean",
            right_on="occsoc",
            how="left",
        )
        m4 = df[["occsoc4"]].merge(onet_4d, on="occsoc4", how="left")
        for c in task_cols:
            if c in df.columns and c in m4.columns:
                df[c] = df[c].fillna(pd.Series(m4[c].to_numpy(), index=df.index))
        df = df.drop(columns=["occsoc", "occsoc4"], errors="ignore")

    xwalk_path = AUX_DIR / "dorn_occ_crosswalks.parquet"
    alm_path = AUX_DIR / "occ1990dd_tasks.parquet"
    if xwalk_path.exists() and alm_path.exists() and "OCC1990" in df.columns:
        xw = pd.read_parquet(xwalk_path)
        xw90 = xw.loc[xw["source_year"] == 1990, ["occ_code", "occ1990dd"]].drop_duplicates("occ_code")
        alm = pd.read_parquet(alm_path)[["occ1990dd", "task_abstract_std", "task_routine_std", "rti_std"]]
        df = df.merge(xw90, left_on="OCC1990", right_on="occ_code", how="left")
        df = df.merge(alm, on="occ1990dd", how="left")
        df = df.drop(columns=["occ_code"], errors="ignore")
    return df


def compute_fullpop_denominators() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute true full-population (native + foreign-born, working-age 18-64) annual-average
    denominators by (enclave_geo_id, period), (state_code, period), and (met2013_code, period)
    from IPUMS USA full-population extracts usa_00011 (1980, 1990, 2000) and usa_00012
    (2007 -> 2005-2007, 2010 -> 2008-2010, 2015 -> 2012-2015, 2018-2019 -> 2017-2019,
    2021-2024 -> 2021-2024).
    """
    if not FULLPOP_1970_2000_PATH.exists() or not FULLPOP_2007_2024_PATH.exists():
        raise FileNotFoundError(
            f"Full-population IPUMS extracts not found at {FULLPOP_1970_2000_PATH} and {FULLPOP_2007_2024_PATH}"
        )
    con = duckdb.connect()
    p11 = str(FULLPOP_1970_2000_PATH)
    p12 = str(FULLPOP_2007_2024_PATH)

    print("  [00_prepare_data] Aggregating full working-age (18-64) metro/state denominators from usa_00011 + usa_00012 (single pass per file)...", flush=True)
    # Single pass over usa_00011 (1980, 1990, 2000)
    agg11 = con.execute(f"""
        SELECT
            CAST(YEAR AS VARCHAR) AS period,
            CAST(COALESCE(TRY_CAST(STATEFIP AS INT), 0) AS VARCHAR) AS state_code,
            COALESCE(TRY_CAST(METAREA AS INT), 0) AS metarea_i,
            COALESCE(TRY_CAST(MET2013 AS INT), 0) AS met2013_i,
            SUM(CAST(PERWT AS DOUBLE)) AS pop
        FROM read_csv_auto('{p11}')
        WHERE YEAR IN (1980, 1990, 2000)
        GROUP BY 1, 2, 3, 4
    """).df()

    # Single pass over usa_00012 (2007, 2010, 2015, 2018-19, 2021-24)
    period_case_12 = """
        CASE
            WHEN YEAR = 2007 THEN '2005-2007'
            WHEN YEAR = 2010 THEN '2008-2010'
            WHEN YEAR = 2015 THEN '2012-2015'
            WHEN YEAR IN (2018, 2019) THEN '2017-2019'
            WHEN YEAR BETWEEN 2021 AND 2024 THEN '2021-2024'
            ELSE 'OTHER'
        END
    """
    nyrs_case_12 = """
        CASE
            WHEN YEAR IN (2018, 2019) THEN 2.0
            WHEN YEAR BETWEEN 2021 AND 2024 THEN 4.0
            ELSE 1.0
        END
    """
    agg12 = con.execute(f"""
        SELECT
            {period_case_12} AS period,
            CAST(COALESCE(TRY_CAST(STATEFIP AS INT), 0) AS VARCHAR) AS state_code,
            COALESCE(TRY_CAST(MET2013 AS INT), 0) AS met2013_i,
            SUM(CAST(PERWT AS DOUBLE) / ({nyrs_case_12})) AS pop
        FROM read_csv_auto('{p12}')
        WHERE YEAR IN (2007, 2010, 2015, 2018, 2019, 2021, 2022, 2023, 2024)
        GROUP BY 1, 2, 3
    """).df()
    con.close()

    agg11["enclave_geo_id"] = np.where(agg11["metarea_i"] > 0, "M_" + agg11["metarea_i"].astype(str), "S_" + agg11["state_code"])
    agg11["met2013_code"] = np.where(agg11["met2013_i"] > 0, "C_" + agg11["met2013_i"].astype(str), "S_" + agg11["state_code"])
    agg12["enclave_geo_id"] = np.where(agg12["met2013_i"] > 0, "C_" + agg12["met2013_i"].astype(str), "S_" + agg12["state_code"])
    agg12["met2013_code"] = agg12["enclave_geo_id"]

    fp_geo = pd.concat([
        agg11.groupby(["period", "enclave_geo_id"], as_index=False)["pop"].sum().rename(columns={"pop": "fullpop_geo"}),
        agg12.groupby(["period", "enclave_geo_id"], as_index=False)["pop"].sum().rename(columns={"pop": "fullpop_geo"}),
    ], ignore_index=True)
    fp_state = pd.concat([
        agg11.groupby(["period", "state_code"], as_index=False)["pop"].sum().rename(columns={"pop": "fullpop_state"}),
        agg12.groupby(["period", "state_code"], as_index=False)["pop"].sum().rename(columns={"pop": "fullpop_state"}),
    ], ignore_index=True)
    fp_m13 = pd.concat([
        agg11.loc[agg11["period"] == "2000"].groupby(["period", "met2013_code"], as_index=False)["pop"].sum().rename(columns={"pop": "fullpop_met2013"}),
        agg12.groupby(["period", "met2013_code"], as_index=False)["pop"].sum().rename(columns={"pop": "fullpop_met2013"}),
    ], ignore_index=True)
    fp_geo.to_parquet(RES_DIR / "fullpop_metro_denominators.parquet", index=False)
    fp_state.to_parquet(RES_DIR / "fullpop_state_denominators.parquet", index=False)
    fp_m13.to_parquet(RES_DIR / "fullpop_met2013_denominators.parquet", index=False)
    return fp_geo, fp_state, fp_m13


def main() -> None:
    print("[00_prepare_data] Loading IPUMS usa_00010.csv.gz...")
    df = load_ipums_data(IPUMS_DIR)
    print(f"  Loaded {len(df):,} raw rows.")

    df["period"] = assign_period(df["YEAR"])
    df = df.loc[df["period"] != "OTHER"].copy()

    # Period weight normalization (divide PERWT by number of distinct years in period)
    yr_counts = df.groupby("period")["YEAR"].transform("nunique").clip(lower=1)
    df["wt"] = df["PERWT"].astype(float) / yr_counts.astype(float)

    # Arrival age
    yrs_us = np.where(
        df["YRSUSA1"].notna() & (df["YRSUSA1"] > 0),
        df["YRSUSA1"],
        np.where(df["YRIMMIG"].notna() & (df["YRIMMIG"] > 0), df["YEAR"] - df["YRIMMIG"], np.nan),
    )
    df["yrs_in_us"] = np.clip(yrs_us, 0, df["AGE"])
    df["age_at_arrival"] = df["AGE"] - df["yrs_in_us"]
    df["is_15gen"] = df["age_at_arrival"].notna() & (df["age_at_arrival"] >= 0) & (df["age_at_arrival"] < 18)
    df["arr_bin"] = pd.cut(
        df["age_at_arrival"],
        bins=[-0.1, 5.5, 11.5, 17.5, 99],
        labels=["0-5", "6-11", "12-17", "18+"],
    ).astype(str)

    # Nativity & origin flags
    df["is_foreign_fs"] = df["BPL"] >= 100
    df["is_us_territory"] = df["BPL"].isin(US_TERRITORY_BPL)
    df["is_english_origin"] = df["BPL"].isin(ENGLISH_SPEAKING_BPL)

    # Geography & enclave size computed on FULL sample (matching scripts/essay2_analysis.py)
    metarea_int = df["METAREA"].fillna(0).astype(int)
    met2013_int = df["MET2013"].fillna(0).astype(int)
    state_int = df["STATEFIP"].fillna(0).astype(int)
    has_metarea = metarea_int > 0
    has_met2013 = met2013_int > 0
    df["in_ident_msa"] = np.where(df["YEAR"] <= 2010, has_metarea, has_met2013)
    df["geo_id"] = np.where(
        df["YEAR"] <= 2010,
        np.where(has_metarea, "M_" + metarea_int.astype(str), "S_" + state_int.astype(str)),
        np.where(has_met2013, "C_" + met2013_int.astype(str), "S_" + state_int.astype(str)),
    )
    df["state_code"] = state_int.astype(str)
    df["state_period"] = df["state_code"] + "_" + df["period"]
    df["met2013_code"] = np.where(has_met2013, "C_" + met2013_int.astype(str), "S_" + state_int.astype(str))
    df["met2013_period"] = df["met2013_code"] + "_" + df["period"]
    df["in_ident_met2013"] = has_met2013
    df["geo_period"] = df["geo_id"] + "_" + df["period"]
    df["bpl_code"] = df["BPL"].fillna(0).astype(int).astype(str)
    df["bpl_period"] = df["bpl_code"] + "_" + df["period"]

    tot_geo = df.groupby("geo_period")["wt"].transform("sum")
    tot_bpl_geo = df.groupby(["geo_period", "bpl_code"])["wt"].transform("sum")
    # Legacy foreign-born-extract denominator (retained explicitly for Table 1 sensitivity row)
    df["enclave_share_fb"] = (tot_bpl_geo / tot_geo.clip(lower=1e-9)).fillna(0.0)
    df["enclave_share_fb_sq"] = df["enclave_share_fb"] ** 2

    # True full-population (native + foreign-born, working-age 18-64) metro/state enclave share
    # Uses METAREA in 1980-2000 (usa_00011) and MET2013 in 2005-2024 (usa_00012)
    df["enclave_geo_id"] = np.where(
        df["YEAR"] <= 2000,
        np.where(has_metarea, "M_" + metarea_int.astype(str), "S_" + state_int.astype(str)),
        np.where(has_met2013, "C_" + met2013_int.astype(str), "S_" + state_int.astype(str)),
    )
    df["enclave_geo_period"] = df["enclave_geo_id"] + "_" + df["period"]
    fp_geo, fp_state, fp_m13 = compute_fullpop_denominators()

    df = df.merge(fp_geo, on=["period", "enclave_geo_id"], how="left")
    df = df.merge(fp_state, on=["period", "state_code"], how="left")
    df = df.merge(fp_m13, on=["period", "met2013_code"], how="left")

    fullpop_match_rate = float(df["fullpop_geo"].notna().mean())
    print(f"  Full-population metro denominator exact match rate: {100.0 * fullpop_match_rate:.3f}%")
    # Any unmatched rare code falls back to full state working-age population
    df["fullpop_geo_clean"] = df["fullpop_geo"].fillna(df["fullpop_state"])

    tot_bpl_egeo = df.groupby(["enclave_geo_period", "bpl_code"])["wt"].transform("sum")
    df["enclave_share"] = (tot_bpl_egeo / df["fullpop_geo_clean"].clip(lower=1e-9)).clip(0.0, 1.0).fillna(0.0)
    df["enclave_share_sq"] = df["enclave_share"] ** 2

    # State-period and MET2013-period enclave shares with true full working-age population denominators
    tot_bpl_st = df.groupby(["state_period", "bpl_code"])["wt"].transform("sum")
    df["enclave_share_state"] = (tot_bpl_st / df["fullpop_state"].clip(lower=1e-9)).clip(0.0, 1.0).fillna(0.0)
    df["enclave_share_state_sq"] = df["enclave_share_state"] ** 2

    tot_bpl_m13 = df.groupby(["met2013_period", "bpl_code"])["wt"].transform("sum")
    df["enclave_share_met2013"] = (tot_bpl_m13 / df["fullpop_met2013"].fillna(df["fullpop_state"]).clip(lower=1e-9)).clip(0.0, 1.0).fillna(0.0)
    df["enclave_share_met2013_sq"] = df["enclave_share_met2013"] ** 2

    # Filter down to 1.5-gen non-English-origin, not-in-school, valid-education universe
    # (saves massive memory & time while keeping all 1.75M 1.5-gen men & women for IPW & cohabitation!)
    df["valid_educ"] = df["EDUC"].between(0, 11)
    base = df.loc[
        df["is_foreign_fs"]
        & (~df["is_english_origin"])
        & df["is_15gen"]
        & df["valid_educ"]
        & (df["SCHOOL"] != 2)
    ].copy()
    del df
    print(f"  Base 1.5-gen universe (men + women, all MARST/LFP): {len(base):,} rows.")

    # Education
    base["educ_cat"] = np.select(
        [
            base["EDUC"].between(0, 5),
            base["EDUC"] == 6,
            base["EDUC"].between(7, 9),
            base["EDUC"].between(10, 11),
        ],
        ["lt_hs", "hs", "some_col", "college"],
        default="missing",
    )
    base["ed_hs"] = (base["educ_cat"] == "hs").astype(int)
    base["ed_sc"] = (base["educ_cat"] == "some_col").astype(int)
    base["ed_col"] = (base["educ_cat"] == "college").astype(int)

    # Spouse education
    base["ed_sp_hs"] = (base["EDUC_SP"] == 6).astype(int)
    base["ed_sp_sc"] = base["EDUC_SP"].between(7, 9).astype(int)
    base["ed_sp_col"] = base["EDUC_SP"].between(10, 11).astype(int)

    # English, children, age^2, citizenship
    base["eng_fluent"] = base["SPEAKENG"].isin([3, 4, 5]).astype(int)
    base["eng_very_well"] = base["SPEAKENG"].isin([3, 4]).astype(int)
    base["has_child"] = (base["NCHILD"] > 0).astype(int)
    base["nchild_clean"] = base["NCHILD"].fillna(0).clip(0, 9).astype(float)
    base["age_sq"] = base["AGE"].astype(float) ** 2
    # Clean citizenship flags:
    # CITIZEN == 1: Born abroad of American parents (U.S. citizen at birth — NOT a true foreign-born immigrant)
    # CITIZEN == 2: Naturalized citizen
    # CITIZEN == 3: Not a citizen
    base["is_born_abroad_us_parent"] = (base["CITIZEN"] == 1)
    base["is_true_immigrant"] = (base["BPL"] >= 150) & base["CITIZEN"].isin([2, 3])
    # Clean naturalized citizen indicator (never conflates CITIZEN==1 with naturalized citizens)
    base["is_citizen"] = (base["CITIZEN"] == 2).astype(int)

    # Race/ethnicity (contaminated vs clean)
    base["fs_white"] = (base["RACE"] == 1).astype(int)
    base["race_clean"] = np.select(
        [
            base["HISPAN"].between(1, 4),
            (base["HISPAN"] == 0) & (base["RACE"] == 1),
            (base["HISPAN"] == 0) & (base["RACE"] == 2),
            (base["HISPAN"] == 0) & (base["RACE"].isin([4, 5, 6])),
        ],
        ["Hispanic", "NH_White", "NH_Black", "NH_Asian"],
        default="NH_Other",
    )

    # Marriage, cohabitation & partner nativity/co-ethnicity
    base["married_sp"] = (base["MARST"] == 1).astype(int)
    base["cohab_strict"] = ((base["MARST"] != 1) & (base["RELATED"] == 1114) & (base["BPL_SP"] > 0)).astype(int)
    base["cohab_any_sploc"] = ((base["MARST"] != 1) & ((base["RELATED"] == 1114) | (base["SPLOC"] > 0)) & (base["BPL_SP"] > 0)).astype(int)
    base["partnered"] = ((base["married_sp"] == 1) | (base["cohab_any_sploc"] == 1)).astype(int)

    # Native spouse (married only, matching F&S) and native partner (married OR cohabiting)
    sp_is_us_born = base["BPL_SP"].between(1, 56)
    base["native_sp"] = ((base["married_sp"] == 1) & sp_is_us_born).astype(int)
    base["native_partner"] = ((base["partnered"] == 1) & sp_is_us_born).astype(int)

    # Co-ethnic vs non-co-ethnic
    same_hisp = base["HISPAN"].between(1, 4) & base["HISPAN_SP"].between(1, 4)
    anc_valid = (
        (base["ANCESTR1"] > 0) & (base["ANCESTR1"] < 900)
        & (base["ANCESTR1_SP"] > 0) & (base["ANCESTR1_SP"] < 900)
    )
    same_anc = anc_valid & ((base["ANCESTR1"] // 10) == (base["ANCESTR1_SP"] // 10))
    same_asian = (
        (base["HISPAN"] == 0) & (base["HISPAN_SP"] == 0)
        & base["RACE"].isin([4, 5, 6]) & base["RACE_SP"].isin([4, 5, 6])
    )
    is_coeth_pair = same_hisp | same_anc | same_asian
    base["coeth_native_sp"] = ((base["native_sp"] == 1) & is_coeth_pair).astype(int)
    base["noncoeth_native_sp"] = ((base["native_sp"] == 1) & (~is_coeth_pair)).astype(int)
    base["coeth_native_partner"] = ((base["native_partner"] == 1) & is_coeth_pair).astype(int)
    base["noncoeth_native_partner"] = ((base["native_partner"] == 1) & (~is_coeth_pair)).astype(int)

    # Labor supply & real hourly wages (1999 dollars via CPI99)
    wks_map = {1: 7.0, 2: 20.0, 3: 33.0, 4: 43.5, 5: 48.5, 6: 51.0}
    wks_mid = base["WKSWORK2"].map(wks_map).fillna(0.0)
    wks = np.where(base["WKSWORK1"].notna() & (base["WKSWORK1"] > 0), base["WKSWORK1"].astype(float), wks_mid)
    base["employed"] = (base["EMPSTAT"] == 1).astype(int)
    base["ftfy"] = ((base["EMPSTAT"] == 1) & (base["UHRSWORK"] >= 35) & (base["WKSWORK2"] == 6)).astype(int)

    denom = wks * base["UHRSWORK"].astype(float)
    base["real_hr_wage"] = np.where(
        (base["INCWAGE"] > 0) & (base["INCWAGE"] < 999998) & (denom > 0),
        (base["INCWAGE"].astype(float) * base["CPI99"].astype(float)) / denom,
        np.nan,
    )
    base["valid_wage"] = base["real_hr_wage"].between(2.0, 200.0)
    base["log_wage"] = np.where(base["valid_wage"], np.log(base["real_hr_wage"]), np.nan)

    # Spouse labor supply & earnings
    sp_real = np.where(
        (base["INCWAGE_SP"] > 0) & (base["INCWAGE_SP"] < 999998),
        base["INCWAGE_SP"].astype(float) * base["CPI99"].astype(float),
        0.0,
    )
    base["log_inc_sp"] = np.log1p(np.maximum(sp_real, 0.0))
    base["sp_employed"] = (base["EMPSTAT_SP"] == 1).astype(int)
    base["sp_uhrs"] = np.where(
        base["UHRSWORK_SP"].notna() & (base["UHRSWORK_SP"] >= 0) & (base["UHRSWORK_SP"] <= 99),
        base["UHRSWORK_SP"].astype(float),
        0.0,
    )
    base["sp_ft"] = ((base["EMPSTAT_SP"] == 1) & (base["sp_uhrs"] >= 35)).astype(int)

    # Explicit period dummies (ref = 1980)
    for p in PERIOD_ORDER[1:]:
        col = f"P_{p.replace('-', '_')}"
        base[col] = (base["period"] == p).astype(int)

    # Merge O*NET + Autor-Dorn tasks
    base = merge_occupation_tasks(base)

    # Keep lean set of columns for fast parquet I/O
    keep_cols = [
        "YEAR", "period", "wt", "SEX", "AGE", "age_sq", "MARST", "RELATED", "SPLOC",
        "RACE", "HISPAN", "BPL", "bpl_code", "bpl_period", "CITIZEN", "is_citizen",
        "is_born_abroad_us_parent", "is_true_immigrant",
        "yrs_in_us", "age_at_arrival", "arr_bin", "is_us_territory",
        "EDUC", "educ_cat", "ed_hs", "ed_sc", "ed_col",
        "ed_sp_hs", "ed_sp_sc", "ed_sp_col",
        "eng_fluent", "eng_very_well", "has_child", "nchild_clean",
        "fs_white", "race_clean",
        "married_sp", "cohab_strict", "cohab_any_sploc", "partnered",
        "native_sp", "native_partner",
        "coeth_native_sp", "noncoeth_native_sp",
        "coeth_native_partner", "noncoeth_native_partner",
        "in_ident_msa", "in_ident_met2013",
        "geo_id", "geo_period", "state_code", "state_period", "met2013_code", "met2013_period",
        "enclave_share", "enclave_share_sq",
        "enclave_share_fb", "enclave_share_fb_sq",
        "enclave_share_state", "enclave_share_state_sq",
        "enclave_share_met2013", "enclave_share_met2013_sq",
        "employed", "ftfy", "valid_wage", "log_wage", "real_hr_wage",
        "log_inc_sp", "sp_employed", "sp_uhrs", "sp_ft",
        "onet_social_skill", "onet_english_comm", "onet_math_analytical",
        "task_abstract_std", "task_routine_std", "rti_std",
    ] + [f"P_{p.replace('-', '_')}" for p in PERIOD_ORDER[1:]]
    keep_cols = [c for c in keep_cols if c in base.columns]
    base = base[keep_cols].copy()

    base.to_parquet(RES_DIR / "base_15gen.parquet", index=False)

    men_ftfy = base.loc[
        (base["SEX"] == 1) & (base["married_sp"] == 1) & (base["ftfy"] == 1)
        & base["valid_wage"] & base["in_ident_msa"]
    ].copy()
    women_ftfy = base.loc[
        (base["SEX"] == 2) & (base["married_sp"] == 1) & (base["ftfy"] == 1)
        & base["valid_wage"] & base["in_ident_msa"]
    ].copy()
    men_ftfy.to_parquet(RES_DIR / "men_ftfy_married.parquet", index=False)
    women_ftfy.to_parquet(RES_DIR / "women_ftfy_married.parquet", index=False)

    meta = {
        "n_base_15gen": int(len(base)),
        "n_base_15gen_true_imm": int(base["is_true_immigrant"].sum()),
        "n_base_men": int((base["SEX"] == 1).sum()),
        "n_base_men_true_imm": int(((base["SEX"] == 1) & base["is_true_immigrant"]).sum()),
        "n_base_women": int((base["SEX"] == 2).sum()),
        "n_base_women_true_imm": int(((base["SEX"] == 2) & base["is_true_immigrant"]).sum()),
        "n_men_ftfy_married_all": int(len(men_ftfy)),
        "n_men_ftfy_married_1980_2010_all": int((men_ftfy["YEAR"] <= 2010).sum()),
        "n_men_ftfy_married_nopr": int((~men_ftfy["is_us_territory"]).sum()),
        "n_men_ftfy_married_cit1": int(men_ftfy["is_born_abroad_us_parent"].sum()),
        "n_men_ftfy_married_true_imm": int(men_ftfy["is_true_immigrant"].sum()),
        "n_men_ftfy_married_true_imm_1980_2010": int(((men_ftfy["YEAR"] <= 2010) & men_ftfy["is_true_immigrant"]).sum()),
        "n_women_ftfy_married_all": int(len(women_ftfy)),
        "n_women_ftfy_married_nopr": int((~women_ftfy["is_us_territory"]).sum()),
        "n_women_ftfy_married_cit1": int(women_ftfy["is_born_abroad_us_parent"].sum()),
        "n_women_ftfy_married_true_imm": int(women_ftfy["is_true_immigrant"].sum()),
        "fullpop_metro_match_rate": round(fullpop_match_rate, 6),
        "mean_enclave_share_fullpop_men": round(float(men_ftfy["enclave_share"].mean()), 6),
        "mean_enclave_share_fb_men": round(float(men_ftfy["enclave_share_fb"].mean()), 6),
        "n_men_cohab_ftfy_true_imm": int((
            (base["SEX"] == 1) & (base["cohab_any_sploc"] == 1) & (base["ftfy"] == 1)
            & base["valid_wage"] & base["in_ident_msa"] & base["is_true_immigrant"]
        ).sum()),
        "n_women_cohab_ftfy_true_imm": int((
            (base["SEX"] == 2) & (base["cohab_any_sploc"] == 1) & (base["ftfy"] == 1)
            & base["valid_wage"] & base["in_ident_msa"] & base["is_true_immigrant"]
        ).sum()),
    }
    with open(RES_DIR / "data_prep_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("[00_prepare_data] Done! Summary:")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
