#!/usr/bin/env python3
"""
Paper 3 — Script 02 (Contribution 1):
Intra-Metro (`CPUMA0010`, 2007-2021 & consistent 2020-Census `PUMA`, 2022-2024)
Neighborhood Hedonic Housing Costs, Crowding, Multigenerational Co-Residence,
and Central-City vs. Suburban Sorting.

Reconciles Saiz (2007) [MSA-level], Saiz & Wachter (2011) [within-MSA neighborhood],
and Song (2017) [co-ethnic native group expenditure] on 2007-2024 IPUMS ACS microdata.
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
from essay3_analysis import (  # noqa: E402
    build_ancestr_sql_case,
    build_bpl_sql_case,
    extract_pf_row,
    load_and_aggregate_real_ipums,
)

ACS_PATH = ROOT / "data" / "ipums" / "raw" / "acs_2007_2024" / "usa_00012.csv.gz"
HOUSING_PARQUET = ROOT / "data" / "aux" / "clean_metro_housing.parquet"
OUT_DIR = ROOT / "papers" / "paper3_enclaves_wfh" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_module1() -> None:
    print("[02] Streaming ACS 2007-2024 for CPUMA0010 (2007-2021) & 2020-PUMA (2022-2024) analysis...")
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")

    bpl_case = build_bpl_sql_case()
    anc1_case = build_ancestr_sql_case("ANCESTR1")
    anc2_case = build_ancestr_sql_case("ANCESTR2")

    con.execute(f"""
        CREATE TEMP VIEW acs_nbhd AS
        SELECT
            YEAR,
            MET2013 AS met2013,
            STATEFIP,
            TRY_CAST(CPUMA0010 AS INTEGER) AS cpuma_raw,
            STATEFIP * 100000 + TRY_CAST(CPUMA0010 AS INTEGER) AS cpuma_id,
            STATEFIP * 100000 + TRY_CAST(PUMA AS INTEGER) AS puma_id,
            CASE WHEN BPL >= 100 THEN 1 ELSE 0 END AS is_immigrant,
            CASE
                WHEN BPL >= 100 THEN COALESCE({bpl_case}, {anc1_case})
                ELSE COALESCE({anc1_case}, {anc2_case})
            END AS eth,
            CAST(PERNUM AS INTEGER) AS PERNUM,
            CAST(PERWT AS DOUBLE) AS PERWT,
            CAST(HHWT AS DOUBLE) AS HHWT,
            CAST(AGE AS DOUBLE) AS AGE,
            CASE WHEN EDUC BETWEEN 1 AND 11 THEN CAST(EDUC AS DOUBLE) ELSE NULL END AS EDUC,
            OWNERSHP,
            CASE WHEN OWNERSHP = 2 AND RENTGRS > 0 THEN LN(RENTGRS * CPI99) ELSE NULL END AS ln_rent,
            CASE WHEN OWNERSHP = 1 AND VALUEH > 0 AND VALUEH < 9999998 THEN LN(VALUEH * CPI99) ELSE NULL END AS ln_val,
            CAST(NULLIF(ROOMS, 0) AS DOUBLE) AS ROOMS,
            CAST(NULLIF(BEDROOMS, 0) AS DOUBLE) AS BEDROOMS,
            CAST(NULLIF(BUILTYR2, 0) AS DOUBLE) AS BUILTYR2,
            CAST(NULLIF(UNITSSTR, 0) AS DOUBLE) AS UNITSSTR,
            CAST(NUMPREC AS DOUBLE) AS NUMPREC,
            MULTGEN,
            METRO
        FROM read_csv_auto('{ACS_PATH}')
        WHERE AGE BETWEEN 18 AND 64 AND SCHOOL != 2 AND MET2013 > 0 AND GQ IN (1, 2)
    """)

    # ------------------------------------------------------------------
    # 1. Neighborhood-level CPUMA0010 panel (2007, 2010, 2015, 2018, 2019, 2021)
    #    - Population shares & adult demographics: person-level weighted by PERWT
    #    - Housing outcomes & crowding/3+ gen: unique households (PERNUM = 1) weighted by HHWT
    # ------------------------------------------------------------------
    print("   [DuckDB] Aggregating CPUMA0010 neighborhood panel (2007-2021)...")
    cpuma_nbhd = con.execute("""
        SELECT
            cpuma_id,
            met2013,
            YEAR,
            SUM(PERWT) AS pop_tot,
            SUM(CASE WHEN PERNUM = 1 THEN HHWT ELSE 0.0 END) AS hh_tot,
            SUM(CASE WHEN is_immigrant = 1 THEN PERWT ELSE 0.0 END) AS imm_tot,
            100.0 * SUM(CASE WHEN is_immigrant = 1 THEN PERWT ELSE 0.0 END) / NULLIF(SUM(PERWT), 0) AS imm_share,
            -- All-household hedonic housing outcomes (unique households PERNUM = 1, HHWT-weighted)
            SUM(CASE WHEN PERNUM = 1 AND ln_rent IS NOT NULL THEN ln_rent * HHWT END) / NULLIF(SUM(CASE WHEN PERNUM = 1 AND ln_rent IS NOT NULL THEN HHWT END), 0) AS ln_rent_all,
            SUM(CASE WHEN PERNUM = 1 AND ln_val IS NOT NULL THEN ln_val * HHWT END) / NULLIF(SUM(CASE WHEN PERNUM = 1 AND ln_val IS NOT NULL THEN HHWT END), 0) AS ln_val_all,
            -- Native-headed household hedonic housing outcomes (PERNUM = 1 & native, HHWT-weighted)
            SUM(CASE WHEN PERNUM = 1 AND is_immigrant = 0 AND ln_rent IS NOT NULL THEN ln_rent * HHWT END) / NULLIF(SUM(CASE WHEN PERNUM = 1 AND is_immigrant = 0 AND ln_rent IS NOT NULL THEN HHWT END), 0) AS ln_rent_nat,
            SUM(CASE WHEN PERNUM = 1 AND is_immigrant = 0 AND ln_val IS NOT NULL THEN ln_val * HHWT END) / NULLIF(SUM(CASE WHEN PERNUM = 1 AND is_immigrant = 0 AND ln_val IS NOT NULL THEN HHWT END), 0) AS ln_val_nat,
            -- Hedonic quality controls (unique households PERNUM = 1, HHWT-weighted)
            SUM(CASE WHEN PERNUM = 1 AND ROOMS IS NOT NULL THEN ROOMS * HHWT END) / NULLIF(SUM(CASE WHEN PERNUM = 1 AND ROOMS IS NOT NULL THEN HHWT END), 0) AS mean_rooms,
            SUM(CASE WHEN PERNUM = 1 AND BEDROOMS IS NOT NULL THEN BEDROOMS * HHWT END) / NULLIF(SUM(CASE WHEN PERNUM = 1 AND BEDROOMS IS NOT NULL THEN HHWT END), 0) AS mean_bedrooms,
            SUM(CASE WHEN PERNUM = 1 AND BUILTYR2 IS NOT NULL THEN BUILTYR2 * HHWT END) / NULLIF(SUM(CASE WHEN PERNUM = 1 AND BUILTYR2 IS NOT NULL THEN HHWT END), 0) AS mean_builtyr,
            SUM(CASE WHEN PERNUM = 1 AND UNITSSTR IS NOT NULL THEN UNITSSTR * HHWT END) / NULLIF(SUM(CASE WHEN PERNUM = 1 AND UNITSSTR IS NOT NULL THEN HHWT END), 0) AS mean_unitsstr,
            -- Demographic controls (working-age adults, PERWT-weighted)
            SUM(AGE * PERWT) / NULLIF(SUM(PERWT), 0) AS mean_age,
            SUM(CASE WHEN EDUC IS NOT NULL THEN EDUC * PERWT END) / NULLIF(SUM(CASE WHEN EDUC IS NOT NULL THEN PERWT END), 0) AS mean_educ,
            -- Mechanisms (unique households PERNUM = 1, HHWT-weighted; MULTGEN = 3 strictly 3+ generations)
            SUM(CASE WHEN PERNUM = 1 AND ROOMS > 0 THEN (NUMPREC / ROOMS) * HHWT END) / NULLIF(SUM(CASE WHEN PERNUM = 1 AND ROOMS > 0 THEN HHWT END), 0) AS crowding_all,
            SUM(CASE WHEN PERNUM = 1 AND is_immigrant = 0 AND ROOMS > 0 THEN (NUMPREC / ROOMS) * HHWT END) / NULLIF(SUM(CASE WHEN PERNUM = 1 AND is_immigrant = 0 AND ROOMS > 0 THEN HHWT END), 0) AS crowding_nat,
            SUM(CASE WHEN PERNUM = 1 AND MULTGEN > 0 THEN (CASE WHEN MULTGEN = 3 THEN 1.0 ELSE 0.0 END) * HHWT END) / NULLIF(SUM(CASE WHEN PERNUM = 1 AND MULTGEN > 0 THEN HHWT END), 0) AS multgen_all,
            SUM(CASE WHEN PERNUM = 1 AND OWNERSHP IN (1, 2) THEN (CASE WHEN OWNERSHP = 1 THEN 1.0 ELSE 0.0 END) * HHWT END) / NULLIF(SUM(CASE WHEN PERNUM = 1 AND OWNERSHP IN (1, 2) THEN HHWT END), 0) AS owner_all,
            SUM(CASE WHEN PERNUM = 1 AND METRO IN (2, 3) THEN (CASE WHEN METRO = 3 THEN 1.0 ELSE 0.0 END) * HHWT END) / NULLIF(SUM(CASE WHEN PERNUM = 1 AND METRO IN (2, 3) THEN HHWT END), 0) AS suburb_share
        FROM acs_nbhd
        WHERE YEAR <= 2021 AND cpuma_raw > 0
        GROUP BY cpuma_id, met2013, YEAR
    """).df()

    # ------------------------------------------------------------------
    # 2. Co-ethnic U.S.-born adult housing experience in CPUMA0010 (eth x cpuma x year, 2007-2021)
    #    - Strictly person-weighted (PERWT) across U.S.-born co-ethnic working-age adults,
    #      measuring the housing conditions experienced by co-ethnic adults (including those
    #      living in parent-headed or 3+ generation households).
    # ------------------------------------------------------------------
    print("   [DuckDB] Aggregating co-ethnic CPUMA0010 panel (2007-2021, PERWT-weighted adult experience)...")
    cpuma_eth = con.execute("""
        WITH cp_pop AS (
            SELECT cpuma_id, met2013, YEAR, SUM(PERWT) AS cp_pop
            FROM acs_nbhd WHERE YEAR <= 2021 AND cpuma_raw > 0
            GROUP BY cpuma_id, met2013, YEAR
        ),
        cp_imm AS (
            SELECT eth, cpuma_id, met2013, YEAR, SUM(PERWT) AS imm_pop
            FROM acs_nbhd WHERE YEAR <= 2021 AND cpuma_raw > 0 AND is_immigrant = 1 AND eth IS NOT NULL
            GROUP BY eth, cpuma_id, met2013, YEAR
        ),
        cp_nat AS (
            SELECT
                eth, cpuma_id, met2013, YEAR,
                SUM(PERWT) AS nat_pop,
                SUM(CASE WHEN ln_rent IS NOT NULL THEN ln_rent * PERWT END) / NULLIF(SUM(CASE WHEN ln_rent IS NOT NULL THEN PERWT END), 0) AS ln_rent,
                SUM(CASE WHEN ln_val IS NOT NULL THEN ln_val * PERWT END) / NULLIF(SUM(CASE WHEN ln_val IS NOT NULL THEN PERWT END), 0) AS ln_val,
                SUM(CASE WHEN ROOMS IS NOT NULL THEN ROOMS * PERWT END) / NULLIF(SUM(CASE WHEN ROOMS IS NOT NULL THEN PERWT END), 0) AS mean_rooms,
                SUM(CASE WHEN BEDROOMS IS NOT NULL THEN BEDROOMS * PERWT END) / NULLIF(SUM(CASE WHEN BEDROOMS IS NOT NULL THEN PERWT END), 0) AS mean_bedrooms,
                SUM(CASE WHEN BUILTYR2 IS NOT NULL THEN BUILTYR2 * PERWT END) / NULLIF(SUM(CASE WHEN BUILTYR2 IS NOT NULL THEN PERWT END), 0) AS mean_builtyr,
                SUM(CASE WHEN UNITSSTR IS NOT NULL THEN UNITSSTR * PERWT END) / NULLIF(SUM(CASE WHEN UNITSSTR IS NOT NULL THEN PERWT END), 0) AS mean_unitsstr,
                SUM(AGE * PERWT) / NULLIF(SUM(PERWT), 0) AS mean_age,
                SUM(CASE WHEN EDUC IS NOT NULL THEN EDUC * PERWT END) / NULLIF(SUM(CASE WHEN EDUC IS NOT NULL THEN PERWT END), 0) AS mean_educ,
                SUM(CASE WHEN ROOMS > 0 THEN (NUMPREC / ROOMS) * PERWT END) / NULLIF(SUM(CASE WHEN ROOMS > 0 THEN PERWT END), 0) AS crowding,
                SUM(CASE WHEN MULTGEN > 0 THEN (CASE WHEN MULTGEN = 3 THEN 1.0 ELSE 0.0 END) * PERWT END) / NULLIF(SUM(CASE WHEN MULTGEN > 0 THEN PERWT END), 0) AS multgen,
                SUM(CASE WHEN OWNERSHP IN (1, 2) THEN (CASE WHEN OWNERSHP = 1 THEN 1.0 ELSE 0.0 END) * PERWT END) / NULLIF(SUM(CASE WHEN OWNERSHP IN (1, 2) THEN PERWT END), 0) AS owner,
                SUM(CASE WHEN METRO IN (2, 3) THEN (CASE WHEN METRO = 3 THEN 1.0 ELSE 0.0 END) * PERWT END) / NULLIF(SUM(CASE WHEN METRO IN (2, 3) THEN PERWT END), 0) AS suburb
            FROM acs_nbhd
            WHERE YEAR <= 2021 AND cpuma_raw > 0 AND is_immigrant = 0 AND eth IS NOT NULL
            GROUP BY eth, cpuma_id, met2013, YEAR
        )
        SELECT
            n.*,
            100.0 * COALESCE(i.imm_pop, 0.0) / p.cp_pop AS coeth_share_cpuma
        FROM cp_nat n
        JOIN cp_pop p ON n.cpuma_id = p.cpuma_id AND n.met2013 = p.met2013 AND n.YEAR = p.YEAR
        LEFT JOIN cp_imm i ON n.eth = i.eth AND n.cpuma_id = i.cpuma_id AND n.met2013 = i.met2013 AND n.YEAR = i.YEAR
    """).df()

    # ------------------------------------------------------------------
    # 3. Post-COVID 2022-2024 consistent 2020-Census PUMA panel
    #    - Housing outcomes strictly on unique households (PERNUM = 1, HHWT-weighted)
    # ------------------------------------------------------------------
    print("   [DuckDB] Aggregating post-COVID 2022-2024 consistent PUMA panel...")
    puma_2224 = con.execute("""
        SELECT
            puma_id,
            met2013,
            YEAR,
            SUM(PERWT) AS pop_tot,
            SUM(CASE WHEN PERNUM = 1 THEN HHWT ELSE 0.0 END) AS hh_tot,
            100.0 * SUM(CASE WHEN is_immigrant = 1 THEN PERWT ELSE 0.0 END) / NULLIF(SUM(PERWT), 0) AS imm_share,
            SUM(CASE WHEN PERNUM = 1 AND ln_rent IS NOT NULL THEN ln_rent * HHWT END) / NULLIF(SUM(CASE WHEN PERNUM = 1 AND ln_rent IS NOT NULL THEN HHWT END), 0) AS ln_rent_all,
            SUM(CASE WHEN PERNUM = 1 AND ln_val IS NOT NULL THEN ln_val * HHWT END) / NULLIF(SUM(CASE WHEN PERNUM = 1 AND ln_val IS NOT NULL THEN HHWT END), 0) AS ln_val_all,
            SUM(CASE WHEN PERNUM = 1 AND ROOMS IS NOT NULL THEN ROOMS * HHWT END) / NULLIF(SUM(CASE WHEN PERNUM = 1 AND ROOMS IS NOT NULL THEN HHWT END), 0) AS mean_rooms,
            SUM(CASE WHEN PERNUM = 1 AND BEDROOMS IS NOT NULL THEN BEDROOMS * HHWT END) / NULLIF(SUM(CASE WHEN PERNUM = 1 AND BEDROOMS IS NOT NULL THEN HHWT END), 0) AS mean_bedrooms,
            SUM(CASE WHEN PERNUM = 1 AND BUILTYR2 IS NOT NULL THEN BUILTYR2 * HHWT END) / NULLIF(SUM(CASE WHEN PERNUM = 1 AND BUILTYR2 IS NOT NULL THEN HHWT END), 0) AS mean_builtyr,
            SUM(CASE WHEN PERNUM = 1 AND UNITSSTR IS NOT NULL THEN UNITSSTR * HHWT END) / NULLIF(SUM(CASE WHEN PERNUM = 1 AND UNITSSTR IS NOT NULL THEN HHWT END), 0) AS mean_unitsstr,
            SUM(AGE * PERWT) / NULLIF(SUM(PERWT), 0) AS mean_age,
            SUM(CASE WHEN EDUC IS NOT NULL THEN EDUC * PERWT END) / NULLIF(SUM(CASE WHEN EDUC IS NOT NULL THEN PERWT END), 0) AS mean_educ,
            SUM(CASE WHEN PERNUM = 1 AND ROOMS > 0 THEN (NUMPREC / ROOMS) * HHWT END) / NULLIF(SUM(CASE WHEN PERNUM = 1 AND ROOMS > 0 THEN HHWT END), 0) AS crowding_all,
            SUM(CASE WHEN PERNUM = 1 AND MULTGEN > 0 THEN (CASE WHEN MULTGEN = 3 THEN 1.0 ELSE 0.0 END) * HHWT END) / NULLIF(SUM(CASE WHEN PERNUM = 1 AND MULTGEN > 0 THEN HHWT END), 0) AS multgen_all,
            SUM(CASE WHEN PERNUM = 1 AND OWNERSHP IN (1, 2) THEN (CASE WHEN OWNERSHP = 1 THEN 1.0 ELSE 0.0 END) * HHWT END) / NULLIF(SUM(CASE WHEN PERNUM = 1 AND OWNERSHP IN (1, 2) THEN HHWT END), 0) AS owner_all,
            SUM(CASE WHEN PERNUM = 1 AND METRO IN (2, 3) THEN (CASE WHEN METRO = 3 THEN 1.0 ELSE 0.0 END) * HHWT END) / NULLIF(SUM(CASE WHEN PERNUM = 1 AND METRO IN (2, 3) THEN HHWT END), 0) AS suburb_share
        FROM acs_nbhd
        WHERE YEAR >= 2022
        GROUP BY puma_id, met2013, YEAR
    """).df()

    # ------------------------------------------------------------------
    # 4. MSA-level benchmark (2007-2024 IPUMS hedonic rent & value, unique households PERNUM = 1)
    # ------------------------------------------------------------------
    print("   [DuckDB] Aggregating MSA-level hedonic benchmark (2007-2024)...")
    msa_bench = con.execute("""
        SELECT
            met2013,
            YEAR,
            SUM(PERWT) AS pop_tot,
            SUM(CASE WHEN PERNUM = 1 THEN HHWT ELSE 0.0 END) AS hh_tot,
            100.0 * SUM(CASE WHEN is_immigrant = 1 THEN PERWT ELSE 0.0 END) / NULLIF(SUM(PERWT), 0) AS imm_share,
            SUM(CASE WHEN PERNUM = 1 AND ln_rent IS NOT NULL THEN ln_rent * HHWT END) / NULLIF(SUM(CASE WHEN PERNUM = 1 AND ln_rent IS NOT NULL THEN HHWT END), 0) AS ln_rent_all,
            SUM(CASE WHEN PERNUM = 1 AND ln_val IS NOT NULL THEN ln_val * HHWT END) / NULLIF(SUM(CASE WHEN PERNUM = 1 AND ln_val IS NOT NULL THEN HHWT END), 0) AS ln_val_all,
            SUM(CASE WHEN PERNUM = 1 AND ROOMS IS NOT NULL THEN ROOMS * HHWT END) / NULLIF(SUM(CASE WHEN PERNUM = 1 AND ROOMS IS NOT NULL THEN HHWT END), 0) AS mean_rooms,
            SUM(CASE WHEN PERNUM = 1 AND BEDROOMS IS NOT NULL THEN BEDROOMS * HHWT END) / NULLIF(SUM(CASE WHEN PERNUM = 1 AND BEDROOMS IS NOT NULL THEN HHWT END), 0) AS mean_bedrooms,
            SUM(CASE WHEN PERNUM = 1 AND BUILTYR2 IS NOT NULL THEN BUILTYR2 * HHWT END) / NULLIF(SUM(CASE WHEN PERNUM = 1 AND BUILTYR2 IS NOT NULL THEN HHWT END), 0) AS mean_builtyr,
            SUM(CASE WHEN PERNUM = 1 AND UNITSSTR IS NOT NULL THEN UNITSSTR * HHWT END) / NULLIF(SUM(CASE WHEN PERNUM = 1 AND UNITSSTR IS NOT NULL THEN HHWT END), 0) AS mean_unitsstr,
            SUM(AGE * PERWT) / NULLIF(SUM(PERWT), 0) AS mean_age,
            SUM(CASE WHEN EDUC IS NOT NULL THEN EDUC * PERWT END) / NULLIF(SUM(CASE WHEN EDUC IS NOT NULL THEN PERWT END), 0) AS mean_educ,
            SUM(CASE WHEN PERNUM = 1 AND ROOMS > 0 THEN (NUMPREC / ROOMS) * HHWT END) / NULLIF(SUM(CASE WHEN PERNUM = 1 AND ROOMS > 0 THEN HHWT END), 0) AS crowding_all,
            SUM(CASE WHEN PERNUM = 1 AND MULTGEN > 0 THEN (CASE WHEN MULTGEN = 3 THEN 1.0 ELSE 0.0 END) * HHWT END) / NULLIF(SUM(CASE WHEN PERNUM = 1 AND MULTGEN > 0 THEN HHWT END), 0) AS multgen_all,
            SUM(CASE WHEN PERNUM = 1 AND OWNERSHP IN (1, 2) THEN (CASE WHEN OWNERSHP = 1 THEN 1.0 ELSE 0.0 END) * HHWT END) / NULLIF(SUM(CASE WHEN PERNUM = 1 AND OWNERSHP IN (1, 2) THEN HHWT END), 0) AS owner_all
        FROM acs_nbhd
        GROUP BY met2013, YEAR
    """).df()

    # Prepare FE keys
    cpuma_nbhd["met_year"] = cpuma_nbhd["met2013"].astype(str) + "_" + cpuma_nbhd["YEAR"].astype(str)
    cpuma_nbhd["cpuma_met"] = cpuma_nbhd["cpuma_id"].astype(str) + "_" + cpuma_nbhd["met2013"].astype(str)

    cpuma_eth["met_year"] = cpuma_eth["met2013"].astype(str) + "_" + cpuma_eth["YEAR"].astype(str)
    cpuma_eth["eth_year"] = cpuma_eth["eth"].astype(str) + "_" + cpuma_eth["YEAR"].astype(str)
    cpuma_eth["eth_cpuma"] = cpuma_eth["eth"].astype(str) + "_" + cpuma_eth["cpuma_id"].astype(str) + "_" + cpuma_eth["met2013"].astype(str)

    puma_2224["met_year"] = puma_2224["met2013"].astype(str) + "_" + puma_2224["YEAR"].astype(str)
    puma_2224["puma_met"] = puma_2224["puma_id"].astype(str) + "_" + puma_2224["met2013"].astype(str)

    hed = "mean_rooms + mean_bedrooms + mean_builtyr + mean_unitsstr"
    struct_all = "mean_builtyr + mean_unitsstr + owner_all"
    struct_eth = "mean_builtyr + mean_unitsstr + owner"
    demo = "mean_age + mean_educ"
    rows = []

    # --- Panel A: MSA-Level Benchmark (2007-2024, met2013 + YEAR FEs) ---
    for ycol, lbl, ctrl in [
        ("ln_rent_all", "MSA_2007_2024_lnRent_StructOnly", f"{struct_all} + {demo}"),
        ("ln_rent_all", "MSA_2007_2024_lnRent_Hedonic", f"{hed} + {demo}"),
        ("ln_val_all", "MSA_2007_2024_lnVal_StructOnly", f"{struct_all} + {demo}"),
        ("ln_val_all", "MSA_2007_2024_lnVal_Hedonic", f"{hed} + {demo}"),
        ("crowding_all", "MSA_2007_2024_Crowding", demo),
        ("multgen_all", "MSA_2007_2024_Multigen", demo),
    ]:
        sub = msa_bench.dropna(subset=[ycol, "imm_share", "mean_rooms", "mean_bedrooms", "mean_builtyr", "mean_unitsstr", "owner_all", "mean_age", "mean_educ", "pop_tot"])
        m = pf.feols(f"{ycol} ~ imm_share + {ctrl} | met2013 + YEAR", data=sub, weights="pop_tot", vcov={"CRV1": "met2013"})
        rows.append(extract_pf_row(m, "imm_share", lbl, {"panel": "A_MSA_2007_2024", "n_units": int(sub["met2013"].nunique())}))

    # --- Panel B: Within-MSA CPUMA0010 Neighborhood (2007-2021, met_year + cpuma_met FEs) ---
    for ycol, lbl, ctrl in [
        ("ln_rent_all", "CPUMA_2007_2021_lnRent_Unadj", demo),
        ("ln_rent_all", "CPUMA_2007_2021_lnRent_StructOnly", f"{struct_all} + {demo}"),
        ("ln_rent_all", "CPUMA_2007_2021_lnRent_Hedonic", f"{hed} + {demo}"),
        ("ln_val_all", "CPUMA_2007_2021_lnVal_Unadj", demo),
        ("ln_val_all", "CPUMA_2007_2021_lnVal_StructOnly", f"{struct_all} + {demo}"),
        ("ln_val_all", "CPUMA_2007_2021_lnVal_Hedonic", f"{hed} + {demo}"),
        ("ln_rent_nat", "CPUMA_2007_2021_lnRent_Native_StructOnly", f"{struct_all} + {demo}"),
        ("ln_rent_nat", "CPUMA_2007_2021_lnRent_Native_Hedonic", f"{hed} + {demo}"),
        ("ln_val_nat", "CPUMA_2007_2021_lnVal_Native_StructOnly", f"{struct_all} + {demo}"),
        ("ln_val_nat", "CPUMA_2007_2021_lnVal_Native_Hedonic", f"{hed} + {demo}"),
        ("crowding_all", "CPUMA_2007_2021_Crowding_All", demo),
        ("crowding_nat", "CPUMA_2007_2021_Crowding_Native", demo),
        ("multgen_all", "CPUMA_2007_2021_Multigen_All", demo),
        ("owner_all", "CPUMA_2007_2021_Homeownership_All", demo),
        ("suburb_share", "CPUMA_2007_2021_Suburban_METRO3", demo),
    ]:
        sub = cpuma_nbhd.dropna(subset=[ycol, "imm_share", "mean_rooms", "mean_bedrooms", "mean_builtyr", "mean_unitsstr", "owner_all", "mean_age", "mean_educ", "pop_tot"])
        m = pf.feols(f"{ycol} ~ imm_share + {ctrl} | met_year + cpuma_met", data=sub, weights="pop_tot", vcov={"CRV1": "met2013"})
        rows.append(extract_pf_row(m, "imm_share", lbl, {"panel": "B_CPUMA_WithinMSA_2007_2021", "n_units": int(sub["cpuma_met"].nunique())}))

    # --- Panel C: Co-Ethnic CPUMA0010 Neighborhood (2007-2021, eth_year + met_year + eth_cpuma FEs) ---
    for ycol, lbl, ctrl in [
        ("ln_rent", "CoEth_CPUMA_2007_2021_lnRent_StructOnly", f"{struct_eth} + {demo}"),
        ("ln_rent", "CoEth_CPUMA_2007_2021_lnRent_Hedonic", f"{hed} + {demo}"),
        ("ln_val", "CoEth_CPUMA_2007_2021_lnVal_StructOnly", f"{struct_eth} + {demo}"),
        ("ln_val", "CoEth_CPUMA_2007_2021_lnVal_Hedonic", f"{hed} + {demo}"),
        ("crowding", "CoEth_CPUMA_2007_2021_Crowding", demo),
        ("multgen", "CoEth_CPUMA_2007_2021_Multigen", demo),
        ("owner", "CoEth_CPUMA_2007_2021_Homeownership", demo),
        ("suburb", "CoEth_CPUMA_2007_2021_Suburban_METRO3", demo),
    ]:
        sub = cpuma_eth.dropna(subset=[ycol, "coeth_share_cpuma", "mean_rooms", "mean_bedrooms", "mean_builtyr", "mean_unitsstr", "owner", "mean_age", "mean_educ", "nat_pop"])
        m = pf.feols(f"{ycol} ~ coeth_share_cpuma + {ctrl} | eth_year + met_year + eth_cpuma", data=sub, weights="nat_pop", vcov={"CRV1": "met2013"})
        rows.append(extract_pf_row(m, "coeth_share_cpuma", lbl, {"panel": "C_CoEth_CPUMA_2007_2021", "n_units": int(sub["eth_cpuma"].nunique())}))

    # --- Panel D: Post-COVID 2022-2024 Consistent 2020-Census PUMA (met_year + puma_met FEs) ---
    for ycol, lbl, ctrl in [
        ("ln_rent_all", "PUMA_2022_2024_lnRent_StructOnly", f"{struct_all} + {demo}"),
        ("ln_rent_all", "PUMA_2022_2024_lnRent_Hedonic", f"{hed} + {demo}"),
        ("ln_val_all", "PUMA_2022_2024_lnVal_StructOnly", f"{struct_all} + {demo}"),
        ("ln_val_all", "PUMA_2022_2024_lnVal_Hedonic", f"{hed} + {demo}"),
        ("crowding_all", "PUMA_2022_2024_Crowding", demo),
        ("multgen_all", "PUMA_2022_2024_Multigen", demo),
        ("owner_all", "PUMA_2022_2024_Homeownership", demo),
    ]:
        sub = puma_2224.dropna(subset=[ycol, "imm_share", "mean_rooms", "mean_bedrooms", "mean_builtyr", "mean_unitsstr", "owner_all", "mean_age", "mean_educ", "pop_tot"])
        m = pf.feols(f"{ycol} ~ imm_share + {ctrl} | met_year + puma_met", data=sub, weights="pop_tot", vcov={"CRV1": "met2013"})
        rows.append(extract_pf_row(m, "imm_share", lbl, {"panel": "D_PUMA_WithinMSA_2022_2024", "n_units": int(sub["puma_met"].nunique())}))

    # ------------------------------------------------------------------
    # 5. Empirical Nickell (1981) Bias Demonstration on Song's Table 7 (1980-2024)
    #    Strictly on the EXACT SAME common cell sample (cells with t-2 lag available)
    #    and the EXACT SAME differenced controls (d_mean_educ_ah + d_mean_ln_wage_ah):
    #      (a) Dynamic FE WITH lagged level hp_{t-1} + eth_met FE:
    #          d_hp_t = gamma * hp_{t-1} + beta * d_share_t + ... => implied persistence rho_FE = 1 + gamma
    #      (b) Static FD WITHOUT lagged hp_{t-1} (same eth_year + met_year + eth_met FEs)
    #      (c) Anderson-Hsiao (1981) IV: d_hp_t = rho_AH * d_hp_{t-1} + beta * d_share_t + ...
    #          with d_hp_{t-1} instrumented by hp_{t-2} (eth_year + met_year FEs; unit FE differenced out)
    # ------------------------------------------------------------------
    print("   [Nickell Demo] Loading 1980-2024 cell panel (common sample, exact rho_FE = 1 + gamma vs rho_AH)...")
    cell, _, _, _, _ = load_and_aggregate_real_ipums()
    cell = cell.sort_values(["eth", "met2013", "YEAR"]).reset_index(drop=True)
    for col in ["mean_ln_rent", "mean_ln_val", "mean_educ", "mean_ln_wage"]:
        cell[f"{col}_lag1"] = cell.groupby(["eth", "met2013"])[col].shift(1)
        cell[f"{col}_lag2"] = cell.groupby(["eth", "met2013"])[col].shift(2)
        cell[f"d_{col}_ah"] = cell[col] - cell[f"{col}_lag1"]
        cell[f"d_{col}_lag1"] = cell[f"{col}_lag1"] - cell[f"{col}_lag2"]

    nickell_rows = []
    fe_3way = "eth_year + met_year + eth_met"
    fe_2way = "eth_year + met_year"

    for hp_col, lbl in [("mean_ln_rent", "Rent"), ("mean_ln_val", "HouseValue")]:
        d_hp = f"d_{hp_col}_ah"
        hp_lag1 = f"{hp_col}_lag1"
        d_hp_lag1 = f"d_{hp_col}_lag1"
        hp_lag2 = f"{hp_col}_lag2"

        # Common sample across ALL 3 estimators (exact same cells, non-singleton in eth_met, same controls!)
        sub_common = cell.dropna(
            subset=[d_hp, "d_share_rct", hp_lag1, d_hp_lag1, hp_lag2, "d_mean_educ_ah", "d_mean_ln_wage_ah", "nat_pop"]
        ).copy()
        sub_common = sub_common[sub_common.groupby("eth_met")["YEAR"].transform("count") >= 2].copy()

        # (a) Song Table 7 Dynamic FE WITH lagged level hp_{t-1} + eth_met FE
        #     d_hp_t = gamma * hp_{t-1} + ... => implied persistence rho_FE = 1 + gamma
        m_a = pf.feols(
            f"{d_hp} ~ d_share_rct + {hp_lag1} + d_mean_educ_ah + d_mean_ln_wage_ah | {fe_3way}",
            data=sub_common, weights="nat_pop", vcov={"CRV1": "met2013"},
        )
        gamma_hat = float(m_a.coef().get(hp_lag1, np.nan))
        gamma_se = float(m_a.se().get(hp_lag1, np.nan))
        rho_fe = 1.0 + gamma_hat
        r_a = extract_pf_row(m_a, "d_share_rct", f"NickellDemo_WithLag_FE_{lbl}", {
            "outcome": lbl, "estimator": "Song_WithLag_FE",
            "lag_coef": gamma_hat,
            "implied_rho": rho_fe,
            "lag_se": gamma_se,
            "lag_tstat": float(m_a.tstat().get(hp_lag1, np.nan)),
        })
        nickell_rows.append(r_a)

        # (b) Static FD WITHOUT lagged level hp_{t-1} (same sample, controls, and 3-way FEs)
        m_b = pf.feols(
            f"{d_hp} ~ d_share_rct + d_mean_educ_ah + d_mean_ln_wage_ah | {fe_3way}",
            data=sub_common, weights="nat_pop", vcov={"CRV1": "met2013"},
        )
        r_b = extract_pf_row(m_b, "d_share_rct", f"NickellDemo_NoLag_FE_{lbl}", {
            "outcome": lbl, "estimator": "StaticFD_NoLag_FE",
            "lag_coef": np.nan, "implied_rho": np.nan, "lag_se": np.nan, "lag_tstat": np.nan,
        })
        nickell_rows.append(r_b)

        # (c) Anderson-Hsiao (1981) IV on exact same sample & controls:
        #     d_hp_t = rho_AH * d_hp_{t-1} + beta * d_share_t + ... | eth_year + met_year | d_hp_{t-1} ~ hp_{t-2}
        m_c = pf.feols(
            f"{d_hp} ~ d_share_rct + d_mean_educ_ah + d_mean_ln_wage_ah | {fe_2way} | {d_hp_lag1} ~ {hp_lag2}",
            data=sub_common, weights="nat_pop", vcov={"CRV1": "met2013"},
        )
        rho_ah = float(m_c.coef().get(d_hp_lag1, np.nan))
        r_c = extract_pf_row(m_c, "d_share_rct", f"NickellDemo_AndersonHsiao_IV_{lbl}", {
            "outcome": lbl, "estimator": "AndersonHsiao_IV",
            "lag_coef": rho_ah,
            "implied_rho": rho_ah,
            "lag_se": float(m_c.se().get(d_hp_lag1, np.nan)),
            "lag_tstat": float(m_c.tstat().get(d_hp_lag1, np.nan)),
        })
        nickell_rows.append(r_c)

    nickell_df = pd.DataFrame(nickell_rows)
    nickell_df.to_csv(OUT_DIR / "module1_nickell_demo.csv", index=False)

    res_df = pd.DataFrame(rows)
    res_df.to_csv(OUT_DIR / "module1_cpuma_housing_crowding.csv", index=False)

    meta = {
        "n_cpuma_nbhd_rows": int(len(cpuma_nbhd)),
        "n_unique_cpumas": int(cpuma_nbhd["cpuma_met"].nunique()),
        "n_cpuma_eth_rows": int(len(cpuma_eth)),
        "n_puma_2224_rows": int(len(puma_2224)),
        "n_unique_pumas_2224": int(puma_2224["puma_met"].nunique()),
        "n_msa_rows": int(len(msa_bench)),
        "results": rows,
        "nickell_demo": nickell_rows,
    }
    (OUT_DIR / "module1_summary.json").write_text(json.dumps(meta, indent=2, default=str))

    print("\n=== MODULE 1: CPUMA0010 & PUMA NEIGHBORHOOD HOUSING & CROWDING RESULTS ===")
    for _, r in res_df.iterrows():
        sig = "***" if r["pval"] < 0.01 else ("**" if r["pval"] < 0.05 else ("*" if r["pval"] < 0.10 else ""))
        print(f"  {r['spec']:44s}: coef={r['coef']:+.5f} ({r['se']:.5f}){sig:3s} | t={r['tstat']:+.2f} | N={int(r['n']):,}")

    print("\n=== MODULE 1: EMPIRICAL NICKELL (1981) BIAS DEMONSTRATION (COMMON SAMPLE, EXACT RHO) ===")
    for _, r in nickell_df.iterrows():
        sig = "***" if r["pval"] < 0.01 else ("**" if r["pval"] < 0.05 else ("*" if r["pval"] < 0.10 else ""))
        lag_s = f"rho={r['implied_rho']:+.4f} (raw={r['lag_coef']:+.4f}, se={r['lag_se']:.4f})" if pd.notna(r["implied_rho"]) else "rho=omitted"
        f_s = f" | F1={r['first_stage_f']:.1f}" if "first_stage_f" in r and pd.notna(r.get("first_stage_f")) else ""
        print(f"  {r['spec']:36s}: beta={r['coef']:+.5f} ({r['se']:.5f}){sig:3s} | {lag_s}{f_s} | N={int(r['n']):,}")


if __name__ == "__main__":
    run_module1()
