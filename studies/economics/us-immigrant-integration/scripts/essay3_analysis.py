#!/usr/bin/env python3
"""
Essay 3 Empirical Replication & Novel Extension Pipeline
(Tao Song PhD Dissertation, Chapter 3: "Why Live in an Ethnic Enclave?")

Implements three core empirical pillars on 1980-2024 IPUMS USA + FHFA/Zillow housing data:
  1. Resolving the Housing Price Puzzle (Table 7 stress-test, hedonic quality controls,
     crowding/multigenerational/suburban sorting, and MSA-level FHFA/Zillow Saiz test).
  2. True Physical Migration vs. Fertility-in-Place (BPL != STATEFIP) &
     Duncan-Trejo (2011, 2017) Direct Ethnic Attrition Test (ANCESTR1 vs ANCESTR2 switching).
  3. Modern Dynamic Shift-Share (Card-Peri 2016 flow spec + PPML `fepois` +
     Jaeger-Ruist-Stuhler 2018 2-instrument dynamic leave-out Bartik IV) &
     2019-2024 Remote-Work (WFH, TRANWORK==80) Pure Amenity Demand Test.

Usage:
  # End-to-end smoke test on synthetic microdata (verifies every estimator, table & figure):
  .venv/bin/python scripts/essay3_analysis.py --smoke-test

  # Run full pipeline on downloaded IPUMS USA extracts:
  .venv/bin/python scripts/essay3_analysis.py
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import duckdb
import matplotlib
import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
IPUMS_RAW = ROOT / "data" / "ipums" / "raw"
HOUSING_PARQUET = ROOT / "data" / "aux" / "clean_metro_housing.parquet"
MSA_CROSSWALK = ROOT / "data" / "aux" / "song_msa_crosswalk.csv"
OCC_TASKS_PARQUET = ROOT / "data" / "aux" / "clean_occ_tasks.parquet"

# ---------------------------------------------------------------------------
# Harmonized Ethnicity Mapping (IPUMS BPL & ANCESTR1/ANCESTR2 -> unified eth_id)
# Consistent 1980-2024 mapping covering major European, Latin American, Asian,
# Caribbean, Middle Eastern, and African origin groups.
# ---------------------------------------------------------------------------
# Each entry: eth_name -> (bpl_codes, ancestr_ranges, attrition_tier)
# attrition_tier: "high" (multi-gen European/Hispanic), "low" (recent Asian/African)
ETHNICITY_MAP: dict[str, tuple[list[int], list[tuple[int, int]], str]] = {
    "Mexican": ([200], [(210, 219)], "high"),
    "Puerto_Rican": ([110], [(261, 261)], "high"),
    "Cuban": ([250], [(271, 271)], "high"),
    "Dominican": ([260], [(275, 275)], "high"),
    "Salvadoran": ([210], [(226, 226)], "high"),
    "Guatemalan": ([210], [(222, 222)], "high"),
    "Colombian": ([300], [(231, 231)], "high"),
    "Peruvian": ([300], [(237, 237)], "high"),
    "Ecuadorian": ([300], [(232, 232)], "high"),
    "German": ([453], [(32, 34)], "high"),
    "Irish": ([414], [(50, 50)], "high"),
    "Italian": ([434], [(51, 51)], "high"),
    "English": ([410], [(22, 22)], "high"),
    "Polish": ([455], [(142, 142)], "high"),
    "French": ([421], [(26, 26)], "high"),
    "Scottish": ([411], [(84, 88)], "high"),
    "Russian": ([465], [(148, 148)], "high"),
    "Dutch": ([425], [(21, 21)], "high"),
    "Swedish": ([405], [(89, 89)], "high"),
    "Norwegian": ([404], [(82, 82)], "high"),
    "Greek": ([433], [(46, 46)], "high"),
    "Portuguese": ([436], [(77, 77)], "high"),
    "Canadian": ([150], [(931, 935)], "high"),
    "Chinese": ([500], [(706, 709)], "low"),
    "Indian": ([521], [(615, 615)], "low"),
    "Filipino": ([515], [(720, 720)], "low"),
    "Vietnamese": ([518], [(785, 785)], "low"),
    "Korean": ([502], [(750, 750)], "low"),
    "Japanese": ([501], [(740, 740)], "low"),
    "Pakistani": ([522], [(680, 680)], "low"),
    "Iranian": ([532], [(416, 416)], "low"),
    "Jamaican": ([260], [(308, 308)], "low"),
    "Haitian": ([260], [(336, 336)], "low"),
    "Nigerian": ([600], [(553, 553)], "low"),
    "Ethiopian": ([600], [(522, 522)], "low"),
}

HIGH_ATTRITION_GROUPS = {k for k, v in ETHNICITY_MAP.items() if v[2] == "high"}
LOW_ATTRITION_GROUPS = {k for k, v in ETHNICITY_MAP.items() if v[2] == "low"}

# Teleworkable major SOC 2-digit prefixes (Dingel & Neiman 2020 high-teleworkability groups)
TELEWORK_SOC2 = {"11", "13", "15", "17", "19", "23", "25", "27", "41", "43"}


# ---------------------------------------------------------------------------
# Exact FWL Alternating-Projection Demeaning + Weighted 2SLS (for 2+ endog vars)
# Required because pyfixest 0.70.0 does not yet support multiple endogenous vars.
# ---------------------------------------------------------------------------
def fwl_demean(
    df: pd.DataFrame,
    cols: list[str],
    fe_cols: list[str],
    weights: np.ndarray,
    tol: float = 1e-9,
    max_iter: int = 200,
) -> np.ndarray:
    """Weighted alternating-projection demeaning across multiple high-dim FEs."""
    mat = df[cols].to_numpy(dtype=float, copy=True)
    w = weights.astype(float)
    fe_codes = [pd.Categorical(df[c]).codes for c in fe_cols]
    n_groups = [int(c.max()) + 1 for c in fe_codes]

    for _ in range(max_iter):
        max_diff = 0.0
        for codes, ng in zip(fe_codes, n_groups):
            w_sum = np.bincount(codes, weights=w, minlength=ng)
            w_sum = np.where(w_sum == 0, 1.0, w_sum)
            for j in range(mat.shape[1]):
                num = np.bincount(codes, weights=w * mat[:, j], minlength=ng)
                mean_g = num / w_sum
                delta = mean_g[codes]
                dmax = float(np.max(np.abs(delta)))
                if dmax > max_diff:
                    max_diff = dmax
                mat[:, j] -= delta
        if max_diff < tol:
            break
    return mat


def fit_fwl_2sls(
    df: pd.DataFrame,
    y_col: str,
    endog_cols: list[str],
    instr_cols: list[str],
    exog_cols: list[str],
    fe_cols: list[str],
    cluster_cols: list[str],
    weight_col: str | None = None,
) -> dict[str, Any]:
    """
    Exact weighted FWL-demeaned 2SLS supporting multiple endogenous variables
    (e.g., Jaeger-Ruist-Stuhler 2018 dynamic shift-share IV), with CRV1 or
    two-way Cameron-Gelbach-Miller (2011) clustered SEs and Cragg-Donald /
    first-stage F-statistics.
    """
    needed = list(dict.fromkeys([y_col, *endog_cols, *instr_cols, *exog_cols, *fe_cols, *cluster_cols]))
    if weight_col:
        needed.append(weight_col)
    sub = df[needed].dropna().copy()
    n = len(sub)
    if n < 30:
        return {"n": n, "status": "insufficient_obs"}

    w = sub[weight_col].to_numpy(dtype=float) if weight_col else np.ones(n, dtype=float)
    w = np.clip(w, 1e-12, None)
    sw = np.sqrt(w)

    all_vars = [y_col, *endog_cols, *exog_cols, *instr_cols]
    dem = fwl_demean(sub, all_vars, fe_cols, w)

    y = dem[:, 0] * sw
    k_end = len(endog_cols)
    k_ex = len(exog_cols)
    k_ins = len(instr_cols)

    X_end = dem[:, 1 : 1 + k_end] * sw[:, None]
    X_ex = dem[:, 1 + k_end : 1 + k_end + k_ex] * sw[:, None] if k_ex > 0 else np.empty((n, 0))
    Z_excl = dem[:, 1 + k_end + k_ex :] * sw[:, None]

    X = np.hstack([X_end, X_ex])
    Z = np.hstack([Z_excl, X_ex])
    x_names = [*endog_cols, *exog_cols]

    # Projection P_Z X = Z (Z'Z)^-1 Z' X
    ZtZ_inv = np.linalg.pinv(Z.T @ Z)
    X_hat = Z @ (ZtZ_inv @ (Z.T @ X))
    XtX_hat_inv = np.linalg.pinv(X_hat.T @ X_hat)
    beta = XtX_hat_inv @ (X_hat.T @ y)
    resid = y - X @ beta  # already sqrt(w)-scaled

    # Clustered meat matrices (CRV1, with CGM 2-way inclusion-exclusion if 2 clusters)
    def crv1_vcov(cluster_series: pd.Series) -> np.ndarray:
        codes = pd.Categorical(cluster_series).codes
        G = int(codes.max()) + 1
        scores = X_hat * resid[:, None]
        k = scores.shape[1]
        meat = np.zeros((k, k))
        for g in range(G):
            sg = scores[codes == g].sum(axis=0)
            meat += np.outer(sg, sg)
        dof = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
        return dof * (XtX_hat_inv @ meat @ XtX_hat_inv)

    if len(cluster_cols) == 1:
        V = crv1_vcov(sub[cluster_cols[0]])
        df_t = max(sub[cluster_cols[0]].nunique() - 1, 1)
    else:
        c1, c2 = cluster_cols[0], cluster_cols[1]
        c12 = sub[c1].astype(str) + "_" + sub[c2].astype(str)
        V = crv1_vcov(sub[c1]) + crv1_vcov(sub[c2]) - crv1_vcov(c12)
        # Ensure PSD diagonal
        eigvals, eigvecs = np.linalg.eigh(V)
        V = eigvecs @ np.diag(np.maximum(eigvals, 1e-16)) @ eigvecs.T
        df_t = max(min(sub[c1].nunique(), sub[c2].nunique()) - 1, 1)

    se = np.sqrt(np.maximum(np.diag(V), 1e-16))
    tstat = beta / se
    pval = 2.0 * stats.t.sf(np.abs(tstat), df=df_t)

    # First-stage F-statistics (per endogenous variable + Cragg-Donald minimum eigenvalue F)
    fs_f = {}
    for j, ecol in enumerate(endog_cols):
        xj = X_end[:, j]
        # Partial out exog if present
        if k_ex > 0:
            P_ex = X_ex @ np.linalg.pinv(X_ex.T @ X_ex) @ X_ex.T
            xj_p = xj - P_ex @ xj
            Z_p = Z_excl - P_ex @ Z_excl
        else:
            xj_p = xj
            Z_p = Z_excl
        gamma = np.linalg.pinv(Z_p.T @ Z_p) @ (Z_p.T @ xj_p)
        u_fs = xj_p - Z_p @ gamma
        rss_u = float(u_fs @ u_fs)
        rss_r = float(xj_p @ xj_p)
        f_val = ((rss_r - rss_u) / max(k_ins, 1)) / max(rss_u / max(n - k_ins - k_ex, 1), 1e-12)
        fs_f[ecol] = f_val

    # Cragg-Donald minimum canonical correlation F-stat for joint weak-IV test
    if k_end >= 2:
        X_end_p = X_end - (X_ex @ np.linalg.pinv(X_ex.T @ X_ex) @ (X_ex.T @ X_end)) if k_ex > 0 else X_end
        Z_p = Z_excl - (X_ex @ np.linalg.pinv(X_ex.T @ X_ex) @ (X_ex.T @ Z_excl)) if k_ex > 0 else Z_excl
        X_end_hat = Z_p @ np.linalg.pinv(Z_p.T @ Z_p) @ (Z_p.T @ X_end_p)
        U_end = X_end_p - X_end_hat
        Sigma_vv = (U_end.T @ U_end) / max(n - k_ins - k_ex, 1)
        M = np.linalg.pinv(Sigma_vv) @ (X_end_hat.T @ X_end_hat) / max(k_ins, 1)
        cd_f = float(np.min(np.real(np.linalg.eigvals(M))))
    else:
        cd_f = next(iter(fs_f.values()))

    return {
        "n": n,
        "status": "ok",
        "coefs": dict(zip(x_names, beta.tolist())),
        "ses": dict(zip(x_names, se.tolist())),
        "tstats": dict(zip(x_names, tstat.tolist())),
        "pvals": dict(zip(x_names, pval.tolist())),
        "first_stage_f": fs_f,
        "cragg_donald_f": cd_f,
    }


def extract_pf_row(model: Any, var: str, label: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extract clean result dictionary from a pyfixest Feols/Fepois object."""
    c = float(model.coef().get(var, np.nan))
    s = float(model.se().get(var, np.nan))
    t = float(model.tstat().get(var, np.nan))
    p = float(model.pvalue().get(var, np.nan))
    out = {
        "spec": label,
        "variable": var,
        "coef": c,
        "se": s,
        "tstat": t,
        "pval": p,
        "n": int(model._N),
        "r2": float(getattr(model, "_r2", np.nan)),
    }
    f1 = getattr(model, "_f_stat_1st_stage", None)
    if f1 is not None and not np.isnan(f1):
        out["first_stage_f"] = float(f1)
    if extra:
        out.update(extra)
    return out


# ---------------------------------------------------------------------------
# Synthetic Microdata Generator (for --smoke-test verification)
# ---------------------------------------------------------------------------
def build_synthetic_microdata(seed: int = 2026) -> pd.DataFrame:
    """
    Generate realistic synthetic IPUMS-schema microdata (1980, 1990, 2000, 2010, 2019, 2024)
    across 30 real MET2013 CBSAs and 12 real ethnic groups so every single regression,
    crosswalk join, IV, PPML, Duncan-Trejo test, WFH interaction, and figure runs end-to-end.
    """
    rng = np.random.default_rng(seed)
    housing = pd.read_parquet(HOUSING_PARQUET)
    # Pick 30 large metros with full 1980-2024 FHFA + Zillow coverage
    top_cbsas = (
        housing[(housing["year"] == 2024) & housing["ln_hpi"].notna() & housing["ln_zori"].notna()]
        .sort_values("zillow_size_rank")
        .head(30)["met2013"]
        .astype(int)
        .tolist()
    )
    years = [1980, 1990, 2000, 2010, 2019, 2024]
    eths = [
        "Mexican", "Chinese", "Indian", "Filipino", "Vietnamese", "Korean",
        "Italian", "German", "Irish", "Polish", "Cuban", "Dominican",
    ]
    # Assign each metro a home state code (1..30)
    metro_state = {m: (i % 25) + 1 for i, m in enumerate(top_cbsas)}

    # Historical 1980 settlement shares pi_{rc,1980} + national origin waves
    pi_1980 = {
        (e, m): float(rng.dirichlet(np.ones(len(top_cbsas)))[j])
        for e in eths
        for j, m in enumerate(top_cbsas)
    }
    # Normalize so each eth sums to 1 across metros
    for e in eths:
        s = sum(pi_1980[(e, m)] for m in top_cbsas)
        for m in top_cbsas:
            pi_1980[(e, m)] /= s

    rows = []
    # Pre-generate cell-level immigrant shocks
    for y_idx, yr in enumerate(years):
        for e_idx, eth in enumerate(eths):
            nat_wave = 1.0 + 0.35 * y_idx * (1.2 if eth in LOW_ATTRITION_GROUPS else 0.6)
            for m in top_cbsas:
                # Immigrant count driven by shift-share + local shock
                imm_n = max(1, int(round(12 * pi_1980[(eth, m)] * nat_wave * rng.uniform(0.7, 1.4))))
                nat_n = max(4, int(round(16 * (0.5 / len(top_cbsas) + 0.5 * pi_1980[(eth, m)]) * rng.uniform(0.7, 1.3))))
                enclave_strength = imm_n / 40.0

                # Immigrants
                for _ in range(imm_n):
                    rows.append({
                        "YEAR": yr,
                        "met2013": m,
                        "STATEFIP": metro_state[m],
                        "BPL": 200 + e_idx,  # foreign-born
                        "is_immigrant": 1,
                        "eth_primary": eth,
                        "eth_secondary": None,
                        "PERWT": float(rng.uniform(10, 50)),
                        "HHWT": float(rng.uniform(10, 50)),
                        "AGE": int(rng.integers(18, 65)),
                        "SEX": int(rng.integers(1, 3)),
                        "MARST": int(rng.choice([1, 6])),
                        "SCHOOL": 1,
                        "EDUC": int(rng.integers(2, 11)),
                        "RACE": 1,
                        "HISPAN": 1 if eth in {"Mexican", "Cuban", "Dominican"} else 0,
                        "resident": int(rng.random() < 0.80),
                        "orig_met2013": m if rng.random() < 0.85 else int(rng.choice(top_cbsas)),
                        "ln_wage": float(rng.normal(2.9, 0.4)),
                        "OWNERSHP": int(rng.choice([1, 2])),
                        "RENTGRS": float(np.exp(rng.normal(7.1, 0.3))),
                        "VALUEH": float(np.exp(rng.normal(12.4, 0.4))),
                        "ROOMS": int(rng.integers(2, 8)),
                        "BEDROOMS": int(rng.integers(1, 5)),
                        "BUILTYR2": int(rng.integers(1, 9)),
                        "UNITSSTR": int(rng.integers(1, 10)),
                        "NUMPREC": int(rng.integers(1, 7)),
                        "MULTGEN": int(rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1])),
                        "METRO": int(rng.choice([2, 3])),
                        "TRANWORK": 80 if (yr >= 2021 and rng.random() < 0.18) else 10,
                        "teleworkable": int(rng.random() < 0.35),
                    })

                # Natives of this ethnicity
                for _ in range(nat_n):
                    born_in_state = rng.random() < (0.55 + 0.12 * enclave_strength)
                    bpl = metro_state[m] if born_in_state else int(rng.integers(1, 50))
                    is_mixed = rng.random() < (0.45 if eth in HIGH_ATTRITION_GROUPS else 0.15)
                    other_eth = str(rng.choice([x for x in eths if x != eth])) if is_mixed else None
                    # Duncan-Trejo identity salience: higher enclave_strength -> higher prob eth is primary
                    if is_mixed and rng.random() < (0.50 - 0.15 * enclave_strength):
                        eth1, eth2 = other_eth, eth
                    else:
                        eth1, eth2 = eth, other_eth

                    stayed = rng.random() < min(0.95, 0.72 + 0.08 * enclave_strength + (0.08 if born_in_state else 0.0))
                    orig_m = m if stayed else int(rng.choice([x for x in top_cbsas if x != m]))
                    wfh = 1 if (yr >= 2021 and rng.random() < 0.22) else (1 if (yr == 2019 and rng.random() < 0.06) else 0)
                    tele = 1 if (wfh or rng.random() < 0.40) else 0

                    rooms = max(1, int(round(rng.normal(5.2 - 0.6 * enclave_strength, 1.2))))
                    bedrooms = max(1, min(rooms - 1, int(rng.integers(1, 5))))
                    age = int(np.clip(rng.normal(40 - 4 * enclave_strength, 11), 18, 64))
                    educ = int(rng.integers(4, 12))
                    owner = 1 if rng.random() < max(0.15, 0.60 - 0.10 * enclave_strength) else 2
                    # Unadjusted rent/value falls with smaller rooms & younger age, but hedonic price rises slightly
                    ln_rent = 6.9 + 0.09 * rooms + 0.04 * bedrooms + 0.005 * age + 0.02 * enclave_strength + rng.normal(0, 0.25)
                    ln_val = 12.1 + 0.12 * rooms + 0.06 * bedrooms + 0.008 * age + 0.03 * enclave_strength + rng.normal(0, 0.30)
                    ln_w = 2.6 + 0.08 * educ + 0.01 * age - 0.02 * enclave_strength * (1 - wfh) + 0.10 * wfh + rng.normal(0, 0.35)

                    rows.append({
                        "YEAR": yr,
                        "met2013": m,
                        "STATEFIP": metro_state[m],
                        "BPL": bpl,
                        "is_immigrant": 0,
                        "eth_primary": eth1,
                        "eth_secondary": eth2,
                        "PERWT": float(rng.uniform(10, 50)),
                        "HHWT": float(rng.uniform(10, 50)),
                        "AGE": age,
                        "SEX": int(rng.integers(1, 3)),
                        "MARST": int(rng.choice([1, 6])),
                        "SCHOOL": 1,
                        "EDUC": educ,
                        "RACE": 1,
                        "HISPAN": 1 if eth1 in {"Mexican", "Cuban", "Dominican"} else 0,
                        "resident": int(stayed),
                        "orig_met2013": orig_m,
                        "ln_wage": float(ln_w),
                        "OWNERSHP": owner,
                        "RENTGRS": float(np.exp(ln_rent)),
                        "VALUEH": float(np.exp(ln_val)),
                        "ROOMS": rooms,
                        "BEDROOMS": bedrooms,
                        "BUILTYR2": int(rng.integers(1, 9)),
                        "UNITSSTR": int(rng.integers(1, 10)),
                        "NUMPREC": int(np.clip(round(rng.normal(2.6 + 0.8 * enclave_strength, 1.0)), 1, 9)),
                        "MULTGEN": int(rng.choice([1, 2, 3], p=[0.65 - 0.1 * min(enclave_strength, 1), 0.25 + 0.05 * min(enclave_strength, 1), 0.10 + 0.05 * min(enclave_strength, 1)])),
                        "METRO": int(rng.choice([2, 3], p=[0.45, 0.55])),
                        "TRANWORK": 80 if wfh else 10,
                        "teleworkable": tele,
                    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Real IPUMS Data Loader via DuckDB (streams 1980-2024 CSV.GZ efficiently)
# ---------------------------------------------------------------------------
def build_bpl_sql_case() -> str:
    """Build SQL CASE mapping IPUMS BPL -> eth_primary for immigrants."""
    cases = []
    for eth, (bpls, _, _) in ETHNICITY_MAP.items():
        # Avoid duplicate BPL codes colliding (specific BPLs handled cleanly)
        if len(bpls) == 1 and bpls[0] not in (210, 260, 300, 600):
            cases.append(f"WHEN BPL = {bpls[0]} THEN '{eth}'")
    # For shared regional BPLs (Central/South America, Caribbean, Africa), fall back to ANCESTR1
    return "CASE " + " ".join(cases) + " ELSE NULL END"


def build_ancestr_sql_case(col: str) -> str:
    """Build SQL CASE mapping IPUMS ANCESTR1/ANCESTR2 -> eth_name."""
    cases = []
    for eth, (_, ranges, _) in ETHNICITY_MAP.items():
        conds = [f"(TRY_CAST({col} AS INTEGER) BETWEEN {lo} AND {hi})" for lo, hi in ranges]
        cases.append(f"WHEN {' OR '.join(conds)} THEN '{eth}'")
    return "CASE " + " ".join(cases) + " ELSE NULL END"


def load_and_aggregate_real_ipums() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stream 1980-2024 IPUMS USA microdata (~40M rows) directly in DuckDB and return:
      - cell: balanced (eth, met2013, YEAR) panel on clean multi-year intervals [1980, 1990, 2000, 2010, 2019, 2024]
      - msa_imm: MSA-level immigration inflows panel [1980, 1990, 2000, 2010, 2019, 2024]
      - mixed_long: exact sufficient-statistic table for Duncan-Trejo mixed-ancestry primary-reporting test
      - wfh_agg: exact sufficient-statistic table for 2019-2024 WFH / teleworkable enclave amenity test
    """
    hist_files = sorted((IPUMS_RAW / "essay1_1970_2000").glob("*.csv*"))
    acs_files = sorted((IPUMS_RAW / "acs_2007_2024").glob("*.csv*"))
    if not hist_files or not acs_files:
        raise FileNotFoundError(
            f"Missing IPUMS raw CSV files in {IPUMS_RAW}/essay1_1970_2000 or acs_2007_2024. "
            "Run with --smoke-test to verify the pipeline without IPUMS files."
        )

    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute(
        f"CREATE TEMP TABLE xwalk AS SELECT DISTINCT "
        f"CAST(metarea_gen AS INTEGER) AS metarea_gen, "
        f"CAST(metarea_det AS INTEGER) AS metarea_det, "
        f"CAST(met2013 AS INTEGER) AS met2013 "
        f"FROM read_csv_auto('{MSA_CROSSWALK}') WHERE met2013 IS NOT NULL"
    )
    con.execute("CREATE TEMP TABLE xwalk_gen AS SELECT DISTINCT metarea_gen, met2013 FROM xwalk")
    con.execute("CREATE TEMP TABLE xwalk_det AS SELECT DISTINCT metarea_det, met2013 FROM xwalk")

    bpl_case = build_bpl_sql_case()
    anc1_case = build_ancestr_sql_case("ANCESTR1")
    anc2_case = build_ancestr_sql_case("ANCESTR2")

    hist_list = "[" + ", ".join(f"'{p}'" for p in hist_files) + "]"
    acs_list = "[" + ", ".join(f"'{p}'" for p in acs_files) + "]"

    # Clean multi-year panel waves for consistent multi-year differences (1980, 1990, 2000, 2010, 2019, 2024)
    # Plus 2021-2023 included in WFH query for maximum 2019-2024 power!
    print("   [DuckDB] Materializing harmonized 1980-2024 microdata view...")
    con.execute(f"""
        CREATE TEMP VIEW micro AS
        SELECT
            h.YEAR,
            COALESCE(NULLIF(TRY_CAST(h.MET2013 AS INTEGER), 0), x.met2013) AS met2013,
            h.STATEFIP,
            h.BPL,
            CASE WHEN h.BPL >= 100 THEN 1 ELSE 0 END AS is_immigrant,
            CASE
                WHEN h.BPL >= 100 THEN COALESCE({bpl_case}, {anc1_case})
                ELSE COALESCE({anc1_case}, {anc2_case})
            END AS eth_primary,
            CASE WHEN h.BPL < 100 THEN {anc2_case} ELSE NULL END AS eth_secondary,
            CAST(h.PERWT AS DOUBLE) AS PERWT,
            CAST(h.HHWT AS DOUBLE) AS HHWT,
            CAST(h.AGE AS DOUBLE) AS AGE,
            h.SEX,
            CASE WHEN h.EDUC BETWEEN 1 AND 11 THEN CAST(h.EDUC AS DOUBLE) ELSE NULL END AS EDUC,
            CASE
                WHEN h.MIGRATE5 = 1 THEN 1.0
                WHEN h.MIGRATE5 = 2 AND xm.met2013 = COALESCE(NULLIF(TRY_CAST(h.MET2013 AS INTEGER), 0), x.met2013) THEN 1.0
                WHEN h.MIGRATE5 IN (3, 4) THEN 0.0
                WHEN h.MIGRATE5 = 2 AND xm.met2013 IS NOT NULL AND xm.met2013 != COALESCE(NULLIF(TRY_CAST(h.MET2013 AS INTEGER), 0), x.met2013) THEN 0.0
                ELSE NULL
            END AS resident,
            CASE
                WHEN h.MIGRATE5 = 1 THEN COALESCE(NULLIF(TRY_CAST(h.MET2013 AS INTEGER), 0), x.met2013)
                WHEN h.MIGRATE5 IN (2, 3) THEN xm.met2013
                ELSE NULL
            END AS orig_met2013,
            CASE
                WHEN TRY_CAST(h.INCWAGE AS DOUBLE) > 0 AND TRY_CAST(h.INCWAGE AS DOUBLE) < 999998
                     AND TRY_CAST(h.UHRSWORK AS DOUBLE) > 0 AND TRY_CAST(h.WKSWORK1 AS DOUBLE) > 0
                THEN LN((TRY_CAST(h.INCWAGE AS DOUBLE) * TRY_CAST(h.CPI99 AS DOUBLE)) / (TRY_CAST(h.UHRSWORK AS DOUBLE) * TRY_CAST(h.WKSWORK1 AS DOUBLE)))
                ELSE NULL
            END AS ln_wage,
            TRY_CAST(h.OWNERSHP AS INTEGER) AS OWNERSHP,
            CASE WHEN TRY_CAST(h.OWNERSHP AS INTEGER) = 2 AND TRY_CAST(h.RENTGRS AS DOUBLE) > 0 THEN LN(TRY_CAST(h.RENTGRS AS DOUBLE) * TRY_CAST(h.CPI99 AS DOUBLE)) ELSE NULL END AS ln_rent,
            CASE WHEN TRY_CAST(h.OWNERSHP AS INTEGER) = 1 AND TRY_CAST(h.VALUEH AS DOUBLE) > 0 AND TRY_CAST(h.VALUEH AS DOUBLE) < 9999998 THEN LN(TRY_CAST(h.VALUEH AS DOUBLE) * TRY_CAST(h.CPI99 AS DOUBLE)) ELSE NULL END AS ln_val,
            NULLIF(TRY_CAST(h.ROOMS AS DOUBLE), 0) AS ROOMS,
            NULLIF(TRY_CAST(h.BEDROOMS AS DOUBLE), 0) AS BEDROOMS,
            NULLIF(TRY_CAST(h.BUILTYR2 AS DOUBLE), 0) AS BUILTYR2,
            NULLIF(TRY_CAST(h.UNITSSTR AS DOUBLE), 0) AS UNITSSTR,
            TRY_CAST(h.NUMPREC AS DOUBLE) AS NUMPREC,
            TRY_CAST(h.MULTGEN AS INTEGER) AS MULTGEN,
            TRY_CAST(h.METRO AS INTEGER) AS METRO,
            0 AS is_wfh,
            0 AS teleworkable
        FROM read_csv_auto({hist_list}) h
        LEFT JOIN xwalk_gen x ON TRY_CAST(h.METAREA AS INTEGER) = x.metarea_gen
        LEFT JOIN xwalk_det xm ON TRY_CAST(h.MIGMETAREA5 AS INTEGER) = xm.metarea_det
        WHERE h.YEAR IN (1980, 1990, 2000) AND h.AGE BETWEEN 18 AND 64 AND h.SCHOOL != 2
          AND COALESCE(NULLIF(TRY_CAST(h.MET2013 AS INTEGER), 0), x.met2013) IS NOT NULL

        UNION ALL

        SELECT
            a.YEAR,
            a.MET2013 AS met2013,
            a.STATEFIP,
            a.BPL,
            CASE WHEN a.BPL >= 100 THEN 1 ELSE 0 END AS is_immigrant,
            CASE
                WHEN a.BPL >= 100 THEN COALESCE({bpl_case}, {anc1_case})
                ELSE COALESCE({anc1_case}, {anc2_case})
            END AS eth_primary,
            CASE WHEN a.BPL < 100 THEN {anc2_case} ELSE NULL END AS eth_secondary,
            CAST(a.PERWT AS DOUBLE) AS PERWT,
            CAST(a.HHWT AS DOUBLE) AS HHWT,
            CAST(a.AGE AS DOUBLE) AS AGE,
            a.SEX,
            CASE WHEN a.EDUC BETWEEN 1 AND 11 THEN CAST(a.EDUC AS DOUBLE) ELSE NULL END AS EDUC,
            CASE
                WHEN a.MIGRATE1 = 1 THEN 1.0
                WHEN a.MIGRATE1 = 2 AND a.MIGMET131 = a.MET2013 THEN 1.0
                WHEN a.MIGRATE1 IN (3, 4) THEN 0.0
                WHEN a.MIGRATE1 = 2 AND a.MIGMET131 >= 10000 AND a.MIGMET131 != a.MET2013 THEN 0.0
                ELSE NULL
            END AS resident,
            CASE
                WHEN a.MIGRATE1 = 1 THEN a.MET2013
                WHEN a.MIGRATE1 IN (2, 3) AND a.MIGMET131 >= 10000 THEN a.MIGMET131
                ELSE NULL
            END AS orig_met2013,
            CASE
                WHEN a.INCWAGE > 0 AND a.INCWAGE < 999998 AND a.UHRSWORK > 0 AND a.WKSWORK2 > 0
                THEN LN((a.INCWAGE * a.CPI99) / (a.UHRSWORK *
                    CASE a.WKSWORK2 WHEN 1 THEN 7.0 WHEN 2 THEN 20.0 WHEN 3 THEN 33.0 WHEN 4 THEN 43.5 WHEN 5 THEN 48.5 WHEN 6 THEN 51.0 ELSE NULL END))
                ELSE NULL
            END AS ln_wage,
            a.OWNERSHP,
            CASE WHEN a.OWNERSHP = 2 AND a.RENTGRS > 0 THEN LN(a.RENTGRS * a.CPI99) ELSE NULL END AS ln_rent,
            CASE WHEN a.OWNERSHP = 1 AND a.VALUEH > 0 AND a.VALUEH < 9999998 THEN LN(a.VALUEH * a.CPI99) ELSE NULL END AS ln_val,
            CAST(NULLIF(a.ROOMS, 0) AS DOUBLE) AS ROOMS,
            CAST(NULLIF(a.BEDROOMS, 0) AS DOUBLE) AS BEDROOMS,
            CAST(NULLIF(a.BUILTYR2, 0) AS DOUBLE) AS BUILTYR2,
            CAST(NULLIF(a.UNITSSTR, 0) AS DOUBLE) AS UNITSSTR,
            CAST(a.NUMPREC AS DOUBLE) AS NUMPREC,
            a.MULTGEN,
            a.METRO,
            CASE WHEN a.TRANWORK = 80 THEN 1 ELSE 0 END AS is_wfh,
            CASE WHEN SUBSTR(CAST(a.OCCSOC AS VARCHAR), 1, 2) IN ('11','13','15','17','19','23','25','27','41','43') THEN 1 ELSE 0 END AS teleworkable
        FROM read_csv_auto({acs_list}) a
        WHERE a.YEAR IN (2010, 2019, 2021, 2022, 2023, 2024) AND a.AGE BETWEEN 18 AND 64 AND a.SCHOOL != 2 AND a.MET2013 > 0
    """)

    panel_years = "(1980, 1990, 2000, 2010, 2019, 2024)"
    print("   [DuckDB] Aggregating MSA totals, immigrant stocks, and native cell outcomes (1980-2024)...")
    msa_pop = con.execute(f"""
        SELECT met2013, YEAR, SUM(PERWT) AS msa_pop, COUNT(*) AS raw_n
        FROM micro
        WHERE YEAR IN {panel_years}
        GROUP BY met2013, YEAR
        ORDER BY met2013, YEAR
    """).df()
    total_raw_n = int(msa_pop["raw_n"].sum())
    print(f"   [DuckDB] Total matched metro working-age rows (6 panel waves): {total_raw_n:,}")

    imm_cell = con.execute(f"""
        SELECT eth_primary AS eth, met2013, YEAR, SUM(PERWT) AS imm_pop
        FROM micro
        WHERE YEAR IN {panel_years} AND is_immigrant = 1 AND eth_primary IS NOT NULL
        GROUP BY eth_primary, met2013, YEAR
    """).df()

    nat_agg = con.execute(f"""
        SELECT
            eth_primary AS eth,
            met2013,
            YEAR,
            SUM(PERWT) AS nat_pop,
            SUM(CASE WHEN resident = 1.0 THEN PERWT ELSE 0.0 END) AS res_pop,
            SUM(CASE WHEN resident = 0.0 THEN PERWT ELSE 0.0 END) AS arr_pop,
            SUM(CASE WHEN resident = 1.0 AND BPL != STATEFIP THEN PERWT ELSE 0.0 END) AS oos_res_pop,
            SUM(CASE WHEN resident = 0.0 AND BPL != STATEFIP THEN PERWT ELSE 0.0 END) AS oos_arr_pop,
            SUM(CASE WHEN ln_rent IS NOT NULL THEN ln_rent * HHWT END) / NULLIF(SUM(CASE WHEN ln_rent IS NOT NULL THEN HHWT END), 0) AS mean_ln_rent,
            SUM(CASE WHEN ln_val IS NOT NULL THEN ln_val * HHWT END) / NULLIF(SUM(CASE WHEN ln_val IS NOT NULL THEN HHWT END), 0) AS mean_ln_val,
            SUM(CASE WHEN ROOMS IS NOT NULL THEN ROOMS * HHWT END) / NULLIF(SUM(CASE WHEN ROOMS IS NOT NULL THEN HHWT END), 0) AS mean_rooms,
            SUM(CASE WHEN BEDROOMS IS NOT NULL THEN BEDROOMS * HHWT END) / NULLIF(SUM(CASE WHEN BEDROOMS IS NOT NULL THEN HHWT END), 0) AS mean_bedrooms,
            SUM(CASE WHEN BUILTYR2 IS NOT NULL THEN BUILTYR2 * HHWT END) / NULLIF(SUM(CASE WHEN BUILTYR2 IS NOT NULL THEN HHWT END), 0) AS mean_builtyr,
            SUM(CASE WHEN UNITSSTR IS NOT NULL THEN UNITSSTR * HHWT END) / NULLIF(SUM(CASE WHEN UNITSSTR IS NOT NULL THEN HHWT END), 0) AS mean_unitsstr,
            SUM(AGE * PERWT) / NULLIF(SUM(PERWT), 0) AS mean_age,
            SUM(CASE WHEN EDUC IS NOT NULL THEN EDUC * PERWT END) / NULLIF(SUM(CASE WHEN EDUC IS NOT NULL THEN PERWT END), 0) AS mean_educ,
            SUM(CASE WHEN ln_wage IS NOT NULL THEN ln_wage * PERWT END) / NULLIF(SUM(CASE WHEN ln_wage IS NOT NULL THEN PERWT END), 0) AS mean_ln_wage,
            SUM(CASE WHEN ROOMS > 0 THEN (NUMPREC / ROOMS) * HHWT END) / NULLIF(SUM(CASE WHEN ROOMS > 0 THEN HHWT END), 0) AS mean_crowding,
            SUM(CASE WHEN MULTGEN > 0 THEN (CASE WHEN MULTGEN >= 2 THEN 1.0 ELSE 0.0 END) * HHWT END) / NULLIF(SUM(CASE WHEN MULTGEN > 0 THEN HHWT END), 0) AS share_multgen,
            SUM(CASE WHEN OWNERSHP IN (1, 2) THEN (CASE WHEN OWNERSHP = 1 THEN 1.0 ELSE 0.0 END) * HHWT END) / NULLIF(SUM(CASE WHEN OWNERSHP IN (1, 2) THEN HHWT END), 0) AS share_owner,
            SUM(CASE WHEN METRO IN (2, 3) THEN (CASE WHEN METRO = 3 THEN 1.0 ELSE 0.0 END) * HHWT END) / NULLIF(SUM(CASE WHEN METRO IN (2, 3) THEN HHWT END), 0) AS share_suburb
        FROM micro
        WHERE YEAR IN {panel_years} AND is_immigrant = 0 AND eth_primary IS NOT NULL
        GROUP BY eth_primary, met2013, YEAR
    """).df()

    print("   [DuckDB] Computing true origin-MSA retention rates (MIGMETAREA5 & MIGMET131)...")
    orig_agg = con.execute(f"""
        SELECT
            eth_primary AS eth,
            orig_met2013 AS met2013,
            YEAR,
            SUM(PERWT) AS orig_cohort_pop,
            SUM(CASE WHEN orig_met2013 = met2013 THEN PERWT ELSE 0.0 END) / NULLIF(SUM(PERWT), 0) AS true_retention_rate
        FROM micro
        WHERE YEAR IN {panel_years} AND is_immigrant = 0 AND eth_primary IS NOT NULL AND orig_met2013 IS NOT NULL
        GROUP BY eth_primary, orig_met2013, YEAR
    """).df()

    print("   [DuckDB] Extracting Duncan-Trejo mixed-ancestry primary-reporting sufficient stats...")
    mixed_long = con.execute(f"""
        WITH m AS (
            SELECT eth_primary AS eth, met2013, YEAR, 1.0 AS is_primary, PERWT, AGE, EDUC
            FROM micro
            WHERE YEAR IN {panel_years} AND is_immigrant = 0
              AND eth_primary IS NOT NULL AND eth_secondary IS NOT NULL AND eth_primary != eth_secondary
            UNION ALL
            SELECT eth_secondary AS eth, met2013, YEAR, 0.0 AS is_primary, PERWT, AGE, EDUC
            FROM micro
            WHERE YEAR IN {panel_years} AND is_immigrant = 0
              AND eth_primary IS NOT NULL AND eth_secondary IS NOT NULL AND eth_primary != eth_secondary
        )
        SELECT
            eth, met2013, YEAR,
            SUM(is_primary * PERWT) / NULLIF(SUM(PERWT), 0) AS is_primary,
            SUM(AGE * PERWT) / NULLIF(SUM(PERWT), 0) AS AGE,
            SUM(CASE WHEN EDUC IS NOT NULL THEN EDUC * PERWT END) / NULLIF(SUM(CASE WHEN EDUC IS NOT NULL THEN PERWT END), 0) AS EDUC,
            SUM(PERWT) AS PERWT
        FROM m
        GROUP BY eth, met2013, YEAR
    """).df()

    print("   [DuckDB] Extracting 2019-2024 WFH / teleworkable enclave amenity sufficient stats...")
    # Compute 2019-2024 annual share_rct for WFH years (2019, 2021, 2022, 2023, 2024)
    wfh_agg = con.execute("""
        WITH msa_p AS (
            SELECT met2013, YEAR, SUM(PERWT) AS mp
            FROM micro WHERE YEAR >= 2019 GROUP BY met2013, YEAR
        ),
        imm_p AS (
            SELECT eth_primary AS eth, met2013, YEAR, SUM(PERWT) AS ip
            FROM micro WHERE YEAR >= 2019 AND is_immigrant = 1 AND eth_primary IS NOT NULL
            GROUP BY eth_primary, met2013, YEAR
        ),
        shr AS (
            SELECT i.eth, i.met2013, i.YEAR, 100.0 * i.ip / m.mp AS share_rct
            FROM imm_p i JOIN msa_p m ON i.met2013 = m.met2013 AND i.YEAR = m.YEAR
        )
        SELECT
            n.eth_primary AS eth,
            n.met2013,
            n.YEAR,
            n.is_wfh,
            n.teleworkable,
            n.SEX,
            COALESCE(s.share_rct, 0.0) AS share_rct,
            SUM(CASE WHEN n.resident IS NOT NULL THEN n.resident * n.PERWT END) / NULLIF(SUM(CASE WHEN n.resident IS NOT NULL THEN n.PERWT END), 0) AS resident,
            SUM(CASE WHEN n.ln_wage IS NOT NULL THEN n.ln_wage * n.PERWT END) / NULLIF(SUM(CASE WHEN n.ln_wage IS NOT NULL THEN n.PERWT END), 0) AS ln_wage,
            SUM(n.AGE * n.PERWT) / NULLIF(SUM(n.PERWT), 0) AS AGE,
            SUM(CASE WHEN n.EDUC IS NOT NULL THEN n.EDUC * n.PERWT END) / NULLIF(SUM(CASE WHEN n.EDUC IS NOT NULL THEN n.PERWT END), 0) AS EDUC,
            SUM(n.PERWT) AS PERWT
        FROM micro n
        LEFT JOIN shr s ON n.eth_primary = s.eth AND n.met2013 = s.met2013 AND n.YEAR = s.YEAR
        WHERE n.YEAR >= 2019 AND n.is_immigrant = 0 AND n.eth_primary IS NOT NULL
        GROUP BY n.eth_primary, n.met2013, n.YEAR, n.is_wfh, n.teleworkable, n.SEX, COALESCE(s.share_rct, 0.0)
    """).df()

    # Assemble balanced cell panel + Card-Peri flows + Leave-Out Bartik IV
    msa_pop = msa_pop.sort_values(["met2013", "YEAR"]).reset_index(drop=True)
    msa_pop["msa_pop_lag"] = msa_pop.groupby("met2013")["msa_pop"].shift(1)

    years = sorted(msa_pop["YEAR"].unique())
    eths = sorted(set(imm_cell["eth"].unique()) | set(nat_agg["eth"].unique()))
    metros = sorted(msa_pop["met2013"].unique())
    grid = pd.MultiIndex.from_product([eths, metros, years], names=["eth", "met2013", "YEAR"]).to_frame(index=False)

    cell = (
        grid.merge(msa_pop[["met2013", "YEAR", "msa_pop", "msa_pop_lag"]], on=["met2013", "YEAR"], how="inner")
        .merge(imm_cell, on=["eth", "met2013", "YEAR"], how="left")
        .merge(nat_agg, on=["eth", "met2013", "YEAR"], how="left")
        .merge(orig_agg, on=["eth", "met2013", "YEAR"], how="left")
    )
    cell["imm_pop"] = cell["imm_pop"].fillna(0.0)
    for c in ["nat_pop", "res_pop", "arr_pop", "oos_res_pop", "oos_arr_pop"]:
        cell[c] = cell[c].fillna(0.0)

    cell["share_rct"] = 100.0 * cell["imm_pop"] / cell["msa_pop"]
    cell = cell.sort_values(["eth", "met2013", "YEAR"]).reset_index(drop=True)
    cell["imm_pop_lag"] = cell.groupby(["eth", "met2013"])["imm_pop"].shift(1)
    cell["share_rct_lag"] = cell.groupby(["eth", "met2013"])["share_rct"].shift(1)
    cell["d_share_rct"] = cell["share_rct"] - cell["share_rct_lag"]
    cell["cp_flow"] = 100.0 * (cell["imm_pop"] - cell["imm_pop_lag"]) / cell["msa_pop_lag"]
    cell["cp_flow_lag"] = cell.groupby(["eth", "met2013"])["cp_flow"].shift(1)

    base_yr = min(years)
    base = cell[cell["YEAR"] == base_yr][["eth", "met2013", "imm_pop"]].copy()
    eth_tot_base = base.groupby("eth")["imm_pop"].transform("sum").replace(0, np.nan)
    base["pi_base"] = (base["imm_pop"] / eth_tot_base).fillna(0.0)
    cell = cell.merge(base[["eth", "met2013", "pi_base"]], on=["eth", "met2013"], how="left")

    nat_imm = cell.groupby(["eth", "YEAR"], as_index=False).agg(nat_imm_pop=("imm_pop", "sum")).sort_values(["eth", "YEAR"])
    nat_imm["nat_imm_pop_lag"] = nat_imm.groupby("eth")["nat_imm_pop"].shift(1)
    cell = cell.merge(nat_imm, on=["eth", "YEAR"], how="left")

    lo_inflow = (cell["nat_imm_pop"] - cell["imm_pop"]) - (cell["nat_imm_pop_lag"] - cell["imm_pop_lag"])
    cell["iv_bartik_flow"] = 100.0 * (cell["pi_base"] * lo_inflow) / cell["msa_pop_lag"]
    cell["iv_bartik_flow_lag"] = cell.groupby(["eth", "met2013"])["iv_bartik_flow"].shift(1)
    cell["iv_bartik_stock"] = 100.0 * (cell["pi_base"] * (cell["nat_imm_pop"] - cell["imm_pop"])) / cell["msa_pop"]

    tot_res = cell.groupby(["met2013", "YEAR"])["res_pop"].transform("sum").replace(0, np.nan)
    tot_arr = cell.groupby(["met2013", "YEAR"])["arr_pop"].transform("sum").replace(0, np.nan)
    cell["r_share"] = 100.0 * cell["res_pop"] / tot_res
    cell["a_share"] = 100.0 * cell["arr_pop"] / tot_arr
    cell["ln_res_pop"] = np.where(cell["res_pop"] > 0, np.log(np.where(cell["res_pop"] > 0, cell["res_pop"], 1.0)), np.nan)
    cell["ln_arr_pop"] = np.where(cell["arr_pop"] > 0, np.log(np.where(cell["arr_pop"] > 0, cell["arr_pop"], 1.0)), np.nan)
    cell["ln_oos_res_pop"] = np.where(cell["oos_res_pop"] > 0, np.log(np.where(cell["oos_res_pop"] > 0, cell["oos_res_pop"], 1.0)), np.nan)
    cell["ln_oos_arr_pop"] = np.where(cell["oos_arr_pop"] > 0, np.log(np.where(cell["oos_arr_pop"] > 0, cell["oos_arr_pop"], 1.0)), np.nan)

    for col in ["mean_ln_rent", "mean_ln_val", "mean_rooms", "mean_bedrooms", "mean_builtyr", "mean_unitsstr", "mean_age", "share_owner"]:
        cell[f"{col}_lag"] = cell.groupby(["eth", "met2013"])[col].shift(1)
        cell[f"d_{col}"] = cell[col] - cell[f"{col}_lag"]

    cell["eth_year"] = cell["eth"].astype(str) + "_" + cell["YEAR"].astype(str)
    cell["met_year"] = cell["met2013"].astype(str) + "_" + cell["YEAR"].astype(str)
    cell["eth_met"] = cell["eth"].astype(str) + "_" + cell["met2013"].astype(str)
    cell["attrition_tier"] = cell["eth"].map(lambda x: "high" if x in HIGH_ATTRITION_GROUPS else "low")

    msa_imm = (
        cell.groupby(["met2013", "YEAR"], as_index=False)
        .agg(
            tot_imm_pop=("imm_pop", "sum"),
            msa_pop=("msa_pop", "first"),
            msa_pop_lag=("msa_pop_lag", "first"),
            msa_bartik_flow=("iv_bartik_flow", "sum"),
        )
        .sort_values(["met2013", "YEAR"])
    )
    msa_imm["tot_imm_lag"] = msa_imm.groupby("met2013")["tot_imm_pop"].shift(1)
    msa_imm["msa_imm_flow"] = 100.0 * (msa_imm["tot_imm_pop"] - msa_imm["tot_imm_lag"]) / msa_imm["msa_pop_lag"]

    return cell, msa_imm, mixed_long, wfh_agg, total_raw_n


# ---------------------------------------------------------------------------
# Cell-Level Panel Construction (including Card-Peri flows & Leave-Out Bartik IV)
# ---------------------------------------------------------------------------
def build_cell_panel(micro: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build balanced (eth, met2013, year) cell panel with:
      - Song's Share_{rct} = 100 * M_{rct} / P_{ct}
      - Card-Peri (2016) flow = 100 * Delta M_{rct} / P_{c,t-1}
      - Leave-out Bartik shift-share IV (1980 base shares) + lagged IV (JRS 2018)
      - Native population counts (all, pre-existing residents, arrivers, out-of-state-born)
      - True origin-MSA retention rate (from orig_met2013)
      - Housing expenditure, hedonic controls, crowding, multigenerational, suburban share
    """
    # Total MSA population P_{ct} (all working-age adults in MSA c at year t)
    msa_pop = (
        micro.groupby(["met2013", "YEAR"], as_index=False)
        .agg(msa_pop=("PERWT", "sum"))
        .sort_values(["met2013", "YEAR"])
    )
    msa_pop["msa_pop_lag"] = msa_pop.groupby("met2013")["msa_pop"].shift(1)

    # Immigrant counts M_{rct}
    imm = micro[(micro["is_immigrant"] == 1) & micro["eth_primary"].notna()]
    imm_cell = (
        imm.groupby(["eth_primary", "met2013", "YEAR"], as_index=False)
        .agg(imm_pop=("PERWT", "sum"))
        .rename(columns={"eth_primary": "eth"})
    )

    # Full balanced grid of observed (eth, met2013) x years for PPML zero-cell inclusion
    years = sorted(micro["YEAR"].unique())
    eths = sorted(micro["eth_primary"].dropna().unique())
    metros = sorted(micro["met2013"].unique())
    grid = pd.MultiIndex.from_product([eths, metros, years], names=["eth", "met2013", "YEAR"]).to_frame(index=False)

    cell = grid.merge(msa_pop, on=["met2013", "YEAR"], how="inner").merge(
        imm_cell, on=["eth", "met2013", "YEAR"], how="left"
    )
    cell["imm_pop"] = cell["imm_pop"].fillna(0.0)
    cell["share_rct"] = 100.0 * cell["imm_pop"] / cell["msa_pop"]

    cell = cell.sort_values(["eth", "met2013", "YEAR"]).reset_index(drop=True)
    cell["imm_pop_lag"] = cell.groupby(["eth", "met2013"])["imm_pop"].shift(1)
    cell["share_rct_lag"] = cell.groupby(["eth", "met2013"])["share_rct"].shift(1)
    cell["d_share_rct"] = cell["share_rct"] - cell["share_rct_lag"]
    cell["cp_flow"] = 100.0 * (cell["imm_pop"] - cell["imm_pop_lag"]) / cell["msa_pop_lag"]
    cell["cp_flow_lag"] = cell.groupby(["eth", "met2013"])["cp_flow"].shift(1)

    # Leave-out Bartik shift-share IV using base year (earliest year, e.g. 1980) shares
    base_yr = min(years)
    base = cell[cell["YEAR"] == base_yr][["eth", "met2013", "imm_pop"]].copy()
    eth_tot_base = base.groupby("eth")["imm_pop"].transform("sum").replace(0, np.nan)
    base["pi_base"] = (base["imm_pop"] / eth_tot_base).fillna(0.0)
    cell = cell.merge(base[["eth", "met2013", "pi_base"]], on=["eth", "met2013"], how="left")

    nat_imm = cell.groupby(["eth", "YEAR"], as_index=False).agg(nat_imm_pop=("imm_pop", "sum"))
    nat_imm = nat_imm.sort_values(["eth", "YEAR"])
    nat_imm["nat_imm_pop_lag"] = nat_imm.groupby("eth")["nat_imm_pop"].shift(1)
    cell = cell.merge(nat_imm, on=["eth", "YEAR"], how="left")

    # Leave-out national inflow: (M_{rt} - M_{rct}) - (M_{r,t-1} - M_{rc,t-1})
    lo_inflow = (cell["nat_imm_pop"] - cell["imm_pop"]) - (cell["nat_imm_pop_lag"] - cell["imm_pop_lag"])
    cell["iv_bartik_flow"] = 100.0 * (cell["pi_base"] * lo_inflow) / cell["msa_pop_lag"]
    cell["iv_bartik_flow_lag"] = cell.groupby(["eth", "met2013"])["iv_bartik_flow"].shift(1)
    cell["iv_bartik_stock"] = 100.0 * (cell["pi_base"] * (cell["nat_imm_pop"] - cell["imm_pop"])) / cell["msa_pop"]

    # Native-born aggregations
    nat = micro[(micro["is_immigrant"] == 0) & micro["eth_primary"].notna()].copy()
    nat["out_of_state_born"] = (nat["BPL"] != nat["STATEFIP"]).astype(int)
    nat["crowding"] = nat["NUMPREC"] / nat["ROOMS"].replace(0, np.nan)
    nat["is_multgen"] = np.where(nat["MULTGEN"] > 0, (nat["MULTGEN"] >= 2).astype(float), np.nan)
    nat["is_owner"] = np.where(nat["OWNERSHP"].isin([1, 2]), (nat["OWNERSHP"] == 1).astype(float), np.nan)
    nat["is_suburb"] = np.where(nat["METRO"].isin([2, 3]), (nat["METRO"] == 3).astype(float), np.nan)
    nat["ln_rent"] = np.where((nat["OWNERSHP"] == 2) & (nat["RENTGRS"] > 0), np.log(nat["RENTGRS"]), np.nan)
    nat["ln_val"] = np.where((nat["OWNERSHP"] == 1) & (nat["VALUEH"] > 0), np.log(nat["VALUEH"]), np.nan)

    def wavg(g: pd.DataFrame, col: str, wcol: str = "PERWT") -> float:
        m = g[col].notna()
        if not m.any():
            return np.nan
        return float(np.average(g.loc[m, col], weights=g.loc[m, wcol]))

    nat_agg = (
        nat.groupby(["eth_primary", "met2013", "YEAR"], as_index=False)
        .apply(
            lambda g: pd.Series({
                "nat_pop": g["PERWT"].sum(),
                "res_pop": g.loc[g["resident"] == 1, "PERWT"].sum(),
                "arr_pop": g.loc[g["resident"] == 0, "PERWT"].sum(),
                "oos_res_pop": g.loc[(g["resident"] == 1) & (g["out_of_state_born"] == 1), "PERWT"].sum(),
                "oos_arr_pop": g.loc[(g["resident"] == 0) & (g["out_of_state_born"] == 1), "PERWT"].sum(),
                "mean_ln_rent": wavg(g, "ln_rent", "HHWT"),
                "mean_ln_val": wavg(g, "ln_val", "HHWT"),
                "mean_rooms": wavg(g, "ROOMS", "HHWT"),
                "mean_bedrooms": wavg(g, "BEDROOMS", "HHWT"),
                "mean_builtyr": wavg(g, "BUILTYR2", "HHWT"),
                "mean_unitsstr": wavg(g, "UNITSSTR", "HHWT"),
                "mean_age": wavg(g, "AGE"),
                "mean_educ": wavg(g, "EDUC"),
                "mean_ln_wage": wavg(g, "ln_wage"),
                "mean_crowding": wavg(g, "crowding", "HHWT"),
                "share_multgen": wavg(g, "is_multgen", "HHWT"),
                "share_owner": wavg(g, "is_owner", "HHWT"),
                "share_suburb": wavg(g, "is_suburb", "HHWT"),
            }),
            include_groups=False,
        )
        .rename(columns={"eth_primary": "eth"})
    )

    cell = cell.merge(nat_agg, on=["eth", "met2013", "YEAR"], how="left")
    for c in ["nat_pop", "res_pop", "arr_pop", "oos_res_pop", "oos_arr_pop"]:
        cell[c] = cell[c].fillna(0.0)

    # Cell shares r_share and a_share (Song Eq. 3)
    tot_res = cell.groupby(["met2013", "YEAR"])["res_pop"].transform("sum").replace(0, np.nan)
    tot_arr = cell.groupby(["met2013", "YEAR"])["arr_pop"].transform("sum").replace(0, np.nan)
    cell["r_share"] = 100.0 * cell["res_pop"] / tot_res
    cell["a_share"] = 100.0 * cell["arr_pop"] / tot_arr
    cell["ln_res_pop"] = np.where(cell["res_pop"] > 0, np.log(cell["res_pop"]), np.nan)
    cell["ln_arr_pop"] = np.where(cell["arr_pop"] > 0, np.log(cell["arr_pop"]), np.nan)
    cell["ln_oos_res_pop"] = np.where(cell["oos_res_pop"] > 0, np.log(cell["oos_res_pop"]), np.nan)
    cell["ln_oos_arr_pop"] = np.where(cell["oos_arr_pop"] > 0, np.log(cell["oos_arr_pop"]), np.nan)

    # True origin-MSA retention rate: of co-ethnic natives living in MSA c at t-5/t-1, share who stayed
    orig = nat[nat["orig_met2013"].notna()].copy()
    orig["stayed_in_orig"] = (orig["orig_met2013"] == orig["met2013"]).astype(float)
    orig_agg = (
        orig.groupby(["eth_primary", "orig_met2013", "YEAR"], as_index=False)
        .apply(
            lambda g: pd.Series({
                "orig_cohort_pop": g["PERWT"].sum(),
                "true_retention_rate": float(np.average(g["stayed_in_orig"], weights=g["PERWT"])),
            }),
            include_groups=False,
        )
        .rename(columns={"eth_primary": "eth", "orig_met2013": "met2013"})
    )
    cell = cell.merge(orig_agg, on=["eth", "met2013", "YEAR"], how="left")

    # First-differenced housing variables for Table 7 stress-test
    cell = cell.sort_values(["eth", "met2013", "YEAR"]).reset_index(drop=True)
    for col in ["mean_ln_rent", "mean_ln_val", "mean_rooms", "mean_bedrooms", "mean_builtyr", "mean_unitsstr", "mean_age", "share_owner"]:
        cell[f"{col}_lag"] = cell.groupby(["eth", "met2013"])[col].shift(1)
        cell[f"d_{col}"] = cell[col] - cell[f"{col}_lag"]

    # Fixed-effect keys
    cell["eth_year"] = cell["eth"].astype(str) + "_" + cell["YEAR"].astype(str)
    cell["met_year"] = cell["met2013"].astype(str) + "_" + cell["YEAR"].astype(str)
    cell["eth_met"] = cell["eth"].astype(str) + "_" + cell["met2013"].astype(str)
    cell["attrition_tier"] = cell["eth"].map(lambda x: "high" if x in HIGH_ATTRITION_GROUPS else "low")

    # MSA-level panel for Pillar 1d (Saiz 2007 MSA housing price test)
    msa_imm = (
        cell.groupby(["met2013", "YEAR"], as_index=False)
        .agg(
            tot_imm_pop=("imm_pop", "sum"),
            msa_pop=("msa_pop", "first"),
            msa_pop_lag=("msa_pop_lag", "first"),
            msa_bartik_flow=("iv_bartik_flow", "sum"),
        )
        .sort_values(["met2013", "YEAR"])
    )
    msa_imm["tot_imm_lag"] = msa_imm.groupby("met2013")["tot_imm_pop"].shift(1)
    msa_imm["msa_imm_flow"] = 100.0 * (msa_imm["tot_imm_pop"] - msa_imm["tot_imm_lag"]) / msa_imm["msa_pop_lag"]

    return cell, msa_imm


# ---------------------------------------------------------------------------
# PILLAR 1: Resolving the Housing Price Puzzle (Table 7 + Saiz MSA Test)
# ---------------------------------------------------------------------------
def run_pillar1_housing(cell: pd.DataFrame, msa_imm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    fe = "eth_year + met_year + eth_met"

    for outcome, lag_col, label_base in [
        ("d_mean_ln_rent", "mean_ln_rent_lag", "Rent"),
        ("d_mean_ln_val", "mean_ln_val_lag", "House Value"),
    ]:
        # 1a. Song Table 7 replication (WITH Nickell-biased lagged level, no hedonics)
        sub1 = cell.dropna(subset=[outcome, "d_share_rct", lag_col, "mean_educ", "mean_ln_wage", "nat_pop"])
        if len(sub1) > 50:
            m1 = pf.feols(
                f"{outcome} ~ d_share_rct + {lag_col} + mean_educ + mean_ln_wage | {fe}",
                data=sub1, weights="nat_pop", vcov={"CRV1": "met2013"},
            )
            rows.append(extract_pf_row(m1, "d_share_rct", f"Table7_SongSpec_WithLag_{label_base}"))

        # 1b. Remove Nickell-biased lagged level
        sub2 = cell.dropna(subset=[outcome, "d_share_rct", "mean_educ", "mean_ln_wage", "nat_pop"])
        if len(sub2) > 50:
            m2 = pf.feols(
                f"{outcome} ~ d_share_rct + mean_educ + mean_ln_wage | {fe}",
                data=sub2, weights="nat_pop", vcov={"CRV1": "met2013"},
            )
            rows.append(extract_pf_row(m2, "d_share_rct", f"Table7_NoNickellLag_{label_base}"))

        # 1c. Add full IPUMS hedonic housing quality controls + cell age (refutes fn. 45!)
        hed_cols = ["d_mean_rooms", "d_mean_bedrooms", "d_mean_builtyr", "d_mean_unitsstr", "d_mean_age", "d_share_owner"]
        sub3 = cell.dropna(subset=[outcome, "d_share_rct", "mean_educ", "mean_ln_wage", "nat_pop", *hed_cols])
        if len(sub3) > 50:
            m3 = pf.feols(
                f"{outcome} ~ d_share_rct + mean_educ + mean_ln_wage + {' + '.join(hed_cols)} | {fe}",
                data=sub3, weights="nat_pop", vcov={"CRV1": "met2013"},
            )
            rows.append(extract_pf_row(m3, "d_share_rct", f"Table7_HedonicQualityControlled_{label_base}"))

    # 1d. Crowding, multigenerational living, homeownership, suburban sorting mechanisms
    for mech_col, mech_label in [
        ("mean_crowding", "Crowding_PersonsPerRoom"),
        ("share_multgen", "Multigenerational_Share"),
        ("share_owner", "Homeownership_Share"),
        ("share_suburb", "Suburban_METRO3_Share"),
    ]:
        sub_m = cell.dropna(subset=[mech_col, "share_rct", "mean_age", "mean_educ", "nat_pop"])
        if len(sub_m) > 50:
            mm = pf.feols(
                f"{mech_col} ~ share_rct + mean_age + mean_educ | {fe}",
                data=sub_m, weights="nat_pop", vcov={"CRV1": "met2013"},
            )
            rows.append(extract_pf_row(mm, "share_rct", f"Mechanism_{mech_label}"))

    # 1e. MSA-level Housing Price Test (FHFA HPI + Zillow ZHVI + Zillow ZORI — Saiz 2007)
    housing = pd.read_parquet(HOUSING_PARQUET)
    msa_h = msa_imm.merge(
        housing[["met2013", "year", "ln_hpi", "ln_zhvi", "ln_zori"]].rename(columns={"year": "YEAR"}),
        on=["met2013", "YEAR"],
        how="inner",
    ).sort_values(["met2013", "YEAR"])
    for hcol in ["ln_hpi", "ln_zhvi", "ln_zori"]:
        msa_h[f"d_{hcol}"] = msa_h.groupby("met2013")[hcol].diff()

    msa_rows = []
    for hcol, hlabel in [("d_ln_hpi", "FHFA_HPI"), ("d_ln_zhvi", "Zillow_ZHVI"), ("d_ln_zori", "Zillow_ZORI")]:
        sub_h = msa_h.dropna(subset=[hcol, "msa_imm_flow", "msa_bartik_flow", "msa_pop_lag"])
        if len(sub_h) >= 20:
            n_yrs = sub_h["YEAR"].nunique()
            fe_str = " | YEAR + met2013" if n_yrs >= 2 else ""
            vcov_spec = {"CRV1": "met2013"} if n_yrs >= 2 else "hetero"
            m_ols = pf.feols(
                f"{hcol} ~ msa_imm_flow{fe_str}",
                data=sub_h, weights="msa_pop_lag", vcov=vcov_spec,
            )
            msa_rows.append(extract_pf_row(m_ols, "msa_imm_flow", f"MSA_{hlabel}_OLS"))
            try:
                m_iv = pf.feols(
                    f"{hcol} ~ 1 + [msa_imm_flow ~ msa_bartik_flow]{fe_str}",
                    data=sub_h, weights="msa_pop_lag", vcov=vcov_spec,
                )
                msa_rows.append(extract_pf_row(m_iv, "msa_imm_flow", f"MSA_{hlabel}_BartikIV"))
            except Exception:
                pass

    return pd.DataFrame(rows), pd.DataFrame(msa_rows)


# ---------------------------------------------------------------------------
# PILLAR 2: True Physical Migration vs. Fertility-in-Place & Duncan-Trejo Attrition
# ---------------------------------------------------------------------------
def run_pillar2_migration_and_attrition(
    micro: pd.DataFrame | None,
    cell: pd.DataFrame,
    mixed_long: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fe = "eth_year + met_year + eth_met"
    mig_rows = []

    # 2a. Song Table 3 Baseline vs. Out-of-State-Born (BPL != STATEFIP)
    for col, label in [
        ("ln_res_pop", "Song_Baseline_ln_Residents"),
        ("ln_oos_res_pop", "OutOfStateBorn_ln_Residents"),
        ("ln_arr_pop", "Song_Baseline_ln_Arrivers"),
        ("ln_oos_arr_pop", "OutOfStateBorn_ln_Arrivers"),
        ("r_share", "Song_Baseline_r_share"),
        ("a_share", "Song_Baseline_a_share"),
    ]:
        sub = cell.dropna(subset=[col, "share_rct"])
        if len(sub) > 50:
            m = pf.feols(f"{col} ~ share_rct | {fe}", data=sub, vcov={"CRV1": "met2013"})
            mig_rows.append(extract_pf_row(m, "share_rct", label))

    # 2b. True Origin-MSA Retention Rate (split by 5-yr 1980-2000 vs. 1-yr 2007-2024 horizon!)
    for tag, cond in [
        ("TrueRetention_AllYears", cell["YEAR"] >= 1980),
        ("TrueRetention_5yr_1980_2000", cell["YEAR"] <= 2000),
        ("TrueRetention_1yr_2007_2024", cell["YEAR"] >= 2007),
    ]:
        sub = cell[cond].dropna(subset=["true_retention_rate", "share_rct", "orig_cohort_pop"])
        if len(sub) > 50:
            m = pf.feols(
                f"true_retention_rate ~ share_rct | {fe}",
                data=sub, weights="orig_cohort_pop", vcov={"CRV1": "met2013"},
            )
            mig_rows.append(extract_pf_row(m, "share_rct", tag))

    # 2c. Direct Duncan-Trejo Ethnic Attrition Test
    att_rows = []
    # Test A: High-attrition (European/Hispanic) vs. Low-attrition (recent Asian/African)
    for tier in ["high", "low"]:
        sub_t = cell[cell["attrition_tier"] == tier].dropna(subset=["ln_res_pop", "share_rct"])
        if len(sub_t) > 40:
            m = pf.feols(f"ln_res_pop ~ share_rct | {fe}", data=sub_t, vcov={"CRV1": "met2013"})
            att_rows.append(extract_pf_row(m, "share_rct", f"AttritionTier_{tier}_ln_Residents"))
        sub_w = cell[cell["attrition_tier"] == tier].dropna(subset=["mean_ln_wage", "share_rct", "nat_pop"])
        if len(sub_w) > 40:
            mw = pf.feols(
                f"mean_ln_wage ~ share_rct + mean_age + mean_educ | {fe}",
                data=sub_w, weights="nat_pop", vcov={"CRV1": "met2013"},
            )
            att_rows.append(extract_pf_row(mw, "share_rct", f"AttritionTier_{tier}_ln_Wage"))

    # Test B: Direct mixed-ancestry primary-reporting test (ANCESTR1 vs. ANCESTR2)
    mlong = None
    if mixed_long is not None and len(mixed_long) >= 50:
        mlong = mixed_long.merge(
            cell[["eth", "met2013", "YEAR", "share_rct", "eth_year", "met_year", "eth_met"]],
            on=["eth", "met2013", "YEAR"],
            how="inner",
        )
    elif micro is not None:
        mixed = micro[
            (micro["is_immigrant"] == 0)
            & micro["eth_primary"].notna()
            & micro["eth_secondary"].notna()
            & (micro["eth_primary"] != micro["eth_secondary"])
        ].copy()
        if len(mixed) >= 50:
            m1 = mixed[["YEAR", "met2013", "PERWT", "AGE", "EDUC", "eth_primary"]].rename(columns={"eth_primary": "eth"})
            m1["is_primary"] = 1.0
            m2 = mixed[["YEAR", "met2013", "PERWT", "AGE", "EDUC", "eth_secondary"]].rename(columns={"eth_secondary": "eth"})
            m2["is_primary"] = 0.0
            mlong = pd.concat([m1, m2], ignore_index=True).merge(
                cell[["eth", "met2013", "YEAR", "share_rct", "eth_year", "met_year", "eth_met"]],
                on=["eth", "met2013", "YEAR"],
                how="inner",
            )
    if mlong is not None and len(mlong) >= 100:
        m_dt = pf.feols(
            "is_primary ~ share_rct + AGE + EDUC | eth_year + met_year + eth_met",
            data=mlong, weights="PERWT", vcov={"CRV1": "met2013"},
        )
        att_rows.append(extract_pf_row(m_dt, "share_rct", "DuncanTrejo_Direct_PrimaryAncestrySwitch"))

    return pd.DataFrame(mig_rows), pd.DataFrame(att_rows)


# ---------------------------------------------------------------------------
# PILLAR 3: Card-Peri Flows, PPML, JRS (2018) Dynamic IV & 2019-2024 WFH Test
# ---------------------------------------------------------------------------
def run_pillar3_modern_iv_and_wfh(
    micro: pd.DataFrame | None,
    cell: pd.DataFrame,
    wfh_agg: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fe = "eth_year + met_year + eth_met"
    iv_rows = []

    # 3a. Card-Peri (2016) flow spec (OLS & PPML fepois keeping all zero cells!)
    sub_cp = cell.dropna(subset=["cp_flow", "iv_bartik_flow"]).copy()
    if len(sub_cp) > 50:
        # PPML on counts (includes all zero-population cells!)
        for cnt_col, lbl in [("res_pop", "PPML_Residents_CardPeriFlow"), ("arr_pop", "PPML_Arrivers_CardPeriFlow")]:
            try:
                mp = pf.fepois(f"{cnt_col} ~ cp_flow | {fe}", data=sub_cp, vcov={"CRV1": "met2013"})
                iv_rows.append(extract_pf_row(mp, "cp_flow", lbl))
            except Exception:
                pass

        # Single-instrument leave-out Bartik IV (1980 base shares) on ln_res_pop, true_retention_rate, mean_ln_wage
        for ycol, lbl, wcol in [
            ("ln_res_pop", "BartikIV_1980_ln_Residents", None),
            ("true_retention_rate", "BartikIV_1980_TrueRetention", "orig_cohort_pop"),
            ("mean_ln_wage", "BartikIV_1980_ln_Wage", "nat_pop"),
        ]:
            s_iv = sub_cp.dropna(subset=[ycol] + ([wcol] if wcol else []))
            if len(s_iv) > 50:
                try:
                    m_iv = pf.feols(
                        f"{ycol} ~ 1 + [cp_flow ~ iv_bartik_flow] | {fe}",
                        data=s_iv, weights=wcol, vcov={"CRV1": "met2013"},
                    )
                    iv_rows.append(extract_pf_row(m_iv, "cp_flow", lbl))
                except Exception:
                    pass

    # 3b. Jaeger-Ruist-Stuhler (2018) 2-instrument dynamic shift-share IV
    # Contemporaneous cp_flow + lagged cp_flow_lag instrumented by iv_bartik_flow + iv_bartik_flow_lag
    sub_jrs = cell.dropna(subset=["cp_flow", "cp_flow_lag", "iv_bartik_flow", "iv_bartik_flow_lag"]).copy()
    for ycol, lbl, wcol in [
        ("ln_res_pop", "JRS2018_DynamicIV_ln_Residents", None),
        ("true_retention_rate", "JRS2018_DynamicIV_TrueRetention", "orig_cohort_pop"),
        ("mean_ln_wage", "JRS2018_DynamicIV_ln_Wage", "nat_pop"),
    ]:
        res = fit_fwl_2sls(
            df=sub_jrs,
            y_col=ycol,
            endog_cols=["cp_flow", "cp_flow_lag"],
            instr_cols=["iv_bartik_flow", "iv_bartik_flow_lag"],
            exog_cols=[],
            fe_cols=["eth_year", "met_year", "eth_met"],
            cluster_cols=["met2013", "eth"],  # Two-way MSA x Ethnicity clustering!
            weight_col=wcol,
        )
        if res.get("status") == "ok":
            for v in ["cp_flow", "cp_flow_lag"]:
                iv_rows.append({
                    "spec": f"{lbl}_{v}",
                    "variable": v,
                    "coef": res["coefs"][v],
                    "se": res["ses"][v],
                    "tstat": res["tstats"][v],
                    "pval": res["pvals"][v],
                    "n": res["n"],
                    "r2": np.nan,
                    "first_stage_f": res["first_stage_f"][v],
                    "cragg_donald_f": res["cragg_donald_f"],
                })

    # 3c. 2019-2024 Remote-Work (WFH, TRANWORK==80) Pure Enclave Amenity Test
    wfh_rows = []
    if wfh_agg is not None and len(wfh_agg) >= 100:
        acs_mod = wfh_agg.copy()
        acs_mod["eth_year"] = acs_mod["eth"].astype(str) + "_" + acs_mod["YEAR"].astype(str)
        acs_mod["met_year"] = acs_mod["met2013"].astype(str) + "_" + acs_mod["YEAR"].astype(str)
        acs_mod["eth_met"] = acs_mod["eth"].astype(str) + "_" + acs_mod["met2013"].astype(str)
        acs_mod["is_wfh"] = acs_mod["is_wfh"].astype(float)
    elif micro is not None:
        acs_mod = micro[(micro["is_immigrant"] == 0) & (micro["YEAR"] >= 2019) & micro["eth_primary"].notna()].copy()
        acs_mod = acs_mod.merge(
            cell[["eth", "met2013", "YEAR", "share_rct", "eth_year", "met_year", "eth_met"]],
            left_on=["eth_primary", "met2013", "YEAR"],
            right_on=["eth", "met2013", "YEAR"],
            how="inner",
        )
        acs_mod["is_wfh"] = (acs_mod["TRANWORK"] == 80).astype(float)
    else:
        acs_mod = pd.DataFrame()

    if len(acs_mod) >= 100:
        acs_mod["share_x_wfh"] = acs_mod["share_rct"] * acs_mod["is_wfh"]
        acs_mod["share_x_tele"] = acs_mod["share_rct"] * acs_mod["teleworkable"].astype(float)

        for ycol, ylbl in [("resident", "Resident_Prob"), ("ln_wage", "Log_Wage")]:
            sub_w = acs_mod.dropna(subset=[ycol, "share_rct", "is_wfh", "AGE", "EDUC"])
            if len(sub_w) >= 100:
                m_wfh = pf.feols(
                    f"{ycol} ~ share_rct + is_wfh + share_x_wfh + AGE + EDUC + SEX | eth_year + met_year + eth_met",
                    data=sub_w, weights="PERWT", vcov={"CRV1": "met2013"},
                )
                wfh_rows.append(extract_pf_row(m_wfh, "share_rct", f"WFH_2019_2024_{ylbl}_OnSiteBase"))
                wfh_rows.append(extract_pf_row(m_wfh, "share_x_wfh", f"WFH_2019_2024_{ylbl}_WFH_Interaction"))

                m_tele = pf.feols(
                    f"{ycol} ~ share_rct + teleworkable + share_x_tele + AGE + EDUC + SEX | eth_year + met_year + eth_met",
                    data=sub_w, weights="PERWT", vcov={"CRV1": "met2013"},
                )
                wfh_rows.append(extract_pf_row(m_tele, "share_x_tele", f"Teleworkable_2019_2024_{ylbl}_Interaction"))

    return pd.DataFrame(iv_rows), pd.DataFrame(wfh_rows)


# ---------------------------------------------------------------------------
# Publication-Grade Figures
# ---------------------------------------------------------------------------
def make_figures(
    t1_housing: pd.DataFrame,
    t2_msa: pd.DataFrame,
    t3_mig: pd.DataFrame,
    t4_att: pd.DataFrame,
    t5_iv: pd.DataFrame,
    t6_wfh: pd.DataFrame,
    fig_dir: Path,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    def coef_plot(ax: Any, df: pd.DataFrame, title: str, xlabel: str) -> None:
        if df.empty:
            ax.set_title(f"{title} (no data)")
            return
        y = np.arange(len(df))
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.errorbar(
            df["coef"], y, xerr=1.96 * df["se"], fmt="o", color="#1f4e79",
            ecolor="#4a7fb5", elinewidth=2, capsize=4, markersize=6,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(df["spec"], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontweight="bold")

    # Figure 1: Housing Price Puzzle
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    coef_plot(axes[0], t1_housing, "Panel A: Table 7 Stress-Test & Mechanisms", "Coefficient (95% CI)")
    coef_plot(axes[1], t2_msa, "Panel B: MSA-Level Housing Prices (Saiz Test)", "Δln(HPI/ZHVI/ZORI) per 1pp Inflow")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_housing_puzzle.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Migration Decomposition & Duncan-Trejo Attrition
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    coef_plot(axes[0], t3_mig, "Panel A: Stock vs. Out-of-State vs. True Retention", "Coefficient on Share_rct (95% CI)")
    coef_plot(axes[1], t4_att, "Panel B: Duncan-Trejo Ethnic Attrition Tests", "Coefficient on Share_rct (95% CI)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_migration_vs_attrition.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: Modern Dynamic Shift-Share IV & 2019-2024 WFH Amenity Test
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    coef_plot(axes[0], t5_iv, "Panel A: Card-Peri PPML & JRS (2018) Dynamic IV", "Coefficient (95% CI)")
    coef_plot(axes[1], t6_wfh, "Panel B: 2019–2024 WFH Enclave Amenity Test", "Coefficient (95% CI)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_jrs_dynamic_and_wfh.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Essay 3 Replication & Novel Extensions Pipeline.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run end-to-end on synthetic microdata and save outputs to results/essay3/smoke/.",
    )
    args = parser.parse_args()

    out_dir = ROOT / "results" / "essay3" / ("smoke" if args.smoke_test else "")
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        print("[SMOKE-TEST] Generating synthetic IPUMS-schema microdata (1980-2024)...")
        micro = build_synthetic_microdata()
        total_raw_n = int(len(micro))
        print(f"   Microdata rows: {total_raw_n:,}, years: {sorted(micro['YEAR'].unique())}, metros: {micro['met2013'].nunique()}")
        print("[STEP 1] Building balanced (eth, met2013, year) cell panel + Card-Peri flows + Leave-Out Bartik IV...")
        cell, msa_imm = build_cell_panel(micro)
        mixed_long, wfh_agg = None, None
    else:
        print("[FULL RUN] Streaming & aggregating real 1980-2024 IPUMS USA microdata in DuckDB...")
        micro = None
        cell, msa_imm, mixed_long, wfh_agg, total_raw_n = load_and_aggregate_real_ipums()
    print(f"   Cell panel shape: {cell.shape}, MSA panel shape: {msa_imm.shape}")

    print("[STEP 2] Pillar 1 — Resolving the Housing Price Puzzle (Table 7 + Saiz MSA Test)...")
    t1_housing, t2_msa = run_pillar1_housing(cell, msa_imm)
    t1_housing.to_csv(out_dir / "table1_housing_puzzle.csv", index=False)
    t2_msa.to_csv(out_dir / "table2_msa_housing_saiz.csv", index=False)

    print("[STEP 3] Pillar 2 — True Migration vs. Fertility-in-Place & Duncan-Trejo Attrition...")
    t3_mig, t4_att = run_pillar2_migration_and_attrition(micro, cell, mixed_long=mixed_long)
    t3_mig.to_csv(out_dir / "table3_migration_decomposition.csv", index=False)
    t4_att.to_csv(out_dir / "table4_duncan_trejo_attrition.csv", index=False)

    print("[STEP 4] Pillar 3 — Card-Peri PPML, JRS (2018) Dynamic IV & 2019-2024 WFH Amenity Test...")
    t5_iv, t6_wfh = run_pillar3_modern_iv_and_wfh(micro, cell, wfh_agg=wfh_agg)
    t5_iv.to_csv(out_dir / "table5_card_peri_ppml_jrs_iv.csv", index=False)
    t6_wfh.to_csv(out_dir / "table6_wfh_amenity_test.csv", index=False)

    print("[STEP 5] Generating publication-grade figures...")
    make_figures(t1_housing, t2_msa, t3_mig, t4_att, t5_iv, t6_wfh, fig_dir)

    summary = {
        "mode": "smoke_test" if args.smoke_test else "real_ipums",
        "micro_rows": int(total_raw_n),
        "cell_rows": int(len(cell)),
        "tables": {
            "table1_housing_puzzle": t1_housing.to_dict(orient="records"),
            "table2_msa_housing_saiz": t2_msa.to_dict(orient="records"),
            "table3_migration_decomposition": t3_mig.to_dict(orient="records"),
            "table4_duncan_trejo_attrition": t4_att.to_dict(orient="records"),
            "table5_card_peri_ppml_jrs_iv": t5_iv.to_dict(orient="records"),
            "table6_wfh_amenity_test": t6_wfh.to_dict(orient="records"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n=== DONE ({'SMOKE-TEST' if args.smoke_test else 'FULL RUN'}) -> {out_dir} ===")
    for name, df in [
        ("Table 1 (Housing Puzzle)", t1_housing),
        ("Table 2 (MSA Saiz Test)", t2_msa),
        ("Table 3 (Migration Decomp)", t3_mig),
        ("Table 4 (Duncan-Trejo)", t4_att),
        ("Table 5 (Card-Peri/PPML/JRS IV)", t5_iv),
        ("Table 6 (2019-2024 WFH Test)", t6_wfh),
    ]:
        print(f"  {name:32s}: {len(df):2d} specs estimated")


if __name__ == "__main__":
    main()
