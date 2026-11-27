#!/usr/bin/env python3
"""
Paper 1 -- ACS Identification: English-Origin vs. Non-English-Origin Immigrants
===============================================================================
Executes the full 2015-2024 ACS empirical suite on data/ipums/raw/acs_2007_2024/usa_00012.csv.gz:
  1. Origin-Comparison Triple-Difference (Non-English-Origin vs. English-Origin Immigrants
     relative to U.S. Natives) across pre-period occupational exposures (O*NET Written English,
     Eloundou et al. 2023/2024 OpenAI Beta, Felten et al. 2021/2023 AIOE/LM, LLM_Writing,
     LLM_Coding_Math, and FaceToFace_Social), for both 2015-2024 and 2021-2024, with 80% MDEs.
  2. Dynamic Annual Event Studies (2015-2024, ref = 2022) for:
       (a) Non-English-Origin vs. Natives
       (b) English-Origin vs. Natives (Selection / Placebo Diagnostic)
       (c) Non-English-Origin vs. English-Origin (Immigrants-Only DDD)
     Saving exact coefficient vectors + cluster-robust VCOV matrices for LP-based HonestDiD.
  3. Simultaneous Task Horse Race: LLM_Writing vs. LLM_Coding_Math vs. FaceToFace_Social
     (plus OpenAI E1 & Felten LM robustness).
  4. Rich Heterogeneity: SPEAKENG (Fluent vs. Non-Fluent), DEGFIELD (College STEM vs.
     College Non-STEM vs. Non-College), Gender (Men vs. Women), Age (18-34, 35-49, 50-64),
     and Years in U.S. (<10 vs. >=10).
  5. Extensive Margin & Occupational Sorting (2015-2024) for Non-English vs. English-Origin.
  6. Exact LP-based Rambachan & Roth (2023) Relative-Magnitude Sensitivity Bounds
     (Delta^RM(Mbar), Mbar = 0, 0.5, 1.0, 1.5, 2.0) using actual per-year step scaling
     for uneven ACS years (2015, 2018, 2019, 2021, 2022, 2023, 2024) via scipy.optimize.linprog.

Outputs written to papers/paper1_genai_immigrants/results/.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import optimize, stats

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[3]
ACS_FILE = ROOT / "data" / "ipums" / "raw" / "acs_2007_2024" / "usa_00012.csv.gz"
RES_DIR = ROOT / "papers" / "paper1_genai_immigrants" / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)

# STEM DEGFIELD codes (IPUMS ACS)
STEM_DEGFIELDS = (21, 24, 25, 36, 37, 50, 51)


def load_acs_cells() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    Build compressed sufficient-statistic cell panel from 2015-2024 ACS microdata.
    Drops U.S. Territories (BPL 100..149) for clean Native vs. Foreign-Born split.
    Classifies English-speaking-origin immigrants per Song (2017) / CIA Factbook:
      - Canada (BPL=150, excluding Quebec/French Canada BPLD>=15080)
      - UK & Ireland (BPL IN 410..414)
      - Australia & New Zealand (BPL=700)
      - English-speaking Caribbean, Belize & Guyana (BPLD IN 21010, 26030, 30040 or 26040..26069)
    """
    con = duckdb.connect()
    t6 = RES_DIR / "tasks_occsoc.parquet"
    t4 = RES_DIR / "tasks_occsoc4.parquet"
    con.execute(f"CREATE VIEW t6 AS SELECT * FROM read_parquet('{t6}')")
    con.execute(f"CREATE VIEW t4 AS SELECT * FROM read_parquet('{t4}')")

    stem_sql = ",".join(str(x) for x in STEM_DEGFIELDS)

    # Origin SQL condition (exact BPL/BPLD)
    eng_origin_sql = """
        (BPL = 150 AND BPLD < 15080)
        OR BPL IN (410, 411, 412, 413, 414, 700)
        OR BPLD IN (21010, 26030, 30040)
        OR (BPLD BETWEEN 26040 AND 26069)
    """

    print("Querying 2015-2024 ACS FTFY wage sample via DuckDB...")
    cell_df = con.execute(f"""
        WITH workers AS (
            SELECT
                YEAR AS year,
                STATEFIP AS statefip,
                CASE
                    WHEN EDUCD < 62 THEN 1
                    WHEN EDUCD BETWEEN 62 AND 64 THEN 2
                    WHEN EDUCD BETWEEN 65 AND 100 THEN 3
                    ELSE 4
                END AS educ4,
                SEX AS sex,
                CASE
                    WHEN BPL < 100 THEN 0
                    WHEN ({eng_origin_sql}) THEN 1
                    ELSE 2
                END AS origin_group,
                CASE WHEN BPL >= 150 AND SPEAKENG IN (1, 5, 6) THEN 1 ELSE 0 END AS non_fluent,
                CASE
                    WHEN EDUCD >= 101 AND DEGFIELD IN ({stem_sql}) THEN 2
                    WHEN EDUCD >= 101 THEN 1
                    ELSE 0
                END AS stem_group,
                CASE
                    WHEN AGE BETWEEN 18 AND 34 THEN 1
                    WHEN AGE BETWEEN 35 AND 49 THEN 2
                    ELSE 3
                END AS age_group,
                CASE WHEN BPL >= 150 AND YRSUSA1 < 10 THEN 1 ELSE 0 END AS recent_arr,
                REGEXP_REPLACE(UPPER(TRIM(CAST(OCCSOC AS VARCHAR))), '[-\\s]', '', 'g') AS occsoc_clean,
                AGE AS age,
                AGE * AGE AS age2,
                CASE WHEN RACE = 2 THEN 1.0 ELSE 0.0 END AS is_black,
                CASE WHEN RACE IN (4, 5, 6) THEN 1.0 ELSE 0.0 END AS is_asian,
                CASE WHEN HISPAN BETWEEN 1 AND 4 THEN 1.0 ELSE 0.0 END AS is_hispanic,
                PERWT AS perwt,
                LN((INCWAGE * CPI99) / (UHRSWORK *
                    CASE WKSWORK2 WHEN 1 THEN 7.0 WHEN 2 THEN 20.0 WHEN 3 THEN 33.0
                         WHEN 4 THEN 43.5 WHEN 5 THEN 48.5 WHEN 6 THEN 51.0 ELSE 45.0 END)) AS ln_wage
            FROM read_csv_auto('{ACS_FILE}', header=true, types={{'OCCSOC': 'VARCHAR'}})
            WHERE YEAR >= 2015
              AND AGE BETWEEN 18 AND 64
              AND EMPSTAT = 1
              AND SCHOOL != 2
              AND EDUCD > 1 AND EDUCD < 999
              AND UHRSWORK >= 35
              AND WKSWORK2 >= 4
              AND INCWAGE > 0 AND INCWAGE < 999998
              AND (BPL < 100 OR BPL >= 150)
              AND ((INCWAGE * CPI99) / (UHRSWORK *
                    CASE WKSWORK2 WHEN 1 THEN 7.0 WHEN 2 THEN 20.0 WHEN 3 THEN 33.0
                         WHEN 4 THEN 43.5 WHEN 5 THEN 48.5 WHEN 6 THEN 51.0 ELSE 45.0 END)) BETWEEN 2.0 AND 200.0
        )
        SELECT
            w.year,
            w.statefip,
            w.educ4,
            w.sex,
            w.origin_group,
            w.non_fluent,
            w.stem_group,
            w.age_group,
            w.recent_arr,
            w.occsoc_clean AS occsoc,
            SUM(w.perwt) AS cell_wt,
            COUNT(*) AS cell_n,
            SUM(w.ln_wage * w.perwt) / SUM(w.perwt) AS ln_wage,
            SUM(w.age * w.perwt) / SUM(w.perwt) AS age,
            SUM(w.age2 * w.perwt) / SUM(w.perwt) AS age2,
            SUM(w.is_black * w.perwt) / SUM(w.perwt) AS is_black,
            SUM(w.is_asian * w.perwt) / SUM(w.perwt) AS is_asian,
            SUM(w.is_hispanic * w.perwt) / SUM(w.perwt) AS is_hispanic,
            COALESCE(t6.llm_openai_beta, t4.llm_openai_beta) AS llm_openai_beta,
            COALESCE(t6.llm_openai_alpha, t4.llm_openai_alpha) AS llm_openai_alpha,
            COALESCE(t6.llm_felten_lm, t4.llm_felten_lm) AS llm_felten_lm,
            COALESCE(t6.written_eng_only, t4.written_eng_only) AS written_eng_only,
            COALESCE(t6.coding_math_only, t4.coding_math_only) AS coding_math_only,
            COALESCE(t6.llm_writing_raw, t4.llm_writing_raw) AS llm_writing_raw,
            COALESCE(t6.llm_coding_math_raw, t4.llm_coding_math_raw) AS llm_coding_math_raw,
            COALESCE(t6.llm_writing_orth, t4.llm_writing_orth) AS llm_writing_orth,
            COALESCE(t6.llm_coding_math_orth, t4.llm_coding_math_orth) AS llm_coding_math_orth,
            COALESCE(t6.face_to_face_social, t4.face_to_face_social) AS face_to_face_social,
            COALESCE(t6.face_to_face_orth, t4.face_to_face_orth) AS face_to_face_orth
        FROM workers w
        LEFT JOIN t6 ON w.occsoc_clean = t6.occsoc
        LEFT JOIN t4 ON SUBSTR(w.occsoc_clean, 1, 4) = t4.occsoc4
        WHERE COALESCE(t6.llm_openai_beta, t4.llm_openai_beta) IS NOT NULL
        GROUP BY
            w.year, w.statefip, w.educ4, w.sex, w.origin_group, w.non_fluent,
            w.stem_group, w.age_group, w.recent_arr, w.occsoc_clean,
            COALESCE(t6.llm_openai_beta, t4.llm_openai_beta),
            COALESCE(t6.llm_openai_alpha, t4.llm_openai_alpha),
            COALESCE(t6.llm_felten_lm, t4.llm_felten_lm),
            COALESCE(t6.written_eng_only, t4.written_eng_only),
            COALESCE(t6.coding_math_only, t4.coding_math_only),
            COALESCE(t6.llm_writing_raw, t4.llm_writing_raw),
            COALESCE(t6.llm_coding_math_raw, t4.llm_coding_math_raw),
            COALESCE(t6.llm_writing_orth, t4.llm_writing_orth),
            COALESCE(t6.llm_coding_math_orth, t4.llm_coding_math_orth),
            COALESCE(t6.face_to_face_social, t4.face_to_face_social),
            COALESCE(t6.face_to_face_orth, t4.face_to_face_orth)
    """).df()

    # Extensive-margin / employment summary by origin group & year (all prime-age non-students)
    print("Querying 2015-2024 ACS extensive-margin & sorting sample...")
    emp_df = con.execute(f"""
        SELECT
            YEAR AS year,
            STATEFIP AS statefip,
            CASE
                WHEN EDUCD < 62 THEN 1
                WHEN EDUCD BETWEEN 62 AND 64 THEN 2
                WHEN EDUCD BETWEEN 65 AND 100 THEN 3
                ELSE 4
            END AS educ4,
            SEX AS sex,
            CASE
                WHEN BPL < 100 THEN 0
                WHEN ({eng_origin_sql}) THEN 1
                ELSE 2
            END AS origin_group,
            SUM(PERWT) AS cell_wt,
            COUNT(*) AS cell_n,
            SUM(CASE WHEN EMPSTAT = 1 THEN PERWT ELSE 0 END) / SUM(PERWT) AS emp_rate,
            SUM(CASE WHEN EMPSTAT = 1 AND UHRSWORK >= 35 AND WKSWORK2 >= 4 THEN PERWT ELSE 0 END) / SUM(PERWT) AS ftfy_rate,
            SUM(AGE * PERWT) / SUM(PERWT) AS age,
            SUM(AGE * AGE * PERWT) / SUM(PERWT) AS age2
        FROM read_csv_auto('{ACS_FILE}', header=true)
        WHERE YEAR >= 2015
          AND AGE BETWEEN 18 AND 64
          AND SCHOOL != 2
          AND EDUCD > 1 AND EDUCD < 999
          AND (BPL < 100 OR BPL >= 150)
        GROUP BY 1, 2, 3, 4, 5
    """).df()

    counts = {
        "n_workers_total": int(cell_df["cell_n"].sum()),
        "n_workers_native": int(cell_df.loc[cell_df["origin_group"] == 0, "cell_n"].sum()),
        "n_workers_eng_imm": int(cell_df.loc[cell_df["origin_group"] == 1, "cell_n"].sum()),
        "n_workers_noneng_imm": int(cell_df.loc[cell_df["origin_group"] == 2, "cell_n"].sum()),
        "n_cells": int(len(cell_df)),
        "n_occsoc_clusters": int(cell_df["occsoc"].nunique()),
        "n_state_clusters": int(cell_df["statefip"].nunique()),
    }
    print(f"Loaded {counts['n_workers_total']:,} FTFY workers ({counts['n_cells']:,} cells): "
          f"Natives={counts['n_workers_native']:,}, Eng-Origin={counts['n_workers_eng_imm']:,}, "
          f"NonEng-Origin={counts['n_workers_noneng_imm']:,}")
    return cell_df, emp_df, counts


MDE_80_MULT = float(stats.norm.ppf(0.975) + stats.norm.ppf(0.80))  # 2.801585


def fit_twoway_logged(
    fml: str,
    data: pd.DataFrame,
    weights: str = "cell_wt",
    c1: str = "occsoc",
    c2: str = "statefip",
    context: str = "",
):
    """
    Estimate feols with two-way CRV1 clustering (c1 + c2).
    NEVER silently drops two-way clustering: if any SE is NaN or non-finite,
    logs an explicit warning and returns the exact se_type label so every
    output CSV records the true inference method used.
    """
    se_label_2w = f"twoway_{c1}_{c2}"
    fit = pf.feols(fml, data=data, weights=weights, vcov={"CRV1": f"{c1} + {c2}"})
    se_arr = fit.se().to_numpy(dtype=float)
    if np.all(np.isfinite(se_arr)) and np.all(se_arr > 0):
        return fit, se_label_2w
    se_label_1w = f"oneway_{c1}_fallback"
    print(f"  [SE_FALLBACK_WARNING] {context}: two-way CRV1 ({c1}+{c2}) yielded NaN/non-pos SE; "
          f"explicitly recording {se_label_1w}")
    fit_1w = pf.feols(fml, data=data, weights=weights, vcov={"CRV1": c1})
    return fit_1w, se_label_1w


def _solve_rm_lp_bounds(
    l_vec: np.ndarray,
    post_dt: np.ndarray,
    max_pre_slope: float,
    mbar: float,
) -> Tuple[float, float]:
    """
    Solve the exact Linear Program (scipy.optimize.linprog, HiGHS) for the
    identified-set bias bounds [b_min, b_max] of l_vec' * delta_post over
    delta_post in Delta^RM(Mbar) with uneven time steps post_dt:
      delta_0 = 0,
      |(delta_s - delta_{s-1}) / post_dt[s-1]| <= mbar * max_pre_slope  for s = 1..T_post.
    Returns (b_min, b_max).
    """
    t_post = len(l_vec)
    if mbar <= 0.0 or max_pre_slope <= 0.0:
        return 0.0, 0.0

    # Build inequality constraints A_ub @ delta <= b_ub:
    # For s = 0:  delta_0_idx <= bound_0  and -delta_0_idx <= bound_0
    # For s >= 1: delta_s - delta_{s-1} <= bound_s  and -(delta_s - delta_{s-1}) <= bound_s
    # where bound_s = mbar * max_pre_slope * post_dt[s].
    A_rows = []
    b_vals = []
    for s in range(t_post):
        bound_s = mbar * max_pre_slope * float(post_dt[s])
        row_pos = np.zeros(t_post)
        row_pos[s] = 1.0
        if s > 0:
            row_pos[s - 1] = -1.0
        A_rows.append(row_pos)
        b_vals.append(bound_s)
        A_rows.append(-row_pos)
        b_vals.append(bound_s)

    A_ub = np.vstack(A_rows)
    b_ub = np.array(b_vals, dtype=float)
    bounds = [(None, None)] * t_post

    res_min = optimize.linprog(c=l_vec, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    res_max = optimize.linprog(c=-l_vec, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not (res_min.success and res_max.success):
        raise RuntimeError(f"HonestDiD LP failed: min={res_min.message}, max={res_max.message}")
    return float(res_min.fun), float(-res_max.fun)


def compute_honestdid_rm_bounds(
    beta_pre: np.ndarray,
    beta_post: np.ndarray,
    vcov_full: np.ndarray,
    l_vec: np.ndarray,
    pre_times: List[float],
    ref_time: float,
    post_times: List[float],
    mbar_grid: List[float] = [0.0, 0.5, 1.0, 1.5, 2.0],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Python LP implementation of Rambachan & Roth (2023, ReStud) Relative Magnitudes
    (Delta^RM(Mbar)) sensitivity bounds with explicit per-period time-step scaling.

    Given pre-treatment event-study estimates beta_pre at times pre_times (< ref_time),
    reference period ref_time (delta = 0), and post-treatment estimates beta_post at
    times post_times (> ref_time), with target parameter theta = l_vec' * tau_post:

    Under time-scaled Delta^RM(Mbar):
      Pre-period per-unit-time slopes: r_j = (delta_{t_{j+1}} - delta_{t_j}) / (t_{j+1} - t_j)
      for consecutive observed pre-periods ending at ref_time (delta = 0).
      M_pre = max_j |r_j|.
      For each post-treatment interval (s_{p-1}, s_p) starting from s_0 = ref_time:
        |(delta_{s_p} - delta_{s_{p-1}}) / (s_p - s_{p-1})| <= Mbar * M_pre.

    We compute:
      1. Exact LP identified-set bounds [theta_hat - b_max, theta_hat - b_min] via
         scipy.optimize.linprog (HiGHS) at the plug-in M_pre.
      2. Fixed-length CI (FLCI) centered at theta_hat with exact bias-aware folded-normal
         critical value cv_alpha(b_max / se_theta) satisfying P(|N(b/se, 1)| <= cv) = 1 - alpha.
      3. Parametric Monte Carlo robust CI (10,000 draws from N(beta_hat, V_hat)) accounting
         for joint estimation uncertainty in beta_post and M_pre.
    """
    t_pre = len(beta_pre)
    t_post = len(beta_post)
    assert len(pre_times) == t_pre and len(post_times) == t_post

    theta_hat = float(l_vec @ beta_post)
    v_post = vcov_full[t_pre:, t_pre:]
    se_theta = float(np.sqrt(max(float(l_vec @ v_post @ l_vec), 1e-16)))

    # Actual elapsed time steps for pre-periods (ending at ref_time) and post-periods (starting from ref_time)
    pre_t_arr = np.array(list(pre_times) + [ref_time], dtype=float)
    pre_dt = np.diff(pre_t_arr)  # e.g., ACS [3, 1, 2, 1] for [2015, 2018, 2019, 2021, 2022]
    post_t_arr = np.array([ref_time] + list(post_times), dtype=float)
    post_dt = np.diff(post_t_arr)  # e.g., ACS [1, 1] for [2022, 2023, 2024]; CPS [2, 1, ..., 1] if 2022Q4 skipped

    pre_with_ref = np.concatenate([beta_pre, [0.0]])
    pre_slopes = np.abs(np.diff(pre_with_ref) / pre_dt)
    max_pre_slope = float(np.max(pre_slopes)) if len(pre_slopes) > 0 else 0.0

    def folded_normal_cv(bias_over_se: float) -> float:
        if bias_over_se <= 1e-12:
            return float(stats.norm.ppf(1.0 - alpha / 2.0))
        def f(c: float) -> float:
            return (stats.norm.cdf(c - bias_over_se) - stats.norm.cdf(-c - bias_over_se)) - (1.0 - alpha)
        return float(optimize.brentq(f, 1.95, bias_over_se + 6.0))

    # Parametric MC draws from N(beta_hat, vcov_full) to incorporate uncertainty in M_pre
    rng = np.random.default_rng(2026)
    beta_full = np.concatenate([beta_pre, beta_post])
    eigvals, eigvecs = np.linalg.eigh(vcov_full)
    eigvals = np.maximum(eigvals, 1e-12)
    vcov_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    draws = rng.multivariate_normal(beta_full, vcov_psd, size=10000)

    # Unit-bound LP (mbar=1, max_pre_slope=1) gives exact linear scaling factors since LP is homogeneous of degree 1
    unit_bmin, unit_bmax = _solve_rm_lp_bounds(l_vec, post_dt, max_pre_slope=1.0, mbar=1.0)

    rows = []
    for mbar in mbar_grid:
        b_min, b_max = _solve_rm_lp_bounds(l_vec, post_dt, max_pre_slope=max_pre_slope, mbar=mbar)
        bias_max = max(abs(b_min), abs(b_max))
        cv_flci = folded_normal_cv(bias_max / se_theta) if se_theta > 0 else 1.96
        flci_lo = theta_hat - cv_flci * se_theta
        flci_hi = theta_hat + cv_flci * se_theta

        # MC draws accounting for uncertainty in both theta_hat and max_pre_slope
        pre_draws = np.hstack([draws[:, :t_pre], np.zeros((len(draws), 1))])
        max_slope_draws = np.max(np.abs(np.diff(pre_draws, axis=1) / pre_dt[None, :]), axis=1)
        theta_draws = draws[:, t_pre:] @ l_vec
        # Identified set for tau = theta - l'delta is [theta - b_max, theta - b_min]
        lo_draws = theta_draws - mbar * max_slope_draws * unit_bmax
        hi_draws = theta_draws - mbar * max_slope_draws * unit_bmin
        rob_ci_lo = float(np.quantile(lo_draws, alpha / 2.0))
        rob_ci_hi = float(np.quantile(hi_draws, 1.0 - alpha / 2.0))

        rows.append({
            "mbar": mbar,
            "theta_hat": theta_hat,
            "se_theta": se_theta,
            "max_pre_slope": max_pre_slope,
            "bias_lp_max": bias_max,
            "id_set_lo": theta_hat - b_max,
            "id_set_hi": theta_hat - b_min,
            "flci_lo": flci_lo,
            "flci_hi": flci_hi,
            "rob_ci_lo": rob_ci_lo,
            "rob_ci_hi": rob_ci_hi,
            "excludes_zero_flci": bool(flci_lo > 0 or flci_hi < 0),
            "excludes_zero_rob": bool(rob_ci_lo > 0 or rob_ci_hi < 0),
            "method": "python_lp_delta_rm_time_scaled",
        })
    return pd.DataFrame(rows)


def run_origin_placebo_suite(cell_df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 1: Origin-Comparison Triple-Difference (Non-English-Origin vs. English-Origin).
    Simultaneously estimates Post2022 x NonEngOrigin x Task and Post2022 x EngOrigin x Task
    (with U.S. Natives as reference), absorbing occsoc:year + statefip:year + educ4:year +
    origin_group:year + sex:year, two-way clustered by occsoc + statefip.
    Logs exact se_type and computes 80%-power MDE (2.8016 * SE) for every specification.
    """
    df = cell_df.copy()
    df["post"] = (df["year"] >= 2023).astype(float)
    df["is_eng"] = (df["origin_group"] == 1).astype(float)
    df["is_noneng"] = (df["origin_group"] == 2).astype(float)

    tasks = [
        ("written_eng_only", "Pre-2022 O*NET Written-English Intensity"),
        ("llm_openai_beta", "OpenAI LLM Exposure (E1+0.5*E2)"),
        ("llm_writing_raw", "LLM Writing (Raw)"),
        ("llm_writing_orth", "LLM Writing (Orthogonalized)"),
        ("llm_coding_math_raw", "LLM Coding & Math (Raw)"),
        ("llm_coding_math_orth", "LLM Coding & Math (Orthogonalized)"),
        ("face_to_face_social", "Face-to-Face Social Skills"),
        ("llm_felten_lm", "Felten Language-Modeling AIOE"),
    ]

    rows = []
    for win_name, sub in [("2015-2024", df), ("2021-2024", df[df["year"] >= 2021].copy())]:
        for tcol, tlab in tasks:
            sub = sub.copy()
            sub["eng_t"] = sub["is_eng"] * sub[tcol]
            sub["noneng_t"] = sub["is_noneng"] * sub[tcol]
            sub["post_eng_t"] = sub["post"] * sub["eng_t"]
            sub["post_noneng_t"] = sub["post"] * sub["noneng_t"]

            fit, se_type = fit_twoway_logged(
                "ln_wage ~ eng_t + noneng_t + post_eng_t + post_noneng_t"
                " + age + age2 + is_black + is_asian + is_hispanic"
                " | occsoc:year + statefip:year + educ4:year + origin_group:year + sex:year",
                data=sub,
                weights="cell_wt",
                c1="occsoc",
                c2="statefip",
                context=f"ACS_origin_{win_name}_{tcol}",
            )
            coefs = fit.coef()
            ses = fit.se()
            pvals = fit.pvalue()
            vcov = fit._vcov
            names = list(coefs.index)
            i_ne = names.index("post_noneng_t")
            i_en = names.index("post_eng_t")

            b_ne = float(coefs["post_noneng_t"])
            se_ne = float(ses["post_noneng_t"])
            p_ne = float(pvals["post_noneng_t"])

            b_en = float(coefs["post_eng_t"])
            se_en = float(ses["post_eng_t"])
            p_en = float(pvals["post_eng_t"])

            diff = b_ne - b_en
            var_diff = float(vcov[i_ne, i_ne] + vcov[i_en, i_en] - 2.0 * vcov[i_ne, i_en])
            se_diff = float(np.sqrt(max(var_diff, 1e-12)))
            t_diff = diff / se_diff
            p_diff = float(2.0 * (1.0 - stats.norm.cdf(abs(t_diff))))

            rows.append({
                "window": win_name,
                "task": tcol,
                "task_label": tlab,
                "b_noneng": b_ne,
                "se_noneng": se_ne,
                "p_noneng": p_ne,
                "mde80_noneng": MDE_80_MULT * se_ne,
                "b_eng": b_en,
                "se_eng": se_en,
                "p_eng": p_en,
                "ddd_diff": diff,
                "ddd_se": se_diff,
                "ddd_p": p_diff,
                "mde80_ddd": MDE_80_MULT * se_diff,
                "se_type": se_type,
                "n_cells": int(fit._N),
                "n_workers": int(sub["cell_n"].sum()),
                "r2": float(fit._r2),
            })
    res = pd.DataFrame(rows)
    res.to_csv(RES_DIR / "acs_origin_placebo.csv", index=False)
    return res


def run_event_studies_and_honestdid(cell_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Dynamic Annual Event Studies (2015-2024, ref = 2022) + Rambachan-Roth HonestDiD bounds.
    Estimates for:
      - Non-English-Origin vs. Natives
      - English-Origin vs. Natives (Placebo)
      - Non-English-Origin vs. English-Origin (Immigrants-Only DDD)
    Across key tasks: llm_openai_beta, llm_writing_raw, llm_writing_orth, llm_coding_math_orth, face_to_face_social.
    """
    es_years = sorted(cell_df["year"].unique().tolist())
    ref_yr = 2022
    pre_years = [y for y in es_years if y < ref_yr]
    post_years = [y for y in es_years if y > ref_yr]

    es_rows = []
    honest_rows = []

    configs = [
        ("NonEng_vs_Native", cell_df[cell_df["origin_group"].isin([0, 2])].copy(), 2),
        ("Eng_vs_Native", cell_df[cell_df["origin_group"].isin([0, 1])].copy(), 1),
        ("NonEng_vs_Eng_ImmiOnly", cell_df[cell_df["origin_group"].isin([1, 2])].copy(), 2),
    ]
    tasks = [
        "llm_openai_beta",
        "llm_writing_raw",
        "llm_writing_orth",
        "llm_coding_math_raw",
        "llm_coding_math_orth",
        "face_to_face_social",
    ]

    for cname_grp, sub, treat_val in configs:
        sub["treat"] = (sub["origin_group"] == treat_val).astype(float)
        for tcol in tasks:
            sub["treat_t"] = sub["treat"] * sub[tcol]
            fit, se_type = fit_twoway_logged(
                f"ln_wage ~ treat_t + i(year, treat_t, ref={ref_yr}) + age + age2 + is_black + is_asian + is_hispanic"
                " | occsoc:year + statefip:year + educ4:year + origin_group:year + sex:year",
                data=sub,
                weights="cell_wt",
                c1="occsoc",
                c2="statefip",
                context=f"ACS_ES_{cname_grp}_{tcol}",
            )
            coefs = fit.coef()
            ses = fit.se()
            pvals = fit.pvalue()
            vcov = fit._vcov
            idx_list = list(coefs.index)

            yr_indices = []
            yr_coefs = []
            for yr in es_years:
                if yr == ref_yr:
                    es_rows.append({
                        "comparison": cname_grp,
                        "task": tcol,
                        "year": int(yr),
                        "coef": 0.0,
                        "se": 0.0,
                        "pval": np.nan,
                        "se_type": se_type,
                        "n_cells": int(fit._N),
                        "n_workers": int(sub["cell_n"].sum()),
                    })
                else:
                    matched = [c for c in idx_list if f"::{yr}:" in c or f"[{yr}]" in c or f"T.{yr}]" in c]
                    k = matched[0]
                    es_rows.append({
                        "comparison": cname_grp,
                        "task": tcol,
                        "year": int(yr),
                        "coef": float(coefs[k]),
                        "se": float(ses[k]),
                        "pval": float(pvals[k]),
                        "se_type": se_type,
                        "n_cells": int(fit._N),
                        "n_workers": int(sub["cell_n"].sum()),
                    })
                    yr_indices.append(idx_list.index(k))
                    yr_coefs.append(float(coefs[k]))

            # Compute LP-based HonestDiD bounds on the average post-treatment effect (0.5 * b_2023 + 0.5 * b_2024)
            # Passing actual calendar years [2015, 2018, 2019, 2021], ref=2022, post=[2023, 2024] for exact per-year step scaling!
            beta_es = np.array(yr_coefs)
            v_es = vcov[np.ix_(yr_indices, yr_indices)]
            n_pre = len(pre_years)
            n_post = len(post_years)
            l_avg = np.ones(n_post) / n_post

            hd = compute_honestdid_rm_bounds(
                beta_pre=beta_es[:n_pre],
                beta_post=beta_es[n_pre:],
                vcov_full=v_es,
                l_vec=l_avg,
                pre_times=[float(y) for y in pre_years],
                ref_time=float(ref_yr),
                post_times=[float(y) for y in post_years],
                mbar_grid=[0.0, 0.5, 1.0, 1.5, 2.0],
            )
            hd.insert(0, "comparison", cname_grp)
            hd.insert(1, "task", tcol)
            hd.insert(2, "target", "Avg_2023_2024")
            hd["se_type"] = se_type
            honest_rows.append(hd)

    es_df = pd.DataFrame(es_rows)
    honest_df = pd.concat(honest_rows, ignore_index=True)
    es_df.to_csv(RES_DIR / "acs_event_studies.csv", index=False)
    honest_df.to_csv(RES_DIR / "acs_honestdid_bounds.csv", index=False)
    return es_df, honest_df


def run_task_horserace(cell_df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 3: Simultaneous Task Horse Race — LLM_Writing vs. LLM_Coding_Math vs. FaceToFace_Social.
    Includes:
      (1) Orthogonalized Writing + Coding_Math_Raw + FaceToFace_Social (exact zero corr between Writing_Orth and Coding_Raw!)
      (2) Raw Writing + Raw Coding_Math + FaceToFace_Social
      (3) OpenAI E1 (alpha) vs. FaceToFace_Social
      (4) Felten LM vs. FaceToFace_Social
    Estimated for both 2015-2024 and 2021-2024, for NonEng vs. Natives, Eng vs. Natives, and Immigrants-Only DDD.
    """
    rows = []
    specs = [
        ("OrthWriting_CodingRaw_Social", ["llm_writing_orth", "llm_coding_math_raw", "face_to_face_social"]),
        ("RawWriting_RawCoding_Social", ["llm_writing_raw", "llm_coding_math_raw", "face_to_face_social"]),
        ("OpenAI_Beta_vs_Social", ["llm_openai_beta", "face_to_face_social"]),
        ("OpenAI_E1_vs_Social", ["llm_openai_alpha", "face_to_face_social"]),
        ("Felten_LM_vs_Social", ["llm_felten_lm", "face_to_face_social"]),
    ]

    for win_name, base_df in [("2015-2024", cell_df), ("2021-2024", cell_df[cell_df["year"] >= 2021])]:
        for grp_name, sub_mask, treat_val in [
            ("NonEng_vs_Native", base_df["origin_group"].isin([0, 2]), 2),
            ("Eng_vs_Native", base_df["origin_group"].isin([0, 1]), 1),
            ("NonEng_vs_Eng_ImmiOnly", base_df["origin_group"].isin([1, 2]), 2),
        ]:
            sub = base_df[sub_mask].copy()
            sub["post"] = (sub["year"] >= 2023).astype(float)
            sub["treat"] = (sub["origin_group"] == treat_val).astype(float)

            for spec_name, tcols in specs:
                rhs_terms = []
                post_terms = []
                for tc in tcols:
                    sub[f"tr_{tc}"] = sub["treat"] * sub[tc]
                    sub[f"ptr_{tc}"] = sub["post"] * sub[f"tr_{tc}"]
                    rhs_terms.extend([f"tr_{tc}", f"ptr_{tc}"])
                    post_terms.append((tc, f"ptr_{tc}"))

                fit, se_type = fit_twoway_logged(
                    f"ln_wage ~ {' + '.join(rhs_terms)} + age + age2 + is_black + is_asian + is_hispanic"
                    " | occsoc:year + statefip:year + educ4:year + origin_group:year + sex:year",
                    data=sub,
                    weights="cell_wt",
                    c1="occsoc",
                    c2="statefip",
                    context=f"ACS_HR_{win_name}_{grp_name}_{spec_name}",
                )
                for tc, pname in post_terms:
                    se_val = float(fit.se()[pname])
                    rows.append({
                        "window": win_name,
                        "comparison": grp_name,
                        "spec": spec_name,
                        "task": tc,
                        "coef": float(fit.coef()[pname]),
                        "se": se_val,
                        "pval": float(fit.pvalue()[pname]),
                        "mde80": MDE_80_MULT * se_val,
                        "se_type": se_type,
                        "n_cells": int(fit._N),
                        "n_workers": int(sub["cell_n"].sum()),
                        "r2": float(fit._r2),
                    })
    res = pd.DataFrame(rows)
    res.to_csv(RES_DIR / "acs_task_horserace.csv", index=False)
    return res


def run_heterogeneity_suite(cell_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mechanism & Heterogeneity Tests (Non-English-Origin Immigrants vs. U.S. Natives):
      - English proficiency (Fluent vs. Non-Fluent SPEAKENG)
      - Education & Field (College STEM, College Non-STEM, Non-College)
      - Gender (Men, Women)
      - Age / Experience (18-34, 35-49, 50-64)
      - Years in U.S. (Recent <10 yrs, Established >=10 yrs)
    Evaluated on llm_openai_beta, llm_writing_orth, and llm_coding_math_orth for 2015-2024 and 2021-2024.
    """
    base = cell_df[cell_df["origin_group"].isin([0, 2])].copy()
    base["post"] = (base["year"] >= 2023).astype(float)
    base["treat"] = (base["origin_group"] == 2).astype(float)

    subgroups = [
        ("English Fluency", "Fluent English (SPEAKENG 3-4)", lambda d: (d["origin_group"] == 0) | (d["non_fluent"] == 0)),
        ("English Fluency", "Non-Fluent English (SPEAKENG 1,5,6)", lambda d: (d["origin_group"] == 0) | (d["non_fluent"] == 1)),
        ("Degree & STEM", "College STEM (DEGFIELD STEM)", lambda d: d["stem_group"] == 2),
        ("Degree & STEM", "College Non-STEM", lambda d: d["stem_group"] == 1),
        ("Degree & STEM", "Non-College (< BA)", lambda d: d["stem_group"] == 0),
        ("Gender", "Men", lambda d: d["sex"] == 1),
        ("Gender", "Women", lambda d: d["sex"] == 2),
        ("Age / Experience", "Young (18-34)", lambda d: d["age_group"] == 1),
        ("Age / Experience", "Prime-Age (35-49)", lambda d: d["age_group"] == 2),
        ("Age / Experience", "Senior (50-64)", lambda d: d["age_group"] == 3),
        ("U.S. Tenure", "Recent Arrivals (<10 yrs)", lambda d: (d["origin_group"] == 0) | (d["recent_arr"] == 1)),
        ("U.S. Tenure", "Established (>=10 yrs)", lambda d: (d["origin_group"] == 0) | (d["recent_arr"] == 0)),
    ]

    rows = []
    for win_name, wdf in [("2015-2024", base), ("2021-2024", base[base["year"] >= 2021])]:
        for dim, slabel, mask_fn in subgroups:
            sub = wdf[mask_fn(wdf)].copy()
            for tcol in ["llm_openai_beta", "llm_writing_raw", "llm_writing_orth", "llm_coding_math_orth"]:
                sub["tr_t"] = sub["treat"] * sub[tcol]
                sub["ptr_t"] = sub["post"] * sub["tr_t"]
                fe_str = "occsoc:year + statefip:year + educ4:year + origin_group:year"
                if dim != "Gender":
                    fe_str += " + sex:year"
                fit, se_type = fit_twoway_logged(
                    f"ln_wage ~ tr_t + ptr_t + age + age2 + is_black + is_asian + is_hispanic | {fe_str}",
                    data=sub,
                    weights="cell_wt",
                    c1="occsoc",
                    c2="statefip",
                    context=f"ACS_het_{win_name}_{slabel}_{tcol}",
                )
                se_val = float(fit.se()["ptr_t"])
                rows.append({
                    "window": win_name,
                    "dimension": dim,
                    "subgroup": slabel,
                    "task": tcol,
                    "coef": float(fit.coef()["ptr_t"]),
                    "se": se_val,
                    "pval": float(fit.pvalue()["ptr_t"]),
                    "mde80": MDE_80_MULT * se_val,
                    "se_type": se_type,
                    "n_cells": int(fit._N),
                    "n_workers": int(sub["cell_n"].sum()),
                    "r2": float(fit._r2),
                })
    res = pd.DataFrame(rows)
    res.to_csv(RES_DIR / "acs_heterogeneity.csv", index=False)
    return res


def run_extensive_and_sorting(cell_df: pd.DataFrame, emp_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    1. Occupational sorting event study (2015-2024, ref=2022) into LLM_OpenAI_Beta,
       LLM_Writing_Orth, LLM_Coding_Math_Orth, and FaceToFace_Social for NonEng & Eng immigrants.
    2. Extensive-margin employment & FTFY rate DiD (2015-2024 & 2021-2024).
    """
    es_years = sorted(cell_df["year"].unique().tolist())
    ref_yr = 2022

    sort_rows = []
    for grp_name, grp_val in [("NonEng_vs_Native", 2), ("Eng_vs_Native", 1)]:
        sub = cell_df[cell_df["origin_group"].isin([0, grp_val])].copy()
        sub["treat"] = (sub["origin_group"] == grp_val).astype(float)
        for out_col in ["llm_openai_beta", "llm_writing_orth", "llm_coding_math_orth", "face_to_face_social"]:
            fit = pf.feols(
                f"{out_col} ~ treat + i(year, treat, ref={ref_yr}) + age + age2 + is_black + is_asian + is_hispanic"
                " | statefip:year + educ4:year + sex:year",
                data=sub,
                weights="cell_wt",
                vcov={"CRV1": "statefip"},
            )
            coefs, ses, pvals = fit.coef(), fit.se(), fit.pvalue()
            for yr in es_years:
                if yr == ref_yr:
                    sort_rows.append({"comparison": grp_name, "outcome": out_col, "year": int(yr), "coef": 0.0, "se": 0.0, "pval": np.nan})
                else:
                    k = [c for c in coefs.index if f"::{yr}:" in c or f"[{yr}]" in c or f"T.{yr}]" in c][0]
                    sort_rows.append({
                        "comparison": grp_name,
                        "outcome": out_col,
                        "year": int(yr),
                        "coef": float(coefs[k]),
                        "se": float(ses[k]),
                        "pval": float(pvals[k]),
                    })
    sort_df = pd.DataFrame(sort_rows)
    sort_df.to_csv(RES_DIR / "acs_occupational_sorting.csv", index=False)

    # Extensive margin: Employment rate & FTFY rate (NonEng vs. Natives and Eng vs. Natives)
    ext_rows = []
    for win_name, edf in [("2015-2024", emp_df), ("2021-2024", emp_df[emp_df["year"] >= 2021])]:
        for grp_name, grp_val in [("NonEng_vs_Native", 2), ("Eng_vs_Native", 1)]:
            sub = edf[edf["origin_group"].isin([0, grp_val])].copy()
            sub["treat"] = (sub["origin_group"] == grp_val).astype(float)
            sub["post"] = (sub["year"] >= 2023).astype(float)
            sub["post_treat"] = sub["post"] * sub["treat"]
            for outcome in ["emp_rate", "ftfy_rate"]:
                fit = pf.feols(
                    f"{outcome} ~ treat + post_treat + age + age2 | statefip:year + educ4:year + sex:year",
                    data=sub,
                    weights="cell_wt",
                    vcov={"CRV1": "statefip"},
                )
                ext_rows.append({
                    "window": win_name,
                    "comparison": grp_name,
                    "outcome": outcome,
                    "coef": float(fit.coef()["post_treat"]),
                    "se": float(fit.se()["post_treat"]),
                    "pval": float(fit.pvalue()["post_treat"]),
                    "n_cells": int(fit._N),
                    "n_obs": int(sub["cell_n"].sum()),
                })
    ext_df = pd.DataFrame(ext_rows)
    ext_df.to_csv(RES_DIR / "acs_extensive_margin.csv", index=False)
    return sort_df, ext_df


def main() -> None:
    cell_df, emp_df, counts = load_acs_cells()
    print("\n1. Running Origin-Placebo Triple-Difference Suite...")
    placebo_df = run_origin_placebo_suite(cell_df)

    print("\n2. Running Dynamic Event Studies & Rambachan-Roth HonestDiD...")
    es_df, honest_df = run_event_studies_and_honestdid(cell_df)

    print("\n3. Running Simultaneous Task Horse Race...")
    hr_df = run_task_horserace(cell_df)

    print("\n4. Running Heterogeneity & Mechanisms Suite...")
    het_df = run_heterogeneity_suite(cell_df)

    print("\n5. Running Extensive Margin & Occupational Sorting...")
    sort_df, ext_df = run_extensive_and_sorting(cell_df, emp_df)

    summary = {
        "sample_counts": counts,
        "origin_placebo_2015_2024": placebo_df[placebo_df["window"] == "2015-2024"][
            ["task", "b_noneng", "se_noneng", "p_noneng", "b_eng", "se_eng", "p_eng", "ddd_diff", "ddd_se", "ddd_p"]
        ].to_dict(orient="records"),
        "origin_placebo_2021_2024": placebo_df[placebo_df["window"] == "2021-2024"][
            ["task", "b_noneng", "se_noneng", "p_noneng", "b_eng", "se_eng", "p_eng", "ddd_diff", "ddd_se", "ddd_p"]
        ].to_dict(orient="records"),
    }
    (RES_DIR / "acs_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n================ TABLE 1: ORIGIN-PLACEBO (2015-2024) ================")
    print(placebo_df[placebo_df["window"] == "2015-2024"][
        ["task", "b_noneng", "se_noneng", "p_noneng", "b_eng", "se_eng", "p_eng", "ddd_diff", "ddd_se", "ddd_p"]
    ].to_string(index=False))

    print("\n================ TABLE 1B: ORIGIN-PLACEBO (2021-2024) ================")
    print(placebo_df[placebo_df["window"] == "2021-2024"][
        ["task", "b_noneng", "se_noneng", "p_noneng", "b_eng", "se_eng", "p_eng", "ddd_diff", "ddd_se", "ddd_p"]
    ].to_string(index=False))

    print("\n================ TASK HORSE RACE (2021-2024, NonEng vs Native) ================")
    print(hr_df[(hr_df["window"] == "2021-2024") & (hr_df["comparison"] == "NonEng_vs_Native")][
        ["spec", "task", "coef", "se", "pval"]
    ].to_string(index=False))

    print("\n================ HONESTDID BOUNDS (llm_openai_beta & llm_writing_orth) ================")
    print(honest_df[(honest_df["comparison"] == "NonEng_vs_Native") & (honest_df["task"].isin(["llm_openai_beta", "llm_writing_orth"]))][
        ["task", "mbar", "theta_hat", "se_theta", "flci_lo", "flci_hi", "rob_ci_lo", "rob_ci_hi"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
