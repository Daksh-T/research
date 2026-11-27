#!/usr/bin/env python3
"""
Paper 1 -- Exact Individual-Level ACS Worker Replication (N = 6.55M / 3.84M)
=============================================================================
Replicates every headline ACS occupational-exposure specification directly on
individual worker microdata rows from data/ipums/raw/acs_2007_2024/usa_00012.csv.gz
(without cell aggregation) using pyfixest.feols with PERWT weights and two-way
CRV1 clustering (OCCSOC + STATEFIP).

Because exact age, age^2, and race/ethnicity indicators vary within broad
age-band cells, cell-mean aggregation is not lossless. This script produces the
authoritative individual-worker estimates and saves a side-by-side comparison
against the cell-grouped estimates in:
  papers/paper1_genai_immigrants/results/acs_individual_vs_cell_comparison.csv
  papers/paper1_genai_immigrants/results/acs_individual_summary.json
"""

from __future__ import annotations

import gc
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[3]
ACS_FILE = ROOT / "data" / "ipums" / "raw" / "acs_2007_2024" / "usa_00012.csv.gz"
RES_DIR = ROOT / "papers" / "paper1_genai_immigrants" / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "papers" / "paper1_genai_immigrants" / "scripts"))
from p1_acs_analysis import MDE_80_MULT, fit_twoway_logged


HEADLINE_TASKS: List[Tuple[str, str]] = [
    ("aei_2026_observed", "AEI 2026 Observed Exposure (Mar 2026)"),
    ("aei_2026_writing_raw", "AEI 2026 Writing (Raw)"),
    ("aei_2026_writing_orth", "AEI 2026 Writing (Orthogonalized)"),
    ("aei_2026_coding_raw", "AEI 2026 Coding & Math (Raw)"),
    ("aei_2026_coding_orth", "AEI 2026 Coding & Math (Orthogonalized)"),
    ("aei_2026_robot", "AEI 2026 Robot Exposure (Manual/Physical)"),
    ("written_eng_only", "O*NET 31.0 Written-English Intensity"),
    ("llm_openai_beta", "OpenAI Beta (Eloundou et al. 2024)"),
    ("llm_writing_orth", "OpenAI Beta Writing (Orthogonalized)"),
    ("llm_coding_math_raw", "OpenAI Beta Coding & Math (Raw)"),
    ("llm_felten_lm", "Felten Language-Modeling AIOE (2023)"),
    ("face_to_face_social", "Face-to-Face Social Skills (O*NET 31.0)"),
]


def load_individual_acs() -> pd.DataFrame:
    """Load exact individual-level FTFY ACS workers (2015-2024) via DuckDB."""
    t0 = time.time()
    con = duckdb.connect()

    t6_legacy = pd.read_parquet(RES_DIR / "tasks_occsoc.parquet")
    t4_legacy = pd.read_parquet(RES_DIR / "tasks_occsoc4.parquet")
    t6_aei = pd.read_parquet(RES_DIR / "aei_2026_tasks_occsoc.parquet")
    aei_cols = [
        "aei_2026_observed",
        "aei_2026_writing_raw",
        "aei_2026_writing_orth",
        "aei_2026_coding_raw",
        "aei_2026_coding_orth",
        "aei_2026_robot",
    ]
    t4_aei = t6_aei.assign(occsoc4=t6_aei["occsoc"].str.slice(0, 4)).groupby("occsoc4", as_index=False)[aei_cols].mean()

    con.register("t6l", t6_legacy)
    con.register("t4l", t4_legacy)
    con.register("t6a", t6_aei[["occsoc"] + aei_cols])
    con.register("t4a", t4_aei)

    eng_origin_sql = """
        (BPL = 150 AND BPLD < 15080)
        OR BPL IN (410, 411, 412, 413, 414, 700)
        OR BPLD IN (21010, 26030, 30040)
        OR (BPLD BETWEEN 26040 AND 26069)
    """

    print("Loading individual-level 2015-2024 ACS FTFY workers via DuckDB...")
    df = con.execute(f"""
        WITH workers AS (
            SELECT
                CAST(YEAR AS SMALLINT) AS year,
                CAST(STATEFIP AS SMALLINT) AS statefip,
                CAST(CASE
                    WHEN EDUCD < 62 THEN 1
                    WHEN EDUCD BETWEEN 62 AND 64 THEN 2
                    WHEN EDUCD BETWEEN 65 AND 100 THEN 3
                    ELSE 4
                END AS TINYINT) AS educ4,
                CAST(SEX AS TINYINT) AS sex,
                CAST(CASE
                    WHEN BPL < 100 THEN 0
                    WHEN ({eng_origin_sql}) THEN 1
                    ELSE 2
                END AS TINYINT) AS origin_group,
                REGEXP_REPLACE(UPPER(TRIM(CAST(OCCSOC AS VARCHAR))), '[-\\s]', '', 'g') AS occsoc,
                CAST(AGE AS DOUBLE) AS age,
                CAST(AGE * AGE AS DOUBLE) AS age2,
                CASE WHEN RACE = 2 THEN 1.0 ELSE 0.0 END AS is_black,
                CASE WHEN RACE IN (4, 5, 6) THEN 1.0 ELSE 0.0 END AS is_asian,
                CASE WHEN HISPAN BETWEEN 1 AND 4 THEN 1.0 ELSE 0.0 END AS is_hispanic,
                CAST(PERWT AS DOUBLE) AS perwt,
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
            w.year, w.statefip, w.educ4, w.sex, w.origin_group, w.occsoc,
            w.age, w.age2, w.is_black, w.is_asian, w.is_hispanic, w.perwt, w.ln_wage,
            COALESCE(t6a.aei_2026_observed, t4a.aei_2026_observed) AS aei_2026_observed,
            COALESCE(t6a.aei_2026_writing_raw, t4a.aei_2026_writing_raw) AS aei_2026_writing_raw,
            COALESCE(t6a.aei_2026_writing_orth, t4a.aei_2026_writing_orth) AS aei_2026_writing_orth,
            COALESCE(t6a.aei_2026_coding_raw, t4a.aei_2026_coding_raw) AS aei_2026_coding_raw,
            COALESCE(t6a.aei_2026_coding_orth, t4a.aei_2026_coding_orth) AS aei_2026_coding_orth,
            COALESCE(t6a.aei_2026_robot, t4a.aei_2026_robot) AS aei_2026_robot,
            COALESCE(t6l.written_eng_only, t4l.written_eng_only) AS written_eng_only,
            COALESCE(t6l.llm_openai_beta, t4l.llm_openai_beta) AS llm_openai_beta,
            COALESCE(t6l.llm_writing_orth, t4l.llm_writing_orth) AS llm_writing_orth,
            COALESCE(t6l.llm_coding_math_raw, t4l.llm_coding_math_raw) AS llm_coding_math_raw,
            COALESCE(t6l.llm_felten_lm, t4l.llm_felten_lm) AS llm_felten_lm,
            COALESCE(t6l.face_to_face_social, t4l.face_to_face_social) AS face_to_face_social
        FROM workers w
        LEFT JOIN t6l ON w.occsoc = t6l.occsoc
        LEFT JOIN t4l ON SUBSTR(w.occsoc, 1, 4) = t4l.occsoc4
        LEFT JOIN t6a ON w.occsoc = t6a.occsoc
        LEFT JOIN t4a ON SUBSTR(w.occsoc, 1, 4) = t4a.occsoc4
        WHERE COALESCE(t6l.llm_openai_beta, t4l.llm_openai_beta) IS NOT NULL
    """).df()

    # Encode occsoc as integer code for fast FE absorption & low memory
    df["occsoc_id"] = df["occsoc"].astype("category").cat.codes.astype(np.int32)
    df.drop(columns=["occsoc"], inplace=True)

    print(f"Loaded {len(df):,} individual ACS FTFY workers in {time.time() - t0:.1f}s "
          f"(2021-2024: {(df['year'] >= 2021).sum():,})")
    return df


def main() -> None:
    df = load_individual_acs()

    # Load existing cell-grouped results for side-by-side comparison
    cell_legacy = pd.read_csv(RES_DIR / "acs_origin_placebo.csv")
    cell_aei = pd.read_csv(RES_DIR / "aei_2026_acs_origin_placebo.csv")

    # Build lookup (window, task) -> cell-grouped stats
    cell_lookup: Dict[Tuple[str, str], Dict[str, float]] = {}
    for _, r in cell_legacy.iterrows():
        cell_lookup[(str(r["window"]), str(r["task"]))] = {
            "cell_b_noneng": float(r["b_noneng"]),
            "cell_se_noneng": float(r["se_noneng"]),
            "cell_p_noneng": float(r["p_noneng"]),
            "cell_b_eng": float(r["b_eng"]),
            "cell_se_eng": float(r["se_eng"]),
            "cell_p_eng": float(r["p_eng"]),
            "cell_ddd_diff": float(r["ddd_diff"]),
            "cell_ddd_se": float(r["ddd_se"]),
            "cell_ddd_p": float(r["ddd_p"]),
            "n_cells": int(r["n_cells"]),
        }
    win_map = {"ACS_2015_2024": "2015-2024", "ACS_2021_2024": "2021-2024"}
    for _, r in cell_aei.iterrows():
        w = win_map.get(str(r["window"]), str(r["window"]))
        key = (w, str(r["task"]))
        if key not in cell_lookup:
            cell_lookup[key] = {
                "cell_b_noneng": float(r["noneng_coef"]),
                "cell_se_noneng": float(r["noneng_se"]),
                "cell_p_noneng": float(r["noneng_p"]),
                "cell_b_eng": float(r["eng_coef"]),
                "cell_se_eng": float(r["eng_se"]),
                "cell_p_eng": float(r["eng_p"]),
                "cell_ddd_diff": float(r["ddd_coef"]),
                "cell_ddd_se": float(r["ddd_se"]),
                "cell_ddd_p": float(r["ddd_p"]),
                "n_cells": int(r["n_cells"]),
            }

    fml = (
        "ln_wage ~ eng_t + noneng_t + post_eng_t + post_noneng_t"
        " + age + age2 + is_black + is_asian + is_hispanic"
        " | occsoc_id:year + statefip:year + educ4:year + origin_group:year + sex:year"
    )

    rows = []
    for win_name, yr_min in [("2021-2024", 2021), ("2015-2024", 2015)]:
        sub = df if yr_min == 2015 else df[df["year"] >= 2021].copy()
        post = (sub["year"] >= 2023).to_numpy(dtype=np.float64)
        is_eng = (sub["origin_group"] == 1).to_numpy(dtype=np.float64)
        is_noneng = (sub["origin_group"] == 2).to_numpy(dtype=np.float64)

        print(f"\n=== Individual-Level ACS Worker Regressions: {win_name} (N = {len(sub):,}) ===")
        for tcol, tlab in HEADLINE_TASKS:
            t1 = time.time()
            tv = sub[tcol].to_numpy(dtype=np.float64)
            sub["eng_t"] = is_eng * tv
            sub["noneng_t"] = is_noneng * tv
            sub["post_eng_t"] = post * sub["eng_t"].to_numpy()
            sub["post_noneng_t"] = post * sub["noneng_t"].to_numpy()

            fit, se_type = fit_twoway_logged(
                fml,
                data=sub,
                weights="perwt",
                c1="occsoc_id",
                c2="statefip",
                context=f"IND_ACS_{win_name}_{tcol}",
            )

            coefs = fit.coef()
            ses = fit.se()
            pvals = fit.pvalue()
            vcov = fit._vcov
            idx = list(coefs.index)
            i_ne = idx.index("post_noneng_t")
            i_e = idx.index("post_eng_t")

            b_ne = float(coefs["post_noneng_t"])
            se_ne = float(ses["post_noneng_t"])
            p_ne = float(pvals["post_noneng_t"])
            b_e = float(coefs["post_eng_t"])
            se_e = float(ses["post_eng_t"])
            p_e = float(pvals["post_eng_t"])

            ddd = b_ne - b_e
            ddd_var = float(vcov[i_ne, i_ne] + vcov[i_e, i_e] - 2.0 * vcov[i_ne, i_e])
            ddd_se = float(np.sqrt(max(ddd_var, 1e-16)))
            ddd_p = float(2.0 * (1.0 - stats.norm.cdf(abs(ddd / ddd_se))))

            c_info = cell_lookup.get((win_name, tcol), {})
            rec = {
                "window": win_name,
                "task": tcol,
                "task_label": tlab,
                "ind_b_noneng": b_ne,
                "ind_se_noneng": se_ne,
                "ind_p_noneng": p_ne,
                "ind_mde80_noneng": MDE_80_MULT * se_ne,
                "ind_b_eng": b_e,
                "ind_se_eng": se_e,
                "ind_p_eng": p_e,
                "ind_ddd_diff": ddd,
                "ind_ddd_se": ddd_se,
                "ind_ddd_p": ddd_p,
                "ind_mde80_ddd": MDE_80_MULT * ddd_se,
                "se_type": "twoway_occsoc_statefip" if se_type == "twoway_occsoc_id_statefip" else se_type,
                "n_workers": int(fit._N),
                "r2": float(fit._r2),
                **c_info,
            }
            if "cell_b_noneng" in c_info:
                rec["diff_b_noneng"] = b_ne - c_info["cell_b_noneng"]
                rec["diff_se_noneng"] = se_ne - c_info["cell_se_noneng"]
                rec["diff_b_eng"] = b_e - c_info["cell_b_eng"]
                rec["diff_ddd"] = ddd - c_info["cell_ddd_diff"]
            rows.append(rec)

            print(
                f"  [{win_name}] {tcol:22s}: Ind NonEng={b_ne:+.4f} ({se_ne:.4f}, p={p_ne:.3f}) "
                f"vs Cell={c_info.get('cell_b_noneng', np.nan):+.4f} ({c_info.get('cell_se_noneng', np.nan):.4f}) | "
                f"Ind Eng={b_e:+.4f} ({se_e:.4f}) | DDD={ddd:+.4f} ({ddd_se:.4f}) [{time.time()-t1:.1f}s]"
            )
            del fit
            gc.collect()

    out_df = pd.DataFrame(rows)
    out_path = RES_DIR / "acs_individual_vs_cell_comparison.csv"
    out_df.to_csv(out_path, index=False)

    summary = {
        "n_workers_2021_2024": int((df["year"] >= 2021).sum()),
        "n_workers_2015_2024": int(len(df)),
        "max_abs_diff_b_noneng": float(out_df["diff_b_noneng"].abs().max()),
        "max_abs_diff_se_noneng": float(out_df["diff_se_noneng"].abs().max()),
        "max_abs_diff_b_eng": float(out_df["diff_b_eng"].abs().max()),
        "max_abs_diff_ddd": float(out_df["diff_ddd"].abs().max()),
        "all_twoway_clustered": bool((out_df["se_type"] == "twoway_occsoc_statefip").all()),
    }
    (RES_DIR / "acs_individual_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\nSaved:", out_path)
    print("Summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
