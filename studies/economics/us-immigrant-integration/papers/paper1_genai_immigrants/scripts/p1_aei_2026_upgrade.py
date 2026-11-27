#!/usr/bin/env python3
"""
Paper 1 — 2026 Anthropic Economic Index Frontier Upgrade
=========================================================
Upgrades Paper 1's empirical core to 2026 Observed AI Exposure (Massenkoff & McCrory,
March 2026, Anthropic Economic Index), while explicitly benchmarking against—and
demoting—pre-2025 theoretical exposure indices (Eloundou et al. 2024; Felten et al. 2023).

Constructs and maps cleanly to BOTH 6-digit OCCSOC (ACS 2015-2024, 6.55M FTFY workers)
and 2018 Census OCC_2018 (CPS ORG 2020 Q1-2026 Q3, 634,326 FT earners):
  1. AEI_2026_Observed        : Observed AI Exposure (Massenkoff & McCrory, March 2026;
                                756 6-digit SOCs, job_exposure.csv).
  2. AEI_2026_Writing (Raw & Orth.) and AEI_2026_Coding_Math (Raw & Orth.):
                                Observed 2026 exposure interacted with O*NET 31.0 Written
                                English vs. Coding/Math intensities, orthogonalized so
                                corr(AEI_2026_Writing_Orth, AEI_2026_Coding_Raw) = 0.000.
  3. AEI_2026_Task_Writing & AEI_2026_Task_Coding:
                                Direct task-level observed 2026 penetration aggregated from
                                17,998 O*NET tasks (task_penetration.csv linked to
                                onet_task_statements.csv) classified into writing/drafting
                                vs. coding/programming/quantitative tasks.
  4. AEI_2026_Robot           : 2026 Robot Exposure Index (robot_exposure_occupations.csv,
                                923 O*NET-SOCs) — 2026 physical-automation contrast to Song (2017).
  5. AEI_2025_Automation vs. AEI_2025_Augmentation:
                                Occupation-level automation (directive + feedback_loop) vs.
                                augmentation (task_iteration + validation + learning)
                                exposure from AEI v2 (March 2025, 3,364 tasks) — flagged
                                with explicit caveat regarding 2025->2026 capability drift.

Outputs (all new files, preserving legacy CSVs untouched) in papers/paper1_genai_immigrants/results/:
  - aei_2026_tasks_occsoc.parquet
  - aei_2026_tasks_occ2018.parquet
  - aei_2026_vs_legacy_diagnostic.csv
  - aei_2026_acs_origin_placebo.csv
  - aei_2026_acs_horserace.csv
  - aei_2026_acs_event_studies.csv
  - aei_2026_acs_honestdid.csv
  - aei_2026_cps_did_persistence.csv
  - aei_2026_cps_event_studies.csv
  - aei_2026_cps_honestdid.csv
  - aei_2026_summary.json
"""

from __future__ import annotations

import json
import re
import sys
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
AEI_DIR = ROOT / "data" / "aux" / "anthropic_openai_2026"
ACS_FILE = ROOT / "data" / "ipums" / "raw" / "acs_2007_2024" / "usa_00012.csv.gz"
CPS_FILE = ROOT / "data" / "ipums" / "raw" / "cps_org_2015_2026" / "cps_org_clean.parquet"
RES_DIR = ROOT / "papers" / "paper1_genai_immigrants" / "results"

sys.path.insert(0, str(ROOT / "papers" / "paper1_genai_immigrants" / "scripts"))
from p1_acs_analysis import MDE_80_MULT, STEM_DEGFIELDS, compute_honestdid_rm_bounds, fit_twoway_logged
from p1_cps_analysis import ENG_PENATVTY, yq_to_seq


def zscore(s: pd.Series) -> pd.Series:
    std = float(s.std(ddof=0))
    if std == 0 or not np.isfinite(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def residualize(y: pd.Series, x: pd.Series) -> pd.Series:
    """OLS residual of y on [1, x], standardized to mean 0, SD 1."""
    mask = y.notna() & x.notna()
    out = pd.Series(np.nan, index=y.index, dtype=float)
    X = np.column_stack([np.ones(mask.sum()), x[mask].to_numpy(dtype=float)])
    yv = y[mask].to_numpy(dtype=float)
    beta = np.linalg.lstsq(X, yv, rcond=None)[0]
    resid = yv - X @ beta
    out.loc[mask] = (resid - resid.mean()) / resid.std(ddof=0)
    return out


# Regex patterns for direct O*NET task-statement classification (Writing vs. Coding/Math)
WRITING_PAT = re.compile(
    r"\b(writ|draft|edit|proofread|translat|interpret|correspond|memorand|report|manuscript|"
    r"article|document|summariz|compose|copyedit|transcript|caption|press release|newsletter|"
    r"speech|brochure|manual|proposal|narrative|editorial|briefs|letters|emails)\w*\b",
    re.IGNORECASE,
)
CODING_PAT = re.compile(
    r"\b(program|code|coding|software|algorithm|database|sql|script|debug|compile|comput|"
    r"mathemat|statist|econometr|model|simulat|data analysis|analyze data|spreadsheet|"
    r"equation|quantitative|calculat|neural|machine learning|network|system architecture|"
    r"cad|blueprint|engineering|technical drawing|cyber|server|api)\w*\b",
    re.IGNORECASE,
)


def build_aei_2026_tasks() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    print("=== Step 1: Constructing 2026 Anthropic Economic Index Occupation Exposures ===")
    t6 = pd.read_parquet(RES_DIR / "tasks_occsoc.parquet").copy()
    t18_base = pd.read_parquet(RES_DIR / "tasks_occ2018.parquet").copy()

    # 1. Massenkoff & McCrory (March 2026) Job Observed Exposure (756 6-digit SOCs)
    je = pd.read_csv(AEI_DIR / "aei_2026_job_exposure.csv")
    je["occsoc"] = je["occ_code"].str.replace("-", "", regex=False).str.strip()
    je = je.groupby("occsoc", as_index=False)["observed_exposure"].mean()

    # 2. Task Penetration (17,998 tasks) linked to O*NET Task Statements
    tp = pd.read_csv(AEI_DIR / "aei_2026_task_penetration.csv")
    ots = pd.read_csv(AEI_DIR / "aei_onet_task_statements.csv")
    tp["task_lc"] = tp["task"].astype(str).str.strip().str.lower()
    ots["task_lc"] = ots["Task"].astype(str).str.strip().str.lower()
    ots["occsoc"] = ots["O*NET-SOC Code"].str.slice(0, 7).str.replace("-", "", regex=False)

    m_tp = ots[["occsoc", "Task", "task_lc"]].merge(
        tp[["task_lc", "penetration"]].drop_duplicates("task_lc"),
        on="task_lc",
        how="inner",
    )
    m_tp["is_writing_task"] = m_tp["Task"].astype(str).str.contains(WRITING_PAT).astype(float)
    m_tp["is_coding_task"] = m_tp["Task"].astype(str).str.contains(CODING_PAT).astype(float)
    # Pure writing (writing & not coding) and coding task penetration shares per occupation
    m_tp["pen_writing"] = m_tp["penetration"] * m_tp["is_writing_task"]
    m_tp["pen_coding"] = m_tp["penetration"] * m_tp["is_coding_task"]

    tp_occ = m_tp.groupby("occsoc", as_index=False).agg(
        aei_2026_task_pen_mean=("penetration", "mean"),
        aei_2026_task_writing_raw=("pen_writing", "mean"),
        aei_2026_task_coding_raw=("pen_coding", "mean"),
        n_tasks_matched=("penetration", "count"),
    )

    # 3. 2026 Robot Exposure (923 8-digit O*NET-SOCs -> 6-digit SOC)
    rob = pd.read_csv(AEI_DIR / "aei_2026_robot_exposure_occupations.csv")
    rob["occsoc"] = rob["onetsoc_code"].str.slice(0, 7).str.replace("-", "", regex=False)
    rob_occ = rob.groupby("occsoc", as_index=False)["robot_exposure_index"].mean()

    # 4. 2025-03 Automation vs. Augmentation by Task (3,364 tasks -> 6-digit SOC)
    # Per Handa et al. (2025): Automation = directive + feedback_loop;
    # Augmentation = task_iteration + validation + learning.
    auto = pd.read_csv(AEI_DIR / "aei_2025_03_automation_vs_augmentation_by_task.csv")
    auto["task_lc"] = auto["task_name"].astype(str).str.strip().str.lower()
    auto["auto_raw"] = auto["directive"] + auto["feedback_loop"]
    auto["aug_raw"] = auto["task_iteration"] + auto["validation"] + auto["learning"]
    auto["unfiltered"] = auto["auto_raw"] + auto["aug_raw"]

    m_auto = ots[["occsoc", "task_lc"]].merge(
        auto[["task_lc", "auto_raw", "aug_raw", "unfiltered"]].drop_duplicates("task_lc"),
        on="task_lc",
        how="inner",
    )
    auto_occ = m_auto.groupby("occsoc", as_index=False).agg(
        auto_sum=("auto_raw", "sum"),
        aug_sum=("aug_raw", "sum"),
        unf_sum=("unfiltered", "sum"),
        n_auto_tasks=("task_lc", "count"),
    )
    auto_occ["aei_2025_auto_share"] = np.where(
        auto_occ["unf_sum"] > 0, auto_occ["auto_sum"] / auto_occ["unf_sum"], 0.0
    )
    auto_occ["aei_2025_aug_share"] = np.where(
        auto_occ["unf_sum"] > 0, auto_occ["aug_sum"] / auto_occ["unf_sum"], 0.0
    )

    # Merge onto 6-digit OCCSOC master table (786 SOCs)
    df = t6.merge(je, on="occsoc", how="left")
    df = df.merge(tp_occ, on="occsoc", how="left")
    df = df.merge(rob_occ, on="occsoc", how="left")
    df = df.merge(
        auto_occ[["occsoc", "aei_2025_auto_share", "aei_2025_aug_share", "n_auto_tasks"]],
        on="occsoc",
        how="left",
    )

    n_exact_je = int(df["observed_exposure"].notna().sum())
    # For the ~38 of 786 SOCs absent from job_exposure.csv, impute from 4-digit SOC mean of observed_exposure
    # (or task_pen_mean fallback) ONLY for complete ACS/CPS coverage, while recording exact match flag
    df["aei_2026_exact_match"] = df["observed_exposure"].notna().astype(int)
    df["occsoc4"] = df["occsoc"].str.slice(0, 4)
    soc4_obs = df.groupby("occsoc4")["observed_exposure"].transform("mean")
    df["aei_2026_obs_raw"] = df["observed_exposure"].fillna(soc4_obs).fillna(df["aei_2026_task_pen_mean"]).fillna(0.0)

    df["aei_2026_task_writing_raw"] = df["aei_2026_task_writing_raw"].fillna(0.0)
    df["aei_2026_task_coding_raw"] = df["aei_2026_task_coding_raw"].fillna(0.0)
    df["robot_exposure_index"] = df["robot_exposure_index"].fillna(0.0)
    df["aei_2025_auto_share"] = df["aei_2025_auto_share"].fillna(0.0)
    df["aei_2025_aug_share"] = df["aei_2025_aug_share"].fillna(0.0)

    # Construct 2026 Observed AI Writing & Coding/Math (interacted with O*NET 31.0 intensities)
    df["raw_aei26_writing"] = df["aei_2026_obs_raw"] * df["written_eng_01"]
    df["raw_aei26_coding"] = df["aei_2026_obs_raw"] * df["coding_math_01"]

    # Construct 2025 Automation vs. Augmentation exposure intensities (2026 observed exposure split by 2025 shares)
    df["raw_aei25_auto_exp"] = df["aei_2026_obs_raw"] * df["aei_2025_auto_share"]
    df["raw_aei25_aug_exp"] = df["aei_2026_obs_raw"] * df["aei_2025_aug_share"]

    # Standardize all 2026 indices (mean 0, SD 1 across 6-digit SOCs)
    df["aei_2026_observed"] = zscore(df["aei_2026_obs_raw"])
    df["aei_2026_writing_raw"] = zscore(df["raw_aei26_writing"])
    df["aei_2026_coding_raw"] = zscore(df["raw_aei26_coding"])
    df["aei_2026_writing_orth"] = residualize(df["raw_aei26_writing"], df["raw_aei26_coding"])
    df["aei_2026_coding_orth"] = residualize(df["raw_aei26_coding"], df["raw_aei26_writing"])

    # Direct O*NET task-statement penetration (standardized + orthogonalized writing)
    df["aei_2026_task_writing"] = zscore(df["aei_2026_task_writing_raw"])
    df["aei_2026_task_coding"] = zscore(df["aei_2026_task_coding_raw"])
    df["aei_2026_task_writing_orth"] = residualize(
        df["aei_2026_task_writing_raw"], df["aei_2026_task_coding_raw"]
    )

    # 2026 Robot exposure & 2025 Automation vs. Augmentation
    df["aei_2026_robot"] = zscore(df["robot_exposure_index"])
    df["aei_2025_auto_exp"] = zscore(df["raw_aei25_auto_exp"])
    df["aei_2025_aug_exp"] = zscore(df["raw_aei25_aug_exp"])

    # Save 6-digit OCCSOC & 4-digit fallback
    df.to_parquet(RES_DIR / "aei_2026_tasks_occsoc.parquet", index=False)

    aei_num_cols = [
        "aei_2026_obs_raw", "aei_2026_observed",
        "aei_2026_writing_raw", "aei_2026_coding_raw",
        "aei_2026_writing_orth", "aei_2026_coding_orth",
        "aei_2026_task_writing", "aei_2026_task_coding", "aei_2026_task_writing_orth",
        "aei_2026_robot", "robot_exposure_index",
        "aei_2025_auto_exp", "aei_2025_aug_exp",
        "aei_2025_auto_share", "aei_2025_aug_share",
        "raw_aei26_writing", "raw_aei26_coding",
        "raw_aei25_auto_exp", "raw_aei25_aug_exp",
    ]
    df4 = df.groupby("occsoc4", as_index=False)[aei_num_cols].mean()

    # Map cleanly to 2018 Census OCC_2018 using tasks_occ2018.parquet's soc_raw crosswalk
    print("Mapping 2026 AEI exposures to 2018 Census OCC_2018 (564 occupations)...")
    soc_lookup = df.set_index("occsoc")[aei_num_cols]
    soc4_lookup = df4.set_index("occsoc4")[aei_num_cols]
    df["occsoc5"] = df["occsoc"].str.slice(0, 5)
    df["occsoc3"] = df["occsoc"].str.slice(0, 3)
    df["occsoc2"] = df["occsoc"].str.slice(0, 2)
    soc5_lookup = df.groupby("occsoc5")[aei_num_cols].mean()
    soc3_lookup = df.groupby("occsoc3")[aei_num_cols].mean()
    soc2_lookup = df.groupby("occsoc2")[aei_num_cols].mean()

    mapped_rows = []
    for _, row in t18_base.iterrows():
        raw_str = str(row["soc_raw"]).strip()
        # Handle multi-SOC or wildcard strings like "11-2030", "11-91XX", "29-1215, 29-1216"
        tokens = [tok.strip().replace("-", "") for tok in re.split(r"[,;/\s]+", raw_str) if tok.strip()]
        matched_vals = []
        for tok in tokens:
            clean = tok.upper()
            if len(clean) == 6 and clean in soc_lookup.index:
                matched_vals.append(soc_lookup.loc[clean])
            elif clean.endswith("0") and clean[:5] in soc5_lookup.index:
                matched_vals.append(soc5_lookup.loc[clean[:5]])
            elif clean[:5].replace("X", "") and len(clean[:5].replace("X", "")) == 5 and clean[:5] in soc5_lookup.index:
                matched_vals.append(soc5_lookup.loc[clean[:5]])
            elif clean[:4] in soc4_lookup.index:
                matched_vals.append(soc4_lookup.loc[clean[:4]])
            elif clean[:3] in soc3_lookup.index:
                matched_vals.append(soc3_lookup.loc[clean[:3]])
            elif clean[:2] in soc2_lookup.index:
                matched_vals.append(soc2_lookup.loc[clean[:2]])
        if matched_vals:
            avg_s = pd.concat(matched_vals, axis=1).mean(axis=1)
        else:
            avg_s = pd.Series(0.0, index=aei_num_cols)
        rec = {"occ2018": int(row["occ2018"])}
        for c in aei_num_cols:
            rec[c] = float(avg_s[c])
        mapped_rows.append(rec)

    t18_aei = pd.DataFrame(mapped_rows)
    # Re-standardize OCC_2018 z-scores and re-orthogonalize writing vs. coding on OCC_2018
    t18_aei["aei_2026_observed"] = zscore(t18_aei["aei_2026_obs_raw"])
    t18_aei["aei_2026_writing_raw"] = zscore(t18_aei["raw_aei26_writing"])
    t18_aei["aei_2026_coding_raw"] = zscore(t18_aei["raw_aei26_coding"])
    t18_aei["aei_2026_writing_orth"] = residualize(t18_aei["raw_aei26_writing"], t18_aei["raw_aei26_coding"])
    t18_aei["aei_2026_coding_orth"] = residualize(t18_aei["raw_aei26_coding"], t18_aei["raw_aei26_writing"])
    t18_aei["aei_2026_task_writing"] = zscore(t18_aei["aei_2026_task_writing"])
    t18_aei["aei_2026_task_coding"] = zscore(t18_aei["aei_2026_task_coding"])
    t18_aei["aei_2026_robot"] = zscore(t18_aei["robot_exposure_index"])
    t18_aei["aei_2025_auto_exp"] = zscore(t18_aei["raw_aei25_auto_exp"])
    t18_aei["aei_2025_aug_exp"] = zscore(t18_aei["raw_aei25_aug_exp"])

    t18_full = t18_base.merge(t18_aei, on="occ2018", how="left")
    t18_full.to_parquet(RES_DIR / "aei_2026_tasks_occ2018.parquet", index=False)

    # Diagnostic: 2026 Observed Exposure vs. Legacy Pre-2025 Benchmarks (Eloundou 2024, Felten 2023)
    exact_df = df[df["aei_2026_exact_match"] == 1].copy()
    diag_cols = [
        "aei_2026_observed",
        "llm_openai_beta",
        "llm_felten_lm",
        "aei_2026_writing_raw",
        "aei_2026_writing_orth",
        "aei_2026_coding_raw",
        "aei_2026_task_writing",
        "aei_2026_task_coding",
        "aei_2025_auto_exp",
        "aei_2025_aug_exp",
        "aei_2026_robot",
        "face_to_face_social",
    ]
    corr_mat = exact_df[diag_cols].corr().round(4)
    corr_mat.to_csv(RES_DIR / "aei_2026_vs_legacy_diagnostic.csv")

    exact_df["divergence_2026_minus_openai"] = exact_df["aei_2026_observed"] - exact_df["llm_openai_beta"]
    exact_df["rank_aei26"] = exact_df["aei_2026_obs_raw"].rank(ascending=False)
    exact_df["rank_openai"] = exact_df["openai_dv_beta"].rank(ascending=False)

    r_openai = float(exact_df["aei_2026_obs_raw"].corr(exact_df["openai_dv_beta"]))
    r_felten = float(exact_df["aei_2026_obs_raw"].corr(exact_df["felten_aioe_lm"]))
    r_robot = float(exact_df["aei_2026_obs_raw"].corr(exact_df["robot_exposure_index"]))
    rho_openai = float(exact_df["aei_2026_obs_raw"].corr(exact_df["openai_dv_beta"], method="spearman"))
    rho_felten = float(exact_df["aei_2026_obs_raw"].corr(exact_df["felten_aioe_lm"], method="spearman"))

    diag_summary = {
        "n_occsoc_total": int(len(df)),
        "n_occsoc_exact_aei26": n_exact_je,
        "n_occ2018_total": int(len(t18_full)),
        "pearson_aei26_vs_openai_beta": round(r_openai, 4),
        "spearman_aei26_vs_openai_beta": round(rho_openai, 4),
        "pearson_aei26_vs_felten_lm": round(r_felten, 4),
        "spearman_aei26_vs_felten_lm": round(rho_felten, 4),
        "pearson_aei26_vs_robot_2026": round(r_robot, 4),
        "share_zero_observed_aei26": round(float((exact_df["aei_2026_obs_raw"] == 0).mean()), 4),
        "share_zero_openai_beta": round(float((exact_df["openai_dv_beta"] == 0).mean()), 4),
        "mean_observed_aei26": round(float(exact_df["aei_2026_obs_raw"].mean()), 4),
        "mean_openai_beta": round(float(exact_df["openai_dv_beta"].mean()), 4),
        "top10_aei_2026_observed": exact_df.nlargest(10, "aei_2026_obs_raw")[
            ["occsoc", "occ_title", "aei_2026_obs_raw", "openai_dv_beta", "robot_exposure_index"]
        ].to_dict(orient="records"),
        "top10_aei_2026_writing_orth": exact_df.nlargest(10, "aei_2026_writing_orth")[
            ["occsoc", "occ_title", "aei_2026_writing_orth", "aei_2026_obs_raw"]
        ].to_dict(orient="records"),
        "top10_overpredicted_by_legacy_openai": exact_df.nsmallest(10, "divergence_2026_minus_openai")[
            ["occsoc", "occ_title", "aei_2026_obs_raw", "openai_dv_beta", "divergence_2026_minus_openai"]
        ].to_dict(orient="records"),
        "top10_underpredicted_by_legacy_openai": exact_df.nlargest(10, "divergence_2026_minus_openai")[
            ["occsoc", "occ_title", "aei_2026_obs_raw", "openai_dv_beta", "divergence_2026_minus_openai"]
        ].to_dict(orient="records"),
        "correlations": corr_mat.to_dict(),
    }

    print(f"AEI 2026 Exact Match: {n_exact_je}/{len(df)} 6-digit SOCs | "
          f"Pearson r(AEI_2026, OpenAI_beta)={r_openai:.3f} (Spearman={rho_openai:.3f}) | "
          f"r(AEI_2026, Felten_LM)={r_felten:.3f} | r(AEI_2026, Robot_2026)={r_robot:.3f}")
    print(f"Mean AEI_2026 observed={exact_df['aei_2026_obs_raw'].mean():.4f} vs. OpenAI beta={exact_df['openai_dv_beta'].mean():.4f}; "
          f"Share zero: AEI_2026={diag_summary['share_zero_observed_aei26']*100:.1f}% vs. OpenAI={diag_summary['share_zero_openai_beta']*100:.1f}%")
    return df, df4, t18_full, diag_summary


def fit_twoway_acs(fml: str, data: pd.DataFrame, weights: str = "cell_wt", context: str = ""):
    return fit_twoway_logged(fml, data=data, weights=weights, c1="occsoc", c2="statefip", context=context)


def fit_twoway_cps(fml: str, data: pd.DataFrame, weights: str = "wt", context: str = ""):
    return fit_twoway_logged(fml, data=data, weights=weights, c1="occ2018", c2="statefip", context=context)


def run_acs_2026_suite(t6_df: pd.DataFrame, t4_df: pd.DataFrame) -> Dict:
    print("\n=== Step 2: Running ACS 2015-2024 2026 AEI Empirical Suite ===")
    con = duckdb.connect()
    con.register("t6", t6_df)
    con.register("t4", t4_df)
    t4_legacy = pd.read_parquet(RES_DIR / "tasks_occsoc4.parquet")
    con.register("t4l", t4_legacy)

    stem_sql = ",".join(str(x) for x in STEM_DEGFIELDS)
    eng_origin_sql = """
        (BPL = 150 AND BPLD < 15080)
        OR BPL IN (410, 411, 412, 413, 414, 700)
        OR BPLD IN (21010, 26030, 30040)
        OR (BPLD BETWEEN 26040 AND 26069)
    """

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
            w.year, w.statefip, w.educ4, w.sex, w.origin_group, w.non_fluent,
            w.stem_group, w.age_group, w.recent_arr,
            w.occsoc_clean AS occsoc,
            SUM(w.perwt) AS cell_wt,
            COUNT(*) AS cell_n,
            SUM(w.ln_wage * w.perwt) / SUM(w.perwt) AS ln_wage,
            SUM(w.age * w.perwt) / SUM(w.perwt) AS age,
            SUM(w.age2 * w.perwt) / SUM(w.perwt) AS age2,
            SUM(w.is_black * w.perwt) / SUM(w.perwt) AS is_black,
            SUM(w.is_asian * w.perwt) / SUM(w.perwt) AS is_asian,
            SUM(w.is_hispanic * w.perwt) / SUM(w.perwt) AS is_hispanic,
            COALESCE(t6.aei_2026_observed, t4.aei_2026_observed) AS aei_2026_observed,
            COALESCE(t6.aei_2026_writing_raw, t4.aei_2026_writing_raw) AS aei_2026_writing_raw,
            COALESCE(t6.aei_2026_writing_orth, t4.aei_2026_writing_orth) AS aei_2026_writing_orth,
            COALESCE(t6.aei_2026_coding_raw, t4.aei_2026_coding_raw) AS aei_2026_coding_raw,
            COALESCE(t6.aei_2026_coding_orth, t4.aei_2026_coding_orth) AS aei_2026_coding_orth,
            COALESCE(t6.aei_2026_task_writing, t4.aei_2026_task_writing) AS aei_2026_task_writing,
            COALESCE(t6.aei_2026_task_coding, t4.aei_2026_task_coding) AS aei_2026_task_coding,
            COALESCE(t6.aei_2026_task_writing_orth, t4.aei_2026_task_writing_orth) AS aei_2026_task_writing_orth,
            COALESCE(t6.aei_2026_robot, t4.aei_2026_robot) AS aei_2026_robot,
            COALESCE(t6.aei_2025_auto_exp, t4.aei_2025_auto_exp) AS aei_2025_auto_exp,
            COALESCE(t6.aei_2025_aug_exp, t4.aei_2025_aug_exp) AS aei_2025_aug_exp,
            COALESCE(t6.written_eng_only, t4l.written_eng_only) AS written_eng_only,
            COALESCE(t6.llm_openai_beta, t4l.llm_openai_beta) AS llm_openai_beta,
            COALESCE(t6.llm_felten_lm, t4l.llm_felten_lm) AS llm_felten_lm,
            COALESCE(t6.face_to_face_social, t4l.face_to_face_social) AS face_to_face_social
        FROM workers w
        LEFT JOIN t6 ON w.occsoc_clean = t6.occsoc
        LEFT JOIN t4 ON SUBSTR(w.occsoc_clean, 1, 4) = t4.occsoc4
        LEFT JOIN t4l ON SUBSTR(w.occsoc_clean, 1, 4) = t4l.occsoc4
        WHERE COALESCE(t6.llm_openai_beta, t4l.llm_openai_beta) IS NOT NULL
        GROUP BY
            w.year, w.statefip, w.educ4, w.sex, w.origin_group, w.non_fluent,
            w.stem_group, w.age_group, w.recent_arr, w.occsoc_clean,
            COALESCE(t6.aei_2026_observed, t4.aei_2026_observed),
            COALESCE(t6.aei_2026_writing_raw, t4.aei_2026_writing_raw),
            COALESCE(t6.aei_2026_writing_orth, t4.aei_2026_writing_orth),
            COALESCE(t6.aei_2026_coding_raw, t4.aei_2026_coding_raw),
            COALESCE(t6.aei_2026_coding_orth, t4.aei_2026_coding_orth),
            COALESCE(t6.aei_2026_task_writing, t4.aei_2026_task_writing),
            COALESCE(t6.aei_2026_task_coding, t4.aei_2026_task_coding),
            COALESCE(t6.aei_2026_task_writing_orth, t4.aei_2026_task_writing_orth),
            COALESCE(t6.aei_2026_robot, t4.aei_2026_robot),
            COALESCE(t6.aei_2025_auto_exp, t4.aei_2025_auto_exp),
            COALESCE(t6.aei_2025_aug_exp, t4.aei_2025_aug_exp),
            COALESCE(t6.written_eng_only, t4l.written_eng_only),
            COALESCE(t6.llm_openai_beta, t4l.llm_openai_beta),
            COALESCE(t6.llm_felten_lm, t4l.llm_felten_lm),
            COALESCE(t6.face_to_face_social, t4l.face_to_face_social)
    """).df()

    print(f"ACS 2026 AEI sample: {int(cell_df['cell_n'].sum()):,} FTFY workers ({len(cell_df):,} cells)")

    # --- 2A. ACS Origin-Placebo Triple-Difference ---
    placebo_tasks = [
        ("aei_2026_observed", "AEI 2026 Observed Exposure (Post-Treatment)"),
        ("aei_2026_writing_raw", "AEI 2026 Writing (Raw)"),
        ("aei_2026_writing_orth", "AEI 2026 Writing (Orthogonalized)"),
        ("aei_2026_coding_raw", "AEI 2026 Coding & Math (Raw)"),
        ("aei_2026_coding_orth", "AEI 2026 Coding & Math (Orth.)"),
        ("aei_2026_task_writing_orth", "AEI 2026 Task-Level Writing (Orth.)"),
        ("aei_2026_task_coding", "AEI 2026 Task-Level Coding"),
        ("aei_2025_auto_exp", "AEI 2025 Automation Exposure (Caveat)"),
        ("aei_2025_aug_exp", "AEI 2025 Augmentation Exposure (Caveat)"),
        ("aei_2026_robot", "AEI 2026 Robot Exposure (Manual/Physical)"),
        ("face_to_face_social", "Face-to-Face Social Skills"),
        ("written_eng_only", "Pre-2022 O*NET Written-English Intensity"),
        ("llm_openai_beta", "Pre-Period: OpenAI Beta (Eloundou 2023/2024)"),
        ("llm_felten_lm", "Pre-Period: Felten LM AIOE (2023)"),
    ]

    placebo_rows = []
    for window, yr_min in [("ACS_2015_2024", 2015), ("ACS_2021_2024", 2021)]:
        sub = cell_df[cell_df["year"] >= yr_min].copy()
        sub["post"] = (sub["year"] >= 2023).astype(float)
        sub["is_eng"] = (sub["origin_group"] == 1).astype(float)
        sub["is_noneng"] = (sub["origin_group"] == 2).astype(float)

        for tcol, tlab in placebo_tasks:
            sub["eng_t"] = sub["is_eng"] * sub[tcol]
            sub["ne_t"] = sub["is_noneng"] * sub[tcol]
            sub["post_eng_t"] = sub["post"] * sub["eng_t"]
            sub["post_ne_t"] = sub["post"] * sub["ne_t"]

            fit, se_type = fit_twoway_acs(
                "ln_wage ~ eng_t + ne_t + post_eng_t + post_ne_t"
                " + age + age2 + is_black + is_asian + is_hispanic"
                " | occsoc:year + statefip:year + educ4:year + origin_group:year + sex:year",
                data=sub,
                context=f"AEI_ACS_origin_{window}_{tcol}",
            )
            coefs, ses, pvals = fit.coef(), fit.se(), fit.pvalue()
            vcov = fit._vcov
            idx = list(coefs.index)
            i_ne = idx.index("post_ne_t")
            i_eng = idx.index("post_eng_t")
            ddd_est = float(coefs["post_ne_t"] - coefs["post_eng_t"])
            ddd_var = float(vcov[i_ne, i_ne] + vcov[i_eng, i_eng] - 2.0 * vcov[i_ne, i_eng])
            ddd_se = float(np.sqrt(max(ddd_var, 1e-16)))
            ddd_p = float(2.0 * (1.0 - stats.norm.cdf(abs(ddd_est / ddd_se))))

            ne_se = float(ses["post_ne_t"])
            placebo_rows.append({
                "window": window,
                "task": tcol,
                "task_label": tlab,
                "noneng_coef": float(coefs["post_ne_t"]),
                "noneng_se": ne_se,
                "noneng_p": float(pvals["post_ne_t"]),
                "noneng_mde80": MDE_80_MULT * ne_se,
                "eng_coef": float(coefs["post_eng_t"]),
                "eng_se": float(ses["post_eng_t"]),
                "eng_p": float(pvals["post_eng_t"]),
                "ddd_coef": ddd_est,
                "ddd_se": ddd_se,
                "ddd_p": ddd_p,
                "ddd_mde80": MDE_80_MULT * ddd_se,
                "se_type": se_type,
                "n_workers": int(sub["cell_n"].sum()),
                "n_cells": int(len(sub)),
                "r2": float(fit._r2),
            })

    placebo_df = pd.DataFrame(placebo_rows)
    placebo_df.to_csv(RES_DIR / "aei_2026_acs_origin_placebo.csv", index=False)
    print("\n--- ACS 2021-2024 Origin-Placebo (2026 AEI vs. Legacy) ---")
    p21 = placebo_df[placebo_df["window"] == "ACS_2021_2024"]
    print(p21[["task", "noneng_coef", "noneng_se", "eng_coef", "ddd_coef", "ddd_p"]].to_string(index=False))

    # --- 2B. ACS Simultaneous Task Horse Race (2026 AEI) ---
    hr_specs = [
        ("HR1_AEI26_Full_15_24", 2015, [0, 2], 2,
         ["aei_2026_writing_orth", "aei_2026_coding_raw", "face_to_face_social", "aei_2026_robot"]),
        ("HR2_AEI26_PostCOVID_21_24", 2021, [0, 2], 2,
         ["aei_2026_writing_orth", "aei_2026_coding_raw", "face_to_face_social", "aei_2026_robot"]),
        ("HR3_AEI26_vs_LegacyOpenAI_21_24", 2021, [0, 2], 2,
         ["aei_2026_observed", "llm_openai_beta", "face_to_face_social", "aei_2026_robot"]),
        ("HR4_AEI25_Auto_vs_Aug_21_24", 2021, [0, 2], 2,
         ["aei_2025_auto_exp", "aei_2025_aug_exp", "face_to_face_social", "aei_2026_robot"]),
        ("HR5_AEI26_DirectTaskPen_21_24", 2021, [0, 2], 2,
         ["aei_2026_task_writing_orth", "aei_2026_task_coding", "face_to_face_social", "aei_2026_robot"]),
        ("HR6_ImmiOnly_DDD_21_24", 2021, [1, 2], 2,
         ["aei_2026_writing_orth", "aei_2026_coding_raw", "face_to_face_social", "aei_2026_robot"]),
    ]

    hr_rows = []
    for spec_name, yr_min, orig_vals, treat_val, tlist in hr_specs:
        sub = cell_df[(cell_df["year"] >= yr_min) & (cell_df["origin_group"].isin(orig_vals))].copy()
        sub["post"] = (sub["year"] >= 2023).astype(float)
        sub["treat"] = (sub["origin_group"] == treat_val).astype(float)
        terms = []
        for t in tlist:
            sub[f"tr_{t}"] = sub["treat"] * sub[t]
            sub[f"ptr_{t}"] = sub["post"] * sub[f"tr_{t}"]
            terms.extend([f"tr_{t}", f"ptr_{t}"])

        fit, se_type = fit_twoway_acs(
            f"ln_wage ~ {' + '.join(terms)} + age + age2 + is_black + is_asian + is_hispanic"
            " | occsoc:year + statefip:year + educ4:year + origin_group:year + sex:year",
            data=sub,
            context=f"AEI_ACS_HR_{spec_name}",
        )
        coefs, ses, pvals = fit.coef(), fit.se(), fit.pvalue()
        for t in tlist:
            k = f"ptr_{t}"
            se_v = float(ses[k])
            hr_rows.append({
                "spec": spec_name,
                "term": t,
                "coef": float(coefs[k]),
                "se": se_v,
                "pval": float(pvals[k]),
                "mde80": MDE_80_MULT * se_v,
                "se_type": se_type,
                "n_workers": int(sub["cell_n"].sum()),
                "n_cells": int(len(sub)),
                "r2": float(fit._r2),
            })

    hr_df = pd.DataFrame(hr_rows)
    hr_df.to_csv(RES_DIR / "aei_2026_acs_horserace.csv", index=False)
    print("\n--- ACS 2026 AEI Simultaneous Horse Races ---")
    print(hr_df[["spec", "term", "coef", "se", "pval"]].to_string(index=False))

    # --- 2C. ACS Annual Event Studies & HonestDiD for AEI_2026_Observed & AEI_2026_Writing_Orth ---
    years = sorted(cell_df["year"].unique().tolist())
    ref_yr = 2022
    pre_yrs = [y for y in years if y < ref_yr]
    post_yrs = [y for y in years if y > ref_yr]
    es_rows = []
    hd_rows = []

    for comp, orig_vals, treat_val in [
        ("NonEng_vs_Native", [0, 2], 2),
        ("Eng_vs_Native", [0, 1], 1),
    ]:
        sub = cell_df[cell_df["origin_group"].isin(orig_vals)].copy()
        sub["treat"] = (sub["origin_group"] == treat_val).astype(float)
        for tcol in ["aei_2026_observed", "aei_2026_writing_orth", "aei_2026_coding_raw", "aei_2026_robot"]:
            sub["treat_t"] = sub["treat"] * sub[tcol]
            fit, se_type = fit_twoway_acs(
                f"ln_wage ~ treat_t + i(year, treat_t, ref={ref_yr})"
                " + age + age2 + is_black + is_asian + is_hispanic"
                " | occsoc:year + statefip:year + educ4:year + origin_group:year + sex:year",
                data=sub,
                context=f"AEI_ACS_ES_{comp}_{tcol}",
            )
            coefs, ses, pvals = fit.coef(), fit.se(), fit.pvalue()
            vcov = fit._vcov
            idx_list = list(coefs.index)
            yr_to_idx, yr_to_coef = {}, {}
            for y in years:
                if y == ref_yr:
                    es_rows.append({
                        "comparison": comp, "task": tcol, "year": int(y),
                        "coef": 0.0, "se": 0.0, "pval": np.nan, "se_type": se_type,
                        "n_workers": int(sub["cell_n"].sum()),
                    })
                else:
                    k = [c for c in idx_list if f"::{y}:" in c or f"[{y}]" in c or f"T.{y}]" in c][0]
                    es_rows.append({
                        "comparison": comp, "task": tcol, "year": int(y),
                        "coef": float(coefs[k]), "se": float(ses[k]), "pval": float(pvals[k]),
                        "se_type": se_type, "n_workers": int(sub["cell_n"].sum()),
                    })
                    yr_to_idx[y] = idx_list.index(k)
                    yr_to_coef[y] = float(coefs[k])

            if comp == "NonEng_vs_Native" and tcol in ("aei_2026_observed", "aei_2026_writing_orth"):
                sel_yrs = pre_yrs + post_yrs
                sel_idx = [yr_to_idx[y] for y in sel_yrs]
                beta_sel = np.array([yr_to_coef[y] for y in sel_yrs])
                v_sel = vcov[np.ix_(sel_idx, sel_idx)]
                hd = compute_honestdid_rm_bounds(
                    beta_pre=beta_sel[:len(pre_yrs)],
                    beta_post=beta_sel[len(pre_yrs):],
                    vcov_full=v_sel,
                    l_vec=np.ones(len(post_yrs)) / len(post_yrs),
                    pre_times=[float(y) for y in pre_yrs],
                    ref_time=float(ref_yr),
                    post_times=[float(y) for y in post_yrs],
                    mbar_grid=[0.0, 0.5, 1.0, 1.5, 2.0],
                )
                hd.insert(0, "comparison", comp)
                hd.insert(1, "task", tcol)
                hd["se_type"] = se_type
                hd_rows.append(hd)

    pd.DataFrame(es_rows).to_csv(RES_DIR / "aei_2026_acs_event_studies.csv", index=False)
    hd_acs = pd.concat(hd_rows, ignore_index=True)
    hd_acs.to_csv(RES_DIR / "aei_2026_acs_honestdid.csv", index=False)

    # Also check English Fluency heterogeneity for AEI_2026_Observed & AEI_2026_Writing_Orth (ACS 2021-2024)
    fluency_rows = []
    sub21 = cell_df[cell_df["year"] >= 2021].copy()
    sub21["post"] = (sub21["year"] >= 2023).astype(float)
    for grp_name, cond in [
        ("Fluent_English_NonEng", (sub21["origin_group"] == 0) | ((sub21["origin_group"] == 2) & (sub21["non_fluent"] == 0))),
        ("NonFluent_English_NonEng", (sub21["origin_group"] == 0) | ((sub21["origin_group"] == 2) & (sub21["non_fluent"] == 1))),
    ]:
        g = sub21[cond].copy()
        g["treat"] = (g["origin_group"] == 2).astype(float)
        for tcol in ["aei_2026_observed", "aei_2026_writing_orth", "aei_2026_coding_raw"]:
            g["tr_t"] = g["treat"] * g[tcol]
            g["ptr_t"] = g["post"] * g["tr_t"]
            fit, se_type = fit_twoway_acs(
                "ln_wage ~ tr_t + ptr_t + age + age2 + is_black + is_asian + is_hispanic"
                " | occsoc:year + statefip:year + educ4:year + origin_group:year + sex:year",
                data=g,
                context=f"AEI_ACS_fluency_{grp_name}_{tcol}",
            )
            se_v = float(fit.se()["ptr_t"])
            fluency_rows.append({
                "subgroup": grp_name, "task": tcol,
                "coef": float(fit.coef()["ptr_t"]), "se": se_v,
                "pval": float(fit.pvalue()["ptr_t"]),
                "mde80": MDE_80_MULT * se_v,
                "se_type": se_type,
                "n_workers": int(g["cell_n"].sum()),
            })
    flu_df = pd.DataFrame(fluency_rows)
    flu_df.to_csv(RES_DIR / "aei_2026_acs_fluency.csv", index=False)
    print("\n--- ACS 2021-2024 Fluency Heterogeneity (AEI 2026) ---")
    print(flu_df.to_string(index=False))

    return {
        "n_workers_total": int(cell_df["cell_n"].sum()),
        "n_cells": int(len(cell_df)),
        "fluency": fluency_rows,
    }


def run_cps_2026_suite(t18_full: pd.DataFrame) -> Dict:
    print("\n=== Step 3: Running CPS ORG 2020 Q1-2026 Q3 2026 AEI Empirical Suite ===")
    cps = pd.read_parquet(CPS_FILE)
    cps["qtr"] = ((cps["MONTH"] - 1) // 3) + 1
    cps["yq_int"] = cps["YEAR"] * 10 + cps["qtr"]
    cps = cps[cps["yq_int"] <= 20263].copy()

    cps = cps[cps["PRCITSHP"].isin([1, 4, 5])].copy()
    is_imm = cps["PRCITSHP"].isin([4, 5])
    is_eng = is_imm & cps["PENATVTY"].isin(ENG_PENATVTY)
    cps["origin_group"] = np.where(~is_imm, 0, np.where(is_eng, 1, 2))

    cps["educ4"] = np.select(
        [cps["PEEDUCA"] < 39, cps["PEEDUCA"] == 39, cps["PEEDUCA"].between(40, 42), cps["PEEDUCA"] >= 43],
        [1, 2, 3, 4],
        default=2,
    )
    cps["age"] = cps["PRTAGE"].astype(float)
    cps["age2"] = cps["age"] ** 2
    cps["female"] = (cps["PESEX"] == 2).astype(int)
    cps["is_black"] = (cps["PTDTRACE"] == 2).astype(float)
    cps["is_asian"] = (cps["PTDTRACE"] == 4).astype(float)
    cps["is_hispanic"] = (cps["PEHSPNON"] == 1).astype(float)
    cps["statefip"] = cps["GESTFIPS"].astype(int)
    cps["wt"] = cps["EARNWT"].astype(float)
    cps["employed"] = cps["PEMLR"].isin([1, 2]).astype(float)
    cps["occ2018"] = cps["OCC_2018"].fillna(-1).astype(int)

    merge_cols = [
        "occ2018", "written_eng_only", "llm_openai_beta", "llm_writing_orth", "face_to_face_social",
        "aei_2026_observed", "aei_2026_writing_raw", "aei_2026_writing_orth",
        "aei_2026_coding_raw", "aei_2026_coding_orth",
        "aei_2026_task_writing", "aei_2026_task_coding",
        "aei_2026_robot", "aei_2025_auto_exp", "aei_2025_aug_exp",
    ]
    cps = cps.merge(t18_full[merge_cols], on="occ2018", how="left")

    wdf = cps[
        (cps["employed"] == 1)
        & (cps["EARNWEEK"] > 100.0)
        & (cps["wt"] > 0)
        & (cps["llm_openai_beta"].notna())
        & ((cps["PEHRUSL1"] >= 35) | ((cps["PEHRUSL1"] == -4) & (cps["EARNWEEK"] >= 300.0)))
    ].copy()

    wdf["earnweek_cap"] = np.minimum(wdf["EARNWEEK"], 2884.61)
    wdf["ln_earnweek_cap"] = np.log(wdf["earnweek_cap"])
    q93 = wdf.groupby("yq_int")["EARNWEEK"].transform(lambda s: s.quantile(0.93))
    wdf["ln_earnweek_q93"] = np.log(np.minimum(wdf["EARNWEEK"], q93))
    wdf["ln_earnweek_raw"] = np.log(wdf["EARNWEEK"])
    hrs_eff = np.where(wdf["PEHRUSL1"] >= 35, wdf["PEHRUSL1"].astype(float), 40.0)
    hr_from_wk = wdf["earnweek_cap"] / hrs_eff
    hr_direct = np.where((wdf["PEERNHRY"] == 1) & (wdf["HOURWAGE"] > 0), np.minimum(wdf["HOURWAGE"], 72.12), np.nan)
    wdf["hrwage_cap"] = np.where(np.isfinite(hr_direct), hr_direct, hr_from_wk)
    wdf = wdf[wdf["hrwage_cap"].between(5.0, 200.0)].copy()
    wdf["ln_hrwage_cap"] = np.log(wdf["hrwage_cap"])

    print(f"CPS ORG 2026 AEI wage sample: N={len(wdf):,} FT earners across {wdf['yq_int'].nunique()} quarters")

    # --- 3A. Early (2023-24) vs. Mature (2025-26) DiD & Origin Placebo ---
    df = wdf[wdf["yq_int"] != 20224].copy()
    df["post_23_24"] = ((df["yq_int"] >= 20231) & (df["yq_int"] <= 20244)).astype(float)
    df["post_25_26"] = (df["yq_int"] >= 20251).astype(float)
    df["is_eng"] = (df["origin_group"] == 1).astype(float)
    df["is_noneng"] = (df["origin_group"] == 2).astype(float)

    cps_tasks = [
        ("aei_2026_observed", "AEI 2026 Observed Exposure (Post-Treatment)"),
        ("aei_2026_writing_orth", "AEI 2026 Writing (Orthogonalized)"),
        ("aei_2026_coding_raw", "AEI 2026 Coding & Math (Raw)"),
        ("aei_2025_auto_exp", "AEI 2025 Automation Exposure (Caveat)"),
        ("aei_2025_aug_exp", "AEI 2025 Augmentation Exposure (Caveat)"),
        ("aei_2026_robot", "AEI 2026 Robot Exposure (Manual/Physical)"),
        ("face_to_face_social", "Face-to-Face Social Skills"),
        ("written_eng_only", "Pre-2022 O*NET Written-English Intensity"),
        ("llm_openai_beta", "Pre-Period: OpenAI Beta (Eloundou 2023/2024)"),
    ]
    outcomes = [
        ("ln_earnweek_cap", "Log Weekly Wage (Capped $2,884.61)"),
        ("ln_earnweek_q93", "Log Weekly Wage (Q93 Winsorized)"),
        ("ln_hrwage_cap", "Log Hourly Wage (Harmonized Cap)"),
        ("ln_earnweek_raw", "Log Weekly Wage (Uncapped Raw)"),
    ]

    did_rows = []
    for ycol, ylab in outcomes:
        tlist = cps_tasks if ycol == "ln_earnweek_cap" else [
            ("aei_2026_observed", "AEI 2026 Observed Exposure (Post-Treatment)"),
            ("aei_2026_writing_orth", "AEI 2026 Writing (Orthogonalized)"),
        ]
        for tcol, tlab in tlist:
            df["eng_t"] = df["is_eng"] * df[tcol]
            df["ne_t"] = df["is_noneng"] * df[tcol]
            df["p2324_eng"] = df["post_23_24"] * df["eng_t"]
            df["p2526_eng"] = df["post_25_26"] * df["eng_t"]
            df["p2324_ne"] = df["post_23_24"] * df["ne_t"]
            df["p2526_ne"] = df["post_25_26"] * df["ne_t"]

            fit, se_type = fit_twoway_cps(
                f"{ycol} ~ eng_t + ne_t + p2324_eng + p2526_eng + p2324_ne + p2526_ne"
                " + age + age2 + is_black + is_asian + is_hispanic"
                " | occ2018:yq_int + statefip:yq_int + educ4:yq_int + origin_group:yq_int + female:yq_int",
                data=df,
                context=f"AEI_CPS_DiD_{ycol}_{tcol}",
            )
            coefs, ses, pvals = fit.coef(), fit.se(), fit.pvalue()
            vcov = fit._vcov
            idx = list(coefs.index)

            def ddd_stats(k1: str, k2: str) -> Tuple[float, float, float]:
                i1, i2 = idx.index(k1), idx.index(k2)
                est = float(coefs[k1] - coefs[k2])
                var = float(vcov[i1, i1] + vcov[i2, i2] - 2.0 * vcov[i1, i2])
                se = float(np.sqrt(max(var, 1e-16)))
                pv = float(2.0 * (1.0 - stats.norm.cdf(abs(est / se))))
                return est, se, pv

            d1, d1_se, d1_p = ddd_stats("p2324_ne", "p2324_eng")
            d2, d2_se, d2_p = ddd_stats("p2526_ne", "p2526_eng")

            did_rows.append({
                "outcome": ycol, "outcome_label": ylab,
                "task": tcol, "task_label": tlab,
                "ne_2324": float(coefs["p2324_ne"]), "ne_2324_se": float(ses["p2324_ne"]), "ne_2324_p": float(pvals["p2324_ne"]),
                "ne_2324_mde80": MDE_80_MULT * float(ses["p2324_ne"]),
                "ne_2526": float(coefs["p2526_ne"]), "ne_2526_se": float(ses["p2526_ne"]), "ne_2526_p": float(pvals["p2526_ne"]),
                "ne_2526_mde80": MDE_80_MULT * float(ses["p2526_ne"]),
                "eng_2324": float(coefs["p2324_eng"]), "eng_2324_se": float(ses["p2324_eng"]), "eng_2324_p": float(pvals["p2324_eng"]),
                "eng_2526": float(coefs["p2526_eng"]), "eng_2526_se": float(ses["p2526_eng"]), "eng_2526_p": float(pvals["p2526_eng"]),
                "ddd_2324": d1, "ddd_2324_se": d1_se, "ddd_2324_p": d1_p,
                "ddd_2526": d2, "ddd_2526_se": d2_se, "ddd_2526_p": d2_p,
                "ddd_2526_mde80": MDE_80_MULT * d2_se,
                "se_type": se_type,
                "n_obs": int(fit._N), "r2": float(fit._r2),
            })

    did_df = pd.DataFrame(did_rows)
    did_df.to_csv(RES_DIR / "aei_2026_cps_did_persistence.csv", index=False)
    print("\n--- CPS ORG 2020 Q1-2026 Q3 Early vs. Mature DiD (Harmonized Cap) ---")
    c_cap = did_df[did_df["outcome"] == "ln_earnweek_cap"]
    print(c_cap[["task", "ne_2324", "ne_2324_se", "ne_2526", "ne_2526_se", "eng_2526", "ddd_2526"]].to_string(index=False))

    # --- 3B. Quarterly Event Studies (2020 Q1 - 2026 Q3, ref = 2022 Q3) + HonestDiD ---
    qtrs = sorted(wdf["yq_int"].unique().tolist())
    ref_q = 20223
    pre_qtrs = [q for q in qtrs if q < ref_q]
    post_qtrs = [q for q in qtrs if q >= 20231]

    es_rows, hd_rows = [], []
    configs = [
        ("NonEng_vs_Native", wdf[wdf["origin_group"].isin([0, 2])].copy(), 2,
         ["aei_2026_observed", "aei_2026_writing_orth", "aei_2026_coding_raw", "aei_2026_robot"]),
        ("Eng_vs_Native", wdf[wdf["origin_group"].isin([0, 1])].copy(), 1,
         ["aei_2026_observed", "aei_2026_writing_orth"]),
    ]

    for comp, sub, treat_val, tasks in configs:
        sub["treat"] = (sub["origin_group"] == treat_val).astype(float)
        for tcol in tasks:
            sub["treat_t"] = sub["treat"] * sub[tcol]
            fit, se_type = fit_twoway_cps(
                f"ln_earnweek_cap ~ treat_t + i(yq_int, treat_t, ref={ref_q})"
                " + age + age2 + is_black + is_asian + is_hispanic"
                " | occ2018:yq_int + statefip:yq_int + educ4:yq_int + origin_group:yq_int + female:yq_int",
                data=sub,
                context=f"AEI_CPS_ES_{comp}_{tcol}",
            )
            coefs, ses, pvals = fit.coef(), fit.se(), fit.pvalue()
            vcov = fit._vcov
            idx_list = list(coefs.index)
            q_to_idx, q_to_coef = {}, {}
            for q in qtrs:
                if q == ref_q:
                    es_rows.append({
                        "comparison": comp, "task": tcol, "yq_int": int(q),
                        "yq_label": f"{q // 10}Q{q % 10}", "coef": 0.0, "se": 0.0,
                        "pval": np.nan, "se_type": se_type, "n_obs": int(fit._N),
                    })
                else:
                    k = [c for c in idx_list if f"::{q}:" in c or f"[{q}]" in c or f"T.{q}]" in c][0]
                    es_rows.append({
                        "comparison": comp, "task": tcol, "yq_int": int(q),
                        "yq_label": f"{q // 10}Q{q % 10}", "coef": float(coefs[k]),
                        "se": float(ses[k]), "pval": float(pvals[k]), "se_type": se_type, "n_obs": int(fit._N),
                    })
                    q_to_idx[q] = idx_list.index(k)
                    q_to_coef[q] = float(coefs[k])

            if comp == "NonEng_vs_Native" and tcol in ("aei_2026_observed", "aei_2026_writing_orth"):
                pre_use = [q for q in pre_qtrs if q >= 20211]
                for post_tag, post_use in [("Post_2023Q1_2024Q4", [q for q in post_qtrs if q <= 20244]),
                                           ("Post_2023Q1_2026Q3", post_qtrs)]:
                    sel_q = pre_use + post_use
                    sel_idx = [q_to_idx[q] for q in sel_q]
                    beta_sel = np.array([q_to_coef[q] for q in sel_q])
                    v_sel = vcov[np.ix_(sel_idx, sel_idx)]
                    n_pre, n_post = len(pre_use), len(post_use)
                    hd = compute_honestdid_rm_bounds(
                        beta_pre=beta_sel[:n_pre],
                        beta_post=beta_sel[n_pre:],
                        vcov_full=v_sel,
                        l_vec=np.ones(n_post) / n_post,
                        pre_times=[yq_to_seq(q) for q in pre_use],
                        ref_time=yq_to_seq(ref_q),
                        post_times=[yq_to_seq(q) for q in post_use],
                        mbar_grid=[0.0, 0.5, 1.0, 1.5, 2.0],
                    )
                    hd.insert(0, "comparison", comp)
                    hd.insert(1, "task", tcol)
                    hd.insert(2, "pre_window", "Pre_2021Q1_2022Q2")
                    hd.insert(3, "post_window", post_tag)
                    hd["se_type"] = se_type
                    hd_rows.append(hd)

    pd.DataFrame(es_rows).to_csv(RES_DIR / "aei_2026_cps_event_studies.csv", index=False)
    pd.concat(hd_rows, ignore_index=True).to_csv(RES_DIR / "aei_2026_cps_honestdid.csv", index=False)

    return {"n_wage_ft": int(len(wdf)), "n_quarters": int(wdf["yq_int"].nunique())}


def main() -> None:
    t6_df, t4_df, t18_full, diag_summary = build_aei_2026_tasks()
    acs_meta = run_acs_2026_suite(t6_df, t4_df)
    cps_meta = run_cps_2026_suite(t18_full)

    summary = {
        "diagnostic": diag_summary,
        "acs_meta": acs_meta,
        "cps_meta": cps_meta,
    }
    (RES_DIR / "aei_2026_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n=== All 2026 AEI upgrade results saved cleanly to results/aei_2026_*.csv ===")


if __name__ == "__main__":
    main()
