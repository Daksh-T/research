#!/usr/bin/env python3
"""
Paper 1 -- High-Frequency Quarterly CPS ORG Event Study, Post-2024 Persistence & LP HonestDiD
=============================================================================================
Uses verified Census Bureau Basic Monthly CPS ORG (81 monthly files spanning Jan 2020 - Oct 2026;
Oct 2025 is missing from Census releases due to the federal shutdown, leaving 2025 Q4 with two
months; 1.187M rows) at data/ipums/raw/cps_org_2015_2026/cps_org_clean.parquet merged 1-to-1
with 2018 Census occupation task exposures (tasks_occ2018.parquet).

Handles the April 2023 Census CPS ORG top-coding & rounding break rigorously:
  - Primary outcome 1: Harmonized capped log weekly earnings (EARNWEEK capped at pre-Apr 2023
    fixed $2,884.61/week across all quarters 2020Q1-2026Q3).
  - Primary outcome 2: Quarterly 93rd-percentile Winsorized log weekly earnings (immune to
    nominal inflation bracket creep into the fixed $2,884.61 cap).
  - Primary outcome 3: Harmonized capped log hourly wage (HOURWAGE or EARNWEEK_cap / PEHRUSL1).
  - Robustness: Uncapped log weekly earnings & hourly-paid-only HOURWAGE.

Outputs written to papers/paper1_genai_immigrants/results/.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[3]
CPS_FILE = ROOT / "data" / "ipums" / "raw" / "cps_org_2015_2026" / "cps_org_clean.parquet"
RES_DIR = ROOT / "papers" / "paper1_genai_immigrants" / "results"

sys.path.insert(0, str(ROOT / "papers" / "paper1_genai_immigrants" / "scripts"))
from p1_acs_analysis import MDE_80_MULT, compute_honestdid_rm_bounds, fit_twoway_logged


def yq_to_seq(yq_int: int) -> float:
    """Convert YYYYQ integer (e.g., 20223) to sequential quarter index (2020Q1 = 0)."""
    return float((yq_int // 10 - 2020) * 4 + (yq_int % 10 - 1))

# Exact Census CPS PENATVTY codes for English-first-language origin countries
ENG_PENATVTY = {
    119,  # Ireland
    138,  # United Kingdom
    139,  # England
    140,  # Scotland
    142,  # Northern Ireland
    300,  # Bermuda
    301,  # Canada
    310,  # Belize
    321,  # Antigua and Barbuda
    323,  # Bahamas
    324,  # Barbados
    328,  # Dominica
    330,  # Grenada
    333,  # Jamaica
    338,  # St. Kitts-Nevis
    339,  # St. Lucia
    340,  # St. Vincent and the Grenadines
    341,  # Trinidad and Tobago
    368,  # Guyana
    501,  # Australia
    515,  # New Zealand
}


def load_cps_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    print("Loading Census CPS ORG Parquet & merging OCC_2018 tasks...")
    cps = pd.read_parquet(CPS_FILE)
    t18 = pd.read_parquet(RES_DIR / "tasks_occ2018.parquet")

    # Quarter index: 2020Q1 .. 2026Q3 (drop partial 2026Q4 with only 1 month Oct 2026 for clean quarterly balance)
    cps["qtr"] = ((cps["MONTH"] - 1) // 3) + 1
    cps["yq_int"] = cps["YEAR"] * 10 + cps["qtr"]
    cps = cps[cps["yq_int"] <= 20263].copy()

    # Origin group:
    #   0 = U.S. Natives (PRCITSHP == 1, PENATVTY == 57)
    #   1 = English-Speaking-Origin Immigrants (PRCITSHP IN (4, 5) & PENATVTY in ENG_PENATVTY)
    #   2 = Non-English-Speaking-Origin Immigrants (PRCITSHP IN (4, 5) & PENATVTY not in ENG_PENATVTY)
    # Drop U.S. territories (PRCITSHP == 2) & born abroad of U.S. parents (PRCITSHP == 3)
    cps = cps[cps["PRCITSHP"].isin([1, 4, 5])].copy()
    is_imm = cps["PRCITSHP"].isin([4, 5])
    is_eng = is_imm & cps["PENATVTY"].isin(ENG_PENATVTY)
    cps["origin_group"] = np.where(~is_imm, 0, np.where(is_eng, 1, 2))

    # 4-category education matching ACS (PEEDUCA: <39 <HS, 39 HS, 40-42 SomeCol/Assoc, >=43 BA+)
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

    # Merge tasks on OCC_2018
    cps["occ2018"] = cps["OCC_2018"].fillna(-1).astype(int)
    task_cols = [
        "occ2018", "written_eng_only", "llm_openai_beta", "llm_openai_alpha", "llm_felten_lm",
        "llm_writing_raw", "llm_coding_math_raw", "llm_writing_orth",
        "llm_coding_math_orth", "face_to_face_social",
    ]
    cps = cps.merge(t18[task_cols], on="occ2018", how="left")

    # Full extensive-margin frame (all prime-age ORG respondents)
    emp_frame = cps[["yq_int", "YEAR", "qtr", "statefip", "educ4", "female", "origin_group",
                     "age", "age2", "is_black", "is_asian", "is_hispanic", "wt", "employed",
                     "occ2018", "llm_openai_beta", "llm_writing_orth", "llm_coding_math_orth",
                     "face_to_face_social"]].copy()

    # Wage sample: Full-time (PEHRUSL1 >= 35 or hours vary PEHRUSL1 == -4 with EARNWEEK >= 300),
    # positive EARNWEEK, valid OCC_2018 matched to tasks, positive EARNWT
    wdf = cps[
        (cps["employed"] == 1)
        & (cps["EARNWEEK"] > 100.0)
        & (cps["wt"] > 0)
        & (cps["llm_openai_beta"].notna())
        & ((cps["PEHRUSL1"] >= 35) | ((cps["PEHRUSL1"] == -4) & (cps["EARNWEEK"] >= 300.0)))
    ].copy()

    # Harmonized wage variables:
    # 1. Fixed pre-April-2023 top-code ($2,884.61/week) applied uniformly across 2020Q1-2026Q3
    wdf["earnweek_cap"] = np.minimum(wdf["EARNWEEK"], 2884.61)
    wdf["ln_earnweek_cap"] = np.log(wdf["earnweek_cap"])

    # 2. Quarterly 93rd-percentile Winsorized weekly earnings (immune to nominal bracket creep)
    q93 = wdf.groupby("yq_int")["EARNWEEK"].transform(lambda s: s.quantile(0.93))
    wdf["ln_earnweek_q93"] = np.log(np.minimum(wdf["EARNWEEK"], q93))

    # 3. Uncapped weekly earnings (raw)
    wdf["ln_earnweek_raw"] = np.log(wdf["EARNWEEK"])

    # 4. Harmonized capped hourly wage (HOURWAGE if hourly-paid, else earnweek_cap / hrs)
    hrs_eff = np.where(wdf["PEHRUSL1"] >= 35, wdf["PEHRUSL1"].astype(float), 40.0)
    hr_from_wk = wdf["earnweek_cap"] / hrs_eff
    hr_direct = np.where((wdf["PEERNHRY"] == 1) & (wdf["HOURWAGE"] > 0), np.minimum(wdf["HOURWAGE"], 72.12), np.nan)
    wdf["hrwage_cap"] = np.where(np.isfinite(hr_direct), hr_direct, hr_from_wk)
    wdf = wdf[wdf["hrwage_cap"].between(5.0, 200.0)].copy()
    wdf["ln_hrwage_cap"] = np.log(wdf["hrwage_cap"])
    wdf["is_hourly_paid"] = ((wdf["PEERNHRY"] == 1) & (wdf["HOURWAGE"].between(5.0, 200.0))).astype(int)

    counts = {
        "n_all_org": int(len(emp_frame)),
        "n_wage_ft": int(len(wdf)),
        "n_wage_native": int((wdf["origin_group"] == 0).sum()),
        "n_wage_eng": int((wdf["origin_group"] == 1).sum()),
        "n_wage_noneng": int((wdf["origin_group"] == 2).sum()),
        "n_quarters": int(wdf["yq_int"].nunique()),
        "n_occ2018_clusters": int(wdf["occ2018"].nunique()),
        "n_state_clusters": int(wdf["statefip"].nunique()),
    }
    print(f"CPS ORG Wage Sample (2020Q1-2026Q3, {counts['n_quarters']} qtrs): "
          f"N={counts['n_wage_ft']:,} (Native={counts['n_wage_native']:,}, "
          f"Eng={counts['n_wage_eng']:,}, NonEng={counts['n_wage_noneng']:,})")
    return wdf, emp_frame, counts


def run_quarterly_event_studies(wdf: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Quarterly event studies (2020Q1 - 2026Q3, ref = 2022Q3 = 20223) + LP-based HonestDiD.
    """
    qtrs = sorted(wdf["yq_int"].unique().tolist())
    ref_q = 20223  # 2022 Q3 (immediately prior to ChatGPT launch Nov 30, 2022)
    pre_qtrs = [q for q in qtrs if q < ref_q]       # 2020Q1..2022Q2 (10 pre quarters)
    post_qtrs = [q for q in qtrs if q >= 20231]     # 2023Q1..2026Q3 (15 post quarters; 2022Q4 is transition)

    es_rows = []
    honest_rows = []

    configs = [
        ("NonEng_vs_Native", wdf[wdf["origin_group"].isin([0, 2])].copy(), 2,
         ["llm_openai_beta", "llm_writing_orth", "llm_coding_math_raw", "face_to_face_social"]),
        ("Eng_vs_Native", wdf[wdf["origin_group"].isin([0, 1])].copy(), 1,
         ["llm_openai_beta", "llm_writing_orth", "llm_coding_math_raw", "face_to_face_social"]),
        ("NonEng_vs_Eng_ImmiOnly", wdf[wdf["origin_group"].isin([1, 2])].copy(), 2,
         ["llm_openai_beta", "llm_writing_orth"]),
    ]

    for comp, sub, treat_val, tasks in configs:
        sub["treat"] = (sub["origin_group"] == treat_val).astype(float)
        for tcol in tasks:
            sub["treat_t"] = sub["treat"] * sub[tcol]
            fit, se_type = fit_twoway_logged(
                f"ln_earnweek_cap ~ treat_t + i(yq_int, treat_t, ref={ref_q})"
                " + age + age2 + is_black + is_asian + is_hispanic"
                " | occ2018:yq_int + statefip:yq_int + educ4:yq_int + origin_group:yq_int + female:yq_int",
                data=sub,
                weights="wt",
                c1="occ2018",
                c2="statefip",
                context=f"CPS_ES_{comp}_{tcol}",
            )
            coefs, ses, pvals = fit.coef(), fit.se(), fit.pvalue()
            vcov = fit._vcov
            idx_list = list(coefs.index)

            q_to_idx = {}
            q_to_coef = {}
            for q in qtrs:
                if q == ref_q:
                    es_rows.append({
                        "comparison": comp, "task": tcol, "yq_int": int(q),
                        "yq_label": f"{q // 10}Q{q % 10}", "coef": 0.0, "se": 0.0, "pval": np.nan,
                        "se_type": se_type, "n_obs": int(fit._N),
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

            # LP HonestDiD with exact quarterly step scaling (note: 2023Q1 is 2 quarters after ref 2022Q3)
            for pre_tag, pre_use in [("Pre_2021Q1_2022Q2", [q for q in pre_qtrs if q >= 20211]),
                                     ("Pre_2020Q1_2022Q2", pre_qtrs)]:
                for post_tag, post_use in [("Post_2023Q1_2024Q4", [q for q in post_qtrs if q <= 20244]),
                                           ("Post_2023Q1_2026Q3", post_qtrs)]:
                    sel_q = pre_use + post_use
                    sel_idx = [q_to_idx[q] for q in sel_q]
                    beta_sel = np.array([q_to_coef[q] for q in sel_q])
                    v_sel = vcov[np.ix_(sel_idx, sel_idx)]
                    n_pre = len(pre_use)
                    n_post = len(post_use)
                    l_avg = np.ones(n_post) / n_post

                    hd = compute_honestdid_rm_bounds(
                        beta_pre=beta_sel[:n_pre],
                        beta_post=beta_sel[n_pre:],
                        vcov_full=v_sel,
                        l_vec=l_avg,
                        pre_times=[yq_to_seq(q) for q in pre_use],
                        ref_time=yq_to_seq(ref_q),
                        post_times=[yq_to_seq(q) for q in post_use],
                        mbar_grid=[0.0, 0.5, 1.0, 1.5, 2.0],
                    )
                    hd.insert(0, "comparison", comp)
                    hd.insert(1, "task", tcol)
                    hd.insert(2, "pre_window", pre_tag)
                    hd.insert(3, "post_window", post_tag)
                    hd["se_type"] = se_type
                    honest_rows.append(hd)

    es_df = pd.DataFrame(es_rows)
    honest_df = pd.concat(honest_rows, ignore_index=True)
    es_df.to_csv(RES_DIR / "cps_quarterly_event_studies.csv", index=False)
    honest_df.to_csv(RES_DIR / "cps_honestdid_bounds.csv", index=False)
    return es_df, honest_df


def run_cps_did_and_persistence(wdf: pd.DataFrame) -> pd.DataFrame:
    """
    Table 2: CPS ORG Quarterly DiD & Post-2024 Persistence (Early 2023-24 vs. Mature 2025-26)
    Excludes transition quarter 2022Q4; compares Pre (2020Q1-2022Q3) vs. Post_23_24 (2023Q1-2024Q4)
    and Post_25_26 (2025Q1-2026Q3), jointly estimating Non-English and English-Origin Placebo!
    """
    df = wdf[wdf["yq_int"] != 20224].copy()
    df["post_23_24"] = ((df["yq_int"] >= 20231) & (df["yq_int"] <= 20244)).astype(float)
    df["post_25_26"] = (df["yq_int"] >= 20251).astype(float)
    df["post_all"] = (df["yq_int"] >= 20231).astype(float)
    df["is_eng"] = (df["origin_group"] == 1).astype(float)
    df["is_noneng"] = (df["origin_group"] == 2).astype(float)

    rows = []
    outcomes = [
        ("ln_earnweek_cap", "Log Weekly Wage (Capped $2,885)"),
        ("ln_earnweek_q93", "Log Weekly Wage (Q93 Winsorized)"),
        ("ln_hrwage_cap", "Log Hourly Wage (Harmonized Cap)"),
        ("ln_earnweek_raw", "Log Weekly Wage (Uncapped Raw)"),
    ]
    tasks = [
        ("written_eng_only", "Pre-2022 O*NET Written-English"),
        ("llm_openai_beta", "OpenAI LLM Beta"),
        ("llm_writing_orth", "LLM Writing (Orth.)"),
        ("llm_coding_math_raw", "LLM Coding & Math"),
        ("face_to_face_social", "Face-to-Face Social"),
    ]

    for ycol, ylab in outcomes:
        tlist = tasks if ycol == "ln_earnweek_cap" else [("llm_openai_beta", "OpenAI LLM Beta"), ("llm_writing_orth", "LLM Writing (Orth.)")]
        for tcol, tlab in tlist:
            df["eng_t"] = df["is_eng"] * df[tcol]
            df["ne_t"] = df["is_noneng"] * df[tcol]
            df["p2324_eng"] = df["post_23_24"] * df["eng_t"]
            df["p2526_eng"] = df["post_25_26"] * df["eng_t"]
            df["p2324_ne"] = df["post_23_24"] * df["ne_t"]
            df["p2526_ne"] = df["post_25_26"] * df["ne_t"]

            fit, se_type = fit_twoway_logged(
                f"{ycol} ~ eng_t + ne_t + p2324_eng + p2526_eng + p2324_ne + p2526_ne"
                " + age + age2 + is_black + is_asian + is_hispanic"
                " | occ2018:yq_int + statefip:yq_int + educ4:yq_int + origin_group:yq_int + female:yq_int",
                data=df,
                weights="wt",
                c1="occ2018",
                c2="statefip",
                context=f"CPS_DiD_{ycol}_{tcol}",
            )
            coefs, ses, pvals = fit.coef(), fit.se(), fit.pvalue()
            vcov = fit._vcov
            names = list(coefs.index)

            def ddd_stats(k_ne: str, k_en: str):
                i_ne, i_en = names.index(k_ne), names.index(k_en)
                diff = float(coefs[k_ne] - coefs[k_en])
                var = float(vcov[i_ne, i_ne] + vcov[i_en, i_en] - 2.0 * vcov[i_ne, i_en])
                se = float(np.sqrt(max(var, 1e-12)))
                p = float(2.0 * (1.0 - stats.norm.cdf(abs(diff / se))))
                return diff, se, p

            d2324, se2324, p2324 = ddd_stats("p2324_ne", "p2324_eng")
            d2526, se2526, p2526 = ddd_stats("p2526_ne", "p2526_eng")

            # Also run pooled post_all (2023Q1-2026Q3)
            df["pall_eng"] = df["post_all"] * df["eng_t"]
            df["pall_ne"] = df["post_all"] * df["ne_t"]
            fit_all, se_type_all = fit_twoway_logged(
                f"{ycol} ~ eng_t + ne_t + pall_eng + pall_ne"
                " + age + age2 + is_black + is_asian + is_hispanic"
                " | occ2018:yq_int + statefip:yq_int + educ4:yq_int + origin_group:yq_int + female:yq_int",
                data=df,
                weights="wt",
                c1="occ2018",
                c2="statefip",
                context=f"CPS_DiD_all_{ycol}_{tcol}",
            )
            ca, sa, pa = fit_all.coef(), fit_all.se(), fit_all.pvalue()
            va = fit_all._vcov
            na = list(ca.index)
            ia_ne, ia_en = na.index("pall_ne"), na.index("pall_eng")
            dall = float(ca["pall_ne"] - ca["pall_eng"])
            seall = float(np.sqrt(max(va[ia_ne, ia_ne] + va[ia_en, ia_en] - 2.0 * va[ia_ne, ia_en], 1e-12)))
            pall_ddd = float(2.0 * (1.0 - stats.norm.cdf(abs(dall / seall))))

            rows.append({
                "outcome": ycol,
                "outcome_label": ylab,
                "task": tcol,
                "task_label": tlab,
                "b_ne_2324": float(coefs["p2324_ne"]), "se_ne_2324": float(ses["p2324_ne"]), "p_ne_2324": float(pvals["p2324_ne"]),
                "mde80_ne_2324": MDE_80_MULT * float(ses["p2324_ne"]),
                "b_ne_2526": float(coefs["p2526_ne"]), "se_ne_2526": float(ses["p2526_ne"]), "p_ne_2526": float(pvals["p2526_ne"]),
                "mde80_ne_2526": MDE_80_MULT * float(ses["p2526_ne"]),
                "b_eng_2324": float(coefs["p2324_eng"]), "se_eng_2324": float(ses["p2324_eng"]), "p_eng_2324": float(pvals["p2324_eng"]),
                "b_eng_2526": float(coefs["p2526_eng"]), "se_eng_2526": float(ses["p2526_eng"]), "p_eng_2526": float(pvals["p2526_eng"]),
                "ddd_2324": d2324, "ddd_se_2324": se2324, "ddd_p_2324": p2324,
                "ddd_2526": d2526, "ddd_se_2526": se2526, "ddd_p_2526": p2526,
                "b_ne_all": float(ca["pall_ne"]), "se_ne_all": float(sa["pall_ne"]), "p_ne_all": float(pa["pall_ne"]),
                "mde80_ne_all": MDE_80_MULT * float(sa["pall_ne"]),
                "b_eng_all": float(ca["pall_eng"]), "se_eng_all": float(sa["pall_eng"]), "p_eng_all": float(pa["pall_eng"]),
                "ddd_all": dall, "ddd_se_all": seall, "ddd_p_all": pall_ddd,
                "se_type": se_type,
                "se_type_all": se_type_all,
                "n_obs": int(fit._N),
                "r2": float(fit._r2),
            })

    res = pd.DataFrame(rows)
    res.to_csv(RES_DIR / "cps_quarterly_did_persistence.csv", index=False)
    return res


def run_cps_extensive_and_entry(emp_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Extensive margin (employment rate) & occupational entry into high-LLM jobs in CPS 2020Q1-2026Q3.
    """
    df = emp_frame[emp_frame["yq_int"] != 20224].copy()
    df["post_23_24"] = ((df["yq_int"] >= 20231) & (df["yq_int"] <= 20244)).astype(float)
    df["post_25_26"] = (df["yq_int"] >= 20251).astype(float)

    rows = []
    # 1. Employment rate (all ORG 18-64)
    for grp_name, grp_val in [("NonEng_vs_Native", 2), ("Eng_vs_Native", 1)]:
        sub = df[df["origin_group"].isin([0, grp_val])].copy()
        sub["treat"] = (sub["origin_group"] == grp_val).astype(float)
        sub["p2324_tr"] = sub["post_23_24"] * sub["treat"]
        sub["p2526_tr"] = sub["post_25_26"] * sub["treat"]
        fit = pf.feols(
            "employed ~ treat + p2324_tr + p2526_tr + age + age2 + is_black + is_asian + is_hispanic"
            " | statefip:yq_int + educ4:yq_int + female:yq_int",
            data=sub, weights="wt", vcov={"CRV1": "statefip"},
        )
        rows.append({
            "comparison": grp_name, "outcome": "emp_rate",
            "b_2324": float(fit.coef()["p2324_tr"]), "se_2324": float(fit.se()["p2324_tr"]), "p_2324": float(fit.pvalue()["p2324_tr"]),
            "b_2526": float(fit.coef()["p2526_tr"]), "se_2526": float(fit.se()["p2526_tr"]), "p_2526": float(fit.pvalue()["p2526_tr"]),
            "n_obs": int(fit._N),
        })

    # 2. Occupational sorting / entry among employed workers with valid OCC_2018
    emp_occ = df[(df["employed"] == 1) & (df["llm_openai_beta"].notna())].copy()
    emp_occ["high_llm_top25"] = (emp_occ["llm_openai_beta"] >= emp_occ["llm_openai_beta"].quantile(0.75)).astype(float)
    for grp_name, grp_val in [("NonEng_vs_Native", 2), ("Eng_vs_Native", 1)]:
        sub = emp_occ[emp_occ["origin_group"].isin([0, grp_val])].copy()
        sub["treat"] = (sub["origin_group"] == grp_val).astype(float)
        sub["p2324_tr"] = sub["post_23_24"] * sub["treat"]
        sub["p2526_tr"] = sub["post_25_26"] * sub["treat"]
        for out_col in ["llm_openai_beta", "llm_writing_orth", "llm_coding_math_orth", "high_llm_top25"]:
            fit = pf.feols(
                f"{out_col} ~ treat + p2324_tr + p2526_tr + age + age2 + is_black + is_asian + is_hispanic"
                " | statefip:yq_int + educ4:yq_int + female:yq_int",
                data=sub, weights="wt", vcov={"CRV1": "statefip"},
            )
            rows.append({
                "comparison": grp_name, "outcome": out_col,
                "b_2324": float(fit.coef()["p2324_tr"]), "se_2324": float(fit.se()["p2324_tr"]), "p_2324": float(fit.pvalue()["p2324_tr"]),
                "b_2526": float(fit.coef()["p2526_tr"]), "se_2526": float(fit.se()["p2526_tr"]), "p_2526": float(fit.pvalue()["p2526_tr"]),
                "n_obs": int(fit._N),
            })

    res = pd.DataFrame(rows)
    res.to_csv(RES_DIR / "cps_extensive_and_entry.csv", index=False)
    return res


def main() -> None:
    wdf, emp_frame, counts = load_cps_data()

    print("\n1. Running CPS Quarterly Event Studies (2020Q1-2026Q3, ref=2022Q3) & HonestDiD...")
    es_df, honest_df = run_quarterly_event_studies(wdf)

    print("\n2. Running CPS Quarterly DiD & Post-2024 Persistence (2023-24 vs. 2025-26)...")
    did_df = run_cps_did_and_persistence(wdf)

    print("\n3. Running CPS Extensive Margin & Occupational Entry (2020Q1-2026Q3)...")
    ext_df = run_cps_extensive_and_entry(emp_frame)

    summary = {
        "counts": counts,
        "did_persistence": did_df.to_dict(orient="records"),
    }
    (RES_DIR / "cps_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n================ CPS QUARTERLY DID & PERSISTENCE (2020Q1-2026Q3) ================")
    print(did_df[["outcome", "task", "b_ne_2324", "se_ne_2324", "p_ne_2324", "b_ne_2526", "se_ne_2526", "p_ne_2526",
                  "b_eng_2324", "b_eng_2526", "ddd_2324", "ddd_2526"]].to_string(index=False))

    print("\n================ CPS HONESTDID (Pre 2021Q1-2022Q2, Post 2023Q1-2026Q3) ================")
    print(honest_df[(honest_df["comparison"] == "NonEng_vs_Native") &
                    (honest_df["pre_window"] == "Pre_2021Q1_2022Q2") &
                    (honest_df["post_window"] == "Post_2023Q1_2026Q3") &
                    (honest_df["task"].isin(["llm_openai_beta", "llm_writing_orth", "llm_coding_math_raw"]))][
        ["task", "mbar", "theta_hat", "se_theta", "flci_lo", "flci_hi", "rob_ci_lo", "rob_ci_hi"]
    ].to_string(index=False))

    print("\n================ CPS EXTENSIVE MARGIN & ENTRY ================")
    print(ext_df.to_string(index=False))


if __name__ == "__main__":
    main()
