#!/usr/bin/env python3
"""
Step 2: Paper 2 Major Revision — Arrival-Age Mechanism, Gender Divergence,
Cohabitation Pooling, Stabilized IPW Selection Correction, and Oster (2019) Bounds.

All regressions use the primary clean true-immigrant sample (BPL >= 150 & CITIZEN in {2, 3}),
excluding U.S. territories and CITIZEN == 1 (born abroad to American parents).
All SEs are computed with explicit, non-silent two-way clustering (origin country + MSA)
and logged in every output CSV.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[3]
PAPER_DIR = ROOT / "papers" / "paper2_intermarriage_comment"
RES_DIR = PAPER_DIR / "results"

PERIOD_ORDER = [
    "1980", "1990", "2000", "2005-2007", "2008-2010",
    "2012-2015", "2017-2019", "2021-2024",
]
PERIODS_90_24 = ["1990", "2000", "2005-2007", "2008-2010", "2012-2015", "2017-2019", "2021-2024"]
PERIODS_00_24 = ["2000", "2005-2007", "2008-2010", "2012-2015", "2017-2019", "2021-2024"]
PERIOD_MIDYEAR = {
    "1980": 1980, "1990": 1990, "2000": 2000, "2005-2007": 2006,
    "2008-2010": 2009, "2012-2015": 2013.5, "2017-2019": 2018, "2021-2024": 2022.5,
}


def build_interaction_terms(var: str, periods: List[str]) -> Tuple[str, List[str]]:
    terms = [var]
    for p in periods[1:]:
        pcol = f"P_{p.replace('-', '_')}"
        terms.append(f"{var}:{pcol}")
    return " + ".join(terms), terms


def run_feols_explicit(
    df: pd.DataFrame,
    fml: str,
    weights: str = "wt",
    cluster_cols: Tuple[str, ...] = ("bpl_code", "geo_id"),
) -> Tuple[object, np.ndarray, str, int, int]:
    """
    Explicit, non-silent clustered feols. Applies standard CGM (2011) PSD eigen-truncation
    when finite-sample V1 + V2 - V12 is non-PSD and explicitly logs it in se_estimator.
    """
    if len(cluster_cols) == 2:
        c1, c2 = cluster_cols
        fit = pf.feols(fml, data=df, weights=weights, vcov={"CRV1": f"{c1} + {c2}"})
        vcov = np.array(fit._vcov, dtype=float)
        vcov = 0.5 * (vcov + vcov.T)
        eigvals, eigvecs = np.linalg.eigh(vcov)
        if np.any(np.diag(vcov) <= 0) or np.min(eigvals) < -1e-12:
            vcov = eigvecs @ np.diag(np.maximum(eigvals, 0.0)) @ eigvecs.T
            vcov = 0.5 * (vcov + vcov.T)
            est_lbl = f"CRV1_2way_CGM_PSD({c1}+{c2})"
        else:
            est_lbl = f"CRV1_2way({c1}+{c2})"
        return fit, vcov, est_lbl, int(df[c1].nunique()), int(df[c2].nunique())
    elif len(cluster_cols) == 1:
        c1 = cluster_cols[0]
        fit = pf.feols(fml, data=df, weights=weights, vcov={"CRV1": c1})
        vcov = np.array(fit._vcov, dtype=float)
        return fit, vcov, f"CRV1_1way({c1})", int(df[c1].nunique()), 0
    fit = pf.feols(fml, data=df, weights=weights, vcov="hetero")
    vcov = np.array(fit._vcov, dtype=float)
    return fit, vcov, "HC1_hetero", 0, 0


def extract_trend_table(
    fit,
    vcov: np.ndarray,
    var: str,
    periods: List[str],
    spec_name: str,
    group: str,
    se_estimator: str,
    n_clusters_bpl: int,
    n_clusters_geo: int,
) -> pd.DataFrame:
    coefs = fit.coef()
    vcov_df = pd.DataFrame(vcov, index=coefs.index, columns=coefs.index)
    ses = pd.Series(np.sqrt(np.maximum(np.diag(vcov), 1e-18)), index=coefs.index)
    if n_clusters_bpl > 1 and n_clusters_geo > 1:
        df_t = min(n_clusters_bpl, n_clusters_geo) - 1
    elif n_clusters_bpl > 1:
        df_t = n_clusters_bpl - 1
    else:
        df_t = int(fit._N) - len(coefs)

    def _pval(b_val: float, se_val: float) -> float:
        if not np.isfinite(b_val) or not np.isfinite(se_val) or se_val <= 0:
            return np.nan
        return float(2.0 * stats.t.sf(abs(b_val / se_val), df=max(df_t, 1)))

    base_b = float(coefs.get(var, np.nan))
    base_se = float(ses.get(var, np.nan))
    base_p = _pval(base_b, base_se)

    rows = [{
        "group": group,
        "spec": spec_name,
        "var": var,
        "period": periods[0],
        "midyear": PERIOD_MIDYEAR[periods[0]],
        "coef_inc": base_b,
        "se_inc": base_se,
        "pval_inc": base_p,
        "total_premium": base_b,
        "se_total": base_se,
        "pval_total": base_p,
        "n_obs": int(fit._N),
        "r2": float(fit._r2),
        "se_estimator": se_estimator,
        "n_clusters_bpl": n_clusters_bpl,
        "n_clusters_geo": n_clusters_geo,
    }]
    for p in periods[1:]:
        pcol = f"P_{p.replace('-', '_')}"
        key = f"{var}:{pcol}"
        if key not in coefs.index:
            key = f"{pcol}:{var}"
        b = float(coefs.get(key, np.nan))
        se = float(ses.get(key, np.nan))
        pv = _pval(b, se)
        tot = base_b + b
        if var in vcov_df.index and key in vcov_df.index:
            var_tot = float(vcov_df.loc[var, var] + vcov_df.loc[key, key] + 2 * vcov_df.loc[var, key])
            se_tot = float(np.sqrt(max(var_tot, 1e-18)))
            p_tot = _pval(tot, se_tot)
        else:
            se_tot, p_tot = np.nan, np.nan
        rows.append({
            "group": group,
            "spec": spec_name,
            "var": var,
            "period": p,
            "midyear": PERIOD_MIDYEAR[p],
            "coef_inc": b,
            "se_inc": se,
            "pval_inc": pv,
            "total_premium": tot,
            "se_total": se_tot,
            "pval_total": p_tot,
            "n_obs": int(fit._N),
            "r2": float(fit._r2),
            "se_estimator": se_estimator,
            "n_clusters_bpl": n_clusters_bpl,
            "n_clusters_geo": n_clusters_geo,
        })
    return pd.DataFrame(rows)


def compute_period_contrast(
    fit,
    vcov: np.ndarray,
    var: str,
    p_from: str,
    p_to: str,
    spec_name: str,
    group: str,
    se_estimator: str,
    n_clusters_bpl: int,
    n_clusters_geo: int,
) -> Dict[str, object]:
    coefs = fit.coef()
    vcov_df = pd.DataFrame(vcov, index=coefs.index, columns=coefs.index)
    df_t = (min(n_clusters_bpl, n_clusters_geo) - 1) if (n_clusters_bpl > 1 and n_clusters_geo > 1) else (int(fit._N) - len(coefs))

    def _find_key(p: str) -> str:
        pcol = f"P_{p.replace('-', '_')}"
        k1 = f"{var}:{pcol}"
        k2 = f"{pcol}:{var}"
        return k1 if k1 in coefs.index else k2

    k_from, k_to = _find_key(p_from), _find_key(p_to)
    b_from, b_to, base_b = float(coefs[k_from]), float(coefs[k_to]), float(coefs[var])
    diff = b_to - b_from
    var_diff = float(vcov_df.loc[k_to, k_to] + vcov_df.loc[k_from, k_from] - 2.0 * vcov_df.loc[k_to, k_from])
    se_diff = float(np.sqrt(max(var_diff, 1e-18)))
    t_stat = diff / se_diff
    pval = float(2.0 * stats.t.sf(abs(t_stat), df=max(df_t, 1)))
    t_crit = float(stats.t.ppf(0.975, df=max(df_t, 1)))
    return {
        "group": group, "spec": spec_name, "var": var,
        "p_from": p_from, "p_to": p_to,
        "total_from": base_b + b_from, "total_to": base_b + b_to,
        "contrast": diff, "se": se_diff, "t_stat": t_stat, "df_t": df_t,
        "pval": pval, "ci95_lo": diff - t_crit * se_diff, "ci95_hi": diff + t_crit * se_diff,
        "n_obs": int(fit._N), "se_estimator": se_estimator,
    }


def compute_oster_delta(beta_raw: float, r2_raw: float, beta_ctrl: float, r2_ctrl: float, rmax_mult: float = 1.3) -> Dict[str, float]:
    r_max = min(1.0, rmax_mult * r2_ctrl)
    denom = (beta_raw - beta_ctrl) * (r_max - r2_ctrl)
    if abs(denom) < 1e-12 or (r2_ctrl - r2_raw) <= 1e-12:
        delta = np.nan
    else:
        delta = (beta_ctrl * (r2_ctrl - r2_raw)) / denom
    if (r2_ctrl - r2_raw) > 1e-12:
        beta_star_d1 = beta_ctrl - (beta_raw - beta_ctrl) * ((r_max - r2_ctrl) / (r2_ctrl - r2_raw))
    else:
        beta_star_d1 = np.nan
    return {
        "beta_raw": beta_raw,
        "r2_raw": r2_raw,
        "beta_ctrl": beta_ctrl,
        "r2_ctrl": r2_ctrl,
        "r_max": r_max,
        "oster_delta": float(delta) if np.isfinite(delta) else np.nan,
        "beta_star_delta1": float(beta_star_d1) if np.isfinite(beta_star_d1) else np.nan,
    }


def main() -> None:
    print("[02_arrival_gender_ipw] Loading cached parquet datasets (Clean True-Immigrant Sample)...")
    base = pd.read_parquet(RES_DIR / "base_15gen.parquet")
    men = pd.read_parquet(RES_DIR / "men_ftfy_married.parquet")
    women = pd.read_parquet(RES_DIR / "women_ftfy_married.parquet")

    # Primary clean sample: true foreign-born immigrants (BPL >= 150 & CITIZEN in {2, 3})
    men_clean = men.loc[men["is_true_immigrant"]].copy()
    women_clean = women.loc[women["is_true_immigrant"]].copy()

    indiv_ctrl = "AGE + age_sq + ed_hs + ed_sc + ed_col + has_child + eng_fluent + yrs_in_us"
    enclave_ctrl = "enclave_share + enclave_share_sq"

    # =========================================================================
    # CONTRIBUTION 3: ARRIVAL-AGE MECHANISM TEST (0-5 vs 6-11 vs 12-17)
    # =========================================================================
    print("\n--- Contribution 3: Arrival-Age Mechanism Test (0-5 vs 6-11 vs 12-17) ---")
    arr_wage_rows = []
    arr_onet_rows = []
    arr_desc_rows = []

    nat_t, _ = build_interaction_terms("native_sp", PERIOD_ORDER)
    nonco_t, _ = build_interaction_terms("noncoeth_native_sp", PERIOD_ORDER)
    co_t, _ = build_interaction_terms("coeth_native_sp", PERIOD_ORDER)

    nat_o, _ = build_interaction_terms("native_sp", PERIODS_00_24)
    nonco_o, _ = build_interaction_terms("noncoeth_native_sp", PERIODS_00_24)
    co_o, _ = build_interaction_terms("coeth_native_sp", PERIODS_00_24)
    soc_o, _ = build_interaction_terms("onet_social_skill", PERIODS_00_24)
    engc_o, _ = build_interaction_terms("onet_english_comm", PERIODS_00_24)
    math_o, _ = build_interaction_terms("onet_math_analytical", PERIODS_00_24)

    for abin in ["0-5", "6-11", "12-17"]:
        sub = men_clean.loc[men_clean["arr_bin"] == abin].copy()
        w_s = sub["wt"].to_numpy()
        arr_desc_rows.append({
            "arr_bin": abin,
            "n_obs": int(len(sub)),
            "intermarriage_rate": float(np.average(sub["native_sp"], weights=w_s)),
            "noncoeth_rate": float(np.average(sub["noncoeth_native_sp"], weights=w_s)),
            "coeth_rate": float(np.average(sub["coeth_native_sp"], weights=w_s)),
            "eng_fluent_rate": float(np.average(sub["eng_fluent"], weights=w_s)),
            "eng_very_well_rate": float(np.average(sub["eng_very_well"], weights=w_s)),
            "college_rate": float(np.average(sub["ed_col"], weights=w_s)),
            "citizen_rate": float(np.average(sub["is_citizen"], weights=w_s)),
        })

        f_tot, v1, se1, nb1, ng1 = run_feols_explicit(
            sub, f"log_wage ~ {nat_t} + {indiv_ctrl} + {enclave_ctrl} | bpl_period + geo_period"
        )
        f_co, v2, se2, nb2, ng2 = run_feols_explicit(
            sub, f"log_wage ~ {nonco_t} + {co_t} + {indiv_ctrl} + {enclave_ctrl} | bpl_period + geo_period"
        )
        arr_wage_rows.extend([
            extract_trend_table(f_tot, v1, "native_sp", PERIOD_ORDER, "Total_Native", f"Arr_{abin}", se1, nb1, ng1),
            extract_trend_table(f_co, v2, "noncoeth_native_sp", PERIOD_ORDER, "NonCoEthnic_Native", f"Arr_{abin}", se2, nb2, ng2),
            extract_trend_table(f_co, v2, "coeth_native_sp", PERIOD_ORDER, "CoEthnic_Native", f"Arr_{abin}", se2, nb2, ng2),
        ])

        sub_o = sub.loc[
            (sub["YEAR"] >= 2000)
            & sub["onet_social_skill"].notna()
            & sub["onet_english_comm"].notna()
            & sub["onet_math_analytical"].notna()
        ].copy()
        for ycol in ["onet_social_skill", "onet_english_comm", "onet_math_analytical"]:
            f_st, vs, ses, nbs, ngs = run_feols_explicit(
                sub_o, f"{ycol} ~ {nat_o} + {indiv_ctrl} + {enclave_ctrl} | bpl_period + geo_period"
            )
            f_sco, vsc, sesc, nbsc, ngsc = run_feols_explicit(
                sub_o, f"{ycol} ~ {nonco_o} + {co_o} + {indiv_ctrl} + {enclave_ctrl} | bpl_period + geo_period"
            )
            arr_onet_rows.extend([
                extract_trend_table(f_st, vs, "native_sp", PERIODS_00_24, f"Sort_{ycol}_Total", f"Arr_{abin}", ses, nbs, ngs),
                extract_trend_table(f_sco, vsc, "noncoeth_native_sp", PERIODS_00_24, f"Sort_{ycol}_NonCoEth", f"Arr_{abin}", sesc, nbsc, ngsc),
                extract_trend_table(f_sco, vsc, "coeth_native_sp", PERIODS_00_24, f"Sort_{ycol}_CoEth", f"Arr_{abin}", sesc, nbsc, ngsc),
            ])

        f_pre, vp, se_p, nb_p, ng_p = run_feols_explicit(
            sub_o, f"log_wage ~ {nat_o} + {indiv_ctrl} + {enclave_ctrl} | bpl_period + geo_period"
        )
        f_post, vpo, se_po, nb_po, ng_po = run_feols_explicit(
            sub_o, f"log_wage ~ {nat_o} + {soc_o} + {engc_o} + {math_o} + {indiv_ctrl} + {enclave_ctrl} | bpl_period + geo_period"
        )
        arr_onet_rows.extend([
            extract_trend_table(f_pre, vp, "native_sp", PERIODS_00_24, "Wage_PreONET", f"Arr_{abin}", se_p, nb_p, ng_p),
            extract_trend_table(f_post, vpo, "native_sp", PERIODS_00_24, "Wage_PostONET_AllTasks", f"Arr_{abin}", se_po, nb_po, ng_po),
        ])

    pd.concat(arr_wage_rows, ignore_index=True).to_csv(RES_DIR / "arrival_age_wages_1980_2024.csv", index=False)
    pd.concat(arr_onet_rows, ignore_index=True).to_csv(RES_DIR / "arrival_age_onet_2000_2024.csv", index=False)
    arr_desc_df = pd.DataFrame(arr_desc_rows)
    arr_desc_df.to_csv(RES_DIR / "arrival_age_descriptives.csv", index=False)
    print("  Arrival-Age Descriptives (Clean True Immigrants):")
    print(arr_desc_df.to_string(index=False))

    # =========================================================================
    # CONTRIBUTION 4: EXPLAINING THE MALE-FEMALE DIVERGENCE
    # =========================================================================
    print("\n--- Contribution 4: Male-Female Divergence Mechanism Tests ---")
    gender_onet_rows = []
    gender_educ_rows = []
    gender_hh_rows = []
    gender_contrast_rows = []

    sp_uhrs_t, _ = build_interaction_terms("sp_uhrs", PERIOD_ORDER)
    sp_emp_t, _ = build_interaction_terms("sp_employed", PERIOD_ORDER)
    nchild_t, _ = build_interaction_terms("nchild_clean", PERIOD_ORDER)
    sp_col_t, _ = build_interaction_terms("ed_sp_col", PERIOD_ORDER)
    sp_sc_t, _ = build_interaction_terms("ed_sp_sc", PERIOD_ORDER)
    log_inc_sp_t, _ = build_interaction_terms("log_inc_sp", PERIOD_ORDER)

    for sex_lbl, d_sex in [("Men", men_clean), ("Women", women_clean)]:
        d_o = d_sex.loc[
            (d_sex["YEAR"] >= 2000)
            & d_sex["onet_social_skill"].notna()
            & d_sex["onet_english_comm"].notna()
            & d_sex["onet_math_analytical"].notna()
        ].copy()
        for ycol in ["onet_social_skill", "onet_english_comm", "onet_math_analytical"]:
            f_st, vs, ses, nbs, ngs = run_feols_explicit(
                d_o, f"{ycol} ~ {nat_o} + {indiv_ctrl} + {enclave_ctrl} + C(arr_bin) | bpl_period + geo_period"
            )
            gender_onet_rows.append(extract_trend_table(f_st, vs, "native_sp", PERIODS_00_24, f"Sort_{ycol}_Total", sex_lbl, ses, nbs, ngs))

        f_wpre, vp, sep, nbp, ngp = run_feols_explicit(
            d_o, f"log_wage ~ {nat_o} + {indiv_ctrl} + {enclave_ctrl} + C(arr_bin) | bpl_period + geo_period"
        )
        f_wpost, vpo, sepo, nbpo, ngpo = run_feols_explicit(
            d_o, f"log_wage ~ {nat_o} + {soc_o} + {engc_o} + {math_o} + {indiv_ctrl} + {enclave_ctrl} + C(arr_bin) | bpl_period + geo_period"
        )
        gender_onet_rows.extend([
            extract_trend_table(f_wpre, vp, "native_sp", PERIODS_00_24, "Wage_PreONET", sex_lbl, sep, nbp, ngp),
            extract_trend_table(f_wpost, vpo, "native_sp", PERIODS_00_24, "Wage_PostONET_AllTasks", sex_lbl, sepo, nbpo, ngpo),
        ])

        for ed_lbl, mask in [("College_Plus", d_sex["ed_col"] == 1), ("Non_College", d_sex["ed_col"] == 0)]:
            sub_e = d_sex.loc[mask]
            ctrl_e = "AGE + age_sq + has_child + eng_fluent + yrs_in_us" if ed_lbl == "College_Plus" else "AGE + age_sq + ed_hs + ed_sc + has_child + eng_fluent + yrs_in_us"
            f_ed, ve, see, nbe, nge = run_feols_explicit(
                sub_e, f"log_wage ~ {nat_t} + {ctrl_e} + {enclave_ctrl} + C(arr_bin) | bpl_period + geo_period"
            )
            gender_educ_rows.append(extract_trend_table(f_ed, ve, "native_sp", PERIOD_ORDER, ed_lbl, sex_lbl, see, nbe, nge))
            gender_contrast_rows.append(compute_period_contrast(f_ed, ve, "native_sp", "2012-2015", "2021-2024", ed_lbl, sex_lbl, see, nbe, nge))

        f_hh0, v0, se0, nb0, ng0 = run_feols_explicit(
            d_sex, f"log_wage ~ {nat_t} + {indiv_ctrl} + {enclave_ctrl} + C(arr_bin) | bpl_period + geo_period"
        )
        f_hh1, v1, se1, nb1, ng1 = run_feols_explicit(
            d_sex,
            f"log_wage ~ {nat_t} + {indiv_ctrl} + {enclave_ctrl} + C(arr_bin) + {sp_uhrs_t} + {sp_emp_t} + {nchild_t} | bpl_period + geo_period",
        )
        f_hh2, v2, se2, nb2, ng2 = run_feols_explicit(
            d_sex,
            f"log_wage ~ {nat_t} + {indiv_ctrl} + {enclave_ctrl} + C(arr_bin) + {sp_uhrs_t} + {sp_emp_t} + {nchild_t} + {sp_col_t} + {sp_sc_t} + ed_sp_hs + {log_inc_sp_t} | bpl_period + geo_period",
        )
        gender_hh_rows.extend([
            extract_trend_table(f_hh0, v0, "native_sp", PERIOD_ORDER, "1_Baseline", sex_lbl, se0, nb0, ng0),
            extract_trend_table(f_hh1, v1, "native_sp", PERIOD_ORDER, "2_Plus_HH_Spec_x_Yr", sex_lbl, se1, nb1, ng1),
            extract_trend_table(f_hh2, v2, "native_sp", PERIOD_ORDER, "3_Plus_HH_Spec_and_SpouseEducInc_x_Yr", sex_lbl, se2, nb2, ng2),
        ])

    pd.concat(gender_onet_rows, ignore_index=True).to_csv(RES_DIR / "gender_onet_mechanism.csv", index=False)
    pd.concat(gender_educ_rows, ignore_index=True).to_csv(RES_DIR / "gender_education_heterogeneity.csv", index=False)
    pd.concat(gender_hh_rows, ignore_index=True).to_csv(RES_DIR / "gender_household_specialization.csv", index=False)

    # =========================================================================
    # CONTRIBUTION 5: COHABITATION SENSITIVITY + IPW SELECTION + OSTER BOUNDS
    # =========================================================================
    print("\n--- Contribution 5A: Cohabitation Pooling Sensitivity (Clean True Immigrants) ---")
    cohab_rows = []
    nat_p_90, _ = build_interaction_terms("native_partner", PERIODS_90_24)
    nat_sp_90, _ = build_interaction_terms("native_sp", PERIODS_90_24)
    nat_p_80, _ = build_interaction_terms("native_partner", PERIOD_ORDER)

    for sex_val, sex_lbl in [(1, "Men"), (2, "Women")]:
        d_pool = base.loc[
            (base["SEX"] == sex_val)
            & base["is_true_immigrant"]
            & (base["ftfy"] == 1)
            & base["valid_wage"]
            & base["in_ident_msa"]
        ].copy()

        d_m90 = d_pool.loc[(d_pool["YEAR"] >= 1990) & (d_pool["married_sp"] == 1)].copy()
        d_mc90 = d_pool.loc[(d_pool["YEAR"] >= 1990) & (d_pool["partnered"] == 1)].copy()
        d_mc90_s = d_pool.loc[(d_pool["YEAR"] >= 1990) & ((d_pool["married_sp"] == 1) | (d_pool["cohab_strict"] == 1))].copy()
        d_mc80 = d_pool.loc[d_pool["partnered"] == 1].copy()

        f_m90, v1, se1, nb1, ng1 = run_feols_explicit(d_m90, f"log_wage ~ {nat_sp_90} + {indiv_ctrl} + {enclave_ctrl} + C(arr_bin) | bpl_period + geo_period")
        f_mc90, v2, se2, nb2, ng2 = run_feols_explicit(d_mc90, f"log_wage ~ {nat_p_90} + cohab_any_sploc + {indiv_ctrl} + {enclave_ctrl} + C(arr_bin) | bpl_period + geo_period")
        f_mc90s, v3, se3, nb3, ng3 = run_feols_explicit(d_mc90_s, f"log_wage ~ {nat_p_90} + cohab_strict + {indiv_ctrl} + {enclave_ctrl} + C(arr_bin) | bpl_period + geo_period")
        f_mc80, v4, se4, nb4, ng4 = run_feols_explicit(d_mc80, f"log_wage ~ {nat_p_80} + cohab_any_sploc + {indiv_ctrl} + {enclave_ctrl} + C(arr_bin) | bpl_period + geo_period")

        cohab_rows.extend([
            extract_trend_table(f_m90, v1, "native_sp", PERIODS_90_24, "Married_Only_1990_2024", sex_lbl, se1, nb1, ng1),
            extract_trend_table(f_mc90, v2, "native_partner", PERIODS_90_24, "Married_Plus_Cohab_1990_2024", sex_lbl, se2, nb2, ng2),
            extract_trend_table(f_mc90s, v3, "native_partner", PERIODS_90_24, "Married_Plus_CohabStrict1114_1990_2024", sex_lbl, se3, nb3, ng3),
            extract_trend_table(f_mc80, v4, "native_partner", PERIOD_ORDER, "Married_Plus_Cohab_1980_2024", sex_lbl, se4, nb4, ng4),
        ])

    pd.concat(cohab_rows, ignore_index=True).to_csv(RES_DIR / "cohabitation_sensitivity.csv", index=False)

    print("\n--- Contribution 5B: IPW Selection Correction & Oster Bounds (Clean True-Immigrant Universe) ---")
    ipw_rows = []
    ipw_diag_rows = []
    oster_rows = []

    for sex_val, sex_lbl in [(1, "Men"), (2, "Women")]:
        univ = base.loc[
            (base["SEX"] == sex_val)
            & base["is_true_immigrant"]
            & base["in_ident_msa"]
        ].copy()
        univ["selected"] = ((univ["married_sp"] == 1) & (univ["ftfy"] == 1) & univ["valid_wage"]).astype(int)

        univ["arr_6_11"] = (univ["arr_bin"] == "6-11").astype(float)
        univ["arr_12_17"] = (univ["arr_bin"] == "12-17").astype(float)
        univ["r_hisp"] = (univ["race_clean"] == "Hispanic").astype(float)
        univ["r_black"] = (univ["race_clean"] == "NH_Black").astype(float)
        univ["r_asian"] = (univ["race_clean"] == "NH_Asian").astype(float)
        feat_cols = [
            "AGE", "age_sq", "ed_hs", "ed_sc", "ed_col",
            "eng_fluent", "is_citizen", "has_child", "nchild_clean",
            "arr_6_11", "arr_12_17", "r_hisp", "r_black", "r_asian",
            "enclave_share",
        ]

        univ["pscore"] = np.nan
        univ["ipw_factor"] = np.nan
        for p in PERIOD_ORDER:
            mask_p = univ["period"] == p
            sub_p = univ.loc[mask_p]
            X_p = sub_p[feat_cols].fillna(0.0).to_numpy(dtype=float)
            y_p = sub_p["selected"].to_numpy(dtype=int)
            w_p = sub_p["wt"].to_numpy(dtype=float)

            scaler = StandardScaler()
            X_sc = scaler.fit_transform(X_p)
            clf = LogisticRegression(C=10.0, max_iter=500, solver="lbfgs")
            clf.fit(X_sc, y_p, sample_weight=w_p)
            phat = np.clip(clf.predict_proba(X_sc)[:, 1], 0.01, 0.99)
            p_bar = float(np.average(y_p, weights=w_p))
            auc = float(roc_auc_score(y_p, phat, sample_weight=w_p))

            sel_mask_p = y_p == 1
            raw_ipw = p_bar / phat
            lo, hi = np.quantile(raw_ipw[sel_mask_p], [0.01, 0.99])
            trimmed_ipw = np.clip(raw_ipw, lo, hi)

            univ.loc[mask_p, "pscore"] = phat
            univ.loc[mask_p, "ipw_factor"] = trimmed_ipw

            ipw_diag_rows.append({
                "sex": sex_lbl,
                "period": p,
                "n_universe": int(len(sub_p)),
                "n_selected": int(sel_mask_p.sum()),
                "selection_rate": p_bar,
                "logit_auc": auc,
                "ipw_p01": float(lo),
                "ipw_p99": float(hi),
                "ipw_mean_selected": float(np.mean(trimmed_ipw[sel_mask_p])),
            })

        sel_df = univ.loc[univ["selected"] == 1].copy()
        sel_df["wt_ipw"] = sel_df["wt"] * sel_df["ipw_factor"]

        f_raw_wt, v1, se1, nb1, ng1 = run_feols_explicit(sel_df, f"log_wage ~ {nat_t} | period", weights="wt")
        f_ctrl_wt, v2, se2, nb2, ng2 = run_feols_explicit(
            sel_df, f"log_wage ~ {nat_t} + {indiv_ctrl} + {enclave_ctrl} + C(arr_bin) | bpl_period + geo_period", weights="wt"
        )
        f_raw_ipw, v3, se3, nb3, ng3 = run_feols_explicit(sel_df, f"log_wage ~ {nat_t} | period", weights="wt_ipw")
        f_ctrl_ipw, v4, se4, nb4, ng4 = run_feols_explicit(
            sel_df, f"log_wage ~ {nat_t} + {indiv_ctrl} + {enclave_ctrl} + C(arr_bin) | bpl_period + geo_period", weights="wt_ipw"
        )

        ed_col_t, _ = build_interaction_terms("ed_col", PERIOD_ORDER)
        ed_sc_t, _ = build_interaction_terms("ed_sc", PERIOD_ORDER)
        ed_hs_t, _ = build_interaction_terms("ed_hs", PERIOD_ORDER)
        eng_t, _ = build_interaction_terms("eng_fluent", PERIOD_ORDER)
        cit_t, _ = build_interaction_terms("is_citizen", PERIOD_ORDER)
        f_full_ipw, v5, se5, nb5, ng5 = run_feols_explicit(
            sel_df,
            f"log_wage ~ {nat_t} + AGE + age_sq + has_child + yrs_in_us + {enclave_ctrl} + C(arr_bin) "
            f"+ {ed_hs_t} + {ed_sc_t} + {ed_col_t} + {eng_t} + {cit_t} "
            f"+ {sp_col_t} + {sp_sc_t} + ed_sp_hs + sp_employed + log_inc_sp | bpl_period + geo_period",
            weights="wt_ipw",
        )

        t_raw_wt = extract_trend_table(f_raw_wt, v1, "native_sp", PERIOD_ORDER, "Raw_Baseline", sex_lbl, se1, nb1, ng1)
        t_ctrl_wt = extract_trend_table(f_ctrl_wt, v2, "native_sp", PERIOD_ORDER, "Ctrl_Baseline", sex_lbl, se2, nb2, ng2)
        t_raw_ipw = extract_trend_table(f_raw_ipw, v3, "native_sp", PERIOD_ORDER, "Raw_IPW", sex_lbl, se3, nb3, ng3)
        t_ctrl_ipw = extract_trend_table(f_ctrl_ipw, v4, "native_sp", PERIOD_ORDER, "Ctrl_IPW", sex_lbl, se4, nb4, ng4)
        t_full_ipw = extract_trend_table(f_full_ipw, v5, "native_sp", PERIOD_ORDER, "FullMech_IPW", sex_lbl, se5, nb5, ng5)
        ipw_rows.extend([t_raw_wt, t_ctrl_wt, t_raw_ipw, t_ctrl_ipw, t_full_ipw])
        gender_contrast_rows.extend([
            compute_period_contrast(f_ctrl_wt, v2, "native_sp", "2012-2015", "2021-2024", "Ctrl_Baseline", sex_lbl, se2, nb2, ng2),
            compute_period_contrast(f_ctrl_ipw, v4, "native_sp", "2012-2015", "2021-2024", "Ctrl_IPW", sex_lbl, se4, nb4, ng4),
            compute_period_contrast(f_full_ipw, v5, "native_sp", "2012-2015", "2021-2024", "FullMech_IPW", sex_lbl, se5, nb5, ng5),
        ])

        for lbl_o, tr_raw, tr_ctrl in [
            ("Baseline", t_raw_wt.set_index("period"), t_ctrl_wt.set_index("period")),
            ("IPW_Corrected", t_raw_ipw.set_index("period"), t_ctrl_ipw.set_index("period")),
        ]:
            for p in PERIOD_ORDER[1:]:
                ost = compute_oster_delta(
                    float(tr_raw.loc[p, "coef_inc"]),
                    float(tr_raw.loc[p, "r2"]),
                    float(tr_ctrl.loc[p, "coef_inc"]),
                    float(tr_ctrl.loc[p, "r2"]),
                )
                ost["sex"] = sex_lbl
                ost["model"] = lbl_o
                ost["period"] = p
                oster_rows.append(ost)

    pd.concat(ipw_rows, ignore_index=True).to_csv(RES_DIR / "ipw_selection_corrected.csv", index=False)
    ipw_diag_df = pd.DataFrame(ipw_diag_rows)
    ipw_diag_df.to_csv(RES_DIR / "ipw_diagnostics.csv", index=False)
    pd.DataFrame(oster_rows).to_csv(RES_DIR / "oster_bounds_men_women_ipw.csv", index=False)
    gc_df = pd.DataFrame(gender_contrast_rows)
    gc_df.to_csv(RES_DIR / "direct_contrasts_gender_ipw.csv", index=False)
    print("\n  Direct 2021-24 minus 2012-15 Contrasts (Gender / Educ / IPW):")
    print(gc_df[["group", "spec", "total_from", "total_to", "contrast", "se", "pval", "ci95_lo", "ci95_hi"]].to_string(index=False))

    print("\n  IPW Diagnostics:")
    print(ipw_diag_df.to_string(index=False))
    print("\n[02_arrival_gender_ipw] Complete!")


if __name__ == "__main__":
    main()
