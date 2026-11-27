#!/usr/bin/env python3
"""
Step 1: Paper 2 Major Revision — Forensic Replication, Wild Cluster Bootstrap,
Consistent-Geography Robustness, Gelbach (2016) Decompositions, and Placebo Task Mediators.

Key Revision Upgrades:
  1. Primary clean sample excludes CITIZEN == 1 (born abroad to U.S. parents, 12.3% of men)
     and U.S. territories (is_true_immigrant: BPL >= 150 & CITIZEN in {2, 3}), fixing the
     citizenship mediator so is_citizen == 1 strictly denotes naturalized citizens (CITIZEN == 2).
     Sensitivity column retains CITIZEN == 1 to match Furtado & Song (2015) directly.
  2. Eliminates all silent SE fallbacks in run_feols; explicitly logs se_estimator,
     n_clusters_bpl, and n_clusters_geo in every CSV.
  3. Audits Hispanic/race small-cluster SEs: reports both two-way CGM SEs, one-way MSA-clustered
     SEs, and restricted Wild Cluster Bootstrap p-values (Webb 6-point weights, 1,999 draws,
     clustered on origin country bpl_code).
  4. Adds consistent-geography robustness checks:
     - 1980-2024 with STATEFIP x Period FEs (zero 2012 boundary break)
     - 2000-2024 with MET2013 x Period FEs (zero 2012 boundary break)
  5. Adds placebo occupational task mediators (O*NET Math/Analytical; ALM Abstract + RTI)
     and joint horse-race regressions (Social + English + Math simultaneously) in both
     Gelbach (2016) decompositions and direct regression tables.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[3]
PAPER_DIR = ROOT / "papers" / "paper2_intermarriage_comment"
RES_DIR = PAPER_DIR / "results"

PERIOD_ORDER = [
    "1980", "1990", "2000", "2005-2007", "2008-2010",
    "2012-2015", "2017-2019", "2021-2024",
]
PERIODS_80_10 = ["1980", "1990", "2000", "2005-2007", "2008-2010"]
PERIODS_00_24 = ["2000", "2005-2007", "2008-2010", "2012-2015", "2017-2019", "2021-2024"]
PERIOD_MIDYEAR = {
    "1980": 1980, "1990": 1990, "2000": 2000, "2005-2007": 2006,
    "2008-2010": 2009, "2012-2015": 2013.5, "2017-2019": 2018, "2021-2024": 2022.5,
}
WEBB_WEIGHTS = np.array([
    -np.sqrt(1.5), -1.0, -np.sqrt(0.5),
    np.sqrt(0.5), 1.0, np.sqrt(1.5),
])


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
    Fit feols with explicit, non-silent clustering. NEVER silently drops to 1-way or HC1.
    If two-way CGM variance V1 + V2 - V12 is non-PSD in finite samples, applies the standard
    Cameron, Gelbach & Miller (2011, Sec. 2.3) eigen-truncation V+ = U max(Lambda, 0) U'
    and explicitly records 'CRV1_2way_CGM_PSD(...)' in se_estimator.
    Returns (fit, vcov_mat, se_estimator_label, n_clusters_dim1, n_clusters_dim2).
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
    else:
        fit = pf.feols(fml, data=df, weights=weights, vcov="hetero")
        vcov = np.array(fit._vcov, dtype=float)
        return fit, vcov, "HC1_hetero", 0, 0


def extract_trend_table(
    fit,
    vcov: np.ndarray,
    var: str,
    periods: List[str],
    spec_name: str,
    window: str,
    se_estimator: str,
    n_clusters_bpl: int,
    n_clusters_geo: int,
) -> pd.DataFrame:
    coefs = fit.coef()
    vcov_df = pd.DataFrame(vcov, index=coefs.index, columns=coefs.index)
    ses = pd.Series(np.sqrt(np.maximum(np.diag(vcov), 1e-18)), index=coefs.index)
    # Degrees of freedom for t-test: min(G1, G2) - 1 for 2-way, G1 - 1 for 1-way
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
        "window": window,
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
            "window": window,
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
    window: str,
    se_estimator: str,
    n_clusters_bpl: int,
    n_clusters_geo: int,
) -> Dict[str, object]:
    """
    Exact direct contrast (total_premium[p_to] - total_premium[p_from]) = beta[p_to] - beta[p_from]
    with exact variance Var(b_to) + Var(b_from) - 2 Cov(b_to, b_from), two-sided t p-value,
    and 95% confidence interval.
    """
    coefs = fit.coef()
    vcov_df = pd.DataFrame(vcov, index=coefs.index, columns=coefs.index)
    if n_clusters_bpl > 1 and n_clusters_geo > 1:
        df_t = min(n_clusters_bpl, n_clusters_geo) - 1
    elif n_clusters_bpl > 1:
        df_t = n_clusters_bpl - 1
    else:
        df_t = int(fit._N) - len(coefs)

    def _find_key(p: str) -> str:
        pcol = f"P_{p.replace('-', '_')}"
        k1 = f"{var}:{pcol}"
        k2 = f"{pcol}:{var}"
        if k1 in coefs.index:
            return k1
        if k2 in coefs.index:
            return k2
        raise KeyError(f"Interaction key for {var} x {p} not found in {list(coefs.index)}")

    k_from = _find_key(p_from)
    k_to = _find_key(p_to)
    b_from = float(coefs[k_from])
    b_to = float(coefs[k_to])
    base_b = float(coefs[var])
    diff = b_to - b_from
    var_diff = float(
        vcov_df.loc[k_to, k_to]
        + vcov_df.loc[k_from, k_from]
        - 2.0 * vcov_df.loc[k_to, k_from]
    )
    se_diff = float(np.sqrt(max(var_diff, 1e-18)))
    t_stat = diff / se_diff
    pval = float(2.0 * stats.t.sf(abs(t_stat), df=max(df_t, 1)))
    t_crit = float(stats.t.ppf(0.975, df=max(df_t, 1)))
    ci_lo = diff - t_crit * se_diff
    ci_hi = diff + t_crit * se_diff
    return {
        "window": window,
        "spec": spec_name,
        "var": var,
        "p_from": p_from,
        "p_to": p_to,
        "total_from": base_b + b_from,
        "total_to": base_b + b_to,
        "contrast": diff,
        "se": se_diff,
        "t_stat": t_stat,
        "df_t": df_t,
        "pval": pval,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "n_obs": int(fit._N),
        "se_estimator": se_estimator,
    }


def wild_cluster_bootstrap_pvals(
    df: pd.DataFrame,
    periods: List[str],
    indiv_ctrl: str,
    enclave_ctrl: str,
    fe_spec: str,
    cluster_col: str = "bpl_code",
    weights_col: str = "wt",
    B: int = 1999,
    seed: int = 20261124,
) -> Dict[str, float]:
    """
    Restricted (null-imposed) Wild Cluster Bootstrap with Webb (2023) 6-point weights,
    clustering on `cluster_col` (origin country bpl_code), via fast exact FWL projection.
    Computes WCB p-values for every interaction term native_sp x period (and 1980 base).
    """
    d = df.copy().reset_index(drop=True)
    x_cols = ["native_sp"]
    per_map = {"native_sp": periods[0]}
    for p in periods[1:]:
        pcol = f"P_{p.replace('-', '_')}"
        cname = f"nat_x_{p.replace('-', '_')}"
        d[cname] = d["native_sp"].astype(float) * d[pcol].astype(float)
        x_cols.append(cname)
        per_map[cname] = p

    d = d.loc[d[weights_col] > 0].dropna(subset=["log_wage", weights_col, cluster_col] + x_cols).reset_index(drop=True)
    fe_cols = [c.strip() for c in fe_spec.split("+")]
    while True:
        prev_n = len(d)
        for fc in fe_cols:
            cnt = d.groupby(fc)[fc].transform("count")
            d = d.loc[cnt > 1]
        if len(d) == prev_n:
            break
    d = d.reset_index(drop=True)
    w = d[weights_col].to_numpy(dtype=float)

    # Exact FWL projection of y and X onto (controls | FE)
    ctrl_fml = f"{indiv_ctrl} + {enclave_ctrl} | {fe_spec}"
    y_t = pf.feols(f"log_wage ~ {ctrl_fml}", data=d, weights=weights_col, fixef_rm="none").resid()
    X_t = np.column_stack([
        pf.feols(f"{c} ~ {ctrl_fml}", data=d, weights=weights_col, fixef_rm="none").resid()
        for c in x_cols
    ])

    # Cluster index mapping
    g_codes, g_uniques = pd.factorize(d[cluster_col], sort=False)
    G = len(g_uniques)
    k = len(x_cols)

    # Unrestricted WLS fit on demeaned data
    XtWX = X_t.T @ (w[:, None] * X_t)
    XtWX_inv = np.linalg.inv(XtWX)
    beta_hat = XtWX_inv @ (X_t.T @ (w * y_t))
    e_hat = y_t - X_t @ beta_hat

    # Cluster scores for unrestricted CRV1 t-stats: S_g = sum_{i in g} x_i w_i e_i
    def _crv1_se_vec(X_mat: np.ndarray, resid: np.ndarray, XtWX_inv_m: np.ndarray) -> np.ndarray:
        xw_e = X_mat * (w * resid)[:, None]
        # Sum by cluster using np.add.at
        S_g = np.zeros((G, X_mat.shape[1]), dtype=float)
        np.add.at(S_g, g_codes, xw_e)
        meat = (G / max(G - 1, 1)) * (S_g.T @ S_g)
        V = XtWX_inv_m @ meat @ XtWX_inv_m
        return np.sqrt(np.maximum(np.diag(V), 1e-18))

    se_orig = _crv1_se_vec(X_t, e_hat, XtWX_inv)
    t_orig = beta_hat / se_orig

    # Pre-aggregate cluster-level X'W X_g and X'W y_g? For restricted WCB per coefficient j:
    # Under H0: beta_j = 0, fit restricted model on X_{-j}, get y_hat_r and e_r.
    # For each draw b: y*_b = y_hat_r + v_{g,b} * e_r -> beta*_b = beta_r + XtWX_inv (X' W (v_b e_r))
    # Since X_t' W y_hat_r = X_t' W X_{-j} beta_r, and beta_j(y_hat_r) = 0 identically!
    # So beta*_j = e_j' XtWX_inv sum_g v_{g,b} s_{g,r} where s_{g,r} = sum_{i in g} x_i w_i e_{r,i}!
    # And residual e*_b = v_b e_r - X_t (beta*_b - beta_r) = v_b e_r - X_t d_b.
    # Thus cluster score for draw b is:
    # S*_{g,b} = sum_{i in g} x_i w_i (v_{g,b} e_{r,i} - x_i' d_b) = v_{g,b} s_{g,r} - M_g d_b!
    # where M_g = sum_{i in g} w_i x_i x_i' is a tiny (k x k) matrix per cluster!
    # This makes 1,999 WCB draws exact to machine precision in < 0.05 seconds!
    rng = np.random.default_rng(seed)
    V_draws = rng.choice(WEBB_WEIGHTS, size=(B, G))  # (B, G)

    # Precompute M_g = sum_{i in g} w_i x_i x_i' of shape (G, k, k)
    M_g = np.zeros((G, k, k), dtype=float)
    for g in range(G):
        idx = (g_codes == g)
        Xg = X_t[idx]
        wg = w[idx]
        M_g[g] = Xg.T @ (wg[:, None] * Xg)

    wcb_pvals: Dict[str, float] = {}
    scale_G = G / max(G - 1, 1)

    for j, cname in enumerate(x_cols):
        # Restricted model dropping column j
        keep = [m for m in range(k) if m != j]
        Xr = X_t[:, keep]
        br = np.linalg.solve(Xr.T @ (w[:, None] * Xr), Xr.T @ (w * y_t))
        e_r = y_t - Xr @ br

        # Cluster restricted scores s_r: (G, k)
        s_r = np.zeros((G, k), dtype=float)
        np.add.at(s_r, g_codes, X_t * (w * e_r)[:, None])

        # For all B draws vectorized:
        # total_score_b = V_draws @ s_r  -> (B, k)
        # d_b = total_score_b @ XtWX_inv.T -> (B, k)
        # beta*_j,b = d_b[:, j]
        d_b = (V_draws @ s_r) @ XtWX_inv.T  # (B, k)
        beta_star_j = d_b[:, j]

        # Compute CRV1 SE for coefficient j across all B draws:
        # S*_{g,b} = v_{g,b} s_{g,r} - M_g d_b  (shape B x G x k)
        # We only need row j of XtWX_inv @ S*_{g,b}: let a_j = XtWX_inv[j, :] (k,)
        # Then scalar score for coef j in cluster g, draw b is:
        # z_{b,g} = v_{b,g} (s_{g,r} @ a_j) - d_b @ (M_g @ a_j)
        a_j = XtWX_inv[j, :]  # (k,)
        sa_g = s_r @ a_j  # (G,)
        Ma_g = np.einsum("gkm,m->gk", M_g, a_j)  # (G, k)
        z_bg = V_draws * sa_g[None, :] - d_b @ Ma_g.T  # (B, G)
        var_star_j = scale_G * np.sum(z_bg ** 2, axis=1)
        se_star_j = np.sqrt(np.maximum(var_star_j, 1e-18))
        t_star_j = beta_star_j / se_star_j

        pval_wcb = float((1 + np.sum(np.abs(t_star_j) >= abs(t_orig[j]))) / (B + 1))
        wcb_pvals[per_map[cname]] = pval_wcb

    return wcb_pvals


def _cluster_var_from_scores(scores: np.ndarray, cluster_ids: np.ndarray) -> np.ndarray:
    df_s = pd.DataFrame(scores)
    df_s["_g"] = cluster_ids
    G_sum = df_s.groupby("_g", sort=False).sum().to_numpy()
    G = len(G_sum)
    scale = G / max(G - 1, 1)
    return scale * (G_sum.T @ G_sum)


def _twoway_cluster_var(scores: np.ndarray, c1: np.ndarray, c2: np.ndarray) -> Tuple[np.ndarray, str]:
    v1 = _cluster_var_from_scores(scores, c1)
    v2 = _cluster_var_from_scores(scores, c2)
    c12 = pd.Series(c1).astype(str) + "__" + pd.Series(c2).astype(str)
    v12 = _cluster_var_from_scores(scores, c12.to_numpy())
    v = v1 + v2 - v12
    if np.any(np.diag(v) <= 0):
        lbl = "CRV1_1way_max(non_psd_2way)"
        return (v2 if np.mean(np.diag(v2)) >= np.mean(np.diag(v1)) else v1), lbl
    return v, "CRV1_2way(bpl_code+geo_id)"


def run_gelbach_decomposition(
    df: pd.DataFrame,
    periods: List[str],
    mediator_blocks: Dict[str, List[str]],
    base_controls: str,
    fe_spec: str,
    contrasts: Dict[str, Tuple[Optional[str], str]],
    label: str,
    weights_col: str = "wt",
) -> pd.DataFrame:
    d = df.copy().reset_index(drop=True)
    x1_cols = ["native_sp"]
    for p in periods[1:]:
        pcol = f"P_{p.replace('-', '_')}"
        cname = f"nat_x_{p.replace('-', '_')}"
        d[cname] = d["native_sp"].astype(float) * d[pcol].astype(float)
        x1_cols.append(cname)

    block_cols: Dict[str, List[str]] = {}
    all_x2_cols: List[str] = []
    for bname, specs in mediator_blocks.items():
        bcols = []
        for item in specs:
            if item.endswith("*PERIOD"):
                base_v = item.replace("*PERIOD", "")
                bcols.append(base_v)
                for p in periods[1:]:
                    pcol = f"P_{p.replace('-', '_')}"
                    cname = f"{base_v}_x_{p.replace('-', '_')}"
                    d[cname] = d[base_v].astype(float) * d[pcol].astype(float)
                    bcols.append(cname)
            else:
                bcols.append(item)
        block_cols[bname] = bcols
        all_x2_cols.extend(bcols)

    req_cols = ["log_wage", weights_col, "bpl_code", "geo_id"] + x1_cols + all_x2_cols
    d = d.dropna(subset=req_cols)
    d = d.loc[d[weights_col] > 0].reset_index(drop=True)

    fe_cols = [c.strip() for c in fe_spec.split("+")]
    while True:
        prev_n = len(d)
        for fc in fe_cols:
            cnt = d.groupby(fc)[fc].transform("count")
            d = d.loc[cnt > 1]
        if len(d) == prev_n:
            break
    d = d.reset_index(drop=True)
    w = d[weights_col].to_numpy(dtype=float)

    all_vars_fml = " + ".join(x1_cols + all_x2_cols)
    fit_full = pf.feols(
        f"log_wage ~ {all_vars_fml} + {base_controls} | {fe_spec}",
        data=d, weights=weights_col, vcov="hetero", fixef_rm="none",
    )
    fit_base = pf.feols(
        f"log_wage ~ {' + '.join(x1_cols)} + {base_controls} | {fe_spec}",
        data=d, weights=weights_col, vcov="hetero", fixef_rm="none",
    )
    assert int(fit_base._N) == int(fit_full._N) == len(d)
    N = int(fit_full._N)

    full_coefs = fit_full.coef()
    active_x2 = [c for c in all_x2_cols if c in full_coefs.index]
    active_block_cols = {
        bname: [c for c in cols if c in full_coefs.index]
        for bname, cols in block_cols.items()
    }

    print(f"  [Gelbach:{label}] Demeaning {len(x1_cols)} X1 + {len(active_x2)} X2 vars (N={N:,})...")
    X1_tilde = np.column_stack([
        pf.feols(f"{c} ~ {base_controls} | {fe_spec}", data=d, weights=weights_col, fixef_rm="none").resid()
        for c in x1_cols
    ])
    X2_tilde = np.column_stack([
        pf.feols(f"{c} ~ {base_controls} | {fe_spec}", data=d, weights=weights_col, fixef_rm="none").resid()
        for c in active_x2
    ])
    y_tilde = pf.feols(f"log_wage ~ {base_controls} | {fe_spec}", data=d, weights=weights_col, fixef_rm="none").resid()

    XtX_1 = X1_tilde.T @ (w[:, None] * X1_tilde)
    XtX_1_inv = np.linalg.inv(XtX_1)
    beta1_base = XtX_1_inv @ (X1_tilde.T @ (w * y_tilde))

    Z_tilde = np.hstack([X1_tilde, X2_tilde])
    ZtZ = Z_tilde.T @ (w[:, None] * Z_tilde)
    ZtZ_inv = np.linalg.inv(ZtZ)
    theta_full = ZtZ_inv @ (Z_tilde.T @ (w * y_tilde))

    p1 = len(x1_cols)
    beta1_full = theta_full[:p1]
    beta2_full = theta_full[p1:]
    eps_full = y_tilde - Z_tilde @ theta_full
    u_base = y_tilde - X1_tilde @ beta1_base

    psi_base = (X1_tilde * (w * u_base)[:, None]) @ XtX_1_inv.T
    psi_theta_full = (Z_tilde * (w * eps_full)[:, None]) @ ZtZ_inv.T
    psi_full = psi_theta_full[:, :p1]
    psi_beta2 = psi_theta_full[:, p1:]

    c1_ids = d["bpl_code"].to_numpy()
    c2_ids = d["geo_id"].to_numpy()
    n_c1 = int(d["bpl_code"].nunique())
    n_c2 = int(d["geo_id"].nunique())

    col_to_x2_idx = {c: i for i, c in enumerate(active_x2)}
    delta_by_block: Dict[str, np.ndarray] = {}
    psi_by_block: Dict[str, np.ndarray] = {}

    for bname, bcols in active_block_cols.items():
        idxs = [col_to_x2_idx[c] for c in bcols]
        X2k = X2_tilde[:, idxs]
        b2k = beta2_full[idxs]
        Gamma_k = XtX_1_inv @ (X1_tilde.T @ (w[:, None] * X2k))
        delta_k = Gamma_k @ b2k
        H_k = X2k @ b2k
        v_k = H_k - X1_tilde @ delta_k
        psi_gamma_term = (X1_tilde * (w * v_k)[:, None]) @ XtX_1_inv.T
        psi_beta_term = psi_beta2[:, idxs] @ Gamma_k.T
        delta_by_block[bname] = delta_k
        psi_by_block[bname] = psi_gamma_term + psi_beta_term

    sum_deltas = sum(delta_by_block.values())
    total_explained = beta1_base - beta1_full
    ident_err = np.max(np.abs(sum_deltas - total_explained))
    print(f"  [Gelbach:{label}] Exact identity max|err| = {ident_err:.2e}")
    assert ident_err < 1e-8

    period_to_x1_idx = {periods[0]: 0}
    for i, p in enumerate(periods[1:], start=1):
        period_to_x1_idx[p] = i

    rows = []
    for cname, (p_from, p_to) in contrasts.items():
        c_vec = np.zeros(p1)
        if p_from is None or p_from == periods[0]:
            c_vec[period_to_x1_idx[p_to]] = 1.0
        else:
            c_vec[period_to_x1_idx[p_to]] = 1.0
            c_vec[period_to_x1_idx[p_from]] = -1.0

        def _stat(est_vec: np.ndarray, psi_mat: np.ndarray) -> Tuple[float, float, float, str]:
            est = float(c_vec @ est_vec)
            sc = (psi_mat @ c_vec)[:, None]
            vmat, v_lbl = _twoway_cluster_var(sc, c1_ids, c2_ids)
            se = float(np.sqrt(max(float(vmat[0, 0]), 1e-18)))
            pval = float(2 * (1 - stats.norm.cdf(abs(est / se))))
            return est, se, pval, v_lbl

        base_est, base_se, base_p, base_vl = _stat(beta1_base, psi_base)
        full_est, full_se, full_p, full_vl = _stat(beta1_full, psi_full)
        expl_est, expl_se, expl_p, expl_vl = _stat(total_explained, psi_base - psi_full)

        rows.append({
            "sample": label, "contrast": cname, "component": "Base_Estimate",
            "estimate": base_est, "se": base_se, "pval": base_p,
            "share_of_base": 1.0, "n_obs": N, "r2_base": float(fit_base._r2), "r2_full": float(fit_full._r2),
            "se_estimator": base_vl, "n_clusters_bpl": n_c1, "n_clusters_geo": n_c2,
        })
        for bname in active_block_cols.keys():
            dk, sek, pk, vl = _stat(delta_by_block[bname], psi_by_block[bname])
            rows.append({
                "sample": label, "contrast": cname, "component": f"Block_{bname}",
                "estimate": dk, "se": sek, "pval": pk,
                "share_of_base": dk / base_est if abs(base_est) > 1e-9 else np.nan,
                "n_obs": N, "r2_base": float(fit_base._r2), "r2_full": float(fit_full._r2),
                "se_estimator": vl, "n_clusters_bpl": n_c1, "n_clusters_geo": n_c2,
            })
        rows.append({
            "sample": label, "contrast": cname, "component": "Total_Explained",
            "estimate": expl_est, "se": expl_se, "pval": expl_p,
            "share_of_base": expl_est / base_est if abs(base_est) > 1e-9 else np.nan,
            "n_obs": N, "r2_base": float(fit_base._r2), "r2_full": float(fit_full._r2),
            "se_estimator": expl_vl, "n_clusters_bpl": n_c1, "n_clusters_geo": n_c2,
        })
        rows.append({
            "sample": label, "contrast": cname, "component": "Full_Unexplained",
            "estimate": full_est, "se": full_se, "pval": full_p,
            "share_of_base": full_est / base_est if abs(base_est) > 1e-9 else np.nan,
            "n_obs": N, "r2_base": float(fit_base._r2), "r2_full": float(fit_full._r2),
            "se_estimator": full_vl, "n_clusters_bpl": n_c1, "n_clusters_geo": n_c2,
        })
    return pd.DataFrame(rows)


def main() -> None:
    print("[01_forensic_and_gelbach] Loading cached men_ftfy_married.parquet & women_ftfy_married.parquet...")
    men_all = pd.read_parquet(RES_DIR / "men_ftfy_married.parquet")
    women_all = pd.read_parquet(RES_DIR / "women_ftfy_married.parquet")

    # Sample definitions:
    # 1. F&S-style sample (retaining CITIZEN == 1, excluding only PR at extract time)
    men_fs_80_10 = men_all.loc[men_all["YEAR"] <= 2010].copy()
    men_nopr_incl_cit1 = men_all.loc[~men_all["is_us_territory"]].copy()
    men_nopr_incl_cit1_80_10 = men_nopr_incl_cit1.loc[men_nopr_incl_cit1["YEAR"] <= 2010].copy()

    # 2. Primary clean sample: true foreign-born immigrants (BPL >= 150 & CITIZEN in {2, 3})
    men_clean = men_all.loc[men_all["is_true_immigrant"]].copy()
    men_clean_80_10 = men_clean.loc[men_clean["YEAR"] <= 2010].copy()
    women_clean = women_all.loc[women_all["is_true_immigrant"]].copy()

    print(f"  Men all (incl CITIZEN=1): {len(men_all):,} | Clean true immigrants: {len(men_clean):,} "
          f"(dropped {len(men_nopr_incl_cit1) - len(men_clean):,} CITIZEN=1 = "
          f"{100*(len(men_nopr_incl_cit1)-len(men_clean))/len(men_nopr_incl_cit1):.2f}%)")
    print(f"  Women all (incl CITIZEN=1): {len(women_all):,} | Clean true immigrants: {len(women_clean):,}")

    indiv_ctrl = "AGE + age_sq + ed_hs + ed_sc + ed_col + has_child + eng_fluent + yrs_in_us"
    enclave_ctrl = "enclave_share + enclave_share_sq"

    # =========================================================================
    # 1A. TABLE 2: FORENSIC REPLICATION + CLEAN PRIMARY + CITIZEN=1 SENSITIVITY
    #     + CONSISTENT-GEOGRAPHY ROBUSTNESS (STATEFIP x Period & MET2013 x Period)
    # =========================================================================
    print("\n--- 1A: Table 2 Side-by-Side + Full-Metro vs FB Denominator + CITIZEN==1 + Consistent Geography ---")
    t2_rows = []
    contrast_rows = []
    enclave_fb_ctrl = "enclave_share_fb + enclave_share_fb_sq"
    for win_lbl, d_fs, d_incl_cit1, d_clean, pers in [
        ("1980-2010", men_fs_80_10, men_nopr_incl_cit1_80_10, men_clean_80_10, PERIODS_80_10),
        ("1980-2024", men_all, men_nopr_incl_cit1, men_clean, PERIOD_ORDER),
    ]:
        nat_t, _ = build_interaction_terms("native_sp", pers)
        specs = [
            ("Col1_Raw_FS", d_fs, f"log_wage ~ {nat_t} | period", ("bpl_code", "geo_id")),
            ("Col2_Indiv_FS", d_fs, f"log_wage ~ {nat_t} + {indiv_ctrl} | period", ("bpl_code", "geo_id")),
            ("Col3_FS_Baseline", d_fs, f"log_wage ~ {nat_t} + {indiv_ctrl} + {enclave_ctrl} | bpl_code + geo_period", ("bpl_code", "geo_id")),
            ("Col3b_FS_Baseline_FBDenom", d_fs, f"log_wage ~ {nat_t} + {indiv_ctrl} + {enclave_fb_ctrl} | bpl_code + geo_period", ("bpl_code", "geo_id")),
            ("Col4_Clean_TrueImm", d_clean, f"log_wage ~ {nat_t} + {indiv_ctrl} + {enclave_ctrl} + C(arr_bin) | bpl_period + geo_period", ("bpl_code", "geo_id")),
            ("Col4b_Clean_FBDenom", d_clean, f"log_wage ~ {nat_t} + {indiv_ctrl} + {enclave_fb_ctrl} + C(arr_bin) | bpl_period + geo_period", ("bpl_code", "geo_id")),
            ("Col5_Sens_InclCit1", d_incl_cit1, f"log_wage ~ {nat_t} + {indiv_ctrl} + {enclave_ctrl} + C(arr_bin) | bpl_period + geo_period", ("bpl_code", "geo_id")),
            ("Col6_Geo_StatePeriod", d_clean, f"log_wage ~ {nat_t} + {indiv_ctrl} + enclave_share_state + enclave_share_state_sq + C(arr_bin) | bpl_period + state_period", ("bpl_code", "state_code")),
        ]
        for sname, df_s, fml, cl in specs:
            fit, vcov, se_est, nb, ng = run_feols_explicit(df_s, fml, cluster_cols=cl)
            t2_rows.append(extract_trend_table(fit, vcov, "native_sp", pers, sname, win_lbl, se_est, nb, ng))
            if "2012-2015" in pers and "2021-2024" in pers:
                contrast_rows.append(
                    compute_period_contrast(fit, vcov, "native_sp", "2012-2015", "2021-2024", sname, win_lbl, se_est, nb, ng)
                )

    # Women Clean Primary (1980-2024) for direct contrast table alongside Men
    nat_full, _ = build_interaction_terms("native_sp", PERIOD_ORDER)
    fit_w, vcov_w, se_w, nb_w, ng_w = run_feols_explicit(
        women_clean,
        f"log_wage ~ {nat_full} + {indiv_ctrl} + {enclave_ctrl} + C(arr_bin) | bpl_period + geo_period",
        cluster_cols=("bpl_code", "geo_id"),
    )
    t2_rows.append(extract_trend_table(fit_w, vcov_w, "native_sp", PERIOD_ORDER, "Women_Clean_TrueImm", "1980-2024", se_w, nb_w, ng_w))
    contrast_rows.append(
        compute_period_contrast(fit_w, vcov_w, "native_sp", "2012-2015", "2021-2024", "Women_Clean_TrueImm", "1980-2024", se_w, nb_w, ng_w)
    )

    # 2000-2024 Consistent MET2013 x Period FE check (zero 2012 boundary break)
    men_clean_00_24_m13 = men_clean.loc[(men_clean["YEAR"] >= 2000) & men_clean["in_ident_met2013"]].copy()
    nat_00, _ = build_interaction_terms("native_sp", PERIODS_00_24)
    fit_m13, vcov_m13, se_m13, nb_m13, ng_m13 = run_feols_explicit(
        men_clean_00_24_m13,
        f"log_wage ~ {nat_00} + {indiv_ctrl} + enclave_share_met2013 + enclave_share_met2013_sq + C(arr_bin) | bpl_period + met2013_period",
        cluster_cols=("bpl_code", "met2013_code"),
    )
    t2_rows.append(extract_trend_table(fit_m13, vcov_m13, "native_sp", PERIODS_00_24, "Col7_Geo_Met2013Period_00_24", "2000-2024", se_m13, nb_m13, ng_m13))
    contrast_rows.append(
        compute_period_contrast(fit_m13, vcov_m13, "native_sp", "2012-2015", "2021-2024", "Col7_Geo_Met2013Period_00_24", "2000-2024", se_m13, nb_m13, ng_m13)
    )

    # Also run 2000-2024 with legacy geo_period on same years for direct comparison
    fit_geo00, vcov_g00, se_g00, nb_g00, ng_g00 = run_feols_explicit(
        men_clean.loc[men_clean["YEAR"] >= 2000].copy(),
        f"log_wage ~ {nat_00} + {indiv_ctrl} + {enclave_ctrl} + C(arr_bin) | bpl_period + geo_period",
        cluster_cols=("bpl_code", "geo_id"),
    )
    t2_rows.append(extract_trend_table(fit_geo00, vcov_g00, "native_sp", PERIODS_00_24, "Col7b_Geo_SwitchedMSA_00_24", "2000-2024", se_g00, nb_g00, ng_g00))
    contrast_rows.append(
        compute_period_contrast(fit_geo00, vcov_g00, "native_sp", "2012-2015", "2021-2024", "Col7b_Geo_SwitchedMSA_00_24", "2000-2024", se_g00, nb_g00, ng_g00)
    )

    t2_df = pd.concat(t2_rows, ignore_index=True)
    t2_df.to_csv(RES_DIR / "forensic_table2_side_by_side.csv", index=False)
    contrasts_df = pd.DataFrame(contrast_rows)
    contrasts_df.to_csv(RES_DIR / "direct_contrasts_2021_24_minus_2012_15.csv", index=False)
    print("\n  Direct 2021-24 minus 2012-15 Contrasts:")
    print(contrasts_df[["spec", "total_from", "total_to", "contrast", "se", "pval", "ci95_lo", "ci95_hi"]].to_string(index=False))

    print("  Table 2 Clean True-Immigrant vs Sensitivity (2012-15 & 2021-24):")
    for sp in ["Col3_FS_Baseline", "Col4_Clean_TrueImm", "Col5_Sens_InclCit1", "Col6_Geo_StatePeriod"]:
        sub = t2_df.loc[(t2_df["window"] == "1980-2024") & (t2_df["spec"] == sp) & t2_df["period"].isin(["2008-2010", "2012-2015", "2021-2024"])]
        vals = ", ".join(f"{r.period}: tot={r.total_premium:.3f} (se={r.se_total:.3f})" for r in sub.itertuples())
        print(f"    {sp}: {vals}")
    for sp in ["Col7_Geo_Met2013Period_00_24", "Col7b_Geo_SwitchedMSA_00_24"]:
        sub = t2_df.loc[(t2_df["window"] == "2000-2024") & (t2_df["spec"] == sp) & t2_df["period"].isin(["2000", "2012-2015", "2021-2024"])]
        vals = ", ".join(f"{r.period}: tot={r.total_premium:.3f} (se={r.se_total:.3f})" for r in sub.itertuples())
        print(f"    {sp}: {vals}")

    # =========================================================================
    # 1B. TABLE 3: BY EDUCATION (Clean True-Immigrant + F&S Sample Comparison)
    # =========================================================================
    print("\n--- 1B: Table 3 By Education (Clean True-Immigrant & F&S Comparison) ---")
    indiv_no_ed = "AGE + age_sq + has_child + eng_fluent + yrs_in_us"
    t3_rows = []
    for win_lbl, d_sub, pers in [
        ("1980-2010_Clean", men_clean_80_10, PERIODS_80_10),
        ("1980-2024_Clean", men_clean, PERIOD_ORDER),
        ("1980-2010_FS_InclCit1", men_fs_80_10, PERIODS_80_10),
    ]:
        nat_t, _ = build_interaction_terms("native_sp", pers)
        fe_s = "bpl_period + geo_period" if "Clean" in win_lbl else "bpl_code + geo_period"
        arr_s = " + C(arr_bin)" if "Clean" in win_lbl else ""
        for ed_code, ed_lbl in [
            ("lt_hs", "Less_Than_HS"),
            ("hs", "High_School"),
            ("some_col", "Some_College"),
            ("college", "College_Plus"),
        ]:
            sub = d_sub.loc[d_sub["educ_cat"] == ed_code]
            fit_e, vcov_e, se_est, nb, ng = run_feols_explicit(
                sub,
                f"log_wage ~ {nat_t} + {indiv_no_ed} + {enclave_ctrl}{arr_s} | {fe_s}",
                cluster_cols=("bpl_code", "geo_id"),
            )
            t3_rows.append(extract_trend_table(fit_e, vcov_e, "native_sp", pers, ed_lbl, win_lbl, se_est, nb, ng))
    t3_df = pd.concat(t3_rows, ignore_index=True)
    t3_df.to_csv(RES_DIR / "forensic_table3_education.csv", index=False)

    # =========================================================================
    # 1C. TABLE 4: BY RACE/ETHNICITY WITH EXPLICIT 1-WAY MSA SEs & WCB P-VALUES
    # =========================================================================
    print("\n--- 1C: Table 4 By Race/Ethnicity + One-Way MSA SEs + Wild Cluster Bootstrap (Webb, B=1999) ---")
    t4_rows = []
    overlap_stats = {}
    for win_lbl, d_sub, pers in [
        ("1980-2010_Clean", men_clean_80_10, PERIODS_80_10),
        ("1980-2024_Clean", men_clean, PERIOD_ORDER),
        ("1980-2010_FS_InclCit1", men_fs_80_10, PERIODS_80_10),
        ("1980-2024_FS_InclCit1", men_all, PERIOD_ORDER),
    ]:
        n_fs_white = int((d_sub["fs_white"] == 1).sum())
        n_overlap = int(((d_sub["RACE"] == 1) & d_sub["HISPAN"].between(1, 4)).sum())
        n_nh_white = int((d_sub["race_clean"] == "NH_White").sum())
        n_hisp = int((d_sub["race_clean"] == "Hispanic").sum())
        n_nh_black = int((d_sub["race_clean"] == "NH_Black").sum())
        n_nh_asian = int((d_sub["race_clean"] == "NH_Asian").sum())
        overlap_stats[win_lbl] = {
            "total_N": int(len(d_sub)),
            "fs_white_N": n_fs_white,
            "white_hispanic_overlap_N": n_overlap,
            "overlap_pct_of_fs_white": round(100.0 * n_overlap / max(n_fs_white, 1), 2),
            "nh_white_clean_N": n_nh_white,
            "hispanic_N": n_hisp,
            "nh_black_N": n_nh_black,
            "nh_asian_N": n_nh_asian,
        }
        nat_t, _ = build_interaction_terms("native_sp", pers)
        fe_s = "bpl_code + geo_period"
        fml = f"log_wage ~ {nat_t} + {indiv_ctrl} + {enclave_ctrl} | {fe_s}"

        for rname, mask in [
            ("FS_White_Contaminated", d_sub["fs_white"] == 1),
            ("NH_White_Clean", d_sub["race_clean"] == "NH_White"),
            ("Hispanic", d_sub["race_clean"] == "Hispanic"),
            ("NH_Black", d_sub["race_clean"] == "NH_Black"),
            ("NH_Asian", d_sub["race_clean"] == "NH_Asian"),
        ]:
            sub = d_sub.loc[mask]
            # 2-way CGM SEs
            fit_2w, vcov_2w, se_est_2w, nb, ng = run_feols_explicit(sub, fml, cluster_cols=("bpl_code", "geo_id"))
            tr = extract_trend_table(fit_2w, vcov_2w, "native_sp", pers, rname, win_lbl, se_est_2w, nb, ng)

            # 1-way MSA-clustered SEs (immune to single-origin dominance like Mexico=67% of Hispanic!)
            fit_1g, vcov_1g, _, _, _ = run_feols_explicit(sub, fml, cluster_cols=("geo_id",))
            tr_1g = extract_trend_table(fit_1g, vcov_1g, "native_sp", pers, rname, win_lbl, "CRV1_1way(geo_id)", nb, ng)
            tr["se_inc_geo"] = tr_1g["se_inc"].values
            tr["pval_inc_geo"] = tr_1g["pval_inc"].values
            tr["se_total_geo"] = tr_1g["se_total"].values
            tr["pval_total_geo"] = tr_1g["pval_total"].values

            # Largest single origin cluster share (diagnoses cluster concentration)
            bpl_shares = sub["bpl_code"].value_counts(normalize=True)
            tr["max_bpl_share"] = float(bpl_shares.iloc[0]) if len(bpl_shares) else np.nan
            tr["top_bpl_code"] = str(bpl_shares.index[0]) if len(bpl_shares) else ""

            # Restricted Wild Cluster Bootstrap (Webb 6-point weights, B=1,999, clustered on bpl_code)
            if "Clean" in win_lbl:
                wcb_map = wild_cluster_bootstrap_pvals(
                    sub, pers, indiv_ctrl, enclave_ctrl, fe_s, cluster_col="bpl_code", B=1999,
                )
                tr["wcb_pval_inc_bpl"] = tr["period"].map(wcb_map)
            else:
                tr["wcb_pval_inc_bpl"] = np.nan

            t4_rows.append(tr)

    t4_df = pd.concat(t4_rows, ignore_index=True)
    t4_df.to_csv(RES_DIR / "forensic_table4_race.csv", index=False)
    with open(RES_DIR / "forensic_table4_overlap.json", "w") as f:
        json.dump(overlap_stats, f, indent=2)

    print("  Table 4 Clean Subsample SE Audit (2008-10, 2012-15, 2021-24):")
    t4_audit = t4_df.loc[
        (t4_df["window"] == "1980-2024_Clean") & t4_df["period"].isin(["2008-2010", "2012-2015", "2021-2024"])
    ][["spec", "period", "coef_inc", "se_inc", "se_inc_geo", "pval_inc_geo", "wcb_pval_inc_bpl", "n_clusters_bpl", "max_bpl_share"]]
    print(t4_audit.to_string(index=False))

    # =========================================================================
    # 2. GELBACH (2016) DECOMPOSITION + PLACEBO TASK MEDIATORS & HORSE RACE
    # =========================================================================
    print("\n--- 2: Gelbach (2016) Decompositions (Clean True-Immigrant + Placebo Mediators) ---")
    base_ctrl_gelbach = "AGE + age_sq + has_child + yrs_in_us + enclave_share + enclave_share_sq + C(arr_bin)"
    fe_gelbach = "bpl_period + geo_period"

    blocks_80_24 = {
        "Own_Educ_x_Period": ["ed_hs*PERIOD", "ed_sc*PERIOD", "ed_col*PERIOD"],
        "English_x_Period": ["eng_fluent*PERIOD"],
        "Citizenship_x_Period": ["is_citizen*PERIOD"],
        "Spouse_Assortative": ["ed_sp_hs", "ed_sp_sc*PERIOD", "ed_sp_col*PERIOD", "sp_employed", "log_inc_sp"],
    }
    contrasts_80_24 = {
        "1980_to_2008_2010": (None, "2008-2010"),
        "1980_to_2012_2015_Rise": (None, "2012-2015"),
        "2012_2015_to_2021_2024_Compression": ("2012-2015", "2021-2024"),
        "1980_to_2021_2024_Net": (None, "2021-2024"),
    }
    gel_m80 = run_gelbach_decomposition(men_clean, PERIOD_ORDER, blocks_80_24, base_ctrl_gelbach, fe_gelbach, contrasts_80_24, "Men_1980_2024")
    gel_w80 = run_gelbach_decomposition(women_clean, PERIOD_ORDER, blocks_80_24, base_ctrl_gelbach, fe_gelbach, contrasts_80_24, "Women_1980_2024")

    # Common 2000-2024 task sample (all O*NET + ALM task columns non-missing so every placebo is apples-to-apples!)
    task_req = ["onet_social_skill", "onet_english_comm", "onet_math_analytical", "task_abstract_std", "rti_std"]
    men_task = men_clean.loc[(men_clean["YEAR"] >= 2000) & men_clean[task_req].notna().all(axis=1)].copy()
    women_task = women_clean.loc[(women_clean["YEAR"] >= 2000) & women_clean[task_req].notna().all(axis=1)].copy()
    print(f"  Common 2000-2024 Task Sample: Men N={len(men_task):,}, Women N={len(women_task):,}")

    contrasts_00_24 = {
        "2000_to_2012_2015_Rise": (None, "2012-2015"),
        "2012_2015_to_2021_2024_Compression": ("2012-2015", "2021-2024"),
        "2000_to_2021_2024_Net": (None, "2021-2024"),
    }

    blocks_soc = {**blocks_80_24, "ONET_Social_Eng_x_Period": ["onet_social_skill*PERIOD", "onet_english_comm*PERIOD"]}
    blocks_math = {**blocks_80_24, "ONET_Math_x_Period": ["onet_math_analytical*PERIOD"]}
    blocks_alm = {**blocks_80_24, "ALM_Abstract_RTI_x_Period": ["task_abstract_std*PERIOD", "rti_std*PERIOD"]}
    blocks_joint = {
        **blocks_80_24,
        "ONET_Social_Eng_x_Period": ["onet_social_skill*PERIOD", "onet_english_comm*PERIOD"],
        "ONET_Math_x_Period": ["onet_math_analytical*PERIOD"],
    }

    gel_m_soc = run_gelbach_decomposition(men_task, PERIODS_00_24, blocks_soc, base_ctrl_gelbach, fe_gelbach, contrasts_00_24, "Men_2000_2024_SocialEng")
    gel_m_math = run_gelbach_decomposition(men_task, PERIODS_00_24, blocks_math, base_ctrl_gelbach, fe_gelbach, contrasts_00_24, "Men_2000_2024_Placebo_Math")
    gel_m_alm = run_gelbach_decomposition(men_task, PERIODS_00_24, blocks_alm, base_ctrl_gelbach, fe_gelbach, contrasts_00_24, "Men_2000_2024_Placebo_ALM")
    gel_m_joint = run_gelbach_decomposition(men_task, PERIODS_00_24, blocks_joint, base_ctrl_gelbach, fe_gelbach, contrasts_00_24, "Men_2000_2024_HorseRace_SocEngMath")
    gel_w_soc = run_gelbach_decomposition(women_task, PERIODS_00_24, blocks_soc, base_ctrl_gelbach, fe_gelbach, contrasts_00_24, "Women_2000_2024_SocialEng")
    gel_w_math = run_gelbach_decomposition(women_task, PERIODS_00_24, blocks_math, base_ctrl_gelbach, fe_gelbach, contrasts_00_24, "Women_2000_2024_Placebo_Math")

    gelbach_all = pd.concat(
        [gel_m80, gel_w80, gel_m_soc, gel_m_math, gel_m_alm, gel_m_joint, gel_w_soc, gel_w_math],
        ignore_index=True,
    )
    gelbach_all.to_csv(RES_DIR / "gelbach_decomposition_all.csv", index=False)

    # =========================================================================
    # 2B. DIRECT MEDIATION PLACEBO & HORSE-RACE REGRESSIONS (2000-2024)
    # =========================================================================
    print("\n--- 2B: Direct Mediation Placebo & Joint Horse Race (2000-2024, Men & Women) ---")
    soc_00, _ = build_interaction_terms("onet_social_skill", PERIODS_00_24)
    engc_00, _ = build_interaction_terms("onet_english_comm", PERIODS_00_24)
    math_00, _ = build_interaction_terms("onet_math_analytical", PERIODS_00_24)
    abs_00, _ = build_interaction_terms("task_abstract_std", PERIODS_00_24)
    rti_00, _ = build_interaction_terms("rti_std", PERIODS_00_24)

    hr_rows = []
    hr_task_coef_rows = []
    for sex_lbl, d_t in [("Men", men_task), ("Women", women_task)]:
        base_fml = f"log_wage ~ {nat_00} + {indiv_ctrl} + {enclave_ctrl} + C(arr_bin)"
        hr_specs = [
            ("1_Base_WithOwnEduc", f"{base_fml} | {fe_gelbach}"),
            ("2_Plus_ONET_Social_Eng", f"{base_fml} + {soc_00} + {engc_00} | {fe_gelbach}"),
            ("3_Placebo_ONET_Math", f"{base_fml} + {math_00} | {fe_gelbach}"),
            ("4_Placebo_ALM_Abs_RTI", f"{base_fml} + {abs_00} + {rti_00} | {fe_gelbach}"),
            ("5_HorseRace_Soc_Eng_Math", f"{base_fml} + {soc_00} + {engc_00} + {math_00} | {fe_gelbach}"),
        ]
        base_inc_24 = None
        for sname, fml in hr_specs:
            fit, vcov, se_est, nb, ng = run_feols_explicit(d_t, fml, cluster_cols=("bpl_code", "geo_id"))
            tr = extract_trend_table(fit, vcov, "native_sp", PERIODS_00_24, sname, f"{sex_lbl}_2000_2024", se_est, nb, ng)
            inc_24 = float(tr.loc[tr["period"] == "2021-2024", "coef_inc"].iloc[0])
            if base_inc_24 is None:
                base_inc_24 = inc_24
            tr["pct_absorbed_vs_base_2024"] = (1.0 - inc_24 / base_inc_24) if abs(base_inc_24) > 1e-9 else np.nan
            hr_rows.append(tr)

            if sname == "5_HorseRace_Soc_Eng_Math":
                c_idx = fit.coef().index
                se_s = pd.Series(np.sqrt(np.maximum(np.diag(vcov), 1e-18)), index=c_idx)
                df_t = min(nb, ng) - 1
                for tvar in ["onet_social_skill", "onet_english_comm", "onet_math_analytical"]:
                    for p in ["2000", "2012-2015", "2021-2024"]:
                        k = tvar if p == "2000" else f"{tvar}:P_{p.replace('-', '_')}"
                        b_v = float(fit.coef()[k])
                        s_v = float(se_s[k])
                        p_v = float(2.0 * stats.t.sf(abs(b_v / s_v), df=max(df_t, 1)))
                        hr_task_coef_rows.append({
                            "sex": sex_lbl,
                            "task_var": tvar,
                            "period": p,
                            "coef": b_v,
                            "se": s_v,
                            "pval": p_v,
                            "n_obs": int(fit._N),
                            "se_estimator": se_est,
                        })

    hr_df = pd.concat(hr_rows, ignore_index=True)
    hr_df.to_csv(RES_DIR / "mediation_placebo_horserace.csv", index=False)
    hr_task_df = pd.DataFrame(hr_task_coef_rows)
    hr_task_df.to_csv(RES_DIR / "mediation_horserace_task_coefs.csv", index=False)

    print("  Men 2000->2021-24 Direct Placebo Absorption:")
    m_hr = hr_df.loc[(hr_df["window"] == "Men_2000_2024") & (hr_df["period"] == "2021-2024")]
    print(m_hr[["spec", "coef_inc", "se_inc", "pval_inc", "pct_absorbed_vs_base_2024", "n_obs"]].to_string(index=False))
    print("\n  Joint Horse-Race Task x 2021-24 Coefficients (Men):")
    print(hr_task_df.loc[(hr_task_df["sex"] == "Men") & (hr_task_df["period"] == "2021-2024")].to_string(index=False))
    print("\n[01_forensic_and_gelbach] Complete!")


if __name__ == "__main__":
    main()
