#!/usr/bin/env python3
"""
Paper 3 — Script 05 (Contribution 4):
Weak-IV-Robust Dynamic Shift-Share Inference (1980-2024).

Upgrades the Jaeger-Ruist-Stuhler (2018) 2-instrument dynamic 2SLS by computing:
  1. Exact Sanderson-Windmeijer (2016) cluster-robust conditional first-stage F-statistics
     (both MSA-clustered and two-way met2013 x eth clustered) for cp_flow & cp_flow_lag.
  2. Exact cluster-robust Anderson-Rubin (1949) weak-IV-robust 95% confidence sets
     (joint 2D AR confidence region + projected 1D 95% AR confidence intervals, plus
     single-IV static Bartik 95% AR CIs) for:
       - ln(Pre-Existing Residents)
       - ln(Out-of-State-Born Residents)
       - True Origin-MSA Retention Rate
       - Log Real Hourly Wage
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from essay3_analysis import fwl_demean, load_and_aggregate_real_ipums  # noqa: E402

OUT_DIR = ROOT / "papers" / "paper3_enclaves_wfh" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def cluster_meat(scores: np.ndarray, codes: np.ndarray, G: int) -> np.ndarray:
    """Fast vectorized cluster sum of outer products using bincount / pandas-free loop."""
    k = scores.shape[1]
    # Sum scores within each cluster g=0..G-1
    csums = np.zeros((G, k), dtype=float)
    np.add.at(csums, codes, scores)
    dof = G / max(G - 1, 1)
    return dof * (csums.T @ csums)


def two_way_meat(
    scores: np.ndarray,
    c1_codes: np.ndarray,
    G1: int,
    c2_codes: np.ndarray,
    G2: int,
    c12_codes: np.ndarray,
    G12: int,
) -> np.ndarray:
    M = (
        cluster_meat(scores, c1_codes, G1)
        + cluster_meat(scores, c2_codes, G2)
        - cluster_meat(scores, c12_codes, G12)
    )
    eigvals, eigvecs = np.linalg.eigh(M)
    return eigvecs @ np.diag(np.maximum(eigvals, 1e-16)) @ eigvecs.T


def two_way_cov_scalar(
    a: np.ndarray,
    b: np.ndarray,
    c1_codes: np.ndarray,
    G1: int,
    c2_codes: np.ndarray | None,
    G2: int,
    c12_codes: np.ndarray | None,
    G12: int,
    one_way_only: bool = False,
) -> float:
    """Exact Cameron-Gelbach-Miller cluster covariance component for 1D score vectors a and b."""
    def _cov(codes: np.ndarray, G: int) -> float:
        sa = np.bincount(codes, weights=a, minlength=G)
        sb = np.bincount(codes, weights=b, minlength=G)
        dof = G / max(G - 1, 1)
        return float(dof * np.dot(sa, sb))

    if one_way_only or c2_codes is None:
        return _cov(c1_codes, G1)
    return _cov(c1_codes, G1) + _cov(c2_codes, G2) - _cov(c12_codes, G12)


def solve_exact_1d_ar(
    y: np.ndarray,
    d: np.ndarray,
    z: np.ndarray,
    c1_codes: np.ndarray,
    G1: int,
    c2_codes: np.ndarray | None,
    G2: int,
    c12_codes: np.ndarray | None,
    G12: int,
    crit: float,
    ar_stat_fn,
) -> dict[str, Any]:
    """
    Exact analytical closed-form quadratic solver for the 1D cluster-robust Anderson-Rubin (1949) /
    Finlay-Magnusson (2009) confidence set:
        AR(beta) = (s_y - beta * s_d)^2 / (v_yy - 2*beta*v_yd + beta^2 * v_dd) <= crit
    <=> (s_d^2 - crit*v_dd)*beta^2 - 2*(s_y*s_d - crit*v_yd)*beta + (s_y^2 - crit*v_yy) <= 0.
    Never clips to an artificial finite search box!
    """
    ztz = float(np.dot(z, z))
    s_y = float(np.dot(z, y))
    s_d = float(np.dot(z, d))

    # Centered auxiliary residuals: e(beta) = (I - P_z)(y - beta * d) = e_y - beta * e_d
    e_y = y - z * (s_y / ztz)
    e_d = d - z * (s_d / ztz)
    sc_y = z * e_y
    sc_d = z * e_d

    v_yy = two_way_cov_scalar(sc_y, sc_y, c1_codes, G1, c2_codes, G2, c12_codes, G12)
    v_yd = two_way_cov_scalar(sc_y, sc_d, c1_codes, G1, c2_codes, G2, c12_codes, G12)
    v_dd = two_way_cov_scalar(sc_d, sc_d, c1_codes, G1, c2_codes, G2, c12_codes, G12)

    # Verify exact numerical equivalence between closed-form quadratic and ar_stat_fn at test points
    for b_test in [0.0, s_y / s_d if abs(s_d) > 1e-12 else 0.0, 0.1, -0.1]:
        denom_test = v_yy - 2.0 * b_test * v_yd + (b_test ** 2) * v_dd
        if denom_test > 1e-14:
            ar_cf = ((s_y - b_test * s_d) ** 2) / denom_test
            ar_num = ar_stat_fn(np.array([b_test]))
            assert abs(ar_cf - ar_num) < 1e-6 * max(1.0, abs(ar_num)), (
                f"Closed-form AR mismatch: {ar_cf} vs {ar_num} at beta={b_test}"
            )

    A = s_d ** 2 - crit * v_dd
    B = -2.0 * (s_y * s_d - crit * v_yd)
    C = s_y ** 2 - crit * v_yy
    disc = B ** 2 - 4.0 * A * C

    if A > 0.0 and disc >= 0.0:
        sq = float(np.sqrt(disc))
        r1 = (-B - sq) / (2.0 * A)
        r2 = (-B + sq) / (2.0 * A)
        lo, hi = (min(r1, r2), max(r1, r2))
        return {
            "ar_type": "bounded",
            "bounds": [float(lo), float(hi)],
            "ar_str": f"[{lo:+.3f}, {hi:+.3f}]",
            "ar_tex": f"$[{lo:+.3f},\\,{hi:+.3f}]$",
        }
    elif A < 0.0 and disc > 0.0:
        sq = float(np.sqrt(disc))
        r1 = (-B - sq) / (2.0 * A)
        r2 = (-B + sq) / (2.0 * A)
        lo, hi = (min(r1, r2), max(r1, r2))
        fmt = "+.4f" if abs(hi - lo) < 0.005 else "+.3f"
        return {
            "ar_type": "unbounded_union",
            "bounds": [float(lo), float(hi)],
            "ar_str": f"(-inf, {lo:{fmt}}] U [{hi:{fmt}}, +inf)",
            "ar_tex": f"$(-\\infty, {lo:{fmt}}] \\cup [{hi:{fmt}}, +\\infty)$",
        }
    else:
        return {
            "ar_type": "unbounded_real_line",
            "bounds": [float(-np.inf), float(np.inf)],
            "ar_str": "(-inf, +inf)",
            "ar_tex": "$(-\\infty, +\\infty)$",
        }


def compute_jrs_weak_iv_robust(
    df: pd.DataFrame,
    y_col: str,
    endog_cols: list[str],
    instr_cols: list[str],
    fe_cols: list[str],
    cluster_cols: list[str],
    weight_col: str | None = None,
) -> dict[str, Any]:
    needed = list(dict.fromkeys([y_col, *endog_cols, *instr_cols, *fe_cols, *cluster_cols]))
    if weight_col:
        needed.append(weight_col)
    sub = df[needed].dropna().copy()
    n = len(sub)
    w = sub[weight_col].to_numpy(dtype=float) if weight_col else np.ones(n, dtype=float)
    w = np.clip(w, 1e-12, None)
    sw = np.sqrt(w)

    all_vars = [y_col, *endog_cols, *instr_cols]
    dem = fwl_demean(sub, all_vars, fe_cols, w)

    y = dem[:, 0] * sw
    K = len(endog_cols)
    L = len(instr_cols)
    X = dem[:, 1 : 1 + K] * sw[:, None]
    Z = dem[:, 1 + K : 1 + K + L] * sw[:, None]

    # Pre-compute cluster integer codes once
    c1_codes = pd.Categorical(sub[cluster_cols[0]]).codes
    G1 = int(c1_codes.max()) + 1
    if len(cluster_cols) == 2:
        c2_codes = pd.Categorical(sub[cluster_cols[1]]).codes
        G2 = int(c2_codes.max()) + 1
        c12_codes = pd.Categorical(sub[cluster_cols[0]].astype(str) + "_" + sub[cluster_cols[1]].astype(str)).codes
        G12 = int(c12_codes.max()) + 1
        df_t = max(min(G1, G2) - 1, 1)
    else:
        c2_codes = c12_codes = None
        G2 = G12 = 0
        df_t = max(G1 - 1, 1)

    def get_meat(scores: np.ndarray, one_way_only: bool = False) -> np.ndarray:
        if one_way_only or len(cluster_cols) == 1:
            return cluster_meat(scores, c1_codes, G1)
        return two_way_meat(scores, c1_codes, G1, c2_codes, G2, c12_codes, G12)

    # 2SLS Point Estimates & Two-Way Clustered SEs
    ZtZ_inv = np.linalg.pinv(Z.T @ Z)
    X_hat = Z @ (ZtZ_inv @ (Z.T @ X))
    XtX_hat_inv = np.linalg.pinv(X_hat.T @ X_hat)
    beta = XtX_hat_inv @ (X_hat.T @ y)
    resid = y - X @ beta

    meat_2w = get_meat(X_hat * resid[:, None], one_way_only=False)
    V_2w = ((n - 1) / max(n - K, 1)) * (XtX_hat_inv @ meat_2w @ XtX_hat_inv)
    se_2w = np.sqrt(np.maximum(np.diag(V_2w), 1e-16))
    tstat_2w = beta / se_2w
    pval_2w = 2.0 * stats.t.sf(np.abs(tstat_2w), df=df_t)

    # 1. Exact Sanderson-Windmeijer (2016) Cluster-Robust Conditional First-Stage F-Stats
    sw_f_1w = {}
    sw_f_2w = {}
    if K == 1:
        # Single endogenous variable: cluster-robust Kleibergen-Paap / Wald F
        gamma = ZtZ_inv @ (Z.T @ X[:, 0])
        u_fs = X[:, 0] - Z @ gamma
        for tag, one_way, store in [("1w", True, sw_f_1w), ("2w", False, sw_f_2w)]:
            M_fs = get_meat(Z * u_fs[:, None], one_way_only=one_way)
            V_fs = ZtZ_inv @ M_fs @ ZtZ_inv
            f_val = float((gamma @ np.linalg.pinv(V_fs) @ gamma) / L)
            store[endog_cols[0]] = f_val
    else:
        for j in range(K):
            other = [idx for idx in range(K) if idx != j]
            xj = X[:, j]
            Xo = X[:, other]
            Xo_hat = Z @ (ZtZ_inv @ (Z.T @ Xo))
            # 2SLS regression of xj on Xo using Z
            delta_jo = np.linalg.pinv(Xo_hat.T @ Xo_hat) @ (Xo_hat.T @ xj)
            xj_tilde = xj - Xo @ delta_jo
            # Orthogonalize Z against Xo_hat -> rank (L - (K - 1)) = 1 for exact ID
            Z_perp = Z - Xo_hat @ (np.linalg.pinv(Xo_hat.T @ Xo_hat) @ (Xo_hat.T @ Z))
            # Extract orthonormal basis of column space of Z_perp (dimension L - K + 1)
            U_svd, s_svd, _ = np.linalg.svd(Z_perp, full_matrices=False)
            rank_z = max(L - K + 1, 1)
            Z_basis = U_svd[:, :rank_z] * s_svd[:rank_z]
            ZbTZb_inv = np.linalg.pinv(Z_basis.T @ Z_basis)
            pi_j = ZbTZb_inv @ (Z_basis.T @ xj_tilde)
            e_j = xj_tilde - Z_basis @ pi_j
            for one_way, store in [(True, sw_f_1w), (False, sw_f_2w)]:
                M_sw = get_meat(Z_basis * e_j[:, None], one_way_only=one_way)
                V_sw = ZbTZb_inv @ M_sw @ ZbTZb_inv
                f_sw = float((pi_j @ np.linalg.pinv(V_sw) @ pi_j) / rank_z)
                store[endog_cols[j]] = f_sw

    # 2. Exact Cluster-Robust Anderson-Rubin (1949) Weak-IV-Robust 95% Confidence Set
    # Test H0: gamma = 0 in auxiliary regression u(b0) = Z * gamma + e(b0), using
    # centered residual e(b0) = u(b0) - Z * gamma_hat(b0) inside the cluster meat
    # (exact Finlay-Magnusson 2009 / Stata `weakiv` cluster-robust AR test).
    def ar_stat(b0: np.ndarray, one_way_only: bool = False) -> float:
        u0 = y - X @ b0
        g_vec = Z.T @ u0
        gam_hat = ZtZ_inv @ g_vec
        e0 = u0 - Z @ gam_hat
        sc_e = Z * e0[:, None]
        Om = get_meat(sc_e, one_way_only=one_way_only)
        return float(g_vec @ np.linalg.pinv(Om) @ g_vec)

    crit_95 = float(stats.chi2.ppf(0.95, df=L))
    ar_p_zero = float(stats.chi2.sf(ar_stat(np.zeros(K)), df=L))

    ar_ci = {}
    ar_meta = {}
    ar_contour = []
    if K == 1:
        sol = solve_exact_1d_ar(
            y=y, d=X[:, 0], z=Z[:, 0],
            c1_codes=c1_codes, G1=G1,
            c2_codes=c2_codes, G2=G2,
            c12_codes=c12_codes, G12=G12,
            crit=crit_95,
            ar_stat_fn=ar_stat,
        )
        ar_ci[endog_cols[0]] = sol["bounds"]
        ar_meta[endog_cols[0]] = {
            "ar_type": sol["ar_type"],
            "ar_str": sol["ar_str"],
            "ar_tex": sol["ar_tex"],
        }
    else:
        # 2D grid around (beta_0, beta_1) for exact joint AR 95% confidence region
        # Verify strictly that NO accepted point touches the outer grid boundary; widen up to ±30 SEs,
        # and if any boundary is still touched, explicitly label as unbounded_2d!
        b0, b1 = float(beta[0]), float(beta[1])
        s0, s1 = float(se_2w[0]), float(se_2w[1])
        hit_boundary = True
        arr = np.empty((0, 2))
        for mult, ngrid in [(8.5, 91), (20.0, 121), (35.0, 141)]:
            g0 = np.linspace(b0 - mult * s0, b0 + mult * s0, ngrid)
            g1 = np.linspace(b1 - mult * s1, b1 + mult * s1, ngrid)
            in_pts = []
            for v0 in g0:
                for v1 in g1:
                    if ar_stat(np.array([v0, v1])) <= crit_95:
                        in_pts.append((float(v0), float(v1)))
            if not in_pts:
                arr = np.empty((0, 2))
                hit_boundary = False
                break
            arr = np.array(in_pts)
            hit_boundary = bool(
                arr[:, 0].min() <= g0[0] + 1e-12
                or arr[:, 0].max() >= g0[-1] - 1e-12
                or arr[:, 1].min() <= g1[0] + 1e-12
                or arr[:, 1].max() >= g1[-1] - 1e-12
            )
            if not hit_boundary:
                break

        if len(arr) > 0:
            if hit_boundary:
                for idx_c, col_c in enumerate(endog_cols):
                    ar_ci[col_c] = [float(-np.inf), float(np.inf)]
                    ar_meta[col_c] = {
                        "ar_type": "unbounded_2d",
                        "ar_str": "Unbounded (2D AR)",
                        "ar_tex": "Unbounded (2D AR)",
                    }
            else:
                for idx_c, col_c in enumerate(endog_cols):
                    lo_c, hi_c = float(arr[:, idx_c].min()), float(arr[:, idx_c].max())
                    ar_ci[col_c] = [lo_c, hi_c]
                    ar_meta[col_c] = {
                        "ar_type": "bounded_2d",
                        "ar_str": f"[{lo_c:+.3f}, {hi_c:+.3f}]",
                        "ar_tex": f"$[{lo_c:+.3f},\\,{hi_c:+.3f}]$",
                    }
                from scipy.spatial import ConvexHull
                if len(arr) >= 4:
                    hull = ConvexHull(arr)
                    hull_pts = arr[hull.vertices]
                    ar_contour = hull_pts.tolist()
        else:
            for col_c in endog_cols:
                ar_ci[col_c] = [np.nan, np.nan]
                ar_meta[col_c] = {"ar_type": "empty", "ar_str": "Empty", "ar_tex": "Empty"}

    return {
        "n": int(n),
        "k_endog": K,
        "coefs": dict(zip(endog_cols, beta.tolist())),
        "ses_2way": dict(zip(endog_cols, se_2w.tolist())),
        "tstats_2way": dict(zip(endog_cols, tstat_2w.tolist())),
        "pvals_2way": dict(zip(endog_cols, pval_2w.tolist())),
        "wald_ci_95": {
            c: [float(beta[i] - 1.96 * se_2w[i]), float(beta[i] + 1.96 * se_2w[i])]
            for i, c in enumerate(endog_cols)
        },
        "sw_f_msa_cluster": sw_f_1w,
        "sw_f_2way_cluster": sw_f_2w,
        "ar_ci_95": ar_ci,
        "ar_meta": ar_meta,
        "ar_joint_zero_pval": ar_p_zero,
        "ar_contour": ar_contour,
    }


def main() -> None:
    print("[05] Loading 1980-2024 cell panel for Uniform 10-Yr Decadal JRS & Separate ACS Extension...")
    cell, _, _, _, _ = load_and_aggregate_real_ipums()

    # Strictly separate panels to guarantee UNIFORM lag spacing and zero 5-yr vs 1-yr mixing:
    # Panel A (Core): Uniform 10-Year Decadal Panel (1980, 1990, 2000, 2010)
    #   - Static Bartik IV uses 1990, 2000, 2010 (all exact 10-year inflows)
    #   - JRS Dynamic 2IV uses 2000, 2010 (exact 10-year contemporaneous + exact 10-year lagged inflows)
    #   - Note: for TrueRetention, 1980-2000 uses 5-yr Census MIGRATE5 while 2010 uses 1-yr ACS MIGRATE1,
    #     so we ALSO run Pure 5-Yr Census Retention (1980, 1990, 2000 -> Static IV on 1990, 2000) separately!
    # Panel B (Extension): Modern ACS 1-Year Migration Panel (2010, 2019, 2024)
    #   - Static Bartik IV uses 2019 (2010->2019) and 2024 (2019->2024), all 1-yr ACS migration windows!
    decadal = cell[cell["YEAR"].isin([1980, 1990, 2000, 2010])].copy()
    census_5yr = cell[cell["YEAR"].isin([1980, 1990, 2000])].copy()
    acs_ext = cell[cell["YEAR"].isin([2010, 2019, 2024])].copy()

    specs = [
        ("ln_res_pop", "ln_Residents", None),
        ("ln_oos_res_pop", "ln_OOS_Residents", None),
        ("true_retention_rate", "TrueRetention", "orig_cohort_pop"),
        ("mean_ln_wage", "LogWage", "nat_pop"),
    ]

    summary_rows = []
    contours = {}

    def record_static(panel_tag: str, df_in: pd.DataFrame, ycol: str, lbl: str, wcol: str | None) -> None:
        sub = df_in.dropna(subset=["cp_flow", "iv_bartik_flow"]).copy()
        res_s = compute_jrs_weak_iv_robust(
            df=sub, y_col=ycol, endog_cols=["cp_flow"], instr_cols=["iv_bartik_flow"],
            fe_cols=["eth_year", "met_year", "eth_met"], cluster_cols=["met2013", "eth"], weight_col=wcol,
        )
        summary_rows.append({
            "panel": panel_tag,
            "model": "Static_Bartik_IV",
            "outcome": lbl,
            "variable": "cp_flow",
            "coef": res_s["coefs"]["cp_flow"],
            "se_2way": res_s["ses_2way"]["cp_flow"],
            "pval_2way": res_s["pvals_2way"]["cp_flow"],
            "wald_ci_lo": res_s["wald_ci_95"]["cp_flow"][0],
            "wald_ci_hi": res_s["wald_ci_95"]["cp_flow"][1],
            "ar_ci_lo": res_s["ar_ci_95"]["cp_flow"][0],
            "ar_ci_hi": res_s["ar_ci_95"]["cp_flow"][1],
            "ar_type": res_s["ar_meta"]["cp_flow"]["ar_type"],
            "ar_ci_str": res_s["ar_meta"]["cp_flow"]["ar_str"],
            "ar_ci_tex": res_s["ar_meta"]["cp_flow"]["ar_tex"],
            "sw_f_msa": res_s["sw_f_msa_cluster"]["cp_flow"],
            "sw_f_2way": res_s["sw_f_2way_cluster"]["cp_flow"],
            "ar_zero_pval": res_s["ar_joint_zero_pval"],
            "n": res_s["n"],
        })

    def record_dynamic(panel_tag: str, df_in: pd.DataFrame, ycol: str, lbl: str, wcol: str | None, save_contour: bool = False) -> None:
        sub = df_in.dropna(subset=["cp_flow", "cp_flow_lag", "iv_bartik_flow", "iv_bartik_flow_lag"]).copy()
        res_d = compute_jrs_weak_iv_robust(
            df=sub, y_col=ycol, endog_cols=["cp_flow", "cp_flow_lag"],
            instr_cols=["iv_bartik_flow", "iv_bartik_flow_lag"],
            fe_cols=["eth_year", "met_year", "eth_met"], cluster_cols=["met2013", "eth"], weight_col=wcol,
        )
        if save_contour:
            contours[lbl] = {
                "beta": res_d["coefs"],
                "se_2way": res_d["ses_2way"],
                "ar_contour": res_d["ar_contour"],
            }
        for v in ["cp_flow", "cp_flow_lag"]:
            summary_rows.append({
                "panel": panel_tag,
                "model": "JRS_Dynamic_2IV",
                "outcome": lbl,
                "variable": v,
                "coef": res_d["coefs"][v],
                "se_2way": res_d["ses_2way"][v],
                "pval_2way": res_d["pvals_2way"][v],
                "wald_ci_lo": res_d["wald_ci_95"][v][0],
                "wald_ci_hi": res_d["wald_ci_95"][v][1],
                "ar_ci_lo": res_d["ar_ci_95"][v][0],
                "ar_ci_hi": res_d["ar_ci_95"][v][1],
                "ar_type": res_d["ar_meta"][v]["ar_type"],
                "ar_ci_str": res_d["ar_meta"][v]["ar_str"],
                "ar_ci_tex": res_d["ar_meta"][v]["ar_tex"],
                "sw_f_msa": res_d["sw_f_msa_cluster"][v],
                "sw_f_2way": res_d["sw_f_2way_cluster"][v],
                "ar_zero_pval": res_d["ar_joint_zero_pval"],
                "n": res_d["n"],
            })

    # --- Panel A: Uniform 10-Year Decadal Panel (1980, 1990, 2000, 2010) ---
    # Exact 10-year lags for ln_Residents, ln_OOS_Residents, LogWage!
    for ycol, lbl, wcol in [
        ("ln_res_pop", "ln_Residents", None),
        ("ln_oos_res_pop", "ln_OOS_Residents", None),
        ("mean_ln_wage", "LogWage", "nat_pop"),
    ]:
        record_static("Decadal_1980_2010_10yr", decadal, ycol, lbl, wcol)
        record_dynamic("Decadal_1980_2010_10yr", decadal, ycol, lbl, wcol, save_contour=True)

    # Pure 5-Year Census Retention (1980, 1990, 2000 — zero 5-yr vs 1-yr mixing!)
    record_static("Census5yr_1980_2000", census_5yr, "true_retention_rate", "TrueRetention_5yr", "orig_cohort_pop")

    # --- Panel B: Modern ACS Extension Panel (2010, 2019, 2024 — uniform 1-yr migration window!) ---
    for ycol, lbl, wcol in specs:
        lbl_acs = "TrueRetention_1yr" if lbl == "TrueRetention" else lbl
        record_static("ACS_2010_2019_2024", acs_ext, ycol, lbl_acs, wcol)

    out_df = pd.DataFrame(summary_rows)
    out_df.to_csv(OUT_DIR / "module4_jrs_weak_iv_robust.csv", index=False)
    (OUT_DIR / "module4_ar_contours.json").write_text(json.dumps(contours, indent=2))
    (OUT_DIR / "module4_summary.json").write_text(json.dumps(summary_rows, indent=2, default=str))

    print("\n=== MODULE 4: UNIFORM 10-YR DECADAL JRS & SEPARATE ACS EXTENSION ===")
    for _, r in out_df.iterrows():
        sig = "***" if r["pval_2way"] < 0.01 else ("**" if r["pval_2way"] < 0.05 else ("*" if r["pval_2way"] < 0.10 else ""))
        print(
            f"  {r['panel']:23s} | {r['model']:16s} | {r['outcome']:18s} | {r['variable']:11s}: "
            f"{r['coef']:+.4f} ({r['se_2way']:.4f}){sig:3s} | "
            f"SW-F(2w)={r['sw_f_2way']:6.1f} | "
            f"AR 95% Set={r['ar_ci_str']} ({r['ar_type']}) | N={int(r['n']):,}"
        )


if __name__ == "__main__":
    main()
