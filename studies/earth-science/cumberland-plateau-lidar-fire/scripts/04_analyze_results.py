"""Phase 2C: Rigorous Statistical Analysis of 3DEP Airborne LiDAR, Multi-Epoch BACI,
Topographically Matched Dose-Response, Time-Since-Burn Trajectory, Cluster-Robust
Inference (Burn-Unit SUID & 2 km Spatial Blocks), and Head-to-Head Sensor Comparison
(3DEP LiDAR vs Landsat 8/9 Multi-Season Optical vs NASA GEDI L2B).

Reads:  data/lidar/plot_metrics.json, data/facts/*_fire.geojson, ../initial_study/burns.geojson
Writes: data/lidar/analysis_summary.json
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd
import pyproj
from scipy import stats
from scipy.spatial import cKDTree
from shapely.geometry import Point, shape
from shapely.strtree import STRtree


def bootstrap_ci(arr: np.ndarray, stat_fn=np.mean, n_boot: int = 4000, seed: int = 42) -> tuple[float, float]:
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boots = stat_fn(arr[idx], axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return round(float(lo), 5), round(float(hi), 5)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sa = np.var(a, ddof=1)
    sb = np.var(b, ddof=1)
    sp = math.sqrt(((len(a) - 1) * sa + (len(b) - 1) * sb) / max(len(a) + len(b) - 2, 1))
    return float((np.mean(a) - np.mean(b)) / sp) if sp > 0 else 0.0


def summarize_group(vals: np.ndarray) -> dict:
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return {"n": 0}
    lo, hi = bootstrap_ci(vals)
    return {
        "n": len(vals),
        "mean": round(float(np.mean(vals)), 5),
        "sd": round(float(np.std(vals, ddof=1)), 5) if len(vals) > 1 else 0.0,
        "median": round(float(np.median(vals)), 5),
        "ci95_lo": lo,
        "ci95_hi": hi,
    }


def mahalanobis_match(treated: pd.DataFrame, controls: pd.DataFrame, covs: list[str]) -> tuple[pd.DataFrame, dict]:
    """1:1 nearest-neighbor Mahalanobis matching (without replacement where possible)
    using the regularized inverse sample covariance matrix S^-1 across topographic
    covariates, returning matched controls + balance SMD table and replacement flag.
    """
    if len(treated) == 0 or len(controls) == 0:
        return controls.iloc[0:0], {}
    pool = pd.concat([treated[covs], controls[covs]], axis=0)
    mu = pool.mean()
    sd = pool.std().replace(0, 1.0)
    zt = ((treated[covs] - mu) / sd).values.astype(float)
    zc = ((controls[covs] - mu) / sd).values.astype(float)

    # True Mahalanobis inverse covariance on standardized covariates (regularized for stability)
    stacked = np.vstack([zt, zc])
    if len(stacked) > len(covs) + 1:
        S = np.cov(stacked, rowvar=False)
        if np.ndim(S) == 0:
            S = np.array([[float(S)]])
        S_inv = np.linalg.pinv(S + 1e-6 * np.eye(len(covs)))
    else:
        S_inv = np.eye(len(covs))

    smd_pre = {c: round(float((treated[c].mean() - controls[c].mean()) / sd[c]), 3) for c in covs}

    used = set()
    matched_idx = []
    allow_replace = len(controls) < len(treated)
    for row in zt:
        diff = zc - row
        dists = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", diff, S_inv, diff), 0.0))
        order = np.argsort(dists)
        picked = None
        for idx in order:
            if allow_replace or idx not in used:
                picked = int(idx)
                used.add(picked)
                break
        if picked is not None:
            matched_idx.append(picked)

    mctrl = controls.iloc[matched_idx].copy()
    smd_post = {c: round(float((treated[c].mean() - mctrl[c].mean()) / sd[c]), 3) for c in covs}
    return mctrl, {
        "smd_pre": smd_pre,
        "smd_post": smd_post,
        "n_pairs": len(matched_idx),
        "unique_controls": len(set(matched_idx)),
        "with_replacement": bool(allow_replace),
    }


def _load_fire_lookup(path: str, id_key: str, date_key: str | None, fy_key: str):
    if not os.path.exists(path):
        return None, [], []
    with open(path) as f:
        fc = json.load(f)["features"]
    geoms, meta = [], []
    for i, f in enumerate(fc):
        g = shape(f["geometry"])
        if not g.is_valid:
            g = g.buffer(0)
        if g.is_empty:
            continue
        pr = f["properties"]
        dc = pr.get(date_key) if date_key else None
        dt_iso = pd.to_datetime(dc, unit="ms").strftime("%Y-%m-%d") if isinstance(dc, (int, float)) else None
        geoms.append(g)
        meta.append({
            "uid": str(pr.get(id_key) or f"poly_{i}"),
            "date": dt_iso,
            "fy": pr.get(fy_key),
        })
    return (STRtree(geoms) if geoms else None), geoms, meta


def build_dataframe(raw: dict[str, dict]) -> pd.DataFrame:
    """Build analysis DataFrame enriched with exact-date FACTS inter-epoch burn &
    mechanical-treatment counts, burn-unit SUID cluster IDs, and 2 km / 5 km
    spatial block IDs (EPSG:5070).
    """
    tr = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    bh_fire = _load_fire_lookup("data/facts/bankhead_fire.geojson", "suid", "date_completed", "fiscal_year_completed")
    db_fire = _load_fire_lookup("data/facts/danielboone_fire.geojson", "suid", "date_completed", "fiscal_year_completed")
    sew_fire = _load_fire_lookup(os.path.join("..", "initial_study", "burns.geojson"), "FID", "Date", "Year")
    bh_mech = _load_fire_lookup("data/facts/bankhead_mech.geojson", "suid", "date_completed", "fiscal_year_completed")
    db_mech = _load_fire_lookup("data/facts/danielboone_mech.geojson", "suid", "date_completed", "fiscal_year_completed")

    rows = []
    for pid, r in raw.items():
        lp = r.get("lidar_primary")
        if not lp:
            continue
        mx, my = tr.transform(r["lon"], r["lat"])
        grid_2km = f"{r['site']}_G2_{int(mx // 2000)}_{int(my // 2000)}"
        grid_5km = f"{r['site']}_G5_{int(mx // 5000)}_{int(my // 5000)}"

        # Query authoritative fire polygons for burn-unit SUID clustering & exact-date BACI timing
        pt = Point(r["lon"], r["lat"])
        tree, geoms, meta = (
            bh_fire if "Bankhead" in r["site"]
            else (db_fire if "DanielBoone" in r["site"] else sew_fire)
        )
        hit_uids, hit_dates = [], []
        if tree is not None:
            for idx in tree.query(pt):
                if geoms[idx].contains(pt):
                    hit_uids.append(meta[idx]["uid"])
                    if meta[idx]["date"]:
                        hit_dates.append((meta[idx]["date"], meta[idx]["fy"]))

        # Query mechanical treatment polygons for exact-date inter-epoch timing audit
        mtree, mgeoms, mmeta = (
            bh_mech if "Bankhead" in r["site"]
            else (db_mech if "DanielBoone" in r["site"] else (None, [], []))
        )
        mech_dates = []
        if mtree is not None:
            for idx in mtree.query(pt):
                if mgeoms[idx].contains(pt) and mmeta[idx]["date"]:
                    mech_dates.append((mmeta[idx]["date"], mmeta[idx]["fy"]))

        if hit_uids and r["n_burns_total_pre_lidar"] > 0:
            cluster_suid = f"{r['site']}_U_{min(hit_uids)}"
        else:
            cluster_suid = grid_2km

        lb = r.get("lidar_baseline")
        acq_p = lp.get("acq_date")
        acq_b = lb.get("acq_date") if lb else None

        # Verify exact inter-epoch burn and mechanical counts using decoded LiDAR acquisition dates
        n_inter_exact = r.get("n_inter_burns")
        n_inter_mech = 0
        stratum_exact = r["stratum"]
        if lb and acq_b and acq_p:
            if hit_dates:
                inter_fys = sorted({fy for dt, fy in hit_dates if acq_b < dt <= acq_p})
                n_inter_exact = len(inter_fys)
                if not r.get("has_mech"):
                    if r["n_burns_total_pre_lidar"] == 0:
                        stratum_exact = "baci_0_control"
                    elif n_inter_exact == 0:
                        stratum_exact = "baci_pre_only"
                    elif n_inter_exact == 1:
                        stratum_exact = "baci_1_inter"
                    else:
                        stratum_exact = "baci_2plus_inter"
            if mech_dates:
                n_inter_mech = len({fy for dt, fy in mech_dates if acq_b < dt <= acq_p})

        row = {
            "plot_id": pid,
            "site": r["site"],
            "stratum_raw": r["stratum"],
            "stratum": stratum_exact,
            "lon": r["lon"],
            "lat": r["lat"],
            "mx": mx,
            "my": my,
            "cluster_suid": cluster_suid,
            "cluster_2km": grid_2km,
            "cluster_5km": grid_5km,
            "n_burns": r["n_burns_total_pre_lidar"],
            "last_burn_year": r.get("last_burn_year"),
            "years_since_burn": r.get("years_since_burn"),
            "n_inter_burns": n_inter_exact,
            "n_inter_mech": n_inter_mech,
            "has_mech": bool(r.get("has_mech", False)),
            "elev_m": r["elev_m"],
            "slope_deg": r["slope_deg"],
            "hli": r["hli"],
            "tpi_450m": r["tpi_450m"],
            "decid_purity": r["decid_purity"],
            "acq_primary": acq_p,
        }
        for k, v in lp.items():
            if k == "nrd_1m" and isinstance(v, dict):
                for bk, bv in v.items():
                    row[f"p_nrd_1m_{bk}"] = bv
            elif k != "acq_date":
                row[f"p_{k}"] = v
        if lb:
            row["acq_baseline"] = acq_b
            for k, v in lb.items():
                if k == "nrd_1m" and isinstance(v, dict):
                    lp_1m = lp.get("nrd_1m") or {}
                    for bk, bv in v.items():
                        row[f"b_nrd_1m_{bk}"] = bv
                        if bk in lp_1m and isinstance(bv, (int, float)) and isinstance(lp_1m[bk], (int, float)):
                            row[f"d_nrd_1m_{bk}"] = round(float(lp_1m[bk] - bv), 5)
                elif k != "acq_date":
                    row[f"b_{k}"] = v
                    if k in lp and isinstance(v, (int, float)) and isinstance(lp[k], (int, float)):
                        row[f"d_{k}"] = round(float(lp[k] - v), 5)
        lpt = r.get("lidar_primary_thinned")
        lbt = r.get("lidar_baseline_thinned")
        if lpt and lbt:
            for k in (
                "nrd_understory", "nrd_low_midstory", "nrd_tall_midstory", "nrd_midstory",
                "canopy_cover_10m", "first_return_canopy_10m", "fhd",
            ):
                if k in lpt and k in lbt and lpt[k] is not None and lbt[k] is not None:
                    row[f"dt_{k}"] = round(float(lpt[k] - lbt[k]), 5)
        opt = r.get("optical", {})
        for k, v in opt.items():
            row[k] = v
        if opt.get("opt_late_win_ndvi") is not None and opt.get("opt_early_win_ndvi") is not None:
            row["d_opt_win_ndvi"] = round(float(opt["opt_late_win_ndvi"] - opt["opt_early_win_ndvi"]), 5)
        if opt.get("opt_late_win_nbr") is not None and opt.get("opt_early_win_nbr") is not None:
            row["d_opt_win_nbr"] = round(float(opt["opt_late_win_nbr"] - opt["opt_early_win_nbr"]), 5)
        if opt.get("opt_late_spr_ndvi") is not None and opt.get("opt_early_spr_ndvi") is not None:
            row["d_opt_spr_ndvi"] = round(float(opt["opt_late_spr_ndvi"] - opt["opt_early_spr_ndvi"]), 5)
        if opt.get("opt_2022_24_spr_ndvi") is not None and opt.get("opt_2016_18_spr_ndvi") is not None:
            row["d_opt_spr_ndvi_db_epoch"] = round(float(opt["opt_2022_24_spr_ndvi"] - opt["opt_2016_18_spr_ndvi"]), 5)
        gedi = r.get("gedi", {})
        for k, v in gedi.items():
            row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)


def _cluster_cov(X: np.ndarray, XtX_inv: np.ndarray, resid: np.ndarray, clusters: np.ndarray) -> tuple[np.ndarray, int]:
    """Liang-Zeger / CR1 finite-sample cluster-robust covariance matrix."""
    n, k = X.shape
    uniq_c = np.unique(clusters)
    G = len(uniq_c)
    meat = np.zeros((k, k))
    for c in uniq_c:
        m = clusters == c
        sc = X[m].T @ resid[m]
        meat += np.outer(sc, sc)
    dfc = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    cov_cl = dfc * (XtX_inv @ meat @ XtX_inv)
    return cov_cl, G


def _cluster_se_p(X: np.ndarray, XtX_inv: np.ndarray, resid: np.ndarray, beta: np.ndarray, clusters: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Liang-Zeger / CR1 finite-sample cluster-robust SEs and t-test p-values (df = G - 1)."""
    cov_cl, G = _cluster_cov(X, XtX_inv, resid, clusters)
    se_cl = np.sqrt(np.maximum(np.diag(cov_cl), 1e-18))
    t_cl = beta / se_cl
    p_cl = 2.0 * (1.0 - stats.t.cdf(np.abs(t_cl), df=max(G - 1, 1)))
    return se_cl, p_cl, G


def wild_cluster_bootstrap_p(
    y: np.ndarray,
    X: np.ndarray,
    col_idx: int,
    clusters: np.ndarray,
    n_boot: int = 9999,
    seed: int = 42,
) -> float:
    """Restricted Rademacher wild cluster bootstrap (WCR-t; Cameron, Gelbach & Miller 2008)
    for testing H0: beta[col_idx] = 0 when treated arms have few clusters."""
    mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
    y_m = np.asarray(y, dtype=float)[mask]
    X_m = np.asarray(X, dtype=float)[mask]
    cl_m = np.asarray(clusters)[mask]
    n, k = X_m.shape
    if n <= k + 1:
        return float("nan")
    XtX_inv = np.linalg.pinv(X_m.T @ X_m)
    beta = XtX_inv @ (X_m.T @ y_m)
    resid = y_m - X_m @ beta
    cov_cl, G = _cluster_cov(X_m, XtX_inv, resid, cl_m)
    if G < 2:
        return float("nan")
    t_obs = float(beta[col_idx] / np.sqrt(max(float(cov_cl[col_idx, col_idx]), 1e-18)))

    keep = [j for j in range(k) if j != col_idx]
    Xr = X_m[:, keep]
    br = np.linalg.pinv(Xr.T @ Xr) @ (Xr.T @ y_m)
    yhat_r = Xr @ br
    er = y_m - yhat_r

    uniq_c, inv_idx = np.unique(cl_m, return_inverse=True)
    rng = np.random.default_rng(seed)
    proj = X_m @ XtX_inv[:, col_idx]
    dfc = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))

    t_boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        w = rng.choice((-1.0, 1.0), size=len(uniq_c))
        yb = yhat_r + er * w[inv_idx]
        bb = XtX_inv @ (X_m.T @ yb)
        rb = yb - X_m @ bb
        scores = np.bincount(inv_idx, weights=proj * rb)
        var_b = dfc * float(np.sum(scores * scores))
        t_boots[b] = float(bb[col_idx] / np.sqrt(max(var_b, 1e-18)))
    return round(float((1 + np.sum(np.abs(t_boots) >= abs(t_obs))) / (n_boot + 1)), 6)


def run_ols_hc3(
    y: np.ndarray,
    X: np.ndarray,
    col_names: list[str],
    clusters_suid: np.ndarray | None = None,
    clusters_2km: np.ndarray | None = None,
    contrasts: list[tuple[str, str, str]] | None = None,
) -> dict:
    """Ordinary least squares with White/HC3 heteroskedasticity-consistent SEs,
    optional CR1 cluster-robust SEs (burn-unit SUID and 2 km spatial blocks),
    and optional exact cluster-robust pairwise linear contrasts `(name, col_a, col_b)`
    testing `H0: beta[col_a] - beta[col_b] = 0`.
    """
    mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
    y = y[mask]
    X = X[mask]
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    # White HC3
    H_diag = np.sum((X @ XtX_inv) * X, axis=1)
    scale = (resid / np.maximum(1.0 - H_diag, 1e-6)) ** 2
    meat = X.T @ (X * scale[:, None])
    cov_hc3 = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(cov_hc3), 1e-18))
    tvals = beta / se
    df = max(n - k, 1)
    pvals = 2.0 * (1.0 - stats.t.cdf(np.abs(tvals), df=df))

    r2 = 1.0 - np.sum(resid**2) / np.sum((y - np.mean(y)) ** 2)
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - k, 1)

    cov_suid = se_suid = p_suid = g_suid = None
    if clusters_suid is not None:
        cov_suid, g_suid = _cluster_cov(X, XtX_inv, resid, np.asarray(clusters_suid)[mask])
        se_suid = np.sqrt(np.maximum(np.diag(cov_suid), 1e-18))
        p_suid = 2.0 * (1.0 - stats.t.cdf(np.abs(beta / se_suid), df=max(g_suid - 1, 1)))
    cov_2km = se_2km = p_2km = g_2km = None
    if clusters_2km is not None:
        cov_2km, g_2km = _cluster_cov(X, XtX_inv, resid, np.asarray(clusters_2km)[mask])
        se_2km = np.sqrt(np.maximum(np.diag(cov_2km), 1e-18))
        p_2km = 2.0 * (1.0 - stats.t.cdf(np.abs(beta / se_2km), df=max(g_2km - 1, 1)))

    coefs = {}
    for i, (name, b, s, t, p) in enumerate(zip(col_names, beta, se, tvals, pvals, strict=False)):
        entry = {
            "coef": round(float(b), 5),
            "se_hc3": round(float(s), 5),
            "t": round(float(t), 3),
            "p": round(float(p), 6),
        }
        if se_suid is not None:
            entry["se_cluster_suid"] = round(float(se_suid[i]), 5)
            entry["p_cluster_suid"] = round(float(p_suid[i]), 6)
        if se_2km is not None:
            entry["se_cluster_2km"] = round(float(se_2km[i]), 5)
            entry["p_cluster_2km"] = round(float(p_2km[i]), 6)
        coefs[name] = entry

    out = {"n": int(n), "r2": round(float(r2), 4), "adj_r2": round(float(adj_r2), 4), "coefs": coefs}
    if g_suid is not None:
        out["n_clusters_suid"] = int(g_suid)
    if g_2km is not None:
        out["n_clusters_2km"] = int(g_2km)

    if contrasts:
        cdict = {}
        name_to_idx = {nm: idx for idx, nm in enumerate(col_names)}
        for cname, col_a, col_b in contrasts:
            if col_a not in name_to_idx or col_b not in name_to_idx:
                continue
            cv = np.zeros(k)
            cv[name_to_idx[col_a]] = 1.0
            cv[name_to_idx[col_b]] = -1.0
            diff = float(cv @ beta)
            s_hc3 = float(np.sqrt(max(float(cv @ cov_hc3 @ cv), 1e-18)))
            t_hc3 = diff / s_hc3
            p_hc3 = float(2.0 * (1.0 - stats.t.cdf(abs(t_hc3), df=df)))
            centry = {
                "diff": round(diff, 5),
                "se_hc3": round(s_hc3, 5),
                "t_hc3": round(t_hc3, 3),
                "p_hc3": round(p_hc3, 6),
            }
            if cov_suid is not None and g_suid is not None:
                s_cl = float(np.sqrt(max(float(cv @ cov_suid @ cv), 1e-18)))
                t_cl = diff / s_cl
                p_cl = float(2.0 * (1.0 - stats.t.cdf(abs(t_cl), df=max(g_suid - 1, 1))))
                centry["se_cluster_suid"] = round(s_cl, 5)
                centry["t_cluster_suid"] = round(t_cl, 3)
                centry["p_cluster_suid"] = round(p_cl, 6)
            if cov_2km is not None and g_2km is not None:
                s_cl2 = float(np.sqrt(max(float(cv @ cov_2km @ cv), 1e-18)))
                t_cl2 = diff / s_cl2
                p_cl2 = float(2.0 * (1.0 - stats.t.cdf(abs(t_cl2), df=max(g_2km - 1, 1))))
                centry["se_cluster_2km"] = round(s_cl2, 5)
                centry["t_cluster_2km"] = round(t_cl2, 3)
                centry["p_cluster_2km"] = round(p_cl2, 6)
            cdict[cname] = centry
        out["contrasts"] = cdict
    return out


def main() -> None:
    with open("data/lidar/plot_metrics.json") as f:
        raw = json.load(f)
    df = build_dataframe(raw)
    print(f"Loaded {len(df)} completed LiDAR plots across sites: {df['site'].value_counts().to_dict()}", flush=True)
    reclassified = int((df["stratum"] != df["stratum_raw"]).sum())
    print(f"Exact-date BACI stratum verification: {reclassified} plot(s) reclassified based on decoded LiDAR acq_date", flush=True)

    # Create unified fire-frequency category across all plots (using total pre-LiDAR burns)
    def burn_cat(r):
        if r["has_mech"]:
            return "mech_plus_fire"
        nb = int(r["n_burns"])
        if nb == 0:
            return "0_burns"
        if nb == 1:
            return "1_burn"
        if nb == 2:
            return "2_burns"
        return "3plus_burns"

    df["burn_cat"] = df.apply(burn_cat, axis=1)
    covs = ["elev_m", "slope_deg", "hli", "tpi_450m"]

    summary = {
        "n_total_plots": len(df),
        "plots_by_site": df["site"].value_counts().to_dict(),
        "plots_by_burn_cat": df["burn_cat"].value_counts().to_dict(),
        "n_clusters_suid": int(df["cluster_suid"].nunique()),
        "n_clusters_2km": int(df["cluster_2km"].nunique()),
        "exact_date_baci_reclassified": reclassified,
    }

    # ---- 1. Cross-sectional burn-frequency dose-response by site & pooled ----
    print("\n=== 1. Cross-Sectional Dose-Response (True Mahalanobis Matching, S^-1) ===", flush=True)
    dose_by_site = {}
    for site_label, sub in [*list(df.groupby("site")), ("POOLED_ALL_SITES", df)]:
        sdict = {}
        ctrl = sub[sub["burn_cat"] == "0_burns"]
        for cat in ("0_burns", "1_burn", "2_burns", "3plus_burns", "mech_plus_fire"):
            g = sub[sub["burn_cat"] == cat]
            if len(g) < 3:
                continue
            entry = {"n": len(g)}
            for m in ("p_nrd_understory", "p_nrd_midstory", "p_nrd_low_midstory", "p_nrd_tall_midstory",
                      "p_first_return_canopy_10m", "p_canopy_gap_10m", "p_canopy_cover_10m", "p_chm_p95",
                      "p_fhd", "opt_win_ndvi", "opt_win_nbr", "opt_spr_ndvi", "opt_aut_ndvi",
                      "gedi_pai_0_5m", "gedi_pai_5_10m"):
                if m in g.columns:
                    entry[m] = summarize_group(g[m].values)
            if cat != "0_burns" and len(ctrl) >= 5:
                _, p_mw = stats.mannwhitneyu(g["p_nrd_midstory"].dropna(), ctrl["p_nrd_midstory"].dropna(), alternative="two-sided")
                d_val = cohens_d(g["p_nrd_midstory"].values, ctrl["p_nrd_midstory"].values)
                entry["vs_0_raw_midstory"] = {
                    "diff": round(float(g["p_nrd_midstory"].mean() - ctrl["p_nrd_midstory"].mean()), 5),
                    "pct_diff": round(100.0 * (g["p_nrd_midstory"].mean() - ctrl["p_nrd_midstory"].mean()) / ctrl["p_nrd_midstory"].mean(), 2),
                    "cohens_d": round(d_val, 3),
                    "p_mannwhitney": round(float(p_mw), 6),
                }
                # True Mahalanobis topographic matching
                mctrl, bal = mahalanobis_match(g, ctrl, covs)
                if len(mctrl) >= 3:
                    diff_m = g["p_nrd_midstory"].values - mctrl["p_nrd_midstory"].values
                    valid_m = ~np.isnan(diff_m)
                    _, p_w = stats.wilcoxon(diff_m[valid_m]) if np.any(diff_m[valid_m] != 0) else (0.0, 1.0)
                    _, p_t = stats.ttest_rel(g["p_nrd_midstory"].values[valid_m], mctrl["p_nrd_midstory"].values[valid_m])
                    lo, hi = bootstrap_ci(diff_m[valid_m])
                    entry["vs_0_matched_midstory"] = {
                        "n_pairs": bal["n_pairs"],
                        "with_replacement": bal["with_replacement"],
                        "treated_mean": round(float(np.nanmean(g["p_nrd_midstory"].values)), 5),
                        "matched_ctrl_mean": round(float(np.nanmean(mctrl["p_nrd_midstory"].values)), 5),
                        "mean_diff": round(float(np.nanmean(diff_m)), 5),
                        "pct_diff": round(100.0 * float(np.nanmean(diff_m)) / float(np.nanmean(mctrl["p_nrd_midstory"].values)), 2),
                        "ci95_lo": lo,
                        "ci95_hi": hi,
                        "cohens_d": round(cohens_d(g["p_nrd_midstory"].values, mctrl["p_nrd_midstory"].values), 3),
                        "p_wilcoxon": round(float(p_w), 6),
                        "p_paired_t": round(float(p_t), 6),
                        "balance": bal,
                    }
                    # Paired-mask safe understory & winter NDVI comparisons
                    diff_u = g["p_nrd_understory"].values - mctrl["p_nrd_understory"].values
                    valid_u = ~np.isnan(diff_u)
                    _, p_u = stats.ttest_rel(g["p_nrd_understory"].values[valid_u], mctrl["p_nrd_understory"].values[valid_u]) if valid_u.sum() >= 3 else (0.0, 1.0)

                    diff_ndvi = g["opt_win_ndvi"].values - mctrl["opt_win_ndvi"].values
                    valid_n = ~np.isnan(diff_ndvi)
                    _, p_nd = stats.ttest_rel(g["opt_win_ndvi"].values[valid_n], mctrl["opt_win_ndvi"].values[valid_n]) if valid_n.sum() >= 3 else (0.0, 1.0)

                    entry["vs_0_matched_understory"] = {
                        "mean_diff": round(float(np.nanmean(diff_u)), 5),
                        "pct_diff": round(100.0 * float(np.nanmean(diff_u)) / float(np.nanmean(mctrl["p_nrd_understory"].values[valid_u])), 2),
                        "p_paired_t": round(float(p_u), 6),
                    }
                    entry["vs_0_matched_win_ndvi"] = {
                        "mean_diff": round(float(np.nanmean(diff_ndvi)), 5),
                        "cohens_d": round(cohens_d(g["opt_win_ndvi"].values[valid_n], mctrl["opt_win_ndvi"].values[valid_n]), 3),
                        "p_paired_t": round(float(p_nd), 6),
                    }
            sdict[cat] = entry
        dose_by_site[site_label] = sdict
        print(f"\n  Site: {site_label}", flush=True)
        for cat, e in sdict.items():
            mid = e["p_nrd_midstory"]["mean"]
            und = e["p_nrd_understory"]["mean"]
            ndv = e["opt_win_ndvi"]["mean"] if e["opt_win_ndvi"].get("n", 0) > 0 else float("nan")
            mstr = ""
            if "vs_0_matched_midstory" in e:
                vm = e["vs_0_matched_midstory"]
                rep_tag = " [w/repl]" if vm["with_replacement"] else ""
                mstr = f" | matched Δmid={vm['mean_diff']:+.4f} ({vm['pct_diff']:+.1f}%, p={vm['p_paired_t']:.4f}, d={vm['cohens_d']:+.2f}){rep_tag}"
            print(f"    {cat:15s} n={e['n']:3d} | NRD_mid={mid:.4f} NRD_und={und:.4f} winNDVI={ndv:.4f}{mstr}", flush=True)

    summary["dose_response_by_site"] = dose_by_site

    # ---- 2. Pooled Multi-Forest Fixed-Effects OLS (HC3 + Burn-Unit SUID + 2 km Spatial Clusters) ----
    print("\n=== 2. Pooled Multi-Forest Fixed-Effects Regressions (HC3 + SUID & 2 km Cluster SEs) ===", flush=True)
    sites_list = sorted(df["site"].unique())
    X_cols = ["intercept", "burn_1", "burn_2", "burn_3plus", "mech_plus_fire", "elev_z", "slope_z", "hli_z", "tpi_z", "dens_z"]
    for s in sites_list[1:]:
        X_cols.append(f"site_{s}")

    def make_X(frame: pd.DataFrame, site_names: list[str]) -> np.ndarray:
        cols = [
            np.ones(len(frame)),
            (frame["burn_cat"] == "1_burn").astype(float).values,
            (frame["burn_cat"] == "2_burns").astype(float).values,
            (frame["burn_cat"] == "3plus_burns").astype(float).values,
            (frame["burn_cat"] == "mech_plus_fire").astype(float).values,
            ((frame["elev_m"] - frame["elev_m"].mean()) / frame["elev_m"].std()).values,
            ((frame["slope_deg"] - frame["slope_deg"].mean()) / frame["slope_deg"].std()).values,
            ((frame["hli"] - frame["hli"].mean()) / frame["hli"].std()).values,
            ((frame["tpi_450m"] - frame["tpi_450m"].mean()) / frame["tpi_450m"].std()).values,
            ((frame["p_pulse_density"] - frame["p_pulse_density"].mean()) / frame["p_pulse_density"].std()).values,
        ]
        for s in site_names[1:]:
            cols.append((frame["site"] == s).astype(float).values)
        return np.column_stack(cols)

    X_all = make_X(df, sites_list)
    std_contrasts = [
        ("mech_minus_3plus", "mech_plus_fire", "burn_3plus"),
        ("burn_3plus_minus_1", "burn_3plus", "burn_1"),
    ]
    ols_models = {}
    for target in (
        "p_nrd_midstory", "p_nrd_understory", "p_nrd_low_midstory", "p_nrd_tall_midstory",
        "p_first_return_canopy_10m", "p_canopy_gap_10m", "p_canopy_cover_10m",
        "p_chm_p95", "p_fhd", "opt_win_ndvi", "opt_win_nbr", "opt_spr_ndvi", "opt_aut_ndvi",
    ):
        if target in df.columns and df[target].notna().sum() >= 20:
            res = run_ols_hc3(
                df[target].values.astype(float),
                X_all,
                X_cols,
                clusters_suid=df["cluster_suid"].values,
                clusters_2km=df["cluster_2km"].values,
                contrasts=std_contrasts,
            )
            ols_models[target] = res
            c = res["coefs"]
            ct = res["contrasts"]["mech_minus_3plus"]
            print(
                f"  OLS {target:25s} (R2={res['r2']:.3f}): "
                f"1b={c['burn_1']['coef']:+.4f}(p={c['burn_1']['p_cluster_suid']:.3f}) "
                f"2b={c['burn_2']['coef']:+.4f}(p={c['burn_2']['p_cluster_suid']:.3f}) "
                f"3+b={c['burn_3plus']['coef']:+.4f}(p={c['burn_3plus']['p_cluster_suid']:.3f}) "
                f"mech={c['mech_plus_fire']['coef']:+.4f}(p={c['mech_plus_fire']['p_cluster_suid']:.3f}) | "
                f"mech-3+={ct['diff']:+.4f}(p_suid={ct['p_cluster_suid']:.4f})",
                flush=True,
            )
    summary["pooled_fe_ols"] = ols_models

    # Continuous 1 m vertical height profile (0.5-10 m) FE-OLS with burn-unit clustering
    print("\n=== 2b. Continuous 1 m Vertical NRD Profile FE-OLS (0.5-10 m) ===", flush=True)
    profile_1m = {}
    fine_labels = ["0.5_1", "1_2", "2_3", "3_4", "4_5", "5_6", "6_7", "7_8", "8_9", "9_10"]
    for lbl in fine_labels:
        col = f"p_nrd_1m_{lbl}"
        if col in df.columns and df[col].notna().sum() >= 20:
            res1 = run_ols_hc3(
                df[col].values.astype(float),
                X_all,
                X_cols,
                clusters_suid=df["cluster_suid"].values,
                clusters_2km=df["cluster_2km"].values,
                contrasts=std_contrasts,
            )
            profile_1m[lbl] = res1
            c = res1["coefs"]
            ct = res1["contrasts"]["mech_minus_3plus"]
            print(
                f"  1m [{lbl:5s} m]: "
                f"1b={c['burn_1']['coef']:+.4f} 2b={c['burn_2']['coef']:+.4f} "
                f"3+b={c['burn_3plus']['coef']:+.4f}(p={c['burn_3plus']['p_cluster_suid']:.3f}) "
                f"mech={c['mech_plus_fire']['coef']:+.4f}(p={c['mech_plus_fire']['p_cluster_suid']:.3f}) | "
                f"mech-3+={ct['diff']:+.4f}(p_suid={ct['p_cluster_suid']:.3f})",
                flush=True,
            )
    summary["pooled_fe_ols_1m_profile"] = profile_1m

    # Cross-site spatial non-overlap sensitivity check (drop any cross-site plot within 220 m of a BACI plot)
    coords = df[["mx", "my"]].values
    tree_xy = cKDTree(coords)
    drop_idx = set()
    for i, j in tree_xy.query_pairs(r=220.0):
        if df.iloc[i]["site"] != df.iloc[j]["site"]:
            # Prefer retaining BACI plot over cross-sectional plot
            if "BACI" in df.iloc[i]["site"]:
                drop_idx.add(j)
            else:
                drop_idx.add(i)
    df_no_ov = df.drop(df.index[list(drop_idx)]).reset_index(drop=True)
    X_no_ov = make_X(df_no_ov, sites_list)
    sens_no_ov = {"n_dropped_cross_site_neighbors": len(drop_idx), "n_remaining": len(df_no_ov), "models": {}}
    for target in ("p_nrd_midstory", "p_nrd_low_midstory", "p_nrd_tall_midstory", "p_first_return_canopy_10m", "p_canopy_cover_10m"):
        if target in df_no_ov.columns and df_no_ov[target].notna().sum() >= 20:
            sens_no_ov["models"][target] = run_ols_hc3(
                df_no_ov[target].values.astype(float),
                X_no_ov,
                X_cols,
                clusters_suid=df_no_ov["cluster_suid"].values,
                clusters_2km=df_no_ov["cluster_2km"].values,
                contrasts=std_contrasts,
            )
    summary["sensitivity_no_cross_overlap"] = sens_no_ov
    print(f"  Sensitivity (drop {len(drop_idx)} cross-site <220m neighbors, n={len(df_no_ov)}): "
          f"3+b mid={sens_no_ov['models']['p_nrd_midstory']['coefs']['burn_3plus']['coef']:+.4f} "
          f"(p_suid={sens_no_ov['models']['p_nrd_midstory']['coefs']['burn_3plus']['p_cluster_suid']:.4f}), "
          f"mech 5-10m={sens_no_ov['models']['p_nrd_tall_midstory']['coefs']['mech_plus_fire']['coef']:+.4f} "
          f"(p_suid={sens_no_ov['models']['p_nrd_tall_midstory']['coefs']['mech_plus_fire']['p_cluster_suid']:.4f})",
          flush=True)

    # ---- 3. TRUE BACI Multi-Epoch Airborne LiDAR Differencing (Exact Dates + Cluster SEs) ----
    print("\n=== 3. TRUE BACI Multi-Epoch Airborne LiDAR Differencing (Exact-Date & Epoch-Verified) ===", flush=True)
    baci_results = {}
    baci_specs = [
        ("Bankhead_North_BACI", "2012", "2021"),
        ("DanielBoone_East_BACI", "2017", "2023"),
    ]
    for site, exp_b_yr, exp_p_yr in baci_specs:
        sub_all = df[(df["site"] == site) & df["d_nrd_midstory"].notna()].copy()
        if len(sub_all) == 0:
            continue
        # Enforce uniform inter-epoch duration by requiring baseline acquisition year == expected epoch
        valid_epoch = sub_all["acq_baseline"].fillna("").str.startswith(exp_b_yr) & sub_all["acq_primary"].fillna("").str.startswith(exp_p_yr)
        excluded_plots = sub_all.loc[~valid_epoch, ["plot_id", "stratum", "acq_baseline", "acq_primary"]].to_dict(orient="records")
        sub = sub_all[valid_epoch].copy()

        ctrl = sub[sub["stratum"] == "baci_0_control"]
        # Fit cluster-robust DiD OLS on ΔNRD_midstory across strata (ref = baci_0_control)
        active_strata = [st for st in ("baci_pre_only", "baci_1_inter", "baci_2plus_inter", "baci_mech_fire") if (sub["stratum"] == st).sum() >= 3]
        Xb = np.column_stack([np.ones(len(sub))] + [(sub["stratum"] == st).astype(float).values for st in active_strata])
        baci_ols = run_ols_hc3(
            sub["d_nrd_midstory"].values.astype(float),
            Xb,
            ["intercept", *active_strata],
            clusters_suid=sub["cluster_suid"].values,
            clusters_2km=sub["cluster_2km"].values,
        )
        # Baseline-adjusted ANCOVA DiD sensitivity check (controls for differing baseline midstory levels)
        Xb_anc = np.column_stack([Xb, (sub["b_nrd_midstory"] - sub["b_nrd_midstory"].mean()).values])
        baci_ancova = run_ols_hc3(
            sub["d_nrd_midstory"].values.astype(float),
            Xb_anc,
            ["intercept", *active_strata, "b_nrd_midstory_c"],
            clusters_suid=sub["cluster_suid"].values,
            clusters_2km=sub["cluster_2km"].values,
        )
        # Endpoint-level (Epoch 2) and baseline-level (Epoch 1) cluster-robust OLS vs control
        baci_base_ols = run_ols_hc3(
            sub["b_nrd_midstory"].values.astype(float),
            Xb,
            ["intercept", *active_strata],
            clusters_suid=sub["cluster_suid"].values,
            clusters_2km=sub["cluster_2km"].values,
        )
        baci_end_ols = run_ols_hc3(
            sub["p_nrd_midstory"].values.astype(float),
            Xb,
            ["intercept", *active_strata],
            clusters_suid=sub["cluster_suid"].values,
            clusters_2km=sub["cluster_2km"].values,
        )
        baci_spr_ols = None
        if "d_opt_spr_ndvi" in sub.columns and sub["d_opt_spr_ndvi"].notna().sum() >= 20:
            baci_spr_ols = run_ols_hc3(
                sub["d_opt_spr_ndvi"].values.astype(float),
                Xb,
                ["intercept", *active_strata],
                clusters_suid=sub["cluster_suid"].values,
                clusters_2km=sub["cluster_2km"].values,
            )
        baci_spr_db_ols = None
        if "d_opt_spr_ndvi_db_epoch" in sub.columns and sub["d_opt_spr_ndvi_db_epoch"].notna().sum() >= 20:
            baci_spr_db_ols = run_ols_hc3(
                sub["d_opt_spr_ndvi_db_epoch"].values.astype(float),
                Xb,
                ["intercept", *active_strata],
                clusters_suid=sub["cluster_suid"].values,
                clusters_2km=sub["cluster_2km"].values,
            )
        baci_frc_ols = baci_frc_base_ols = baci_frc_end_ols = baci_frc_anc_ols = None
        if "d_first_return_canopy_10m" in sub.columns and sub["d_first_return_canopy_10m"].notna().sum() >= 20:
            baci_frc_ols = run_ols_hc3(
                sub["d_first_return_canopy_10m"].values.astype(float),
                Xb,
                ["intercept", *active_strata],
                clusters_suid=sub["cluster_suid"].values,
                clusters_2km=sub["cluster_2km"].values,
            )
            baci_frc_base_ols = run_ols_hc3(
                sub["b_first_return_canopy_10m"].values.astype(float),
                Xb,
                ["intercept", *active_strata],
                clusters_suid=sub["cluster_suid"].values,
                clusters_2km=sub["cluster_2km"].values,
            )
            baci_frc_end_ols = run_ols_hc3(
                sub["p_first_return_canopy_10m"].values.astype(float),
                Xb,
                ["intercept", *active_strata],
                clusters_suid=sub["cluster_suid"].values,
                clusters_2km=sub["cluster_2km"].values,
            )
            Xb_anc_frc = np.column_stack([Xb, (sub["b_first_return_canopy_10m"] - sub["b_first_return_canopy_10m"].mean()).values])
            baci_frc_anc_ols = run_ols_hc3(
                sub["d_first_return_canopy_10m"].values.astype(float),
                Xb_anc_frc,
                ["intercept", *active_strata, "b_frc_c"],
                clusters_suid=sub["cluster_suid"].values,
                clusters_2km=sub["cluster_2km"].values,
            )

        # Small-cluster-robust restricted wild cluster bootstrap (WCR-t, Rademacher, B=9999)
        wcb_p_suid = {}
        for s_idx, st_name in enumerate(active_strata, start=1):
            wcb_p_suid[st_name] = wild_cluster_bootstrap_p(
                sub["d_nrd_midstory"].values.astype(float),
                Xb,
                s_idx,
                sub["cluster_suid"].values,
                n_boot=9999,
                seed=42 + s_idx,
            )

        flight_diag = {
            "baseline_mean_pulse_density": round(float(sub["b_pulse_density"].mean()), 3),
            "primary_mean_pulse_density": round(float(sub["p_pulse_density"].mean()), 3),
            "baseline_mean_frac_first_returns": round(float(sub["b_frac_first_returns"].mean()), 4) if "b_frac_first_returns" in sub else None,
            "primary_mean_frac_first_returns": round(float(sub["p_frac_first_returns"].mean()), 4) if "p_frac_first_returns" in sub else None,
            "baseline_acq_dates": sorted(sub["acq_baseline"].dropna().unique().tolist())[:5],
            "primary_acq_dates": sorted(sub["acq_primary"].dropna().unique().tolist())[:5],
        }

        sres = {
            "n_total": len(sub),
            "n_excluded_off_epoch": len(excluded_plots),
            "excluded_off_epoch": excluded_plots,
            "n_clusters_suid": baci_ols.get("n_clusters_suid"),
            "n_clusters_2km": baci_ols.get("n_clusters_2km"),
            "flight_diagnostics": flight_diag,
            "strata": {},
        }
        print(
            f"\n  BACI Site: {site} (n={len(sub)}, excluded off-epoch={len(excluded_plots)}, "
            f"G_suid={sres['n_clusters_suid']}, G_2km={sres['n_clusters_2km']}, "
            f"frac_first: {flight_diag['baseline_mean_frac_first_returns']} -> {flight_diag['primary_mean_frac_first_returns']})",
            flush=True,
        )
        for st in ("baci_0_control", "baci_pre_only", "baci_1_inter", "baci_2plus_inter", "baci_mech_fire"):
            g = sub[sub["stratum"] == st]
            if len(g) < 3:
                continue
            suid_counts = g["cluster_suid"].value_counts()
            entry = {
                "n": len(g),
                "n_clusters_suid": len(suid_counts),
                "top_suid_share": round(float(suid_counts.iloc[0] / len(g)), 3),
                "n_inter_mech_plots": int((g["n_inter_mech"] > 0).sum()),
                "baseline_nrd_midstory": summarize_group(g["b_nrd_midstory"].values),
                "primary_nrd_midstory": summarize_group(g["p_nrd_midstory"].values),
                "baseline_nrd_understory": summarize_group(g["b_nrd_understory"].values),
                "primary_nrd_understory": summarize_group(g["p_nrd_understory"].values),
                "baseline_first_return_canopy_10m": summarize_group(g["b_first_return_canopy_10m"].values) if "b_first_return_canopy_10m" in g else {},
                "primary_first_return_canopy_10m": summarize_group(g["p_first_return_canopy_10m"].values) if "p_first_return_canopy_10m" in g else {},
                "delta_nrd_midstory": summarize_group(g["d_nrd_midstory"].values),
                "delta_nrd_midstory_thinned": summarize_group(g["dt_nrd_midstory"].values) if "dt_nrd_midstory" in g else {},
                "delta_nrd_understory": summarize_group(g["d_nrd_understory"].values),
                "delta_nrd_understory_thinned": summarize_group(g["dt_nrd_understory"].values) if "dt_nrd_understory" in g else {},
                "delta_nrd_low_midstory": summarize_group(g["d_nrd_low_midstory"].values),
                "delta_nrd_tall_midstory": summarize_group(g["d_nrd_tall_midstory"].values),
                "delta_first_return_canopy_10m": summarize_group(g["d_first_return_canopy_10m"].values) if "d_first_return_canopy_10m" in g else {},
                "delta_canopy_10m": summarize_group(g["d_canopy_cover_10m"].values),
                "delta_chm_p95": summarize_group(g["d_chm_p95"].values),
                "delta_fhd": summarize_group(g["d_fhd"].values),
                "delta_opt_win_ndvi": summarize_group(g["d_opt_win_ndvi"].values) if "d_opt_win_ndvi" in g else {},
                "delta_opt_win_nbr": summarize_group(g["d_opt_win_nbr"].values) if "d_opt_win_nbr" in g else {},
                "delta_opt_spr_ndvi": summarize_group(g["d_opt_spr_ndvi"].values) if "d_opt_spr_ndvi" in g else {},
            }
            _, p_zero = stats.ttest_1samp(g["d_nrd_midstory"].dropna(), 0.0)
            entry["delta_nrd_midstory"]["p_vs_zero"] = round(float(p_zero), 6)
            if "dt_nrd_midstory" in g and g["dt_nrd_midstory"].notna().sum() >= 3:
                _, p_zero_t = stats.ttest_1samp(g["dt_nrd_midstory"].dropna(), 0.0)
                entry["delta_nrd_midstory_thinned"]["p_vs_zero"] = round(float(p_zero_t), 6)

            if st != "baci_0_control" and len(ctrl) >= 5:
                did = float(g["d_nrd_midstory"].mean() - ctrl["d_nrd_midstory"].mean())
                _, p_did_t = stats.ttest_ind(g["d_nrd_midstory"].dropna(), ctrl["d_nrd_midstory"].dropna(), equal_var=False)
                _, p_did_mw = stats.mannwhitneyu(g["d_nrd_midstory"].dropna(), ctrl["d_nrd_midstory"].dropna())
                d_did = cohens_d(g["d_nrd_midstory"].values, ctrl["d_nrd_midstory"].values)

                did_thin = float(g["dt_nrd_midstory"].mean() - ctrl["dt_nrd_midstory"].mean())
                _, p_did_thin = stats.ttest_ind(g["dt_nrd_midstory"].dropna(), ctrl["dt_nrd_midstory"].dropna(), equal_var=False)

                mctrl, bal = mahalanobis_match(g, ctrl, covs)
                did_matched = float(np.mean(g["d_nrd_midstory"].values - mctrl["d_nrd_midstory"].values))
                _, p_did_matched = stats.ttest_rel(g["d_nrd_midstory"].values, mctrl["d_nrd_midstory"].values)

                did_ndvi = float(g["d_opt_win_ndvi"].mean() - ctrl["d_opt_win_ndvi"].mean())
                _, p_did_ndvi = stats.ttest_ind(g["d_opt_win_ndvi"].dropna(), ctrl["d_opt_win_ndvi"].dropna(), equal_var=False)

                cl_coef = baci_ols["coefs"][st]
                anc_coef = baci_ancova["coefs"][st]
                base_coef = baci_base_ols["coefs"][st]
                end_coef = baci_end_ols["coefs"][st]
                _, p_base_w = stats.ttest_ind(g["b_nrd_midstory"].dropna(), ctrl["b_nrd_midstory"].dropna(), equal_var=False)
                _, p_end_w = stats.ttest_ind(g["p_nrd_midstory"].dropna(), ctrl["p_nrd_midstory"].dropna(), equal_var=False)
                _, p_end_und_w = stats.ttest_ind(g["p_nrd_understory"].dropna(), ctrl["p_nrd_understory"].dropna(), equal_var=False)

                did_entry = {
                    "did_midstory": round(did, 5),
                    "cohens_d": round(d_did, 3),
                    "p_welch": round(float(p_did_t), 6),
                    "p_mannwhitney": round(float(p_did_mw), 6),
                    "se_cluster_suid": cl_coef["se_cluster_suid"],
                    "p_cluster_suid": cl_coef["p_cluster_suid"],
                    "p_wcb_suid": wcb_p_suid.get(st),
                    "se_cluster_2km": cl_coef["se_cluster_2km"],
                    "p_cluster_2km": cl_coef["p_cluster_2km"],
                    "did_midstory_ancova": anc_coef["coef"],
                    "p_ancova_suid": anc_coef["p_cluster_suid"],
                    "baseline_diff_vs_ctrl": base_coef["coef"],
                    "p_baseline_welch": round(float(p_base_w), 6),
                    "p_baseline_suid": base_coef["p_cluster_suid"],
                    "endpoint_diff_vs_ctrl": end_coef["coef"],
                    "p_endpoint_welch": round(float(p_end_w), 6),
                    "p_endpoint_suid": end_coef["p_cluster_suid"],
                    "endpoint_understory_diff_vs_ctrl": round(float(g["p_nrd_understory"].mean() - ctrl["p_nrd_understory"].mean()), 5),
                    "p_endpoint_understory_welch": round(float(p_end_und_w), 6),
                    "did_midstory_thinned": round(did_thin, 5),
                    "p_welch_thinned": round(float(p_did_thin), 6),
                    "did_midstory_matched": round(did_matched, 5),
                    "p_matched_t": round(float(p_did_matched), 6),
                    "did_opt_win_ndvi": round(did_ndvi, 5),
                    "p_opt_win_ndvi": round(float(p_did_ndvi), 6),
                    "balance": bal,
                }
                if baci_spr_ols is not None and st in baci_spr_ols["coefs"]:
                    did_spr = float(g["d_opt_spr_ndvi"].mean() - ctrl["d_opt_spr_ndvi"].mean())
                    _, p_did_spr = stats.ttest_ind(g["d_opt_spr_ndvi"].dropna(), ctrl["d_opt_spr_ndvi"].dropna(), equal_var=False)
                    did_entry["did_opt_spr_ndvi"] = round(did_spr, 5)
                    did_entry["p_opt_spr_ndvi_welch"] = round(float(p_did_spr), 6)
                    did_entry["p_opt_spr_ndvi_suid"] = baci_spr_ols["coefs"][st]["p_cluster_suid"]
                if baci_spr_db_ols is not None and st in baci_spr_db_ols["coefs"]:
                    did_entry["did_opt_spr_ndvi_db_epoch"] = baci_spr_db_ols["coefs"][st]["coef"]
                    did_entry["p_opt_spr_ndvi_db_epoch_suid"] = baci_spr_db_ols["coefs"][st]["p_cluster_suid"]
                if baci_frc_ols is not None and st in baci_frc_ols["coefs"]:
                    did_entry["did_first_return_canopy_10m"] = baci_frc_ols["coefs"][st]["coef"]
                    did_entry["p_first_return_canopy_suid"] = baci_frc_ols["coefs"][st]["p_cluster_suid"]
                    if baci_frc_base_ols is not None and st in baci_frc_base_ols["coefs"]:
                        did_entry["baseline_frc_diff_vs_ctrl"] = baci_frc_base_ols["coefs"][st]["coef"]
                        did_entry["p_baseline_frc_suid"] = baci_frc_base_ols["coefs"][st]["p_cluster_suid"]
                    if baci_frc_end_ols is not None and st in baci_frc_end_ols["coefs"]:
                        did_entry["endpoint_frc_diff_vs_ctrl"] = baci_frc_end_ols["coefs"][st]["coef"]
                        did_entry["p_endpoint_frc_suid"] = baci_frc_end_ols["coefs"][st]["p_cluster_suid"]
                    if baci_frc_anc_ols is not None and st in baci_frc_anc_ols["coefs"]:
                        did_entry["did_frc_ancova"] = baci_frc_anc_ols["coefs"][st]["coef"]
                        did_entry["p_frc_ancova_suid"] = baci_frc_anc_ols["coefs"][st]["p_cluster_suid"]
                entry["did_vs_control"] = did_entry
            sres["strata"][st] = entry
            dmid = entry["delta_nrd_midstory"]["mean"]
            dmid_t = entry["delta_nrd_midstory_thinned"].get("mean", float("nan"))
            dund = entry["delta_nrd_understory"]["mean"]
            did_str = ""
            if "did_vs_control" in entry:
                dc = entry["did_vs_control"]
                did_str = (
                    f" | DiD={dc['did_midstory']:+.4f} (p_welch={dc['p_welch']:.4f}, "
                    f"p_suid={dc['p_cluster_suid']:.4f}, p_wcb={dc.get('p_wcb_suid', float('nan')):.4f}, "
                    f"p_2km={dc['p_cluster_2km']:.4f}, "
                    f"d={dc['cohens_d']:+.2f}; thin={dc['did_midstory_thinned']:+.4f}; "
                    f"ancova={dc['did_midstory_ancova']:+.4f} p={dc['p_ancova_suid']:.3f}; "
                    f"end_vs_ctrl={dc['endpoint_diff_vs_ctrl']:+.4f} p={dc['p_endpoint_welch']:.3f})"
                )
            print(f"    {st:18s} n={entry['n']:3d} (G={entry['n_clusters_suid']:2d}, inter_mech={entry['n_inter_mech_plots']}) | Δmid={dmid:+.4f} (thin={dmid_t:+.4f}) Δund={dund:+.4f}{did_str}", flush=True)
        baci_results[site] = sres
    summary["baci"] = baci_results

    # ---- 4. Time-Since-Burn Trajectory (Fire-Only) ----
    print("\n=== 4. Time-Since-Burn Trajectory (Fire-Only) ===", flush=True)
    fire_only = df[(~df["has_mech"]) & (df["n_burns"] >= 1) & (df["years_since_burn"].notna())].copy()
    tsb_out = {}
    for freq_lbl, fsub in [("single_burn", fire_only[fire_only["n_burns"] == 1]),
                           ("multi_burn_2plus", fire_only[fire_only["n_burns"] >= 2])]:
        bins = [
            ("0_1yr", fsub[fsub["years_since_burn"] <= 1]),
            ("2_3yr", fsub[(fsub["years_since_burn"] >= 2) & (fsub["years_since_burn"] <= 3)]),
            ("4plus_yr", fsub[fsub["years_since_burn"] >= 4]),
        ]
        tsb_out[freq_lbl] = {}
        for blbl, bdf in bins:
            if len(bdf) < 3:
                continue
            tsb_out[freq_lbl][blbl] = {
                "n": len(bdf),
                "nrd_understory": summarize_group(bdf["p_nrd_understory"].values),
                "nrd_midstory": summarize_group(bdf["p_nrd_midstory"].values),
                "opt_win_ndvi": summarize_group(bdf["opt_win_ndvi"].values),
            }
            print(
                f"  {freq_lbl:16s} {blbl:9s} n={len(bdf):3d} | NRD_und={bdf['p_nrd_understory'].mean():.4f} "
                f"NRD_mid={bdf['p_nrd_midstory'].mean():.4f} winNDVI={bdf['opt_win_ndvi'].mean():.4f}",
                flush=True,
            )
    summary["time_since_burn"] = tsb_out

    # ---- 5. Head-to-Head Sensor Comparison (3DEP LiDAR vs Optical Multi-Season vs GEDI L2B) ----
    print("\n=== 5. Head-to-Head Sensor Correlation with True 3DEP LiDAR Midstory (2-10 m) ===", flush=True)
    sensor_corr = {}
    cand_sensors = [
        "opt_win_ndvi", "opt_win_nbr", "opt_win_ndmi",
        "opt_spr_ndvi", "opt_sum_ndvi", "opt_aut_ndvi",
        "gedi_pai_0_5m", "gedi_pai_5_10m", "gedi_pai_total", "gedi_fhd",
    ]
    for col in cand_sensors:
        if col not in df.columns:
            continue
        valid = df[["site", "p_nrd_midstory", "p_nrd_understory", col]].dropna()
        if len(valid) < 10:
            continue
        r_mid, p_mid = stats.pearsonr(valid["p_nrd_midstory"], valid[col])
        rho_mid, _ = stats.spearmanr(valid["p_nrd_midstory"], valid[col])
        r_und, p_und = stats.pearsonr(valid["p_nrd_understory"], valid[col])
        # Within-site (site-demeaned) correlation removes between-site confounding
        dm_x = valid["p_nrd_midstory"] - valid.groupby("site")["p_nrd_midstory"].transform("mean")
        dm_y = valid[col] - valid.groupby("site")[col].transform("mean")
        r_within, p_within = stats.pearsonr(dm_x, dm_y)
        by_site = {}
        for sname, sg in valid.groupby("site"):
            if len(sg) >= 10:
                rs, ps = stats.pearsonr(sg["p_nrd_midstory"], sg[col])
                by_site[sname] = {"n": len(sg), "pearson_r": round(float(rs), 4), "p": round(float(ps), 6)}
        sensor_corr[col] = {
            "n": len(valid),
            "pearson_r_midstory": round(float(r_mid), 4),
            "r2_midstory": round(float(r_mid**2), 4),
            "p_midstory": round(float(p_mid), 6),
            "within_site_r_midstory": round(float(r_within), 4),
            "within_site_r2_midstory": round(float(r_within**2), 4),
            "within_site_p_midstory": round(float(p_within), 6),
            "by_site_r_midstory": by_site,
            "spearman_rho_midstory": round(float(rho_mid), 4),
            "pearson_r_understory": round(float(r_und), 4),
            "p_understory": round(float(p_und), 6),
        }
        print(
            f"  {col:16s} (n={len(valid):3d}): pooled r={r_mid:+.4f} | within-site r={r_within:+.4f} "
            f"(p={p_within:.3g}) | rho={rho_mid:+.4f} | r_und={r_und:+.4f}",
            flush=True,
        )

    # BACI delta correlation strictly over the epoch-verified BACI plots (excluding the 2 2012-12 DB plots)
    valid_baci_mask = (
        ((df["site"] == "Bankhead_North_BACI") & df["acq_baseline"].fillna("").str.startswith("2012"))
        | ((df["site"] == "DanielBoone_East_BACI") & df["acq_baseline"].fillna("").str.startswith("2017"))
    )
    baci_df = df[valid_baci_mask & df["d_nrd_midstory"].notna()]
    baci_corr = {}
    if len(baci_df) >= 10:
        for ocol in ("d_opt_win_ndvi", "d_opt_win_nbr", "d_opt_spr_ndvi", "d_opt_spr_ndvi_db_epoch"):
            if ocol not in baci_df.columns:
                continue
            sub_c = baci_df[["site", "d_nrd_midstory", "d_nrd_understory", ocol]].dropna()
            if len(sub_c) < 10:
                continue
            rm, pm = stats.pearsonr(sub_c["d_nrd_midstory"], sub_c[ocol])
            rhom, _ = stats.spearmanr(sub_c["d_nrd_midstory"], sub_c[ocol])
            dm_x = sub_c["d_nrd_midstory"] - sub_c.groupby("site")["d_nrd_midstory"].transform("mean")
            dm_y = sub_c[ocol] - sub_c.groupby("site")[ocol].transform("mean")
            rw, pw = stats.pearsonr(dm_x, dm_y)
            by_site_b = {}
            for sname, sg in sub_c.groupby("site"):
                if len(sg) >= 10:
                    rs, ps = stats.pearsonr(sg["d_nrd_midstory"], sg[ocol])
                    by_site_b[sname] = {"n": len(sg), "pearson_r": round(float(rs), 4), "p": round(float(ps), 6)}
            baci_corr[ocol] = {
                "n": len(sub_c),
                "pearson_r_d_midstory": round(float(rm), 4),
                "r2_d_midstory": round(float(rm**2), 4),
                "p_d_midstory": round(float(pm), 6),
                "within_site_r_d_midstory": round(float(rw), 4),
                "within_site_p_d_midstory": round(float(pw), 6),
                "by_site": by_site_b,
                "spearman_rho_d_midstory": round(float(rhom), 4),
            }
            print(
                f"  BACI Δ-corr {ocol:23s} vs ΔNRD_midstory (n={len(sub_c)}): "
                f"pooled r={rm:+.4f} (p={pm:.3g}) | within-site r={rw:+.4f} (p={pw:.3g}) | by_site={by_site_b}",
                flush=True,
            )

    summary["sensor_comparison"] = {"cross_sectional": sensor_corr, "baci_deltas": baci_corr}

    out_path = "data/lidar/analysis_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
