"""Shared utilities for Stage 3 profiling experiments (P1-P6)."""
import os
import sys
import json
import math
import urllib.request
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import least_squares, minimize
from skimage.morphology import reconstruction

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.dirname(HERE)
CACHE_DIR = os.path.join(HERE, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

if STUDY_DIR not in sys.path:
    sys.path.insert(0, STUDY_DIR)

import ee
import eslib as L

PROJECT = L.PROJECT


def init_ee():
    ee.Initialize(project=PROJECT)


def get_domain():
    return L.domain()


def get_dem_3dep():
    """3DEP 10m with explicit default projection (avoids mosaic projection-strip bug)."""
    col = ee.ImageCollection('USGS/3DEP/10m_collection')
    proj = col.first().projection()
    return col.mosaic().setDefaultProjection(proj).select('elevation')


def get_nlcd_2021():
    return (ee.ImageCollection('USGS/NLCD_RELEASES/2021_REL/NLCD')
            .filter(ee.Filter.eq('system:index', '2021'))
            .first().select('landcover'))


def get_evt():
    return ee.ImageCollection('LANDFIRE/Vegetation/EVT/v1_4_0').filterBounds(
        ee.Geometry.Point([-85.92, 35.15])
    ).first().select('EVT')


def priority_flood_fill(dem, nodata_mask=None):
    """Exact 8-connected depression filling via morphological reconstruction by erosion.
    Boundary pixels (and nodata edges) act as drainage outlets."""
    arr = np.asarray(dem, dtype=np.float64)
    if nodata_mask is None:
        nodata_mask = ~np.isfinite(arr)
    else:
        nodata_mask = np.asarray(nodata_mask, dtype=bool) | (~np.isfinite(arr))

    valid = ~nodata_mask
    if not np.any(valid):
        return arr.copy(), np.zeros_like(arr)

    max_val = float(np.nanmax(arr[valid])) + 1.0
    work = np.where(valid, arr, max_val)

    # Seed = max_val inside, actual elevation on outer border and pixels adjacent to nodata
    seed = np.full_like(work, max_val)
    seed[0, :] = work[0, :]
    seed[-1, :] = work[-1, :]
    seed[:, 0] = work[:, 0]
    seed[:, -1] = work[:, -1]
    if np.any(nodata_mask):
        seed[nodata_mask] = work[nodata_mask]
        # Pixels bordering nodata can drain out
        from scipy.ndimage import binary_dilation
        border_nd = binary_dilation(nodata_mask, structure=np.ones((3, 3))) & valid
        seed[border_nd] = work[border_nd]

    filled = reconstruction(seed, work, method='erosion', footprint=np.ones((3, 3)))
    fill_depth = np.where(valid, np.maximum(0.0, filled - work), np.nan)
    filled = np.where(valid, filled, np.nan)
    return filled, fill_depth


def double_logistic_eval(t, params):
    vmin, amp, t1, k1, t2, k2 = params
    s1 = 1.0 / (1.0 + np.exp(-np.clip(k1 * (t - t1), -40, 40)))
    s2 = 1.0 / (1.0 + np.exp(-np.clip(k2 * (t - t2), -40, 40)))
    return vmin + amp * (s1 - s2)


def fit_double_logistic(doys, vals, min_obs=8, min_spring=3, min_autumn=3, drop_outliers=True):
    """Robust 2-pass 6-parameter double-logistic phenology fit with QC gates.
    Pass 1 uses soft_l1 loss; if severe single-date cloud-leakage outliers exist
    (resid < -0.15 or resid > 0.22, <=15% of obs), Pass 2 refits on inliers so
    raw R2 is not falsely destroyed by 1-2 unmasked cloud drops.
    Returns dict with sos15/50/85, eos85/50/15, gsl, r2, rmse, qc_pass, etc."""
    t = np.asarray(doys, dtype=np.float64)
    y = np.asarray(vals, dtype=np.float64)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]

    n_obs = len(t)
    n_spr = int(np.sum((t >= 60) & (t <= 165)))
    n_aut = int(np.sum((t >= 245) & (t <= 350)))
    out = {
        'n_obs': n_obs, 'n_spring': n_spr, 'n_autumn': n_aut, 'n_outliers_dropped': 0,
        'sos15': np.nan, 'sos50': np.nan, 'sos85': np.nan,
        'eos85': np.nan, 'eos50': np.nan, 'eos15': np.nan,
        'gsl': np.nan, 'vmin': np.nan, 'vmax': np.nan, 'amp': np.nan,
        'k1': np.nan, 'k2': np.nan, 'r2': np.nan, 'rmse': np.nan,
        'qc_pass': False,
    }
    if n_obs < min_obs or n_spr < min_spring or n_aut < min_autumn:
        return out

    def _fit_once(tt, yy):
        win_m = (tt <= 65) | (tt >= 325)
        sum_m = (tt >= 155) & (tt <= 245)
        vmin0 = float(np.percentile(yy[win_m], 35)) if np.sum(win_m) >= 2 else float(np.percentile(yy, 10))
        vmax0 = float(np.percentile(yy[sum_m], 75)) if np.sum(sum_m) >= 2 else float(np.percentile(yy, 90))
        amp0 = max(0.05, vmax0 - vmin0)
        vmin0 = np.clip(vmin0, -0.2, 0.7)
        amp0 = np.clip(amp0, 0.05, 0.9)

        p0 = np.array([vmin0, amp0, 110.0, 0.12, 300.0, 0.10], dtype=np.float64)
        lb = np.array([-0.3, 0.03, 65.0, 0.02, 240.0, 0.02], dtype=np.float64)
        ub = np.array([0.8, 1.10, 175.0, 0.80, 355.0, 0.80], dtype=np.float64)
        p0 = np.clip(p0, lb + 1e-4, ub - 1e-4)
        res = least_squares(lambda p: double_logistic_eval(tt, p) - yy, p0,
                            bounds=(lb, ub), loss='soft_l1', f_scale=0.05, max_nfev=800)
        return res.x

    try:
        p = _fit_once(t, y)
    except Exception:
        return out

    n_dropped = 0
    if drop_outliers and n_obs >= max(10, min_obs + 2):
        resid1 = y - double_logistic_eval(t, p)
        inlier = (resid1 > -0.15) & (resid1 < 0.22)
        n_bad = int(np.sum(~inlier))
        if 0 < n_bad <= max(2, int(0.15 * n_obs)):
            t_in, y_in = t[inlier], y[inlier]
            if (len(t_in) >= min_obs
                    and int(np.sum((t_in >= 60) & (t_in <= 165))) >= min_spring
                    and int(np.sum((t_in >= 245) & (t_in <= 350))) >= min_autumn):
                try:
                    p = _fit_once(t_in, y_in)
                    t, y = t_in, y_in
                    n_dropped = n_bad
                except Exception:
                    pass

    yhat = double_logistic_eval(t, p)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else np.nan
    rmse = math.sqrt(ss_res / len(t))

    vmin, amp, t1, k1, t2, k2 = [float(v) for v in p]
    # Exact analytical percentile crossings for logistic branches:
    # 1/(1+exp(-k(t-t0))) = q  =>  t = t0 + ln(q/(1-q))/k
    logit85 = math.log(0.85 / 0.15)  # ~1.7346
    sos15 = t1 - logit85 / k1
    sos50 = t1
    sos85 = t1 + logit85 / k1
    eos85 = t2 - logit85 / k2
    eos50 = t2
    eos15 = t2 + logit85 / k2
    gsl = eos50 - sos50

    qc = (
        np.isfinite(r2) and r2 >= 0.85
        and len(t) >= max(6, min_obs)
        and amp >= 0.08
        and 70.0 <= sos50 <= 165.0
        and 245.0 <= eos50 <= 345.0
        and gsl > 80.0
    )
    out.update({
        'n_obs': int(len(t)),
        'n_outliers_dropped': int(n_dropped),
        'sos15': round(sos15, 3), 'sos50': round(sos50, 3), 'sos85': round(sos85, 3),
        'eos85': round(eos85, 3), 'eos50': round(eos50, 3), 'eos15': round(eos15, 3),
        'gsl': round(gsl, 3), 'vmin': round(vmin, 4), 'vmax': round(vmin + amp, 4),
        'amp': round(amp, 4), 'k1': round(k1, 4), 'k2': round(k2, 4),
        'r2': round(r2, 4), 'rmse': round(rmse, 4), 'qc_pass': bool(qc),
    })
    return out


def load_noaa_sewanee(force=False):
    """Download & parse NOAA GHCN-Daily Sewanee (USC00408184) daily TMAX, TMIN, PRCP."""
    path = os.path.join(CACHE_DIR, 'USC00408184.csv')
    if force or not os.path.exists(path):
        url = 'https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/USC00408184.csv'
        urllib.request.urlretrieve(url, path)
    df = pd.read_csv(path, low_memory=False)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['year'] = df['DATE'].dt.year
    df['month'] = df['DATE'].dt.month
    df['doy'] = df['DATE'].dt.dayofyear
    for col in ['TMAX', 'TMIN', 'PRCP']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce') / 10.0  # tenths of C / mm
        else:
            df[col] = np.nan
    return df[['DATE', 'year', 'month', 'doy', 'TMAX', 'TMIN', 'PRCP']].copy()


def fetch_era5_daily(year_start=2000, year_end=2024, force=False):
    """Fetch daily ERA5-Land meteorology at Sewanee (-85.92, 35.15) from GEE and cache."""
    path = os.path.join(CACHE_DIR, f'era5_daily_{year_start}_{year_end}.json')
    if not force and os.path.exists(path):
        return pd.DataFrame(json.load(open(path)))

    init_ee()
    pt = ee.Geometry.Point([-85.92, 35.15])
    rows = []
    for yr in range(year_start, year_end + 1):
        col = (ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
               .filterDate(f'{yr}-01-01', f'{yr+1}-01-01')
               .select([
                   'temperature_2m_min', 'temperature_2m_max', 'temperature_2m',
                   'u_component_of_wind_10m', 'v_component_of_wind_10m',
                   'total_precipitation_sum', 'surface_net_solar_radiation_sum'
               ]))
        def _ext(img):
            d = img.reduceRegion(ee.Reducer.first(), pt, 10000)
            return ee.Feature(None, d.set('date', img.date().format('YYYY-MM-dd')))
        feats = col.map(_ext).getInfo()['features']
        for f in feats:
            p = f['properties']
            u = p.get('u_component_of_wind_10m')
            v = p.get('v_component_of_wind_10m')
            wspd = math.hypot(u, v) if (u is not None and v is not None) else None
            rows.append({
                'date': p.get('date'),
                'year': yr,
                't2m_min_C': None if p.get('temperature_2m_min') is None else round(p['temperature_2m_min'] - 273.15, 3),
                't2m_max_C': None if p.get('temperature_2m_max') is None else round(p['temperature_2m_max'] - 273.15, 3),
                't2m_C': None if p.get('temperature_2m') is None else round(p['temperature_2m'] - 273.15, 3),
                'u10': None if u is None else round(u, 3),
                'v10': None if v is None else round(v, 3),
                'wspd_ms': None if wspd is None else round(wspd, 3),
                'precip_mm': None if p.get('total_precipitation_sum') is None else round(p['total_precipitation_sum'] * 1000.0, 3),
                'ssr_MJ': None if p.get('surface_net_solar_radiation_sum') is None else round(p['surface_net_solar_radiation_sum'] / 1e6, 3),
            })
    json.dump(rows, open(path, 'w'))
    return pd.DataFrame(rows)


def ols_cluster_se(X, y, clusters=None, add_const=True, col_names=None):
    """OLS with optional Cameron-Miller cluster-robust SEs (HC1 if clusters=None)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[ok], y[ok]
    if clusters is not None:
        clusters = np.asarray(clusters)[ok]
    if add_const:
        X = np.column_stack([np.ones(len(y)), X])
        names = ['intercept'] + (list(col_names) if col_names else [f'x{i}' for i in range(X.shape[1] - 1)])
    else:
        names = list(col_names) if col_names else [f'x{i}' for i in range(X.shape[1])]

    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(1, n - k)

    if clusters is None:
        # HC1
        scale = n / max(1, n - k)
        meat = (X * (resid[:, None] ** 2)).T @ X
        vcov = scale * XtX_inv @ meat @ XtX_inv
        df = max(1, n - k)
    else:
        uniq = np.unique(clusters)
        G = len(uniq)
        meat = np.zeros((k, k))
        for g in uniq:
            m = clusters == g
            score = X[m].T @ resid[m]
            meat += np.outer(score, score)
        scale = (G / max(1, G - 1)) * ((n - 1) / max(1, n - k))
        vcov = scale * XtX_inv @ meat @ XtX_inv
        df = max(1, G - 1)

    se = np.sqrt(np.maximum(np.diag(vcov), 1e-18))
    tstat = beta / se
    pval = 2.0 * stats.t.sf(np.abs(tstat), df=df)

    coefs = {}
    for i, nm in enumerate(names):
        coefs[nm] = {
            'beta': round(float(beta[i]), 4),
            'se': round(float(se[i]), 4),
            't': round(float(tstat[i]), 3),
            'p': float(f'{pval[i]:.4g}'),
        }
    return {
        'n': int(n), 'k': int(k), 'df': int(df),
        'r2': round(float(r2), 4), 'adj_r2': round(float(adj_r2), 4),
        'rmse': round(float(math.sqrt(ss_res / max(1, n))), 4),
        'coefs': coefs, 'vcov': vcov.tolist(), 'beta_vec': beta.tolist(),
    }


def empirical_variogram(coords_m, values, max_dist=3500.0, n_bins=14, max_pairs=250000, seed=42):
    """Compute empirical semivariogram and fit spherical range (meters)."""
    coords = np.asarray(coords_m, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v) & np.all(np.isfinite(coords), axis=1)
    coords, v = coords[ok], v[ok]
    n = len(v)
    rng = np.random.default_rng(seed)
    if n > 800:
        idx = rng.choice(n, 800, replace=False)
        coords, v = coords[idx], v[idx]
        n = len(v)

    dx = coords[:, 0:1] - coords[:, 0:1].T
    dy = coords[:, 1:2] - coords[:, 1:2].T
    dist = np.hypot(dx, dy)
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    sv = 0.5 * (v[iu[0]] - v[iu[1]]) ** 2

    edges = np.linspace(0.0, max_dist, n_bins + 1)
    lags, gamma, counts = [], [], []
    for i in range(n_bins):
        m = (d >= edges[i]) & (d < edges[i + 1])
        if np.sum(m) >= 20:
            lags.append(float(np.mean(d[m])))
            gamma.append(float(np.mean(sv[m])))
            counts.append(int(np.sum(m)))

    lags = np.array(lags)
    gamma = np.array(gamma)

    # Fit spherical variogram: nugget + sill * (1.5*(h/r) - 0.5*(h/r)^3) for h < r
    def sph(h, c0, c1, r):
        x = np.minimum(h / max(r, 1.0), 1.0)
        return c0 + c1 * (1.5 * x - 0.5 * x ** 3)

    if len(lags) >= 4:
        var_tot = float(np.var(v))
        res = least_squares(
            lambda p: sph(lags, p[0], p[1], p[2]) - gamma,
            x0=[var_tot * 0.2, var_tot * 0.8, 800.0],
            bounds=([0.0, 0.0, 50.0], [var_tot * 2.0, var_tot * 3.0, max_dist * 1.5]),
        )
        nugget, sill, rng_m = [float(x) for x in res.x]
    else:
        nugget, sill, rng_m = np.nan, np.nan, 800.0

    return {
        'lags_m': [round(x, 1) for x in lags],
        'semivariance': [round(x, 4) for x in gamma],
        'pair_counts': counts,
        'nugget': round(nugget, 4) if np.isfinite(nugget) else None,
        'partial_sill': round(sill, 4) if np.isfinite(sill) else None,
        'spherical_range_m': round(rng_m, 1) if np.isfinite(rng_m) else None,
    }


def manski_diff_bounds(y1, p_miss1, y0, p_miss0, y_lo, y_hi):
    """Manski worst-case bounds on E[Y1] - E[Y0] under missingness rates p_miss1, p_miss0."""
    m1 = float(np.nanmean(y1))
    m0 = float(np.nanmean(y0))
    e1_lo = (1.0 - p_miss1) * m1 + p_miss1 * y_lo
    e1_hi = (1.0 - p_miss1) * m1 + p_miss1 * y_hi
    e0_lo = (1.0 - p_miss0) * m0 + p_miss0 * y_lo
    e0_hi = (1.0 - p_miss0) * m0 + p_miss0 * y_hi
    return {
        'naive_diff': round(m1 - m0, 3),
        'manski_lo': round(e1_lo - e0_hi, 3),
        'manski_hi': round(e1_hi - e0_lo, 3),
        'p_miss1': round(float(p_miss1), 4),
        'p_miss0': round(float(p_miss0), 4),
    }


def sanitize_json(obj):
    """Recursively replace non-finite floats (NaN/Inf) with None and numpy scalars with builtins."""
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_json(name, obj):
    path = os.path.join(HERE, name)
    clean_obj = sanitize_json(obj)
    with open(path, 'w') as f:
        json.dump(clean_obj, f, indent=2, allow_nan=False, default=str)
    print(f'WROTE {path}')
    return path
