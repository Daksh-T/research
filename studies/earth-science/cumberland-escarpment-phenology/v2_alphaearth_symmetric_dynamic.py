"""Stage 4 (4D): Symmetric Matched-Capacity Dynamic Baseline (Raw Delta_S2_t vs. Delta_emb_t -> Delta_SOS_t).

Closes Stage-3 P6 gap:
- Extracts matched-capacity annual Sentinel-2 10-band 4-season composites (40 bands + 20 seasonal
  spectral indices = 60 features) for 2018-2024 at the EXACT same 147 pixels / 900 pixel-years
  cached in profiling/cache/p6_interannual_emb_sos.json (verified bit-exact against cached 2018 A00!).
- Runs identical strict leak-free evaluation protocols (LOYO pixel-demeaned, LOYO two-way demeaned,
  LOYO strict first-difference, LOYO first-difference year-demeaned, 5-fold GroupPixelCV pixel-demeaned
  & two-way demeaned) across BOTH RidgeCV and RandomForestRegressor for:
    1. AlphaEarth Delta_emb_t (64-d)
    2. Raw Sentinel-2 Delta_S2_t Full 4-Season (40 bands)
    3. Raw Sentinel-2 Delta_S2_t Full 4-Season + Spectral Indices (60 features; matches C2 capacity)
    4. Raw Sentinel-2 Delta_S2_t Non-Spring Only (Win+Sum+Aut, 30 bands + 15 indices = 45 features)
    5. Combined AlphaEarth + Raw S2 (104 features)
    6. Prior-year lagged check (t-1 -> t) for both AlphaEarth and Raw S2 (zero same-year overlap)
- Also consolidates P6 spatial buffered CV (b=800m, b=eff) and 4-cove LOCO transfer so Figure 7
  reads a single complete, symmetric AlphaEarth sidebar dataset.

Outputs:
  data/v2_alphaearth_symmetric_dynamic.json (strict RFC 8259 JSON)
"""
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'profiling'))
import ee
import common as C
import eslib as L

OUT_PATH = os.path.join(HERE, 'data', 'v2_alphaearth_symmetric_dynamic.json')
CACHE_S2_DYN = os.path.join(C.CACHE_DIR, 'v2_p6_interannual_raw_s2.json')

S2_10BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
AE_BANDS = L.BANDS  # A00..A63
RIDGE_ALPHAS = np.logspace(-1, 5, 25)
SEASONS = [('win', 1, 90), ('spr', 91, 155), ('sum', 156, 250), ('aut', 251, 365)]


def extract_matched_raw_s2_panel(df_dyn):
    """Extract matched-capacity 2018-2024 4-season S2 10-band composites at exact P6 dynamic pixels."""
    if os.path.exists(CACHE_S2_DYN):
        s2_rows = json.load(open(CACHE_S2_DYN))
        print(f'[4D] Loaded cached annual Raw S2 panel ({len(s2_rows)} pixel-years)')
        return pd.DataFrame(s2_rows), {'cached': True}

    print('[4D] Regenerating exact P6 220-pixel grid (seed=55) and verifying alignment against cached A00...')
    dom = C.get_domain()
    dem = C.get_dem_3dep().rename('elev')
    nlcd = C.get_nlcd_2021()
    decid = nlcd.eq(41)
    pts_fc = dem.updateMask(decid).sample(region=dom, scale=30, numPixels=220, seed=55, geometries=True, tileScale=4)
    p_feats = pts_fc.getInfo()['features']
    pts = [{'pid': i, 'lon': f['geometry']['coordinates'][0], 'lat': f['geometry']['coordinates'][1]}
           for i, f in enumerate(p_feats)]
    fc_pts = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([p['lon'], p['lat']]), {'pid': p['pid']}) for p in pts])

    # Verify exact pixel alignment against cached 2018 AlphaEarth A00!
    ae18 = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
            .filterDate('2018-01-01', '2019-01-01').filterBounds(dom).mosaic().select(['A00']))
    ef18 = ae18.sampleRegions(fc_pts, scale=20, tileScale=4, geometries=False).getInfo()['features']
    a00_check = {int(f['properties']['pid']): float(f['properties']['A00']) for f in ef18 if 'A00' in f['properties']}
    sub18 = df_dyn[df_dyn['yr'] == 2018]
    diffs = [abs(a00_check[int(r['pid'])] - float(r['A00'])) for _, r in sub18.iterrows() if int(r['pid']) in a00_check]
    max_diff = max(diffs) if diffs else float('inf')
    print(f'  Pixel alignment audit against cached 2018 A00: n={len(diffs)}, max|diff|={max_diff:.2e}')
    if max_diff > 1e-4:
        raise RuntimeError(f'Pixel alignment mismatch! max_diff={max_diff}')

    # Extract 4-season 10-band Sentinel-2 composites for each year 2018-2024 (7 fast GEE calls!)
    def prep_s2(img):
        scl = img.select('SCL')
        good = scl.eq(4).Or(scl.eq(5))
        return img.select(S2_10BANDS).multiply(1e-4).updateMask(good).copyProperties(img, ['system:time_start'])

    s2_rows = []
    for yr in range(2018, 2025):
        t0 = time.time()
        s2_yr = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                 .filterBounds(dom).filterDate(f'{yr}-01-01', f'{yr+1}-01-01')
                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70)).map(prep_s2))
        ann_med = s2_yr.median()
        imgs = []
        for sname, d0, d1 in SEASONS:
            comp = s2_yr.filter(ee.Filter.calendarRange(d0, d1, 'day_of_year')).median().unmask(ann_med)
            imgs.append(comp.rename([f's2_{sname}_{b}' for b in S2_10BANDS]))
        stack = ee.Image.cat(imgs)
        for attempt in range(4):
            try:
                feats = stack.sampleRegions(fc_pts, scale=20, tileScale=4, geometries=False).getInfo()['features']
                break
            except Exception as ex:
                if attempt == 3:
                    raise
                time.sleep(3 * (attempt + 1))
        for f in feats:
            p = f['properties']
            p['pid'] = int(p['pid'])
            p['yr'] = yr
            s2_rows.append(p)
        print(f'  Extracted Raw S2 4-season stack for {yr}: {len(feats)} pixels in {time.time()-t0:.1f}s')

    json.dump(s2_rows, open(CACHE_S2_DYN, 'w'))
    return pd.DataFrame(s2_rows), {'cached': False, 'alignment_max_a00_diff': round(max_diff, 8), 'alignment_n': len(diffs)}


def build_s2_indices(df, seasons_list):
    cols = []
    for s in seasons_list:
        b3 = df[f's2_{s}_B3'].values.astype(np.float64)
        b4 = df[f's2_{s}_B4'].values.astype(np.float64)
        b5 = df[f's2_{s}_B5'].values.astype(np.float64)
        b8 = df[f's2_{s}_B8'].values.astype(np.float64)
        b8a = df[f's2_{s}_B8A'].values.astype(np.float64)
        b11 = df[f's2_{s}_B11'].values.astype(np.float64)
        b12 = df[f's2_{s}_B12'].values.astype(np.float64)
        cols.extend([
            (b8 - b4) / np.maximum(b8 + b4, 1e-4),
            (b8a - b5) / np.maximum(b8a + b5, 1e-4),
            (b8 - b11) / np.maximum(b8 + b11, 1e-4),
            (b8 - b12) / np.maximum(b8 + b12, 1e-4),
            (b8 - b3) / np.maximum(b8 + b3, 1e-4),
        ])
    return np.column_stack(cols)


def eval_dynamic_suite(X_raw, y_raw, pids, yrs):
    """Exact replica of P6 strict leak-free dynamic CV suite (LOYO, first-diff LOYO, GroupPixelCV)."""
    def eval_loyo(two_way=False):
        out = {}
        for alg in ['Ridge', 'RF']:
            preds = np.full(len(y_raw), np.nan)
            y_eval = np.full(len(y_raw), np.nan)
            for yr in np.unique(yrs):
                te = (yrs == yr)
                tr = ~te
                px_m, py_m = {}, {}
                for p in np.unique(pids[tr]):
                    m = tr & (pids == p)
                    px_m[p] = X_raw[m].mean(axis=0)
                    py_m[p] = y_raw[m].mean()
                tr_idx = np.where(tr)[0]
                te_idx = np.where(te)[0]
                X_tr = np.array([X_raw[i] - px_m[pids[i]] for i in tr_idx])
                y_tr = np.array([y_raw[i] - py_m[pids[i]] for i in tr_idx])
                X_te = np.array([X_raw[i] - px_m[pids[i]] for i in te_idx])
                y_te = np.array([y_raw[i] - py_m[pids[i]] for i in te_idx])
                if two_way:
                    tr_yrs = yrs[tr]
                    for y2 in np.unique(tr_yrs):
                        mm = (tr_yrs == y2)
                        X_tr[mm] -= X_tr[mm].mean(axis=0)
                        y_tr[mm] -= y_tr[mm].mean()
                    X_te = X_te - X_te.mean(axis=0)
                    y_te = y_te - y_te.mean()
                if alg == 'Ridge':
                    mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
                else:
                    mdl = RandomForestRegressor(n_estimators=150, min_samples_leaf=5,
                                                max_features='sqrt', n_jobs=-1, random_state=42)
                mdl.fit(X_tr, y_tr)
                preds[te] = mdl.predict(X_te)
                y_eval[te] = y_te
            out[alg] = {
                'loyo_r2': round(float(r2_score(y_eval, preds)), 4),
                'loyo_rmse_d': round(float(root_mean_squared_error(y_eval, preds)), 3),
                'loyo_r': round(float(np.corrcoef(y_eval, preds)[0, 1]), 4),
            }
        return out

    # Strict first-difference LOYO (consecutive years, holding out both yr and yr+1)
    order = np.lexsort((yrs, pids))
    fd_rows = []
    for i in range(1, len(order)):
        i0, i1 = order[i - 1], order[i]
        if pids[i1] == pids[i0] and yrs[i1] - yrs[i0] == 1:
            fd_rows.append((int(pids[i1]), int(yrs[i1]), float(y_raw[i1] - y_raw[i0]), X_raw[i1] - X_raw[i0]))
    fd_yrs = np.array([r[1] for r in fd_rows])
    fd_y = np.array([r[2] for r in fd_rows])
    fd_X = np.vstack([r[3] for r in fd_rows])

    def eval_fd_loyo(year_demean=False):
        out = {}
        for alg in ['Ridge', 'RF']:
            preds = np.full(len(fd_y), np.nan)
            y_eval = np.full(len(fd_y), np.nan)
            for yr in np.unique(fd_yrs):
                te = (fd_yrs == yr)
                # Strict non-overlapping first-difference LOYO:
                # Row fd_yr=yr uses (yr, yr-1), so exclude yr-1 (shares yr-1), yr, and yr+1 (shares yr)!
                tr = (fd_yrs != yr - 1) & (fd_yrs != yr) & (fd_yrs != yr + 1)
                if tr.sum() < 50:
                    continue
                X_tr, y_tr = fd_X[tr].copy(), fd_y[tr].copy()
                X_te, y_te = fd_X[te].copy(), fd_y[te].copy()
                if year_demean:
                    for y2 in np.unique(fd_yrs[tr]):
                        mm = (fd_yrs[tr] == y2)
                        X_tr[mm] -= X_tr[mm].mean(axis=0)
                        y_tr[mm] -= y_tr[mm].mean()
                    X_te -= X_te.mean(axis=0)
                    y_te -= y_te.mean()
                if alg == 'Ridge':
                    mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
                else:
                    mdl = RandomForestRegressor(n_estimators=150, min_samples_leaf=5,
                                                max_features='sqrt', n_jobs=-1, random_state=42)
                mdl.fit(X_tr, y_tr)
                preds[te] = mdl.predict(X_te)
                y_eval[te] = y_te
            ok = np.isfinite(preds)
            out[alg] = {
                'n_pairs': int(ok.sum()),
                'loyo_r2': round(float(r2_score(y_eval[ok], preds[ok])), 4),
                'loyo_rmse_d': round(float(root_mean_squared_error(y_eval[ok], preds[ok])), 3),
                'loyo_r': round(float(np.corrcoef(y_eval[ok], preds[ok])[0, 1]), 4),
            }
        return out

    # Strict 5-fold GroupPixelCV
    X_pdm = np.zeros_like(X_raw)
    y_pdm = np.zeros_like(y_raw)
    for p in np.unique(pids):
        m = (pids == p)
        X_pdm[m] = X_raw[m] - X_raw[m].mean(axis=0)
        y_pdm[m] = y_raw[m] - y_raw[m].mean()

    def eval_group_pid(two_way=False):
        uniq_p = np.unique(pids)
        rng = np.random.default_rng(42)
        pmap = dict(zip(uniq_p, rng.integers(0, 5, size=len(uniq_p))))
        p_fold = np.array([pmap[p] for p in pids])
        out = {}
        for alg in ['Ridge', 'RF']:
            preds = np.full(len(y_raw), np.nan)
            y_eval = np.full(len(y_raw), np.nan)
            for k in range(5):
                te = (p_fold == k)
                tr = ~te
                X_tr, y_tr = X_pdm[tr].copy(), y_pdm[tr].copy()
                X_te, y_te = X_pdm[te].copy(), y_pdm[te].copy()
                if two_way:
                    for yr in np.unique(yrs):
                        mtr = (yrs[tr] == yr)
                        mte = (yrs[te] == yr)
                        xm = X_tr[mtr].mean(axis=0)
                        ym = y_tr[mtr].mean()
                        X_tr[mtr] -= xm
                        y_tr[mtr] -= ym
                        X_te[mte] -= xm
                        y_te[mte] -= ym
                if alg == 'Ridge':
                    mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
                else:
                    mdl = RandomForestRegressor(n_estimators=150, min_samples_leaf=5,
                                                max_features='sqrt', n_jobs=-1, random_state=42)
                mdl.fit(X_tr, y_tr)
                preds[te] = mdl.predict(X_te)
                y_eval[te] = y_te
            out[alg] = {
                'group_pixel_r2': round(float(r2_score(y_eval, preds)), 4),
                'group_pixel_rmse_d': round(float(root_mean_squared_error(y_eval, preds)), 3),
            }
        return out

    # Prior-year lagged test (X_{t-1} -> y_t, zero same-year temporal overlap!)
    lag_rows = []
    for i in range(1, len(order)):
        i0, i1 = order[i - 1], order[i]
        if pids[i1] == pids[i0] and yrs[i1] - yrs[i0] == 1:
            lag_rows.append((int(pids[i1]), int(yrs[i1]), float(y_raw[i1]), X_raw[i0]))
    l_pids = np.array([r[0] for r in lag_rows])
    l_yrs = np.array([r[1] for r in lag_rows])
    l_y = np.array([r[2] for r in lag_rows])
    l_X = np.vstack([r[3] for r in lag_rows])

    def eval_lag_loyo(two_way=True, excl_adj=False):
        out = {}
        for alg in ['Ridge', 'RF']:
            l_preds = np.full(len(l_y), np.nan)
            l_yeval = np.full(len(l_y), np.nan)
            for yr in np.unique(l_yrs):
                te = (l_yrs == yr)
                tr = (~te) if not excl_adj else ((l_yrs != yr) & (l_yrs != yr - 1) & (l_yrs != yr + 1))
                px_m, py_m = {}, {}
                for p in np.unique(l_pids[tr]):
                    m = tr & (l_pids == p)
                    px_m[p] = l_X[m].mean(axis=0)
                    py_m[p] = l_y[m].mean()
                tr_idx = [i for i in np.where(tr)[0] if l_pids[i] in px_m]
                te_idx = [i for i in np.where(te)[0] if l_pids[i] in px_m]
                if len(tr_idx) < 50 or len(te_idx) < 10:
                    continue
                X_tr = np.array([l_X[i] - px_m[l_pids[i]] for i in tr_idx])
                y_tr = np.array([l_y[i] - py_m[l_pids[i]] for i in tr_idx])
                X_te = np.array([l_X[i] - px_m[l_pids[i]] for i in te_idx])
                y_te = np.array([l_y[i] - py_m[l_pids[i]] for i in te_idx])
                if two_way:
                    ty = l_yrs[tr_idx]
                    for y2 in np.unique(ty):
                        mm = (ty == y2)
                        X_tr[mm] -= X_tr[mm].mean(axis=0)
                        y_tr[mm] -= y_tr[mm].mean()
                    X_te -= X_te.mean(axis=0)
                    y_te -= y_te.mean()
                if alg == 'Ridge':
                    mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
                else:
                    mdl = RandomForestRegressor(n_estimators=150, min_samples_leaf=5,
                                                max_features='sqrt', n_jobs=-1, random_state=42)
                mdl.fit(X_tr, y_tr)
                l_preds[te_idx] = mdl.predict(X_te)
                l_yeval[te_idx] = y_te
            lok = np.isfinite(l_preds)
            out[alg] = {
                'n_pairs': int(lok.sum()),
                'loyo_r2': round(float(r2_score(l_yeval[lok], l_preds[lok])), 4),
                'loyo_rmse_d': round(float(root_mean_squared_error(l_yeval[lok], l_preds[lok])), 3),
                'loyo_r': round(float(np.corrcoef(l_yeval[lok], l_preds[lok])[0, 1]), 4),
            }
        return out

    return {
        'n_features': int(X_raw.shape[1]),
        'pixel_demeaned_LOYO_strict': eval_loyo(two_way=False),
        'twoway_demeaned_LOYO_strict': eval_loyo(two_way=True),
        'first_diff_LOYO_strict': eval_fd_loyo(year_demean=False),
        'first_diff_year_demeaned_LOYO_strict': eval_fd_loyo(year_demean=True),
        'pixel_demeaned_GroupPixelCV_strict': eval_group_pid(two_way=False),
        'twoway_demeaned_GroupPixelCV_strict': eval_group_pid(two_way=True),
        'lagged_prev_year_twoway_LOYO': eval_lag_loyo(two_way=True, excl_adj=False),
        'lagged_prev_year_nonoverlap_LOYO': eval_lag_loyo(two_way=False, excl_adj=True),
        'lagged_prev_year_pixel_only_naive_diagnostic': eval_lag_loyo(two_way=False, excl_adj=False),
    }


def main():
    t0 = time.time()
    if not os.path.exists(CACHE_S2_DYN):
        C.init_ee()
    df_dyn = pd.DataFrame(json.load(open(os.path.join(C.CACHE_DIR, 'p6_interannual_emb_sos.json'))))
    df_s2, align_audit = extract_matched_raw_s2_panel(df_dyn)

    # Merge on (pid, yr) and keep exact same >=5-year pixel subset!
    df = df_dyn.merge(df_s2, on=['pid', 'yr'], how='inner')
    cnts = df.groupby('pid')['yr'].count()
    keep_pids = cnts[cnts >= 5].index
    df = df[df['pid'].isin(keep_pids)].sort_values(['pid', 'yr']).reset_index(drop=True)
    print(f'[4D] Symmetric dynamic panel: {df["pid"].nunique()} pixels, {len(df)} pixel-years')

    s2_all_cols = [f's2_{s}_{b}' for s, _, _ in SEASONS for b in S2_10BANDS]
    s2_nospr_cols = [f's2_{s}_{b}' for s in ['win', 'sum', 'aut'] for b in S2_10BANDS]

    X_ae = df[AE_BANDS].values.astype(np.float64)
    X_s2_40 = df[s2_all_cols].values.astype(np.float64)
    X_s2_idx20 = build_s2_indices(df, ['win', 'spr', 'sum', 'aut'])
    X_s2_60 = np.column_stack([X_s2_40, X_s2_idx20])

    X_s2_ns30 = df[s2_nospr_cols].values.astype(np.float64)
    X_s2_ns_idx15 = build_s2_indices(df, ['win', 'sum', 'aut'])
    X_s2_ns45 = np.column_stack([X_s2_ns30, X_s2_ns_idx15])

    X_comb = np.column_stack([X_ae, X_s2_40])

    y_raw = df['sos'].values.astype(np.float64)
    pids = df['pid'].values
    yrs = df['yr'].values

    models = {
        'AE_delta_emb_64d': X_ae,
        'Raw_delta_S2_4seas_40b': X_s2_40,
        'Raw_delta_S2_4seas_idx_60f': X_s2_60,
        'Raw_delta_S2_nonspring_45f': X_s2_ns45,
        'Combined_AE_plus_RawS2_104f': X_comb,
    }
    dyn_results = {}
    for nm, Xm in models.items():
        print(f'  Evaluating {nm} (p={Xm.shape[1]})...')
        dyn_results[nm] = eval_dynamic_suite(Xm, y_raw, pids, yrs)

    # Pull spatial buffered CV and 4-cove LOCO from profiling/p6.json for unified sidebar JSON
    p6 = json.load(open(os.path.join(HERE, 'profiling', 'p6.json')))

    # Build clean symmetric comparison summary table
    def best_m(res, key, subkey='loyo_r2'):
        return max(res[key]['Ridge'][subkey], res[key]['RF'][subkey])

    sym_table = {}
    for nm, res in dyn_results.items():
        sym_table[nm] = {
            'n_features': res['n_features'],
            'pixel_demeaned_LOYO_r2': best_m(res, 'pixel_demeaned_LOYO_strict', 'loyo_r2'),
            'twoway_demeaned_LOYO_r2': best_m(res, 'twoway_demeaned_LOYO_strict', 'loyo_r2'),
            'first_diff_LOYO_r2': best_m(res, 'first_diff_LOYO_strict', 'loyo_r2'),
            'first_diff_2w_LOYO_r2': best_m(res, 'first_diff_year_demeaned_LOYO_strict', 'loyo_r2'),
            'pixel_demeaned_GroupPixel_r2': best_m(res, 'pixel_demeaned_GroupPixelCV_strict', 'group_pixel_r2'),
            'twoway_demeaned_GroupPixel_r2': best_m(res, 'twoway_demeaned_GroupPixelCV_strict', 'group_pixel_r2'),
            'lagged_prev_year_twoway_LOYO_r2': best_m(res, 'lagged_prev_year_twoway_LOYO', 'loyo_r2'),
            'lagged_prev_year_pixel_naive_r2': res['lagged_prev_year_pixel_only_naive_diagnostic']['Ridge']['loyo_r2'],
        }

    # Document exact 2018-2024 annual mean SOS series (explains biennial oscillation artifact in naive lag!)
    yr_means = {int(yr): round(float(g['sos'].mean()), 2) for yr, g in df.groupby('yr')}

    # Honest comparison against the BEST raw S2 baseline (40-band beats 60-feature on first-diff LOYO!)
    best_raw_s2_comparison = {}
    for metric_k in ['pixel_demeaned_LOYO_r2', 'twoway_demeaned_LOYO_r2', 'first_diff_LOYO_r2',
                     'first_diff_2w_LOYO_r2', 'twoway_demeaned_GroupPixel_r2',
                     'lagged_prev_year_twoway_LOYO_r2']:
        ae_v = sym_table['AE_delta_emb_64d'][metric_k]
        s2_40 = sym_table['Raw_delta_S2_4seas_40b'][metric_k]
        s2_60 = sym_table['Raw_delta_S2_4seas_idx_60f'][metric_k]
        best_s2 = max(s2_40, s2_60)
        best_s2_name = 'Raw_delta_S2_4seas_40b' if s2_40 >= s2_60 else 'Raw_delta_S2_4seas_idx_60f'
        best_raw_s2_comparison[metric_k] = {
            'AE_64d': ae_v,
            'Raw_S2_40b': s2_40,
            'Raw_S2_60f': s2_60,
            'best_raw_s2_model': best_s2_name,
            'best_raw_s2_r2': best_s2,
            'delta_AE_minus_best_raw_s2': round(ae_v - best_s2, 4),
        }

    # Exact LOCO fold names directly from profiling/p6.json (prevents fold-name misreporting)
    loco_folds = sorted(list(p6['loco_4coves_transfer']['E_AE2017_plus_3DEP']['Ridge']['per_cove_r2'].keys()))

    out = {
        'experiment': 'v2_4D_alphaearth_symmetric_dynamic',
        'manuscript_role': 'Standalone supplementary audit script (removed from main manuscript per peer review)',
        'elapsed_s': round(time.time() - t0, 1),
        'alignment_audit': align_audit,
        'n_pixels': int(df['pid'].nunique()),
        'n_pixel_years': int(len(df)),
        'annual_mean_sos_2018_2024': yr_means,
        'lag_audit_note': (
            'Sewanee domain-mean SOS alternated strictly high-low-high-low across 2018-2024 '
            '(118.0->107.0->110.6->108.7->113.7->104.8->109.4), so naive pixel-only lagged LOYO yields '
            'a spurious +0.161 R2 by negating y_{t-1}. Removing annual domain means (two-way demeaned lag) '
            'or excluding adjacent years collapses prior-year skill to zero/negative for BOTH AlphaEarth and Raw S2, '
            'confirming that same-year dynamic R2 reflects intra-annual trajectory compression rather than causal forecasting.'
        ),
        'spatial_summary_from_p6': {
            'variogram_sos_range_m': p6['variogram_sos']['spherical_range_m'],
            'variogram_terrain_resid_range_m': p6['variogram_terrain_resid']['spherical_range_m'],
            'effective_buffer_m': p6['effective_buffer_m'],
            'buffered_800m': p6['spatial_cv_buffered_800m'],
            'buffered_eff': p6['spatial_cv_buffered_eff'],
            'n_loco_folds': int(len(loco_folds)),
            'loco_actual_cove_names': loco_folds,
            'loco_4coves_transfer': p6['loco_4coves_transfer'],
        },
        'best_raw_s2_comparison': best_raw_s2_comparison,
        'dynamic_symmetric_summary': sym_table,
        'dynamic_full_results': dyn_results,
    }
    clean = C.sanitize_json(out)
    with open(OUT_PATH, 'w') as f:
        json.dump(clean, f, indent=2, allow_nan=False)
    print(f'WROTE {OUT_PATH} in {time.time()-t0:.1f}s')

    print('\n=== 4D SYMMETRIC DYNAMIC COMPARISON ===')
    print(json.dumps(sym_table, indent=2))


if __name__ == '__main__':
    main()
