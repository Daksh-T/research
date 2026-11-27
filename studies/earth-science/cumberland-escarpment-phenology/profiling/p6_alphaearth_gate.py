"""Experiment P6: Airtight AlphaEarth 2017 Sidebar Gate.

Tests under ZERO temporal overlap (2017 features -> 2018-2024 S2 SOS) and ZERO spatial leakage:
1. Empirical semivariogram of SOS residuals -> verifies practical autocorrelation range r,
   enforcing spatial buffer b >= max(800 m, r) (plus b=800m and b=1200m checks).
2. Variogram-Buffered Spatial Block CV (4 quadrants with >=800m exclusion buffer) comparing
   BOTH RidgeCV (standardized) AND tuned RandomForestRegressor across:
     (A) 3DEP morphometrics (elev, slope, cos/sin aspect, TPI 150/500/1200m, roughness, hillshade)
     (B) A + LANDFIRE EVT (one-hot)
     (C) Matched-capacity Raw 2017 S2 10-band seasonal composites (40 bands) + A
     (D) AlphaEarth 2017 only (64-d)
     (E) AlphaEarth 2017 + A (74-d)
3. Leave-One-Cove-Out (LOCO) Regional Transfer across 4 distinct Southern Cumberland coves/gorges
   (Sewanee Domain, Sinking Cove, Fiery Gizzard, Savage Gulf; 10-35 km apart).
4. Pixel-Demeaned Interannual Dynamic Test (Delta_emb_t -> Delta_SOS_t, 2018-2024) under
   Leave-One-Year-Out CV and Spatial-Block CV (both pixel-demeaned and two-way pixel+year demeaned).

Hard Keep/Drop Gate:
  Keep AlphaEarth sidebar ONLY IF (D)/(E) beats (C) by Delta R2 >= +0.03 on BOTH buffered spatial CV
  AND LOCO transfer, OR pixel-demeaned Delta_emb_t -> Delta_SOS_t achieves R2 >= 0.15.
"""
import os
import sys
import json
import math
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, root_mean_squared_error

RIDGE_ALPHAS = np.logspace(-1, 5, 25)


def fit_spatial_ridge(X_tr, y_tr, groups_tr):
    """Select Ridge alpha via inner group-blocked CV (prevents GCV under-regularization under spatial autocorrelation)."""
    uniq_g = np.unique(groups_tr)
    if len(uniq_g) < 2:
        mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
        mdl.fit(X_tr, y_tr)
        return mdl
    best_a, best_mse = float(RIDGE_ALPHAS[0]), float('inf')
    for a in RIDGE_ALPHAS:
        sq_errs = []
        for g in uniq_g:
            iv = (groups_tr == g)
            it = ~iv
            if it.sum() < 10 or iv.sum() < 5:
                continue
            m = make_pipeline(StandardScaler(), Ridge(alpha=float(a)))
            m.fit(X_tr[it], y_tr[it])
            p = m.predict(X_tr[iv])
            sq_errs.append(np.mean((y_tr[iv] - p) ** 2))
        if sq_errs and np.mean(sq_errs) < best_mse:
            best_mse = float(np.mean(sq_errs))
            best_a = float(a)
    mdl = make_pipeline(StandardScaler(), Ridge(alpha=best_a))
    mdl.fit(X_tr, y_tr)
    return mdl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ee
import common as C
import eslib as L

S2_10BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
AE_BANDS = L.BANDS  # A00..A63


def build_feature_stack(region, use_asset_sos=False):
    """Build aligned 2017 feature stack + 2018-2024 S2 SOS over any region."""
    dem = C.get_dem_3dep().rename('elev')
    slope = ee.Terrain.slope(dem).rename('slope')
    asp = ee.Terrain.aspect(dem).multiply(math.pi / 180.0)
    cos_asp = asp.cos().rename('cos_asp')
    sin_asp = asp.sin().rename('sin_asp')
    tpi150 = dem.subtract(dem.focalMean(150, 'circle', 'meters')).rename('tpi150')
    tpi500 = dem.subtract(dem.focalMean(500, 'circle', 'meters')).rename('tpi500')
    tpi1200 = dem.subtract(dem.focalMean(1200, 'circle', 'meters')).rename('tpi1200')
    rough90 = dem.reduceNeighborhood(ee.Reducer.stdDev(), ee.Kernel.circle(90, 'meters')).rename('rough90')
    hill_nov = ee.Terrain.hillshade(dem, 152, 37).rename('hill_nov')  # Nov 1 ~10:30 AM
    terrain = ee.Image.cat([dem, slope, cos_asp, sin_asp, tpi150, tpi500, tpi1200, rough90, hill_nov])

    evt = C.get_evt().rename('evt')
    nlcd = C.get_nlcd_2021()
    decid = nlcd.eq(41)

    # Raw 2017 Sentinel-2 10-band seasonal composites (4 seasons -> 40 bands)
    def prep_s2(img):
        scl = img.select('SCL')
        good = scl.eq(4).Or(scl.eq(5))
        return img.select(S2_10BANDS).multiply(1e-4).updateMask(good).copyProperties(img, ['system:time_start'])

    s2_17 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
             .filterBounds(region).filterDate('2017-01-01', '2018-01-01')
             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70)).map(prep_s2))

    # 4 seasonal windows covering full 2017 phenology cycle (fallback to annual median if a pixel is cloudy in one window)
    ann_med = s2_17.median()
    seas = [
        ('win', 1, 90),
        ('spr', 91, 155),
        ('sum', 156, 250),
        ('aut', 251, 365),
    ]
    s2_imgs = []
    for sname, d0, d1 in seas:
        comp = s2_17.filter(ee.Filter.calendarRange(d0, d1, 'day_of_year')).median().unmask(ann_med)
        s2_imgs.append(comp.rename([f's2_{sname}_{b}' for b in S2_10BANDS]))
    s2_stack = ee.Image.cat(s2_imgs)

    # AlphaEarth 2017 (64 bands)
    ae17 = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
            .filterDate('2017-01-01', '2018-01-01').filterBounds(region).mosaic().select(AE_BANDS))

    # Target: 2018-2024 S2 SOS (zero temporal overlap with 2017!)
    if use_asset_sos:
        sos = ee.Image(os.environ.get('EE_SOS_ASSET', f"projects/{os.environ.get('EE_PROJECT', '')}/assets/escarpment_sos")).rename('sos')
    else:
        sos = L.build_sos(region, years=('2018-01-01', '2025-01-01'), decid_only=True).rename('sos')

    return ee.Image.cat([sos, terrain, evt, s2_stack, ae17]).updateMask(decid).updateMask(sos.mask())


def sample_domain_and_loco():
    cache_dom = os.path.join(C.CACHE_DIR, 'p6_domain_2017_stack.json')
    cache_loco = os.path.join(C.CACHE_DIR, 'p6_loco_4coves_stack.json')

    dom = C.get_domain()
    if os.path.exists(cache_dom):
        df_dom = pd.DataFrame(json.load(open(cache_dom)))
        print(f'[P6] Loaded cached Domain stack ({len(df_dom)} pixels)')
    else:
        print('[P6] Sampling Sewanee Domain (2017 S2 + AE2017 + 3DEP + EVT -> 2018-2024 SOS)...')
        stack = build_feature_stack(dom, use_asset_sos=True)
        feats = stack.sample(region=dom, scale=30, numPixels=2600, seed=77, geometries=True, tileScale=8).getInfo()['features']
        rows = []
        for f in feats:
            p = f['properties']
            if 'sos' not in p or 'A00' not in p or 's2_spr_B8' not in p:
                continue
            lon, lat = f['geometry']['coordinates']
            p['lon'] = round(lon, 6)
            p['lat'] = round(lat, 6)
            rows.append(p)
        df_dom = pd.DataFrame(rows)
        json.dump(rows, open(cache_dom, 'w'))
        print(f'  Sampled {len(df_dom)} complete Domain pixels')

    if os.path.exists(cache_loco):
        df_loco = pd.DataFrame(json.load(open(cache_loco)))
        print(f'[P6] Loaded cached 4-cove LOCO stack ({len(df_loco)} pixels)')
    else:
        print('[P6] Sampling 4 distinct regional coves for Leave-One-Cove-Out (LOCO) transfer...')
        coves = [
            ('Sewanee_LostCove', dom),
            ('Sinking_Cove', ee.Geometry.Point([-85.992, 35.014]).buffer(3500)),
            ('Fiery_Gizzard', ee.Geometry.Point([-85.760, 35.245]).buffer(3500)),
            ('Savage_Gulf', ee.Geometry.Point([-85.565, 35.448]).buffer(3500)),
        ]
        all_rows = []
        for cname, reg in coves:
            t0 = time.time()
            st = build_feature_stack(reg, use_asset_sos=False)
            feats = st.sample(region=reg, scale=30, numPixels=700, seed=88, geometries=True, tileScale=8).getInfo()['features']
            cnt = 0
            for f in feats:
                p = f['properties']
                if 'sos' not in p or 'A00' not in p or 's2_spr_B8' not in p:
                    continue
                lon, lat = f['geometry']['coordinates']
                p['lon'] = round(lon, 6)
                p['lat'] = round(lat, 6)
                p['cove'] = cname
                all_rows.append(p)
                cnt += 1
            print(f'  [{cname}] {cnt} pixels in {time.time()-t0:.1f}s')
        df_loco = pd.DataFrame(all_rows)
        json.dump(all_rows, open(cache_loco, 'w'))

    return df_dom, df_loco


def build_feature_matrices(df):
    terrain_cols = ['elev', 'slope', 'cos_asp', 'sin_asp', 'tpi150', 'tpi500', 'tpi1200', 'rough90', 'hill_nov']
    s2_cols = [f's2_{s}_{b}' for s in ['win', 'spr', 'sum', 'aut'] for b in S2_10BANDS]
    ae_cols = AE_BANDS

    X_A = df[terrain_cols].values.astype(np.float64)
    # One-hot EVT (keep categories present >= 10 times to avoid degenerate columns)
    evt_vals = df['evt'].fillna(-1).astype(int).values
    top_evts = [k for k, v in pd.Series(evt_vals).value_counts().items() if v >= 10]
    X_evt = np.column_stack([(evt_vals == e).astype(np.float64) for e in top_evts])
    X_B = np.column_stack([X_A, X_evt])
    X_S2 = df[s2_cols].values.astype(np.float64)
    # Also construct seasonal spectral indices (NDVI, NDRE, NDMI, NBR, GNDVI x 4 seasons = 20 features)
    idx_cols = []
    for s in ['win', 'spr', 'sum', 'aut']:
        b3 = df[f's2_{s}_B3'].values.astype(np.float64)
        b4 = df[f's2_{s}_B4'].values.astype(np.float64)
        b5 = df[f's2_{s}_B5'].values.astype(np.float64)
        b8 = df[f's2_{s}_B8'].values.astype(np.float64)
        b8a = df[f's2_{s}_B8A'].values.astype(np.float64)
        b11 = df[f's2_{s}_B11'].values.astype(np.float64)
        b12 = df[f's2_{s}_B12'].values.astype(np.float64)
        idx_cols.extend([
            (b8 - b4) / np.maximum(b8 + b4, 1e-4),
            (b8a - b5) / np.maximum(b8a + b5, 1e-4),
            (b8 - b11) / np.maximum(b8 + b11, 1e-4),
            (b8 - b12) / np.maximum(b8 + b12, 1e-4),
            (b8 - b3) / np.maximum(b8 + b3, 1e-4),
        ])
    X_S2_idx = np.column_stack(idx_cols)
    X_C = np.column_stack([X_S2, X_A])
    X_C2 = np.column_stack([X_S2, X_S2_idx, X_A])
    X_D = df[ae_cols].values.astype(np.float64)
    X_E = np.column_stack([X_D, X_A])
    y = df['sos'].values.astype(np.float64)

    return {
        'A_3DEP_morphometrics': X_A,
        'B_3DEP_plus_EVT': X_B,
        'C_Raw2017S2_plus_3DEP': X_C,
        'C2_Raw2017S2_Indices_plus_3DEP': X_C2,
        'C0_Raw2017S2_only': X_S2,
        'D_AE2017_only': X_D,
        'E_AE2017_plus_3DEP': X_E,
    }, y


def lonlat_to_local_m(lon, lat):
    lat0 = np.mean(lat)
    lon0 = np.mean(lon)
    x = (lon - lon0) * 111320.0 * math.cos(math.radians(lat0))
    y = (lat - lat0) * 111320.0
    return np.column_stack([x, y])


def eval_buffered_spatial_cv(df_dom, X_dict, y, buffer_m=800.0):
    coords = lonlat_to_local_m(df_dom['lon'].values, df_dom['lat'].values)
    # 4 spatial quadrants split at median x, y
    mx, my = np.median(coords[:, 0]), np.median(coords[:, 1])
    quad = ((coords[:, 0] >= mx).astype(int) * 2) + (coords[:, 1] >= my).astype(int)

    # Precompute pairwise distances for exact exclusion buffer
    dx = coords[:, 0:1] - coords[:, 0:1].T
    dy = coords[:, 1:2] - coords[:, 1:2].T
    dist = np.hypot(dx, dy)

    results = {}
    for mname, X in X_dict.items():
        results[mname] = {}
        for alg in ['Ridge', 'RF']:
            preds = np.full(len(y), np.nan)
            train_counts = []
            for q in range(4):
                te_m = (quad == q)
                # Exclude ANY training point within buffer_m of ANY test point!
                min_d_to_te = dist[:, te_m].min(axis=1)
                tr_m = (~te_m) & (min_d_to_te >= buffer_m)
                train_counts.append(int(tr_m.sum()))
                if alg == 'Ridge':
                    model = fit_spatial_ridge(X[tr_m], y[tr_m], quad[tr_m])
                else:
                    model = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features='sqrt',
                                                  n_jobs=-1, random_state=42)
                    model.fit(X[tr_m], y[tr_m])
                preds[te_m] = model.predict(X[te_m])
            ok = np.isfinite(preds)
            r2 = float(r2_score(y[ok], preds[ok]))
            rmse = float(root_mean_squared_error(y[ok], preds[ok]))
            results[mname][alg] = {
                'r2': round(r2, 4), 'rmse_d': round(rmse, 3),
                'mean_train_n': int(round(np.mean(train_counts)))
            }
    return results


def eval_loco(df_loco, X_dict, y):
    coves = sorted(df_loco['cove'].unique())
    results = {}
    for mname, X in X_dict.items():
        results[mname] = {}
        for alg in ['Ridge', 'RF']:
            preds = np.full(len(y), np.nan)
            per_cove = {}
            for c in coves:
                te_m = (df_loco['cove'].values == c)
                tr_m = ~te_m
                if alg == 'Ridge':
                    model = fit_spatial_ridge(X[tr_m], y[tr_m], df_loco['cove'].values[tr_m])
                else:
                    model = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features='sqrt',
                                                  n_jobs=-1, random_state=42)
                    model.fit(X[tr_m], y[tr_m])
                p = model.predict(X[te_m])
                preds[te_m] = p
                per_cove[c] = round(float(r2_score(y[te_m], p)), 4)
            results[mname][alg] = {
                'overall_r2': round(float(r2_score(y, preds)), 4),
                'overall_rmse_d': round(float(root_mean_squared_error(y, preds)), 3),
                'mean_cove_r2': round(float(np.mean(list(per_cove.values()))), 4),
                'per_cove_r2': per_cove,
            }
    return results


def run_interannual_delta_emb_test():
    """Test pixel-demeaned interannual Delta_emb_t -> Delta_SOS_t (2018-2024) on ~200 Domain pixels."""
    cache_dyn = os.path.join(C.CACHE_DIR, 'p6_interannual_emb_sos.json')
    if os.path.exists(cache_dyn):
        df_dyn = pd.DataFrame(json.load(open(cache_dyn)))
        print(f'[P6] Loaded cached interannual Delta_emb panel ({len(df_dyn)} pixel-years)')
    else:
        print('[P6] Extracting per-pixel annual S2 NDVI + AlphaEarth embeddings (2018-2024) for dynamic test...')
        dom = C.get_domain()
        # Sample 220 stratified NLCD-41 pixels across Domain elevation gradient (Lost Cove + Escarpment + Plateau)
        dem = C.get_dem_3dep().rename('elev')
        nlcd = C.get_nlcd_2021()
        decid = nlcd.eq(41)
        pts_fc = dem.updateMask(decid).sample(region=dom, scale=30, numPixels=220, seed=55, geometries=True, tileScale=4)
        p_feats = pts_fc.getInfo()['features']
        pts = [{'pid': i, 'lon': f['geometry']['coordinates'][0], 'lat': f['geometry']['coordinates'][1],
                'elev': f['properties']['elev']} for i, f in enumerate(p_feats)]
        fc_pts = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([p['lon'], p['lat']]), {'pid': p['pid']}) for p in pts])

        # Extract per-year AlphaEarth embeddings (2018-2024) at these 220 pixels (7 fast calls!)
        emb_rows = {}
        for yr in range(2018, 2025):
            e_img = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
                     .filterDate(f'{yr}-01-01', f'{yr+1}-01-01').filterBounds(dom).mosaic().select(AE_BANDS))
            ef = e_img.sampleRegions(fc_pts, scale=20, tileScale=4, geometries=False).getInfo()['features']
            for f in ef:
                p = f['properties']
                emb_rows[(int(p['pid']), yr)] = [p[b] for b in AE_BANDS]

        # Extract per-year S2 NDVI at these 220 pixels (batch 12 images at a time: 12*220=2640 < 5000!)
        rows = []
        for yr in range(2018, 2025):
            t0 = time.time()
            s2_yr = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterBounds(dom).filterDate(f'{yr}-01-01', f'{yr+1}-01-01')
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 55)))
            ids = s2_yr.aggregate_array('system:index').getInfo()
            sel = [i for i in ids if '16SEE' in i] or ids
            # Deduplicate by date prefix YYYYMMDD
            by_dt = {}
            for i in sel:
                by_dt.setdefault(i[:8], i)
            sel = sorted(by_dt.values())

            def _samp(img):
                scl = img.select('SCL')
                good = scl.eq(4).Or(scl.eq(5))
                ndvi = img.normalizedDifference(['B8', 'B4']).updateMask(good).rename('ndvi')
                return ndvi.sampleRegions(fc_pts, scale=20, tileScale=4, geometries=False).map(
                    lambda f: f.set('doy', img.date().getRelative('day', 'year'))
                )

            yr_obs = []
            for k in range(0, len(sel), 8):
                b = sel[k:k + 8]
                for attempt in range(4):
                    try:
                        fs = s2_yr.filter(ee.Filter.inList('system:index', b)).map(_samp).flatten().getInfo()['features']
                        break
                    except Exception as ex:
                        if attempt == 3:
                            raise
                        time.sleep(3 * (attempt + 1))
                for f in fs:
                    yr_obs.append(f['properties'])
            if not yr_obs:
                continue
            df_o = pd.DataFrame(yr_obs)
            ok_cnt = 0
            for pid, grp in df_o.groupby('pid'):
                fit = C.fit_double_logistic(grp['doy'].values, grp['ndvi'].values,
                                            min_obs=7, min_spring=2, min_autumn=2)
                if fit['qc_pass'] and (int(pid), yr) in emb_rows:
                    rec = {'pid': int(pid), 'yr': yr, 'sos': fit['sos50'], 'eos': fit['eos50'], 'r2': fit['r2']}
                    for b, v in zip(AE_BANDS, emb_rows[(int(pid), yr)]):
                        rec[b] = v
                    rows.append(rec)
                    ok_cnt += 1
            print(f'  Dynamic {yr}: {ok_cnt} QC-pass pixel fits in {time.time()-t0:.1f}s')
        df_dyn = pd.DataFrame(rows)
        json.dump(rows, open(cache_dyn, 'w'))

    # Keep pixels with >= 5 valid years
    cnts = df_dyn.groupby('pid')['yr'].count()
    keep_pids = cnts[cnts >= 5].index
    df_d = df_dyn[df_dyn['pid'].isin(keep_pids)].copy()

    # Strict LEAK-FREE demeaning (means always computed strictly inside training folds!)
    X_raw = df_d[AE_BANDS].values.astype(np.float64)
    y_raw = df_d['sos'].values.astype(np.float64)
    pids = df_d['pid'].values
    yrs = df_d['yr'].values

    # 1. Strict Leak-Free Pixel-Demeaned & Two-Way Demeaned Leave-One-Year-Out (LOYO) CV:
    #    For test year t*, pixel means are computed STRICTLY on training years (t != t*).
    def eval_loyo_strict(two_way=False):
        out = {}
        for alg in ['Ridge', 'RF']:
            preds = np.full(len(y_raw), np.nan)
            y_eval = np.full(len(y_raw), np.nan)
            for yr in np.unique(yrs):
                te = (yrs == yr)
                tr = ~te
                # Compute per-pixel baseline strictly on training years
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
                    mdl = RandomForestRegressor(n_estimators=150, min_samples_leaf=5, max_features='sqrt', n_jobs=-1, random_state=42)
                mdl.fit(X_tr, y_tr)
                preds[te] = mdl.predict(X_te)
                y_eval[te] = y_te
            out[alg] = {
                'loyo_r2': round(float(r2_score(y_eval, preds)), 4),
                'loyo_rmse_d': round(float(root_mean_squared_error(y_eval, preds)), 3),
                'loyo_r': round(float(np.corrcoef(y_eval, preds)[0, 1]), 4),
            }
        return out

    # 2. Strict Year-to-Year First-Difference LOYO (emb_t - emb_{t-1} -> sos_t - sos_{t-1}),
    #    holding out BOTH (yr, yr-1) and (yr+1, yr) from training so test year t* never touches train!
    df_s = df_d.sort_values(['pid', 'yr'])
    fd_rows = []
    for p, g in df_s.groupby('pid'):
        g = g.sort_values('yr')
        for i in range(1, len(g)):
            if g['yr'].iloc[i] - g['yr'].iloc[i - 1] == 1:
                dx = g[AE_BANDS].iloc[i].values.astype(np.float64) - g[AE_BANDS].iloc[i - 1].values.astype(np.float64)
                dy = float(g['sos'].iloc[i] - g['sos'].iloc[i - 1])
                fd_rows.append((int(p), int(g['yr'].iloc[i]), dy, dx))
    fd_yrs = np.array([r[1] for r in fd_rows])
    fd_y = np.array([r[2] for r in fd_rows])
    fd_X = np.vstack([r[3] for r in fd_rows])

    def eval_first_diff_loyo(year_demean=False):
        out = {}
        for alg in ['Ridge', 'RF']:
            preds = np.full(len(fd_y), np.nan)
            y_eval = np.full(len(fd_y), np.nan)
            for yr in np.unique(fd_yrs):
                te = (fd_yrs == yr)
                tr = (fd_yrs != yr) & (fd_yrs != yr + 1)
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
                    mdl = RandomForestRegressor(n_estimators=150, min_samples_leaf=5, max_features='sqrt', n_jobs=-1, random_state=42)
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

    # 3. Strict Leak-Free 5-fold GroupPixelCV (within-pixel demeaning has zero cross-pixel leakage;
    #    for two-way demeaning, per-year means are estimated strictly on training pixels!)
    X_pdm = np.zeros_like(X_raw)
    y_pdm = np.zeros_like(y_raw)
    for p in np.unique(pids):
        m = (pids == p)
        X_pdm[m] = X_raw[m] - X_raw[m].mean(axis=0)
        y_pdm[m] = y_raw[m] - y_raw[m].mean()

    def eval_group_pid_strict(two_way=False):
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
                    mdl = RandomForestRegressor(n_estimators=150, min_samples_leaf=5, max_features='sqrt', n_jobs=-1, random_state=42)
                mdl.fit(X_tr, y_tr)
                preds[te] = mdl.predict(X_te)
                y_eval[te] = y_te
            out[alg] = {
                'group_pixel_r2': round(float(r2_score(y_eval, preds)), 4),
                'group_pixel_rmse_d': round(float(root_mean_squared_error(y_eval, preds)), 3),
            }
        return out

    return {
        'n_pixels': int(len(keep_pids)),
        'n_pixel_years': int(len(df_d)),
        'pixel_demeaned_LOYO_strict': eval_loyo_strict(two_way=False),
        'twoway_demeaned_LOYO_strict': eval_loyo_strict(two_way=True),
        'first_diff_LOYO_strict': eval_first_diff_loyo(year_demean=False),
        'first_diff_year_demeaned_LOYO_strict': eval_first_diff_loyo(year_demean=True),
        'pixel_demeaned_GroupPixelCV_strict': eval_group_pid_strict(two_way=False),
        'twoway_demeaned_GroupPixelCV_strict': eval_group_pid_strict(two_way=True),
    }


def main():
    t0 = time.time()
    C.init_ee()
    df_dom, df_loco = sample_domain_and_loco()

    X_dom, y_dom = build_feature_matrices(df_dom)
    X_loco, y_loco = build_feature_matrices(df_loco)

    # 1. Empirical semivariogram of SOS (and terrain-residual SOS!)
    coords_dom = lonlat_to_local_m(df_dom['lon'].values, df_dom['lat'].values)
    vario_sos = C.empirical_variogram(coords_dom, y_dom, max_dist=3500.0, n_bins=14)
    # Residual after 3DEP morphometrics
    m_ter = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25))).fit(X_dom['A_3DEP_morphometrics'], y_dom)
    resid_ter = y_dom - m_ter.predict(X_dom['A_3DEP_morphometrics'])
    vario_resid = C.empirical_variogram(coords_dom, resid_ter, max_dist=3500.0, n_bins=14)
    eff_buf = max(800.0, min(1500.0, float(vario_resid.get('spherical_range_m') or 800.0)))
    print(f'[P6] Variogram range: SOS={vario_sos["spherical_range_m"]}m | Terrain-resid={vario_resid["spherical_range_m"]}m -> buffer={eff_buf:.0f}m')

    # 2. Spatial CV: unbuffered (b=0), b=800m, b=1200m, and b=eff_buf (variogram-guided)
    cv_0 = eval_buffered_spatial_cv(df_dom, X_dom, y_dom, buffer_m=0.0)
    cv_800 = eval_buffered_spatial_cv(df_dom, X_dom, y_dom, buffer_m=800.0)
    cv_1200 = eval_buffered_spatial_cv(df_dom, X_dom, y_dom, buffer_m=1200.0)
    cv_eff = eval_buffered_spatial_cv(df_dom, X_dom, y_dom, buffer_m=eff_buf)

    # 3. LOCO Regional Transfer
    loco_res = eval_loco(df_loco, X_loco, y_loco)

    # 4. Pixel-Demeaned Interannual Dynamic Test (strict leak-free in-fold demeaning)
    dyn_res = run_interannual_delta_emb_test()

    # Evaluate Hard Pre-Registered Keep/Drop Gate
    def best_alg_r2(res_dict, key, metric='r2'):
        return max(res_dict[key]['Ridge'][metric], res_dict[key]['RF'][metric])

    c_b800 = max(best_alg_r2(cv_800, 'C_Raw2017S2_plus_3DEP', 'r2'),
                 best_alg_r2(cv_800, 'C2_Raw2017S2_Indices_plus_3DEP', 'r2'))
    de_b800 = max(best_alg_r2(cv_800, 'D_AE2017_only', 'r2'), best_alg_r2(cv_800, 'E_AE2017_plus_3DEP', 'r2'))
    delta_b800 = round(de_b800 - c_b800, 4)

    c_loco = max(best_alg_r2(loco_res, 'C_Raw2017S2_plus_3DEP', 'overall_r2'),
                 best_alg_r2(loco_res, 'C2_Raw2017S2_Indices_plus_3DEP', 'overall_r2'))
    de_loco = max(best_alg_r2(loco_res, 'D_AE2017_only', 'overall_r2'), best_alg_r2(loco_res, 'E_AE2017_plus_3DEP', 'overall_r2'))
    delta_loco = round(de_loco - c_loco, 4)

    dyn_loyo_r2 = max(dyn_res['pixel_demeaned_LOYO_strict']['Ridge']['loyo_r2'],
                      dyn_res['pixel_demeaned_LOYO_strict']['RF']['loyo_r2'])
    dyn_2w_loyo_r2 = max(dyn_res['twoway_demeaned_LOYO_strict']['Ridge']['loyo_r2'],
                         dyn_res['twoway_demeaned_LOYO_strict']['RF']['loyo_r2'])
    dyn_fd_r2 = max(dyn_res['first_diff_LOYO_strict']['Ridge']['loyo_r2'],
                    dyn_res['first_diff_LOYO_strict']['RF']['loyo_r2'])
    dyn_2w_gp_r2 = max(dyn_res['twoway_demeaned_GroupPixelCV_strict']['Ridge']['group_pixel_r2'],
                       dyn_res['twoway_demeaned_GroupPixelCV_strict']['RF']['group_pixel_r2'])

    keep_spatial = (delta_b800 >= 0.03) and (delta_loco >= 0.03)
    keep_dynamic = (dyn_loyo_r2 >= 0.15) or (dyn_2w_loyo_r2 >= 0.15)
    verdict = 'KEEP_ALPHAEARTH_SIDEBAR' if (keep_spatial or keep_dynamic) else 'DROP_ALPHAEARTH_OR_NEGATIVE_RESULT'

    out = {
        'experiment': 'P6_alphaearth_gate',
        'elapsed_s': round(time.time() - t0, 1),
        'n_domain_pixels': int(len(df_dom)),
        'n_loco_pixels': int(len(df_loco)),
        'variogram_sos': vario_sos,
        'variogram_terrain_resid': vario_resid,
        'effective_buffer_m': round(eff_buf, 1),
        'spatial_cv_unbuffered_0m': cv_0,
        'spatial_cv_buffered_800m': cv_800,
        'spatial_cv_buffered_1200m': cv_1200,
        'spatial_cv_buffered_eff': cv_eff,
        'loco_4coves_transfer': loco_res,
        'interannual_delta_emb': dyn_res,
        'gate_check': {
            'b800_best_C_Raw2017S2_3DEP_r2': c_b800,
            'b800_best_AE2017_DE_r2': de_b800,
            'b800_delta_AE_minus_RawS2': delta_b800,
            'loco_best_C_Raw2017S2_3DEP_r2': c_loco,
            'loco_best_AE2017_DE_r2': de_loco,
            'loco_delta_AE_minus_RawS2': delta_loco,
            'dyn_pixel_demeaned_LOYO_strict_r2': dyn_loyo_r2,
            'dyn_twoway_demeaned_LOYO_strict_r2': dyn_2w_loyo_r2,
            'dyn_first_diff_LOYO_strict_r2': dyn_fd_r2,
            'dyn_twoway_demeaned_GroupPixel_strict_r2': dyn_2w_gp_r2,
            'keep_spatial_pass': keep_spatial,
            'keep_dynamic_pass': keep_dynamic,
            'verdict': verdict,
        }
    }
    C.save_json('p6.json', out)
    print('\n=== P6 SUMMARY ===')
    print(json.dumps(out['gate_check'], indent=2))
    print('Buffered 800m CV:', json.dumps(cv_800, indent=2))
    print('LOCO Transfer:', json.dumps({k: {a: v['overall_r2'] for a, v in d.items()} for k, d in loco_res.items()}, indent=2))


if __name__ == '__main__':
    main()
