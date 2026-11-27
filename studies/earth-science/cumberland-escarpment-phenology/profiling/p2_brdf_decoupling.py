"""Experiment P2: Optical/BRDF Invariance (SCS+C & HLS NBAR) & Vertical Cold-Pool Decoupling.

Fast, zero-concurrency-error architecture:
1. Sample ~1,600 stratified NLCD-41 deciduous pixels across all 15 Domain strata (Signal A: Lost/OpenN
   overall & EVT 3321/3317; Signal B: Escarpment steep, PlatFlat, PlatSteep overall, EVT 3317, and
   4 aspect quadrants N/E/S/W) with exact 10m 3DEP elevation, slope, aspect, and LANDFIRE EVT.
2. Extract per-date Sentinel-2 SR (B4, B5, B8, B8A, B11, SCL + SZA, SAA) at those exact points (2018-2024)
   and compute exact Soenen et al. (2005) SCS+C topographic correction locally per acquisition date.
3. Extract per-date NASA HLS v2.0 c-factor NBAR (HLSL30 + HLSS30, Fmask-filtered) at those exact points.
4. Fit 6-parameter double-logistic curves (15%/50%/85% crossings) per (mode, index, stratum, year)
   for NDVI, NDRE, NIRv, NDMI.
5. Evaluate Signal A invariance, Signal B aspect-quadrant invariance, and vertical cold-pool decoupling
   (dSOS_NDVI - dSOS_NDRE) in freeze vs. mild springs (NOAA USC00408184).
"""
import os
import sys
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ee
import common as C


def sample_domain_points(dom):
    cache_pts = os.path.join(C.CACHE_DIR, 'p2_domain_points.json')
    if os.path.exists(cache_pts):
        pts = json.load(open(cache_pts))
        print(f'[P2] Loaded {len(pts)} cached stratified Domain points')
        return pts

    print('[P2] Sampling stratified NLCD-41 points across 15 Domain strata...')
    dem = C.get_dem_3dep().rename('elev')
    slope = ee.Terrain.slope(dem).rename('slope')
    aspect = ee.Terrain.aspect(dem).rename('aspect')
    nlcd = C.get_nlcd_2021()
    evt = C.get_evt().rename('evt')
    decid = nlcd.eq(41)

    lost_box = ee.Geometry.Rectangle([-85.945, 35.121, -85.880, 35.168])
    openn_box = ee.Geometry.Rectangle([-85.991, 35.178, -85.859, 35.231])
    in_lost = ee.Image.constant(1).clip(lost_box).unmask(0).eq(1)
    in_openn = ee.Image.constant(1).clip(openn_box).unmask(0).eq(1)

    band_a = decid.And(dem.gte(330)).And(dem.lte(385))
    plat = decid.And(dem.gte(550))
    plat_flat = plat.And(slope.lte(8))
    plat_steep = plat.And(slope.gt(10))
    esc_steep = decid.And(dem.gte(480)).And(dem.lte(540)).And(slope.gt(10))

    # Build integer stratum classes to sample cleanly (non-overlapping core strata + aspect sub-strata)
    # 1=lost_3321, 2=lost_3317, 3=lost_other, 4=openn_3321, 5=openn_3317, 6=openn_other,
    # 7=esc_steep, 8=plat_flat, 9=steep_N, 10=steep_E, 11=steep_S, 12=steep_W
    cls = ee.Image.constant(0).toInt()
    cls = cls.where(band_a.And(in_lost), 3)
    cls = cls.where(band_a.And(in_lost).And(evt.eq(3321)), 1)
    cls = cls.where(band_a.And(in_lost).And(evt.eq(3317)), 2)
    cls = cls.where(band_a.And(in_openn), 6)
    cls = cls.where(band_a.And(in_openn).And(evt.eq(3321)), 4)
    cls = cls.where(band_a.And(in_openn).And(evt.eq(3317)), 5)
    cls = cls.where(esc_steep, 7)
    cls = cls.where(plat_flat, 8)
    cls = cls.where(plat_steep.And(aspect.gte(315).Or(aspect.lt(45))), 9)
    cls = cls.where(plat_steep.And(aspect.gte(45)).And(aspect.lt(135)), 10)
    cls = cls.where(plat_steep.And(aspect.gte(135)).And(aspect.lt(225)), 11)
    cls = cls.where(plat_steep.And(aspect.gte(225)).And(aspect.lt(315)), 12)
    cls = cls.updateMask(cls.gt(0)).rename('cls')

    stack = ee.Image.cat([cls, dem, slope, aspect, evt])
    fc = stack.stratifiedSample(
        numPoints=75, classBand='cls', region=dom, scale=20,
        seed=42, geometries=True, tileScale=4
    )
    feats = fc.getInfo()['features']
    pts = []
    for i, f in enumerate(feats):
        lon, lat = f['geometry']['coordinates']
        p = f['properties']
        pts.append({
            'pid': i, 'lon': round(lon, 6), 'lat': round(lat, 6),
            'cls': int(p['cls']), 'elev': round(float(p['elev']), 2),
            'slope': round(float(p['slope']), 2), 'aspect': round(float(p['aspect']), 2),
            'evt': int(p['evt']) if p.get('evt') is not None else -1,
        })
    json.dump(pts, open(cache_pts, 'w'))
    print(f'  Sampled {len(pts)} points across 12 mutually-exclusive classes')
    return pts


def assign_strata(df_pts):
    """Map each point to all overlapping analytical strata."""
    c = df_pts['cls'].values
    evt = df_pts['evt'].values
    strata_masks = {
        'lost_all': np.isin(c, [1, 2, 3]),
        'openn_all': np.isin(c, [4, 5, 6]),
        'lost_3321': c == 1,
        'openn_3321': c == 4,
        'lost_3317': c == 2,
        'openn_3317': c == 5,
        'esc_steep': c == 7,
        'plat_flat': c == 8,
        'plat_steep': np.isin(c, [9, 10, 11, 12]),
        'plat_flat_3317': (c == 8) & (evt == 3317),
        'plat_steep_3317': np.isin(c, [9, 10, 11, 12]) & (evt == 3317),
        'steep_N': c == 9,
        'steep_E': c == 10,
        'steep_S': c == 11,
        'steep_W': c == 12,
    }
    return strata_masks


def fetch_s2_points_and_compute_scsc(pts, dom):
    cache_path = os.path.join(C.CACHE_DIR, 'p2_s2_stratum_ts.json')
    if os.path.exists(cache_path):
        print('[P2] Loaded cached S2 Raw + SCS+C stratum time series')
        return pd.DataFrame(json.load(open(cache_path)))

    print('[P2] Extracting S2 SR at stratified points (2018-2024) and computing exact SCS+C locally...')
    df_pts = pd.DataFrame(pts).set_index('pid')
    strata_masks = assign_strata(df_pts)
    fc_pts = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([p['lon'], p['lat']]), {'pid': p['pid']})
        for p in pts
    ])

    slope_rad = np.deg2rad(df_pts['slope'].values)
    aspect_rad = np.deg2rad(df_pts['aspect'].values)
    cos_a = np.cos(slope_rad)
    sin_a = np.sin(slope_rad)
    pid_order = df_pts.index.values

    def extract_year(yr):
        t0 = time.time()
        s2_yr = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                 .filterBounds(dom)
                 .filterDate(f'{yr}-01-01', f'{yr+1}-01-01')
                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
                 .select(['B4', 'B5', 'B8', 'B8A', 'B11', 'SCL']))
        # Fetch metadata for SZA, SAA, DOY keyed by system:index
        meta_list = s2_yr.map(lambda im: ee.Feature(None, {
            'id': im.get('system:index'),
            'doy': im.date().getRelative('day', 'year'),
            'sza': im.get('MEAN_SOLAR_ZENITH_ANGLE'),
            'saa': im.get('MEAN_SOLAR_AZIMUTH_ANGLE'),
        })).getInfo()['features']
        meta = {f['properties']['id']: f['properties'] for f in meta_list}

        # Deduplicate overlapping MGRS tiles by filtering to 16SEE (covers entire Sewanee Domain) or grouping
        # Process in batches of 4 images (4 * 900 = 3600 < 5000 GEE getInfo limit!)
        year_rows = []
        ids_all = sorted(meta.keys())
        # Keep only one tile per doy (prefer 16SEE which covers the full domain)
        by_doy = {}
        for iid in ids_all:
            d = meta[iid]['doy']
            if d not in by_doy or '16SEE' in iid:
                by_doy[d] = iid
        sel_ids = [by_doy[d] for d in sorted(by_doy.keys())]

        def _samp(img):
            scl = img.select('SCL')
            good = scl.eq(4).Or(scl.eq(5))
            im_m = img.select(['B4', 'B5', 'B8', 'B8A', 'B11']).updateMask(good)
            return im_m.sampleRegions(
                collection=fc_pts, scale=20, tileScale=4, geometries=False
            ).map(lambda f: f.set('iid', img.get('system:index')))

        for k in range(0, len(sel_ids), 4):
            batch_ids = sel_ids[k:k + 4]
            sub_col = s2_yr.filter(ee.Filter.inList('system:index', batch_ids))
            feats = sub_col.map(_samp).flatten().getInfo()['features']
            if not feats:
                continue
            df_c = pd.DataFrame([f['properties'] for f in feats])
            for iid, grp in df_c.groupby('iid'):
                if iid not in meta or len(grp) < 120:
                    continue
                md = meta[iid]
                sza = math.radians(float(md['sza']))
                saa = math.radians(float(md['saa']))
                doy = int(md['doy'])

                # Align to pid_order
                g = grp.drop_duplicates('pid').set_index('pid')
                present = np.isin(pid_order, g.index.values)
                if present.sum() < 120:
                    continue
                pids_p = pid_order[present]
                sub_g = g.loc[pids_p]
                B4 = sub_g['B4'].values.astype(np.float64) * 1e-4
                B5 = sub_g['B5'].values.astype(np.float64) * 1e-4
                B8 = sub_g['B8'].values.astype(np.float64) * 1e-4
                B8A = sub_g['B8A'].values.astype(np.float64) * 1e-4
                B11 = sub_g['B11'].values.astype(np.float64) * 1e-4

                ca = cos_a[present]
                sa = sin_a[present]
                ar = aspect_rad[present]
                cos_sz = math.cos(sza)
                sin_sz = math.sin(sza)
                cos_i = cos_sz * ca + sin_sz * sa * np.cos(saa - ar)
                ill_ok = cos_i >= 0.15
                if ill_ok.sum() < 100:
                    continue

                # Exact Soenen et al. (2005) SCS+C per band
                def scsc_band(b):
                    x = cos_i[ill_ok]
                    y = b[ill_ok]
                    ok = np.isfinite(x) & np.isfinite(y)
                    if ok.sum() < 50:
                        return b
                    slp, icp = np.polyfit(x[ok], y[ok], 1)
                    c_val = float(np.clip(icp / slp, 0.05, 5.0)) if slp > 0.005 else 1.0
                    corr = b * (cos_sz * ca + c_val) / (cos_i + c_val)
                    return np.clip(corr, 0.0, 1.2)

                B4c, B5c, B8c, B8Ac, B11c = [scsc_band(b) for b in (B4, B5, B8, B8A, B11)]

                def calc_vis(r, re, nir, nir8a, sw):
                    ndvi = (nir - r) / np.maximum(nir + r, 1e-6)
                    ndre = (nir8a - re) / np.maximum(nir8a + re, 1e-6)
                    nirv = ndvi * nir
                    ndmi = (nir - sw) / np.maximum(nir + sw, 1e-6)
                    return {'ndvi': ndvi, 'ndre': ndre, 'nirv': nirv, 'ndmi': ndmi}

                vi_raw = calc_vis(B4, B5, B8, B8A, B11)
                vi_scsc = calc_vis(B4c, B5c, B8c, B8Ac, B11c)

                rec = {'yr': yr, 'doy': doy, 'iid': iid, 'n_pts': int(ill_ok.sum())}
                for sname, smask in strata_masks.items():
                    m_loc = smask[present] & ill_ok
                    if m_loc.sum() < 8:
                        continue
                    for idx in ['ndvi', 'ndre', 'nirv', 'ndmi']:
                        rec[f'{sname}__raw_{idx}'] = float(np.nanmean(vi_raw[idx][m_loc]))
                        rec[f'{sname}__scsc_{idx}'] = float(np.nanmean(vi_scsc[idx][m_loc]))
                year_rows.append(rec)
        print(f'  S2 {yr}: {len(year_rows)} clear acquisitions in {time.time()-t0:.1f}s')
        return year_rows

    all_rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(extract_year, yr) for yr in range(2018, 2025)]
        for fut in as_completed(futs):
            all_rows.extend(fut.result())

    json.dump(all_rows, open(cache_path, 'w'))
    return pd.DataFrame(all_rows)


def fetch_hls_points(pts, dom):
    cache_path = os.path.join(C.CACHE_DIR, 'p2_hls_stratum_ts.json')
    if os.path.exists(cache_path):
        print('[P2] Loaded cached HLS NBAR stratum time series')
        return pd.DataFrame(json.load(open(cache_path)))

    print('[P2] Extracting NASA HLS v2.0 c-factor NBAR (HLSL30 + HLSS30, 2018-2024)...')
    df_pts = pd.DataFrame(pts).set_index('pid')
    strata_masks = assign_strata(df_pts)
    pid_order = df_pts.index.values
    fc_pts = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([p['lon'], p['lat']]), {'pid': p['pid']})
        for p in pts
    ])

    def hls_clear(img):
        fm = img.select('Fmask').toInt()
        bad = fm.bitwiseAnd((1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)).neq(0)
        return bad.Not()

    def extract_year(yr):
        t0 = time.time()
        year_rows = []
        # Filter to MGRS tile 16SEE (or deduplicate per doy) and batch 4 images at a time (< 5000 elements)
        s30_yr = (ee.ImageCollection('NASA/HLS/HLSS30/v002')
                  .filterBounds(dom).filterDate(f'{yr}-01-01', f'{yr+1}-01-01')
                  .filter(ee.Filter.lt('CLOUD_COVERAGE', 60)))
        l30_yr = (ee.ImageCollection('NASA/HLS/HLSL30/v002')
                  .filterBounds(dom).filterDate(f'{yr}-01-01', f'{yr+1}-01-01')
                  .filter(ee.Filter.lt('CLOUD_COVERAGE', 60)))

        def samp_s30(img):
            g = hls_clear(img)
            im = ee.Image.cat([
                img.normalizedDifference(['B8A', 'B4']).rename('ndvi'),
                img.normalizedDifference(['B8A', 'B5']).rename('ndre'),
                img.normalizedDifference(['B8A', 'B4']).multiply(img.select('B8A')).rename('nirv'),
                img.normalizedDifference(['B8A', 'B11']).rename('ndmi'),
            ]).updateMask(g)
            return im.sampleRegions(fc_pts, scale=30, tileScale=4, geometries=False).map(
                lambda f: f.set('doy', img.date().getRelative('day', 'year'), 'iid', img.get('system:index'))
            )

        def samp_l30(img):
            g = hls_clear(img)
            im = ee.Image.cat([
                img.normalizedDifference(['B5', 'B4']).rename('ndvi'),
                ee.Image.constant(-9999.0).rename('ndre'),
                img.normalizedDifference(['B5', 'B4']).multiply(img.select('B5')).rename('nirv'),
                img.normalizedDifference(['B5', 'B6']).rename('ndmi'),
            ]).updateMask(g)
            return im.sampleRegions(fc_pts, scale=30, tileScale=4, geometries=False).map(
                lambda f: f.set('doy', img.date().getRelative('day', 'year'), 'iid', img.get('system:index'))
            )

        for col, samp_fn in [(s30_yr, samp_s30), (l30_yr, samp_l30)]:
            all_ids = col.aggregate_array('system:index').getInfo()
            # Keep tile 16SEE (covers full Sewanee Domain) if present, else dedup by date prefix
            sel = [i for i in all_ids if '16SEE' in i] or all_ids
            for k in range(0, len(sel), 4):
                b_ids = sel[k:k + 4]
                feats = col.filter(ee.Filter.inList('system:index', b_ids)).map(samp_fn).flatten().getInfo()['features']
                if not feats:
                    continue
                df_c = pd.DataFrame([f['properties'] for f in feats])
                for (iid, doy), grp in df_c.groupby(['iid', 'doy']):
                    if len(grp) < 100:
                        continue
                    g = grp.drop_duplicates('pid').set_index('pid')
                    present = np.isin(pid_order, g.index.values)
                    if present.sum() < 100:
                        continue
                    sub_g = g.loc[pid_order[present]]
                    rec = {'yr': yr, 'doy': int(doy), 'iid': iid, 'n_pts': int(present.sum())}
                    for sname, smask in strata_masks.items():
                        m_loc = smask[present]
                        if m_loc.sum() < 8:
                            continue
                        for idx in ['ndvi', 'ndre', 'nirv', 'ndmi']:
                            vals = sub_g[idx].values[m_loc].astype(np.float64)
                            vals = vals[vals > -100]
                            if len(vals) >= 8:
                                rec[f'{sname}__hls_{idx}'] = float(np.nanmean(vals))
                    year_rows.append(rec)
        print(f'  HLS {yr}: {len(year_rows)} clear acquisitions in {time.time()-t0:.1f}s')
        return year_rows

    all_rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(extract_year, yr) for yr in range(2018, 2025)]
        for fut in as_completed(futs):
            all_rows.extend(fut.result())

    json.dump(all_rows, open(cache_path, 'w'))
    return pd.DataFrame(all_rows)


def fit_all_strata(df, mode_prefixes, strata):
    num_cols = [c for c in df.columns if '__' in c]
    df_day = df.groupby(['yr', 'doy'])[num_cols].median().reset_index()

    fits = []
    for yr, grp in df_day.groupby('yr'):
        for pref in mode_prefixes:
            for idx in ['ndvi', 'ndre', 'nirv', 'ndmi']:
                for s in strata:
                    col = f'{s}__{pref}_{idx}'
                    if col not in grp.columns:
                        continue
                    res = C.fit_double_logistic(grp['doy'].values, grp[col].values,
                                                min_obs=6, min_spring=2, min_autumn=2)
                    fits.append({'yr': int(yr), 'mode': pref, 'index': idx, 'stratum': s, **res})
    return pd.DataFrame(fits)


def summarize_contrast(df_fit, s1, s0):
    out = {}
    for (mode, idx), grp in df_fit.groupby(['mode', 'index']):
        a = grp[(grp['stratum'] == s1) & (grp['qc_pass'])].set_index('yr')
        b = grp[(grp['stratum'] == s0) & (grp['qc_pass'])].set_index('yr')
        yrs = sorted(a.index.intersection(b.index))
        if not yrs:
            continue
        rec = {'n_yrs': len(yrs), 'years': yrs}
        for m in ['sos15', 'sos50', 'sos85', 'eos85', 'eos50', 'eos15', 'gsl']:
            diff = (a.loc[yrs, m] - b.loc[yrs, m]).values
            t_val, p_val = stats.ttest_1samp(diff, 0.0) if len(diff) >= 3 else (np.nan, np.nan)
            rec[f'd_{m}_mean'] = round(float(np.mean(diff)), 3)
            rec[f'd_{m}_sd'] = round(float(np.std(diff, ddof=1)), 3) if len(diff) > 1 else 0.0
            rec[f'd_{m}_p'] = float(f'{p_val:.4g}') if np.isfinite(p_val) else None
            if m in ['sos50', 'eos50', 'gsl']:
                rec[f'd_{m}_by_yr'] = {int(y): round(float(v), 2) for y, v in zip(yrs, diff)}
        out[f'{mode}_{idx}'] = rec
    return out


def vertical_decoupling_test(df_fit):
    noaa = C.load_noaa_sewanee()
    apr = noaa[(noaa['year'].between(2018, 2024)) & (noaa['month'] == 4)]
    mar_apr = noaa[(noaa['year'].between(2018, 2024)) & (noaa['doy'].between(75, 130))]

    wx_rows = []
    for yr in range(2018, 2025):
        a = apr[apr['year'] == yr]
        ma = mar_apr[mar_apr['year'] == yr]
        post_bud = ma[ma['doy'].between(100, 125)]
        wx_rows.append({
            'yr': yr,
            'apr_tmin_min_C': round(float(a['TMIN'].min()), 2),
            'post_doy100_tmin_min_C': round(float(post_bud['TMIN'].min()), 2),
            'has_post_doy100_frost_le1C': int((post_bud['TMIN'] <= 1.0).any()),
            'apr_freeze_days_le0': int((a['TMIN'] <= 0.0).sum()),
            'apr_frost_days_le2': int((a['TMIN'] <= 2.0).sum()),
            'doy80_125_frost_le2': int((ma[ma['doy'].between(80, 125)]['TMIN'] <= 2.0).sum()),
            'apr_mean_tmin_C': round(float(a['TMIN'].mean()), 2),
        })
    df_wx = pd.DataFrame(wx_rows).set_index('yr')

    results = {}
    for mode in ['raw', 'scsc', 'hls']:
        rows = []
        for yr in range(2018, 2025):
            def get_val(idx, st):
                sub = df_fit[(df_fit['mode'] == mode) & (df_fit['index'] == idx) &
                             (df_fit['stratum'] == st) & (df_fit['yr'] == yr) & (df_fit['qc_pass'])]
                return float(sub['sos50'].iloc[0]) if len(sub) else np.nan
            dsos_ndvi = get_val('ndvi', 'lost_all') - get_val('ndvi', 'openn_all')
            dsos_ndre = get_val('ndre', 'lost_all') - get_val('ndre', 'openn_all')
            decoup = dsos_ndvi - dsos_ndre
            if np.isfinite(decoup):
                rows.append({
                    'yr': yr,
                    'dSOS_ndvi': round(dsos_ndvi, 3),
                    'dSOS_ndre': round(dsos_ndre, 3),
                    'decoupling_ndvi_minus_ndre': round(decoup, 3),
                    **df_wx.loc[yr].to_dict(),
                })
        if len(rows) >= 4:
            dfd = pd.DataFrame(rows)
            freeze = dfd[dfd['apr_tmin_min_C'] <= 0.5]['decoupling_ndvi_minus_ndre']
            mild = dfd[dfd['apr_tmin_min_C'] > 0.5]['decoupling_ndvi_minus_ndre']
            pb_frz = dfd[dfd['has_post_doy100_frost_le1C'] == 1]['decoupling_ndvi_minus_ndre']
            pb_mld = dfd[dfd['has_post_doy100_frost_le1C'] == 0]['decoupling_ndvi_minus_ndre']
            r_tmin, p_tmin = stats.pearsonr(dfd['apr_tmin_min_C'], dfd['decoupling_ndvi_minus_ndre'])
            r_pb, p_pb = stats.pearsonr(dfd['post_doy100_tmin_min_C'], dfd['decoupling_ndvi_minus_ndre'])
            r_fd, p_fd = stats.pearsonr(dfd['apr_frost_days_le2'], dfd['decoupling_ndvi_minus_ndre'])
            t_fm, p_fm = stats.ttest_ind(freeze, mild, equal_var=False) if (len(freeze) >= 2 and len(mild) >= 2) else (np.nan, np.nan)
            _, p_pbfm = stats.ttest_ind(pb_frz, pb_mld, equal_var=False) if (len(pb_frz) >= 2 and len(pb_mld) >= 2) else (np.nan, np.nan)
            results[mode] = {
                'n_yrs': len(dfd),
                'mean_decoupling_d': round(float(dfd['decoupling_ndvi_minus_ndre'].mean()), 3),
                'freeze_springs_mean_d': round(float(freeze.mean()), 3) if len(freeze) else None,
                'freeze_springs_n': int(len(freeze)),
                'mild_springs_mean_d': round(float(mild.mean()), 3) if len(mild) else None,
                'mild_springs_n': int(len(mild)),
                'freeze_minus_mild_d': round(float(freeze.mean() - mild.mean()), 3) if (len(freeze) and len(mild)) else None,
                'freeze_vs_mild_p': float(f'{p_fm:.4g}') if np.isfinite(p_fm) else None,
                'r_with_apr_tmin_min': round(float(r_tmin), 3),
                'p_with_apr_tmin_min': float(f'{p_tmin:.4g}'),
                'r_with_apr_frost_days_le2': round(float(r_fd), 3),
                'p_with_apr_frost_days_le2': float(f'{p_fd:.4g}'),
                'exploratory_post_doy100_frost_mean_d': round(float(pb_frz.mean()), 3) if len(pb_frz) else None,
                'exploratory_no_post_doy100_frost_mean_d': round(float(pb_mld.mean()), 3) if len(pb_mld) else None,
                'exploratory_post_doy100_diff_p': float(f'{p_pbfm:.4g}') if np.isfinite(p_pbfm) else None,
                'exploratory_r_post_doy100_tmin': round(float(r_pb), 3),
                'exploratory_p_post_doy100_tmin': float(f'{p_pb:.4g}'),
                'by_year': rows,
            }
    return results


def main():
    t0 = time.time()
    C.init_ee()
    dom = C.get_domain()
    pts = sample_domain_points(dom)
    strata = list(assign_strata(pd.DataFrame(pts)).keys())

    df_s2 = fetch_s2_points_and_compute_scsc(pts, dom)
    df_hls = fetch_hls_points(pts, dom)

    fit_s2 = fit_all_strata(df_s2, ['raw', 'scsc'], strata)
    fit_hls = fit_all_strata(df_hls, ['hls'], strata)
    df_fit = pd.concat([fit_s2, fit_hls], ignore_index=True)

    sig_a_all = summarize_contrast(df_fit, 'lost_all', 'openn_all')
    sig_a_3321 = summarize_contrast(df_fit, 'lost_3321', 'openn_3321')
    sig_a_3317 = summarize_contrast(df_fit, 'lost_3317', 'openn_3317')

    sig_b_all = summarize_contrast(df_fit, 'plat_flat', 'plat_steep')
    sig_b_3317 = summarize_contrast(df_fit, 'plat_flat_3317', 'plat_steep_3317')
    sig_b_aspects = {q: summarize_contrast(df_fit, 'plat_flat', f'steep_{q}') for q in ['N', 'E', 'S', 'W']}
    sig_b_contact = summarize_contrast(df_fit, 'plat_steep', 'esc_steep')
    decoupling = vertical_decoupling_test(df_fit)

    a_scsc_ndre = sig_a_all.get('scsc_ndre', {})
    a_raw_ndre = sig_a_all.get('raw_ndre', {})
    a_hls_ndvi = sig_a_all.get('hls_ndvi', {})
    a_hls_ndre = sig_a_all.get('hls_ndre', {})

    b_scsc_ndre = sig_b_all.get('scsc_ndre', {})
    b_hls_ndre = sig_b_all.get('hls_ndre', {})
    b_asp_scsc_ndre = {q: sig_b_aspects[q].get('scsc_ndre', {}).get('d_sos50_mean') for q in ['N', 'E', 'S', 'W']}
    b_asp_hls_ndvi = {q: sig_b_aspects[q].get('hls_ndvi', {}).get('d_sos50_mean') for q in ['N', 'E', 'S', 'W']}

    sig_a_pass = (
        (a_scsc_ndre.get('d_sos50_mean', 0) >= 2.5)
        and (a_scsc_ndre.get('d_eos50_mean', 0) <= -3.5)
        and (a_hls_ndvi.get('d_sos50_mean', 0) >= 2.5)
        and (a_hls_ndvi.get('d_eos50_mean', 0) <= -3.5)
    )
    asp_vals = [v for v in b_asp_scsc_ndre.values() if v is not None]
    sig_b_pass = (
        (b_scsc_ndre.get('d_sos50_mean', 0) >= 2.0)
        and len(asp_vals) == 4
        and all(v >= 1.5 for v in asp_vals)
    )
    sig_b_reject = (b_scsc_ndre.get('d_sos50_mean', 99) < 1.0) or any(v <= 0.0 for v in asp_vals)

    out = {
        'experiment': 'P2_brdf_decoupling',
        'elapsed_s': round(time.time() - t0, 1),
        'n_points_sampled': len(pts),
        'signal_A_all': sig_a_all,
        'signal_A_EVT3321': sig_a_3321,
        'signal_A_EVT3317': sig_a_3317,
        'signal_B_all': sig_b_all,
        'signal_B_EVT3317': sig_b_3317,
        'signal_B_by_aspect': sig_b_aspects,
        'signal_B_contact_steep_only': sig_b_contact,
        'vertical_decoupling': decoupling,
        'gate_check': {
            'sigA_scsc_ndre_dSOS': a_scsc_ndre.get('d_sos50_mean'),
            'sigA_scsc_ndre_dEOS': a_scsc_ndre.get('d_eos50_mean'),
            'sigA_scsc_retention_SOS_pct': round(100.0 * a_scsc_ndre.get('d_sos50_mean', 0) / max(1e-6, a_raw_ndre.get('d_sos50_mean', 1)), 1) if a_raw_ndre else None,
            'sigA_scsc_retention_EOS_pct': round(100.0 * a_scsc_ndre.get('d_eos50_mean', 0) / min(-1e-6, a_raw_ndre.get('d_eos50_mean', -1)), 1) if a_raw_ndre else None,
            'sigA_hls_ndvi_dSOS': a_hls_ndvi.get('d_sos50_mean'),
            'sigA_hls_ndvi_dEOS': a_hls_ndvi.get('d_eos50_mean'),
            'sigA_hls_ndre_dSOS': a_hls_ndre.get('d_sos50_mean'),
            'sigA_hls_ndre_dEOS': a_hls_ndre.get('d_eos50_mean'),
            'sigA_pass': sig_a_pass,
            'sigB_scsc_ndre_dSOS': b_scsc_ndre.get('d_sos50_mean'),
            'sigB_hls_ndre_dSOS': b_hls_ndre.get('d_sos50_mean'),
            'sigB_aspect_scsc_ndre_dSOS': b_asp_scsc_ndre,
            'sigB_aspect_hls_ndvi_dSOS': b_asp_hls_ndvi,
            'sigB_pass': sig_b_pass,
            'sigB_rejected_as_brdf': sig_b_reject,
            'verdict': 'PASS_BOTH' if (sig_a_pass and sig_b_pass) else ('PASS_A_ONLY' if sig_a_pass else 'PARTIAL_OR_FAIL'),
        }
    }
    C.save_json('p2.json', out)
    print('\n=== P2 SUMMARY ===')
    print(json.dumps(out['gate_check'], indent=2))
    print('Vertical decoupling (scsc):', json.dumps(decoupling.get('scsc', {}), indent=2))


if __name__ == '__main__':
    main()
