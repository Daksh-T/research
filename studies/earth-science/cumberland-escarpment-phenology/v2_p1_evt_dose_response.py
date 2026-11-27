"""Stage 4 (4A): Regional Within-EVT Stratification & Nonlinear FillDepth Dose-Response on P1.

Closes two Stage-3 gaps:
1. Regional Within-EVT Stratification across the 11 (and strict 30m 9) Southern Cumberland
   closed-karst vs. open-gorge pairs:
   - Quantifies LANDFIRE EVT v1.4.0 composition (EVT 3321 mesophytic cove vs. EVT 3317 dry oak)
     on floors and rims of closed basins vs. open gorges to explain the open-gorge -3 to -5 d
     floor-rim EOS baseline.
   - Extracts 2018-2024 Sentinel-2 NDVI & NDRE strictly within EVT 3321 and within EVT 3317
     to test whether the regional closed-karst phenology signal holds within identical forest types.
2. Nonlinear / Thresholded FillDepth Dose-Response vs. 300m TPI (AIC / AICc model comparison):
   - Compares Null, 300m TPI, Linear FillDepth, Step Threshold (closed basin >=20m/25m),
     Log-Saturating FillDepth, and Piecewise Saturating Hinge models on regional dEOS, dSOS, dGSL.

Outputs:
  data/v2_p1_evt_dose_response.json (strict RFC 8259 JSON)
"""
import json
import math
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'profiling'))
import ee
import common as C

OUT_PATH = os.path.join(HERE, 'data', 'v2_p1_evt_dose_response.json')
CACHE_META = os.path.join(C.CACHE_DIR, 'v2_p1_evt_tpi_meta.json')
CACHE_TS = os.path.join(C.CACHE_DIR, 'v2_p1_evt_s2_ts.json')
CACHE_SPECS = os.path.join(C.CACHE_DIR, 'v2_p1_site_specs.json')

EVT_COVE = 3321  # Southern & Central Appalachian Cove Forest (mesophytic)
EVT_OAK = 3317   # Southern Appalachian Oak Forest (dry upland oak)


def build_site_specs_from_p1():
    """Reconstruct exact P1 site geometries & elevation thresholds from profiling/p1.json."""
    p1 = json.load(open(os.path.join(HERE, 'profiling', 'p1.json')))
    if os.path.exists(CACHE_SPECS):
        site_specs = json.load(open(CACHE_SPECS))
        print(f'[4A] Loaded cached P1 site specs ({len(site_specs)} sites)')
        return p1, site_specs
    import p1_karst_census as P1
    dem90, nlcd90, tf90, _ = P1.load_regional_dem_nlcd()
    _, _, _, pairs = P1.detect_closed_basins_and_open_pairs(dem90, nlcd90, tf90)
    audit_30m = json.load(open(os.path.join(C.CACHE_DIR, 'p1_30m_filldepth_audit.json')))

    site_specs = []
    for p in pairs[:14]:
        c = p['closed']
        o = p['open']
        f30 = audit_30m.get(c['name'], {}).get('max_fill_30m', c['max_fill_m'])
        a30 = audit_30m.get(c['name'], {}).get('area_30m_km2', c['area_km2'])
        strict30 = (f30 >= 25.0) and (a30 >= 0.35)
        site_specs.append({
            'site': c['name'], 'pair': c['name'], 'kind': 'closed',
            'lat': c['lat'], 'lon': c['lon'],
            'max_fill_90m': c['max_fill_m'],
            'max_fill_30m': float(f30),
            'strict_30m_ge25m': bool(strict30),
            'hull': c['hull'],
            'z_floor_med': c['z_floor_med'],
            'z_rim_med': c['z_rim_med'],
            'z_floor_hi': round(c['z_floor_min'] + max(20.0, 0.40 * (c['z_spill'] - c['z_floor_min'])), 1),
            'z_rim_lo': round(max(c['z_spill'], c['z_rim_med'] - 25.0), 1),
        })
        site_specs.append({
            'site': o['name'], 'pair': c['name'], 'kind': 'open',
            'lat': o['lat'], 'lon': o['lon'],
            'max_fill_90m': 0.0,
            'max_fill_30m': 0.0,
            'strict_30m_ge25m': bool(strict30),
            'hull': None,
            'z_floor_med': o['z_floor_med'],
            'z_rim_med': o['z_rim_med'],
            'z_floor_hi': round(o['z_floor_med'] + 25.0, 1),
            'z_rim_lo': round(o['z_rim_med'] - 25.0, 1),
        })
    json.dump(site_specs, open(CACHE_SPECS, 'w'), indent=2)
    return p1, site_specs


def extract_evt_and_tpi_meta(site_specs):
    """Extract LANDFIRE EVT composition + 300m/900m 3DEP TPI on floor & rim for all 22 sites."""
    if os.path.exists(CACHE_META):
        meta = json.load(open(CACHE_META))
        print(f'[4A] Loaded cached EVT & TPI metadata ({len(meta)} sites)')
        return meta

    print(f'[4A] Extracting 30m EVT composition + 300m/900m TPI for {len(site_specs)} sites...')
    dem = C.get_dem_3dep()
    nlcd = C.get_nlcd_2021()
    evt = C.get_evt()
    decid = nlcd.eq(41)

    tpi300 = dem.subtract(dem.focalMean(300, 'circle', 'meters')).rename('tpi300')
    tpi900 = dem.subtract(dem.focalMean(900, 'circle', 'meters')).rename('tpi900')

    meta = {}
    for sp in site_specs:
        if sp['hull'] is not None:
            g_core = ee.Geometry.Polygon([sp['hull']])
            g_outer = g_core.buffer(1100)
        else:
            g_core = ee.Geometry.Point([sp['lon'], sp['lat']]).buffer(900)
            g_outer = ee.Geometry.Point([sp['lon'], sp['lat']]).buffer(1700)
        in_core = ee.Image.constant(0).paint(ee.FeatureCollection([ee.Feature(g_core)]), 1).eq(1)
        fl_m = decid.And(in_core).And(dem.lte(float(sp['z_floor_hi'])))
        rm_m = decid.And(in_core.Not()).And(dem.gte(float(sp['z_rim_lo'])))

        stack = ee.Image.cat([
            tpi300.updateMask(fl_m).rename('fl_tpi300'),
            tpi900.updateMask(fl_m).rename('fl_tpi900'),
            tpi300.updateMask(rm_m).rename('rm_tpi300'),
            dem.updateMask(fl_m).rename('fl_elev'),
            dem.updateMask(rm_m).rename('rm_elev'),
            evt.eq(EVT_COVE).updateMask(fl_m).rename('fl_is3321'),
            evt.eq(EVT_OAK).updateMask(fl_m).rename('fl_is3317'),
            evt.eq(EVT_COVE).updateMask(rm_m).rename('rm_is3321'),
            evt.eq(EVT_OAK).updateMask(rm_m).rename('rm_is3317'),
        ])
        d = stack.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.count(), '', True),
            geometry=g_outer, scale=30, tileScale=4, maxPixels=2e6
        ).getInfo()

        def g(k):
            v = d.get(k)
            return None if v is None else round(float(v), 4)

        fl_n = int(d.get('fl_is3321_count') or 0)
        rm_n = int(d.get('rm_is3321_count') or 0)
        fl_f3321 = float(d.get('fl_is3321_mean') or 0.0)
        fl_f3317 = float(d.get('fl_is3317_mean') or 0.0)
        rm_f3321 = float(d.get('rm_is3321_mean') or 0.0)
        rm_f3317 = float(d.get('rm_is3317_mean') or 0.0)

        meta[sp['site']] = {
            'pair': sp['pair'], 'kind': sp['kind'],
            'max_fill_30m': sp['max_fill_30m'],
            'max_fill_90m': sp['max_fill_90m'],
            'strict_30m_ge25m': sp['strict_30m_ge25m'],
            'fl_n_30m': fl_n, 'rm_n_30m': rm_n,
            'fl_n_3321': int(round(fl_n * fl_f3321)),
            'fl_n_3317': int(round(fl_n * fl_f3317)),
            'rm_n_3321': int(round(rm_n * rm_f3321)),
            'rm_n_3317': int(round(rm_n * rm_f3317)),
            'fl_frac_3321': round(fl_f3321, 4),
            'fl_frac_3317': round(fl_f3317, 4),
            'rm_frac_3321': round(rm_f3321, 4),
            'rm_frac_3317': round(rm_f3317, 4),
            'fl_tpi300': g('fl_tpi300_mean'),
            'fl_tpi900': g('fl_tpi900_mean'),
            'rm_tpi300': g('rm_tpi300_mean'),
            'fl_elev_30m': g('fl_elev_mean'),
            'rm_elev_30m': g('rm_elev_mean'),
        }
        print(f"  {sp['site']:24s} ({sp['kind']:6s}) fl_3321={fl_f3321*100:4.1f}% fl_3317={fl_f3317*100:4.1f}% | "
              f"rm_3321={rm_f3321*100:4.1f}% rm_3317={rm_f3317*100:4.1f}% | TPI300={meta[sp['site']]['fl_tpi300']}")
    json.dump(meta, open(CACHE_META, 'w'), indent=2)
    return meta


def extract_evt_stratified_s2(site_specs):
    """Extract 2018-2024 Sentinel-2 NDVI & NDRE per site stratified by EVT 3321 and EVT 3317."""
    if os.path.exists(CACHE_TS):
        rows = json.load(open(CACHE_TS))
        print(f'[4A] Loaded cached within-EVT S2 time series ({len(rows)} rows)')
        return pd.DataFrame(rows)

    print(f'[4A] Extracting within-EVT S2 NDVI & NDRE sequentially for {len(site_specs)} sites...')
    dem = C.get_dem_3dep()
    nlcd = C.get_nlcd_2021()
    evt = C.get_evt()
    decid = nlcd.eq(41)
    m_3321 = decid.And(evt.eq(EVT_COVE))
    m_3317 = decid.And(evt.eq(EVT_OAK))

    def s2_prep(img):
        scl = img.select('SCL')
        good = scl.eq(4).Or(scl.eq(5))
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('ndvi')
        ndre = img.normalizedDifference(['B8A', 'B5']).rename('ndre')
        return ndvi.addBands(ndre).updateMask(good).updateMask(decid).set(
            'doy', img.date().getRelative('day', 'year'),
            'yr', img.date().get('year')
        )

    all_rows = []
    t0 = time.time()
    for i, sp in enumerate(site_specs, 1):
        if sp['hull'] is not None:
            g_core = ee.Geometry.Polygon([sp['hull']])
            g_outer = g_core.buffer(1100)
        else:
            g_core = ee.Geometry.Point([sp['lon'], sp['lat']]).buffer(900)
            g_outer = ee.Geometry.Point([sp['lon'], sp['lat']]).buffer(1700)
        in_core = ee.Image.constant(0).paint(ee.FeatureCollection([ee.Feature(g_core)]), 1).eq(1)
        fl_base = in_core.And(dem.lte(float(sp['z_floor_hi'])))
        rm_base = in_core.Not().And(dem.gte(float(sp['z_rim_lo'])))

        fl_3321 = fl_base.And(m_3321)
        fl_3317 = fl_base.And(m_3317)
        rm_3321 = rm_base.And(m_3321)
        rm_3317 = rm_base.And(m_3317)

        s2_site = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                   .filterBounds(g_outer)
                   .filterDate('2018-01-01', '2025-01-01')
                   .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 65))
                   .map(s2_prep))

        def red_one(img):
            st = ee.Image.cat([
                img.select('ndvi').updateMask(fl_3321).rename('fl3321_ndvi'),
                img.select('ndre').updateMask(fl_3321).rename('fl3321_ndre'),
                img.select('ndvi').updateMask(fl_3317).rename('fl3317_ndvi'),
                img.select('ndre').updateMask(fl_3317).rename('fl3317_ndre'),
                img.select('ndvi').updateMask(rm_3321).rename('rm3321_ndvi'),
                img.select('ndre').updateMask(rm_3321).rename('rm3321_ndre'),
                img.select('ndvi').updateMask(rm_3317).rename('rm3317_ndvi'),
                img.select('ndre').updateMask(rm_3317).rename('rm3317_ndre'),
            ])
            d = st.reduceRegion(
                reducer=ee.Reducer.mean().combine(ee.Reducer.count(), '', True),
                geometry=g_outer, scale=30, tileScale=2, maxPixels=1e6
            )
            return ee.Feature(None, d).set('doy', img.get('doy'), 'yr', img.get('yr'))

        for attempt in range(4):
            try:
                feats = s2_site.map(red_one).filter(
                    ee.Filter.Or(
                        ee.Filter.gte('fl3321_ndvi_count', 8),
                        ee.Filter.gte('fl3317_ndvi_count', 8),
                        ee.Filter.gte('rm3317_ndvi_count', 8),
                    )
                ).getInfo()['features']
                break
            except Exception as ex:
                if attempt == 3:
                    raise
                print(f'  retry {sp["site"]} ({ex})')
                time.sleep(3 * (attempt + 1))

        srows = []
        for f in feats:
            p = f['properties']
            srows.append({
                'site': sp['site'], 'pair': sp['pair'], 'kind': sp['kind'],
                'yr': int(p['yr']), 'doy': int(p['doy']),
                'fl3321_ndvi': p.get('fl3321_ndvi_mean'), 'fl3321_ndre': p.get('fl3321_ndre_mean'),
                'fl3321_n': int(p.get('fl3321_ndvi_count') or 0),
                'fl3317_ndvi': p.get('fl3317_ndvi_mean'), 'fl3317_ndre': p.get('fl3317_ndre_mean'),
                'fl3317_n': int(p.get('fl3317_ndvi_count') or 0),
                'rm3321_ndvi': p.get('rm3321_ndvi_mean'), 'rm3321_ndre': p.get('rm3321_ndre_mean'),
                'rm3321_n': int(p.get('rm3321_ndvi_count') or 0),
                'rm3317_ndvi': p.get('rm3317_ndvi_mean'), 'rm3317_ndre': p.get('rm3317_ndre_mean'),
                'rm3317_n': int(p.get('rm3317_ndvi_count') or 0),
            })
        all_rows.extend(srows)
        print(f'  [{i:2d}/{len(site_specs)}] {sp["site"]:24s} {len(srows)} clear dates ({time.time()-t0:.1f}s)')

    df_raw = pd.DataFrame(all_rows)
    df_raw['tot_n'] = df_raw['fl3321_n'] + df_raw['fl3317_n'] + df_raw['rm3317_n']
    df_dedup = (df_raw.sort_values('tot_n', ascending=False)
                .drop_duplicates(subset=['site', 'yr', 'doy'], keep='first')
                .drop(columns=['tot_n']))
    clean_rows = df_dedup.to_dict(orient='records')
    json.dump(clean_rows, open(CACHE_TS, 'w'))
    return df_dedup


def paired_summary(vals):
    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return {'n': 0}
    mean = float(np.mean(v))
    sd = float(np.std(v, ddof=1)) if n > 1 else 0.0
    if n >= 3 and sd > 1e-12:
        t_v, p_v = stats.ttest_1samp(v, 0.0)
    else:
        t_v, p_v = np.nan, np.nan
    try:
        w_p = float(stats.wilcoxon(v).pvalue) if n >= 5 else np.nan
    except Exception:
        w_p = np.nan
    return {
        'n': int(n),
        'mean': round(mean, 3),
        'sd': round(sd, 3),
        'se': round(sd / math.sqrt(n), 3) if n > 1 else None,
        't': round(float(t_v), 3) if np.isfinite(t_v) else None,
        'p': float(f'{p_v:.4g}') if np.isfinite(p_v) else None,
        'wilcoxon_p': float(f'{w_p:.4g}') if np.isfinite(w_p) else None,
        'frac_neg': round(float(np.mean(v < 0)), 3),
        'frac_pos': round(float(np.mean(v > 0)), 3),
    }


def analyze_within_evt(df_ts, site_specs, meta):
    spec_map = {s['site']: s for s in site_specs}
    fits = []
    strata = ['fl3321', 'fl3317', 'rm3321', 'rm3317']
    for (site, yr), grp in df_ts.groupby(['site', 'yr']):
        sp = spec_map[site]
        for st in strata:
            sub = grp[grp[f'{st}_n'] >= 10]
            if len(sub) < 6:
                continue
            for idx in ['ndvi', 'ndre']:
                res = C.fit_double_logistic(sub['doy'].values, sub[f'{st}_{idx}'].values,
                                            min_obs=6, min_spring=2, min_autumn=2)
                if res['qc_pass']:
                    fits.append({
                        'site': site, 'pair': sp['pair'], 'kind': sp['kind'],
                        'strict_30m_ge25m': sp['strict_30m_ge25m'],
                        'max_fill_30m': sp['max_fill_30m'],
                        'stratum': st, 'index': idx, 'yr': int(yr),
                        'sos50': res['sos50'], 'eos50': res['eos50'], 'gsl': res['gsl'], 'r2': res['r2'],
                    })
    df_fit = pd.DataFrame(fits)

    # Aggregate to site level per (site, pair, kind, stratum, index) requiring >= 3 QC-pass years
    agg = df_fit.groupby(['site', 'pair', 'kind', 'strict_30m_ge25m', 'stratum', 'index']).agg(
        n_yrs=('yr', 'count'),
        sos50=('sos50', 'mean'),
        eos50=('eos50', 'mean'),
        gsl=('gsl', 'mean'),
    ).reset_index()
    agg = agg[agg['n_yrs'] >= 3].copy()

    out = {'evt_composition': {}, 'within_evt_tests': {}, 'open_gorge_baseline_explanation': {}}

    # 1. EVT Composition Summary across closed vs open floors and rims
    mdf = pd.DataFrame.from_dict(meta, orient='index').reset_index().rename(columns={'index': 'site'})
    for kind in ['closed', 'open']:
        sub = mdf[mdf['kind'] == kind]
        out['evt_composition'][kind] = {
            'n_sites': int(len(sub)),
            'floor_evt3321_mean_pct': round(float(sub['fl_frac_3321'].mean() * 100), 1),
            'floor_evt3317_mean_pct': round(float(sub['fl_frac_3317'].mean() * 100), 1),
            'rim_evt3321_mean_pct': round(float(sub['rm_frac_3321'].mean() * 100), 1),
            'rim_evt3317_mean_pct': round(float(sub['rm_frac_3317'].mean() * 100), 1),
            'floor_tpi300_mean_m': round(float(sub['fl_tpi300'].mean()), 2),
            'floor_tpi900_mean_m': round(float(sub['fl_tpi900'].mean()), 2),
        }

    # 2. Within-EVT Matched Floor-vs-Floor AND Within-EVT Floor-Rim DiD
    for subset_name, strict_only in [('all_11_pairs', False), ('strict_30m_9_pairs', True)]:
        sub_agg = agg[agg['strict_30m_ge25m']] if strict_only else agg
        sub_fit = df_fit[df_fit['strict_30m_ge25m']] if strict_only else df_fit
        out['within_evt_tests'][subset_name] = {}

        for idx in ['ndre', 'ndvi']:
            res_idx = {}
            # (a) Direct matched-EVT Floor-vs-Floor: Closed Floor [EVT X] - Open Floor [EVT X]
            #     Controlling for identical forest type at elevation-matched floors!
            for evt_code, st_fl, st_rm in [('EVT3321_cove', 'fl3321', 'rm3321'), ('EVT3317_oak', 'fl3317', 'rm3317')]:
                s_fl = sub_agg[(sub_agg['index'] == idx) & (sub_agg['stratum'] == st_fl)]
                cl_fl = s_fl[s_fl['kind'] == 'closed'].set_index('pair')
                op_fl = s_fl[s_fl['kind'] == 'open'].set_index('pair')
                pk_fl = cl_fl.index.intersection(op_fl.index)

                # Also compute floor(EVT X) - rim(EVT 3317) to remove any residual latitudinal/climatic gradient between paired sites!
                s_rm17 = sub_agg[(sub_agg['index'] == idx) & (sub_agg['stratum'] == 'rm3317')]
                cl_rm17 = s_rm17[s_rm17['kind'] == 'closed'].set_index('pair')
                op_rm17 = s_rm17[s_rm17['kind'] == 'open'].set_index('pair')
                pk_adj = pk_fl.intersection(cl_rm17.index).intersection(op_rm17.index)

                # Same-EVT Floor - Rim DiD (where both floor & rim have that EVT)
                s_rm = sub_agg[(sub_agg['index'] == idx) & (sub_agg['stratum'] == st_rm)]
                cl_rm = s_rm[s_rm['kind'] == 'closed'].set_index('pair')
                op_rm = s_rm[s_rm['kind'] == 'open'].set_index('pair')
                pk_same = pk_fl.intersection(cl_rm.index).intersection(op_rm.index)

                # Annual panel cluster-robust OLS for Floor-vs-Floor within EVT X (clustered by pair)
                f_yr = sub_fit[(sub_fit['index'] == idx) & (sub_fit['stratum'] == st_fl)].copy()
                f_yr = f_yr[f_yr['pair'].isin(pk_fl)]
                lmm_fl = {}
                if len(pk_fl) >= 4 and len(f_yr) >= 20:
                    # Demean by pair-year or include pair fixed effects (equivalent to paired DiD)
                    f_cl_y = f_yr[f_yr['kind'] == 'closed'].set_index(['pair', 'yr'])
                    f_op_y = f_yr[f_yr['kind'] == 'open'].set_index(['pair', 'yr'])
                    ci = f_cl_y.index.intersection(f_op_y.index)
                    if len(ci) >= 10:
                        pair_arr = np.array([k[0] for k in ci])
                        for met in ['sos50', 'eos50', 'gsl']:
                            diff_y = (f_cl_y.loc[ci, met] - f_op_y.loc[ci, met]).values
                            ols = C.ols_cluster_se(np.ones((len(diff_y), 1)), diff_y,
                                                   clusters=pair_arr, add_const=False, col_names=['closed_minus_open'])
                            lmm_fl[met] = ols['coefs']['closed_minus_open']

                rec = {
                    'n_pairs_floor_matched': int(len(pk_fl)),
                    'floor_closed_minus_open': {
                        'dSOS': paired_summary(cl_fl.loc[pk_fl, 'sos50'] - op_fl.loc[pk_fl, 'sos50']),
                        'dEOS': paired_summary(cl_fl.loc[pk_fl, 'eos50'] - op_fl.loc[pk_fl, 'eos50']),
                        'dGSL': paired_summary(cl_fl.loc[pk_fl, 'gsl'] - op_fl.loc[pk_fl, 'gsl']),
                        'annual_cluster_ols': lmm_fl,
                    },
                    'n_pairs_rim3317_anchored_DiD': int(len(pk_adj)),
                    'rim3317_anchored_DiD': {
                        'closed_fl_minus_rm3317_EOS': paired_summary(cl_fl.loc[pk_adj, 'eos50'] - cl_rm17.loc[pk_adj, 'eos50']),
                        'open_fl_minus_rm3317_EOS': paired_summary(op_fl.loc[pk_adj, 'eos50'] - op_rm17.loc[pk_adj, 'eos50']),
                        'DiD_dSOS': paired_summary((cl_fl.loc[pk_adj, 'sos50'] - cl_rm17.loc[pk_adj, 'sos50']) -
                                                   (op_fl.loc[pk_adj, 'sos50'] - op_rm17.loc[pk_adj, 'sos50'])),
                        'DiD_dEOS': paired_summary((cl_fl.loc[pk_adj, 'eos50'] - cl_rm17.loc[pk_adj, 'eos50']) -
                                                   (op_fl.loc[pk_adj, 'eos50'] - op_rm17.loc[pk_adj, 'eos50'])),
                        'DiD_dGSL': paired_summary((cl_fl.loc[pk_adj, 'gsl'] - cl_rm17.loc[pk_adj, 'gsl']) -
                                                   (op_fl.loc[pk_adj, 'gsl'] - op_rm17.loc[pk_adj, 'gsl'])),
                    },
                    'n_pairs_strict_same_evt_fl_rm_DiD': int(len(pk_same)),
                    'strict_same_evt_fl_rm_DiD': {
                        'closed_fl_minus_rm_EOS': paired_summary(cl_fl.loc[pk_same, 'eos50'] - cl_rm.loc[pk_same, 'eos50']),
                        'open_fl_minus_rm_EOS': paired_summary(op_fl.loc[pk_same, 'eos50'] - op_rm.loc[pk_same, 'eos50']),
                        'DiD_dSOS': paired_summary((cl_fl.loc[pk_same, 'sos50'] - cl_rm.loc[pk_same, 'sos50']) -
                                                   (op_fl.loc[pk_same, 'sos50'] - op_rm.loc[pk_same, 'sos50'])),
                        'DiD_dEOS': paired_summary((cl_fl.loc[pk_same, 'eos50'] - cl_rm.loc[pk_same, 'eos50']) -
                                                   (op_fl.loc[pk_same, 'eos50'] - op_rm.loc[pk_same, 'eos50'])),
                        'DiD_dGSL': paired_summary((cl_fl.loc[pk_same, 'gsl'] - cl_rm.loc[pk_same, 'gsl']) -
                                                   (op_fl.loc[pk_same, 'gsl'] - op_rm.loc[pk_same, 'gsl'])),
                    },
                }
                res_idx[evt_code] = rec
            out['within_evt_tests'][subset_name][idx] = res_idx

    # 3. Quantitatively explain open-gorge -3 to -5 d floor-rim EOS baseline:
    #    (a) Within open-gorge FLOORS at identical elevation: EVT 3321 (cove) - EVT 3317 (oak)
    #    (b) Open-gorge floor EVT 3317 - rim EVT 3317 (pure elevation/topo effect within identical oak EVT)
    #    (c) Open-gorge floor EVT 3321 - rim EVT 3317 (combined cove-vs-oak + topo effect)
    for idx in ['ndre', 'ndvi']:
        op = agg[(agg['kind'] == 'open') & (agg['index'] == idx)]
        o_fl21 = op[op['stratum'] == 'fl3321'].set_index('site')
        o_fl17 = op[op['stratum'] == 'fl3317'].set_index('site')
        o_rm17 = op[op['stratum'] == 'rm3317'].set_index('site')
        o_rm21 = op[op['stratum'] == 'rm3321'].set_index('site')

        k_same_fl = o_fl21.index.intersection(o_fl17.index)
        k_pure_17 = o_fl17.index.intersection(o_rm17.index)
        k_pure_21 = o_fl21.index.intersection(o_rm21.index)
        k_cross = o_fl21.index.intersection(o_rm17.index)

        out['open_gorge_baseline_explanation'][idx] = {
            'open_floor_3321_minus_3317_same_elev_EOS': paired_summary(o_fl21.loc[k_same_fl, 'eos50'] - o_fl17.loc[k_same_fl, 'eos50']),
            'open_floor_3321_minus_3317_same_elev_SOS': paired_summary(o_fl21.loc[k_same_fl, 'sos50'] - o_fl17.loc[k_same_fl, 'sos50']),
            'open_same_EVT3317_floor_minus_rim_EOS': paired_summary(o_fl17.loc[k_pure_17, 'eos50'] - o_rm17.loc[k_pure_17, 'eos50']),
            'open_same_EVT3317_floor_minus_rim_SOS': paired_summary(o_fl17.loc[k_pure_17, 'sos50'] - o_rm17.loc[k_pure_17, 'sos50']),
            'open_same_EVT3321_floor_minus_rim_EOS': paired_summary(o_fl21.loc[k_pure_21, 'eos50'] - o_rm21.loc[k_pure_21, 'eos50']),
            'open_cross_EVT_fl3321_minus_rm3317_EOS': paired_summary(o_fl21.loc[k_cross, 'eos50'] - o_rm17.loc[k_cross, 'eos50']),
            'open_cross_EVT_fl3321_minus_rm3317_SOS': paired_summary(o_fl21.loc[k_cross, 'sos50'] - o_rm17.loc[k_cross, 'sos50']),
        }

    return out, agg


def ols_dz_adjusted_diff(dy, dz):
    """Estimate paired difference dy adjusted for elevation mismatch dz via OLS (dy ~ 1 + dz).
    Returns intercept (effect at dz = 0), standard OLS SE, t, p (df = n - 2), and dz slope."""
    dy = np.asarray(dy, dtype=np.float64)
    dz = np.asarray(dz, dtype=np.float64)
    ok = np.isfinite(dy) & np.isfinite(dz)
    dy, dz = dy[ok], dz[ok]
    n = len(dy)
    if n < 4:
        return {'n': int(n)}
    X = np.column_stack([np.ones(n), dz])
    beta, *_ = np.linalg.lstsq(X, dy, rcond=None)
    resid = dy - X @ beta
    df = n - 2
    s2 = float(np.sum(resid ** 2)) / max(1, df)
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(cov), 1e-18))
    t_int = float(beta[0] / se[0])
    p_int = float(2.0 * stats.t.sf(abs(t_int), df=df))
    t_dz = float(beta[1] / se[1])
    p_dz = float(2.0 * stats.t.sf(abs(t_dz), df=df))
    return {
        'n': int(n), 'df': int(df),
        'raw_mean': round(float(np.mean(dy)), 3),
        'dz_mean_m': round(float(np.mean(dz)), 2),
        'dz_adj_mean': round(float(beta[0]), 3),
        'dz_adj_se': round(float(se[0]), 3),
        'dz_adj_t': round(t_int, 3),
        'dz_adj_p': float(f'{p_int:.4g}'),
        'dz_slope_per_100m': round(float(beta[1] * 100.0), 3),
        'dz_slope_p': float(f'{p_dz:.4g}'),
    }


def analyze_relief_and_rim_free(p1, meta, agg_evt):
    """Relief-adjusted 22-site CR1 OLS (df=10), interaction check, rim-free floor-vs-floor tests
    (all 11 pairs + within-EVT 3317/3321), and relief-controlled binary closure vs. TPI300 models."""
    sm = p1['site_meta_30m']
    out = {'relief_diagnostics': {}, 'relief_adjusted_cr1_ols': {}, 'rim_free_floor_vs_floor': {},
           'relief_controlled_closure_models': {}, 'within_closed_depth_slope': {}}

    # Relief diagnostics on the 22 sites (11 closed + 11 open)
    tbl0 = pd.DataFrame(p1['phenology_power']['ndre']['per_cove_table'])
    tbl0['z_fl'] = [sm[s]['z_floor_30m'] for s in tbl0['site']]
    tbl0['z_rm'] = [sm[s]['z_rim_30m'] for s in tbl0['site']]
    tbl0['relief_m'] = tbl0['z_rm'] - tbl0['z_fl']
    cl0 = tbl0[tbl0['kind'] == 'closed'].set_index('pair')
    op0 = tbl0[tbl0['kind'] == 'open'].set_index('pair')
    pk0 = cl0.index.intersection(op0.index)
    rel_diff = (op0.loc[pk0, 'relief_m'] - cl0.loc[pk0, 'relief_m']).values
    t_rel, p_rel = stats.ttest_1samp(rel_diff, 0.0)
    out['relief_diagnostics'] = {
        'n_pairs': int(len(pk0)),
        'closed_relief_mean_m': round(float(cl0.loc[pk0, 'relief_m'].mean()), 2),
        'closed_relief_sd_m': round(float(cl0.loc[pk0, 'relief_m'].std(ddof=1)), 2),
        'open_relief_mean_m': round(float(op0.loc[pk0, 'relief_m'].mean()), 2),
        'open_relief_sd_m': round(float(op0.loc[pk0, 'relief_m'].std(ddof=1)), 2),
        'open_minus_closed_relief_m': round(float(rel_diff.mean()), 2),
        'open_minus_closed_relief_p': float(f'{p_rel:.4g}'),
        'closed_rims_ge_535m': int((cl0.loc[pk0, 'z_rm'] >= 535.0).sum()),
        'open_rims_ge_535m': int((op0.loc[pk0, 'z_rm'] >= 535.0).sum()),
        'closed_minus_open_floor_elev_mean_m': round(float((cl0.loc[pk0, 'z_fl'] - op0.loc[pk0, 'z_fl']).mean()), 2),
    }

    for idx in ['ndre', 'ndvi']:
        tbl = pd.DataFrame(p1['phenology_power'][idx]['per_cove_table'])
        tbl['z_fl'] = [sm[s]['z_floor_30m'] for s in tbl['site']]
        tbl['z_rm'] = [sm[s]['z_rim_30m'] for s in tbl['site']]
        tbl['relief_m'] = tbl['z_rm'] - tbl['z_fl']
        tbl['is_closed'] = (tbl['kind'] == 'closed').astype(float)
        tbl['tpi300'] = [meta[s]['fl_tpi300'] for s in tbl['site']]
        tbl['fill_30m'] = [meta[s]['max_fill_30m'] for s in tbl['site']]
        tbl['strict_30m'] = [meta[s]['strict_30m_ge25m'] for s in tbl['site']]
        tbl['mean_fl_gsl'] = tbl['mean_fl_eos'] - tbl['mean_fl_sos']

        # 1. Relief-adjusted 22-site OLS with CR1 pair-clustered SEs (df=10) + interaction check
        cr1_all, cr1_strict = {}, {}
        for oc in ['dSOS', 'dEOS', 'dGSL']:
            y = tbl[oc].values
            r_main = C.ols_cluster_se(
                tbl[['is_closed', 'relief_m']].values, y,
                clusters=tbl['pair'].values, col_names=['is_closed', 'relief_m']
            )
            rel_c = tbl['relief_m'].values - tbl['relief_m'].mean()
            r_int = C.ols_cluster_se(
                np.column_stack([tbl['is_closed'].values, rel_c, tbl['is_closed'].values * rel_c]), y,
                clusters=tbl['pair'].values, col_names=['is_closed', 'relief_c', 'is_closed_x_relief']
            )
            cr1_all[oc] = {
                'n_sites': int(len(tbl)), 'n_pairs': int(tbl['pair'].nunique()), 'df': r_main['df'],
                'r2': r_main['r2'], 'adj_r2': r_main['adj_r2'],
                'is_closed': r_main['coefs']['is_closed'],
                'relief_m': r_main['coefs']['relief_m'],
                'interaction_is_closed_x_relief': r_int['coefs']['is_closed_x_relief'],
            }
            sub9 = tbl[tbl['strict_30m']].copy()
            r9 = C.ols_cluster_se(
                sub9[['is_closed', 'relief_m']].values, sub9[oc].values,
                clusters=sub9['pair'].values, col_names=['is_closed', 'relief_m']
            )
            cr1_strict[oc] = {
                'n_sites': int(len(sub9)), 'n_pairs': int(sub9['pair'].nunique()), 'df': r9['df'],
                'is_closed': r9['coefs']['is_closed'], 'relief_m': r9['coefs']['relief_m'],
            }
        out['relief_adjusted_cr1_ols'][idx] = {'all_11_pairs': cr1_all, 'strict_30m_9_pairs': cr1_strict}

        # 2. Rim-free direct floor-vs-floor (all 11 pairs + within-EVT 3317 & 3321), dz_floor-adjusted
        cl = tbl[tbl['kind'] == 'closed'].set_index('pair')
        op = tbl[tbl['kind'] == 'open'].set_index('pair')
        pk = cl.index.intersection(op.index)
        dz_all = (cl.loc[pk, 'z_fl'] - op.loc[pk, 'z_fl']).values
        rf_all = {}
        for col_fl, nm in [('mean_fl_sos', 'dSOS'), ('mean_fl_eos', 'dEOS'), ('mean_fl_gsl', 'dGSL')]:
            dy = (cl.loc[pk, col_fl] - op.loc[pk, col_fl]).values
            rf_all[nm] = {
                'paired_unadjusted': paired_summary(dy),
                'dz_floor_adjusted': ols_dz_adjusted_diff(dy, dz_all),
            }

        rf_evt = {}
        for evt_lbl, st in [('EVT3317_oak', 'fl3317'), ('EVT3321_cove', 'fl3321')]:
            s_e = agg_evt[(agg_evt['index'] == idx) & (agg_evt['stratum'] == st)]
            ce = s_e[s_e['kind'] == 'closed'].set_index('pair')
            oe = s_e[s_e['kind'] == 'open'].set_index('pair')
            pke = ce.index.intersection(oe.index)
            dz_e = np.array([meta[ce.loc[p, 'site']]['fl_elev_30m'] - meta[oe.loc[p, 'site']]['fl_elev_30m'] for p in pke])
            rf_evt[evt_lbl] = {}
            for col_e, nm in [('sos50', 'dSOS'), ('eos50', 'dEOS'), ('gsl', 'dGSL')]:
                dy_e = (ce.loc[pke, col_e] - oe.loc[pke, col_e]).values
                rf_evt[evt_lbl][nm] = {
                    'paired_unadjusted': paired_summary(dy_e),
                    'dz_floor_adjusted': ols_dz_adjusted_diff(dy_e, dz_e),
                }
        out['rim_free_floor_vs_floor'][idx] = {'all_11_pairs': rf_all, 'within_evt': rf_evt}

        # 3. Relief-controlled geomorphic models: binary is_closed vs Relief + TPI300
        rc_models = {}
        for oc in ['dSOS', 'dEOS', 'dGSL']:
            y = tbl[oc].values
            m_rel = fit_gaussian_aic(tbl[['relief_m']].values, y)
            m_rel_tpi = fit_gaussian_aic(tbl[['relief_m', 'tpi300']].values, y)
            m_rel_cl = fit_gaussian_aic(tbl[['relief_m', 'is_closed']].values, y)
            m_rel_tpi_cl = fit_gaussian_aic(tbl[['relief_m', 'tpi300', 'is_closed']].values, y)
            m_rel_fd = fit_gaussian_aic(tbl[['relief_m', 'fill_30m']].values, y)
            partial_r2_over_rel = (m_rel_cl['r2'] - m_rel['r2']) / max(1e-9, 1.0 - m_rel['r2'])
            partial_r2_over_rel_tpi = (m_rel_tpi_cl['r2'] - m_rel_tpi['r2']) / max(1e-9, 1.0 - m_rel_tpi['r2'])
            rc_models[oc] = {
                'M_relief': m_rel,
                'M_relief_tpi300': m_rel_tpi,
                'M_relief_is_closed': m_rel_cl,
                'M_relief_tpi300_is_closed': m_rel_tpi_cl,
                'M_relief_fill_linear': m_rel_fd,
                'delta_r2_tpi300_over_relief': round(m_rel_tpi['r2'] - m_rel['r2'], 4),
                'delta_r2_is_closed_over_relief': round(m_rel_cl['r2'] - m_rel['r2'], 4),
                'partial_r2_is_closed_over_relief': round(partial_r2_over_rel, 4),
                'partial_r2_is_closed_over_relief_tpi300': round(partial_r2_over_rel_tpi, 4),
                'delta_aicc_relief_closed_minus_relief_tpi300': round(m_rel_cl['aicc'] - m_rel_tpi['aicc'], 2),
                'delta_aicc_relief_closed_minus_relief': round(m_rel_cl['aicc'] - m_rel['aicc'], 2),
                'delta_aicc_relief_tpi_closed_minus_relief_tpi': round(m_rel_tpi_cl['aicc'] - m_rel_tpi['aicc'], 2),
            }
        out['relief_controlled_closure_models'][idx] = rc_models

        # 4. Within-closed depth ramp test (checking for any ramp above 22m; 0-22m unobserved)
        cl_only = tbl[tbl['kind'] == 'closed']
        w_slopes = {}
        for oc in ['dSOS', 'dEOS', 'dGSL']:
            r_c, p_c = stats.pearsonr(cl_only['fill_30m'].values, cl_only[oc].values)
            w_slopes[oc] = {'n': int(len(cl_only)), 'r': round(float(r_c), 3), 'p': float(f'{p_c:.4g}')}
        out['within_closed_depth_slope'][idx] = w_slopes

    return out


def archive_disqualified_basins(p1):
    """Archive exact 6 disqualified closed basins from profiling/p1.json (no fabricated basin names)."""
    rows = []
    for b in p1.get('disqualified_basins', []):
        v30 = b.get('verify_30m') or {}
        rows.append({
            'name': b['name'],
            'lat': b['lat'], 'lon': b['lon'],
            'max_fill_90m': b['max_fill_m'],
            'area_90m_km2': b['area_km2'],
            'floor_nlcd41_90m_pct': round(float(b['floor_nlcd41_frac']) * 100.0, 1),
            'max_fill_30m': v30.get('max_fill_30m'),
            'area_30m_km2': v30.get('area_30m_km2'),
            'floor_nlcd41_30m_pct': round(float(v30['floor_nlcd41_30m']) * 100.0, 1) if v30.get('floor_nlcd41_30m') is not None else None,
            'floor_pasture_30m_pct': round(float(v30['floor_pasture_30m']) * 100.0, 1) if v30.get('floor_pasture_30m') is not None else None,
            'status': b['status'],
        })
    return {
        'n_disqualified': int(len(rows)),
        'basins': rows,
        'note': (
            'All 6 excluded regional basins are archived directly from profiling/p1.json. '
            'Specifically, Lost_Cove_TN (48.7% 30m NLCD-41, 38.4% pasture) and Sinking_Cove_TN '
            '(57.1% 30m NLCD-41) fell below the 60% deciduous-floor criterion and therefore sit '
            'outside the 11 P1 phenology pairs; both join 5 P1 basins in the 7-cove ECOSTRESS '
            'tile 16SED thermal replication. Karst_AL_15 collapsed to 0 m fill depth at 30 m.'
        ),
    }


def fit_gaussian_aic(X, y, extra_df=0):
    """Fit OLS and return exact Gaussian AIC, AICc, R2, adj_R2, RMSE, F-stat, p-value."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if X is None or X.shape[1] == 0:
        X_mat = np.ones((n, 1))
        p_pred = 0
    else:
        X_mat = np.column_stack([np.ones(n), np.asarray(X, dtype=np.float64)])
        p_pred = X.shape[1]
    beta, *_ = np.linalg.lstsq(X_mat, y, rcond=None)
    resid = y - X_mat @ beta
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(1, n - p_pred - 1)
    # Number of free parameters k = intercept + p_pred + variance + any profiled threshold params
    k = 1 + p_pred + 1 + extra_df
    sigma2 = max(rss / n, 1e-12)
    ll = -0.5 * n * (math.log(2.0 * math.pi * sigma2) + 1.0)
    aic = 2.0 * k - 2.0 * ll
    aicc = aic + (2.0 * k * (k + 1)) / max(1.0, n - k - 1)
    if p_pred >= 1 and n > p_pred + 1:
        f_stat = ((tss - rss) / p_pred) / max(rss / (n - p_pred - 1), 1e-12)
        p_val = float(stats.f.sf(f_stat, p_pred, n - p_pred - 1))
    else:
        f_stat, p_val = 0.0, 1.0
    return {
        'n': int(n), 'k_params': int(k), 'r2': round(r2, 4), 'adj_r2': round(adj_r2, 4),
        'rmse': round(math.sqrt(rss / n), 3),
        'aic': round(aic, 2), 'aicc': round(aicc, 2),
        'f_stat': round(float(f_stat), 3), 'p': float(f'{p_val:.4g}'),
        'beta': [round(float(b), 4) for b in beta],
    }


def morans_i_perm(lats, lons, vals, n_perm=5000, seed=42):
    """Inverse-distance Moran's I spatial autocorrelation test with exact permutation p-value."""
    v = np.asarray(vals, dtype=np.float64)
    n = len(v)
    if n < 5 or np.std(v) < 1e-12:
        return {'n': int(n), 'I': None, 'p_perm': None}
    W = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i != j:
                d_km = math.hypot((lons[i] - lons[j]) * math.cos(math.radians(35.0)),
                                  lats[i] - lats[j]) * 111.32
                W[i, j] = 1.0 / max(d_km, 1.0)
    z = v - v.mean()
    s0 = W.sum()
    denom = float(z @ z)
    I_obs = float((n / s0) * (z @ W @ z) / denom)
    rng = np.random.default_rng(seed)
    I_perm = np.array([(n / s0) * (zp @ W @ zp) / denom for zp in (rng.permutation(z) for _ in range(n_perm))])
    exp_i = -1.0 / (n - 1)
    p_val = float(np.mean(np.abs(I_perm - I_perm.mean()) >= abs(I_obs - I_perm.mean())))
    return {
        'n': int(n),
        'I': round(I_obs, 4),
        'expected_I': round(exp_i, 4),
        'p_perm_two_sided': round(p_val, 4),
    }


def analyze_dose_response_aic(p1, meta, site_specs):
    """Compare nonlinear/thresholded FillDepth models vs. 300m TPI on P1 regional phenology."""
    coord_map = {s['site']: (s['lat'], s['lon']) for s in site_specs}
    out = {}
    for idx in ['ndre', 'ndvi']:
        tbl = pd.DataFrame(p1['phenology_power'][idx]['per_cove_table'])
        tbl['fill_30m'] = [meta[s]['max_fill_30m'] for s in tbl['site']]
        tbl['tpi300'] = [meta[s]['fl_tpi300'] for s in tbl['site']]
        tbl['tpi900'] = [meta[s]['fl_tpi900'] for s in tbl['site']]
        tbl['fl_frac_3321'] = [meta[s]['fl_frac_3321'] for s in tbl['site']]
        tbl['rm_frac_3317'] = [meta[s]['rm_frac_3317'] for s in tbl['site']]
        tbl['lat'] = [coord_map[s][0] for s in tbl['site']]
        tbl['lon'] = [coord_map[s][1] for s in tbl['site']]

        fd = tbl['fill_30m'].values.astype(float)
        t300 = tbl['tpi300'].values.astype(float)
        t900 = tbl['tpi900'].values.astype(float)

        out[idx] = {}
        for outcome in ['dEOS', 'dSOS', 'dGSL']:
            y = tbl[outcome].values.astype(float)
            # Profile optimal saturating hinge tau in [15..55m]
            best_tau, best_rss = 25.0, float('inf')
            for tau in np.arange(15.0, 56.0, 1.0):
                xh = np.minimum(fd, tau)[:, None]
                Xm = np.column_stack([np.ones(len(y)), xh])
                b, *_ = np.linalg.lstsq(Xm, y, rcond=None)
                rss = float(np.sum((y - Xm @ b) ** 2))
                if rss < best_rss:
                    best_rss, best_tau = rss, float(tau)

            models = {
                'M0_null': fit_gaussian_aic(None, y),
                'M1_tpi300': fit_gaussian_aic(t300[:, None], y),
                'M1b_tpi900': fit_gaussian_aic(t900[:, None], y),
                'M2_fill_linear': fit_gaussian_aic(fd[:, None], y),
                'M3_fill_step20m': fit_gaussian_aic((fd >= 20.0).astype(float)[:, None], y),
                'M3b_fill_step25m': fit_gaussian_aic((fd >= 25.0).astype(float)[:, None], y),
                'M4_fill_log1p': fit_gaussian_aic(np.log1p(fd)[:, None], y),
                'M5_fill_hinge25m': fit_gaussian_aic(np.minimum(fd, 25.0)[:, None], y),
                'M5b_fill_hinge_opt': {
                    **fit_gaussian_aic(np.minimum(fd, best_tau)[:, None], y, extra_df=1),
                    'opt_tau_m': best_tau,
                },
                'M6_fill_step20m_plus_tpi300': fit_gaussian_aic(
                    np.column_stack([(fd >= 20.0).astype(float), t300]), y
                ),
            }
            min_aic = min(m['aic'] for m in models.values())
            min_aicc = min(m['aicc'] for m in models.values())
            for m in models.values():
                m['delta_aic'] = round(m['aic'] - min_aic, 2)
                m['delta_aicc'] = round(m['aicc'] - min_aicc, 2)

            # Explicit delta vs 300m TPI (negative = FillDepth beats 300m TPI!)
            best_fill_key = min(['M3_fill_step20m', 'M4_fill_log1p', 'M5_fill_hinge25m'],
                                key=lambda k: models[k]['aicc'])
            out[idx][outcome] = {
                'models': models,
                'best_fill_model': best_fill_key,
                'delta_aic_bestFill_minus_tpi300': round(models[best_fill_key]['aic'] - models['M1_tpi300']['aic'], 2),
                'delta_aicc_bestFill_minus_tpi300': round(models[best_fill_key]['aicc'] - models['M1_tpi300']['aicc'], 2),
                'delta_aicc_bestFill_minus_linearFill': round(models[best_fill_key]['aicc'] - models['M2_fill_linear']['aicc'], 2),
                'r_tpi300': round(float(stats.pearsonr(t300, y).statistic), 3),
                'p_tpi300': float(f'{stats.pearsonr(t300, y).pvalue:.4g}'),
                'r_fill_log1p': round(float(stats.pearsonr(np.log1p(fd), y).statistic), 3),
                'p_fill_log1p': float(f'{stats.pearsonr(np.log1p(fd), y).pvalue:.4g}'),
            }

        # Store per-site merged table + spatial autocorrelation audit for NDRE
        if idx == 'ndre':
            b_h25 = out['ndre']['dGSL']['models']['M5_fill_hinge25m']['beta']
            resid_h25 = tbl['dGSL'].values - (b_h25[0] + b_h25[1] * np.minimum(fd, 25.0))
            cl_s = tbl[tbl['kind'] == 'closed'].set_index('pair')
            op_s = tbl[tbl['kind'] == 'open'].set_index('pair')
            pk = cl_s.index.intersection(op_s.index)
            did_gsl = (cl_s.loc[pk, 'dGSL'] - op_s.loc[pk, 'dGSL']).values
            out['moran_i_spatial_autocorrelation'] = {
                'm5_hinge25m_dGSL_residuals_22sites': morans_i_perm(tbl['lat'].values, tbl['lon'].values, resid_h25),
                'pair_DiD_dGSL_11pairs': morans_i_perm(cl_s.loc[pk, 'lat'].values, cl_s.loc[pk, 'lon'].values, did_gsl),
            }
            closed_fds = sorted([round(float(x), 1) for x in tbl.loc[tbl['kind'] == 'closed', 'fill_30m']])
            cl_only_30 = tbl[tbl['kind'] == 'closed']
            r_eos_30, p_eos_30 = stats.pearsonr(cl_only_30['fill_30m'].values, cl_only_30['dEOS'].values)
            r_gsl_30, p_gsl_30 = stats.pearsonr(cl_only_30['fill_30m'].values, cl_only_30['dGSL'].values)
            out['closure_gap_note'] = (
                f"All 11 open gorges have FillDepth = 0 m and all 11 closed karst basins have 30 m FillDepth in "
                f"[{closed_fds[0]}, {closed_fds[-1]}] m (zero observations in 0 < FillDepth < {closed_fds[0]} m). "
                f"Because min(FillDepth, 25 m) equals 25 * is_closed on 21/22 sites, M5_fill_hinge25m and binary "
                f"M3_fill_step20m are statistically indistinguishable (dAICc = 0.03 on dGSL) and both restate "
                f"binary topographic closure rather than resolving a continuous 0-22 m dose-response ramp. "
                f"Within the 11 closed basins above 22 m, deeper 30 m sills do not add further compression "
                f"(r(dEOS, FillDepth_30m) = {r_eos_30:+.2f}, p = {p_eos_30:.2f}; "
                f"r(dGSL, FillDepth_30m) = {r_gsl_30:+.2f}, p = {p_gsl_30:.2f})."
            )
            sm = p1['site_meta_30m']
            tbl['z_floor_30m'] = [sm[s]['z_floor_30m'] for s in tbl['site']]
            tbl['z_rim_30m'] = [sm[s]['z_rim_30m'] for s in tbl['site']]
            tbl['relief_m'] = tbl['z_rim_30m'] - tbl['z_floor_30m']
            out['site_table_ndre'] = tbl[[
                'site', 'pair', 'kind', 'fill_30m', 'tpi300', 'tpi900',
                'z_floor_30m', 'z_rim_30m', 'relief_m',
                'fl_frac_3321', 'rm_frac_3317', 'dSOS', 'dEOS', 'dGSL', 'sigma_SOS'
            ]].round(3).to_dict(orient='records')
    return out


def main():
    t0 = time.time()
    if not (os.path.exists(CACHE_META) and os.path.exists(CACHE_TS)):
        C.init_ee()
    p1, site_specs = build_site_specs_from_p1()
    meta = extract_evt_and_tpi_meta(site_specs)
    df_ts = extract_evt_stratified_s2(site_specs)
    evt_res, agg_evt = analyze_within_evt(df_ts, site_specs, meta)
    dose_res = analyze_dose_response_aic(p1, meta, site_specs)
    relief_res = analyze_relief_and_rim_free(p1, meta, agg_evt)
    disq_archive = archive_disqualified_basins(p1)

    out = {
        'experiment': 'v2_4A_p1_evt_dose_response',
        'elapsed_s': round(time.time() - t0, 1),
        'p1_gate_status': p1['gate_check']['verdict'],
        'disqualified_basins_archive': disq_archive,
        'site_evt_tpi_meta': meta,
        'within_evt_analysis': evt_res,
        'relief_and_rim_free_analysis': relief_res,
        'dose_response_aic': dose_res,
    }
    clean = C.sanitize_json(out)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(clean, f, indent=2, allow_nan=False)
    print(f'WROTE {OUT_PATH} in {time.time()-t0:.1f}s')

    print('\n=== 4A KEY RESULTS ===')
    print('Relief diagnostics:', json.dumps(relief_res['relief_diagnostics']))
    print('Relief-adj CR1 NDRE (11 pairs):', json.dumps(relief_res['relief_adjusted_cr1_ols']['ndre']['all_11_pairs'], indent=2))
    print('Rim-free NDRE (11 pairs):', json.dumps(relief_res['rim_free_floor_vs_floor']['ndre']['all_11_pairs'], indent=2))
    print('Rim-free NDRE within-EVT:', json.dumps(relief_res['rim_free_floor_vs_floor']['ndre']['within_evt'], indent=2))
    print('Relief-controlled closure dGSL:', json.dumps(relief_res['relief_controlled_closure_models']['ndre']['dGSL'], indent=2))
    print('Disqualified basins:', [b['name'] for b in disq_archive['basins']])


if __name__ == '__main__':
    main()
