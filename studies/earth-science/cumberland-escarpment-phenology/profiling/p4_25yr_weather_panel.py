"""Experiment P4: 25-Year (2000-2024) Within-Cove Interannual Weather Identification Panel.

Extracts annual floor-minus-rim dSOS_ct and dEOS_ct from harmonized Landsat 5/7/8/9 (2000-2024)
across 4 replicated closed karst basins + 3 open-drainage control gorges around Sewanee,
regressed on orthogonalized NOAA USC00408184 + ERA5-Land synoptic weather:
  - CalmOctFreeze (DOY 265-305 cold + calm radiative nights -> CAP frost driver)
  - WindyOctCold (DOY 265-305 cold + windy advective nights -> physical placebo)
  - AprLateFrost (DOY 80-125 calm late-frost severity -> spring CAP retardation)
  - MarAprPrecip (DOY 60-120 total precip -> karst ponding / soil wetness control)
with cove fixed effects and year-clustered Cameron-Miller standard errors.
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

# Replicated Southern Cumberland Escarpment coves (4 Closed Karst + 3 Open Gorges near Sewanee weather station)
SITES = [
    # Closed karst basins (floor = interior depression below z_fl; rim = outer plateau >= z_rim)
    {'site': 'Lost_Cove_TN', 'kind': 'closed', 'lat': 35.140, 'lon': -85.910,
     'r_core': 1400, 'r_outer': 2600, 'z_fl': 385.0, 'z_rim': 545.0, 'box': [-85.945, 35.121, -85.880, 35.168]},
    {'site': 'Sinking_Cove_TN', 'kind': 'closed', 'lat': 35.014, 'lon': -85.992,
     'r_core': 1200, 'r_outer': 2300, 'z_fl': 320.0, 'z_rim': 495.0, 'box': None},
    {'site': 'Karst_TN_02_Franklin', 'kind': 'closed', 'lat': 35.050, 'lon': -86.016,
     'r_core': 950, 'r_outer': 2000, 'z_fl': 365.0, 'z_rim': 460.0, 'box': None},
    {'site': 'Karst_TN_08_Grundy', 'kind': 'closed', 'lat': 35.588, 'lon': -85.677,
     'r_core': 1100, 'r_outer': 2200, 'z_fl': 380.0, 'z_rim': 440.0, 'box': None},
    # Open-drainage control gorges (verified 3DEP floor & plateau rim elevations)
    {'site': 'OpenN_Sewanee', 'kind': 'open', 'lat': 35.205, 'lon': -85.925,
     'r_core': 1800, 'r_outer': 2800, 'z_fl': 385.0, 'z_rim': 545.0, 'box': [-85.991, 35.178, -85.859, 35.231]},
    {'site': 'Fiery_Gizzard_OPEN', 'kind': 'open', 'lat': 35.195, 'lon': -85.745,
     'r_core': 1500, 'r_outer': 2600, 'z_fl': 430.0, 'z_rim': 550.0, 'box': None},
    {'site': 'Collins_Gulf_OPEN', 'kind': 'open', 'lat': 35.435, 'lon': -85.640,
     'r_core': 1500, 'r_outer': 2600, 'z_fl': 410.0, 'z_rim': 540.0, 'box': None},
]


def extract_landsat_25yr():
    cache_p = os.path.join(C.CACHE_DIR, 'p4_landsat_2000_2024_ts.json')
    if os.path.exists(cache_p):
        print('[P4] Loaded cached 25-yr Landsat 5-9 time series')
        return pd.DataFrame(json.load(open(cache_p)))

    print('[P4] Extracting 25-yr (2000-2024) harmonized Landsat 5/7/8/9 NDVI across 7 coves...')
    dem = C.get_dem_3dep()
    nlcd = C.get_nlcd_2021()
    decid = nlcd.eq(41)

    def mask_l57(img):
        qa = img.select('QA_PIXEL')
        # Bits 1 (dilated cloud), 2 (cirrus), 3 (cloud), 4 (cloud shadow), 5 (snow) == 0
        clear = qa.bitwiseAnd((1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)).eq(0)
        red = img.select('SR_B3').multiply(2.75e-5).add(-0.2)
        nir = img.select('SR_B4').multiply(2.75e-5).add(-0.2)
        ndvi_etm = nir.subtract(red).divide(nir.add(red).max(1e-4))
        # Roy et al. (2016) ETM+ -> OLI NDVI harmonization
        ndvi = ndvi_etm.multiply(0.9723).add(0.0235).rename('ndvi')
        return ndvi.updateMask(clear).updateMask(decid).updateMask(ndvi.gt(0.05).And(ndvi.lt(1.0))).set(
            'doy', img.date().getRelative('day', 'year'),
            'yr', img.date().get('year'),
            'sat', 'L57'
        )

    def mask_l89(img):
        qa = img.select('QA_PIXEL')
        clear = qa.bitwiseAnd((1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)).eq(0)
        red = img.select('SR_B4').multiply(2.75e-5).add(-0.2)
        nir = img.select('SR_B5').multiply(2.75e-5).add(-0.2)
        ndvi = nir.subtract(red).divide(nir.add(red).max(1e-4)).rename('ndvi')
        return ndvi.updateMask(clear).updateMask(decid).updateMask(ndvi.gt(0.05).And(ndvi.lt(1.0))).set(
            'doy', img.date().getRelative('day', 'year'),
            'yr', img.date().get('year'),
            'sat', 'L89'
        )

    def extract_site_chunk(sp, y0, y1):
        if sp['box'] is not None:
            g_core = ee.Geometry.Rectangle(sp['box'])
            g_outer = g_core.buffer(1500)
        else:
            pt = ee.Geometry.Point([sp['lon'], sp['lat']])
            g_core = pt.buffer(sp['r_core'])
            g_outer = pt.buffer(sp['r_outer'])

        in_core = ee.Image.constant(0).paint(ee.FeatureCollection([ee.Feature(g_core)]), 1).eq(1)
        if sp['site'] in ('Lost_Cove_TN', 'OpenN_Sewanee'):
            fl_m = decid.And(in_core).And(dem.gte(330)).And(dem.lte(sp['z_fl']))
            rm_m = decid.And(dem.gte(sp['z_rim']))
        else:
            fl_m = decid.And(in_core).And(dem.lte(sp['z_fl']))
            rm_m = decid.And(in_core.Not()).And(dem.gte(sp['z_rim']))

        d0, d1 = f'{y0}-01-01', f'{y1+1}-01-01'
        cols = []
        if y0 <= 2011:
            cols.append(ee.ImageCollection('LANDSAT/LT05/C02/T1_L2').filterBounds(g_outer).filterDate(d0, d1)
                        .filter(ee.Filter.lt('CLOUD_COVER', 65)).map(mask_l57))
        if y0 <= 2021:
            cols.append(ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').filterBounds(g_outer).filterDate(d0, d1)
                        .filter(ee.Filter.lt('CLOUD_COVER', 65)).map(mask_l57))
        if y1 >= 2013:
            cols.append(ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(g_outer).filterDate(d0, d1)
                        .filter(ee.Filter.lt('CLOUD_COVER', 65)).map(mask_l89))
        if y1 >= 2021:
            cols.append(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').filterBounds(g_outer).filterDate(d0, d1)
                        .filter(ee.Filter.lt('CLOUD_COVER', 65)).map(mask_l89))

        merged = cols[0]
        for c in cols[1:]:
            merged = merged.merge(c)

        def red_one(img):
            st = ee.Image.cat([
                img.updateMask(fl_m).rename('fl_ndvi'),
                img.updateMask(rm_m).rename('rm_ndvi'),
            ])
            d = st.reduceRegion(ee.Reducer.mean().combine(ee.Reducer.count(), '', True),
                                g_outer, 30, tileScale=2, maxPixels=1e6)
            return ee.Feature(None, d).set('doy', img.get('doy'), 'yr', img.get('yr'), 'sat', img.get('sat'))

        feats = merged.map(red_one).filter(
            ee.Filter.And(ee.Filter.gte('fl_ndvi_count', 15), ee.Filter.gte('rm_ndvi_count', 15))
        ).getInfo()['features']
        rows = []
        for f in feats:
            p = f['properties']
            rows.append({
                'site': sp['site'], 'kind': sp['kind'], 'yr': int(p['yr']), 'doy': int(p['doy']),
                'sat': p['sat'],
                'fl_ndvi': p.get('fl_ndvi_mean'), 'fl_n': int(p.get('fl_ndvi_count', 0) or 0),
                'rm_ndvi': p.get('rm_ndvi_mean'), 'rm_n': int(p.get('rm_ndvi_count', 0) or 0),
            })
        return rows

    chunks = [(2000, 2006), (2007, 2012), (2013, 2018), (2019, 2024)]
    tasks = [(sp, y0, y1) for sp in SITES for (y0, y1) in chunks]
    all_rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(extract_site_chunk, sp, y0, y1): (sp['site'], y0, y1) for sp, y0, y1 in tasks}
        for fut in as_completed(futs):
            snm, y0, y1 = futs[fut]
            r = fut.result()
            all_rows.extend(r)
            print(f'  [{snm} {y0}-{y1}] {len(r)} clear scenes ({time.time()-t0:.1f}s)')

    df = pd.DataFrame(all_rows)
    # Deduplicate same-day overlapping path/rows by keeping max pixel coverage
    df['tot_n'] = df['fl_n'] + df['rm_n']
    df = (df.sort_values('tot_n', ascending=False)
          .drop_duplicates(subset=['site', 'kind', 'yr', 'doy'], keep='first')
          .drop(columns=['tot_n']))
    json.dump(df.to_dict(orient='records'), open(cache_p, 'w'))
    return df


def fetch_era5_night_wind(year_start=2000, year_end=2024):
    """Fetch true nighttime (00-12 UTC = 18:00-06:00 CST) scalar 10m wind speed from ERA5-Land HOURLY
    for Spring (DOY 80-125) and Autumn (DOY 265-305) across 2000-2024 using fast getRegion()."""
    cache_w = os.path.join(C.CACHE_DIR, f'p4_era5_night_wind_{year_start}_{year_end}.json')
    if os.path.exists(cache_w):
        return pd.DataFrame(json.load(open(cache_w)))

    print('[P4] Fetching nighttime (00-12 UTC) scalar 10m wind from ERA5-Land HOURLY (2000-2024)...')
    pt = ee.Geometry.Point([-85.92, 35.15])
    by_d = {}
    # 5-year chunks x 2 seasons -> 10 fast getRegion calls (< 15s total!)
    for y0 in range(year_start, year_end + 1, 5):
        y1 = min(year_end, y0 + 4)
        for d0, d1 in [(79, 126), (264, 306)]:
            col = (ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
                   .filterDate(f'{y0}-01-01', f'{y1+1}-01-01')
                   .filter(ee.Filter.calendarRange(d0, d1, 'day_of_year'))
                   .filter(ee.Filter.calendarRange(0, 11, 'hour'))
                   .select(['u_component_of_wind_10m', 'v_component_of_wind_10m']))
            tbl = col.getRegion(pt, 10000).getInfo()
            hdr = tbl[0]
            i_id = hdr.index('id')
            i_u = hdr.index('u_component_of_wind_10m')
            i_v = hdr.index('v_component_of_wind_10m')
            for r in tbl[1:]:
                u, v = r[i_u], r[i_v]
                if u is None or v is None:
                    continue
                # id format: YYYYMMDDTHH
                s = str(r[i_id])
                dt_s = f'{s[0:4]}-{s[4:6]}-{s[6:8]}'
                by_d.setdefault(dt_s, []).append(math.hypot(float(u), float(v)))
        print(f'  ERA5 night wind {y0}-{y1}: {len(by_d)} total days cached')
    rows = [{'date': dt_s, 'night_wspd_ms': round(float(np.mean(vals)), 3)} for dt_s, vals in sorted(by_d.items())]
    json.dump(rows, open(cache_w, 'w'))
    return pd.DataFrame(rows)


def build_annual_synoptic_weather():
    """Combine NOAA USC00408184 daily TMIN/PRCP (ERA5 fallback for missing days like Jan-May 2000)
    with ERA5-Land true nighttime (00-12 UTC) scalar wind speed (2000-2024)."""
    noaa = C.load_noaa_sewanee()
    era5 = C.fetch_era5_daily(2000, 2024)
    era5['DATE'] = pd.to_datetime(era5['date'])
    nw = fetch_era5_night_wind(2000, 2024)
    nw['DATE'] = pd.to_datetime(nw['date'])

    # Right join onto complete ERA5 2000-2024 dates so missing NOAA rows (e.g. Jan-May 2000) are NEVER dropped!
    m = pd.merge(noaa[noaa['year'].between(2000, 2024)][['DATE', 'TMAX', 'TMIN', 'PRCP']],
                 era5[['DATE', 'wspd_ms', 't2m_min_C', 'ssr_MJ', 'precip_mm']],
                 on='DATE', how='right')
    m = pd.merge(m, nw[['DATE', 'night_wspd_ms']], on='DATE', how='left')
    m['year'] = m['DATE'].dt.year
    m['month'] = m['DATE'].dt.month
    m['doy'] = m['DATE'].dt.dayofyear

    # Use NOAA TMIN/PRCP with ERA5-Land fallback where NOAA is missing
    m['tmin'] = m['TMIN'].fillna(m['t2m_min_C'])
    m['prcp'] = m['PRCP'].fillna(m['precip_mm'])
    m['w_night'] = m['night_wspd_ms'].fillna(m['wspd_ms'])

    # True tail separation on nighttime scalar wind speed (Q30 calm vs Q75 windy!)
    oct_all = m[m['doy'].between(265, 305)]
    w_calm = float(oct_all['w_night'].quantile(0.30))
    w_wind = float(oct_all['w_night'].quantile(0.75))
    w_spr_calm = float(m[m['doy'].between(80, 125)]['w_night'].quantile(0.35))
    print(f'[P4] Nighttime scalar wind thresholds (Oct DOY 265-305): calm <= {w_calm:.2f} m/s (Q30), windy >= {w_wind:.2f} m/s (Q75)')

    wx = []
    for yr, grp in m.groupby('year'):
        oct_w = grp[grp['doy'].between(265, 305)]  # Sep 22 - Nov 1 (pre-abscission / leaf-on frost window)
        spr_w = grp[grp['doy'].between(80, 125)]   # Mar 21 - May 5 (spring budbreak frost window)
        prcp_w = grp[grp['doy'].between(60, 120)]  # Mar-Apr precip (karst ponding)

        # Continuous cooling degree-days below 5 C split into CALM radiative (<=Q30) vs WINDY advective (>=Q75) nights!
        calm_oct_mask = (oct_w['w_night'] <= w_calm)
        wind_oct_mask = (oct_w['w_night'] >= w_wind)

        calm_oct_cdd = float(np.maximum(0.0, 5.0 - oct_w.loc[calm_oct_mask, 'tmin']).sum())
        wind_oct_cdd = float(np.maximum(0.0, 5.0 - oct_w.loc[wind_oct_mask, 'tmin']).sum())
        calm_oct_days = int((calm_oct_mask & (oct_w['tmin'] <= 4.0)).sum())
        wind_oct_days = int((wind_oct_mask & (oct_w['tmin'] <= 4.0)).sum())
        oct_night_wspd = float(oct_w['w_night'].mean())
        oct_daily_wspd = float(oct_w['wspd_ms'].mean())

        # Also strict freeze <= 2 C on calm vs windy nights
        calm_oct_frz = int((calm_oct_mask & (oct_w['tmin'] <= 2.0)).sum())
        wind_oct_frz = int((wind_oct_mask & (oct_w['tmin'] <= 2.0)).sum())

        # First-principles nonlinear 0 C differential-freeze physics:
        # (a) CalmCapOnly: plateau TMIN in [1.5, 7.0] C & calm -> rim stays >0 C while CAP freezes floor!
        # (b) HardRegionalFreeze: plateau TMIN <= 0.0 C -> BOTH rim and floor freeze, shrinking |dEOS|!
        calm_cap_only = int((calm_oct_mask & (oct_w['tmin'] >= 1.5) & (oct_w['tmin'] <= 7.0)).sum())
        hard_reg_freeze = int((oct_w['tmin'] <= 0.0).sum())
        oct_mean_tmin = float(oct_w['tmin'].mean())

        # Spring late frost (calm nighttime)
        spr_calm_mask = (spr_w['w_night'] <= w_spr_calm)
        apr_late_frost_cdd = float(np.maximum(0.0, 3.0 - spr_w.loc[spr_calm_mask, 'tmin']).sum())
        apr_all_frost_cdd = float(np.maximum(0.0, 3.0 - spr_w['tmin']).sum())
        apr_min_tmin = float(grp[grp['month'] == 4]['tmin'].min())
        mar_apr_precip = float(prcp_w['prcp'].sum())

        wx.append({
            'yr': int(yr),
            'CalmOctFreeze': round(calm_oct_cdd, 3),
            'WindyOctCold': round(wind_oct_cdd, 3),
            'OctNightWspd': round(oct_night_wspd, 3),
            'OctDailyWspd': round(oct_daily_wspd, 3),
            'CalmCapOnlyNights': calm_cap_only,
            'HardRegionalFreezeNights': hard_reg_freeze,
            'OctMeanTmin': round(oct_mean_tmin, 2),
            'calm_oct_days_le4': calm_oct_days,
            'wind_oct_days_le4': wind_oct_days,
            'calm_oct_frz_le2': calm_oct_frz,
            'wind_oct_frz_le2': wind_oct_frz,
            'AprLateFrost': round(apr_late_frost_cdd, 3),
            'AprAllFrost': round(apr_all_frost_cdd, 3),
            'AprMinTmin': round(apr_min_tmin, 2),
            'MarAprPrecip': round(mar_apr_precip, 1),
        })
    df_wx = pd.DataFrame(wx)
    for col in ['CalmOctFreeze', 'WindyOctCold', 'OctNightWspd', 'OctDailyWspd',
                'CalmCapOnlyNights', 'HardRegionalFreezeNights', 'AprLateFrost', 'AprAllFrost', 'MarAprPrecip']:
        sd = df_wx[col].std()
        df_wx[col + '_z'] = (df_wx[col] - df_wx[col].mean()) / (sd if sd > 1e-8 else 1.0)
    return df_wx, {'oct_calm_night_thresh_ms': round(w_calm, 3), 'oct_windy_night_thresh_ms': round(w_wind, 3)}


def fit_panel_and_regress(df_ts, df_wx):
    fits = []
    for (site, kind, yr), grp in df_ts.groupby(['site', 'kind', 'yr']):
        rf = C.fit_double_logistic(grp['doy'].values, grp['fl_ndvi'].values,
                                   min_obs=6, min_spring=2, min_autumn=2)
        rr = C.fit_double_logistic(grp['doy'].values, grp['rm_ndvi'].values,
                                   min_obs=6, min_spring=2, min_autumn=2)
        # Landsat 2000-2012 has ~12-18 clear obs/yr; keep strict R2 >= 0.80 (and verify >= 0.85 subset!)
        qc80 = (np.isfinite(rf['r2']) and rf['r2'] >= 0.80 and np.isfinite(rr['r2']) and rr['r2'] >= 0.80
                and np.isfinite(rf['sos50']) and np.isfinite(rr['sos50'])
                and 70 <= rf['sos50'] <= 165 and 70 <= rr['sos50'] <= 165
                and 245 <= rf['eos50'] <= 345 and 245 <= rr['eos50'] <= 345)
        fits.append({
            'site': site, 'kind': kind, 'yr': int(yr),
            'fl_sos': rf['sos50'], 'rm_sos': rr['sos50'],
            'fl_eos': rf['eos50'], 'rm_eos': rr['eos50'],
            'dSOS': (rf['sos50'] - rr['sos50']) if (np.isfinite(rf['sos50']) and np.isfinite(rr['sos50'])) else np.nan,
            'dEOS': (rf['eos50'] - rr['eos50']) if (np.isfinite(rf['eos50']) and np.isfinite(rr['eos50'])) else np.nan,
            'dGSL': (rf['gsl'] - rr['gsl']) if (np.isfinite(rf['gsl']) and np.isfinite(rr['gsl'])) else np.nan,
            'fl_r2': rf['r2'], 'rm_r2': rr['r2'], 'n_obs': rf['n_obs'],
            'qc80': bool(qc80), 'qc85': bool(rf['qc_pass'] and rr['qc_pass']),
        })
    df_p = pd.DataFrame(fits)
    df_p = pd.merge(df_p, df_wx, on='yr', how='left')

    # Also compute direct Lost Cove - OpenN Sewanee 25-yr panel (elevation-matched 330-385m!)
    lc = df_p[(df_p['site'] == 'Lost_Cove_TN') & (df_p['qc80'])].set_index('yr')
    op = df_p[(df_p['site'] == 'OpenN_Sewanee') & (df_p['qc80'])].set_index('yr')
    common_y = sorted(lc.index.intersection(op.index))
    df_lo = pd.DataFrame({
        'yr': common_y,
        'dSOS_lost_minus_openn': (lc.loc[common_y, 'fl_sos'] - op.loc[common_y, 'fl_sos']).values,
        'dEOS_lost_minus_openn': (lc.loc[common_y, 'fl_eos'] - op.loc[common_y, 'fl_eos']).values,
        'dGSL_lost_minus_openn': ((lc.loc[common_y, 'fl_eos'] - lc.loc[common_y, 'fl_sos']) -
                                  (op.loc[common_y, 'fl_eos'] - op.loc[common_y, 'fl_sos'])).values,
    })
    df_lo = pd.merge(df_lo, df_wx, on='yr', how='left')

    def run_fe_model(sub, y_col, x_cols):
        sub = sub.dropna(subset=[y_col] + x_cols).copy()
        if len(sub) < 10:
            return {'n': len(sub), 'note': 'insufficient_obs'}
        # Within-cove demeaning (exact Cove Fixed Effects!)
        sub[y_col + '_dm'] = sub[y_col] - sub.groupby('site')[y_col].transform('mean')
        X_dm = np.column_stack([
            (sub[c] - sub.groupby('site')[c].transform('mean')).values for c in x_cols
        ])
        res = C.ols_cluster_se(X_dm, sub[y_col + '_dm'].values, clusters=sub['yr'].values,
                               add_const=False, col_names=x_cols)
        res['n_years'] = int(sub['yr'].nunique())
        res['n_coves'] = int(sub['site'].nunique())
        return res

    results = {}
    for qc_col in ['qc80', 'qc85']:
        valid = df_p[df_p[qc_col]].copy()
        cl = valid[valid['kind'] == 'closed']
        opg = valid[valid['kind'] == 'open']

        # EOS identification: CalmOctFreeze_z + WindyOctCold_z + MarAprPrecip_z
        eos_cl = run_fe_model(cl, 'dEOS', ['CalmOctFreeze_z', 'WindyOctCold_z', 'MarAprPrecip_z'])
        eos_op = run_fe_model(opg, 'dEOS', ['CalmOctFreeze_z', 'WindyOctCold_z', 'MarAprPrecip_z'])
        # Nonlinear differential-freeze threshold model: CalmCapOnlyNights_z + HardRegionalFreezeNights_z + WindyOctCold_z
        eos_cl_thresh = run_fe_model(cl, 'dEOS', ['CalmCapOnlyNights_z', 'HardRegionalFreezeNights_z', 'WindyOctCold_z'])
        eos_op_thresh = run_fe_model(opg, 'dEOS', ['CalmCapOnlyNights_z', 'HardRegionalFreezeNights_z', 'WindyOctCold_z'])
        # Also bivariate (CalmOctFreeze only & WindyOctCold only) on annual closed-cove mean dEOS_t
        ann_cl = cl.groupby('yr').agg(dEOS=('dEOS', 'mean'), dSOS=('dSOS', 'mean'), n_c=('site', 'count')).reset_index()
        ann_cl = pd.merge(ann_cl, df_wx, on='yr', how='inner')
        ann_op = opg.groupby('yr').agg(dEOS=('dEOS', 'mean'), dSOS=('dSOS', 'mean'), n_c=('site', 'count')).reset_index()
        ann_op = pd.merge(ann_op, df_wx, on='yr', how='inner')

        # Annual Closed - Open DiD time series!
        ann_did = pd.merge(ann_cl[['yr', 'dEOS', 'dSOS']], ann_op[['yr', 'dEOS', 'dSOS']],
                           on='yr', suffixes=('_cl', '_op'))
        ann_did['DiD_dEOS'] = ann_did['dEOS_cl'] - ann_did['dEOS_op']
        ann_did['DiD_dSOS'] = ann_did['dSOS_cl'] - ann_did['dSOS_op']
        ann_did = pd.merge(ann_did, df_wx, on='yr', how='inner')

        # SOS identification: AprLateFrost_z + MarAprPrecip_z
        sos_cl = run_fe_model(cl, 'dSOS', ['AprLateFrost_z', 'MarAprPrecip_z'])
        sos_op = run_fe_model(opg, 'dSOS', ['AprLateFrost_z', 'MarAprPrecip_z'])

        # Wald test |beta_calm| > |beta_windy| (testing beta_calm - beta_windy < 0 since calm frost makes EOS EARLIER -> negative)
        wald_diff = None
        if 'vcov' in eos_cl:
            b = np.array(eos_cl['beta_vec'])
            V = np.array(eos_cl['vcov'])
            # Contrast c = [1, -1, 0] -> test beta_calm - beta_windy = 0
            c = np.array([1.0, -1.0, 0.0])
            diff_b = float(c @ b)
            se_d = math.sqrt(max(1e-12, float(c @ V @ c)))
            t_d = diff_b / se_d
            p_d = float(2.0 * stats.t.sf(abs(t_d), df=eos_cl['df']))
            wald_diff = {'beta_calm_minus_windy': round(diff_b, 4), 'se': round(se_d, 4), 't': round(t_d, 3), 'p': float(f'{p_d:.4g}')}

        # Variance explained of annual closed-cove mean dEOS_t by CalmOctFreeze
        r_calm, p_calm = stats.pearsonr(ann_cl['CalmOctFreeze'], ann_cl['dEOS']) if len(ann_cl) >= 6 else (np.nan, np.nan)
        r_wind, p_wind = stats.pearsonr(ann_cl['WindyOctCold'], ann_cl['dEOS']) if len(ann_cl) >= 6 else (np.nan, np.nan)
        r_nw, p_nw = stats.pearsonr(ann_cl['OctNightWspd'], ann_cl['dEOS']) if len(ann_cl) >= 6 else (np.nan, np.nan)
        r_did_calm, p_did_calm = stats.pearsonr(ann_did['CalmOctFreeze'], ann_did['DiD_dEOS']) if len(ann_did) >= 6 else (np.nan, np.nan)
        r_did_wind, p_did_wind = stats.pearsonr(ann_did['WindyOctCold'], ann_did['DiD_dEOS']) if len(ann_did) >= 6 else (np.nan, np.nan)
        r_did_nw, p_did_nw = stats.pearsonr(ann_did['OctNightWspd'], ann_did['DiD_dEOS']) if len(ann_did) >= 6 else (np.nan, np.nan)
        r_sos_fr, p_sos_fr = stats.pearsonr(ann_cl['AprLateFrost'], ann_cl['dSOS']) if len(ann_cl) >= 6 else (np.nan, np.nan)
        r_sos_pr, p_sos_pr = stats.pearsonr(ann_cl['MarAprPrecip'], ann_cl['dSOS']) if len(ann_cl) >= 6 else (np.nan, np.nan)

        results[qc_col] = {
            'closed_cove_years': int(len(cl)),
            'open_cove_years': int(len(opg)),
            'closed_mean_dEOS': round(float(cl['dEOS'].mean()), 3),
            'open_mean_dEOS': round(float(opg['dEOS'].mean()), 3),
            'closed_mean_dSOS': round(float(cl['dSOS'].mean()), 3),
            'open_mean_dSOS': round(float(opg['dSOS'].mean()), 3),
            'fe_eos_closed': {k: v for k, v in eos_cl.items() if k not in ('vcov', 'beta_vec')},
            'fe_eos_open': {k: v for k, v in eos_op.items() if k not in ('vcov', 'beta_vec')},
            'fe_eos_closed_threshold_model': {k: v for k, v in eos_cl_thresh.items() if k not in ('vcov', 'beta_vec')},
            'fe_eos_open_threshold_model': {k: v for k, v in eos_op_thresh.items() if k not in ('vcov', 'beta_vec')},
            'wald_calm_vs_windy_closed': wald_diff,
            'fe_sos_closed': {k: v for k, v in sos_cl.items() if k not in ('vcov', 'beta_vec')},
            'fe_sos_open': {k: v for k, v in sos_op.items() if k not in ('vcov', 'beta_vec')},
            'annual_closed_corr': {
                'n_yrs': int(len(ann_cl)),
                'r_CalmOctFreeze_dEOS': round(float(r_calm), 3), 'r2_CalmOctFreeze_dEOS': round(float(r_calm ** 2), 3), 'p_CalmOctFreeze': float(f'{p_calm:.4g}'),
                'r_WindyOctCold_dEOS': round(float(r_wind), 3), 'p_WindyOctCold': float(f'{p_wind:.4g}'),
                'r_OctNightWspd_dEOS': round(float(r_nw), 3), 'p_OctNightWspd': float(f'{p_nw:.4g}'),
                'r_AprLateFrost_dSOS': round(float(r_sos_fr), 3), 'p_AprLateFrost': float(f'{p_sos_fr:.4g}'),
                'r_MarAprPrecip_dSOS': round(float(r_sos_pr), 3), 'p_MarAprPrecip': float(f'{p_sos_pr:.4g}'),
            },
            'annual_DiD_closed_minus_open_corr': {
                'n_yrs': int(len(ann_did)),
                'mean_DiD_dEOS': round(float(ann_did['DiD_dEOS'].mean()), 3),
                'r_CalmOctFreeze_DiD_dEOS': round(float(r_did_calm), 3), 'r2_CalmOctFreeze_DiD_dEOS': round(float(r_did_calm ** 2), 3), 'p_Calm': float(f'{p_did_calm:.4g}'),
                'r_WindyOctCold_DiD_dEOS': round(float(r_did_wind), 3), 'p_Windy': float(f'{p_did_wind:.4g}'),
                'r_OctNightWspd_DiD_dEOS': round(float(r_did_nw), 3), 'p_OctNightWspd': float(f'{p_did_nw:.4g}'),
            }
        }

    # Direct Lost Cove - OpenN 25-year panel regressions + wind-mixing correlation
    lo_eos = C.ols_cluster_se(df_lo[['CalmOctFreeze_z', 'WindyOctCold_z', 'MarAprPrecip_z']].values,
                              df_lo['dEOS_lost_minus_openn'].values,
                              col_names=['CalmOctFreeze_z', 'WindyOctCold_z', 'MarAprPrecip_z'])
    lo_sos = C.ols_cluster_se(df_lo[['AprLateFrost_z', 'MarAprPrecip_z']].values,
                              df_lo['dSOS_lost_minus_openn'].values,
                              col_names=['AprLateFrost_z', 'MarAprPrecip_z'])
    r_lo_nw, p_lo_nw = stats.pearsonr(df_lo['OctNightWspd'], df_lo['dEOS_lost_minus_openn'])
    r_lo_dw, p_lo_dw = stats.pearsonr(df_lo['OctDailyWspd'], df_lo['dEOS_lost_minus_openn'])

    # Cross-sensor validation against P2 HLS 2018-2024 (if p2.json exists)
    cross_sensor = None
    p2_path = os.path.join(C.HERE, 'p2.json')
    if os.path.exists(p2_path):
        p2j = json.load(open(p2_path))
        hls_s = p2j.get('signal_A_all', {}).get('hls_ndvi', {}).get('d_sos50_by_yr', {})
        hls_e = p2j.get('signal_A_all', {}).get('hls_ndvi', {}).get('d_eos50_by_yr', {})
        lo_idx = df_lo.set_index('yr')
        cy = sorted(int(y) for y in hls_e.keys() if int(y) in lo_idx.index)
        if len(cy) >= 4:
            rs, ps = stats.pearsonr([lo_idx.loc[y, 'dSOS_lost_minus_openn'] for y in cy], [hls_s[str(y)] for y in cy])
            re_, pe_ = stats.pearsonr([lo_idx.loc[y, 'dEOS_lost_minus_openn'] for y in cy], [hls_e[str(y)] for y in cy])
            cross_sensor = {
                'n_yrs': len(cy),
                'r_landsat_vs_hls_dSOS': round(float(rs), 3), 'p_dSOS': float(f'{ps:.4g}'),
                'r_landsat_vs_hls_dEOS': round(float(re_), 3), 'p_dEOS': float(f'{pe_:.4g}'),
            }

    lo_summary = {
        'n_years': int(len(df_lo)),
        'mean_dSOS': round(float(df_lo['dSOS_lost_minus_openn'].mean()), 3),
        'mean_dEOS': round(float(df_lo['dEOS_lost_minus_openn'].mean()), 3),
        'mean_dGSL': round(float(df_lo['dGSL_lost_minus_openn'].mean()), 3),
        'frac_yrs_dSOS_gt0': round(float((df_lo['dSOS_lost_minus_openn'] > 0).mean()), 3),
        'frac_yrs_dEOS_lt0': round(float((df_lo['dEOS_lost_minus_openn'] < 0).mean()), 3),
        'r_OctNightWspd_dEOS': round(float(r_lo_nw), 3), 'p_OctNightWspd': float(f'{p_lo_nw:.4g}'),
        'r_OctDailyWspd_dEOS': round(float(r_lo_dw), 3), 'p_OctDailyWspd': float(f'{p_lo_dw:.4g}'),
        'cross_sensor_2018_2024_vs_HLS': cross_sensor,
        'eos_ols': {k: v for k, v in lo_eos.items() if k not in ('vcov', 'beta_vec')},
        'sos_ols': {k: v for k, v in lo_sos.items() if k not in ('vcov', 'beta_vec')},
        'by_year': df_lo[['yr', 'dSOS_lost_minus_openn', 'dEOS_lost_minus_openn', 'CalmOctFreeze', 'WindyOctCold', 'OctNightWspd', 'AprLateFrost', 'MarAprPrecip']].round(2).to_dict(orient='records'),
    }

    # Cross-sensor check against P2 S2 2018-2024 if cached
    return results, lo_summary, df_p


def main():
    t0 = time.time()
    C.init_ee()
    df_ts = extract_landsat_25yr()
    df_wx, wx_meta = build_annual_synoptic_weather()
    results, lo_summary, df_p = fit_panel_and_regress(df_ts, df_wx)

    # Evaluate pre-registered P4 gates (primary = qc80 full 25-yr panel)
    q80 = results['qc80']
    c_eos = q80['fe_eos_closed'].get('coefs', {})
    b_calm = c_eos.get('CalmOctFreeze_z', {}).get('beta', 0.0)
    p_calm = c_eos.get('CalmOctFreeze_z', {}).get('p', 1.0)
    b_wind = c_eos.get('WindyOctCold_z', {}).get('beta', 0.0)
    t_wind = c_eos.get('WindyOctCold_z', {}).get('t', 99.0)
    r2_calm = q80['annual_closed_corr']['r2_CalmOctFreeze_dEOS']
    c_sos = q80['fe_sos_closed'].get('coefs', {})
    b_prcp_sos = c_sos.get('MarAprPrecip_z', {}).get('beta', 0.0)
    p_prcp_sos = c_sos.get('MarAprPrecip_z', {}).get('p', 1.0)

    passed = (
        abs(b_calm) > abs(b_wind)
        and b_calm < 0 and p_calm < 0.05
        and r2_calm >= 0.35
        and abs(t_wind) < 1.5
    )
    partial = (abs(b_calm) > abs(b_wind) and b_calm < 0 and p_calm < 0.10 and abs(t_wind) < 1.8)

    out = {
        'experiment': 'P4_25yr_weather_panel',
        'elapsed_s': round(time.time() - t0, 1),
        'weather_thresholds': wx_meta,
        'weather_cov_corr': df_wx[['CalmOctFreeze', 'WindyOctCold', 'AprLateFrost', 'MarAprPrecip']].corr().round(3).to_dict(),
        'panel_results': results,
        'lost_minus_openn_25yr': lo_summary,
        'gate_check': {
            'beta_CalmOctFreeze_z': b_calm,
            'p_CalmOctFreeze_z': p_calm,
            'beta_WindyOctCold_z': b_wind,
            't_WindyOctCold_z': t_wind,
            'r2_CalmOctFreeze_annual_closed_dEOS': r2_calm,
            'beta_MarAprPrecip_on_dSOS': b_prcp_sos,
            'p_MarAprPrecip_on_dSOS': p_prcp_sos,
            'lost_openn_25yr_mean_dSOS': lo_summary['mean_dSOS'],
            'lost_openn_25yr_mean_dEOS': lo_summary['mean_dEOS'],
            'lost_openn_25yr_frac_dEOS_lt0': lo_summary['frac_yrs_dEOS_lt0'],
            'verdict': 'PASS' if passed else ('PARTIAL' if partial else 'FAIL_STATIC_OR_NOISY'),
        }
    }
    C.save_json('p4.json', out)
    print('\n=== P4 SUMMARY ===')
    print(json.dumps(out['gate_check'], indent=2))


if __name__ == '__main__':
    main()
