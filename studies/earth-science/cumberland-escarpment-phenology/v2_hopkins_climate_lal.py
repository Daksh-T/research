"""Stage 4 (4C): Clean Hopkins (1918) Baseline, Multi-Decadal Climate Context, & Public Sewanee LAL Controls.

Delivers:
1. Clean apples-to-apples MODIS MCD12Q2 v061 (2001-2023) baseline:
   - MidGreenup_1 (50% threshold matching S2 SOS50) + MidGreendown_1 (50% EOS)
   - Strict QA_Overall_1 <= 1 and NLCD-41 deciduous forest mask + 3DEP elevation
   - Transparent contrast against legacy 05_lapse_trend.py (Greenup_1, Hansen >50%, no QA)
     and legacy 06_landsat_trend.py (1984-2024 unharmonized LEDAPS->LaSRC 2013 step artifact)
2. Sewanee Domain Hopkins Decomposition (Escarpment core lapse vs. Signal A cold-air pool
   vs. Signal B plateau-cap oak guild turnover).
3. NOAA GHCN-Daily Sewanee USC00408184 (1958-2025) + ERA5-Land (2000-2024) spring/autumn
   warming, freeze-date, and thermal phenology sensitivity (days / deg C) context.
4. Public Sewanee Landscape Analysis Lab (LAL) ArcGIS REST controls (no institutional login):
   - SewaneeHarvestRecord_WFL1 (120 timber harvest polygons)
   - Domain_Watershed_Boundaries (51 watershed polygons)
   - Sensitivity re-test of Signal A, Signal B, and P5 Flat-vs-Steep after masking harvested stands.

Outputs:
  data/v2_hopkins_climate_lal.json (strict RFC 8259 JSON)
"""
import json
import math
import os
import sys
import time
import numpy as np
import pandas as pd
import requests
from matplotlib.path import Path
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'profiling'))
import ee
import common as C
import eslib as L

OUT_PATH = os.path.join(HERE, 'data', 'v2_hopkins_climate_lal.json')
CACHE_MODIS = os.path.join(C.CACHE_DIR, 'v2_modis_midgreenup_clean_2001_2023.json')
CACHE_MODIS_DECOMP = os.path.join(C.CACHE_DIR, 'v2_modis_3factor_decomp.json')
CACHE_LAL_HARV = os.path.join(C.CACHE_DIR, 'lal_harvests.geojson')
CACHE_LAL_WSHD = os.path.join(C.CACHE_DIR, 'lal_watersheds.geojson')

LAL_HARV_URL = (
    'https://services1.arcgis.com/RpVCghfSj7XHqAKQ/arcgis/rest/services/'
    'SewaneeHarvestRecord_WFL1/FeatureServer/0/query'
)
LAL_WSHD_URL = (
    'https://services1.arcgis.com/RpVCghfSj7XHqAKQ/arcgis/rest/services/'
    'Domain_Watershed_Boundaries/FeatureServer/15/query'
)


def trend_stats(x, y):
    """OLS + Theil-Sen + Kendall tau trend summary."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 5:
        return {'n': int(n)}
    lr = stats.linregress(x, y)
    ts = stats.theilslopes(y, x)
    kt = stats.kendalltau(x, y)
    return {
        'n': int(n),
        'year_min': int(x.min()), 'year_max': int(x.max()),
        'mean': round(float(np.mean(y)), 3),
        'sd': round(float(np.std(y, ddof=1)), 3),
        'se': round(float(np.std(y, ddof=1) / math.sqrt(n)), 3),
        'ols_per_decade': round(float(lr.slope * 10.0), 4),
        'ols_se_per_decade': round(float(lr.stderr * 10.0), 4),
        'ols_r2': round(float(lr.rvalue ** 2), 4),
        'ols_t': round(float(lr.slope / lr.stderr), 3) if lr.stderr > 0 else None,
        'ols_p': float(f'{lr.pvalue:.4g}'),
        'theilsen_per_decade': round(float(ts.slope * 10.0), 4),
        'kendall_tau': round(float(kt.statistic), 3),
        'kendall_p': float(f'{kt.pvalue:.4g}'),
    }


def fetch_clean_modis_2001_2023():
    """Run clean apples-to-apples MODIS MCD12Q2 v061 MidGreenup_1 & MidGreendown_1 (QA<=1, NLCD-41)."""
    if os.path.exists(CACHE_MODIS):
        rows = json.load(open(CACHE_MODIS))
        print(f'[4C] Loaded cached clean MODIS MCD12Q2 baseline ({len(rows)} years)')
        return rows

    print('[4C] Running clean MODIS MCD12Q2 MidGreenup_1 (QA_Overall_1<=1, NLCD-41, 2001-2023)...')
    dom = C.get_domain()
    region = dom.bounds().buffer(12000).bounds()
    dem = C.get_dem_3dep().rename('elev')
    nlcd = C.get_nlcd_2021()
    decid = nlcd.eq(41)
    mcd = ee.ImageCollection('MODIS/061/MCD12Q2')

    rows = []
    for y in range(2001, 2024):
        img = mcd.filterDate(f'{y}-01-01', f'{y}-12-31').first()
        qa = img.select('QA_Overall_1')
        days_to_year = ee.Date.fromYMD(y, 1, 1).difference(ee.Date('1970-01-01'), 'day')

        mg = img.select('MidGreenup_1')
        sos = mg.subtract(days_to_year).rename('sos')
        good_sos = (qa.lte(1).And(mg.neq(32767)).And(sos.gt(45)).And(sos.lt(180)).And(decid))
        sos_m = sos.updateMask(good_sos)

        md = img.select('MidGreendown_1')
        eos = md.subtract(days_to_year).rename('eos')
        good_eos = (qa.lte(1).And(md.neq(32767)).And(eos.gt(235)).And(eos.lt(355)).And(decid))
        eos_m = eos.updateMask(good_eos)

        gsl_m = eos_m.subtract(sos_m).rename('gsl')

        fit_sos = dem.rename('x').addBands(sos_m.rename('y')).updateMask(good_sos).reduceRegion(
            ee.Reducer.linearFit(), region, 500, maxPixels=1e9, tileScale=8, bestEffort=True
        ).getInfo()
        fit_eos = dem.rename('x').addBands(eos_m.rename('y')).updateMask(good_eos).reduceRegion(
            ee.Reducer.linearFit(), region, 500, maxPixels=1e9, tileScale=8, bestEffort=True
        ).getInfo()
        stats_img = ee.Image.cat([sos_m, eos_m, gsl_m]).reduceRegion(
            ee.Reducer.mean().combine(ee.Reducer.count(), '', True),
            region, 500, maxPixels=1e9, tileScale=8, bestEffort=True
        ).getInfo()

        if fit_sos.get('scale') is None:
            continue
        rows.append({
            'year': y,
            'midgreenup_mean_doy': round(float(stats_img['sos_mean']), 2),
            'midgreenup_lapse_d_per_100m': round(float(fit_sos['scale'] * 100.0), 3),
            'midgreenup_n': int(stats_img.get('sos_count') or 0),
            'midgreendown_mean_doy': round(float(stats_img['eos_mean']), 2) if stats_img.get('eos_mean') is not None else None,
            'midgreendown_lapse_d_per_100m': round(float(fit_eos['scale'] * 100.0), 3) if fit_eos.get('scale') is not None else None,
            'gsl_mean_d': round(float(stats_img['gsl_mean']), 2) if stats_img.get('gsl_mean') is not None else None,
        })
        print(f"  {y}: MidGreenup={rows[-1]['midgreenup_mean_doy']:.1f} (lapse={rows[-1]['midgreenup_lapse_d_per_100m']:.2f} d/100m, n={rows[-1]['midgreenup_n']}) | "
              f"MidGreendown={rows[-1]['midgreendown_mean_doy']}")
    json.dump(rows, open(CACHE_MODIS, 'w'), indent=2)
    return rows


def fetch_modis_3factor_decomposition():
    """Decompose the 4.09 -> 3.19 d/100m MODIS lapse difference across three sequential factors:
    (1) QA_Overall_1 <= 1 filter, (2) NLCD-41 deciduous mask vs. Hansen treecover2000 > 50%,
    and (3) MidGreenup_1 (50%) vs. Greenup_1 (15% onset) phenological metric definition."""
    if os.path.exists(CACHE_MODIS_DECOMP):
        res = json.load(open(CACHE_MODIS_DECOMP))
        print('[4C] Loaded cached 3-factor MODIS decomposition')
        return res

    print('[4C] Running 3-factor MODIS MCD12Q2 decomposition in GEE (2001-2023)...')
    try:
        C.init_ee()
        dom = C.get_domain()
        region = dom.bounds().buffer(12000).bounds()
        dem = C.get_dem_3dep().rename('x')
        nlcd_decid = C.get_nlcd_2021().eq(41)
        hansen_tc50 = ee.Image('UMD/hansen/global_forest_change_2023_v1_11').select('treecover2000').gt(50)
        mcd = ee.ImageCollection('MODIS/061/MCD12Q2')

        years = ee.List.sequence(2001, 2023)

        def per_year(y_ee):
            y_ee = ee.Number(y_ee)
            img = mcd.filterDate(ee.Date.fromYMD(y_ee, 1, 1), ee.Date.fromYMD(y_ee, 12, 31)).first()
            qa = img.select('QA_Overall_1')
            d0 = ee.Date.fromYMD(y_ee, 1, 1).difference(ee.Date('1970-01-01'), 'day')

            g15 = img.select('Greenup_1').subtract(d0).rename('y')
            g50 = img.select('MidGreenup_1').subtract(d0).rename('y')
            ok15 = img.select('Greenup_1').neq(32767).And(g15.gt(45)).And(g15.lt(180))
            ok50 = img.select('MidGreenup_1').neq(32767).And(g50.gt(45)).And(g50.lt(180))

            def get_slope(band_img, mask_img):
                fit = dem.addBands(band_img).updateMask(mask_img).reduceRegion(
                    ee.Reducer.linearFit(), region, 500, maxPixels=1e9, tileScale=8, bestEffort=True
                )
                return ee.Number(fit.get('scale')).multiply(100.0)

            return ee.Feature(None, {
                'year': y_ee,
                'A_g15_hansen_noqa': get_slope(g15, ok15.And(hansen_tc50)),
                'B_g15_hansen_qa1': get_slope(g15, ok15.And(hansen_tc50).And(qa.lte(1))),
                'C_g15_nlcd41_qa1': get_slope(g15, ok15.And(nlcd_decid).And(qa.lte(1))),
                'D_g50_nlcd41_qa1': get_slope(g50, ok50.And(nlcd_decid).And(qa.lte(1))),
                'E_g50_hansen_noqa': get_slope(g50, ok50.And(hansen_tc50)),
            })

        feats = ee.FeatureCollection(years.map(per_year)).getInfo()['features']
        rows = [f['properties'] for f in feats]
        df = pd.DataFrame(rows).sort_values('year')

        def summ(col):
            v = df[col].dropna().values.astype(float)
            return {
                'n': int(len(v)),
                'mean_d_per_100m': round(float(np.mean(v)), 3),
                'sd_d_per_100m': round(float(np.std(v, ddof=1)), 3),
                'se_d_per_100m': round(float(np.std(v, ddof=1) / math.sqrt(len(v))), 3),
            }

        sA = summ('A_g15_hansen_noqa')
        sB = summ('B_g15_hansen_qa1')
        sC = summ('C_g15_nlcd41_qa1')
        sD = summ('D_g50_nlcd41_qa1')
        sE = summ('E_g50_hansen_noqa')
        res = {
            'status': 'COMPLETED_GEE',
            'configs': {
                'A_Greenup15_Hansen50_NoQA': sA,
                'B_Greenup15_Hansen50_QA1': sB,
                'C_Greenup15_NLCD41_QA1': sC,
                'D_MidGreenup50_NLCD41_QA1': sD,
                'E_MidGreenup50_Hansen50_NoQA': sE,
            },
            'sequential_deltas_d_per_100m': {
                'delta_qa_filter_B_minus_A': round(sB['mean_d_per_100m'] - sA['mean_d_per_100m'], 3),
                'delta_nlcd41_mask_C_minus_B': round(sC['mean_d_per_100m'] - sB['mean_d_per_100m'], 3),
                'delta_50pct_vs_15pct_D_minus_C': round(sD['mean_d_per_100m'] - sC['mean_d_per_100m'], 3),
                'total_D_minus_A': round(sD['mean_d_per_100m'] - sA['mean_d_per_100m'], 3),
            },
            'per_year': rows,
        }
        json.dump(C.sanitize_json(res), open(CACHE_MODIS_DECOMP, 'w'), indent=2)
        print('  3-factor MODIS decomposition complete:', json.dumps(res['sequential_deltas_d_per_100m']))
        return res
    except Exception as ex:
        print(f'[4C] WARNING: 3-factor MODIS GEE decomposition failed ({ex}); recording unavailable status')
        return {'status': 'UNAVAILABLE_GEE_ERROR', 'error': str(ex)}


def diagnose_legacy_landsat_step():
    """Diagnose the LEDAPS (L4/5/7, 1984-2012) -> LaSRC (L8/9, 2013-2024) collinearity / non-identifiability
    in legacy data/landsat_trend.json."""
    lt = json.load(open(os.path.join(HERE, 'data', 'landsat_trend.json')))
    df = pd.DataFrame(lt['per_year'])
    yrs = df['year'].values.astype(float)
    sos = df['mean_sos'].values.astype(float)
    is_oli = (yrs >= 2013).astype(float)

    naive = trend_stats(yrs, sos)
    pre = df[df['year'] < 2013]
    post = df[df['year'] >= 2013]

    # Step-dummy regression: SOS ~ (year - 2000)/10 + I(year >= 2013)
    X_step = np.column_stack([(yrs - 2000.0) / 10.0, is_oli])
    ols_step = C.ols_cluster_se(X_step, sos, col_names=['decade_slope', 'oli_2013_step_d'])
    r_collin = float(stats.pearsonr(yrs, is_oli).statistic)
    vif = 1.0 / max(1e-9, 1.0 - r_collin ** 2)

    t_era, p_era = stats.ttest_ind(post['mean_sos'], pre['mean_sos'], equal_var=False)
    return {
        'naive_1984_2024': naive,
        'pre_2013_ledaps_mean_sos': round(float(pre['mean_sos'].mean()), 2),
        'pre_2013_n': int(len(pre)),
        'pre_2013_trend': trend_stats(pre['year'].values, pre['mean_sos'].values),
        'post_2013_lasrc_mean_sos': round(float(post['mean_sos'].mean()), 2),
        'post_2013_n': int(len(post)),
        'post_2013_trend': trend_stats(post['year'].values, post['mean_sos'].values),
        'raw_era_shift_post_minus_pre_d': round(float(post['mean_sos'].mean() - pre['mean_sos'].mean()), 2),
        'era_welch_p': float(f'{p_era:.4g}'),
        'collinearity_r_year_vs_oli2013': round(r_collin, 3),
        'collinearity_vif': round(vif, 2),
        'step_dummy_regression': {
            'r2': ols_step['r2'],
            'decade_slope_controlling_for_2013_step': ols_step['coefs']['decade_slope'],
            'oli_2013_step_d': ols_step['coefs']['oli_2013_step_d'],
            'interpretation': (
                'Collinearity non-identifiability (r(year, I[year>=2013]) = '
                f'{r_collin:.2f}, VIF = {vif:.1f}): neither the 2013 processor-era step '
                f'({ols_step["coefs"]["oli_2013_step_d"]["beta"]:+.2f} +/- '
                f'{ols_step["coefs"]["oli_2013_step_d"]["se"]:.2f} d, '
                f'p = {ols_step["coefs"]["oli_2013_step_d"]["p"]}) nor the secular slope '
                f'({ols_step["coefs"]["decade_slope"]["beta"]:+.2f} +/- '
                f'{ols_step["coefs"]["decade_slope"]["se"]:.2f} d/dec, '
                f'p = {ols_step["coefs"]["decade_slope"]["p"]}) can be separated cleanly in '
                'unharmonized Collection 2 Landsat.'
            ),
        },
        'per_year': lt['per_year'],
    }


def analyze_noaa_and_era5(modis_rows):
    """Multi-decadal climate context from NOAA Sewanee USC00408184 (1895-1943 & 2000-2025;
    1944-1999 temp hiatus documented) + ERA5-Land (2000-2024)."""
    df = C.load_noaa_sewanee()
    df['TMEAN'] = 0.5 * (df['TMAX'] + df['TMIN'])

    annual = []
    for yr, g in df.groupby('year'):
        if yr < 1895 or yr > 2025:
            continue
        spr = g[g['month'].isin([3, 4])]
        aut = g[g['month'].isin([9, 10, 11])]
        oct_g = g[g['month'] == 10]
        jan_jun = g[g['month'] <= 6]
        jul_dec = g[g['month'] >= 7]

        rec = {'year': int(yr)}
        # Require >=80% completeness (49/61 spring days, 73/91 autumn days)
        if spr['TMEAN'].notna().sum() >= 49:
            rec['spr_tmean_C'] = round(float(spr['TMEAN'].mean()), 3)
            rec['spr_tmin_C'] = round(float(spr['TMIN'].mean()), 3)
            rec['spr_tmax_C'] = round(float(spr['TMAX'].mean()), 3)
            rec['spr_gdd5'] = round(float(np.maximum(0.0, spr['TMEAN'].dropna() - 5.0).sum()), 1)
        if aut['TMEAN'].notna().sum() >= 73:
            rec['aut_tmean_C'] = round(float(aut['TMEAN'].mean()), 3)
            rec['aut_tmin_C'] = round(float(aut['TMIN'].mean()), 3)
            rec['aut_tmax_C'] = round(float(aut['TMAX'].mean()), 3)
        if oct_g['TMEAN'].notna().sum() >= 25:
            rec['oct_tmean_C'] = round(float(oct_g['TMEAN'].mean()), 3)
            rec['oct_tmin_C'] = round(float(oct_g['TMIN'].mean()), 3)
        if jan_jun['TMIN'].notna().sum() >= 145:
            frz_s = jan_jun[jan_jun['TMIN'] <= 0.0]
            if len(frz_s):
                rec['last_spring_freeze_doy'] = int(frz_s['doy'].max())
        if jul_dec['TMIN'].notna().sum() >= 145:
            frz_a = jul_dec[jul_dec['TMIN'] <= 0.0]
            if len(frz_a):
                rec['first_autumn_freeze_doy'] = int(frz_a['doy'].min())
        if 'last_spring_freeze_doy' in rec and 'first_autumn_freeze_doy' in rec:
            rec['frost_free_days'] = int(rec['first_autumn_freeze_doy'] - rec['last_spring_freeze_doy'])
        if len(rec) > 1:
            annual.append(rec)

    adf = pd.DataFrame(annual)
    noaa_modern = {}
    noaa_century = {}
    noaa_era_shift = {}
    cols_eval = ['spr_tmean_C', 'spr_tmin_C', 'spr_tmax_C', 'spr_gdd5',
                 'aut_tmean_C', 'aut_tmin_C', 'oct_tmean_C', 'oct_tmin_C',
                 'last_spring_freeze_doy', 'first_autumn_freeze_doy', 'frost_free_days']
    for col in cols_eval:
        if col in adf.columns:
            sub_all = adf[['year', col]].dropna()
            sub_mod = sub_all[sub_all['year'] >= 2000]
            sub_hop = sub_all[sub_all['year'] <= 1945]
            noaa_modern[col] = trend_stats(sub_mod['year'].values, sub_mod[col].values)
            noaa_century[col] = trend_stats(sub_all['year'].values, sub_all[col].values)
            if len(sub_hop) >= 10 and len(sub_mod) >= 10:
                t_e, p_e = stats.ttest_ind(sub_mod[col].values, sub_hop[col].values, equal_var=False)
                noaa_era_shift[col] = {
                    'hopkins_era_1895_1943_n': int(len(sub_hop)),
                    'hopkins_era_mean': round(float(sub_hop[col].mean()), 3),
                    'modern_era_2000_2025_n': int(len(sub_mod)),
                    'modern_era_mean': round(float(sub_mod[col].mean()), 3),
                    'shift_modern_minus_hopkins': round(float(sub_mod[col].mean() - sub_hop[col].mean()), 3),
                    'welch_t': round(float(t_e), 3),
                    'welch_p': float(f'{p_e:.4g}'),
                }

    # Also compute 2001-2023 NOAA trends & thermal sensitivity of MODIS MidGreenup_1
    mdf = pd.DataFrame(modis_rows)
    merged = mdf.merge(adf, on='year', how='inner')
    sub_sens = merged[['year', 'midgreenup_mean_doy', 'spr_tmean_C']].dropna()
    lr_sens = stats.linregress(sub_sens['spr_tmean_C'], sub_sens['midgreenup_mean_doy'])
    sens_eos_sub = merged[['year', 'midgreendown_mean_doy', 'oct_tmean_C']].dropna()
    lr_eos = stats.linregress(sens_eos_sub['oct_tmean_C'], sens_eos_sub['midgreendown_mean_doy']) if len(sens_eos_sub) >= 8 else None

    # ERA5-Land 2000-2024 trends
    e5 = C.fetch_era5_daily(2000, 2024)
    e5['dt'] = pd.to_datetime(e5['date'])
    e5['month'] = e5['dt'].dt.month
    e5_spr = e5[e5['month'].isin([3, 4])].groupby('year').agg(
        spr_t2m_C=('t2m_C', 'mean'), spr_tmin_C=('t2m_min_C', 'mean'), spr_wspd=('wspd_ms', 'mean')
    ).reset_index()
    e5_aut = e5[e5['month'].isin([9, 10, 11])].groupby('year').agg(
        aut_t2m_C=('t2m_C', 'mean'), aut_tmin_C=('t2m_min_C', 'mean'), aut_wspd=('wspd_ms', 'mean')
    ).reset_index()

    return {
        'noaa_station': 'USC00408184 (Sewanee, TN; elev 564 m; raw GHCN-Daily 1895-1943 & 2000-2025, 1944-1999 sensor hiatus)',
        'noaa_homogenization_caveat': (
            'Raw GHCN-Daily station records are unhomogenized for time-of-observation bias (TOBS), '
            'sensor transitions (liquid-in-glass to MMTS/electronic), or micro-site relocations '
            'across the 1944-1999 hiatus; century-scale era shifts provide contextual background '
            'rather than formal climate-attribution estimates.'
        ),
        'noaa_2000_2025_modern_trends': noaa_modern,
        'noaa_1895_2025_century_trends': noaa_century,
        'noaa_modern_vs_hopkins_era_shift': noaa_era_shift,
        'noaa_2001_2023_spr_tmean_trend': trend_stats(sub_sens['year'].values, sub_sens['spr_tmean_C'].values),
        'modis_thermal_sensitivity': {
            'n_years': int(len(sub_sens)),
            'days_per_degC_spring': round(float(lr_sens.slope), 3),
            'se_days_per_degC': round(float(lr_sens.stderr), 3),
            'r': round(float(lr_sens.rvalue), 3),
            'r2': round(float(lr_sens.rvalue ** 2), 4),
            'p': float(f'{lr_sens.pvalue:.4g}'),
            'autumn_oct_days_per_degC': round(float(lr_eos.slope), 3) if lr_eos else None,
            'autumn_oct_r': round(float(lr_eos.rvalue), 3) if lr_eos else None,
            'autumn_oct_p': float(f'{lr_eos.pvalue:.4g}') if lr_eos else None,
        },
        'era5_2000_2024_trends': {
            'spr_t2m_C': trend_stats(e5_spr['year'].values, e5_spr['spr_t2m_C'].values),
            'spr_tmin_C': trend_stats(e5_spr['year'].values, e5_spr['spr_tmin_C'].values),
            'aut_t2m_C': trend_stats(e5_aut['year'].values, e5_aut['aut_t2m_C'].values),
            'aut_tmin_C': trend_stats(e5_aut['year'].values, e5_aut['aut_tmin_C'].values),
        },
        'noaa_annual_series': adf.to_dict(orient='records'),
    }


def fetch_geojson_arcgis(url, cache_path):
    if os.path.exists(cache_path):
        return json.load(open(cache_path))
    r = requests.get(url, params={'where': '1=1', 'outFields': '*', 'f': 'geojson'}, timeout=30)
    r.raise_for_status()
    gj = r.json()
    json.dump(gj, open(cache_path, 'w'))
    return gj


def points_in_geojson_polygons(lons, lats, features):
    """Exact point-in-polygon matching using matplotlib.path.Path; returns matched feature index (-1 if none)."""
    pts = np.column_stack([lons, lats])
    hit_idx = np.full(len(pts), -1, dtype=np.int32)
    for fi, feat in enumerate(features):
        geom = feat.get('geometry')
        if not geom:
            continue
        gtype = geom['type']
        polys = [geom['coordinates']] if gtype == 'Polygon' else (geom['coordinates'] if gtype == 'MultiPolygon' else [])
        for poly in polys:
            if not poly or len(poly[0]) < 3:
                continue
            outer = np.asarray(poly[0], dtype=np.float64)
            # Fast bbox pre-filter
            bb = ((pts[:, 0] >= outer[:, 0].min()) & (pts[:, 0] <= outer[:, 0].max()) &
                  (pts[:, 1] >= outer[:, 1].min()) & (pts[:, 1] <= outer[:, 1].max()))
            if not np.any(bb):
                continue
            idx_cand = np.where(bb)[0]
            inside = Path(outer[:, :2]).contains_points(pts[idx_cand])
            # Exclude holes if present
            if len(poly) > 1 and np.any(inside):
                for hole in poly[1:]:
                    if len(hole) >= 3:
                        inside &= ~Path(np.asarray(hole)[:, :2]).contains_points(pts[idx_cand])
            hit_idx[idx_cand[inside]] = fi
    return hit_idx


def audit_lal_public_controls():
    """Download public Sewanee LAL Harvest & Watershed layers (no login) and run harvest-masking sensitivity."""
    print('[4C] Fetching public Sewanee LAL GeoJSON layers (Harvests & Domain Watersheds)...')
    harv_gj = fetch_geojson_arcgis(LAL_HARV_URL, CACHE_LAL_HARV)
    wshd_gj = fetch_geojson_arcgis(LAL_WSHD_URL, CACHE_LAL_WSHD)
    h_feats = harv_gj.get('features', [])
    w_feats = wshd_gj.get('features', [])
    print(f'  Downloaded {len(h_feats)} harvest polygons, {len(w_feats)} watershed polygons (public REST, HTTP 200)')

    # Summarize harvest record years & types
    h_years = [f['properties'].get('Year') for f in h_feats if f['properties'].get('Year')]
    h_recent = [f for f in h_feats if (f['properties'].get('Year') or 0) >= 1990]
    h_types = pd.Series([f['properties'].get('Type') or 'Unknown' for f in h_feats]).value_counts().head(8).to_dict()

    # 1. Test harvest overlap on P5 plateau cap pixels (Signal B / guild turnover dataset!)
    p5_px = pd.DataFrame(json.load(open(os.path.join(C.CACHE_DIR, 'p5_plateau_pixels.json'))))
    hit_all_p5 = points_in_geojson_polygons(p5_px['lon'].values, p5_px['lat'].values, h_feats) >= 0
    hit_rec_p5 = points_in_geojson_polygons(p5_px['lon'].values, p5_px['lat'].values, h_recent) >= 0
    p5_px['in_lal_harvest_all'] = hit_all_p5
    p5_px['in_lal_harvest_ge1990'] = hit_rec_p5

    def flat_steep_stats(sub):
        fl = sub[sub['pcls'] == 1]['sos'].dropna().values
        st = sub[sub['pcls'] == 2]['sos'].dropna().values
        t_v, p_v = stats.ttest_ind(fl, st, equal_var=False)
        return {
            'n_total': int(len(sub)), 'n_flat': int(len(fl)), 'n_steep': int(len(st)),
            'flat_mean_sos': round(float(fl.mean()), 3),
            'steep_mean_sos': round(float(st.mean()), 3),
            'diff_flat_minus_steep_d': round(float(fl.mean() - st.mean()), 3),
            'welch_t': round(float(t_v), 3),
            'welch_p': float(f'{p_v:.4g}'),
        }

    p5_unmasked = flat_steep_stats(p5_px)
    p5_no_recent_harv = flat_steep_stats(p5_px[~p5_px['in_lal_harvest_ge1990']])
    p5_no_any_harv = flat_steep_stats(p5_px[~p5_px['in_lal_harvest_all']])

    # 2. Test harvest overlap & watershed assignment on P6 Domain pixels (full Domain elevation gradient!)
    p6_dom = pd.DataFrame(json.load(open(os.path.join(C.CACHE_DIR, 'p6_domain_2017_stack.json'))))
    hit_all_dom = points_in_geojson_polygons(p6_dom['lon'].values, p6_dom['lat'].values, h_feats) >= 0
    hit_rec_dom = points_in_geojson_polygons(p6_dom['lon'].values, p6_dom['lat'].values, h_recent) >= 0
    w_idx_dom = points_in_geojson_polygons(p6_dom['lon'].values, p6_dom['lat'].values, w_feats)
    p6_dom['in_lal_harvest_all'] = hit_all_dom
    p6_dom['in_lal_harvest_ge1990'] = hit_rec_dom
    p6_dom['wshed'] = [w_feats[i]['properties'].get('WshedName') if i >= 0 else None for i in w_idx_dom]

    # Signal A check (Lost Cove 330-385m vs OpenN 330-385m) with vs. without LAL harvests!
    in_lost = (p6_dom['lon'].between(-85.945, -85.880) & p6_dom['lat'].between(35.121, 35.168) &
               p6_dom['elev'].between(330, 385))
    in_openn = (p6_dom['lon'].between(-85.991, -85.859) & p6_dom['lat'].between(35.178, 35.231) &
                p6_dom['elev'].between(330, 385))

    def sig_a_check(mask_keep):
        vl = p6_dom.loc[in_lost & mask_keep, 'sos'].values
        vo = p6_dom.loc[in_openn & mask_keep, 'sos'].values
        return {
            'n_lost': int(len(vl)), 'n_openn': int(len(vo)),
            'lost_mean_sos': round(float(vl.mean()), 3),
            'openn_mean_sos': round(float(vo.mean()), 3),
            'lost_sd_sos': round(float(vl.std(ddof=1)), 3) if len(vl) > 1 else None,
            'openn_sd_sos': round(float(vo.std(ddof=1)), 3) if len(vo) > 1 else None,
            'diff_lost_minus_openn_d': round(float(vl.mean() - vo.mean()), 3),
            'inference_note': 'Descriptive single-cove (N=1) harvest sensitivity; pixel-level p-value omitted due to spatial pseudoreplication.',
        }

    # Hopkins decomposition on p6_dom (exact 2018-2024 S2 SOS across Domain elevation!)
    # Compare: (a) Full Domain, (b) Escarpment core (385-535m, excluding Lost Cove & plateau cap),
    # (c) Unharvested Escarpment core!
    def slope_100m(sub):
        lr = stats.linregress(sub['elev'].values, sub['sos'].values)
        return {
            'n': int(len(sub)),
            'slope_d_per_100m': round(float(lr.slope * 100.0), 3),
            'se_d_per_100m': round(float(lr.stderr * 100.0), 3),
            'r2': round(float(lr.rvalue ** 2), 4),
            'p': float(f'{lr.pvalue:.4g}'),
        }

    escarp_core = p6_dom[(p6_dom['elev'] >= 385) & (p6_dom['elev'] <= 535)]
    escarp_noharv = escarp_core[~escarp_core['in_lal_harvest_all']]
    # Binned profile (15m bins) separating Lost Cove vs Open Escarpment vs Plateau Cap
    bins = []
    for e0 in range(300, 600, 15):
        e1 = e0 + 15
        b_all = p6_dom[(p6_dom['elev'] >= e0) & (p6_dom['elev'] < e1)]
        b_lost = b_all[b_all['lon'].between(-85.945, -85.880) & b_all['lat'].between(35.121, 35.168)]
        b_open = b_all[~(b_all['lon'].between(-85.945, -85.880) & b_all['lat'].between(35.121, 35.168))]
        if len(b_all) >= 8:
            bins.append({
                'elev_mid': e0 + 7.5,
                'all_sos_mean': round(float(b_all['sos'].mean()), 2),
                'all_sos_se': round(float(b_all['sos'].std(ddof=1) / math.sqrt(len(b_all))), 3),
                'all_n': int(len(b_all)),
                'lost_sos_mean': round(float(b_lost['sos'].mean()), 2) if len(b_lost) >= 5 else None,
                'lost_n': int(len(b_lost)),
                'open_sos_mean': round(float(b_open['sos'].mean()), 2) if len(b_open) >= 5 else None,
                'open_n': int(len(b_open)),
            })

    lost_wsheds = p6_dom.loc[in_lost, 'wshed'].value_counts().head(5).to_dict()
    openn_wsheds = p6_dom.loc[in_openn, 'wshed'].value_counts().head(5).to_dict()

    return {
        'access_status': 'PUBLIC_ARCGIS_REST_NO_LOGIN_HTTP_200',
        'harvest_layer_summary': {
            'n_polygons': int(len(h_feats)),
            'n_ge1990': int(len(h_recent)),
            'year_min': int(min(h_years)) if h_years else None,
            'year_max': int(max(h_years)) if h_years else None,
            'top_types': h_types,
        },
        'watershed_layer_summary': {
            'n_polygons': int(len(w_feats)),
            'lost_cove_band_watersheds': lost_wsheds,
            'openn_band_watersheds': openn_wsheds,
        },
        'harvest_overlap_pct': {
            'domain_all_harvests_pct': round(float(hit_all_dom.mean() * 100), 2),
            'domain_ge1990_harvests_pct': round(float(hit_rec_dom.mean() * 100), 2),
            'plateau_p5_all_harvests_pct': round(float(hit_all_p5.mean() * 100), 2),
            'plateau_p5_ge1990_harvests_pct': round(float(hit_rec_p5.mean() * 100), 2),
            'lost_cove_330_385_all_harvests_pct': round(float(p6_dom.loc[in_lost, 'in_lal_harvest_all'].mean() * 100), 2),
            'openn_330_385_all_harvests_pct': round(float(p6_dom.loc[in_openn, 'in_lal_harvest_all'].mean() * 100), 2),
        },
        'signal_A_harvest_sensitivity': {
            'unmasked': sig_a_check(pd.Series(True, index=p6_dom.index)),
            'excluding_ge1990_harvests': sig_a_check(~p6_dom['in_lal_harvest_ge1990']),
            'excluding_all_harvests': sig_a_check(~p6_dom['in_lal_harvest_all']),
        },
        'signal_B_p5_harvest_sensitivity': {
            'unmasked': p5_unmasked,
            'excluding_ge1990_harvests': p5_no_recent_harv,
            'excluding_all_harvests': p5_no_any_harv,
        },
        'hopkins_domain_decomposition': {
            'hopkins_1918_benchmark_d_per_100m': 3.281,
            'full_domain_290_610m': slope_100m(p6_dom),
            'full_domain_unharvested': slope_100m(p6_dom[~p6_dom['in_lal_harvest_all']]),
            'escarpment_core_385_535m': slope_100m(escarp_core),
            'escarpment_core_unharvested': slope_100m(escarp_noharv),
            'elevation_bins_15m': bins,
        },
    }


def main():
    t0 = time.time()
    if not os.path.exists(CACHE_MODIS):
        C.init_ee()
    modis_rows = fetch_clean_modis_2001_2023()
    modis_decomp = fetch_modis_3factor_decomposition()
    mdf = pd.DataFrame(modis_rows)
    legacy_modis = json.load(open(os.path.join(HERE, 'data', 'lapse_trend.json')))
    landsat_diag = diagnose_legacy_landsat_step()
    climate = analyze_noaa_and_era5(modis_rows)
    lal = audit_lal_public_controls()

    out = {
        'experiment': 'v2_4C_hopkins_climate_lal',
        'elapsed_s': round(time.time() - t0, 1),
        'clean_modis_2001_2023': {
            'spec': 'MODIS/061/MCD12Q2 MidGreenup_1 & MidGreendown_1, QA_Overall_1<=1, NLCD-41 deciduous, 3DEP elev',
            'midgreenup_trend': trend_stats(mdf['year'].values, mdf['midgreenup_mean_doy'].values),
            'lapse_rate_trend': trend_stats(mdf['year'].values, mdf['midgreenup_lapse_d_per_100m'].values),
            'midgreendown_trend': trend_stats(mdf['year'].values, mdf['midgreendown_mean_doy'].values),
            'gsl_trend': trend_stats(mdf['year'].values, mdf['gsl_mean_d'].values),
            'per_year': modis_rows,
        },
        'modis_3factor_decomposition': modis_decomp,
        'legacy_modis_comparison': {
            'spec': 'Internal round-1 script 05_lapse_trend.py: Greenup_1 (15%), Hansen TC>50%, no QA',
            'lapse_rate_mean_d_per_100m': legacy_modis['lapse_rate_mean_d_per_100m'],
            'mean_greenup_trend_days_per_decade': legacy_modis['mean_greenup_trend_days_per_decade'],
            'lapse_rate_trend_d_per_100m_per_decade': legacy_modis['lapse_rate_trend_d_per_100m_per_decade'],
        },
        'legacy_landsat_step_artifact': landsat_diag,
        'climate_context': climate,
        'lal_public_controls': lal,
    }
    clean = C.sanitize_json(out)
    with open(OUT_PATH, 'w') as f:
        json.dump(clean, f, indent=2, allow_nan=False)
    print(f'WROTE {OUT_PATH} in {time.time()-t0:.1f}s')

    print('\n=== 4C KEY RESULTS ===')
    print('Clean MODIS MidGreenup trend:', out['clean_modis_2001_2023']['midgreenup_trend'])
    print('Clean MODIS Lapse rate:', out['clean_modis_2001_2023']['lapse_rate_trend'])
    print('Clean MODIS MidGreendown:', out['clean_modis_2001_2023']['midgreendown_trend'])
    print('Legacy Landsat step artifact:', out['legacy_landsat_step_artifact']['step_dummy_regression'])
    print('NOAA 2000-2025 spr_tmean:', climate['noaa_2000_2025_modern_trends']['spr_tmean_C'])
    print('NOAA 2000-2025 aut_tmean:', climate['noaa_2000_2025_modern_trends']['aut_tmean_C'])
    print('NOAA Era shift (Modern vs Hopkins 1895-1943):', json.dumps(climate['noaa_modern_vs_hopkins_era_shift'], indent=2))
    print('MODIS thermal sens:', climate['modis_thermal_sensitivity'])
    print('LAL harvest overlap:', lal['harvest_overlap_pct'])
    print('Signal A harvest sens:', lal['signal_A_harvest_sensitivity'])
    print('Signal B harvest sens:', lal['signal_B_p5_harvest_sensitivity'])
    print('Hopkins core lapse:', lal['hopkins_domain_decomposition']['escarpment_core_385_535m'])


if __name__ == '__main__':
    main()
