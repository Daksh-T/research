"""Experiment P5: Signal B Plateau Mechanism (Pedological Thermal Inertia vs. Oak/Rim Guild Turnover).

Tests why Flat Plateau (>=550m, slope<=8) lags Steep Sandstone Rim (>=550m, slope>10) by +3.6 d
despite identical elevation (+8m), identical winter NDVI, identical ECOSTRESS LST, and confirmed
aspect-quadrant BRDF invariance (P2):
  5A. POLARIS (30m ksat, clay, bd, sand, om, theta_s at surface & fragipan 30-100cm depths) +
      gNATSGO/SSURGO official drainage class, hydrologic group, bedrock depth, and soil series (USDA SDA).
  5B. 25-yr (2000-2024) harmonized Landsat 5-9 annual dSOS_(flat - steep) regressed on NOAA
      Sewanee March-April precipitation (wet-fragipan thermal inertia interaction).
  5C. USFS TreeMap (FORTYPCD, STANDHT, BALIVE, CANOPYPCT), LANDFIRE EVT, and Sentinel-2 October
      mARI anthocyanin pigment index ((1/B3 - 1/B5)*B8) guild-turnover test + variance decomposition.
"""
import os
import sys
import json
import math
import time
import requests
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ee
import common as C

# FIA Forest Type Codes common on Southern Cumberland Plateau
FORTYP_NAMES = {
    401: 'Eastern white pine/N. red oak/white ash',
    403: 'Longleaf/oak', 404: 'Shortleaf pine/oak', 405: 'Virginia pine/S. red oak',
    406: 'Loblolly pine/hardwood', 409: 'Other pine/hardwood',
    501: 'Post oak/blackjack oak', 502: 'Chestnut oak', 503: 'White oak/red oak/hickory',
    504: 'White oak', 505: 'Northern red oak', 506: 'Yellow-poplar/white oak/N. red oak',
    509: 'Bur oak', 511: 'Yellow-poplar', 512: 'Black walnut', 513: 'Black locust',
    514: 'Southern scrub oak', 515: 'Chestnut oak/black oak/scarlet oak',
    516: 'Cherry/white ash/yellow-poplar', 519: 'Red maple/oak', 520: 'Mixed upland hardwoods',
    801: 'Sugar maple/beech/yellow birch', 809: 'Red maple/upland',
}


def sample_plateau_pixels(dom):
    cache_p = os.path.join(C.CACHE_DIR, 'p5_plateau_pixels.json')
    if os.path.exists(cache_p):
        print('[P5] Loaded cached plateau pixel dataset')
        return pd.DataFrame(json.load(open(cache_p)))

    print('[P5] Sampling plateau pixels (>=550m NLCD-41) with SOS, mARI, POLARIS, gNATSGO, TreeMap, EVT...')
    dem = C.get_dem_3dep().rename('elev')
    slope = ee.Terrain.slope(dem).rename('slope')
    aspect = ee.Terrain.aspect(dem).rename('aspect')
    tpi150 = dem.subtract(dem.focalMean(150, 'circle', 'meters')).rename('tpi150')
    nlcd = C.get_nlcd_2021()
    evt = C.get_evt().rename('evt')
    sos = ee.Image(os.environ.get('EE_SOS_ASSET', f"projects/{os.environ.get('EE_PROJECT', '')}/assets/escarpment_sos")).rename('sos')

    # Sentinel-2 seasonal composites (2018-2024, strict SCL=4|5):
    # - Autumn peak pigment window (DOY 285-315) mARI = (1/B3 - 1/B5) * B8 and autumn NDRE/NDVI
    # - Winter NDVI (DOY 1-45) & Summer NDVI (DOY 175-235)
    def s2_prep(img):
        scl = img.select('SCL')
        good = scl.eq(4).Or(scl.eq(5))
        sr = img.select(['B3', 'B4', 'B5', 'B8', 'B8A', 'B11']).multiply(1e-4).updateMask(good)
        ndvi = sr.normalizedDifference(['B8', 'B4']).rename('ndvi')
        ndre = sr.normalizedDifference(['B8A', 'B5']).rename('ndre')
        # mARI = (1/Green - 1/RedEdge1) * NIR  (Gitelson et al. 2006)
        mari = sr.select('B3').max(0.005).pow(-1).subtract(sr.select('B5').max(0.005).pow(-1)).multiply(sr.select('B8')).rename('mari')
        return ee.Image.cat([ndvi, ndre, mari]).copyProperties(img, ['system:time_start'])

    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(dom).filterDate('2018-01-01', '2025-01-01')
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 55)).map(s2_prep))
    oct_comp = s2.filter(ee.Filter.calendarRange(285, 315, 'day_of_year')).median()
    oct_mari = oct_comp.select('mari').rename('oct_mari')
    oct_ndvi = oct_comp.select('ndvi').rename('oct_ndvi')
    oct_ndre = oct_comp.select('ndre').rename('oct_ndre')
    win_ndvi = s2.filter(ee.Filter.calendarRange(1, 50, 'day_of_year')).median().select('ndvi').rename('win_ndvi')
    sum_ndvi = s2.filter(ee.Filter.calendarRange(175, 235, 'day_of_year')).median().select('ndvi').rename('sum_ndvi')
    # Spring mid-greenup window (DOY 105-125) NDRE as independent per-pixel check on SOS
    spr_ndre = s2.filter(ee.Filter.calendarRange(105, 125, 'day_of_year')).median().select('ndre').rename('spr_ndre')

    # POLARIS 30m soil layers (surface 0-5cm, subsoil 30-60cm & 60-100cm fragipan zone)
    def pol(prop, depth_id, out_name):
        return (ee.ImageCollection(f'projects/sat-io/open-datasets/polaris/{prop}_mean')
                .filter(ee.Filter.eq('system:index', f'{prop}_{depth_id}')).first().rename(out_name))

    polaris = ee.Image.cat([
        pol('ksat', '0_5', 'ksat_0_5'),
        pol('ksat', '30_60', 'ksat_30_60'),
        pol('ksat', '60_100', 'ksat_60_100'),
        pol('clay', '0_5', 'clay_0_5'),
        pol('clay', '30_60', 'clay_30_60'),
        pol('clay', '60_100', 'clay_60_100'),
        pol('bd', '30_60', 'bd_30_60'),
        pol('bd', '60_100', 'bd_60_100'),
        pol('sand', '0_5', 'sand_0_5'),
        pol('om', '0_5', 'om_0_5'),
        pol('theta_s', '30_60', 'thetas_30_60'),
    ])

    # gNATSGO mukey
    mukey = ee.ImageCollection('projects/sat-io/open-datasets/gNATSGO/raster/mukey').filterBounds(dom).mosaic().rename('mukey')

    # USFS TreeMap 2016
    tm = ee.ImageCollection('USFS/GTAC/TreeMap/v2016').first().select(
        ['FORTYPCD', 'FLDTYPCD', 'STANDHT', 'BALIVE', 'CANOPYPCT', 'QMD_RMRS']
    )

    # Stratified sample across PlatFlat (>=550m, slope<=8, class=1) and PlatSteep (>=550m, slope>10, class=2)
    plat_m = nlcd.eq(41).And(dem.gte(550)).And(sos.gt(60))
    pcls = ee.Image.constant(0).toInt().where(plat_m.And(slope.lte(8)), 1).where(plat_m.And(slope.gt(10)), 2)
    pcls = pcls.updateMask(pcls.gt(0)).rename('pcls')

    stack = ee.Image.cat([
        pcls, sos, dem, slope, aspect, tpi150, evt.unmask(-9999),
        oct_mari.unmask(-9999), oct_ndvi.unmask(-9999), oct_ndre.unmask(-9999),
        win_ndvi.unmask(-9999), sum_ndvi.unmask(-9999), spr_ndre.unmask(-9999),
        polaris.unmask(-9999), mukey.unmask(-9999), tm.unmask(-9999)
    ])
    fc = stack.stratifiedSample(
        numPoints=700, classBand='pcls', region=dom, scale=30, seed=101, geometries=True, tileScale=4
    )
    feats = fc.getInfo()['features']
    rows = []
    for f in feats:
        p = {k: (np.nan if (isinstance(v, (int, float)) and v <= -9990) else v) for k, v in f['properties'].items()}
        lon, lat = f['geometry']['coordinates']
        p['lon'] = round(lon, 6)
        p['lat'] = round(lat, 6)
        rows.append(p)
    print(f'  Sampled {len(rows)} plateau pixels (flat={sum(r["pcls"]==1 for r in rows)}, steep={sum(r["pcls"]==2 for r in rows)})')
    json.dump(rows, open(cache_p, 'w'))
    return pd.DataFrame(rows)


def attach_ssurgo_attributes(df):
    """Query official USDA NRCS Soil Data Access (SDA) API for every unique mukey in our plateau sample."""
    cache_s = os.path.join(C.CACHE_DIR, 'p5_ssurgo_mukeys.json')
    mukeys = sorted({str(int(m)) for m in df['mukey'].dropna().unique() if m > 0})
    if os.path.exists(cache_s):
        lut = json.load(open(cache_s))
    else:
        print(f'[P5] Querying USDA NRCS SDA for {len(mukeys)} SSURGO mukeys...')
        in_clause = ','.join(f"'{m}'" for m in mukeys)
        q = (f"SELECT m.mukey, m.musym, m.muname, a.drclassdcd, a.hydgrpdcd, a.brockdepmin, a.wtdepannmin "
             f"FROM mapunit m INNER JOIN muaggatt a ON m.mukey = a.mukey WHERE m.mukey IN ({in_clause})")
        r = requests.post('https://SDMDataAccess.sc.egov.usda.gov/Tabular/post.rest',
                          data={'query': q, 'format': 'JSON+COLUMNNAME'}, timeout=30)
        lut = {}
        if r.status_code == 200 and 'Table' in r.json():
            tbl = r.json()['Table']
            hdr = tbl[0]
            for row in tbl[1:]:
                rec = dict(zip(hdr, row))
                lut[str(rec['mukey'])] = rec
        # Also query component table for fragipan horizon presence!
        q2 = (f"SELECT c.mukey, MAX(CASE WHEN ch.desgnmaster LIKE '%x%' OR ch.hzname LIKE '%x%' OR "
              f"c.taxsubgrp LIKE '%Fragi%' THEN 1 ELSE 0 END) AS has_fragipan, "
              f"MAX(c.compname) AS compname "
              f"FROM component c LEFT JOIN chorizon ch ON c.cokey = ch.cokey "
              f"WHERE c.mukey IN ({in_clause}) AND c.majcompflag = 'Yes' GROUP BY c.mukey")
        r2 = requests.post('https://SDMDataAccess.sc.egov.usda.gov/Tabular/post.rest',
                           data={'query': q2, 'format': 'JSON+COLUMNNAME'}, timeout=30)
        if r2.status_code == 200 and 'Table' in r2.json():
            tbl2 = r2.json()['Table']
            hdr2 = tbl2[0]
            for row in tbl2[1:]:
                rec2 = dict(zip(hdr2, row))
                mk = str(rec2['mukey'])
                if mk in lut:
                    lut[mk]['has_fragipan'] = int(rec2.get('has_fragipan') or 0)
                    lut[mk]['compname'] = rec2.get('compname')
        json.dump(lut, open(cache_s, 'w'), indent=2)
        print(f'  Retrieved SSURGO attributes for {len(lut)}/{len(mukeys)} mukeys')

    df = df.copy()
    df['mukey_str'] = df['mukey'].apply(lambda x: str(int(x)) if pd.notna(x) and x > 0 else '')
    df['muname'] = df['mukey_str'].map(lambda k: lut.get(k, {}).get('muname', 'Unknown'))
    df['compname'] = df['mukey_str'].map(lambda k: lut.get(k, {}).get('compname', 'Unknown'))
    df['drclass'] = df['mukey_str'].map(lambda k: lut.get(k, {}).get('drclassdcd', 'Unknown'))
    df['hydgrp'] = df['mukey_str'].map(lambda k: lut.get(k, {}).get('hydgrpdcd', 'Unknown'))
    df['brockdepmin'] = pd.to_numeric(df['mukey_str'].map(lambda k: lut.get(k, {}).get('brockdepmin')), errors='coerce')
    df['wtdepannmin'] = pd.to_numeric(df['mukey_str'].map(lambda k: lut.get(k, {}).get('wtdepannmin')), errors='coerce')
    df['has_fragipan'] = pd.to_numeric(df['mukey_str'].map(lambda k: lut.get(k, {}).get('has_fragipan', 0)), errors='coerce').fillna(0).astype(int)
    return df, lut


def run_25yr_precip_interaction(dom):
    """Test 5B: 25-yr (2000-2024) Landsat dSOS_(flat - steep) vs NOAA Sewanee Mar-Apr precipitation."""
    cache_p = os.path.join(C.CACHE_DIR, 'p5_landsat_flat_steep_25yr.json')
    if os.path.exists(cache_p):
        df_ts = pd.DataFrame(json.load(open(cache_p)))
        print('[P5] Loaded cached 25-yr PlatFlat vs PlatSteep Landsat time series')
    else:
        print('[P5] Extracting 25-yr (2000-2024) Landsat 5-9 NDVI for PlatFlat vs PlatSteep...')
        dem = C.get_dem_3dep()
        slope = ee.Terrain.slope(dem)
        nlcd = C.get_nlcd_2021()
        plat = nlcd.eq(41).And(dem.gte(550))
        flat_m = plat.And(slope.lte(8))
        steep_m = plat.And(slope.gt(10))

        def mask_l57(img):
            qa = img.select('QA_PIXEL')
            clear = qa.bitwiseAnd((1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)).eq(0)
            r = img.select('SR_B3').multiply(2.75e-5).add(-0.2)
            n = img.select('SR_B4').multiply(2.75e-5).add(-0.2)
            ndvi = n.subtract(r).divide(n.add(r).max(1e-4)).multiply(0.9723).add(0.0235).rename('ndvi')
            return ndvi.updateMask(clear).updateMask(ndvi.gt(0.05).And(ndvi.lt(1.0))).set(
                'doy', img.date().getRelative('day', 'year'), 'yr', img.date().get('year')
            )

        def mask_l89(img):
            qa = img.select('QA_PIXEL')
            clear = qa.bitwiseAnd((1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)).eq(0)
            r = img.select('SR_B4').multiply(2.75e-5).add(-0.2)
            n = img.select('SR_B5').multiply(2.75e-5).add(-0.2)
            ndvi = n.subtract(r).divide(n.add(r).max(1e-4)).rename('ndvi')
            return ndvi.updateMask(clear).updateMask(ndvi.gt(0.05).And(ndvi.lt(1.0))).set(
                'doy', img.date().getRelative('day', 'year'), 'yr', img.date().get('year')
            )

        rows = []
        for y0, y1 in [(2000, 2006), (2007, 2012), (2013, 2018), (2019, 2024)]:
            d0, d1 = f'{y0}-01-01', f'{y1+1}-01-01'
            cols = []
            if y0 <= 2011:
                cols.append(ee.ImageCollection('LANDSAT/LT05/C02/T1_L2').filterBounds(dom).filterDate(d0, d1)
                            .filter(ee.Filter.lt('CLOUD_COVER', 65)).map(mask_l57))
            if y0 <= 2021:
                cols.append(ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').filterBounds(dom).filterDate(d0, d1)
                            .filter(ee.Filter.lt('CLOUD_COVER', 65)).map(mask_l57))
            if y1 >= 2013:
                cols.append(ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(dom).filterDate(d0, d1)
                            .filter(ee.Filter.lt('CLOUD_COVER', 65)).map(mask_l89))
            if y1 >= 2021:
                cols.append(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').filterBounds(dom).filterDate(d0, d1)
                            .filter(ee.Filter.lt('CLOUD_COVER', 65)).map(mask_l89))
            m = cols[0]
            for c in cols[1:]:
                m = m.merge(c)

            def red(img):
                st = ee.Image.cat([
                    img.updateMask(flat_m).rename('flat_ndvi'),
                    img.updateMask(steep_m).rename('steep_ndvi'),
                ])
                d = st.reduceRegion(ee.Reducer.mean().combine(ee.Reducer.count(), '', True),
                                    dom, 30, tileScale=4, maxPixels=1e6)
                return ee.Feature(None, d).set('doy', img.get('doy'), 'yr', img.get('yr'))

            feats = m.map(red).filter(
                ee.Filter.And(ee.Filter.gte('flat_ndvi_count', 50), ee.Filter.gte('steep_ndvi_count', 30))
            ).getInfo()['features']
            for f in feats:
                p = f['properties']
                rows.append({
                    'yr': int(p['yr']), 'doy': int(p['doy']),
                    'flat_ndvi': p.get('flat_ndvi_mean'), 'steep_ndvi': p.get('steep_ndvi_mean')
                })
        df_ts = pd.DataFrame(rows).groupby(['yr', 'doy'], as_index=False).median()
        json.dump(df_ts.to_dict(orient='records'), open(cache_p, 'w'))

    noaa = C.load_noaa_sewanee()
    era5 = C.fetch_era5_daily(2000, 2024)
    era5['DATE'] = pd.to_datetime(era5['date'])
    # Right-join onto complete ERA5 2000-2024 dates so missing NOAA days (e.g. Jan-May 2000) fall back to ERA5!
    wx_m = pd.merge(noaa[noaa['year'].between(2000, 2024)][['DATE', 'TMIN', 'PRCP']],
                    era5[['DATE', 't2m_min_C', 'precip_mm']], on='DATE', how='right')
    wx_m['year'] = wx_m['DATE'].dt.year
    wx_m['doy'] = wx_m['DATE'].dt.dayofyear
    wx_m['tmin'] = wx_m['TMIN'].fillna(wx_m['t2m_min_C'])
    wx_m['prcp'] = wx_m['PRCP'].fillna(wx_m['precip_mm'])

    wx_rows = []
    for yr in range(2000, 2025):
        g = wx_m[wx_m['year'] == yr]
        wx_rows.append({
            'yr': yr,
            'mar_apr_precip_mm': round(float(g[g['doy'].between(60, 120)]['prcp'].sum()), 1),
            'feb_apr_precip_mm': round(float(g[g['doy'].between(32, 120)]['prcp'].sum()), 1),
            'mar_apr_tmin_C': round(float(g[g['doy'].between(60, 120)]['tmin'].mean()), 2),
        })
    df_wx = pd.DataFrame(wx_rows).set_index('yr')

    yr_fits = []
    for yr, grp in df_ts.groupby('yr'):
        rf = C.fit_double_logistic(grp['doy'].values, grp['flat_ndvi'].values, min_obs=6, min_spring=2, min_autumn=2)
        rs = C.fit_double_logistic(grp['doy'].values, grp['steep_ndvi'].values, min_obs=6, min_spring=2, min_autumn=2)
        if np.isfinite(rf['r2']) and rf['r2'] >= 0.80 and np.isfinite(rs['r2']) and rs['r2'] >= 0.80:
            yr_fits.append({
                'yr': int(yr),
                'flat_sos': rf['sos50'], 'steep_sos': rs['sos50'],
                'dSOS_flat_minus_steep': round(rf['sos50'] - rs['sos50'], 3),
                'dEOS_flat_minus_steep': round(rf['eos50'] - rs['eos50'], 3),
                **df_wx.loc[int(yr)].to_dict(),
            })
    dfd = pd.DataFrame(yr_fits)
    r_ma, p_ma = stats.pearsonr(dfd['mar_apr_precip_mm'], dfd['dSOS_flat_minus_steep'])
    r_fa, p_fa = stats.pearsonr(dfd['feb_apr_precip_mm'], dfd['dSOS_flat_minus_steep'])
    r_tm, p_tm = stats.pearsonr(dfd['mar_apr_tmin_C'], dfd['dSOS_flat_minus_steep'])
    t_m, p_m = stats.ttest_1samp(dfd['dSOS_flat_minus_steep'], 0.0)
    return {
        'n_years': int(len(dfd)),
        'mean_dSOS_flat_minus_steep': round(float(dfd['dSOS_flat_minus_steep'].mean()), 3),
        'sd_dSOS_flat_minus_steep': round(float(dfd['dSOS_flat_minus_steep'].std()), 3),
        'frac_yrs_gt0': round(float((dfd['dSOS_flat_minus_steep'] > 0).mean()), 3),
        'onesamp_p': float(f'{p_m:.4g}'),
        'r_mar_apr_precip': round(float(r_ma), 3), 'p_mar_apr_precip': float(f'{p_ma:.4g}'),
        'r_feb_apr_precip': round(float(r_fa), 3), 'p_feb_apr_precip': float(f'{p_fa:.4g}'),
        'r_mar_apr_tmin': round(float(r_tm), 3), 'p_mar_apr_tmin': float(f'{p_tm:.4g}'),
        'by_year': yr_fits,
    }


def main():
    t0 = time.time()
    C.init_ee()
    dom = C.get_domain()

    df = sample_plateau_pixels(dom)
    df, ssurgo_lut = attach_ssurgo_attributes(df)
    df['is_flat'] = (df['pcls'] == 1).astype(int)

    flat = df[df['is_flat'] == 1]
    steep = df[df['is_flat'] == 0]

    # Overall raw Signal B on this sample
    raw_diff = float(flat['sos'].mean() - steep['sos'].mean())
    t_raw, p_raw = stats.ttest_ind(flat['sos'], steep['sos'], equal_var=False)

    # --- Test 5A: Soils / Pedological Thermal Inertia ---
    # 1. Within FLAT (slope <= 8) ONLY: wet/fragipan vs well-drained
    flat_frag = flat[(flat['has_fragipan'] == 1) | (~flat['drclass'].isin(['Well drained', 'Somewhat excessively drained', 'Excessively drained']))]
    flat_well = flat[(flat['has_fragipan'] == 0) & (flat['drclass'].isin(['Well drained', 'Somewhat excessively drained', 'Excessively drained']))]
    frag_vs_well_flat = None
    if len(flat_frag) >= 15 and len(flat_well) >= 15:
        tf, pf = stats.ttest_ind(flat_frag['sos'], flat_well['sos'], equal_var=False)
        frag_vs_well_flat = {
            'n_wet_frag': int(len(flat_frag)), 'n_well_drained': int(len(flat_well)),
            'mean_sos_wet_frag': round(float(flat_frag['sos'].mean()), 3),
            'mean_sos_well_drained': round(float(flat_well['sos'].mean()), 3),
            'diff_wet_minus_well_d': round(float(flat_frag['sos'].mean() - flat_well['sos'].mean()), 3),
            'p': float(f'{pf:.4g}'),
        }

    # 2. Within FLAT ONLY: POLARIS subsoil ksat_30_60 (bottom Q1 vs top Q4) & bd_60_100 (top Q4 vs bottom Q1)
    q25_k, q75_k = flat['ksat_30_60'].quantile(0.25), flat['ksat_30_60'].quantile(0.75)
    f_lo_k = flat[flat['ksat_30_60'] <= q25_k]['sos']
    f_hi_k = flat[flat['ksat_30_60'] >= q75_k]['sos']
    _, pk = stats.ttest_ind(f_lo_k, f_hi_k, equal_var=False)
    r_ksat_flat, p_ksat_flat = stats.pearsonr(flat['ksat_30_60'].dropna(), flat.loc[flat['ksat_30_60'].notna(), 'sos'])
    r_bd_flat, p_bd_flat = stats.pearsonr(flat['bd_60_100'].dropna(), flat.loc[flat['bd_60_100'].notna(), 'sos'])
    r_clay_flat, p_clay_flat = stats.pearsonr(flat['clay_30_60'].dropna(), flat.loc[flat['clay_30_60'].notna(), 'sos'])

    # 3. Holding WELL-DRAINED SSURGO soils constant: Flat vs Steep!
    steep_well = steep[(steep['has_fragipan'] == 0) & (steep['drclass'].isin(['Well drained', 'Somewhat excessively drained', 'Excessively drained']))]
    _, p_wd = stats.ttest_ind(flat_well['sos'], steep_well['sos'], equal_var=False) if (len(flat_well) >= 15 and len(steep_well) >= 15) else (np.nan, np.nan)
    flat_vs_steep_well_drained = {
        'n_flat_well': int(len(flat_well)), 'n_steep_well': int(len(steep_well)),
        'flat_well_sos': round(float(flat_well['sos'].mean()), 3),
        'steep_well_sos': round(float(steep_well['sos'].mean()), 3),
        'diff_d': round(float(flat_well['sos'].mean() - steep_well['sos'].mean()), 3),
        'p': float(f'{p_wd:.4g}') if np.isfinite(p_wd) else None,
    }

    # Breakdown by SSURGO drainage class and top soil series
    dr_table = df.groupby(['drclass', 'is_flat']).agg(n=('sos', 'count'), mean_sos=('sos', 'mean'), mean_elev=('elev', 'mean'), mean_slope=('slope', 'mean')).round(2).reset_index().to_dict(orient='records')
    series_table = df.groupby(['compname', 'is_flat']).agg(n=('sos', 'count'), mean_sos=('sos', 'mean'), has_frag=('has_fragipan', 'max')).round(2).reset_index()
    series_table = series_table[series_table['n'] >= 15].sort_values(['compname', 'is_flat']).to_dict(orient='records')

    # --- Test 5C: Guild Turnover (TreeMap FORTYPCD, STANDHT, BALIVE, EVT, mARI) ---
    guild_compare = {}
    for col in ['oct_mari', 'oct_ndvi', 'oct_ndre', 'spr_ndre', 'win_ndvi', 'sum_ndvi',
                'STANDHT', 'BALIVE', 'CANOPYPCT', 'QMD_RMRS', 'ksat_30_60', 'clay_30_60', 'bd_60_100', 'tpi150']:
        a = flat[col].dropna()
        b = steep[col].dropna()
        if len(a) >= 10 and len(b) >= 10:
            tv, pv = stats.ttest_ind(a, b, equal_var=False)
            guild_compare[col] = {
                'flat_mean': round(float(a.mean()), 4), 'steep_mean': round(float(b.mean()), 4),
                'diff': round(float(a.mean() - b.mean()), 4), 'p': float(f'{pv:.4g}')
            }

    # FORTYPCD distribution & within-FORTYPCD flat vs steep jump!
    ft_dist = []
    for ft, grp in df.groupby('FORTYPCD'):
        f1 = grp[grp['is_flat'] == 1]
        f0 = grp[grp['is_flat'] == 0]
        if len(grp) >= 20:
            diff_ft = float(f1['sos'].mean() - f0['sos'].mean()) if (len(f1) >= 10 and len(f0) >= 10) else None
            p_ft = float(stats.ttest_ind(f1['sos'], f0['sos'], equal_var=False).pvalue) if diff_ft is not None else None
            ft_dist.append({
                'FORTYPCD': int(ft), 'name': FORTYP_NAMES.get(int(ft), f'Code_{int(ft)}'),
                'n_flat': int(len(f1)), 'n_steep': int(len(f0)),
                'pct_of_flat': round(100.0 * len(f1) / len(flat), 1),
                'pct_of_steep': round(100.0 * len(f0) / len(steep), 1),
                'flat_sos': round(float(f1['sos'].mean()), 2) if len(f1) >= 5 else None,
                'steep_sos': round(float(f0['sos'].mean()), 2) if len(f0) >= 5 else None,
                'diff_d': round(diff_ft, 2) if diff_ft is not None else None,
                'p': float(f'{p_ft:.4g}') if p_ft is not None else None,
            })

    # Nested OLS decomposition: how much of beta(is_flat) survives controlling for elev+TPI, soils, TreeMap+EVT, mARI?
    reg_df = df.dropna(subset=['sos', 'is_flat', 'elev', 'tpi150', 'ksat_30_60', 'clay_30_60', 'bd_60_100',
                               'has_fragipan', 'STANDHT', 'BALIVE', 'CANOPYPCT', 'oct_mari']).copy()
    # One-hot top FORTYPCD & EVT
    top_ft = reg_df['FORTYPCD'].value_counts().head(5).index.tolist()
    ft_dum = np.column_stack([(reg_df['FORTYPCD'] == c).astype(float).values for c in top_ft[1:]])
    top_evt = reg_df['evt'].value_counts().head(4).index.tolist()
    evt_dum = np.column_stack([(reg_df['evt'] == c).astype(float).values for c in top_evt[1:]])

    y_sos = reg_df['sos'].values
    m1 = C.ols_cluster_se(reg_df[['is_flat']].values, y_sos, col_names=['is_flat'])
    m2 = C.ols_cluster_se(reg_df[['is_flat', 'elev', 'tpi150']].values, y_sos, col_names=['is_flat', 'elev', 'tpi150'])
    m3_soil = C.ols_cluster_se(
        reg_df[['is_flat', 'elev', 'tpi150', 'ksat_30_60', 'clay_30_60', 'bd_60_100', 'has_fragipan']].values,
        y_sos, col_names=['is_flat', 'elev', 'tpi150', 'ksat_30_60', 'clay_30_60', 'bd_60_100', 'has_fragipan']
    )
    X_guild = np.column_stack([reg_df[['is_flat', 'elev', 'tpi150', 'STANDHT', 'BALIVE', 'CANOPYPCT', 'oct_mari']].values, ft_dum, evt_dum])
    m4_guild = C.ols_cluster_se(X_guild, y_sos, col_names=['is_flat', 'elev', 'tpi150', 'STANDHT', 'BALIVE', 'CANOPYPCT', 'oct_mari'] + [f'ft_{c}' for c in top_ft[1:]] + [f'evt_{c}' for c in top_evt[1:]])
    X_full = np.column_stack([reg_df[['is_flat', 'elev', 'tpi150', 'ksat_30_60', 'clay_30_60', 'bd_60_100', 'has_fragipan', 'STANDHT', 'BALIVE', 'CANOPYPCT', 'oct_mari']].values, ft_dum, evt_dum])
    m5_full = C.ols_cluster_se(X_full, y_sos, col_names=['is_flat', 'elev', 'tpi150', 'ksat_30_60', 'clay_30_60', 'bd_60_100', 'has_fragipan', 'STANDHT', 'BALIVE', 'CANOPYPCT', 'oct_mari'] + [f'ft_{c}' for c in top_ft[1:]] + [f'evt_{c}' for c in top_evt[1:]])

    # Test 5B: 25-yr Mar-Apr precip interaction
    precip_25yr = run_25yr_precip_interaction(dom)

    # Evaluate pre-registered P5 gates
    soil_pass = (
        frag_vs_well_flat is not None
        and frag_vs_well_flat['diff_wet_minus_well_d'] >= 2.0
        and precip_25yr['r_mar_apr_precip'] >= 0.50
    )
    guild_pass = (
        flat_vs_steep_well_drained['diff_d'] >= 2.5
        and abs(guild_compare.get('oct_mari', {}).get('diff', 0)) > 0
        and guild_compare.get('oct_mari', {}).get('p', 1.0) < 0.01
    )

    out = {
        'experiment': 'P5_signalb_mechanism',
        'elapsed_s': round(time.time() - t0, 1),
        'n_pixels': int(len(df)),
        'raw_flat_minus_steep_d': round(raw_diff, 3),
        'raw_p': float(f'{p_raw:.4g}'),
        'test_5a_soils': {
            'fragipan_vs_welldrained_within_flat': frag_vs_well_flat,
            'polaris_within_flat': {
                'low_ksat_minus_high_ksat_d': round(float(f_lo_k.mean() - f_hi_k.mean()), 3),
                'low_ksat_vs_high_ksat_p': float(f'{pk:.4g}'),
                'r_ksat30_60_sos': round(float(r_ksat_flat), 3), 'p_ksat': float(f'{p_ksat_flat:.4g}'),
                'r_bd60_100_sos': round(float(r_bd_flat), 3), 'p_bd': float(f'{p_bd_flat:.4g}'),
                'r_clay30_60_sos': round(float(r_clay_flat), 3), 'p_clay': float(f'{p_clay_flat:.4g}'),
            },
            'flat_vs_steep_holding_well_drained': flat_vs_steep_well_drained,
            'drainage_class_table': dr_table,
            'soil_series_table': series_table,
        },
        'test_5b_25yr_precip_interaction': precip_25yr,
        'test_5c_guild_and_pigment': {
            'flat_vs_steep_features': guild_compare,
            'fortypcd_breakdown': ft_dist,
            'nested_ols_beta_is_flat': {
                'm1_unadjusted': {'beta': m1['coefs']['is_flat']['beta'], 'r2': m1['r2']},
                'm2_elev_tpi': {'beta': m2['coefs']['is_flat']['beta'], 'r2': m2['r2']},
                'm3_plus_soils': {'beta': m3_soil['coefs']['is_flat']['beta'], 'r2': m3_soil['r2']},
                'm4_plus_guild_mari': {'beta': m4_guild['coefs']['is_flat']['beta'], 'r2': m4_guild['r2'], 'oct_mari_coef': m4_guild['coefs']['oct_mari']},
                'm5_full': {'beta': m5_full['coefs']['is_flat']['beta'], 'r2': m5_full['r2']},
            }
        },
        'gate_check': {
            'fragipan_within_flat_diff_d': frag_vs_well_flat['diff_wet_minus_well_d'] if frag_vs_well_flat else None,
            'r_25yr_mar_apr_precip': precip_25yr['r_mar_apr_precip'],
            'flat_vs_steep_well_drained_d': flat_vs_steep_well_drained['diff_d'],
            'oct_mari_diff': guild_compare.get('oct_mari', {}).get('diff'),
            'oct_mari_p': guild_compare.get('oct_mari', {}).get('p'),
            'beta_is_flat_unadj': m1['coefs']['is_flat']['beta'],
            'beta_is_flat_full_controls': m5_full['coefs']['is_flat']['beta'],
            'soil_inertia_verdict': 'PASS' if soil_pass else 'FAIL',
            'guild_turnover_verdict': 'PASS' if guild_pass else 'PARTIAL_OR_FAIL',
        }
    }
    C.save_json('p5.json', out)
    print('\n=== P5 SUMMARY ===')
    print(json.dumps(out['gate_check'], indent=2))
    print('Nested OLS beta(is_flat):', json.dumps(out['test_5c_guild_and_pigment']['nested_ols_beta_is_flat'], indent=2))


if __name__ == '__main__':
    main()
