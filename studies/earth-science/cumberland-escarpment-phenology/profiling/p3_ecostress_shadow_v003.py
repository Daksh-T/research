"""Experiment P3: ECOSTRESS 10m 3DEP Ray-Traced Shadow Audit + Live ECO_L2T_LSTE.003 (Spring & Autumn).

Tests:
(a) 10m 3DEP ray-traced solar cast shadow & post-sunrise cumulative direct insolation (I_cum)
    on all cached 04-11h ECOSTRESS scenes (data/ecostress/), testing whether the 09-10h -1.80 C
    anomaly is eastern-rim cast shadow vs. true Whiteman convective breakup lag.
(b) Synoptic ERA5-Land hourly wind speed (calm-clear vs. windy placebo) & Manski MNAR censoring
    bounds on pre-dawn (02-06h) cached scenes.
(c) Live NASA LPCLOUD ECO_L2T_LSTE.003 (C3998139651-LPCLOUD) retrieval of LST, QC, cloud, EmisWB,
    and view_zenith for BOTH Spring (Feb-Apr) and Leaf-On Autumn (Sep-Nov, esp. Oct) across
    Lost Cove vs. OpenN (330-385m NLCD-41 + FillDepth > 40m).
"""
import os
import sys
import re
import glob
import json
import math
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ee
import common as C

LON0 = -85.92
LAT0 = 35.15
DOM_BUF_BBOX = [-86.02, 35.09, -85.83, 35.26]


def load_domain_10m_dem_nlcd():
    """Download & cache 10m 3DEP DEM + NLCD 2021 over Sewanee Domain + 3km buffer for exact ray tracing."""
    cache_tif = os.path.join(C.CACHE_DIR, 'sewanee_3dep_nlcd_10m_buf.tif')
    if not os.path.exists(cache_tif):
        print('[P3] Downloading 10m 3DEP DEM + NLCD 2021 over Sewanee + 3km buffer...')
        region = ee.Geometry.Rectangle(DOM_BUF_BBOX)
        dem = ee.Image('USGS/3DEP/10m').select('elevation')
        nlcd = C.get_nlcd_2021()
        url = dem.addBands(nlcd).getDownloadURL({
            'region': region, 'scale': 10, 'crs': 'EPSG:32616', 'format': 'GEO_TIFF'
        })
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        with open(cache_tif, 'wb') as f:
            f.write(r.content)
        print(f'  cached {cache_tif} ({len(r.content)/1e6:.2f} MB)')
    with rasterio.open(cache_tif) as ds:
        dem = ds.read(1).astype(np.float32)
        nlcd = ds.read(2).astype(np.int16)
        prof = ds.profile
        tf = ds.transform
        crs = ds.crs
    dem[dem < -100] = np.nan
    return dem, nlcd, prof, tf, crs


def solar_position(doy, utc_hr, lat_deg=LAT0, lon_deg=LON0):
    """NOAA/Spencer solar altitude & azimuth (degrees, az clockwise from North)."""
    gamma = 2.0 * math.pi / 365.0 * (doy - 1 + (utc_hr - 12.0) / 24.0)
    eqtime = 229.18 * (
        0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma) - 0.040849 * math.sin(2 * gamma)
    )
    decl = (
        0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma)
    )
    tst_min = (utc_hr * 60.0 + eqtime + 4.0 * lon_deg) % 1440.0
    ha = math.radians(tst_min / 4.0 - 180.0)
    lat = math.radians(lat_deg)
    sin_alt = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(ha)
    sin_alt = max(-1.0, min(1.0, sin_alt))
    alt = math.asin(sin_alt)
    cos_az = (math.sin(decl) - math.sin(alt) * math.sin(lat)) / max(1e-8, math.cos(alt) * math.cos(lat))
    cos_az = max(-1.0, min(1.0, cos_az))
    az = math.acos(cos_az)
    if ha > 0:
        az = 2.0 * math.pi - az
    return math.degrees(alt), math.degrees(az), tst_min / 60.0


def trace_cast_shadow(dem, px_m, alt_deg, az_deg, max_dist_m=3000.0, step_m=20.0):
    """Exact 10m 3DEP ray-traced cast shadow (True = in topographic shadow)."""
    if alt_deg <= 0.0:
        return np.ones(dem.shape, dtype=bool)
    H, W = dem.shape
    tan_alt = math.tan(math.radians(alt_deg))
    # Direction FROM pixel TOWARD sun in (row, col): North is -row, East is +col
    az = math.radians(az_deg)
    dr_unit = -math.cos(az) / px_m
    dc_unit = math.sin(az) / px_m

    shadow = np.zeros((H, W), dtype=bool)
    n_steps = int(max_dist_m / step_m)
    rr0 = np.arange(H)
    cc0 = np.arange(W)
    for s in range(1, n_steps + 1):
        dist = s * step_m
        dr = int(round(dr_unit * dist))
        dc = int(round(dc_unit * dist))
        if dr == 0 and dc == 0:
            continue
        r_src0 = max(0, -dr)
        r_src1 = min(H, H - dr)
        c_src0 = max(0, -dc)
        c_src1 = min(W, W - dc)
        if r_src1 <= r_src0 or c_src1 <= c_src0:
            break
        z_here = dem[r_src0:r_src1, c_src0:c_src1]
        z_there = dem[r_src0 + dr:r_src1 + dr, c_src0 + dc:c_src1 + dc]
        blocked = z_there > (z_here + dist * tan_alt)
        shadow[r_src0:r_src1, c_src0:c_src1] |= blocked
    return shadow


def compute_slope_aspect(dem, px_m):
    dz_dy, dz_dx = np.gradient(dem, px_m)
    # y increases downward (South), so uphill in (East, North) is (dz_dx, -dz_dy),
    # and downhill aspect vector in (East, North) is (-dz_dx, +dz_dy).
    # Clockwise azimuth from North = atan2(East, North) = atan2(-dz_dx, dz_dy) mod 2*pi.
    slope = np.arctan(np.hypot(dz_dx, dz_dy))
    aspect = np.arctan2(-dz_dx, dz_dy) % (2.0 * math.pi)
    return slope, aspect


def compute_cum_insolation(dem, slope, aspect, px_m, doy, utc_hr, step_min=20):
    """Integrate clear-sky direct beam cos(i)*I(sunlit) from sunrise to overpass utc_hr (arbitrary MJ-equiv)."""
    # Find approximate sunrise in UTC (roughly 10.5 to 13.5 UTC at -85.92)
    ic = np.zeros(dem.shape, dtype=np.float32)
    t_start = max(10.0, utc_hr - 6.0)
    t = t_start
    dt_hr = step_min / 60.0
    cos_a = np.cos(slope)
    sin_a = np.sin(slope)
    while t <= utc_hr + 1e-6:
        alt_d, az_d, _ = solar_position(doy, t)
        if alt_d > 1.0:
            sh = trace_cast_shadow(dem, px_m, alt_d, az_d, max_dist_m=2800.0, step_m=30.0)
            sza = math.radians(90.0 - alt_d)
            saa = math.radians(az_d)
            cos_i = math.cos(sza) * cos_a + math.sin(sza) * sin_a * np.cos(saa - aspect)
            # Beer-Lambert clear-sky direct beam ~ 1.0 * 0.7^(1/sin(alt))
            airmass = 1.0 / max(0.05, math.sin(math.radians(alt_d)))
            beam = (0.75 ** min(airmass, 10.0)) * np.maximum(0.0, cos_i) * (~sh)
            ic += (beam * dt_hr).astype(np.float32)
        t += dt_hr
    return ic


def build_10m_masks(dem, nlcd, tf, crs):
    """Build exact Lost Cove vs OpenN (330-385m NLCD-41) + exact 10m FillDepth masks on 10m UTM grid."""
    H, W = dem.shape
    # Exact 10m Priority-Flood (no 20m decimation so narrow 10m saddles are preserved)
    _, fill_depth = C.priority_flood_fill(dem)

    # Convert pixel grid to lon/lat bounds using rasterio.warp.transform
    from rasterio.warp import transform as warp_transform
    rows_g = np.arange(0, H, 10)
    cols_g = np.arange(0, W, 10)
    xs = tf.c + (np.arange(W) + 0.5) * tf.a
    ys = tf.f + (np.arange(H) + 0.5) * tf.e
    # Fast linear lat/lon approx on small 20km UTM16 tile (verified exact to <0.0002 deg)
    c_lon, c_lat = warp_transform(crs, 'EPSG:4326', [xs[0], xs[-1], xs[0]], [ys[0], ys[0], ys[-1]])
    lon_1d = np.linspace(c_lon[0], c_lon[1], W)
    lat_1d = np.linspace(c_lat[0], c_lat[2], H)
    lon_2d = np.broadcast_to(lon_1d[None, :], (H, W))
    lat_2d = np.broadcast_to(lat_1d[:, None], (H, W))

    decid = (nlcd == 41)
    band_a = decid & np.isfinite(dem) & (dem >= 330.0) & (dem <= 385.0)
    in_lost_box = (lon_2d >= -85.945) & (lon_2d <= -85.880) & (lat_2d >= 35.121) & (lat_2d <= 35.168)
    in_openn_box = (lon_2d >= -85.991) & (lon_2d <= -85.859) & (lat_2d >= 35.178) & (lat_2d <= 35.231)

    m_lost = band_a & in_lost_box
    m_openn = band_a & in_openn_box
    m_lost_deep = decid & in_lost_box & (fill_depth >= 30.0)
    m_lost_fd40 = decid & in_lost_box & (fill_depth >= 40.0)
    m_plateau = decid & np.isfinite(dem) & (dem >= 550.0) & (lat_2d >= 35.121) & (lat_2d <= 35.231) & (lon_2d >= -85.991) & (lon_2d <= -85.859)

    print(f'  10m masks: Lost(330-385)={m_lost.sum():,} | OpenN(330-385)={m_openn.sum():,} | '
          f'Lost(FD>=30)={m_lost_deep.sum():,} | Lost(FD>=40)={m_lost_fd40.sum():,} | '
          f'max Lost FD={float(np.nanmax(fill_depth[in_lost_box])):.1f}m')
    return {
        'lost': m_lost, 'openn': m_openn, 'lost_fd30': m_lost_deep, 'lost_fd40': m_lost_fd40,
        'plateau': m_plateau, 'fill_depth': fill_depth, 'max_lost_fd_10m': float(np.nanmax(fill_depth[in_lost_box]))
    }


def fetch_era5_hourly_for_timestamps(ts_list):
    """Fetch ERA5-Land hourly 10m wind speed & 2m temp at Sewanee for specific UTC timestamps."""
    cache_p = os.path.join(C.CACHE_DIR, 'p3_era5_hourly_lookup.json')
    lut = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    missing = [t for t in ts_list if t not in lut]
    if missing:
        print(f'[P3] Querying ERA5-Land hourly meteorology for {len(missing)} ECOSTRESS overpass times...')
        pt = ee.Geometry.Point([LON0, LAT0])
        col = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY').select([
            'u_component_of_wind_10m', 'v_component_of_wind_10m', 'temperature_2m'
        ])
        # Build FeatureCollection of timestamps and map lookup in batches of 40
        for i in range(0, len(missing), 40):
            chunk = missing[i:i + 40]
            fc = ee.FeatureCollection([ee.Feature(None, {'ts': t}) for t in chunk])

            def _get(f):
                ts = ee.String(f.get('ts'))
                dt = ee.Date(ts)
                im = col.filterDate(dt.advance(-35, 'minute'), dt.advance(35, 'minute')).first()
                d = ee.Algorithms.If(
                    im,
                    ee.Image(im).reduceRegion(ee.Reducer.first(), pt, 10000),
                    ee.Dictionary({'u_component_of_wind_10m': None, 'v_component_of_wind_10m': None, 'temperature_2m': None})
                )
                return f.set(ee.Dictionary(d))

            feats = fc.map(_get).getInfo()['features']
            for f in feats:
                p = f['properties']
                u = p.get('u_component_of_wind_10m')
                v = p.get('v_component_of_wind_10m')
                t2 = p.get('temperature_2m')
                lut[p['ts']] = {
                    'wspd_ms': round(math.hypot(u, v), 3) if (u is not None and v is not None) else None,
                    't2m_C': round(t2 - 273.15, 2) if t2 is not None else None,
                }
        json.dump(lut, open(cache_p, 'w'))
    return lut


def audit_cached_v002_scenes(dem, slope, aspect, tf, crs, masks):
    """Part A: Audit cached data/ecostress/ v002 scenes with 10m shadow ray-tracing, I_cum, ERA5, Manski."""
    src = os.path.join(C.STUDY_DIR, 'data', 'ecostress')
    tifs = glob.glob(os.path.join(src, '*.tif'))
    acq = {}
    for f in tifs:
        m = re.search(r'_(LST|QC|cloud)_doy(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})', os.path.basename(f))
        if not m:
            continue
        layer, yr, doy, hh, mm, ss = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)), int(m.group(6))
        utc_hr = hh + mm / 60.0 + ss / 3600.0
        alt_d, az_d, sol_hr = solar_position(doy, utc_hr)
        key = f'{yr}{doy:03d}{hh:02d}{mm:02d}{ss:02d}'
        dt_iso = datetime.strptime(f'{yr}-{doy:03d} {hh:02d}:{mm:02d}', '%Y-%j %H:%M').strftime('%Y-%m-%dT%H:00:00Z')
        acq.setdefault(key, {'meta': {
            'key': key, 'yr': yr, 'doy': doy, 'utc_hr': utc_hr, 'solar_hr': sol_hr,
            'alt_deg': alt_d, 'az_deg': az_d, 'dt_iso': dt_iso
        }})[layer] = f

    keys_lst = [k for k, v in acq.items() if 'LST' in v]
    print(f'[P3-A] Auditing {len(keys_lst)} cached v002 LST scenes...')
    era5_lut = fetch_era5_hourly_for_timestamps(list({acq[k]['meta']['dt_iso'] for k in keys_lst}))

    H, W = dem.shape
    px_m = abs(tf.a)
    m_lost = masks['lost']
    m_openn = masks['openn']
    m_plat = masks['plateau']

    scene_rows = []
    for k in sorted(keys_lst):
        rec = acq[k]
        meta = rec['meta']
        with rasterio.open(rec['LST']) as ds:
            raw_lst = ds.read(1).astype(np.float32)
            nod = ds.nodata
            src_tf, src_crs = ds.transform, ds.crs
        if nod is not None:
            raw_lst[raw_lst == nod] = np.nan
        raw_lst[raw_lst == 0] = np.nan
        if np.isfinite(raw_lst).sum() < 20:
            continue
        if np.nanmedian(raw_lst) > 1000:
            raw_lst = raw_lst * 0.02
        if np.nanmedian(raw_lst) > 150:
            raw_lst = raw_lst - 273.15
        raw_lst[(raw_lst < -35) | (raw_lst > 50)] = np.nan

        # Load cloud and QC
        cld_arr = None
        if 'cloud' in rec:
            with rasterio.open(rec['cloud']) as cs:
                c = cs.read(1)
            if c.shape == raw_lst.shape:
                cld_arr = (c > 0).astype(np.uint8)
        qc_bad = None
        if 'QC' in rec:
            with rasterio.open(rec['QC']) as qs:
                q = qs.read(1)
            if q.shape == raw_lst.shape:
                # Bits 0-1: 0=best, 1=nominal, 2=cloud, 3=not produced
                qc_bad = ((q & 3) >= 2).astype(np.uint8)

        # Reproject LST & cloud onto 10m grid (nearest preserves exact 70m ECOSTRESS values)
        lst10 = np.full((H, W), np.nan, np.float32)
        reproject(raw_lst, lst10, src_transform=src_tf, src_crs=src_crs,
                  dst_transform=tf, dst_crs=crs, src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.nearest)

        cld10 = np.zeros((H, W), np.uint8)
        if cld_arr is not None:
            reproject(cld_arr, cld10, src_transform=src_tf, src_crs=src_crs,
                      dst_transform=tf, dst_crs=crs, resampling=Resampling.nearest)
        qcb10 = np.zeros((H, W), np.uint8)
        if qc_bad is not None:
            reproject(qc_bad, qcb10, src_transform=src_tf, src_crs=src_crs,
                      dst_transform=tf, dst_crs=crs, resampling=Resampling.nearest)

        # Censoring diagnostics (swath-conditioned: fraction flagged cloud among in-swath pixels!)
        in_swath_lost = m_lost & (np.isfinite(lst10) | (cld10 > 0) | (qcb10 > 0))
        in_swath_openn = m_openn & (np.isfinite(lst10) | (cld10 > 0) | (qcb10 > 0))
        sw_l_frac = float(in_swath_lost.sum() / max(1, m_lost.sum()))
        sw_o_frac = float(in_swath_openn.sum() / max(1, m_openn.sum()))
        p_cld_lost = float(np.mean(cld10[in_swath_lost] > 0)) if in_swath_lost.sum() >= 50 else np.nan
        p_cld_openn = float(np.mean(cld10[in_swath_openn] > 0)) if in_swath_openn.sum() >= 50 else np.nan

        clear_lst = np.where((cld10 == 0) & (qcb10 == 0), lst10, np.nan)
        v_lost = clear_lst[m_lost]
        v_openn = clear_lst[m_openn]
        v_plat = clear_lst[m_plat]
        f_lost = float(np.isfinite(v_lost).mean())
        f_openn = float(np.isfinite(v_openn).mean())

        if sw_l_frac < 0.50 or sw_o_frac < 0.50:
            continue

        has_cov = (f_lost >= 0.35) and (f_openn >= 0.35)
        mean_l = float(np.nanmean(v_lost)) if np.isfinite(v_lost).sum() >= 20 else np.nan
        mean_o = float(np.nanmean(v_openn)) if np.isfinite(v_openn).sum() >= 20 else np.nan
        sd_l = float(np.nanstd(v_lost)) if np.isfinite(v_lost).sum() >= 20 else np.nan
        sd_o = float(np.nanstd(v_openn)) if np.isfinite(v_openn).sum() >= 20 else np.nan
        mean_p = float(np.nanmean(v_plat)) if np.isfinite(v_plat).sum() > 50 else np.nan
        diff = (mean_l - mean_o) if (np.isfinite(mean_l) and np.isfinite(mean_o)) else np.nan

        # Flag C2 cloud leakage outliers (e.g. half-clouded scene where spatial SD > 4.5 C or |diff| > 10 C)
        is_clean = (
            has_cov and np.isfinite(diff)
            and (abs(diff) < 10.0) and (sd_l < 4.5) and (sd_o < 4.5)
            and (f_lost >= 0.50) and (f_openn >= 0.50)
        )

        row = {
            'key': k, 'yr': meta['yr'], 'doy': meta['doy'],
            'solar_hr': round(meta['solar_hr'], 2),
            'alt_deg': round(meta['alt_deg'], 2), 'az_deg': round(meta['az_deg'], 1),
            'wspd_ms': era5_lut.get(meta['dt_iso'], {}).get('wspd_ms'),
            'era5_t2m_C': era5_lut.get(meta['dt_iso'], {}).get('t2m_C'),
            'f_valid_lost': round(f_lost, 3), 'f_valid_openn': round(f_openn, 3),
            'p_cld_lost': round(p_cld_lost, 4) if np.isfinite(p_cld_lost) else None,
            'p_cld_openn': round(p_cld_openn, 4) if np.isfinite(p_cld_openn) else None,
            'lst_lost': round(mean_l, 3) if np.isfinite(mean_l) else None,
            'lst_openn': round(mean_o, 3) if np.isfinite(mean_o) else None,
            'lst_plat': round(mean_p, 3) if np.isfinite(mean_p) else None,
            'diff_lost_minus_openn': round(diff, 3) if np.isfinite(diff) else None,
            'sd_lost': round(sd_l, 3) if np.isfinite(sd_l) else None,
            'sd_openn': round(sd_o, 3) if np.isfinite(sd_o) else None,
            'has_cov': bool(has_cov),
            'is_clean': bool(is_clean),
        }

        # For morning scenes (07.0 <= solar_hr <= 11.5) and clean: run exact 10m ray-traced shadow + I_cum!
        if is_clean and (6.5 <= meta['solar_hr'] <= 11.5) and meta['alt_deg'] > 0.5:
            sh = trace_cast_shadow(dem, px_m, meta['alt_deg'], meta['az_deg'], max_dist_m=3000.0, step_m=20.0)
            sh_lost = float(sh[m_lost & np.isfinite(clear_lst)].mean())
            sh_openn = float(sh[m_openn & np.isfinite(clear_lst)].mean())

            sun_l = clear_lst[m_lost & (~sh)]
            shd_l = clear_lst[m_lost & sh]
            sun_o = clear_lst[m_openn & (~sh)]

            row['shadow_frac_lost'] = round(sh_lost, 3)
            row['shadow_frac_openn'] = round(sh_openn, 3)
            row['sunlit_diff_L_minus_O'] = round(float(np.nanmean(sun_l) - np.nanmean(sun_o)), 3) if (np.isfinite(sun_l).sum() >= 30 and np.isfinite(sun_o).sum() >= 30) else None
            row['shadowed_L_minus_sunlit_O'] = round(float(np.nanmean(shd_l) - np.nanmean(sun_o)), 3) if (np.isfinite(shd_l).sum() >= 30 and np.isfinite(sun_o).sum() >= 30) else None

            # Cumulative post-sunrise insolation regression for 08-11h scenes
            if 8.0 <= meta['solar_hr'] <= 11.0:
                ic = compute_cum_insolation(dem, slope, aspect, px_m, meta['doy'], meta['utc_hr'], step_min=20)
                row['icum_lost_mean'] = round(float(np.nanmean(ic[m_lost])), 4)
                row['icum_openn_mean'] = round(float(np.nanmean(ic[m_openn])), 4)
                # Pixel-level OLS (subsampled every 7th pixel ~70m native resolution): LST ~ is_lost + I_cum + elev
                m_comb = (m_lost | m_openn) & np.isfinite(clear_lst)
                rr, cc = np.where(m_comb)
                sub_s = (rr % 7 == 0) & (cc % 7 == 0)
                rr, cc = rr[sub_s], cc[sub_s]
                if len(rr) >= 50:
                    y = clear_lst[rr, cc]
                    is_l = m_lost[rr, cc].astype(float)
                    X = np.column_stack([is_l, ic[rr, cc], dem[rr, cc]])
                    ols = C.ols_cluster_se(X, y, col_names=['is_lost', 'I_cum', 'elev'])
                    row['icum_adjusted_diff_L_minus_O'] = ols['coefs']['is_lost']['beta']
                    row['icum_beta'] = ols['coefs']['I_cum']['beta']
                    row['icum_r2'] = ols['r2']

        scene_rows.append(row)

    df_v2 = pd.DataFrame(scene_rows)
    clean = df_v2[df_v2['is_clean']].copy()

    # Aggregate 08.5-10.0h (strict 09-10h morning window) and 08.5-10.2h shadow audit
    m910 = clean[(clean['solar_hr'] >= 8.5) & (clean['solar_hr'] <= 10.0)]
    m910_calm = m910[m910['wspd_ms'].notna() & (m910['wspd_ms'] < 2.5)]
    m910_wide = clean[(clean['solar_hr'] >= 8.5) & (clean['solar_hr'] < 10.2)]
    m0811 = clean[(clean['solar_hr'] >= 8.0) & (clean['solar_hr'] < 11.0)]
    predawn = clean[(clean['solar_hr'] >= 1.5) & (clean['solar_hr'] < 6.5)]

    def grp_stats(sub, col='diff_lost_minus_openn'):
        v = sub[col].dropna().values
        if len(v) == 0:
            return {'n': 0}
        t_v, p_v = stats.ttest_1samp(v, 0.0) if len(v) >= 3 else (np.nan, np.nan)
        return {
            'n': int(len(v)),
            'mean_C': round(float(np.mean(v)), 3),
            'sd_C': round(float(np.std(v, ddof=1)), 3) if len(v) > 1 else 0.0,
            'p10_C': round(float(np.percentile(v, 10)), 3),
            't': round(float(t_v), 3) if np.isfinite(t_v) else None,
            'p': float(f'{p_v:.4g}') if np.isfinite(p_v) else None,
        }

    # Pre-dawn calm vs windy placebo
    pd_calm = predawn[predawn['wspd_ms'].notna() & (predawn['wspd_ms'] < 2.5)]
    pd_mod = predawn[predawn['wspd_ms'].notna() & (predawn['wspd_ms'] >= 2.5) & (predawn['wspd_ms'] < 4.5)]
    pd_windy = predawn[predawn['wspd_ms'].notna() & (predawn['wspd_ms'] >= 4.0)]

    # Manski bounds on ALL in-swath cold pre-dawn nights (ERA5 t2m < 5 C, including heavily cloud-censored nights!)
    cold_pd_all = df_v2[(df_v2['solar_hr'] >= 1.5) & (df_v2['solar_hr'] < 6.5) &
                        (df_v2['era5_t2m_C'].notna()) & (df_v2['era5_t2m_C'] < 5.0)]
    cold_pd_obs = cold_pd_all[cold_pd_all['is_clean']]
    if len(cold_pd_obs) >= 3:
        p_miss_l_all = float(cold_pd_all['p_cld_lost'].dropna().mean())
        p_miss_o_all = float(cold_pd_all['p_cld_openn'].dropna().mean())
        p_miss_l_obs = float(cold_pd_obs['p_cld_lost'].dropna().mean())
        p_miss_o_obs = float(cold_pd_obs['p_cld_openn'].dropna().mean())
        y_lo = float(np.nanpercentile(cold_pd_obs['lst_lost'], 5)) - 3.0
        y_hi = float(np.nanpercentile(cold_pd_obs['lst_openn'], 95)) + 1.0
        manski_pd_all = C.manski_diff_bounds(cold_pd_obs['lst_lost'].values, p_miss_l_all,
                                             cold_pd_obs['lst_openn'].values, p_miss_o_all, y_lo, y_hi)
        manski_pd_all['n_cold_in_swath'] = int(len(cold_pd_all))
        manski_pd_all['n_cold_clean'] = int(len(cold_pd_obs))
        manski_pd_obs = C.manski_diff_bounds(cold_pd_obs['lst_lost'].values, p_miss_l_obs,
                                             cold_pd_obs['lst_openn'].values, p_miss_o_obs, y_lo, y_hi)
        manski_pd = {'all_cold_swath_censoring': manski_pd_all, 'clean_scene_pixel_censoring': manski_pd_obs}
    else:
        manski_pd = {'note': 'too few cold predawn scenes'}

    return {
        'total_cached_v002_scenes': len(keys_lst),
        'in_swath_scenes': len(df_v2),
        'valid_coverage_scenes': int(df_v2['has_cov'].sum()),
        'clean_scenes': len(clean),
        'predawn_02_06h_all': grp_stats(predawn),
        'predawn_calm_lt2p5ms': grp_stats(pd_calm),
        'predawn_mod_2p5_4p5ms': grp_stats(pd_mod),
        'predawn_windy_ge4ms': grp_stats(pd_windy),
        'predawn_cold_manski': manski_pd,
        'morning_085_100h_all': grp_stats(m910, 'diff_lost_minus_openn'),
        'morning_085_100h_sunlit_only': grp_stats(m910, 'sunlit_diff_L_minus_O'),
        'morning_085_100h_icum_adjusted': grp_stats(m910, 'icum_adjusted_diff_L_minus_O'),
        'morning_085_100h_calm_sunlit': grp_stats(m910_calm, 'sunlit_diff_L_minus_O'),
        'morning_085_100h_calm_icum_adjusted': grp_stats(m910_calm, 'icum_adjusted_diff_L_minus_O'),
        'morning_085_102h_all': grp_stats(m910_wide, 'diff_lost_minus_openn'),
        'morning_085_102h_sunlit_only': grp_stats(m910_wide, 'sunlit_diff_L_minus_O'),
        'morning_085_102h_icum_adjusted': grp_stats(m910_wide, 'icum_adjusted_diff_L_minus_O'),
        'morning_085_100h_shadow_frac_lost_mean': round(float(m910['shadow_frac_lost'].mean()), 3) if 'shadow_frac_lost' in m910 and m910['shadow_frac_lost'].notna().any() else None,
        'morning_085_100h_shadow_frac_openn_mean': round(float(m910['shadow_frac_openn'].mean()), 3) if 'shadow_frac_openn' in m910 and m910['shadow_frac_openn'].notna().any() else None,
        'morning_085_100h_icum_lost_mean': round(float(m910['icum_lost_mean'].mean()), 4) if 'icum_lost_mean' in m910 and m910['icum_lost_mean'].notna().any() else None,
        'morning_085_100h_icum_openn_mean': round(float(m910['icum_openn_mean'].mean()), 4) if 'icum_openn_mean' in m910 and m910['icum_openn_mean'].notna().any() else None,
        'morning_08_11h_scenes': m0811.to_dict(orient='records'),
        'predawn_scenes': predawn.to_dict(orient='records'),
    }


def make_edl_session():
    env = {}
    for line in open(os.path.join(os.path.dirname(C.STUDY_DIR), '.env')):
        if '=' in line and not line.strip().startswith('#'):
            k, v = line.strip().split('=', 1)
            env[k] = v

    class EDLSession(requests.Session):
        def rebuild_auth(self, prepared_request, response):
            headers = prepared_request.headers
            url = prepared_request.url
            if 'Authorization' in headers:
                orig = requests.utils.urlparse(response.request.url).hostname
                new = requests.utils.urlparse(url).hostname
                if orig != new and new != 'urs.earthdata.nasa.gov':
                    del headers['Authorization']

    s = EDLSession()
    s.auth = (env['EARTHDATA_USER'], env['EARTHDATA_PASS'])
    return s


def fetch_and_audit_live_v003(dem, tf, crs, masks):
    """Part B: Live NASA LPCLOUD ECO_L2T_LSTE.003 retrieval for Spring (Feb-Apr) and Leaf-On Autumn (Sep-Nov)."""
    cache_v3 = os.path.join(C.CACHE_DIR, 'p3_v003_spring_autumn_scenes.json')
    if os.path.exists(cache_v3):
        rows = json.load(open(cache_v3))
        print(f'[P3-B] Loaded {len(rows)} cached live ECO_L2T_LSTE.003 scenes')
    else:
        print('[P3-B] Querying NASA CMR for live ECO_L2T_LSTE.003 (Feb-Apr & Sep-Nov, 2018-2026)...')
        granules = []
        for yr in [2018, 2019, 2025, 2026]:
            for season, m0, m1 in [('spring', '02-01', '04-30'), ('autumn', '09-01', '11-30')]:
                r = requests.get('https://cmr.earthdata.nasa.gov/search/granules.json', params={
                    'collection_concept_id': 'C3998139651-LPCLOUD',
                    'point': '-85.91,35.145',
                    'temporal': f'{yr}-{m0}T00:00:00Z,{yr}-{m1}T23:59:59Z',
                    'page_size': 200,
                }, timeout=30)
                if r.status_code != 200:
                    continue
                for e in r.json()['feed']['entry']:
                    title = e['title']
                    # Keep 16SED (covers Lost Cove + Sewanee Domain)
                    if '_16SED_' not in title:
                        continue
                    t_str = e['time_start']
                    dt = datetime.fromisoformat(t_str.replace('Z', '+00:00'))
                    doy = dt.timetuple().tm_yday
                    utc_hr = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
                    alt_d, az_d, sol_hr = solar_position(doy, utc_hr)
                    # Focus on pre-dawn (00-07h) & morning (07-11h) & late-evening (21-24h)
                    if not (sol_hr < 11.0 or sol_hr >= 21.0):
                        continue
                    links = {}
                    for l in e.get('links', []):
                        href = l.get('href', '')
                        if not href.startswith('https'):
                            continue
                        for suf in ['_LST.tif', '_QC.tif', '_cloud.tif', '_EmisWB.tif', '_view_zenith.tif']:
                            if href.endswith(suf):
                                links[suf[1:-4]] = href
                    if 'LST' in links and 'cloud' in links:
                        granules.append({
                            'title': title, 'season': season, 'yr': dt.year, 'month': dt.month,
                            'doy': doy, 'utc_hr': utc_hr, 'solar_hr': sol_hr,
                            'dt_iso': dt.strftime('%Y-%m-%dT%H:00:00Z'),
                            'links': links,
                        })

        # Deduplicate by timestamp (keep highest build number)
        by_t = {}
        for g in granules:
            k = (g['yr'], g['doy'], int(round(g['utc_hr'] * 10)))
            if k not in by_t or g['title'] > by_t[k]['title']:
                by_t[k] = g
        sel_g = sorted(by_t.values(), key=lambda x: (x['season'], x['yr'], x['doy'], x['utc_hr']))
        print(f'  Found {len(sel_g)} unique 16SED night/morning v003 granules '
              f'(spring={sum(g["season"]=="spring" for g in sel_g)}, autumn={sum(g["season"]=="autumn" for g in sel_g)})')

        # Sample up to 35 spring + 35 autumn pre-dawn/night/morning scenes (prefer 00-07h pre-dawn + Oct!)
        def prio(g):
            # Prioritize pre-dawn 01-06.5h and Oct (month==10) / Mar-Apr (month in 3,4)
            pd_bonus = 0 if (1.0 <= g['solar_hr'] <= 6.5) else (1 if g['solar_hr'] >= 21.5 or g['solar_hr'] < 1.0 else 2)
            mo_bonus = 0 if g['month'] in (3, 4, 10) else 1
            return (pd_bonus, mo_bonus, g['yr'], g['doy'])

        spr_g = sorted([g for g in sel_g if g['season'] == 'spring'], key=prio)[:40]
        aut_g = sorted([g for g in sel_g if g['season'] == 'autumn'], key=prio)[:45]
        target_g = spr_g + aut_g
        print(f'  Downloading & analyzing {len(target_g)} live v003 granules (LST, cloud, QC, EmisWB, view_zenith)...')

        H, W = dem.shape
        m_lost = masks['lost']
        m_openn = masks['openn']
        m_fd40 = masks['lost_fd40']
        m_fd30 = masks['lost_fd30']
        m_plat = masks['plateau']

        def proc_granule(g):
            s = make_edl_session()
            try:
                def load_layer(lname):
                    if lname not in g['links']:
                        return None, None, None
                    resp = s.get(g['links'][lname], timeout=35)
                    if resp.status_code != 200:
                        return None, None, None
                    with rasterio.MemoryFile(resp.content) as mf:
                        with mf.open() as ds:
                            arr = ds.read(1).astype(np.float32)
                            nod = ds.nodata
                            if nod is not None and np.isfinite(nod):
                                arr[arr == nod] = np.nan
                            return arr, ds.transform, ds.crs

                lst, stf, scrs = load_layer('LST')
                if lst is None:
                    return None
                # Reproject onto 10m Domain grid
                def to10(a, nearest=True):
                    out = np.full((H, W), np.nan, np.float32)
                    reproject(a, out, src_transform=stf, src_crs=scrs, dst_transform=tf, dst_crs=crs,
                              src_nodata=np.nan, dst_nodata=np.nan,
                              resampling=Resampling.nearest if nearest else Resampling.bilinear)
                    return out

                lst10 = to10(lst)
                if np.isfinite(lst10[m_lost]).sum() < 20:
                    return None
                if np.nanmedian(lst10) > 1000:
                    lst10 = lst10 * 0.02
                if np.nanmedian(lst10) > 150:
                    lst10 = lst10 - 273.15
                lst10[(lst10 < -35) | (lst10 > 55)] = np.nan

                cld, _, _ = load_layer('cloud')
                cld10 = to10(cld) if cld is not None else np.zeros((H, W), np.float32)
                qc, _, _ = load_layer('QC')
                qc10 = to10(qc) if qc is not None else np.zeros((H, W), np.float32)

                p_cld_l = float(np.nanmean((cld10[m_lost] > 0).astype(float)))
                p_cld_o = float(np.nanmean((cld10[m_openn] > 0).astype(float)))

                # v003 clear mask: cloud == 0 and QC bits 0-1 <= 1
                q_int = np.where(np.isfinite(qc10), qc10, 3).astype(np.int32)
                clear = (cld10 == 0) & ((q_int & 3) <= 1) & np.isfinite(lst10)
                clst = np.where(clear, lst10, np.nan)

                f_l = float(np.isfinite(clst[m_lost]).mean())
                f_o = float(np.isfinite(clst[m_openn]).mean())
                if f_l < 0.35 or f_o < 0.35:
                    return {
                        'title': g['title'], 'season': g['season'], 'yr': g['yr'], 'month': g['month'],
                        'doy': g['doy'], 'solar_hr': round(g['solar_hr'], 2), 'dt_iso': g['dt_iso'],
                        'f_valid_lost': round(f_l, 3), 'f_valid_openn': round(f_o, 3),
                        'p_cld_lost': round(p_cld_l, 4), 'p_cld_openn': round(p_cld_o, 4),
                        'usable': False,
                    }

                # Also fetch EmisWB and view_zenith only for clear scenes!
                emis, _, _ = load_layer('EmisWB')
                vza, _, _ = load_layer('view_zenith')
                d_emis = None
                if emis is not None:
                    e10 = to10(emis)
                    if np.nanmedian(e10) > 2.0:
                        e10 = e10 * 0.002 + 0.49
                    e10 = np.where(clear & (e10 > 0.5) & (e10 <= 1.0), e10, np.nan)
                    if np.isfinite(e10[m_lost]).sum() >= 30 and np.isfinite(e10[m_openn]).sum() >= 30:
                        d_emis = round(float(np.nanmean(e10[m_lost]) - np.nanmean(e10[m_openn])), 5)
                vza_mean = None
                if vza is not None:
                    v10 = to10(vza)
                    if np.isfinite(v10[m_lost]).sum() >= 30:
                        vza_mean = round(float(np.nanmean(v10[m_lost])), 2)

                ml = float(np.nanmean(clst[m_lost]))
                mo = float(np.nanmean(clst[m_openn]))
                m_fd30_val = float(np.nanmean(clst[m_fd30])) if np.isfinite(clst[m_fd30]).sum() >= 20 else np.nan
                m_fd40_val = float(np.nanmean(clst[m_fd40])) if np.isfinite(clst[m_fd40]).sum() >= 15 else np.nan
                mp = float(np.nanmean(clst[m_plat])) if np.isfinite(clst[m_plat]).sum() >= 50 else np.nan
                sd_l = float(np.nanstd(clst[m_lost]))
                sd_o = float(np.nanstd(clst[m_openn]))
                diff = ml - mo
                is_clean = (abs(diff) < 10.0) and (sd_l < 4.5) and (sd_o < 4.5) and (f_l >= 0.50) and (f_o >= 0.50)

                return {
                    'title': g['title'], 'season': g['season'], 'yr': g['yr'], 'month': g['month'],
                    'doy': g['doy'], 'solar_hr': round(g['solar_hr'], 2), 'dt_iso': g['dt_iso'],
                    'f_valid_lost': round(f_l, 3), 'f_valid_openn': round(f_o, 3),
                    'p_cld_lost': round(p_cld_l, 4), 'p_cld_openn': round(p_cld_o, 4),
                    'lst_lost': round(ml, 3), 'lst_openn': round(mo, 3),
                    'lst_plat': round(mp, 3) if np.isfinite(mp) else None,
                    'diff_lost_minus_openn': round(diff, 3),
                    'diff_fd30_minus_openn': round(m_fd30_val - mo, 3) if np.isfinite(m_fd30_val) else None,
                    'diff_fd40_minus_openn': round(m_fd40_val - mo, 3) if np.isfinite(m_fd40_val) else None,
                    'd_emis_wb': d_emis, 'vza_deg': vza_mean,
                    'sd_lost': round(sd_l, 3), 'sd_openn': round(sd_o, 3),
                    'usable': True, 'is_clean': bool(is_clean),
                }
            except Exception as ex:
                return None

        rows = []
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = [ex.submit(proc_granule, g) for g in target_g]
            for fut in as_completed(futs):
                r = fut.result()
                if r is not None:
                    rows.append(r)
        print(f'  Processed {len(rows)} v003 granules in {time.time()-t0:.1f}s '
              f'(usable={sum(r.get("usable", False) for r in rows)}, clean={sum(r.get("is_clean", False) for r in rows)})')
        json.dump(rows, open(cache_v3, 'w'))

    # Attach ERA5-Land hourly wind speed for all v003 rows (available through ~2026-10 or latest ERA5-Land)
    ts_all = list({r['dt_iso'] for r in rows if r.get('dt_iso')})
    era5_lut = fetch_era5_hourly_for_timestamps(ts_all)
    for r in rows:
        e = era5_lut.get(r.get('dt_iso'), {})
        r['wspd_ms'] = e.get('wspd_ms')
        r['era5_t2m_C'] = e.get('t2m_C')

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return {'status': 'no_v003_scenes'}
    cln = df[df.get('is_clean', False) == True].copy()

    def sstat(sub, col='diff_lost_minus_openn'):
        v = sub[col].dropna().values
        if len(v) == 0:
            return {'n': 0}
        t_v, p_v = stats.ttest_1samp(v, 0.0) if len(v) >= 3 else (np.nan, np.nan)
        return {
            'n': int(len(v)),
            'mean_C': round(float(np.mean(v)), 3),
            'sd_C': round(float(np.std(v, ddof=1)), 3) if len(v) > 1 else 0.0,
            'p10_C': round(float(np.percentile(v, 10)), 3),
            't': round(float(t_v), 3) if np.isfinite(t_v) else None,
            'p': float(f'{p_v:.4g}') if np.isfinite(p_v) else None,
        }

    is_night_pd = (cln['solar_hr'] < 7.0) | (cln['solar_hr'] >= 21.0)
    spr_pd = cln[(cln['season'] == 'spring') & is_night_pd]
    aut_pd = cln[(cln['season'] == 'autumn') & is_night_pd]
    oct_pd = cln[(cln['month'] == 10) & is_night_pd]
    all_pd = cln[is_night_pd]
    calm_pd = all_pd[all_pd['wspd_ms'].notna() & (all_pd['wspd_ms'] < 2.5)]
    windy_pd = all_pd[all_pd['wspd_ms'].notna() & (all_pd['wspd_ms'] >= 4.0)]
    vza_lo = all_pd[all_pd['vza_deg'].notna() & (all_pd['vza_deg'] < 25.0)]

    return {
        'n_granules_inspected': int(len(df)),
        'n_clean_scenes': int(len(cln)),
        'spring_feb_apr_night_predawn': sstat(spr_pd),
        'spring_feb_apr_fd40': sstat(spr_pd, 'diff_fd40_minus_openn'),
        'autumn_sep_nov_night_predawn': sstat(aut_pd),
        'autumn_oct_only_night_predawn': sstat(oct_pd),
        'autumn_oct_fd40': sstat(oct_pd, 'diff_fd40_minus_openn'),
        'all_night_calm_lt2p5ms': sstat(calm_pd),
        'all_night_windy_ge4ms': sstat(windy_pd),
        'all_night_vza_lt25deg': sstat(vza_lo),
        'mean_d_EmisWB': round(float(cln['d_emis_wb'].dropna().mean()), 5) if cln['d_emis_wb'].notna().any() else None,
        'max_abs_d_EmisWB': round(float(cln['d_emis_wb'].dropna().abs().max()), 5) if cln['d_emis_wb'].notna().any() else None,
        'v003_cloud_censor_lost_mean': round(float(df['p_cld_lost'].mean()), 4),
        'v003_cloud_censor_openn_mean': round(float(df['p_cld_openn'].mean()), 4),
        'clean_scene_list': cln.to_dict(orient='records'),
    }


def main():
    t0 = time.time()
    C.init_ee()
    dem, nlcd, prof, tf, crs = load_domain_10m_dem_nlcd()
    px_m = abs(tf.a)
    slope, aspect = compute_slope_aspect(dem, px_m)
    masks = build_10m_masks(dem, nlcd, tf, crs)

    part_a = audit_cached_v002_scenes(dem, slope, aspect, tf, crs, masks)
    part_b = fetch_and_audit_live_v003(dem, tf, crs, masks)

    # Gate checks (strict 08.5-10.0h morning window + calm sunlit breakdown)
    sunlit_910 = part_a.get('morning_085_100h_sunlit_only', {})
    calm_sun_910 = part_a.get('morning_085_100h_calm_sunlit', {})
    icum_910 = part_a.get('morning_085_100h_icum_adjusted', {})
    whiteman_keep = (
        sunlit_910.get('n', 0) >= 3
        and sunlit_910.get('mean_C', 0) <= -1.0
        and (sunlit_910.get('p') is not None and sunlit_910['p'] < 0.05)
    )
    whiteman_drop = (sunlit_910.get('n', 0) >= 2 and abs(sunlit_910.get('mean_C', 0)) < 0.5)

    v3_spr = part_b.get('spring_feb_apr_night_predawn', {})
    v3_aut = part_b.get('autumn_sep_nov_night_predawn', {})
    v3_oct = part_b.get('autumn_oct_only_night_predawn', {})
    v3_emis = part_b.get('mean_d_EmisWB')

    out = {
        'experiment': 'P3_ecostress_shadow_v003',
        'elapsed_s': round(time.time() - t0, 1),
        'max_lost_cove_filldepth_10m': round(masks['max_lost_fd_10m'], 1),
        'part_a_v002_shadow_and_synoptic': part_a,
        'part_b_live_v003_spring_autumn': part_b,
        'gate_check': {
            'whiteman_085_100h_all_C': part_a['morning_085_100h_all'].get('mean_C'),
            'whiteman_085_100h_sunlit_only_C': sunlit_910.get('mean_C'),
            'whiteman_085_100h_sunlit_p': sunlit_910.get('p'),
            'whiteman_085_100h_calm_sunlit_C': calm_sun_910.get('mean_C'),
            'whiteman_085_100h_calm_sunlit_p': calm_sun_910.get('p'),
            'whiteman_085_100h_icum_adj_C': icum_910.get('mean_C'),
            'whiteman_085_100h_shadow_frac_lost': part_a.get('morning_085_100h_shadow_frac_lost_mean'),
            'whiteman_085_100h_icum_lost': part_a.get('morning_085_100h_icum_lost_mean'),
            'whiteman_085_100h_icum_openn': part_a.get('morning_085_100h_icum_openn_mean'),
            'whiteman_lag_verdict': 'KEEP' if whiteman_keep else ('DROP_SHADOW_ARTIFACT' if whiteman_drop else 'INCONCLUSIVE_CALM_ONLY'),
            'v002_predawn_calm_C': part_a['predawn_calm_lt2p5ms'].get('mean_C'),
            'v002_predawn_windy_C': part_a['predawn_windy_ge4ms'].get('mean_C'),
            'v003_spring_night_C': v3_spr.get('mean_C'),
            'v003_spring_night_n': v3_spr.get('n'),
            'v003_autumn_night_C': v3_aut.get('mean_C'),
            'v003_autumn_night_n': v3_aut.get('n'),
            'v003_oct_night_C': v3_oct.get('mean_C'),
            'v003_oct_night_n': v3_oct.get('n'),
            'v003_mean_d_EmisWB': v3_emis,
            'emis_pass': (v3_emis is not None and abs(v3_emis) < 0.008),
        }
    }
    C.save_json('p3.json', out)
    print('\n=== P3 SUMMARY ===')
    print(json.dumps(out['gate_check'], indent=2))


if __name__ == '__main__':
    main()
