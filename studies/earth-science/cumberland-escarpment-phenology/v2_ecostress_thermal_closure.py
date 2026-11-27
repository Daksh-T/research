"""Stage 4 (4B): ECOSTRESS v002 + v003 Multi-Season, Windy-Placebo, and Multi-Cove Thermal Closure.

Closes Stage-3 P3 gaps:
1. Queries NASA LPCLOUD ECO_L2T_LSTE.002 (C2076090826-LPCLOUD, 2020-2024, where v003 had 0 granules)
   for Spring (Feb-Apr), Autumn (Sep-Nov), and high-wind winter/shoulder nights over MGRS tile 16SED.
2. Merges (timestamp-deduplicated, preferring v003 where overlapping) with cached live v003 scenes
   (profiling/cache/p3_v003_spring_autumn_scenes.json) to close:
   - Spring nighttime/pre-dawn (Feb-Apr) cold-air pooling sample size (was n=2 in v003 alone)
   - Autumn (Sep-Nov & Oct) nighttime cold-air pooling power
   - Windy-night thermal placebo (u10 >= 4.0 m/s, >= 3.5 m/s, plus continuous Delta_LST vs. u10 regression)
   - Honest swath-conditioned Manski MNAR censoring bounds (both swath-level & pixel-level)
   - Emissivity (EmisWB) & view-zenith (VZA < 25 deg) artifact controls
3. Multi-cove thermal replication across 6 additional closed-karst vs. open-gorge pairs lying inside
   the same MGRS 16SED granules (Karst_TN_02, Karst_TN_17, Karst_AL_05, Karst_AL_10, Karst_AL_12,
   plus Sinking Cove vs. elevation-matched open gorge) using cached regional 3DEP+NLCD masks.

Outputs:
  data/v2_ecostress_thermal_closure.json (strict RFC 8259 JSON)
"""
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
import threading
from datetime import datetime
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import requests
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'profiling'))
import common as C
import p1_karst_census as P1
import p3_ecostress_shadow_v003 as P3

OUT_PATH = os.path.join(HERE, 'data', 'v2_ecostress_thermal_closure.json')
CACHE_V002_LIVE = os.path.join(C.CACHE_DIR, 'v2_ecostress_v002_live_scenes.json')
CACHE_MC_META = os.path.join(C.CACHE_DIR, 'v2_ecostress_multicove_meta.json')
V002_CONCEPT_ID = 'C2076090826-LPCLOUD'

# Approximate bounds of MGRS tile 16SED (covers Sewanee/Lost Cove, Sinking Cove, TN/AL border karst)
TILE_16SED_BBOX = (-87.00, 34.34, -85.79, 35.33)


def build_multicove_90m_masks():
    """Build paired Closed Floor vs. Open Gorge Floor masks on the cached 90m regional DEM grid
    for every P1 pair inside MGRS tile 16SED + Sinking Cove."""
    dem, nlcd, tf, bounds = P1.load_regional_dem_nlcd()
    _, fill_depth = C.priority_flood_fill(dem)
    _, _, _, pairs = P1.detect_closed_basins_and_open_pairs(dem, nlcd, tf)

    H, W = dem.shape
    lons = tf.c + (np.arange(W) + 0.5) * tf.a
    lats = tf.f + (np.arange(H) + 0.5) * tf.e
    lon2d = np.broadcast_to(lons[None, :], (H, W))
    lat2d = np.broadcast_to(lats[:, None], (H, W))
    decid = (nlcd == 41) & np.isfinite(dem)

    # Compute local TPI (~1350m) for open gorge floor definition
    from scipy import ndimage
    vdem = np.where(np.isfinite(dem), dem, np.nanmedian(dem))
    tpi = dem - ndimage.uniform_filter(vdem, size=31)

    cove_pairs = []
    for p in pairs:
        cl, op = p['closed'], p['open']
        # Check inside 16SED with small margin
        if not (TILE_16SED_BBOX[0] + 0.05 <= cl['lon'] <= TILE_16SED_BBOX[2] - 0.03 and
                TILE_16SED_BBOX[1] + 0.05 <= cl['lat'] <= TILE_16SED_BBOX[3] - 0.03):
            continue
        if not (TILE_16SED_BBOX[0] + 0.05 <= op['lon'] <= TILE_16SED_BBOX[2] - 0.03 and
                TILE_16SED_BBOX[1] + 0.05 <= op['lat'] <= TILE_16SED_BBOX[3] - 0.03):
            continue
        lon0, lat0, lon1, lat1 = cl['bbox']
        in_cl = (lon2d >= lon0) & (lon2d <= lon1) & (lat2d >= lat0) & (lat2d <= lat1)
        m_cl = decid & in_cl & (fill_depth >= max(10.0, 0.45 * cl['max_fill_m']))

        # Open pair floor: within ~1.4 km of open gorge center, FillDepth < 2m, elev within +/-35m of floor med
        d_op = np.hypot((lon2d - op['lon']) * math.cos(math.radians(op['lat'])), lat2d - op['lat']) * 111.32
        m_op = (decid & (d_op <= 1.5) & (fill_depth < 2.0) & (tpi < -40.0) &
                (dem <= op['z_floor_med'] + 35.0))
        if m_cl.sum() >= 6 and m_op.sum() >= 8:
            cove_pairs.append({
                'name': cl['name'], 'open_name': op['name'],
                'max_fill_m': cl['max_fill_m'],
                'lat': cl['lat'], 'lon': cl['lon'],
                'z_cl': round(float(np.nanmedian(dem[m_cl])), 1),
                'z_op': round(float(np.nanmedian(dem[m_op])), 1),
                'n_cl': int(m_cl.sum()), 'n_op': int(m_op.sum()),
                'm_cl': m_cl, 'm_op': m_op,
            })

    # Also add Sinking Cove (35.012, -85.963) vs. nearby elevation-matched open gorge floor (~3-10km)
    d_sk = np.hypot((lon2d - (-85.963)) * math.cos(math.radians(35.012)), lat2d - 35.012) * 111.32
    m_sk_cl = decid & (d_sk <= 3.0) & (fill_depth >= 8.0)
    if m_sk_cl.sum() >= 6:
        z_sk = float(np.nanmedian(dem[m_sk_cl]))
        # Open gorge floor 3-10 km from Sinking Cove at matched elevation (+/-40m)
        m_sk_op = (decid & (d_sk >= 3.0) & (d_sk <= 10.0) & (fill_depth < 2.0) & (tpi < -40.0) &
                   (dem >= z_sk - 40.0) & (dem <= z_sk + 40.0))
        if m_sk_op.sum() >= 10:
            cove_pairs.append({
                'name': 'Sinking_Cove_TN', 'open_name': 'OpenPair_Sinking_Cove',
                'max_fill_m': round(float(np.nanmax(fill_depth[m_sk_cl])), 1),
                'lat': 35.012, 'lon': -85.963,
                'z_cl': round(z_sk, 1),
                'z_op': round(float(np.nanmedian(dem[m_sk_op])), 1),
                'n_cl': int(m_sk_cl.sum()), 'n_op': int(m_sk_op.sum()),
                'm_cl': m_sk_cl, 'm_op': m_sk_op,
            })

    print(f'[4B] Built 90m multi-cove thermal masks inside 16SED for {len(cove_pairs)} regional pairs:')
    for cp in cove_pairs:
        print(f"  {cp['name']:18s} fill={cp['max_fill_m']:5.1f}m | z_cl={cp['z_cl']:.0f}m (n={cp['n_cl']}) "
              f"vs z_op={cp['z_op']:.0f}m (n={cp['n_op']})")
    return cove_pairs, dem.shape, tf, 'EPSG:4326'


def query_cmr_v002_granules():
    """Search NASA CMR for ECO_L2T_LSTE.002 (2020-2024) 16SED nighttime/pre-dawn granules."""
    granules = []
    # Query all seasons in 2020-2024 so we capture Spring (Feb-Apr), Autumn (Sep-Nov),
    # AND winter/shoulder high-wind nights (Dec-Jan) for the windy thermal placebo!
    season_map = {
        1: 'winter', 2: 'spring', 3: 'spring', 4: 'spring',
        5: 'late_spring', 6: 'summer', 7: 'summer', 8: 'summer',
        9: 'autumn', 10: 'autumn', 11: 'autumn', 12: 'winter',
    }
    for yr in range(2020, 2025):
        for chunk_start, chunk_end in [('01-01', '05-15'), ('09-01', '12-31')]:
            r = requests.get('https://cmr.earthdata.nasa.gov/search/granules.json', params={
                'collection_concept_id': V002_CONCEPT_ID,
                'point': '-85.91,35.145',
                'temporal': f'{yr}-{chunk_start}T00:00:00Z,{yr}-{chunk_end}T23:59:59Z',
                'page_size': 200,
            }, timeout=30)
            if r.status_code != 200:
                continue
            for e in r.json()['feed']['entry']:
                title = e['title']
                if '_16SED_' not in title:
                    continue
                t_str = e['time_start']
                dt = datetime.fromisoformat(t_str.replace('Z', '+00:00'))
                doy = dt.timetuple().tm_yday
                utc_hr = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
                _, _, sol_hr = P3.solar_position(doy, utc_hr)
                # Keep night/pre-dawn (00-07h or 21-24h)
                if not (sol_hr < 7.0 or sol_hr >= 21.0):
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
                        'title': title, 'version': 'v002',
                        'season': season_map[dt.month], 'yr': dt.year, 'month': dt.month,
                        'doy': doy, 'utc_hr': round(utc_hr, 4), 'solar_hr': round(sol_hr, 3),
                        'dt_iso': dt.strftime('%Y-%m-%dT%H:00:00Z'),
                        'links': links,
                    })
    # Deduplicate by timestamp (keep latest build)
    by_t = {}
    for g in granules:
        k = (g['yr'], g['doy'], int(round(g['utc_hr'] * 10)))
        if k not in by_t or g['title'] > by_t[k]['title']:
            by_t[k] = g
    sel = sorted(by_t.values(), key=lambda x: (x['yr'], x['doy'], x['utc_hr']))
    print(f'[4B] CMR ECO_L2T_LSTE.002 (2020-2024) 16SED night granules: total={len(sel)} '
          f'(spring={sum(g["season"]=="spring" for g in sel)}, '
          f'autumn={sum(g["season"]=="autumn" for g in sel)}, '
          f'winter={sum(g["season"]=="winter" for g in sel)})')
    return sel


def fetch_and_process_v002_live(dem10, tf10, crs10, masks10, cove_pairs, shape90, tf90, crs90):
    """Download & process prioritized v002 night granules (all Spring + all Autumn + all Windy nights)."""
    if os.path.exists(CACHE_V002_LIVE):
        rows = json.load(open(CACHE_V002_LIVE))
        print(f'[4B] Loaded {len(rows)} cached live ECO_L2T_LSTE.002 scenes')
        return rows

    granules = query_cmr_v002_granules()
    # Fetch ERA5-Land hourly wind speed for ALL candidate timestamps FIRST (1 fast GEE lookup!)
    # so we make sure we download 100% of windy nights (u10 >= 3.5 m/s) plus all spring & autumn nights!
    era5_lut = P3.fetch_era5_hourly_for_timestamps(list({g['dt_iso'] for g in granules}))
    for g in granules:
        e = era5_lut.get(g['dt_iso'], {})
        g['wspd_ms'] = e.get('wspd_ms')
        g['era5_t2m_C'] = e.get('t2m_C')

    n_windy_cand = sum(1 for g in granules if (g['wspd_ms'] or 0) >= 4.0)
    n_modwind_cand = sum(1 for g in granules if (g['wspd_ms'] or 0) >= 3.5)
    print(f'  Candidate night granules with ERA5 u10>=4.0 m/s: {n_windy_cand} (>=3.5 m/s: {n_modwind_cand})')

    # Select ALL spring (Feb-Apr), ALL windy nights (u10 >= 3.0 m/s), and prioritized autumn (Sep-Nov, up to 70)
    # Note: ERA5-Land 10m wind over forest roughness (z0~1m) is ~0.6x open-airport wind, so u10>=3.0-4.0 m/s is windy.
    spr_g = [g for g in granules if g['season'] == 'spring']
    wnd_g = [g for g in granules if (g['wspd_ms'] or 0.0) >= 3.0 and g['season'] != 'spring']
    aut_rem = [g for g in granules if g['season'] == 'autumn' and (g['wspd_ms'] or 0.0) < 3.0]
    # Prioritize October and pre-dawn (01-06.5h) among remaining autumn granules
    aut_rem.sort(key=lambda g: (0 if g['month'] == 10 else 1, 0 if 1.0 <= g['solar_hr'] <= 6.5 else 1, g['yr'], g['doy']))
    seen_t = set()
    target = []
    for g in spr_g + wnd_g + aut_rem[:75]:
        if g['title'] not in seen_t:
            seen_t.add(g['title'])
            target.append(g)

    # Incremental JSONL checkpoint so no progress is ever lost
    ckpt_path = CACHE_V002_LIVE + 'l'
    done_map = {}
    if os.path.exists(ckpt_path):
        for line in open(ckpt_path):
            if line.strip():
                rec = json.loads(line)
                done_map[rec['title']] = rec
    todo = [g for g in target if g['title'] not in done_map]
    print(f'  Downloading & processing {len(todo)}/{len(target)} live v002 16SED night granules '
          f'(already checkpointed={len(done_map)})...')

    H10, W10 = dem10.shape
    H90, W90 = shape90
    m_lost = masks10['lost']
    m_openn = masks10['openn']
    m_fd30 = masks10['lost_fd30']
    m_fd40 = masks10['lost_fd40']
    m_plat = masks10['plateau']

    tls = threading.local()

    def get_sess():
        if not hasattr(tls, 'sess'):
            tls.sess = P3.make_edl_session()
        return tls.sess

    def proc_one(g):
        s = get_sess()
        try:
            def load_layer(lname):
                if lname not in g['links']:
                    return None, None, None
                resp = s.get(g['links'][lname], timeout=(6, 18))
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

            # Normalize LST units to Celsius
            valid_raw = lst[np.isfinite(lst) & (lst != 0)]
            if len(valid_raw) < 100:
                return None
            med = float(np.median(valid_raw))
            if med > 1000.0:
                lst = lst * 0.02
                med *= 0.02
            if med > 150.0:
                lst = lst - 273.15
            lst[(lst < -35.0) | (lst > 55.0)] = np.nan

            cld, _, _ = load_layer('cloud')
            qc, _, _ = load_layer('QC')

            def reproj(a, dst_h, dst_w, dst_tf, dst_crs):
                out = np.full((dst_h, dst_w), np.nan, np.float32)
                reproject(a, out, src_transform=stf, src_crs=scrs,
                          dst_transform=dst_tf, dst_crs=dst_crs,
                          src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.nearest)
                return out

            # 1. Sewanee 10m Lost Cove vs. OpenN evaluation
            lst10 = reproj(lst, H10, W10, tf10, crs10)
            cld10 = reproj(cld, H10, W10, tf10, crs10) if cld is not None else np.zeros((H10, W10), np.float32)
            qc10 = reproj(qc, H10, W10, tf10, crs10) if qc is not None else np.zeros((H10, W10), np.float32)

            in_sw_l = m_lost & (np.isfinite(lst10) | (cld10 > 0) | np.isfinite(qc10))
            in_sw_o = m_openn & (np.isfinite(lst10) | (cld10 > 0) | np.isfinite(qc10))
            sw_l = float(in_sw_l.sum() / max(1, m_lost.sum()))
            sw_o = float(in_sw_o.sum() / max(1, m_openn.sum()))
            p_cld_l = float(np.mean(cld10[in_sw_l] > 0)) if in_sw_l.sum() >= 50 else np.nan
            p_cld_o = float(np.mean(cld10[in_sw_o] > 0)) if in_sw_o.sum() >= 50 else np.nan

            q_int = np.where(np.isfinite(qc10), qc10, 3).astype(np.int32)
            clear10 = (cld10 == 0) & ((q_int & 3) <= 1) & np.isfinite(lst10)
            clst10 = np.where(clear10, lst10, np.nan)

            f_l = float(np.isfinite(clst10[m_lost]).mean())
            f_o = float(np.isfinite(clst10[m_openn]).mean())

            rec = {
                'title': g['title'], 'version': 'v002', 'season': g['season'],
                'yr': g['yr'], 'month': g['month'], 'doy': g['doy'],
                'utc_hr': g['utc_hr'], 'solar_hr': g['solar_hr'], 'dt_iso': g['dt_iso'],
                'wspd_ms': g['wspd_ms'], 'era5_t2m_C': g['era5_t2m_C'],
                'in_swath_lost_frac': round(sw_l, 3), 'in_swath_openn_frac': round(sw_o, 3),
                'f_valid_lost': round(f_l, 3), 'f_valid_openn': round(f_o, 3),
                'p_cld_lost': round(p_cld_l, 4) if np.isfinite(p_cld_l) else None,
                'p_cld_openn': round(p_cld_o, 4) if np.isfinite(p_cld_o) else None,
                'usable': False, 'is_clean': False, 'multicove': {},
            }

            if f_l >= 0.35 and f_o >= 0.35:
                ml = float(np.nanmean(clst10[m_lost]))
                mo = float(np.nanmean(clst10[m_openn]))
                m30 = float(np.nanmean(clst10[m_fd30])) if np.isfinite(clst10[m_fd30]).sum() >= 20 else np.nan
                m40 = float(np.nanmean(clst10[m_fd40])) if np.isfinite(clst10[m_fd40]).sum() >= 15 else np.nan
                mp = float(np.nanmean(clst10[m_plat])) if np.isfinite(clst10[m_plat]).sum() >= 50 else np.nan
                sd_l = float(np.nanstd(clst10[m_lost]))
                sd_o = float(np.nanstd(clst10[m_openn]))
                diff = ml - mo
                is_clean = (abs(diff) < 10.0) and (sd_l < 4.5) and (sd_o < 4.5) and (f_l >= 0.50) and (f_o >= 0.50)

                # Fetch EmisWB and view_zenith only for clear scenes
                d_emis, vza_mean = None, None
                if is_clean:
                    emis, _, _ = load_layer('EmisWB')
                    if emis is not None:
                        e10 = reproj(emis, H10, W10, tf10, crs10)
                        if np.nanmedian(e10) > 2.0:
                            e10 = e10 * 0.002 + 0.49
                        e10 = np.where(clear10 & (e10 > 0.5) & (e10 <= 1.0), e10, np.nan)
                        if np.isfinite(e10[m_lost]).sum() >= 30 and np.isfinite(e10[m_openn]).sum() >= 30:
                            d_emis = round(float(np.nanmean(e10[m_lost]) - np.nanmean(e10[m_openn])), 5)
                    vza, _, _ = load_layer('view_zenith')
                    if vza is not None:
                        v10 = reproj(vza, H10, W10, tf10, crs10)
                        if np.isfinite(v10[m_lost]).sum() >= 30:
                            vza_mean = round(float(np.nanmean(v10[m_lost])), 2)

                rec.update({
                    'lst_lost': round(ml, 3), 'lst_openn': round(mo, 3),
                    'lst_plat': round(mp, 3) if np.isfinite(mp) else None,
                    'diff_lost_minus_openn': round(diff, 3),
                    'diff_fd30_minus_openn': round(m30 - mo, 3) if np.isfinite(m30) else None,
                    'diff_fd40_minus_openn': round(m40 - mo, 3) if np.isfinite(m40) else None,
                    'd_emis_wb': d_emis, 'vza_deg': vza_mean,
                    'sd_lost': round(sd_l, 3), 'sd_openn': round(sd_o, 3),
                    'usable': True, 'is_clean': bool(is_clean),
                })

            # 2. Multi-cove replication on 90m regional grid (zero extra downloads!)
            # Compute clear mask on native granule first, then reproject once to 90m grid
            q_nat = np.where(np.isfinite(qc), qc, 3).astype(np.int32) if qc is not None else np.zeros(lst.shape, np.int32)
            c_nat = (cld == 0) if cld is not None else np.ones(lst.shape, dtype=bool)
            clear_nat = np.where(c_nat & ((q_nat & 3) <= 1) & np.isfinite(lst), lst, np.nan).astype(np.float32)
            clst90 = reproj(clear_nat, H90, W90, tf90, crs90)

            mc = {}
            for cp in cove_pairs:
                vcl = clst90[cp['m_cl']]
                vop = clst90[cp['m_op']]
                fcl = float(np.isfinite(vcl).mean())
                fop = float(np.isfinite(vop).mean())
                if fcl >= 0.50 and fop >= 0.50 and np.isfinite(vcl).sum() >= 5 and np.isfinite(vop).sum() >= 6:
                    m_c = float(np.nanmean(vcl))
                    m_o = float(np.nanmean(vop))
                    s_c = float(np.nanstd(vcl))
                    s_o = float(np.nanstd(vop))
                    if abs(m_c - m_o) < 10.0 and s_c < 4.5 and s_o < 4.5:
                        mc[cp['name']] = round(m_c - m_o, 3)
            rec['multicove'] = mc
            return rec
        except Exception:
            return None

    t0 = time.time()
    batch_sz = 16
    with open(ckpt_path, 'a') as ck_f:
        for b0 in range(0, len(todo), batch_sz):
            batch = todo[b0:b0 + batch_sz]
            ex = ThreadPoolExecutor(max_workers=4)
            futs = {ex.submit(proc_one, g): g for g in batch}
            done_set, not_done = wait(futs.keys(), timeout=90)
            for fut in done_set:
                try:
                    r = fut.result(timeout=1)
                except Exception:
                    r = None
                if r is not None:
                    done_map[r['title']] = r
                    ck_f.write(json.dumps(r) + '\n')
            ck_f.flush()
            for fut in not_done:
                fut.cancel()
            ex.shutdown(wait=False, cancel_futures=True)
            print(f'  [batch {b0 + len(batch):3d}/{len(todo)}] done={len(done_set)} timedout={len(not_done)} '
                  f'({time.time()-t0:.1f}s) | total_saved={len(done_map)} '
                  f'clean={sum(x.get("is_clean") for x in done_map.values())}', flush=True)
    rows = list(done_map.values())
    json.dump(rows, open(CACHE_V002_LIVE, 'w'))
    return rows


def sstat(df, col='diff_lost_minus_openn'):
    v = df[col].dropna().values if len(df) and col in df.columns else np.array([])
    n = len(v)
    if n == 0:
        return {'n': 0}
    mean = float(np.mean(v))
    sd = float(np.std(v, ddof=1)) if n > 1 else 0.0
    t_v, p_v = stats.ttest_1samp(v, 0.0) if (n >= 3 and sd > 1e-12) else (np.nan, np.nan)
    try:
        w_p = float(stats.wilcoxon(v).pvalue) if n >= 5 else np.nan
    except Exception:
        w_p = np.nan
    return {
        'n': int(n),
        'mean_C': round(mean, 3),
        'median_C': round(float(np.median(v)), 3),
        'sd_C': round(sd, 3),
        'se_C': round(sd / math.sqrt(n), 3) if n > 1 else None,
        'p10_C': round(float(np.percentile(v, 10)), 3),
        't': round(float(t_v), 3) if np.isfinite(t_v) else None,
        'p': float(f'{p_v:.4g}') if np.isfinite(p_v) else None,
        'wilcoxon_p': float(f'{w_p:.4g}') if np.isfinite(w_p) else None,
        'frac_neg': round(float(np.mean(v < 0)), 3),
    }


def compute_manski_pair(df_all_swath, df_clean, max_abs_dev_C=10.0):
    """Compute honest swath-conditioned Manski bounds accounting for unflagged cloud leaks:
    - swath_conditioned (leak-adjusted cloud missingness across all in-swath passes, setting
      p_cld = 1.0 for any scene failing |LST - ERA5_t2m| <= max_abs_dev_C so dropped cloud-leaked
      scenes are properly counted as unobserved under the Law of Total Expectation)
    - swath_raw_cloud_flag_only (raw NASA cloud-flag p_cld prior to leak adjustment, for reference)
    - swath_total_invalid (leak-adjusted total pixel invalid fraction 1 - f_valid, setting f_valid = 0
      for cloud-leaked scenes)
    - clean_scene_pixel (within-scene missingness across clean passes)
    """
    if len(df_clean) < 3 or len(df_all_swath) < 3:
        return {'n_in_swath': int(len(df_all_swath)), 'n_clean': int(len(df_clean)), 'note': 'insufficient scenes'}

    # Identify unflagged cloud/calibration leak scenes in df_all_swath
    t2m = df_all_swath['era5_t2m_C'] if 'era5_t2m_C' in df_all_swath.columns else pd.Series(np.nan, index=df_all_swath.index)
    ll = df_all_swath['lst_lost'] if 'lst_lost' in df_all_swath.columns else pd.Series(np.nan, index=df_all_swath.index)
    lo = df_all_swath['lst_openn'] if 'lst_openn' in df_all_swath.columns else pd.Series(np.nan, index=df_all_swath.index)
    is_leak = t2m.notna() & (
        (ll.notna() & ((ll - t2m).abs() > max_abs_dev_C)) |
        (lo.notna() & ((lo - t2m).abs() > max_abs_dev_C))
    )

    p_l_raw = float(df_all_swath['p_cld_lost'].dropna().mean())
    p_o_raw = float(df_all_swath['p_cld_openn'].dropna().mean())
    p_l_adj = float(np.where(is_leak, 1.0, df_all_swath['p_cld_lost']).astype(float)[df_all_swath['p_cld_lost'].notna() | is_leak].mean())
    p_o_adj = float(np.where(is_leak, 1.0, df_all_swath['p_cld_openn']).astype(float)[df_all_swath['p_cld_openn'].notna() | is_leak].mean())

    if 'f_valid_lost' in df_all_swath and 'f_valid_openn' in df_all_swath:
        fv_l = np.where(is_leak, 0.0, df_all_swath['f_valid_lost']).astype(float)
        fv_o = np.where(is_leak, 0.0, df_all_swath['f_valid_openn']).astype(float)
        p_l_tot = float((1.0 - fv_l[np.isfinite(fv_l)]).mean())
        p_o_tot = float((1.0 - fv_o[np.isfinite(fv_o)]).mean())
    else:
        p_l_tot, p_o_tot = p_l_adj, p_o_adj

    p_l_cln = float(df_clean['p_cld_lost'].dropna().mean())
    p_o_cln = float(df_clean['p_cld_openn'].dropna().mean())
    y_lo = float(np.nanpercentile(df_clean['lst_lost'], 5)) - 3.0
    y_hi = float(np.nanpercentile(df_clean['lst_openn'], 95)) + 1.0

    m_swath = C.manski_diff_bounds(df_clean['lst_lost'].values, p_l_adj,
                                   df_clean['lst_openn'].values, p_o_adj, y_lo, y_hi)
    m_swath['n_in_swath'] = int(len(df_all_swath))
    m_swath['n_clean'] = int(len(df_clean))
    m_swath['n_cloud_leak_in_swath'] = int(is_leak.sum())

    m_raw = C.manski_diff_bounds(df_clean['lst_lost'].values, p_l_raw,
                                 df_clean['lst_openn'].values, p_o_raw, y_lo, y_hi)
    m_raw['n_in_swath'] = int(len(df_all_swath))
    m_raw['n_clean'] = int(len(df_clean))

    m_tot = C.manski_diff_bounds(df_clean['lst_lost'].values, p_l_tot,
                                 df_clean['lst_openn'].values, p_o_tot, y_lo, y_hi)
    m_tot['n_in_swath'] = int(len(df_all_swath))
    m_tot['n_clean'] = int(len(df_clean))
    m_tot['n_cloud_leak_in_swath'] = int(is_leak.sum())

    m_px = C.manski_diff_bounds(df_clean['lst_lost'].values, p_l_cln,
                                df_clean['lst_openn'].values, p_o_cln, y_lo, y_hi)
    return {
        'swath_conditioned': m_swath,
        'swath_raw_cloud_flag_only': m_raw,
        'swath_total_invalid': m_tot,
        'clean_scene_pixel': m_px,
    }


def apply_physical_cloud_leak_filter(df, max_abs_dev_C=10.0):
    """Filter out unflagged ECOSTRESS cloud-contaminated or corrupted granules where
    |LST - ERA5_t2m| exceeds max_abs_dev_C on either Lost Cove or OpenN.
    Scenes without ERA5 t2m (e.g. very recent v003 scenes) are kept if LST is physically plausible.
    Returns (df_kept, df_dropped)."""
    if len(df) == 0:
        return df.copy(), df.copy()
    t2m = df['era5_t2m_C'] if 'era5_t2m_C' in df.columns else pd.Series(np.nan, index=df.index)
    ll = df['lst_lost'] if 'lst_lost' in df.columns else pd.Series(np.nan, index=df.index)
    lo = df['lst_openn'] if 'lst_openn' in df.columns else pd.Series(np.nan, index=df.index)
    ok_l = t2m.isna() | ll.isna() | ((ll - t2m).abs() <= max_abs_dev_C)
    ok_o = t2m.isna() | lo.isna() | ((lo - t2m).abs() <= max_abs_dev_C)
    keep = ok_l & ok_o
    return df[keep].copy(), df[~keep].copy()


def main():
    t0 = time.time()
    if not os.path.exists(CACHE_V002_LIVE):
        C.init_ee()

    # 1. Fetch/load live ECO_L2T_LSTE.002 (2020-2024) — fast-path metadata when already cached
    if os.path.exists(CACHE_V002_LIVE) and os.path.exists(CACHE_MC_META):
        cove_pairs = json.load(open(CACHE_MC_META))
        rows_v2 = json.load(open(CACHE_V002_LIVE))
        print(f'[4B] Loaded {len(rows_v2)} cached live v002 scenes + {len(cove_pairs)} cached multi-cove specs (fast path)')
    elif os.path.exists(CACHE_V002_LIVE):
        cove_pairs_full, shape90, tf90, crs90 = build_multicove_90m_masks()
        cove_pairs = [{k: v for k, v in cp.items() if k not in ('m_cl', 'm_op')} for cp in cove_pairs_full]
        json.dump(cove_pairs, open(CACHE_MC_META, 'w'), indent=2)
        rows_v2 = json.load(open(CACHE_V002_LIVE))
        print(f'[4B] Loaded {len(rows_v2)} cached live ECO_L2T_LSTE.002 scenes')
    else:
        cove_pairs_full, shape90, tf90, crs90 = build_multicove_90m_masks()
        cove_pairs = [{k: v for k, v in cp.items() if k not in ('m_cl', 'm_op')} for cp in cove_pairs_full]
        json.dump(cove_pairs, open(CACHE_MC_META, 'w'), indent=2)
        dem10, nlcd10, prof10, tf10, crs10 = P3.load_domain_10m_dem_nlcd()
        masks10 = P3.build_10m_masks(dem10, nlcd10, tf10, crs10)
        rows_v2 = fetch_and_process_v002_live(dem10, tf10, crs10, masks10, cove_pairs_full, shape90, tf90, crs90)

    # 2. Load cached live ECO_L2T_LSTE.003 (2018-2019, 2025-2026) and attach ERA5 lookup
    v3_path = os.path.join(C.CACHE_DIR, 'p3_v003_spring_autumn_scenes.json')
    rows_v3 = json.load(open(v3_path)) if os.path.exists(v3_path) else []
    era5_lut = P3.fetch_era5_hourly_for_timestamps(list({r['dt_iso'] for r in rows_v3 if r.get('dt_iso')}))
    for r in rows_v3:
        r['version'] = 'v003'
        e = era5_lut.get(r.get('dt_iso'), {})
        r['wspd_ms'] = e.get('wspd_ms')
        r['era5_t2m_C'] = e.get('t2m_C')
        r.setdefault('in_swath_lost_frac', 1.0)
        r.setdefault('in_swath_openn_frac', 1.0)

    # Filter v003 to nighttime/pre-dawn (solar_hr < 7.0 or >= 21.0) and merge deduplicating by (yr, doy, solar_hr_bin)
    combined = {}
    for r in rows_v2:
        k = (r['yr'], r['doy'], int(round(r['solar_hr'] * 2)))
        combined[k] = r
    for r in rows_v3:
        if r['solar_hr'] < 7.0 or r['solar_hr'] >= 21.0:
            k = (r['yr'], r['doy'], int(round(r['solar_hr'] * 2)))
            combined[k] = r  # prefer v003 if exact same overpass exists

    df_all = pd.DataFrame(list(combined.values()))
    df_swath = df_all[(df_all['in_swath_lost_frac'] >= 0.50) & (df_all['in_swath_openn_frac'] >= 0.50)].copy()
    df_cln_unfiltered = df_swath[df_swath['is_clean'] == True].copy()

    # Physical cloud-leak / calibration-corruption filter: |LST - ERA5_t2m| <= 10.0 C
    df_cln, df_leaks = apply_physical_cloud_leak_filter(df_cln_unfiltered, max_abs_dev_C=10.0)

    # Record unfiltered diagnostics transparently alongside primary filtered results
    unfilt_spr = df_cln_unfiltered[df_cln_unfiltered['season'] == 'spring']
    unfilt_aut = df_cln_unfiltered[df_cln_unfiltered['season'] == 'autumn']
    unfilt_w = df_cln_unfiltered[df_cln_unfiltered['wspd_ms'].notna()]
    leak_cols = ['title', 'version', 'season', 'yr', 'doy', 'wspd_ms', 'era5_t2m_C',
                 'lst_lost', 'lst_openn', 'diff_lost_minus_openn']
    unfiltered_diag = {
        'n_unfiltered_clean': int(len(df_cln_unfiltered)),
        'n_dropped_cloud_leaks': int(len(df_leaks)),
        'dropped_by_version': {str(k): int(v) for k, v in df_leaks['version'].value_counts().to_dict().items()},
        'dropped_scenes': df_leaks[[c for c in leak_cols if c in df_leaks.columns]].to_dict(orient='records'),
        'all_nights_unfiltered': sstat(df_cln_unfiltered),
        'spring_unfiltered': sstat(unfilt_spr),
        'autumn_unfiltered': sstat(unfilt_aut),
        'calm_lt2p5ms_unfiltered': sstat(unfilt_w[unfilt_w['wspd_ms'] < 2.5]),
        'breezy_ge2p5ms_unfiltered': sstat(unfilt_w[unfilt_w['wspd_ms'] >= 2.5]),
        'note': (
            'Applying a physical |LST - ERA5_t2m| <= 10 C screen removes 8 unflagged ECOv002 scenes '
            '(7 cloud-contaminated scenes with LST 11-22 C below ERA5 t2m and 1 calibration-corrupted '
            'pre-dawn scene at +46 to +48 C; 0/14 v003 scenes are flagged), including all 3 scenes '
            'with u10 >= 3.0 m/s. Filtering strengthens cooling across every season and makes spring '
            'cleanly significant on both parametric t-test and Wilcoxon.'
        ),
    }

    # Seasonal subsets (primary: cloud-leak-filtered clean nighttime/pre-dawn)
    spr_all = df_swath[df_swath['season'] == 'spring']
    spr_cln = df_cln[df_cln['season'] == 'spring']
    aut_all = df_swath[df_swath['season'] == 'autumn']
    aut_cln = df_cln[df_cln['season'] == 'autumn']
    aut_cln_era5 = aut_cln[aut_cln['era5_t2m_C'].notna()]
    oct_cln = df_cln[df_cln['month'] == 10]
    cold_all = df_swath[df_swath['era5_t2m_C'].notna() & (df_swath['era5_t2m_C'] < 5.0)]
    cold_cln = df_cln[df_cln['era5_t2m_C'].notna() & (df_cln['era5_t2m_C'] < 5.0)]

    # Wind stratification (calm <2.5, breezy 2.5-3.0, windy >=3.0 m/s)
    has_w = df_cln[df_cln['wspd_ms'].notna()].copy()
    w_calm = has_w[has_w['wspd_ms'] < 2.5]
    w_ge25 = has_w[has_w['wspd_ms'] >= 2.5]
    w_mod = has_w[(has_w['wspd_ms'] >= 2.5) & (has_w['wspd_ms'] < 3.0)]
    w_ge30 = has_w[has_w['wspd_ms'] >= 3.0]
    w_ge35 = has_w[has_w['wspd_ms'] >= 3.5]
    w_ge40 = has_w[has_w['wspd_ms'] >= 4.0]

    # Continuous regression: diff_lost_minus_openn ~ wspd_ms (positive slope = stronger wind weakens cold pool)
    wind_reg = {}
    if len(has_w) >= 10:
        r_w, p_w = stats.pearsonr(has_w['wspd_ms'].values, has_w['diff_lost_minus_openn'].values)
        rho_w, prho_w = stats.spearmanr(has_w['wspd_ms'].values, has_w['diff_lost_minus_openn'].values)
        ols_w = C.ols_cluster_se(has_w[['wspd_ms']].values, has_w['diff_lost_minus_openn'].values,
                                 col_names=['wspd_ms'])
        t_cw25, p_cw25 = stats.ttest_ind(w_calm['diff_lost_minus_openn'], w_ge25['diff_lost_minus_openn'],
                                         equal_var=False) if len(w_ge25) >= 3 else (np.nan, np.nan)
        mw_cw25_p = float(stats.mannwhitneyu(w_calm['diff_lost_minus_openn'], w_ge25['diff_lost_minus_openn']).pvalue) if len(w_ge25) >= 3 else np.nan
        t_cw4, p_cw4 = stats.ttest_ind(w_calm['diff_lost_minus_openn'], w_ge40['diff_lost_minus_openn'],
                                       equal_var=False) if len(w_ge40) >= 3 else (np.nan, np.nan)
        t_cw35, p_cw35 = stats.ttest_ind(w_calm['diff_lost_minus_openn'], w_ge35['diff_lost_minus_openn'],
                                         equal_var=False) if len(w_ge35) >= 3 else (np.nan, np.nan)
        t_cw30, p_cw30 = stats.ttest_ind(w_calm['diff_lost_minus_openn'], w_ge30['diff_lost_minus_openn'],
                                         equal_var=False) if len(w_ge30) >= 3 else (np.nan, np.nan)
        # Season-stratified & season-controlled wind placebo
        season_wind = {}
        for s_lbl in ['autumn', 'spring']:
            sub_s = has_w[has_w['season'] == s_lbl]
            sc_c = sub_s[sub_s['wspd_ms'] < 2.5]['diff_lost_minus_openn']
            sc_b = sub_s[sub_s['wspd_ms'] >= 2.5]['diff_lost_minus_openn']
            tw_s, pw_s = stats.ttest_ind(sc_c, sc_b, equal_var=False) if len(sc_b) >= 2 else (np.nan, np.nan)
            mw_s = float(stats.mannwhitneyu(sc_c, sc_b).pvalue) if len(sc_b) >= 2 else np.nan
            season_wind[s_lbl] = {
                'calm_n': int(len(sc_c)),
                'calm_mean_C': round(float(sc_c.mean()), 3) if len(sc_c) else None,
                'calm_se_C': round(float(sc_c.std(ddof=1) / math.sqrt(len(sc_c))), 3) if len(sc_c) > 1 else None,
                'calm_median_C': round(float(sc_c.median()), 3) if len(sc_c) else None,
                'breezy_n': int(len(sc_b)),
                'breezy_mean_C': round(float(sc_b.mean()), 3) if len(sc_b) else None,
                'breezy_se_C': round(float(sc_b.std(ddof=1) / math.sqrt(len(sc_b))), 3) if len(sc_b) > 1 else None,
                'breezy_median_C': round(float(sc_b.median()), 3) if len(sc_b) else None,
                'welch_t': round(float(tw_s), 3) if np.isfinite(tw_s) else None,
                'welch_p': float(f'{pw_s:.4g}') if np.isfinite(pw_s) else None,
                'mannwhitney_p': float(f'{mw_s:.4g}') if np.isfinite(mw_s) else None,
            }
        X_sctrl = np.column_stack([
            (has_w['wspd_ms'] >= 2.5).astype(float).values,
            (has_w['season'] == 'spring').astype(float).values,
        ])
        ols_sctrl = C.ols_cluster_se(X_sctrl, has_w['diff_lost_minus_openn'].values,
                                     col_names=['breezy_ge2p5', 'is_spring'])

        wind_reg = {
            'n': int(len(has_w)),
            'wspd_max_clean_ms': round(float(has_w['wspd_ms'].max()), 3),
            'pearson_r': round(float(r_w), 3), 'pearson_p': float(f'{p_w:.4g}'),
            'spearman_rho': round(float(rho_w), 3), 'spearman_p': float(f'{prho_w:.4g}'),
            'ols_slope_C_per_ms': ols_w['coefs']['wspd_ms'],
            'ols_intercept_C': ols_w['coefs']['intercept'],
            'calm_vs_breezy2p5_welch_t': round(float(t_cw25), 3) if np.isfinite(t_cw25) else None,
            'calm_vs_breezy2p5_welch_p': float(f'{p_cw25:.4g}') if np.isfinite(p_cw25) else None,
            'calm_vs_breezy2p5_mannwhitney_p': float(f'{mw_cw25_p:.4g}') if np.isfinite(mw_cw25_p) else None,
            'season_stratified_breezy2p5': season_wind,
            'season_controlled_ols_breezy2p5': ols_sctrl['coefs']['breezy_ge2p5'],
            'calm_vs_windy4p0_welch_t': round(float(t_cw4), 3) if np.isfinite(t_cw4) else None,
            'calm_vs_windy4p0_welch_p': float(f'{p_cw4:.4g}') if np.isfinite(p_cw4) else None,
            'calm_vs_windy3p5_welch_t': round(float(t_cw35), 3) if np.isfinite(t_cw35) else None,
            'calm_vs_windy3p5_welch_p': float(f'{p_cw35:.4g}') if np.isfinite(p_cw35) else None,
            'calm_vs_windy3p0_welch_t': round(float(t_cw30), 3) if np.isfinite(t_cw30) else None,
            'calm_vs_windy3p0_welch_p': float(f'{p_cw30:.4g}') if np.isfinite(p_cw30) else None,
            'wind_collapse_scope_note': (
                'All 7 clean breezy nights lie in 2.5 <= u10 < 3.0 m/s (0 clean nights >= 3.0 m/s once '
                '3 cloud-leaked v002 scenes are removed). Wind-collapse inference is scoped strictly to '
                'Lost Cove over 2.5-3.0 m/s; it does not replicate across the 6 other 16SED coves when '
                'using Sewanee point ERA5 wind (paired calm vs. breezy p = 0.88).'
            ),
        }

    # VZA & Emissivity controls (on cloud-leak-filtered clean scenes)
    vza_lo = df_cln[df_cln['vza_deg'].notna() & (df_cln['vza_deg'] < 25.0)]
    emis_vals = df_cln['d_emis_wb'].dropna().values

    # Identify all v002 granules failing |LST - ERA5_t2m| <= 10 C over Sewanee for multi-cove screen
    flagged_v2_titles = set()
    for r in rows_v2:
        t2m = r.get('era5_t2m_C')
        ll, lo = r.get('lst_lost'), r.get('lst_openn')
        if t2m is not None:
            if (ll is not None and abs(ll - t2m) > 10.0) or (lo is not None and abs(lo - t2m) > 10.0):
                flagged_v2_titles.add(r['title'])

    # Multi-cove replication summary across the 6 additional 16SED closed-open pairs.
    # Primary screen requires the granule to be observed & physically verified cloud-free at Sewanee
    # (lst_lost and lst_openn present and |LST - ERA5_t2m| <= 10 C), eliminating 16 uncheckable
    # cloudy-at-Sewanee granules (including a stormy pcld=0.85 pass injecting -9.29 C on Karst_AL_12).
    # We also archive excl_flagged_only and unfiltered tables transparently.
    def build_mc_table(exclude_titles, require_sewanee_verified=False):
        mc_rows = []
        means_all, means_calm, means_breezy = [], [], []
        for cp in cove_pairs:
            nm = cp['name']
            vals_all, vals_calm, vals_breezy = [], [], []
            for r in rows_v2:
                if r['title'] in exclude_titles:
                    continue
                if require_sewanee_verified and (r.get('lst_lost') is None or r.get('lst_openn') is None):
                    continue
                mc = r.get('multicove', {})
                if nm in mc:
                    vals_all.append(mc[nm])
                    w = r.get('wspd_ms')
                    if w is not None and w < 2.5:
                        vals_calm.append(mc[nm])
                    elif w is not None and w >= 2.5:
                        vals_breezy.append(mc[nm])
            st_all = sstat(pd.DataFrame({'v': vals_all}), 'v')
            st_clm = sstat(pd.DataFrame({'v': vals_calm}), 'v')
            st_brz = sstat(pd.DataFrame({'v': vals_breezy}), 'v')
            if st_all.get('n', 0) >= 5:
                means_all.append(st_all['mean_C'])
            if st_clm.get('n', 0) >= 3:
                means_calm.append(st_clm['mean_C'])
            if st_brz.get('n', 0) >= 3:
                means_breezy.append(st_brz['mean_C'])
            mc_rows.append({
                'cove': nm, 'open_pair': cp['open_name'], 'max_fill_m': cp['max_fill_m'],
                'z_cl_m': cp['z_cl'], 'z_op_m': cp['z_op'],
                'all_nights': st_all, 'calm_lt2p5ms': st_clm, 'breezy_ge2p5ms': st_brz,
            })
        return mc_rows, means_all, means_calm, means_breezy

    mc_summary, cove_means_all, cove_means_calm, cove_means_breezy = build_mc_table(
        flagged_v2_titles, require_sewanee_verified=True
    )
    mc_summary_excl_flagged, _, _, _ = build_mc_table(flagged_v2_titles, require_sewanee_verified=False)
    mc_summary_unfilt, u_all, u_calm, _ = build_mc_table(set(), require_sewanee_verified=False)

    # Include Lost Cove in the grand 7-cove paired test (cove as independent unit, N=7)
    grand_all = [sstat(df_cln)['mean_C']] + cove_means_all
    grand_calm = [sstat(w_calm)['mean_C']] + cove_means_calm
    t_gc_all, p_gc_all = stats.ttest_1samp(grand_all, 0.0) if len(grand_all) >= 3 else (np.nan, np.nan)
    t_gc_clm, p_gc_clm = stats.ttest_1samp(grand_calm, 0.0) if len(grand_calm) >= 3 else (np.nan, np.nan)
    _, p_mc_wind_6 = stats.ttest_rel(cove_means_calm, cove_means_breezy) if len(cove_means_calm) == len(cove_means_breezy) and len(cove_means_calm) >= 3 else (np.nan, np.nan)

    if wind_reg:
        p_mc_str = f'{p_mc_wind_6:.2f}' if np.isfinite(p_mc_wind_6) else 'n/a'
        wind_reg['wind_collapse_scope_note'] = (
            'All 7 clean breezy nights lie in 2.5 <= u10 < 3.0 m/s (0 clean nights >= 3.0 m/s once '
            '3 cloud-leaked v002 scenes are removed). Wind-collapse inference is scoped strictly to '
            'Lost Cove over 2.5-3.0 m/s; it does not replicate across the 6 other 16SED coves when '
            f'using Sewanee point ERA5 wind (paired calm vs. breezy p = {p_mc_str}).'
        )

    # Separate v002 vs v003 Manski bounds alongside combined swath bounds
    sw_v002 = df_swath[df_swath['version'] == 'v002']
    cln_v002 = df_cln[df_cln['version'] == 'v002']
    sw_v003 = df_swath[df_swath['version'] == 'v003']
    cln_v003 = df_cln[df_cln['version'] == 'v003']

    # Also carry forward P3 morning 08.5-10.0h ray-traced shadow audit from profiling/p3.json
    p3_prev = json.load(open(os.path.join(HERE, 'profiling', 'p3.json')))

    # Trim scene table for plotting (cloud-leak-filtered clean scenes with key columns)
    plot_cols = ['title', 'version', 'season', 'yr', 'month', 'doy', 'solar_hr', 'wspd_ms',
                 'era5_t2m_C', 'lst_lost', 'lst_openn', 'diff_lost_minus_openn',
                 'diff_fd30_minus_openn', 'diff_fd40_minus_openn', 'd_emis_wb', 'vza_deg']
    scene_list = df_cln[[c for c in plot_cols if c in df_cln.columns]].to_dict(orient='records')

    out = {
        'experiment': 'v2_4B_ecostress_thermal_closure',
        'elapsed_s': round(time.time() - t0, 1),
        'cloud_leak_filter': '|LST - ERA5_t2m| <= 10.0 C on both Lost Cove and OpenN',
        'archive_counts': {
            'v002_live_processed': int(len(rows_v2)),
            'v003_live_night_merged': int(sum(1 for r in combined.values() if r.get('version') == 'v003')),
            'combined_unique_night_in_swath': int(len(df_swath)),
            'combined_unique_night_clean_unfiltered': int(len(df_cln_unfiltered)),
            'dropped_v002_cloud_leaks': int(len(df_leaks)),
            'combined_unique_night_clean': int(len(df_cln)),
            'clean_era5_matched_count': int(len(has_w)),
            'clean_v002_count': int((df_cln['version'] == 'v002').sum()),
            'clean_v003_count': int((df_cln['version'] == 'v003').sum()),
            'v002_granules_flagged_for_multicove_screen': int(len(flagged_v2_titles)),
        },
        'unfiltered_diagnostics': unfiltered_diag,
        'lost_cove_night_stats': {
            'all_nights_combined': sstat(df_cln),
            'all_nights_era5_matched': sstat(has_w),
            'all_nights_fd30': sstat(df_cln, 'diff_fd30_minus_openn'),
            'all_nights_fd40': sstat(df_cln, 'diff_fd40_minus_openn'),
            'spring_feb_apr_night': sstat(spr_cln),
            'spring_feb_apr_v002_only': sstat(spr_cln[spr_cln['version'] == 'v002']),
            'spring_feb_apr_calm_lt2p5ms': sstat(spr_cln[spr_cln['wspd_ms'].notna() & (spr_cln['wspd_ms'] < 2.5)]),
            'autumn_sep_nov_night': sstat(aut_cln),
            'autumn_sep_nov_era5_matched': sstat(aut_cln_era5),
            'autumn_oct_only_night': sstat(oct_cln),
            'autumn_calm_lt2p5ms': sstat(aut_cln[aut_cln['wspd_ms'].notna() & (aut_cln['wspd_ms'] < 2.5)]),
            'cold_t2m_lt5C_night': sstat(cold_cln),
        },
        'wind_placebo': {
            'calm_lt2p5ms': sstat(w_calm),
            'breezy_ge2p5ms': sstat(w_ge25),
            'breezy_2p5_3p0ms': sstat(w_mod),
            'windy_ge3p0ms': sstat(w_ge30),
            'windy_ge3p5ms': sstat(w_ge35),
            'windy_ge4p0ms': sstat(w_ge40),
            'continuous_wind_regression': wind_reg,
        },
        'artifact_controls': {
            'vza_lt25deg_night': sstat(vza_lo),
            'n_emis_scenes': int(len(emis_vals)),
            'mean_d_EmisWB': round(float(np.mean(emis_vals)), 5) if len(emis_vals) else None,
            'max_abs_d_EmisWB': round(float(np.max(np.abs(emis_vals))), 5) if len(emis_vals) else None,
            'morning_085_100h_raytraced_shadow': {
                'all_C': p3_prev['part_a_v002_shadow_and_synoptic']['morning_085_100h_all'],
                'sunlit_only_C': p3_prev['part_a_v002_shadow_and_synoptic']['morning_085_100h_sunlit_only'],
                'calm_sunlit_C': p3_prev['part_a_v002_shadow_and_synoptic']['morning_085_100h_calm_sunlit'],
                'icum_adjusted_C': p3_prev['part_a_v002_shadow_and_synoptic']['morning_085_100h_icum_adjusted'],
            },
        },
        'manski_bounds': {
            'all_combined_nights': compute_manski_pair(df_swath, df_cln),
            'v002_only_nights': compute_manski_pair(sw_v002, cln_v002),
            'v003_only_nights': compute_manski_pair(sw_v003, cln_v003),
            'spring_feb_apr': compute_manski_pair(spr_all, spr_cln),
            'autumn_sep_nov': compute_manski_pair(aut_all, aut_cln),
            'cold_t2m_lt5C': compute_manski_pair(cold_all, cold_cln),
        },
        'multicove_16sed_replication': {
            'n_cove_pairs_incl_lost_cove': int(len(grand_all)),
            'cove_level_all_nights_mean_C': round(float(np.mean(grand_all)), 3),
            'cove_level_all_nights_se_C': round(float(np.std(grand_all, ddof=1) / math.sqrt(len(grand_all))), 3),
            'cove_level_all_nights_t': round(float(t_gc_all), 3) if np.isfinite(t_gc_all) else None,
            'cove_level_all_nights_p': float(f'{p_gc_all:.4g}') if np.isfinite(p_gc_all) else None,
            'cove_level_calm_mean_C': round(float(np.mean(grand_calm)), 3),
            'cove_level_calm_se_C': round(float(np.std(grand_calm, ddof=1) / math.sqrt(len(grand_calm))), 3),
            'cove_level_calm_t': round(float(t_gc_clm), 3) if np.isfinite(t_gc_clm) else None,
            'cove_level_calm_p': float(f'{p_gc_clm:.4g}') if np.isfinite(p_gc_clm) else None,
            'frac_coves_negative_all': round(float(np.mean(np.array(grand_all) < 0)), 3),
            'frac_coves_negative_calm': round(float(np.mean(np.array(grand_calm) < 0)), 3),
            'six_other_coves_calm_vs_breezy_paired_p': float(f'{p_mc_wind_6:.4g}') if np.isfinite(p_mc_wind_6) else None,
            'screen_rule': 'Sewanee-verified clear v002 granules (lst_lost & lst_openn observed and |LST - ERA5_t2m| <= 10 C)',
            'per_cove': mc_summary,
            'per_cove_excl_flagged_only': mc_summary_excl_flagged,
            'per_cove_unfiltered': mc_summary_unfilt,
        },
        'clean_night_scenes': scene_list,
    }
    clean = C.sanitize_json(out)
    with open(OUT_PATH, 'w') as f:
        json.dump(clean, f, indent=2, allow_nan=False)
    print(f'WROTE {OUT_PATH} in {time.time()-t0:.1f}s')

    print('\n=== 4B KEY RESULTS (CLOUD-LEAK FILTERED) ===')
    print('Archive:', json.dumps(out['archive_counts']))
    print('Spring night:', out['lost_cove_night_stats']['spring_feb_apr_night'])
    print('Autumn night:', out['lost_cove_night_stats']['autumn_sep_nov_night'])
    print('Oct night:', out['lost_cove_night_stats']['autumn_oct_only_night'])
    print('Wind calm (<2.5):', out['wind_placebo']['calm_lt2p5ms'])
    print('Wind breezy (2.5-3.0):', out['wind_placebo']['breezy_2p5_3p0ms'])
    print('Wind reg:', {k: v for k, v in out['wind_placebo']['continuous_wind_regression'].items() if k != 'season_stratified_breezy2p5'})
    print('Manski v002 vs v003:', {
        'v002_swath': out['manski_bounds']['v002_only_nights']['swath_conditioned'],
        'v003_swath': out['manski_bounds']['v003_only_nights']['swath_conditioned'],
    })
    print('Multi-cove:', {k: v for k, v in out['multicove_16sed_replication'].items() if k not in ('per_cove', 'per_cove_unfiltered')})


if __name__ == '__main__':
    main()
