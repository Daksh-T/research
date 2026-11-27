"""Experiment P1: Regional Forested Karst Census & Power across Southern Cumberland Plateau.

Steps:
1. Download & cache 90m 3DEP DEM + NLCD 2021 across Southern Cumberland Plateau (TN/AL).
2. Run offline exact Priority-Flood (morphological reconstruction) -> FillDepth.
3. Detect all closed karst depressions (FillDepth >= 25m, area >= 0.35 km2), verify at 30m 3DEP,
   filter forested floors (NLCD-41 >= 60%, water < 5%, dev/barren < 15%), and log disqualified uvalas.
4. Automatically pair each forested closed basin 1:1 with a nearby elevation-matched open-drainage gorge.
5. Extract per-year (2018-2024) Sentinel-2 (strict SCL=4|5) NDVI & NDRE time series per stratum
   (floor, mid, rim) in GEE and fit 6-parameter double-logistic phenology with QC gates.
6. Run cove-level mixed-effects / paired tests, check r(sigma_SOS, FillDepth), and diagnose Swagerty Cove.
"""
import os
import sys
import json
import math
import time
import requests
import numpy as np
import pandas as pd
import rasterio
from scipy import stats, ndimage
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ee
import common as C

# Southern Cumberland Plateau bounding box (northern AL to central TN karst belt)
BBOX = [-86.70, 34.25, -84.95, 36.15]

KNOWN_COVES = [
    ('Lost_Cove_TN', 35.145, -85.910),
    ('Sinking_Cove_TN', 35.012, -85.963),
    ('Swagerty_Cove_TN', 35.870, -85.370),
    ('Grassy_Cove_TN', 35.833, -85.083),
    ('Crab_Orchard_TN', 35.910, -84.880),
    ('Newsome_Sinks_AL', 34.432, -86.600),
    ('Sauta_Cave_Karst_AL', 34.618, -86.130),
    ('Mink_Cove_AL', 34.720, -86.100),
    ('Cagle_Cove_TN', 35.510, -85.460),
    ('Fiery_Gizzard_OPEN', 35.245, -85.755),
    ('Savage_Gulf_OPEN', 35.448, -85.565),
]


def load_regional_dem_nlcd():
    cache_tif = os.path.join(C.CACHE_DIR, 'plateau_3dep_nlcd_90m.tif')
    if not os.path.exists(cache_tif):
        print('[P1] Downloading 90m 3DEP + NLCD 2021 over Southern Cumberland Plateau...')
        region = ee.Geometry.Rectangle(BBOX)
        dem = ee.Image('USGS/3DEP/10m').select('elevation')
        nlcd = C.get_nlcd_2021()
        url = dem.addBands(nlcd).getDownloadURL({
            'region': region, 'scale': 90, 'crs': 'EPSG:4326', 'format': 'GEO_TIFF'
        })
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        with open(cache_tif, 'wb') as f:
            f.write(r.content)
        print(f'  cached {cache_tif} ({len(r.content)/1e6:.2f} MB)')
    with rasterio.open(cache_tif) as ds:
        dem = ds.read(1).astype(np.float64)
        nlcd = ds.read(2).astype(np.int32)
        transform = ds.transform
        bounds = ds.bounds
    dem[dem < -100] = np.nan
    return dem, nlcd, transform, bounds


def rc_to_lonlat(transform, r, c):
    lon, lat = rasterio.transform.xy(transform, r, c)
    return float(lon), float(lat)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(min(1.0, a)))


def match_known_name(lat, lon, idx):
    best_name, best_d = None, 999.0
    for nm, klat, klon in KNOWN_COVES:
        d = haversine_km(lat, lon, klat, klon)
        if d < best_d:
            best_d, best_name = d, nm
    if best_d <= 4.0:
        return best_name
    st = 'AL' if lat < 34.99 else 'TN'
    return f'Karst_{st}_{idx:02d}'


def detect_closed_basins_and_open_pairs(dem, nlcd, transform):
    print('[P1] Running offline Priority-Flood depression fill...')
    t0 = time.time()
    filled, fill_depth = C.priority_flood_fill(dem)
    print(f'  Priority-Flood complete in {time.time()-t0:.2f}s | max FillDepth={np.nanmax(fill_depth):.1f} m')

    # Pixel area in km2 at ~35.2N
    px_lat_m = abs(transform.e) * 111320.0
    px_lon_m = abs(transform.a) * 111320.0 * math.cos(math.radians(35.2))
    px_area_km2 = (px_lat_m * px_lon_m) / 1e6

    # Label closed depressions with FillDepth >= 5m (then require max_fill >= 25m and area >= 0.35 km2)
    mask5 = np.isfinite(fill_depth) & (fill_depth >= 5.0)
    labels, n_feat = ndimage.label(mask5, structure=np.ones((3, 3)))
    print(f'  Raw FillDepth>=5m connected components: {n_feat}')

    candidates = []
    H, W = dem.shape
    for lbl in range(1, n_feat + 1):
        rr, cc = np.where(labels == lbl)
        if len(rr) * px_area_km2 < 0.30:
            continue
        fd = fill_depth[rr, cc]
        max_fd = float(np.max(fd))
        if max_fd < 22.5:
            continue
        # Exclude edge-touching artifacts
        if rr.min() <= 2 or rr.max() >= H - 3 or cc.min() <= 2 or cc.max() >= W - 3:
            continue
        # Deepest point
        deep_i = int(np.argmax(fd))
        dr, dc = int(rr[deep_i]), int(cc[deep_i])
        lon_c, lat_c = rc_to_lonlat(transform, dr, dc)

        # Floor pixels: FillDepth >= max(12.0, 0.50 * max_fd)
        fl_m = fd >= max(12.0, 0.50 * max_fd)
        fl_nlcd = nlcd[rr[fl_m], cc[fl_m]]
        if len(fl_nlcd) < 5:
            continue
        nlcd41_frac = float(np.mean(fl_nlcd == 41))
        forest_all_frac = float(np.mean(np.isin(fl_nlcd, [41, 42, 43])))
        pasture_frac = float(np.mean(np.isin(fl_nlcd, [81, 82])))
        water_frac = float(np.mean(fl_nlcd == 11))
        # Distinguish quarries / heavy development (23, 24, 31) from rural roads / open space (21, 22)
        dev_bar_frac = float(np.mean(np.isin(fl_nlcd, [23, 24, 31])))
        road_open_frac = float(np.mean(np.isin(fl_nlcd, [21, 22])))

        z_vals = dem[rr, cc]
        z_floor_min = float(np.min(z_vals))
        z_floor_med = float(np.median(z_vals[fl_m]))
        z_spill = float(np.median(filled[rr[fl_m], cc[fl_m]]))

        # Check rim elevation in 500-1500m ring around basin
        r0, r1 = max(0, rr.min() - 15), min(H, rr.max() + 16)
        c0, c1 = max(0, cc.min() - 15), min(W, cc.max() + 16)
        sub_lbl = (labels[r0:r1, c0:c1] == lbl)
        ring = ndimage.binary_dilation(sub_lbl, iterations=12) & (~ndimage.binary_dilation(sub_lbl, iterations=3))
        ring_z = dem[r0:r1, c0:c1][ring]
        ring_nlcd = nlcd[r0:r1, c0:c1][ring]
        ring_z_41 = ring_z[(ring_nlcd == 41) & np.isfinite(ring_z)]
        z_rim_med = float(np.median(ring_z_41)) if len(ring_z_41) >= 10 else float(np.nanpercentile(ring_z, 75))

        # Bounding box of depression (expanded slightly for rim sampling)
        lon_min, lat_max = rc_to_lonlat(transform, int(rr.min()), int(cc.min()))
        lon_max, lat_min = rc_to_lonlat(transform, int(rr.max()), int(cc.max()))

        # Convex hull in lon/lat for exact basin containment in GEE
        pts_ll = np.array([rc_to_lonlat(transform, int(r), int(c)) for r, c in zip(rr, cc)])
        if len(pts_ll) >= 4:
            try:
                hull = ConvexHull(pts_ll)
                hull_coords = pts_ll[hull.vertices].tolist()
                hull_coords.append(hull_coords[0])
            except Exception:
                hull_coords = [[lon_min, lat_min], [lon_max, lat_min], [lon_max, lat_max], [lon_min, lat_max], [lon_min, lat_min]]
        else:
            hull_coords = [[lon_min, lat_min], [lon_max, lat_min], [lon_max, lat_max], [lon_min, lat_max], [lon_min, lat_min]]

        status = 'FORESTED_KARST'
        if water_frac >= 0.05 or dev_bar_frac >= 0.15:
            status = 'DISQUALIFIED_WATER_QUARRY'
        elif nlcd41_frac < 0.60:
            status = 'DISQUALIFIED_PASTURE_NONFOREST'

        candidates.append({
            'lbl': int(lbl),
            'lat': round(lat_c, 5), 'lon': round(lon_c, 5),
            'max_fill_m': round(max_fd, 1),
            'area_km2': round(float(len(rr) * px_area_km2), 3),
            'n_floor_px': int(np.sum(fl_m)),
            'floor_nlcd41_frac': round(nlcd41_frac, 3),
            'floor_forest_all_frac': round(forest_all_frac, 3),
            'floor_pasture_frac': round(pasture_frac, 3),
            'floor_water_frac': round(water_frac, 3),
            'floor_dev_bar_frac': round(dev_bar_frac, 3),
            'floor_road_open_frac': round(road_open_frac, 3),
            'z_floor_min': round(z_floor_min, 1),
            'z_floor_med': round(z_floor_med, 1),
            'z_spill': round(z_spill, 1),
            'z_rim_med': round(z_rim_med, 1),
            'status': status,
            'bbox': [round(lon_min, 5), round(lat_min, 5), round(lon_max, 5), round(lat_max, 5)],
            'hull': [[round(x, 5), round(y, 5)] for x, y in hull_coords],
        })

    # Sort candidates by max_fill_m desc
    candidates.sort(key=lambda d: d['max_fill_m'], reverse=True)
    used_names = set()
    for i, c in enumerate(candidates, 1):
        nm = match_known_name(c['lat'], c['lon'], i)
        if nm in used_names:
            nm = f'{nm}_{i}'
        used_names.add(nm)
        c['name'] = nm

    forested = [c for c in candidates if c['status'] == 'FORESTED_KARST']
    disq = [c for c in candidates if c['status'] != 'FORESTED_KARST']
    print(f'  Detected {len(candidates)} closed karst basins (>=25m, >=0.35 km2): '
          f'{len(forested)} FORESTED_KARST, {len(disq)} DISQUALIFIED')
    for c in forested[:20]:
        print(f"    [KEEP] {c['name']:22s} fill={c['max_fill_m']:5.1f}m area={c['area_km2']:5.2f}km2 "
              f"NLCD41={c['floor_nlcd41_frac']*100:4.1f}% z_fl={c['z_floor_med']:.0f}m z_rim={c['z_rim_med']:.0f}m")
    for c in disq[:10]:
        print(f"    [DISQ] {c['name']:22s} fill={c['max_fill_m']:5.1f}m area={c['area_km2']:5.2f}km2 "
              f"NLCD41={c['floor_nlcd41_frac']*100:4.1f}% past={c['floor_pasture_frac']*100:4.1f}% ({c['status']})")

    # --- Detect open-drainage gorges on the same DEM ---
    # TPI at ~1350m radius (15 px)
    valid_dem = np.where(np.isfinite(dem), dem, np.nanmedian(dem))
    tpi = dem - ndimage.uniform_filter(valid_dem, size=31)
    # Open gorge floor mask: FillDepth < 2m, TPI < -55m, NLCD==41, plateau escarpment elevation (180-480m)
    open_mask = (
        np.isfinite(dem) & (fill_depth < 2.0) & (tpi < -55.0)
        & (nlcd == 41) & (dem >= 160.0) & (dem <= 480.0)
    )
    # Compute local 7x7 forest density to ensure forested gorge floor
    f41_smooth = ndimage.uniform_filter((nlcd == 41).astype(np.float64), size=7)
    open_mask = open_mask & (f41_smooth >= 0.65)
    # Exclude pixels within 25 pixels (~2.2 km) of any closed basin
    closed_any = np.zeros_like(open_mask)
    for c in candidates:
        closed_any |= (labels == c['lbl'])
    near_closed = ndimage.binary_dilation(closed_any, iterations=20)
    open_mask = open_mask & (~near_closed)

    olbl, n_open = ndimage.label(open_mask, structure=np.ones((3, 3)))
    open_gorges = []
    for lbl in range(1, n_open + 1):
        rr, cc = np.where(olbl == lbl)
        if len(rr) < 12:  # >= ~0.10 km2 of deep gorge bottom
            continue
        if rr.min() <= 15 or rr.max() >= H - 16 or cc.min() <= 15 or cc.max() >= W - 16:
            continue
        med_r, med_c = int(np.median(rr)), int(np.median(cc))
        lon_c, lat_c = rc_to_lonlat(transform, med_r, med_c)
        z_fl = float(np.median(dem[rr, cc]))
        # Check surrounding rim within 1.5 km has plateau relief >= 80 m above floor
        r0, r1 = max(0, rr.min() - 14), min(H, rr.max() + 15)
        c0, c1 = max(0, cc.min() - 14), min(W, cc.max() + 15)
        sub_z = dem[r0:r1, c0:c1]
        sub_n = nlcd[r0:r1, c0:c1]
        rim_z = sub_z[(sub_n == 41) & np.isfinite(sub_z)]
        if len(rim_z) < 40:
            continue
        z_rim = float(np.percentile(rim_z, 85))
        if (z_rim - z_fl) < 75.0:
            continue
        open_gorges.append({
            'id': f'Open_{lbl}',
            'lat': round(lat_c, 5), 'lon': round(lon_c, 5),
            'z_floor_med': round(z_fl, 1),
            'z_rim_med': round(z_rim, 1),
            'relief_m': round(z_rim - z_fl, 1),
            'n_px': int(len(rr)),
            'max_fill_m': round(float(np.max(fill_depth[rr, cc])), 1),
        })
    print(f'  Detected {len(open_gorges)} candidate forested open-drainage gorges')

    # Pair each forested closed basin 1:1 (greedy minimum distance + elev match, distinct open gorges >= 4km apart)
    used_open = []
    pairs = []
    # Cap to top 16 forested closed basins (if more exist, still report total count!)
    for c in forested:
        best_o, best_score = None, 1e9
        for o in open_gorges:
            d_km = haversine_km(c['lat'], c['lon'], o['lat'], o['lon'])
            if d_km < 3.5 or d_km > 40.0:
                continue
            dz = abs(c['z_floor_med'] - o['z_floor_med'])
            if dz > 80.0:
                continue
            # Ensure open gorges aren't duplicates (< 3km apart)
            if any(haversine_km(o['lat'], o['lon'], u['lat'], u['lon']) < 3.0 for u in used_open):
                continue
            score = d_km + 0.25 * dz
            if score < best_score:
                best_score, best_o = score, o
        if best_o is not None:
            used_open.append(best_o)
            onm = match_known_name(best_o['lat'], best_o['lon'], len(pairs) + 1)
            if not onm.endswith('_OPEN'):
                onm = f"OpenPair_{c['name']}"
            best_o = dict(best_o, name=onm, dist_to_closed_km=round(haversine_km(c['lat'], c['lon'], best_o['lat'], best_o['lon']), 1))
            pairs.append({'closed': c, 'open': best_o})

    print(f'  Matched {len(pairs)} 1:1 forested closed-karst <-> open-gorge pairs')
    return candidates, forested, disq, pairs


def verify_30m_priority_flood(candidates):
    """Run exact 30m 3DEP Priority-Flood locally around each 90m-screened depression."""
    cache_30 = os.path.join(C.CACHE_DIR, 'p1_30m_filldepth_audit.json')
    if os.path.exists(cache_30):
        audit = json.load(open(cache_30))
        print(f'[P1] Loaded cached 30m Priority-Flood verification ({len(audit)} basins)')
        return audit

    print(f'[P1] Running local 30m 3DEP Priority-Flood verification for {len(candidates)} candidate basins...')
    dem_ee = ee.Image('USGS/3DEP/10m').select('elevation')
    nlcd_ee = C.get_nlcd_2021()
    audit = {}
    for c in candidates:
        lon0, lat0, lon1, lat1 = c['bbox']
        # Pad by ~1.5 km (0.016 deg) so true spillover sill is well inside the tile
        pad = 0.018
        reg = ee.Geometry.Rectangle([lon0 - pad, lat0 - pad, lon1 + pad, lat1 + pad])
        try:
            url = dem_ee.addBands(nlcd_ee).getDownloadURL({
                'region': reg, 'scale': 30, 'crs': 'EPSG:32616', 'format': 'GEO_TIFF'
            })
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            with rasterio.MemoryFile(r.content) as mf:
                with mf.open() as ds:
                    d30 = ds.read(1).astype(np.float64)
                    n30 = ds.read(2).astype(np.int32)
                    px_km2 = (abs(ds.transform.a) * abs(ds.transform.e)) / 1e6
            d30[d30 < -100] = np.nan
            _, fd30 = C.priority_flood_fill(d30)
            # Identify central depression overlapping interior (excluding 10-px border)
            m5 = np.isfinite(fd30) & (fd30 >= 5.0)
            lbls, nf = ndimage.label(m5, structure=np.ones((3, 3)))
            best_rec = {'max_fill_30m': 0.0, 'area_30m_km2': 0.0, 'floor_nlcd41_30m': None}
            H, W = d30.shape
            for l in range(1, nf + 1):
                rr, cc = np.where(lbls == l)
                if rr.min() <= 2 or rr.max() >= H - 3 or cc.min() <= 2 or cc.max() >= W - 3:
                    continue
                # Must lie near center of tile
                if not (H * 0.2 <= np.median(rr) <= H * 0.8 and W * 0.2 <= np.median(cc) <= W * 0.8):
                    continue
                mfd = float(np.max(fd30[rr, cc]))
                if mfd > best_rec['max_fill_30m']:
                    fl_m = fd30[rr, cc] >= max(12.0, 0.50 * mfd)
                    fl_n = n30[rr[fl_m], cc[fl_m]]
                    best_rec = {
                        'max_fill_30m': round(mfd, 1),
                        'area_30m_km2': round(float(len(rr) * px_km2), 3),
                        'floor_nlcd41_30m': round(float(np.mean(fl_n == 41)), 3) if len(fl_n) else None,
                        'floor_forest_all_30m': round(float(np.mean(np.isin(fl_n, [41, 42, 43]))), 3) if len(fl_n) else None,
                        'floor_pasture_30m': round(float(np.mean(np.isin(fl_n, [81, 82]))), 3) if len(fl_n) else None,
                        'floor_water_30m': round(float(np.mean(fl_n == 11)), 3) if len(fl_n) else None,
                    }
            audit[c['name']] = best_rec
            print(f"  [30m] {c['name']:22s} 90m={c['max_fill_m']:5.1f}m -> 30m={best_rec['max_fill_30m']:5.1f}m | "
                  f"area={best_rec['area_30m_km2']:.3f}km2 | NLCD41={best_rec['floor_nlcd41_30m']}")
        except Exception as ex:
            audit[c['name']] = {'error': str(ex)}
    json.dump(audit, open(cache_30, 'w'), indent=2)
    return audit


def verify_30m_filldepth_and_extract_s2(pairs):
    """Extract per-year (2018-2024) Sentinel-2 NDVI & NDRE per stratum for all pairs in GEE."""
    # Use up to 14 pairs (all if <= 14) to keep GEE fast while maximizing power
    sel_pairs = pairs[:14]
    dem = C.get_dem_3dep()
    nlcd = C.get_nlcd_2021()
    decid = nlcd.eq(41)

    # Build a single FeatureCollection of (site, type, stratum) with geometry + elev bounds
    # We'll create one combined masked image per stratum? Better: pass geometries with elev bounds
    # and compute per-date spatial means in GEE!
    # Wait: even cleaner and faster — for each site, we have 3 strata (floor, mid, rim).
    # We can represent each stratum as an ee.Geometry + elevation range, then mask S2 per-feature?
    # Actually reduceRegions over a multi-band image where each band is masked by elev is fastest if
    # we do per-site, OR even simpler: mask by elevation inside a server-side map over features!
    # Let's check: 14 pairs * 2 = 28 sites. Grouping by site (28 tiny regions) or batching 28 sites
    # over 7 years is super fast if we extract at 30m scale!

    site_specs = []
    for p in sel_pairs:
        c = p['closed']
        o = p['open']
        # Closed site: hull for floor/mid, 1200m ring around hull for rim
        site_specs.append({
            'site': c['name'], 'pair': c['name'], 'kind': 'closed',
            'lat': c['lat'], 'lon': c['lon'],
            'max_fill_m': c['max_fill_m'],
            'hull': c['hull'],
            'z_floor_hi': round(c['z_floor_min'] + max(20.0, 0.40 * (c['z_spill'] - c['z_floor_min'])), 1),
            'z_spill': c['z_spill'],
            'z_rim_lo': round(max(c['z_spill'], c['z_rim_med'] - 25.0), 1),
        })
        site_specs.append({
            'site': o['name'], 'pair': c['name'], 'kind': 'open',
            'lat': o['lat'], 'lon': o['lon'],
            'max_fill_m': 0.0,
            'hull': None,
            'z_floor_hi': round(o['z_floor_med'] + 25.0, 1),
            'z_spill': round(0.5 * (o['z_floor_med'] + o['z_rim_med']), 1),
            'z_rim_lo': round(o['z_rim_med'] - 25.0, 1),
        })

    # Verify exact 30m 3DEP elevation & floor NLCD41 fraction for each site + build stratum masks
    print(f'[P1] Extracting 2018-2024 Sentinel-2 NDVI & NDRE for {len(site_specs)} sites ({len(sel_pairs)} pairs)...')

    def s2_prep(img):
        scl = img.select('SCL')
        good = scl.eq(4).Or(scl.eq(5))
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('ndvi')
        ndre = img.normalizedDifference(['B8A', 'B5']).rename('ndre')
        return ndvi.addBands(ndre).updateMask(good).updateMask(decid).set(
            'doy', img.date().getRelative('day', 'year'),
            'yr', img.date().get('year')
        )

    # Create a single labeled stratum image across the region? Sites don't overlap!
    # Wait: building ONE global stratum mask image (3 bands: floor_mask, mid_mask, rim_mask)
    # and reducing across a 28-feature FeatureCollection of site polygons (each buffered to include rim)
    # lets us make ONLY 7 GEE calls (1 per year) for ALL 28 sites x 3 strata x 2 indices!
    # Let's verify: non-overlapping site polygons -> each site has its own z_floor_hi and z_rim_lo!
    # We can paint z_floor_hi and z_rim_lo from the FeatureCollection onto images using reduceToImage!
    feats_site = []
    feats_hull = []
    for idx, sp in enumerate(site_specs):
        if sp['hull'] is not None:
            g_core = ee.Geometry.Polygon([sp['hull']])
            g_outer = g_core.buffer(1100)
        else:
            g_core = ee.Geometry.Point([sp['lon'], sp['lat']]).buffer(900)
            g_outer = ee.Geometry.Point([sp['lon'], sp['lat']]).buffer(1700)
        feats_site.append(ee.Feature(g_outer, {
            'sid': idx, 'site': sp['site'], 'kind': sp['kind'],
            'z_fl': float(sp['z_floor_hi']), 'z_rim': float(sp['z_rim_lo']),
        }))
        feats_hull.append(ee.Feature(g_core, {'in_core': 1}))

    fc_site = ee.FeatureCollection(feats_site)
    fc_hull = ee.FeatureCollection(feats_hull)
    region_all = fc_site.geometry().bounds()

    z_fl_img = fc_site.reduceToImage(['z_fl'], ee.Reducer.first())
    z_rim_img = fc_site.reduceToImage(['z_rim'], ee.Reducer.first())
    in_core_img = fc_hull.reduceToImage(['in_core'], ee.Reducer.max()).unmask(0)

    floor_m = decid.And(in_core_img.eq(1)).And(dem.lte(z_fl_img))
    mid_m = decid.And(in_core_img.eq(1)).And(dem.gt(z_fl_img)).And(dem.lt(z_rim_img))
    rim_m = decid.And(in_core_img.eq(0)).And(dem.gte(z_rim_img))

    # Check pixel counts per site/stratum first
    cnt_img = ee.Image.cat([
        floor_m.selfMask().rename('n_floor'),
        mid_m.selfMask().rename('n_mid'),
        rim_m.selfMask().rename('n_rim'),
        dem.updateMask(floor_m).rename('z_floor'),
        dem.updateMask(rim_m).rename('z_rim'),
    ])
    cnt_res = cnt_img.reduceRegions(
        collection=fc_site,
        reducer=ee.Reducer.count().combine(ee.Reducer.mean(), '', True),
        scale=30, tileScale=8
    ).getInfo()['features']
    site_meta = {}
    for f in cnt_res:
        p = f['properties']
        site_meta[p['site']] = {
            'n_floor': int(p.get('n_floor_count', 0) or 0),
            'n_mid': int(p.get('n_mid_count', 0) or 0),
            'n_rim': int(p.get('n_rim_count', 0) or 0),
            'z_floor_30m': round(float(p['z_floor_mean']), 1) if p.get('z_floor_mean') is not None else None,
            'z_rim_30m': round(float(p['z_rim_mean']), 1) if p.get('z_rim_mean') is not None else None,
        }
    print(f'  30m stratum check complete for {len(site_meta)} sites')

    # Cache S2 time series — extract PER SITE in parallel (tiny 3km bbox per site = zero GEE memory errors!)
    ts_cache = os.path.join(C.CACHE_DIR, 'p1_s2_ts_2018_2024.json')
    if os.path.exists(ts_cache):
        raw_ts = json.load(open(ts_cache))
        print(f'  Loaded cached S2 time series ({len(raw_ts)} rows)')
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def extract_one_site(sp):
            if sp['hull'] is not None:
                g_core = ee.Geometry.Polygon([sp['hull']])
                g_outer = g_core.buffer(1100)
            else:
                g_core = ee.Geometry.Point([sp['lon'], sp['lat']]).buffer(900)
                g_outer = ee.Geometry.Point([sp['lon'], sp['lat']]).buffer(1700)
            in_core = ee.Image.constant(0).paint(ee.FeatureCollection([ee.Feature(g_core)]), 1).eq(1)
            fl_m = decid.And(in_core).And(dem.lte(float(sp['z_floor_hi'])))
            md_m = decid.And(in_core).And(dem.gt(float(sp['z_floor_hi']))).And(dem.lt(float(sp['z_rim_lo'])))
            rm_m = decid.And(in_core.Not()).And(dem.gte(float(sp['z_rim_lo'])))

            s2_site = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                       .filterBounds(g_outer)
                       .filterDate('2018-01-01', '2025-01-01')
                       .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 65))
                       .map(s2_prep))

            def red_one(img):
                stacked = ee.Image.cat([
                    img.select('ndvi').updateMask(fl_m).rename('fl_ndvi'),
                    img.select('ndre').updateMask(fl_m).rename('fl_ndre'),
                    img.select('ndvi').updateMask(md_m).rename('md_ndvi'),
                    img.select('ndre').updateMask(md_m).rename('md_ndre'),
                    img.select('ndvi').updateMask(rm_m).rename('rm_ndvi'),
                    img.select('ndre').updateMask(rm_m).rename('rm_ndre'),
                ])
                d = stacked.reduceRegion(
                    reducer=ee.Reducer.mean().combine(ee.Reducer.count(), '', True),
                    geometry=g_outer, scale=30, tileScale=2, maxPixels=1e6
                )
                return ee.Feature(None, d).set('doy', img.get('doy'), 'yr', img.get('yr'))

            feats = s2_site.map(red_one).filter(
                ee.Filter.Or(ee.Filter.gte('fl_ndvi_count', 12), ee.Filter.gte('rm_ndvi_count', 12))
            ).getInfo()['features']
            out_rows = []
            for f in feats:
                p = f['properties']
                out_rows.append({
                    'site': sp['site'], 'kind': sp['kind'], 'yr': int(p['yr']), 'doy': int(p['doy']),
                    'fl_ndvi': p.get('fl_ndvi_mean'), 'fl_ndre': p.get('fl_ndre_mean'), 'fl_n': int(p.get('fl_ndvi_count', 0) or 0),
                    'md_ndvi': p.get('md_ndvi_mean'), 'md_ndre': p.get('md_ndre_mean'), 'md_n': int(p.get('md_ndvi_count', 0) or 0),
                    'rm_ndvi': p.get('rm_ndvi_mean'), 'rm_ndre': p.get('rm_ndre_mean'), 'rm_n': int(p.get('rm_ndvi_count', 0) or 0),
                })
            return sp['site'], out_rows

        raw_ts = []
        t_all = time.time()
        with ThreadPoolExecutor(max_workers=5) as ex:
            futs = [ex.submit(extract_one_site, sp) for sp in site_specs]
            for fut in as_completed(futs):
                snm, srows = fut.result()
                raw_ts.extend(srows)
                print(f'  [{snm}] {len(srows)} clear dates ({time.time()-t_all:.1f}s elapsed)')
        # Deduplicate overlapping same-day S2 tiles by selecting the tile with highest pixel coverage
        df_raw = pd.DataFrame(raw_ts)
        df_raw['tot_n'] = df_raw['fl_n'] + df_raw['rm_n']
        df_dedup = (df_raw.sort_values('tot_n', ascending=False)
                    .drop_duplicates(subset=['site', 'kind', 'yr', 'doy'], keep='first')
                    .drop(columns=['tot_n']))
        raw_ts = df_dedup.to_dict(orient='records')
        json.dump(raw_ts, open(ts_cache, 'w'))

    return site_specs, site_meta, pd.DataFrame(raw_ts)


def analyze_phenology_and_power(site_specs, site_meta, df_ts, pairs):
    spec_map = {s['site']: s for s in site_specs}
    fits = []
    for (site, yr), grp in df_ts.groupby(['site', 'yr']):
        sp = spec_map[site]
        for st_code, st_name in [('fl', 'floor'), ('md', 'mid'), ('rm', 'rim')]:
            sub = grp[grp[f'{st_code}_n'] >= 15]
            for idx in ['ndvi', 'ndre']:
                res = C.fit_double_logistic(sub['doy'].values, sub[f'{st_code}_{idx}'].values,
                                            min_obs=6, min_spring=2, min_autumn=2)
                fits.append({
                    'site': site, 'pair': sp['pair'], 'kind': sp['kind'],
                    'max_fill_m': sp['max_fill_m'],
                    'stratum': st_name, 'index': idx, 'yr': int(yr), **res
                })
    df_fit = pd.DataFrame(fits)

    # Compute floor - rim differences per (site, yr, index) where both floor and rim pass QC
    fl = df_fit[(df_fit['stratum'] == 'floor') & (df_fit['qc_pass'])].set_index(['site', 'yr', 'index'])
    rm = df_fit[(df_fit['stratum'] == 'rim') & (df_fit['qc_pass'])].set_index(['site', 'yr', 'index'])
    common_idx = fl.index.intersection(rm.index)
    diff_rows = []
    for key in common_idx:
        f_r, r_r = fl.loc[key], rm.loc[key]
        diff_rows.append({
            'site': key[0], 'yr': key[1], 'index': key[2],
            'pair': f_r['pair'], 'kind': f_r['kind'], 'max_fill_m': f_r['max_fill_m'],
            'fl_sos': f_r['sos50'], 'rm_sos': r_r['sos50'], 'dSOS_fl_rm': f_r['sos50'] - r_r['sos50'],
            'fl_eos': f_r['eos50'], 'rm_eos': r_r['eos50'], 'dEOS_fl_rm': f_r['eos50'] - r_r['eos50'],
            'fl_gsl': f_r['gsl'], 'rm_gsl': r_r['gsl'], 'dGSL_fl_rm': f_r['gsl'] - r_r['gsl'],
            'fl_r2': f_r['r2'], 'fl_rmse': f_r['rmse'],
        })
    df_diff = pd.DataFrame(diff_rows)

    summary = {}
    for idx in ['ndvi', 'ndre']:
        sub = df_diff[df_diff['index'] == idx]
        # Aggregate to cove level (cove = independent experimental unit!)
        cove_agg = sub.groupby(['site', 'pair', 'kind', 'max_fill_m']).agg(
            n_yrs=('yr', 'count'),
            dSOS=('dSOS_fl_rm', 'mean'),
            dEOS=('dEOS_fl_rm', 'mean'),
            dGSL=('dGSL_fl_rm', 'mean'),
            sigma_SOS=('fl_sos', 'std'),
            mean_fl_sos=('fl_sos', 'mean'),
            mean_fl_eos=('fl_eos', 'mean'),
            mean_rm_eos=('rm_eos', 'mean'),
        ).reset_index()

        cl = cove_agg[cove_agg['kind'] == 'closed'].set_index('pair')
        op = cove_agg[cove_agg['kind'] == 'open'].set_index('pair')
        paired_keys = cl.index.intersection(op.index)
        cl_p = cl.loc[paired_keys]
        op_p = op.loc[paired_keys]

        # Paired Closed vs Open tests on floor-rim dEOS, dSOS, dGSL
        did_eos = (cl_p['dEOS'] - op_p['dEOS']).values
        did_sos = (cl_p['dSOS'] - op_p['dSOS']).values
        did_gsl = (cl_p['dGSL'] - op_p['dGSL']).values

        t_eos, p_eos = stats.ttest_1samp(did_eos, 0.0) if len(did_eos) >= 3 else (np.nan, np.nan)
        t_sos, p_sos = stats.ttest_1samp(did_sos, 0.0) if len(did_sos) >= 3 else (np.nan, np.nan)
        t_gsl, p_gsl = stats.ttest_1samp(did_gsl, 0.0) if len(did_gsl) >= 3 else (np.nan, np.nan)
        t_cl_eos, p_cl_eos = stats.ttest_1samp(cl['dEOS'].values, 0.0) if len(cl) >= 3 else (np.nan, np.nan)
        try:
            w_eos_p = float(stats.wilcoxon(did_eos).pvalue) if len(did_eos) >= 5 else np.nan
        except Exception:
            w_eos_p = np.nan

        # Also fit cove-level random-intercept mixed-effects model on annual (site, yr) observations:
        # dEOS_ij = b0 + b1 * I(closed) + u_site + v_pair + e_ij  -> approximated cleanly by cluster-robust OLS clustered by pair
        sub_reg = sub.copy()
        sub_reg['is_closed'] = (sub_reg['kind'] == 'closed').astype(float)
        lmm_eos = C.ols_cluster_se(sub_reg[['is_closed']].values, sub_reg['dEOS_fl_rm'].values,
                                   clusters=sub_reg['pair'].values, col_names=['is_closed'])
        lmm_sos = C.ols_cluster_se(sub_reg[['is_closed']].values, sub_reg['dSOS_fl_rm'].values,
                                   clusters=sub_reg['pair'].values, col_names=['is_closed'])
        lmm_gsl = C.ols_cluster_se(sub_reg[['is_closed']].values, sub_reg['dGSL_fl_rm'].values,
                                   clusters=sub_reg['pair'].values, col_names=['is_closed'])

        # Diagnostics: r(sigma_SOS, FillDepth) across all coves (and closed only)
        ok_sig = cove_agg['sigma_SOS'].notna()
        r_sig_all, p_sig_all = stats.pearsonr(cove_agg.loc[ok_sig, 'sigma_SOS'], cove_agg.loc[ok_sig, 'max_fill_m']) if ok_sig.sum() >= 4 else (np.nan, np.nan)
        ok_sig_cl = cl['sigma_SOS'].notna()
        r_sig_cl, p_sig_cl = stats.pearsonr(cl.loc[ok_sig_cl, 'sigma_SOS'], cl.loc[ok_sig_cl, 'max_fill_m']) if ok_sig_cl.sum() >= 4 else (np.nan, np.nan)
        r_eos_fill, p_eos_fill = stats.pearsonr(cl['dEOS'], cl['max_fill_m']) if len(cl) >= 4 else (np.nan, np.nan)

        frac_early_eos = float(np.mean(cl['dEOS'] < 0)) if len(cl) else np.nan
        frac_neg_gsl = float(np.mean(cl['dGSL'] < op['dGSL'].reindex(cl.index))) if len(paired_keys) else np.nan

        summary[idx] = {
            'n_closed_valid': int(len(cl)),
            'n_open_valid': int(len(op)),
            'n_pairs_valid': int(len(paired_keys)),
            'closed_mean_dEOS_fl_rm': round(float(cl['dEOS'].mean()), 3),
            'closed_sd_dEOS_fl_rm': round(float(cl['dEOS'].std()), 3),
            'closed_dEOS_onesamp_t': round(float(t_cl_eos), 3),
            'closed_dEOS_onesamp_p': float(f'{p_cl_eos:.4g}'),
            'closed_frac_early_floor_EOS': round(frac_early_eos, 3),
            'open_mean_dEOS_fl_rm': round(float(op['dEOS'].mean()), 3),
            'open_sd_dEOS_fl_rm': round(float(op['dEOS'].std()), 3),
            'closed_mean_dSOS_fl_rm': round(float(cl['dSOS'].mean()), 3),
            'open_mean_dSOS_fl_rm': round(float(op['dSOS'].mean()), 3),
            'closed_mean_dGSL_fl_rm': round(float(cl['dGSL'].mean()), 3),
            'open_mean_dGSL_fl_rm': round(float(op['dGSL'].mean()), 3),
            'paired_DiD_dEOS_mean': round(float(np.mean(did_eos)), 3),
            'paired_DiD_dEOS_t': round(float(t_eos), 3),
            'paired_DiD_dEOS_p': float(f'{p_eos:.4g}'),
            'paired_DiD_dEOS_wilcoxon_p': float(f'{w_eos_p:.4g}') if np.isfinite(w_eos_p) else None,
            'paired_DiD_dSOS_mean': round(float(np.mean(did_sos)), 3),
            'paired_DiD_dSOS_t': round(float(t_sos), 3),
            'paired_DiD_dSOS_p': float(f'{p_sos:.4g}'),
            'paired_DiD_dGSL_mean': round(float(np.mean(did_gsl)), 3),
            'paired_DiD_dGSL_t': round(float(t_gsl), 3),
            'paired_DiD_dGSL_p': float(f'{p_gsl:.4g}'),
            'cluster_mixed_dEOS': lmm_eos['coefs']['is_closed'],
            'cluster_mixed_dSOS': lmm_sos['coefs']['is_closed'],
            'cluster_mixed_dGSL': lmm_gsl['coefs']['is_closed'],
            'r_sigmaSOS_filldepth_all': round(float(r_sig_all), 3) if np.isfinite(r_sig_all) else None,
            'r_sigmaSOS_filldepth_closed': round(float(r_sig_cl), 3) if np.isfinite(r_sig_cl) else None,
            'r_dEOS_filldepth_closed': round(float(r_eos_fill), 3) if np.isfinite(r_eos_fill) else None,
            'p_dEOS_filldepth_closed': float(f'{p_eos_fill:.4g}') if np.isfinite(p_eos_fill) else None,
            'per_cove_table': cove_agg.round(3).to_dict(orient='records'),
        }

    # Diagnose Swagerty Cove specifically
    swag_diag = {}
    swag_rows = df_fit[df_fit['site'].str.contains('Swagerty', case=False, na=False)]
    if len(swag_rows) > 0:
        swag_diag['found_in_census'] = True
        swag_diag['per_stratum_summary'] = (
            swag_rows.groupby(['stratum', 'index']).agg(
                n_total=('yr', 'count'),
                n_qc_pass=('qc_pass', 'sum'),
                mean_sos50=('sos50', 'mean'),
                mean_eos50=('eos50', 'mean'),
                mean_r2=('r2', 'mean'),
                mean_n_spring=('n_spring', 'mean'),
            ).round(3).reset_index().to_dict(orient='records')
        )
    else:
        swag_diag['found_in_census'] = False
        # Run targeted Swagerty Cove audit directly in GEE!
        swag_diag['targeted_audit'] = audit_swagerty_cove()

    return summary, swag_diag, df_fit


def audit_swagerty_cove():
    """Targeted diagnostic on Swagerty Cove (35.870, -85.370) for its Stage-1 mid-slope SOS=128 anomaly."""
    pt = ee.Geometry.Point([-85.370, 35.870]).buffer(1800)
    dem = C.get_dem_3dep()
    nlcd = C.get_nlcd_2021()
    evt = C.get_evt()
    decid = nlcd.eq(41)
    # Check elevation distribution & EVT
    hist = evt.updateMask(decid).reduceRegion(ee.Reducer.frequencyHistogram(), pt, 30, maxPixels=1e7).getInfo().get('EVT', {})
    # Check floor (<350m), mid (350-480m), rim (>500m) S2 NDVI/NDRE across 2018-2024
    def prep(img):
        scl = img.select('SCL')
        good = scl.eq(4).Or(scl.eq(5))
        return (img.normalizedDifference(['B8', 'B4']).rename('ndvi')
                .addBands(img.normalizedDifference(['B8A', 'B5']).rename('ndre'))
                .updateMask(good).updateMask(decid)
                .set('doy', img.date().getRelative('day', 'year'), 'yr', img.date().get('year')))
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(pt).filterDate('2018-01-01', '2025-01-01')
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 65)).map(prep))
    fl_m = decid.And(dem.lte(360))
    md_m = decid.And(dem.gt(360)).And(dem.lte(490))
    rm_m = decid.And(dem.gte(510))
    # Also check evergreen / mixed fraction on mid-slope
    lc_mid = nlcd.updateMask(dem.gt(360).And(dem.lte(490))).reduceRegion(
        ee.Reducer.frequencyHistogram(), pt, 30, maxPixels=1e7).getInfo().get('landcover', {})
    def ext(img):
        st = ee.Image.cat([
            img.select('ndvi').updateMask(fl_m).rename('fl_ndvi'),
            img.select('ndvi').updateMask(md_m).rename('md_ndvi'),
            img.select('ndvi').updateMask(rm_m).rename('rm_ndvi'),
            img.select('ndre').updateMask(md_m).rename('md_ndre'),
        ])
        d = st.reduceRegion(ee.Reducer.mean(), pt, 30, tileScale=4, maxPixels=1e7)
        return ee.Feature(None, d).set('doy', img.get('doy'), 'yr', img.get('yr'))
    feats = s2.map(ext).getInfo()['features']
    df = pd.DataFrame([f['properties'] for f in feats])
    out = {'evt_top5': dict(sorted(hist.items(), key=lambda kv: -kv[1])[:5]),
           'mid_slope_nlcd_hist': lc_mid, 'per_year_mid_ndvi': []}
    for yr, grp in df.groupby('yr'):
        r_fl = C.fit_double_logistic(grp['doy'].values, grp['fl_ndvi'].values, min_obs=6, min_spring=2, min_autumn=2)
        r_md = C.fit_double_logistic(grp['doy'].values, grp['md_ndvi'].values, min_obs=6, min_spring=2, min_autumn=2)
        r_rm = C.fit_double_logistic(grp['doy'].values, grp['rm_ndvi'].values, min_obs=6, min_spring=2, min_autumn=2)
        out['per_year_mid_ndvi'].append({
            'yr': int(yr),
            'fl_sos': r_fl['sos50'], 'fl_eos': r_fl['eos50'], 'fl_qc': r_fl['qc_pass'],
            'md_sos': r_md['sos50'], 'md_eos': r_md['eos50'], 'md_r2': r_md['r2'], 'md_qc': r_md['qc_pass'],
            'rm_sos': r_rm['sos50'], 'rm_eos': r_rm['eos50'], 'rm_qc': r_rm['qc_pass'],
        })
    return out


def main():
    t0 = time.time()
    C.init_ee()
    dem, nlcd, transform, bounds = load_regional_dem_nlcd()
    candidates, forested, disq, pairs = detect_closed_basins_and_open_pairs(dem, nlcd, transform)
    audit_30m = verify_30m_priority_flood(candidates)
    for c in candidates:
        c['verify_30m'] = audit_30m.get(c['name'], {})
    site_specs, site_meta, df_ts = verify_30m_filldepth_and_extract_s2(pairs)
    summary, swag_diag, df_fit = analyze_phenology_and_power(site_specs, site_meta, df_ts, pairs)

    # Strict >=25.0m & >=0.35km2 (at 30m Priority-Flood) sensitivity check
    strict_30m_names = {
        c['name'] for c in forested
        if audit_30m.get(c['name'], {}).get('max_fill_30m', 0) >= 25.0
        and audit_30m.get(c['name'], {}).get('area_30m_km2', 0) >= 0.35
    }
    strict_90m_names = {
        c['name'] for c in forested
        if c['max_fill_m'] >= 25.0 and c['area_km2'] >= 0.35
    }
    strict_check = {}
    for label, keep_set in [('strict_30m_ge25m', strict_30m_names), ('strict_90m_ge25m', strict_90m_names)]:
        strict_check[label] = {'n_pairs': len(keep_set)}
        for idx in ['ndvi', 'ndre']:
            tbl = pd.DataFrame(summary[idx]['per_cove_table'])
            tbl_s = tbl[tbl['pair'].isin(keep_set)]
            cl_s = tbl_s[tbl_s['kind'] == 'closed'].set_index('pair')
            op_s = tbl_s[tbl_s['kind'] == 'open'].set_index('pair')
            pk = cl_s.index.intersection(op_s.index)
            if len(pk) >= 3:
                de = (cl_s.loc[pk, 'dEOS'] - op_s.loc[pk, 'dEOS']).values
                ds = (cl_s.loc[pk, 'dSOS'] - op_s.loc[pk, 'dSOS']).values
                dg = (cl_s.loc[pk, 'dGSL'] - op_s.loc[pk, 'dGSL']).values
                strict_check[label][idx] = {
                    'n_pairs': int(len(pk)),
                    'DiD_dEOS_mean': round(float(de.mean()), 3),
                    'DiD_dEOS_p': float(f'{stats.ttest_1samp(de, 0.0).pvalue:.4g}'),
                    'DiD_dSOS_mean': round(float(ds.mean()), 3),
                    'DiD_dSOS_p': float(f'{stats.ttest_1samp(ds, 0.0).pvalue:.4g}'),
                    'DiD_dGSL_mean': round(float(dg.mean()), 3),
                    'DiD_dGSL_p': float(f'{stats.ttest_1samp(dg, 0.0).pvalue:.4g}'),
                }

    # Evaluate pre-registered P1 Pass/Fail
    ndvi_s = summary['ndvi']
    ndre_s = summary['ndre']
    n_pairs = min(ndvi_s['n_pairs_valid'], ndre_s['n_pairs_valid'])
    p_ndvi = ndvi_s['cluster_mixed_dEOS']['p']
    p_ndre = ndre_s['cluster_mixed_dEOS']['p']
    frac_early = ndvi_s['closed_frac_early_floor_EOS']
    open_deos = ndvi_s['open_mean_dEOS_fl_rm']
    r_sig = ndvi_s['r_sigmaSOS_filldepth_all']

    passed = (
        n_pairs >= 8
        and p_ndvi < 0.01 and p_ndre < 0.01
        and frac_early >= 0.75
        and open_deos >= 0.0
    )
    verdict = 'PASS' if (passed and (r_sig is None or abs(r_sig) < 0.15)) else (
        'PARTIAL_PASS' if (n_pairs >= 6 and p_ndvi < 0.05 and p_ndre < 0.05) else 'FAIL'
    )

    out = {
        'experiment': 'P1_karst_census',
        'elapsed_s': round(time.time() - t0, 1),
        'region_bbox': BBOX,
        'census_counts': {
            'total_closed_90m_ge22p5m_ge030km2': len(candidates),
            'total_closed_30m_ge25m_ge035km2': sum(
                1 for c in candidates
                if audit_30m.get(c['name'], {}).get('max_fill_30m', 0) >= 25.0
                and audit_30m.get(c['name'], {}).get('area_30m_km2', 0) >= 0.35
            ),
            'forested_karst_nlcd41_ge60pct': len(forested),
            'forested_karst_strict_30m_ge25m': len(strict_30m_names),
            'forested_karst_strict_90m_ge25m': len(strict_90m_names),
            'disqualified_basins': len(disq),
            'matched_1to1_pairs_total': len(pairs),
            'evaluated_pairs': len(pairs[:14]),
        },
        'forested_basins': forested,
        'disqualified_basins': disq,
        'strict_threshold_sensitivity': strict_check,
        'pairs_evaluated': [{'closed': p['closed']['name'], 'open': p['open']['name'],
                             'closed_fill_90m': p['closed']['max_fill_m'],
                             'closed_fill_30m': audit_30m.get(p['closed']['name'], {}).get('max_fill_30m'),
                             'dist_km': p['open']['dist_to_closed_km'],
                             'z_floor_closed': p['closed']['z_floor_med'],
                             'z_floor_open': p['open']['z_floor_med']} for p in pairs[:14]],
        'site_meta_30m': site_meta,
        'phenology_power': summary,
        'swagerty_diagnostic': swag_diag,
        'gate_check': {
            'n_pairs_valid_ge_8': n_pairs >= 8,
            'n_pairs_valid': n_pairs,
            'n_pairs_strict_30m_ge25m': len(strict_30m_names),
            'mixed_dEOS_p_ndvi': p_ndvi,
            'mixed_dEOS_p_ndre': p_ndre,
            'closed_frac_early_floor_EOS': frac_early,
            'open_mean_dEOS_fl_rm': open_deos,
            'r_sigmaSOS_filldepth': r_sig,
            'verdict': verdict,
        }
    }
    C.save_json('p1.json', out)
    print('\n=== P1 SUMMARY ===')
    print(json.dumps(out['gate_check'], indent=2))
    print('NDVI:', {k: v for k, v in ndvi_s.items() if k != 'per_cove_table'})
    print('NDRE:', {k: v for k, v in ndre_s.items() if k != 'per_cove_table'})


if __name__ == '__main__':
    main()
