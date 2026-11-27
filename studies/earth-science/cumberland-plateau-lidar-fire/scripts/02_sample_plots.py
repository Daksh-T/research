"""Phase 2A: Build Stratified, Pure-Deciduous, Fire-History Sampling Design
Across Three Cumberland Plateau Landscapes (Bankhead NF AL, Daniel Boone NF KY,
and Sewanee Domain + Franklin SF TN).

Uses authoritative USFS FACTS (Layer 3 Completed Fire + Layer 5 Completed
Mechanical) and Sewanee LAL (burns.geojson) polygons, intersects them with
exact 3DEP S3 EPT LiDAR footprints, filters strictly to continuous pure
deciduous upland forest (Annual NLCD 41 in 2001, 2010, 2021 + Hansen
treecover2000 >= 60% and loss == 0) via Google Earth Engine, and attaches
10 m 3DEP topographic covariates (elevation, slope, HLI, TPI_450m) to every
30 m-radius circular plot (minimum spacing >= 200 m).

Outputs: data/lidar/sample_plots.json
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter, defaultdict

import ee
import numpy as np
import pyproj
import requests
from shapely.errors import ShapelyError
from shapely.geometry import Point, box, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import plateau_lib as P


def load_polygons_with_years(path: str, year_key: str) -> list[tuple[object, int]]:
    with open(path) as f:
        fc = json.load(f)
    out = []
    for feat in fc["features"]:
        yr = feat["properties"].get(year_key)
        geom = feat.get("geometry")
        if yr is None or geom is None:
            continue
        try:
            g = shape(geom)
            if not g.is_valid:
                g = g.buffer(0)
            if not g.is_empty:
                out.append((g, int(yr)))
        except (ShapelyError, TypeError, ValueError, KeyError, AttributeError) as exc:
            print(f"  [WARN] Skipping invalid polygon in {path}: {exc}", flush=True)
    return out


def fetch_3dep_footprint(workunit: str):
    url = "https://index.nationalmap.gov/arcgis/rest/services/3DEPElevationIndex/MapServer/24/query"
    r = requests.get(url, params={"where": f"workunit='{workunit}'", "outSR": "4326", "f": "geojson"}, timeout=30).json()
    geoms = [shape(f["geometry"]).buffer(0) for f in r.get("features", [])]
    return unary_union(geoms)


def generate_candidate_points(
    site_name: str,
    domain_geom,
    fire_polys: list[tuple[object, int]],
    mech_polys: list[tuple[object, int]],
    epoch1_yr: int,
    epoch2_yr: int | None,
    ept_primary: str,
    ept_baseline: str | None,
    min_spacing_m: float = 220.0,
    target_per_bin: int = 45,
    rng_seed: int = 42,
) -> list[dict]:
    """Generate spatially spaced (>= min_spacing_m) candidate points stratified by
    exact burn count & mechanical treatment status BEFORE `epoch1_yr` (or between
    `epoch1_yr` and `epoch2_yr` for BACI).
    """
    rng = np.random.default_rng(rng_seed)
    tr_to_m = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)

    # Spatial indices for fast point-in-polygon queries
    fire_geoms = [g for g, _ in fire_polys]
    fire_yrs = [y for _, y in fire_polys]
    fire_tree = STRtree(fire_geoms) if fire_geoms else None

    mech_geoms = [g for g, _ in mech_polys]
    mech_yrs = [y for _, y in mech_polys]
    mech_tree = STRtree(mech_geoms) if mech_geoms else None

    # Build a dense candidate pool from:
    # (a) random points inside burned polygons (ensures rare 3+ burn strata are well-sampled)
    # (b) random points across the full domain bounding box (ensures abundant unburned controls)
    b = domain_geom.bounds
    candidates_ll: list[tuple[float, float]] = []

    # Sample inside each fire polygon intersecting domain
    for g, _ in fire_polys:
        if not g.intersects(domain_geom):
            continue
        inter = g.intersection(domain_geom)
        if inter.is_empty:
            continue
        gb = inter.bounds
        for _ in range(18):
            lx = rng.uniform(gb[0], gb[2])
            ly = rng.uniform(gb[1], gb[3])
            p = Point(lx, ly)
            if inter.contains(p):
                candidates_ll.append((lx, ly))

    # Sample across domain for controls
    n_bg = 3500
    xs = rng.uniform(b[0], b[2], size=n_bg)
    ys = rng.uniform(b[1], b[3], size=n_bg)
    for lx, ly in zip(xs, ys, strict=False):
        if domain_geom.contains(Point(lx, ly)):
            candidates_ll.append((float(lx), float(ly)))

    rng.shuffle(candidates_ll)

    # Classify each candidate into strata and enforce minimum spacing
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    accepted_xy: list[tuple[float, float]] = []

    for lx, ly in candidates_ll:
        p = Point(lx, ly)
        # Fire history
        hit_fire_yrs = []
        if fire_tree is not None:
            for idx in fire_tree.query(p):
                if fire_geoms[idx].contains(p):
                    hit_fire_yrs.append(fire_yrs[idx])
        # Deduplicate multiple FACTS rows in the same fiscal year at the same point
        uniq_fire_yrs = sorted(set(hit_fire_yrs))

        # Mechanical history
        hit_mech_yrs = []
        if mech_tree is not None:
            for idx in mech_tree.query(p):
                if mech_geoms[idx].contains(p):
                    hit_mech_yrs.append(mech_yrs[idx])
        uniq_mech_yrs = sorted(set(hit_mech_yrs))

        # Primary cross-sectional stratum (burns strictly <= target LiDAR year)
        ref_yr = epoch2_yr if epoch2_yr is not None else epoch1_yr
        pre_burns = [y for y in uniq_fire_yrs if y <= ref_yr]
        pre_mech = [y for y in uniq_mech_yrs if y <= ref_yr]
        n_burns = len(pre_burns)
        has_mech = len(pre_mech) > 0

        # For BACI sites, also count inter-epoch burns strictly in (epoch1_yr, epoch2_yr]
        if epoch2_yr is not None:
            inter_burns = [y for y in uniq_fire_yrs if epoch1_yr < y <= epoch2_yr]
            n_inter = len(inter_burns)
            if has_mech:
                if n_inter >= 1:
                    stratum = "baci_mech_fire"
                else:
                    continue
            else:
                if n_burns == 0:
                    stratum = "baci_0_control"
                elif n_inter == 0:
                    # Burned only BEFORE epoch1 — useful recovery/legacy check, keep modest sample
                    stratum = "baci_pre_only"
                elif n_inter == 1:
                    stratum = "baci_1_inter"
                else:
                    stratum = "baci_2plus_inter"
        else:
            if has_mech:
                if n_burns >= 1:
                    stratum = "mech_plus_fire"
                else:
                    continue
            else:
                if n_burns == 0:
                    stratum = "0_burns"
                elif n_burns == 1:
                    stratum = "1_burn"
                elif n_burns == 2:
                    stratum = "2_burns"
                else:
                    stratum = "3plus_burns"

        cap = target_per_bin * 3  # Oversample 3x before GEE pure-deciduous filter
        if len(by_stratum[stratum]) >= cap:
            continue

        mx, my = tr_to_m.transform(lx, ly)
        # Check spacing against already accepted points
        too_close = False
        for ax, ay in accepted_xy:
            if (mx - ax) ** 2 + (my - ay) ** 2 < min_spacing_m ** 2:
                too_close = True
                break
        if too_close:
            continue

        accepted_xy.append((mx, my))
        by_stratum[stratum].append({
            "site": site_name,
            "stratum": stratum,
            "lon": round(lx, 6),
            "lat": round(ly, 6),
            "n_burns_total_pre_lidar": n_burns,
            "burn_years_pre_lidar": pre_burns,
            "last_burn_year": max(pre_burns) if pre_burns else None,
            "years_since_burn": (ref_yr - max(pre_burns)) if pre_burns else None,
            "n_inter_burns": len([y for y in uniq_fire_yrs if epoch1_yr < y <= epoch2_yr]) if epoch2_yr else None,
            "has_mech": has_mech,
            "mech_years": pre_mech,
            "ept_primary": ept_primary,
            "ept_baseline": ept_baseline,
            "epoch1_yr": epoch1_yr,
            "epoch2_yr": epoch2_yr,
        })

    all_pts = [pt for lst in by_stratum.values() for pt in lst]
    print(f"  [{site_name}] raw candidates={len(all_pts)} -> {dict(Counter(p['stratum'] for p in all_pts))}", flush=True)
    return all_pts


def filter_pure_deciduous_and_topo_gee(candidates: list[dict], max_per_stratum: int = 38) -> list[dict]:
    """Filter candidate plots on GEE to strictly continuous pure deciduous upland forest
    (Annual NLCD 41 in 2001, 2010, and 2021; Hansen treecover2000 >= 60 and loss == 0)
    and attach 10 m 3DEP elevation, slope, HLI, and metric TPI (450 m).
    """
    P.init()
    gfc = ee.Image(P.HANSEN_ASSET)
    lc01 = P.annual_nlcd_image(2001).rename("lc01")
    lc10 = P.annual_nlcd_image(2010).rename("lc10")
    lc21 = P.annual_nlcd_image(2021).rename("lc21")
    dem = P.get_dem(use_srtm=False).rename("elev")
    slope = ee.Terrain.slope(dem).rename("slope")
    aspect = ee.Terrain.aspect(dem)
    d2r = math.pi / 180.0
    slope_r = slope.multiply(d2r)
    lat_r = ee.Image.pixelLonLat().select("latitude").multiply(d2r)
    folded = ee.Image(180.0).subtract(aspect.subtract(225.0).abs()).multiply(d2r)
    hli = (
        ee.Image(0.339)
        .add(slope_r.cos().multiply(lat_r.cos()).multiply(0.808))
        .subtract(slope_r.sin().multiply(lat_r.sin()).multiply(0.196))
        .subtract(folded.cos().multiply(slope_r.sin()).multiply(0.482))
        .rename("hli")
    )
    tpi = dem.subtract(dem.focalMean(450, "circle", "meters")).rename("tpi")

    # 30 m radius neighborhood mean of deciduous purity so plot edges aren't non-forest/pine
    decid_pure = lc01.eq(41).And(lc10.eq(41)).And(lc21.eq(41)).And(gfc.select("treecover2000").gte(60)).And(gfc.select("loss").Not()).float().rename("decid_purity")
    stack = ee.Image.cat([decid_pure, dem, slope, hli, tpi])

    kept: list[dict] = []
    # Process in batches of 250 points
    batch_size = 250
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        feats = [
            ee.Feature(ee.Geometry.Point([p["lon"], p["lat"]]).buffer(30), {"idx": i + j})
            for j, p in enumerate(batch)
        ]
        fc = ee.FeatureCollection(feats)
        sampled = stack.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=30, tileScale=4).getInfo()
        for f in sampled["features"]:
            pr = f["properties"]
            idx = int(pr["idx"])
            purity = pr.get("decid_purity")
            elev = pr.get("elev")
            slp = pr.get("slope")
            hli_v = pr.get("hli")
            tpi_v = pr.get("tpi")
            if purity is None or elev is None or slp is None:
                continue
            # Require >= 80% pure continuous deciduous intact forest inside the 30 m radius plot
            if purity < 0.80:
                continue
            pt = dict(candidates[idx])
            pt["decid_purity"] = round(float(purity), 3)
            pt["elev_m"] = round(float(elev), 1)
            pt["slope_deg"] = round(float(slp), 2)
            pt["hli"] = round(float(hli_v), 4)
            pt["tpi_450m"] = round(float(tpi_v), 2)
            kept.append(pt)
        print(f"  GEE filtered batch {i}..{i+len(batch)}: total pure-deciduous kept so far = {len(kept)}", flush=True)

    # Cap per (site, stratum) while keeping generous controls for matching
    final = []
    counts = Counter()
    for pt in kept:
        key = (pt["site"], pt["stratum"])
        cap = max_per_stratum * 2 if ("0_" in pt["stratum"] or "control" in pt["stratum"]) else max_per_stratum
        if counts[key] < cap:
            counts[key] += 1
            pt["plot_id"] = f"{pt['site']}_{counts[key]:03d}_{pt['stratum']}"
            final.append(pt)
    print("Final pure-deciduous plot counts:", dict(counts), flush=True)
    return final


def main() -> None:
    os.makedirs("data/lidar", exist_ok=True)
    print("=== Loading USFS FACTS & Sewanee LAL fire/mechanical polygons ===", flush=True)
    bh_fire = load_polygons_with_years("data/facts/bankhead_fire.geojson", "fiscal_year_completed")
    bh_mech = load_polygons_with_years("data/facts/bankhead_mech.geojson", "fiscal_year_completed")
    db_fire = load_polygons_with_years("data/facts/danielboone_fire.geojson", "fiscal_year_completed")
    db_mech = load_polygons_with_years("data/facts/danielboone_mech.geojson", "fiscal_year_completed")
    sew_fire = load_polygons_with_years(os.path.join("..", "initial_study", "burns.geojson"), "Year")

    # Forest boundaries
    # Bankhead NF is the northwestern unit of National Forests in Alabama (clip to Bankhead bbox)
    with open("data/facts/bankhead_boundary.geojson") as f:
        bh_all = unary_union([shape(feat["geometry"]).buffer(0) for feat in json.load(f)["features"]])
    bh_bound = bh_all.intersection(box(-87.60, 34.10, -87.05, 34.50))
    with open("data/facts/danielboone_boundary.geojson") as f:
        db_bound = unary_union([shape(feat["geometry"]).buffer(0) for feat in json.load(f)["features"]])

    # Sewanee Domain + Franklin State Forest boundaries
    with open("data/areas.geojson") as f:
        areas_fc = json.load(f)["features"]
    sew_bound = shape(next(feat["geometry"] for feat in areas_fc if feat["properties"]["name"] == "Sewanee Domain")).buffer(0)
    fsf_bound = shape(next(feat["geometry"] for feat in areas_fc if feat["properties"]["name"] == "Franklin State Forest")).buffer(0)

    print("=== Fetching 3DEP LiDAR workunit footprints ===", flush=True)
    wu_al_2012 = fetch_3dep_footprint("AL_LawrenceCo_2012")
    wu_al_2017 = fetch_3dep_footprint("AL_25Co_B1_2017")
    wu_al_2020 = fetch_3dep_footprint("AL_17Co_1_2020")
    wu_ky_2023 = fetch_3dep_footprint("KY_CentralEast_3_A23")

    all_cands: list[dict] = []

    # 1. Bankhead North — BACI 2012 -> 2020/21 (8.9 yr repeat leaf-off LiDAR!)
    bh_north_baci = bh_bound.intersection(wu_al_2012).intersection(wu_al_2020)
    all_cands.extend(generate_candidate_points(
        site_name="Bankhead_North_BACI",
        domain_geom=bh_north_baci,
        fire_polys=bh_fire,
        mech_polys=bh_mech,
        epoch1_yr=2012,
        epoch2_yr=2020,
        ept_primary="AL_17Co_1_2020",
        ept_baseline="USGS_LPC_AL_LawrenceCo_2012_LAS_2018",
        min_spacing_m=220.0,
        target_per_bin=35,
        rng_seed=101,
    ))

    # 2. Bankhead South — Cross-sectional dose-response (2017 QL2 LiDAR, 2007-2016 burns)
    bh_south = bh_bound.intersection(wu_al_2017)
    all_cands.extend(generate_candidate_points(
        site_name="Bankhead_South_2017",
        domain_geom=bh_south,
        fire_polys=bh_fire,
        mech_polys=bh_mech,
        epoch1_yr=2016,
        epoch2_yr=None,
        ept_primary="USGS_LPC_AL_25Co_B1_2017",
        ept_baseline=None,
        min_spacing_m=220.0,
        target_per_bin=35,
        rng_seed=202,
    ))

    # 3. Daniel Boone East — BACI 2017 -> 2023 (6.0 yr QL2 -> QL2 leaf-off LiDAR!)
    db_east_baci = db_bound.intersection(wu_ky_2023)
    all_cands.extend(generate_candidate_points(
        site_name="DanielBoone_East_BACI",
        domain_geom=db_east_baci,
        fire_polys=db_fire,
        mech_polys=db_mech,
        epoch1_yr=2017,
        epoch2_yr=2022,
        ept_primary="KY_CentralEast_3_A23",
        ept_baseline="KY_FullState",
        min_spacing_m=220.0,
        target_per_bin=30,
        rng_seed=303,
    ))

    # 4. Daniel Boone Main — Cross-sectional dose-response (2017 QL2 LiDAR, 2006-2016 burns)
    all_cands.extend(generate_candidate_points(
        site_name="DanielBoone_Main_2017",
        domain_geom=db_bound,
        fire_polys=db_fire,
        mech_polys=db_mech,
        epoch1_yr=2016,
        epoch2_yr=None,
        ept_primary="KY_FullState",
        ept_baseline=None,
        min_spacing_m=250.0,
        target_per_bin=35,
        rng_seed=404,
    ))

    # 5. Sewanee Domain + Franklin SF (TN) — 2015/16 QL2 LiDAR
    sew_fsf = unary_union([sew_bound, fsf_bound])
    all_cands.extend(generate_candidate_points(
        site_name="Sewanee_Franklin_2015",
        domain_geom=sew_fsf,
        fire_polys=sew_fire,
        mech_polys=[],
        epoch1_yr=2015,
        epoch2_yr=None,
        ept_primary="USGS_LPC_TN_27County_blk4_2015_LAS_2018",
        ept_baseline=None,
        min_spacing_m=180.0,
        target_per_bin=30,
        rng_seed=505,
    ))

    print(f"\n=== Filtering {len(all_cands)} candidates on GEE for pure deciduous upland forest ===", flush=True)
    final_plots = filter_pure_deciduous_and_topo_gee(all_cands, max_per_stratum=35)

    with open("data/lidar/sample_plots.json", "w") as f:
        json.dump(final_plots, f, indent=1)
    print(f"Wrote data/lidar/sample_plots.json with {len(final_plots)} pure-deciduous plots", flush=True)


if __name__ == "__main__":
    main()
