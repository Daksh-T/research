"""Phase 2B: Extract 3DEP Airborne LiDAR Vertical Structure (Cross-Sectional +
Multi-Epoch BACI + Density-Matched Sensitivity) and Matched Multi-Season
Landsat Optical + GEDI L2B Metrics for Every Sampled Plot.

Reads:  data/lidar/sample_plots.json
Writes: data/lidar/plot_metrics.json (incrementally checkpointed)
"""

from __future__ import annotations

import json
import os
import struct
import sys
import time
from collections import defaultdict

import ee
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import lidar_ept as L
import plateau_lib as P


def extract_all_lidar(plots: list[dict], out_path: str, radius_m: float = 30.0) -> dict[str, dict]:
    """Stream 3DEP EPT point clouds per dataset (maximizing octree node cache hits)
    and compute vertical structure + BACI density-matched sensitivity checks.
    """
    results: dict[str, dict] = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        print(f"Resuming from {out_path} with {len(results)} completed plots", flush=True)

    # Group by (ept_primary, ept_baseline) so each EPT client reuses its octree cache
    def _has_full_metrics(m: dict | None) -> bool:
        return bool(m) and ("first_return_canopy_10m" in m) and ("nrd_1m" in m)

    groups = defaultdict(list)
    for pt in plots:
        pid = pt["plot_id"]
        if (
            pid in results
            and _has_full_metrics(results[pid].get("lidar_primary"))
            and (pt["ept_baseline"] is None or _has_full_metrics(results[pid].get("lidar_baseline")))
        ):
            continue
        groups[(pt["ept_primary"], pt["ept_baseline"])].append(pt)

    def _unpack_query(qres):
        if len(qres) == 6:
            return qres
        x, y, z, c, gt = qres
        return x, y, z, c, None, gt

    for (ds_prim, ds_base), grp in groups.items():
        print(f"\n=== Extracting 3DEP LiDAR: primary={ds_prim}, baseline={ds_base} ({len(grp)} plots) ===", flush=True)
        # Sort spatially (lon, lat) so consecutive plots share nearby EPT octree nodes!
        grp.sort(key=lambda p: (round(p["lon"], 2), p["lat"]))

        cli_prim = L.EPTClient(ds_prim, max_workers=12)
        cli_base = L.EPTClient(ds_base, max_workers=12) if ds_base else None

        t_start = time.time()
        for idx, pt in enumerate(grp, 1):
            pid = pt["plot_id"]
            rec = results.get(pid, dict(pt))
            try:
                xp, yp, zp, cp, rnp, gtp = _unpack_query(cli_prim.query_plot_points(pt["lon"], pt["lat"], radius_m=radius_m))
                mp = L.compute_plot_metrics(xp, yp, zp, cp, pt["lon"], pt["lat"], radius_m=radius_m, gps_time=gtp, rn=rnp)
                if mp is None:
                    print(f"  [SKIP] {pid}: insufficient primary points ({len(xp)})", flush=True)
                    continue
                rec["lidar_primary"] = mp

                if cli_base is not None:
                    xb, yb, zb, cb, rnb, gtb = _unpack_query(cli_base.query_plot_points(pt["lon"], pt["lat"], radius_m=radius_m))
                    mb = L.compute_plot_metrics(xb, yb, zb, cb, pt["lon"], pt["lat"], radius_m=radius_m, gps_time=gtb, rn=rnb)
                    if mb is None:
                        print(f"  [SKIP BACI] {pid}: insufficient baseline points ({len(xb)})", flush=True)
                        continue
                    rec["lidar_baseline"] = mb

                    # Pulse-density invariance check: subsample the denser epoch down to the exact
                    # pulse density of the sparser epoch and recompute NRD!
                    d_p = mp["pulse_density"]
                    d_b = mb["pulse_density"]
                    rng = np.random.default_rng(abs(hash(pid)) % (2**32))
                    if d_p > d_b * 1.05:
                        frac = d_b / d_p
                        mp_thin = L.compute_plot_metrics(
                            xp, yp, zp, cp, pt["lon"], pt["lat"], radius_m=radius_m,
                            gps_time=gtp, subsample_frac=frac, rng=rng, rn=rnp,
                        )
                        rec["lidar_primary_thinned"] = mp_thin
                        rec["lidar_baseline_thinned"] = mb
                    elif d_b > d_p * 1.05:
                        frac = d_p / d_b
                        mb_thin = L.compute_plot_metrics(
                            xb, yb, zb, cb, pt["lon"], pt["lat"], radius_m=radius_m,
                            gps_time=gtb, subsample_frac=frac, rng=rng, rn=rnb,
                        )
                        rec["lidar_primary_thinned"] = mp
                        rec["lidar_baseline_thinned"] = mb_thin
                    else:
                        rec["lidar_primary_thinned"] = mp
                        rec["lidar_baseline_thinned"] = mb

                results[pid] = rec
            except (OSError, ValueError, RuntimeError, KeyError, EOFError, struct.error, L.laspy.LaspyException) as e:
                print(f"  [ERR] {pid}: {e}", flush=True)
                continue

            if idx % 15 == 0 or idx == len(grp):
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=1)
                elapsed = time.time() - t_start
                print(
                    f"  [{ds_prim}] {idx}/{len(grp)} done ({elapsed:.1f}s, {elapsed/idx:.2f}s/plot) | "
                    f"sample {pid}: nrd_mid={mp['nrd_midstory']:.3f} acq={mp['acq_date']} dens={mp['pulse_density']:.1f}/m2",
                    flush=True,
                )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    return results


def _season_med(
    col: ee.ImageCollection, y1: int, y2: int, m1: int, m2: int, band: str, name: str
) -> ee.Image:
    return (
        P.annual_seasonal_composites(col, y1, y2, m1, m2, band)
        .select("val")
        .median()
        .rename(name)
    )


def attach_optical_and_gedi(results: dict[str, dict], out_path: str) -> dict[str, dict]:
    """For every plot with valid 3DEP LiDAR, extract matched Landsat 8/9 multi-season
    optical indices (winter JF, spring MA, summer JA, autumn ON; NDVI/NBR/NDMI),
    BACI optical deltas, and NASA GEDI L2B vertical PAI (2019-2023) via GEE.
    """
    P.init()
    print(f"\n=== Extracting matched Landsat 8/9 multi-season optical & GEDI L2B for {len(results)} plots ===", flush=True)

    # Group plots by site (small bounding boxes keep GEE fast)
    by_site = defaultdict(list)
    for pid, rec in results.items():
        by_site[rec["site"]].append((pid, rec))

    gedi_l2b = (
        ee.ImageCollection("LARSE/GEDI/GEDI02_B_002_MONTHLY")
        .filterDate("2019-04-01", "2023-04-01")
        .map(lambda im: im.updateMask(im.select("l2b_quality_flag").eq(1)).updateMask(im.select("degrade_flag").eq(0)))
    )

    for site, items in by_site.items():
        lons = [r["lon"] for _, r in items]
        lats = [r["lat"] for _, r in items]
        bbox = ee.Geometry.Rectangle([min(lons) - 0.05, min(lats) - 0.05, max(lons) + 0.05, max(lats) + 0.05])
        l89 = P.landsat_sensor_collection(bbox, "L8").merge(P.landsat_sensor_collection(bbox, "L9"))

        # Choose 3-year window centered around the site's primary LiDAR epoch
        e2 = items[0][1].get("epoch2_yr")
        e1 = items[0][1].get("epoch1_yr")
        ref_yr = e2 if e2 is not None else e1
        y_start = max(2014, ref_yr - 1)
        y_end = min(2025, ref_yr + 1)

        bands = [
            _season_med(l89, y_start, y_end, 1, 2, "NDVI", "opt_win_ndvi"),
            _season_med(l89, y_start, y_end, 1, 2, "NBR", "opt_win_nbr"),
            _season_med(l89, y_start, y_end, 1, 2, "NDMI", "opt_win_ndmi"),
            _season_med(l89, y_start, y_end, 3, 4, "NDVI", "opt_spr_ndvi"),
            _season_med(l89, y_start, y_end, 6, 8, "NDVI", "opt_sum_ndvi"),
            _season_med(l89, y_start, y_end, 10, 11, "NDVI", "opt_aut_ndvi"),
            # Also 2014-2016 early OLI & 2020-2023 late OLI for BACI optical differencing (winter & spring)
            _season_med(l89, 2014, 2016, 1, 2, "NDVI", "opt_early_win_ndvi"),
            _season_med(l89, 2014, 2016, 1, 2, "NBR", "opt_early_win_nbr"),
            _season_med(l89, 2020, 2023, 1, 2, "NDVI", "opt_late_win_ndvi"),
            _season_med(l89, 2020, 2023, 1, 2, "NBR", "opt_late_win_nbr"),
            _season_med(l89, 2014, 2016, 3, 4, "NDVI", "opt_early_spr_ndvi"),
            _season_med(l89, 2020, 2023, 3, 4, "NDVI", "opt_late_spr_ndvi"),
            # Epoch-matched 2016-2018 vs 2022-2024 spring windows for Daniel Boone (2017->2023) check
            _season_med(l89, 2016, 2018, 3, 4, "NDVI", "opt_2016_18_spr_ndvi"),
            _season_med(l89, 2022, 2024, 3, 4, "NDVI", "opt_2022_24_spr_ndvi"),
        ]

        opt_stack = ee.Image.cat(bands)

        # Extract optical at exact 30 m radius plot
        feats_30 = [ee.Feature(ee.Geometry.Point([r["lon"], r["lat"]]).buffer(30), {"pid": pid}) for pid, r in items]
        opt_fc = opt_stack.reduceRegions(ee.FeatureCollection(feats_30), ee.Reducer.mean(), scale=30, tileScale=4).getInfo()
        for f in opt_fc["features"]:
            pr = f["properties"]
            pid = pr["pid"]
            results[pid]["optical"] = {
                k: round(float(v), 5) if v is not None else None
                for k, v in pr.items()
                if k != "pid"
            }

        # Reuse existing GEDI L2B when already extracted for every plot in this site
        need_gedi = any("gedi" not in r for _, r in items)
        if need_gedi:
            # GEDI L2B vertical PAI/PAVD profile (0-5 m, 5-10 m, total PAI, FHD, cover)
            # Note: in GEDI02_B_002, pai_z0 = cumulative PAI >=0 m, pai_z1 >=5 m, pai_z2 >=10 m
            # so 0-5 m layer PAI = pai_z0 - pai_z1, and 5-10 m layer PAI = pai_z1 - pai_z2.
            gedi_sub = gedi_l2b.filterBounds(bbox)
            gedi_img = gedi_sub.select(["pai", "fhd_normal", "cover", "pai_z0", "pai_z1", "pai_z2", "pavd_z0", "pavd_z1"]).mean()
            pai_0_5 = gedi_img.select("pai_z0").subtract(gedi_img.select("pai_z1")).rename("gedi_pai_0_5m")
            pai_5_10 = gedi_img.select("pai_z1").subtract(gedi_img.select("pai_z2")).rename("gedi_pai_5_10m")
            pavd_0_5 = gedi_img.select("pavd_z0").rename("gedi_pavd_0_5m")
            pavd_5_10 = gedi_img.select("pavd_z1").rename("gedi_pavd_5_10m")
            pai_tot = gedi_img.select("pai").rename("gedi_pai_total")
            fhd_g = gedi_img.select("fhd_normal").rename("gedi_fhd")
            cov_g = gedi_img.select("cover").rename("gedi_cover")
            gedi_stack = ee.Image.cat([pai_0_5, pai_5_10, pavd_0_5, pavd_5_10, pai_tot, fhd_g, cov_g])

            # Extract GEDI L2B with 75 m radius (since GEDI footprints are spaced ~60 m along track)
            feats_75 = [ee.Feature(ee.Geometry.Point([r["lon"], r["lat"]]).buffer(75), {"pid": pid}) for pid, r in items]
            gedi_fc = gedi_stack.reduceRegions(ee.FeatureCollection(feats_75), ee.Reducer.mean(), scale=25, tileScale=4).getInfo()
            n_gedi = 0
            for f in gedi_fc["features"]:
                pr = f["properties"]
                pid = pr["pid"]
                if pr.get("gedi_pai_total") is not None:
                    n_gedi += 1
                results[pid]["gedi"] = {
                    k: round(float(v), 4) if v is not None else None
                    for k, v in pr.items()
                    if k != "pid"
                }
        else:
            n_gedi = sum(1 for _, r in items if (r.get("gedi") or {}).get("gedi_pai_total") is not None)
        print(f"  [{site}] attached optical ({len(items)} plots) + GEDI L2B ({n_gedi}/{len(items)} with shots)", flush=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"Saved enriched {out_path}", flush=True)
    return results


def main() -> None:
    with open("data/lidar/sample_plots.json") as f:
        plots = json.load(f)
    out_path = "data/lidar/plot_metrics.json"
    results = extract_all_lidar(plots, out_path=out_path, radius_m=30.0)
    attach_optical_and_gedi(results, out_path=out_path)


if __name__ == "__main__":
    main()
