"""Phase 1: Complete Optical Diagnostic Audit & Bug-Fixed Regional/8-Area Baseline.

Runs live against Google Earth Engine (configured via EE_PROJECT env var) and writes:
  1. data/diagnostic_optical.json
     - Inter-sensor winter/summer overlap offsets (L5 vs L7 2000-2011; L7 vs L8 2013-2021)
       across NDVI, NBR, NDMI on Sewanee and Regional Plateau Top.
     - Unharmonized vs Roy (2016) RMA vs Empirical Overlap-Harmonized vs Single-Sensor
       (L5, L7, L8+9) trends on Sewanee & Regional Plateau Top.
     - Residual L5 (1985-2011) deciduous rise attribution:
         (a) Stand-age / pre-2000 forest recovery via Annual NLCD 1985 -> 2001 -> 2021
         (b) Winter climate (Daymet V4 Jan-Feb & Dec-Mar tmin, tmax, SWE, frost days)
         (c) Month-by-month (Dec, Jan, Feb, Mar) & spectral index (NDVI, NBR, NDMI) split
     - Sewanee burn-cohort persistence test (2013-18 ONLY vs 2019-21 vs 2022-25 vs >=2x)
  2. data/area_metrics.json
     - All 8 Tennessee comparison properties (with fixed full Fall Creek Falls boundary,
       consistent >=430 m plateau-top masking, consistent stable-forest denominators,
       Annual NLCD 2001/2021 composition, unharmonized vs harmonized vs single-sensor trends,
       and MTBS/MODIS documented burn overlay).
  3. data/region_metrics.json
     - Bug-fixed regional plateau-top (>=430 m) summary with Annual NLCD 1985/2001/2021/2024,
       Hansen gross loss, and calibrated multi-era trends.
"""

from __future__ import annotations

import json
import os
import sys

import ee

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import plateau_lib as P


def measure_sensor_overlap(geom: ee.Geometry, mask: ee.Image, scale: int = 30) -> dict:
    """Measure exact pixel-matched inter-sensor differences on overlapping calendar years."""
    l5 = P.landsat_sensor_collection(geom, "L5")
    l7 = P.landsat_sensor_collection(geom, "L7")
    l8 = P.landsat_sensor_collection(geom, "L8")

    seasons = {
        "dec_mar": (12, 3),
        "jan_feb": (1, 2),
        "march": (3, 3),
        "jun_aug": (6, 8),
        "oct_nov": (10, 11),
    }
    out = {}
    for skey, (m1, m2) in seasons.items():
        s5_00_11 = P.annual_seasonal_composites(l5, 2000, 2011, m1, m2, "NDVI").select("val").median().updateMask(mask)
        s7_00_11 = P.annual_seasonal_composites(l7, 2000, 2011, m1, m2, "NDVI").select("val").median().updateMask(mask)
        s7_13_21 = P.annual_seasonal_composites(l7, 2013, 2021, m1, m2, "NDVI").select("val").median().updateMask(mask)
        s8_13_21 = P.annual_seasonal_composites(l8, 2013, 2021, m1, m2, "NDVI").select("val").median().updateMask(mask)

        # Evaluate on common valid mask for each pair
        m57 = s5_00_11.mask().And(s7_00_11.mask())
        m78 = s7_13_21.mask().And(s8_13_21.mask())
        v5 = P.mean_value(s5_00_11.updateMask(m57), geom, "val", scale=scale).getInfo()
        v7a = P.mean_value(s7_00_11.updateMask(m57), geom, "val", scale=scale).getInfo()
        v7b = P.mean_value(s7_13_21.updateMask(m78), geom, "val", scale=scale).getInfo()
        v8 = P.mean_value(s8_13_21.updateMask(m78), geom, "val", scale=scale).getInfo()
        out[skey] = {
            "L5_2000_2011": round(v5, 4),
            "L7_2000_2011": round(v7a, 4),
            "L7_minus_L5": round(v7a - v5, 4),
            "L7_2013_2021": round(v7b, 4),
            "L8_2013_2021": round(v8, 4),
            "L8_minus_L7": round(v8 - v7b, 4),
            "total_L5_to_L8": round((v7a - v5) + (v8 - v7b), 4),
        }
        print(f"  overlap {skey}: L7-L5={v7a-v5:+.4f}, L8-L7={v8-v7b:+.4f}, total={(v7a-v5)+(v8-v7b):+.4f}", flush=True)
    return out


def diagnose_l5_residual(region: ee.Geometry, plateau: ee.Image, scale: int = 150) -> dict:
    """Solve the open diagnostic: what drove the +0.012 to +0.014/dec single-sensor
    L5 (1985-2011) deciduous winter NDVI rise, and why did it flatten in L7 and reverse in L8+9?
    Tests: (1) pre-2000 stand age/regrowth via Annual NLCD 1985->2001->2021,
           (2) Daymet V4 winter climate (tmin, tmax, SWE, frost days),
           (3) Month-by-month (Dec, Jan, Feb, Mar) & spectral index (NDVI, NBR, NDMI).
    """
    print("\n--- Diagnosing residual L5 (1985-2011) deciduous winter trend ---", flush=True)
    gfc = ee.Image(P.HANSEN_ASSET).clip(region)
    stable = gfc.select("treecover2000").gte(P.FOREST_THRESHOLD).And(gfc.select("loss").Not()).And(plateau)
    lc85 = P.annual_nlcd_image(1985).clip(region)
    lc01 = P.annual_nlcd_image(2001).clip(region)
    lc21 = P.annual_nlcd_image(2021).clip(region)

    # Cohort 1: Continuous old deciduous (NLCD 41 in 1985, 2001, AND 2021)
    old_decid = stable.And(lc85.eq(41)).And(lc01.eq(41)).And(lc21.eq(41))
    # Cohort 2: Regrowing deciduous (non-forest in 1985 -> NLCD 41 in 2001 & 2021)
    regrow_decid = stable.And(lc85.lt(41).Or(lc85.gt(43))).And(lc01.eq(41)).And(lc21.eq(41))
    # Cohort 3: Old evergreen (NLCD 42 in 1985, 2001, 2021)
    old_ev = stable.And(lc85.eq(42)).And(lc01.eq(42)).And(lc21.eq(42))
    # Cohort 4: Regrowing evergreen / 1980s-90s pine plantations (non-forest 1985 -> NLCD 42 in 2001 & 2021)
    regrow_ev = stable.And(lc85.lt(41).Or(lc85.gt(43))).And(lc01.eq(42)).And(lc21.eq(42))
    # Cohort 5: Pure deciduous 2001 & 2021 (regardless of 1985)
    pure_decid = stable.And(lc01.eq(41)).And(lc21.eq(41))

    l5 = P.landsat_sensor_collection(region, "L5")
    l7 = P.landsat_sensor_collection(region, "L7")
    l89 = P.landsat_sensor_collection(region, "L8").merge(P.landsat_sensor_collection(region, "L9"))

    cohorts = {
        "old_deciduous_1985_2021": old_decid,
        "regrown_deciduous_1985_2001": regrow_decid,
        "pure_deciduous_2001_2021": pure_decid,
        "old_evergreen_1985_2021": old_ev,
        "regrown_evergreen_1985_2001": regrow_ev,
    }
    cohort_results = {}
    for cname, cmask in cohorts.items():
        area_km2 = P.masked_area_ha(cmask, region, scale=90, tile_scale=12).divide(100).getInfo()
        s5_jf = P.mean_value(P.sens_slope_decade(l5, 1985, 2011, 1, 2, cmask, "NDVI"), region, scale=scale).getInfo()
        s5_dm = P.mean_value(P.sens_slope_decade(l5, 1985, 2011, 12, 3, cmask, "NDVI"), region, scale=scale).getInfo()
        s7_jf = P.mean_value(P.sens_slope_decade(l7, 2000, 2021, 1, 2, cmask, "NDVI"), region, scale=scale).getInfo()
        s89_jf = P.mean_value(P.sens_slope_decade(l89, 2014, 2025, 1, 2, cmask, "NDVI"), region, scale=scale).getInfo()
        cohort_results[cname] = {
            "area_km2": round(area_km2, 1),
            "L5_1985_2011_JF": round(s5_jf, 4),
            "L5_1985_2011_DM": round(s5_dm, 4),
            "L7_2000_2021_JF": round(s7_jf, 4),
            "L89_2014_2025_JF": round(s89_jf, 4),
        }
        print(f"  cohort {cname} ({area_km2:.0f} km2): L5_JF={s5_jf:+.4f}, L7_JF={s7_jf:+.4f}, L89_JF={s89_jf:+.4f}", flush=True)

    # Monthly & spectral index decomposition on old_deciduous_1985_2021
    monthly = {}
    for m_name, m_num in [("dec", 12), ("jan", 1), ("feb", 2), ("mar", 3)]:
        s5_m = P.mean_value(P.sens_slope_decade(l5, 1985, 2011, m_num, m_num, old_decid, "NDVI"), region, scale=scale).getInfo()
        s89_m = P.mean_value(P.sens_slope_decade(l89, 2014, 2025, m_num, m_num, old_decid, "NDVI"), region, scale=scale).getInfo()
        monthly[m_name] = {"L5_1985_2011": round(s5_m, 4), "L89_2014_2025": round(s89_m, 4)}
        print(f"  month {m_name} (old decid): L5={s5_m:+.4f}, L89={s89_m:+.4f}", flush=True)

    indices = {}
    for idx in ("NDVI", "NBR", "NDMI"):
        s5_i = P.mean_value(P.sens_slope_decade(l5, 1985, 2011, 1, 2, old_decid, idx), region, scale=scale).getInfo()
        s89_i = P.mean_value(P.sens_slope_decade(l89, 2014, 2025, 1, 2, old_decid, idx), region, scale=scale).getInfo()
        indices[idx] = {"L5_1985_2011_JF": round(s5_i, 4), "L89_2014_2025_JF": round(s89_i, 4)}
        print(f"  index {idx} (old decid JF): L5={s5_i:+.4f}, L89={s89_i:+.4f}", flush=True)

    # Daymet V4 winter climate trends (1985-2011 vs 2014-2023) over plateau top
    daymet = ee.ImageCollection("NASA/ORNL/DAYMET_V4").filterBounds(region)

    def daymet_era(y1: int, y2: int, m1: int, m2: int) -> dict:
        yrs = ee.List.sequence(y1, y2)

        def yr_stats(y):
            y = ee.Number(y).toInt()
            d1 = ee.Date.fromYMD(y, m1, 1)
            d2 = ee.Date.fromYMD(y, m2, 1).advance(1, "month")
            sub = daymet.filterDate(d1, d2)
            tmin = sub.select("tmin").mean()
            tmax = sub.select("tmax").mean()
            swe = sub.select("swe").mean()
            snow_days = sub.map(lambda im: im.select("swe").gt(0)).sum()
            frost_days = sub.map(lambda im: im.select("tmin").lt(0)).sum()
            img = ee.Image.cat([
                tmin.rename("tmin"), tmax.rename("tmax"), swe.rename("swe"),
                snow_days.rename("snow_days"), frost_days.rename("frost_days"),
            ]).updateMask(plateau)
            st = img.reduceRegion(ee.Reducer.mean(), region, 1000, maxPixels=1e9, tileScale=4)
            return ee.Feature(None, st.set("year", y))

        fc = ee.FeatureCollection(yrs.map(yr_stats)).getInfo()
        rows = [f["properties"] for f in fc["features"]]
        # Compute simple linear & Theil-Sen slopes per decade in Python
        import numpy as np
        from scipy.stats import theilslopes

        out = {"n_years": len(rows)}
        t = np.array([r["year"] for r in rows], dtype=float)
        for var in ("tmin", "tmax", "swe", "snow_days", "frost_days"):
            vals = np.array([r[var] for r in rows], dtype=float)
            sl, _, _, _ = theilslopes(vals, t)
            out[var] = {
                "mean": round(float(np.mean(vals)), 3),
                "slope_per_decade": round(float(sl * 10.0), 3),
            }
        return out

    climate = {
        "JF_1985_2011": daymet_era(1985, 2011, 1, 2),
        "JF_2014_2023": daymet_era(2014, 2023, 1, 2),
        "Mar_1985_2011": daymet_era(1985, 2011, 3, 3),
        "Mar_2014_2023": daymet_era(2014, 2023, 3, 3),
    }
    print("  Daymet JF 1985-2011:", climate["JF_1985_2011"], flush=True)
    print("  Daymet Mar 1985-2011:", climate["Mar_1985_2011"], flush=True)
    return {"cohorts": cohort_results, "monthly_old_decid": monthly, "indices_old_decid": indices, "daymet_climate": climate}


def sewanee_burn_persistence(sewanee_geom: ee.Geometry) -> dict:
    """Reproduce and save the Sewanee burn-cohort persistence test on plateau-top (>=530 m)
    pure NLCD 41 deciduous forest using pure OLI (2014 -> 2022-25) & S2 (2022-25)."""
    print("\n--- Running Sewanee burn-cohort persistence test ---", flush=True)
    burns_path = os.path.join("..", "initial_study", "burns.geojson")
    with open(burns_path) as f:
        burns_fc = json.load(f)

    # Build rasterized burn-count and last-burn-year images from burns.geojson
    ee_feats = []
    for f in burns_fc["features"]:
        yr = f["properties"].get("Year")
        if yr is not None:
            ee_feats.append(ee.Feature(ee.Geometry(f["geometry"]), {"Year": int(yr), "one": 1}))
    fc = ee.FeatureCollection(ee_feats)

    burn_count = fc.reduceToImage(["one"], ee.Reducer.sum()).unmask(0).clip(sewanee_geom)
    # Count burns strictly in windows
    def has_burn(y1: int, y2: int) -> ee.Image:
        sub = fc.filter(ee.Filter.gte("Year", y1)).filter(ee.Filter.lte("Year", y2))
        return sub.reduceToImage(["one"], ee.Reducer.sum()).unmask(0).gt(0).clip(sewanee_geom)

    dem = P.get_dem(use_srtm=False).clip(sewanee_geom)
    top_decid = P.pure_deciduous_mask(sewanee_geom, plateau_only=False).And(dem.gte(530))

    cohorts = {
        "control_never_burned": top_decid.And(burn_count.eq(0)),
        "burned_2013_2018_only": top_decid.And(burn_count.eq(1)).And(has_burn(2013, 2018)),
        "burned_2019_2021_only": top_decid.And(burn_count.eq(1)).And(has_burn(2019, 2021)),
        "burned_2022_2025_only": top_decid.And(burn_count.eq(1)).And(has_burn(2022, 2025)),
        "burned_2plus_times": top_decid.And(burn_count.gte(2)),
    }

    l89 = P.landsat_sensor_collection(sewanee_geom, "L8").merge(P.landsat_sensor_collection(sewanee_geom, "L9"))
    oli_2014 = P.annual_seasonal_composites(l89, 2014, 2014, 1, 2, "NDVI").select("val").median()
    oli_2225 = P.annual_seasonal_composites(l89, 2022, 2025, 1, 2, "NDVI").select("val").median()

    def s2_jf(y):
        def mk(im):
            scl = im.select("SCL")
            good = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
            return im.updateMask(good)
        return (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(sewanee_geom)
            .filterDate(f"{y}-01-01", f"{y}-02-28")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
            .map(mk)
            .median()
            .normalizedDifference(["B8", "B4"])
            .rename("val")
        )

    s2_2225 = ee.ImageCollection([s2_jf(y) for y in (2022, 2023, 2024, 2025)]).median()

    out = {}
    for name, mask in cohorts.items():
        ha = P.masked_area_ha(mask, sewanee_geom, scale=30).getInfo()
        v14 = P.mean_value(oli_2014.updateMask(mask), sewanee_geom, "val", scale=30).getInfo()
        v25 = P.mean_value(oli_2225.updateMask(mask), sewanee_geom, "val", scale=30).getInfo()
        vs2 = P.mean_value(s2_2225.updateMask(mask), sewanee_geom, "val", scale=30).getInfo()
        out[name] = {
            "area_ha": round(ha, 1),
            "oli_2014_jf": round(v14, 4) if v14 is not None else None,
            "oli_2022_25_jf": round(v25, 4) if v25 is not None else None,
            "delta_oli": round(v25 - v14, 4) if (v14 is not None and v25 is not None) else None,
            "s2_2022_25_jf": round(vs2, 4) if vs2 is not None else None,
        }
        print(f"  {name}: {ha:.1f} ha | OLI 2014={v14:.4f} -> 2022-25={v25:.4f} (Δ={v25-v14:+.4f}) | S2={vs2:.4f}", flush=True)
    return out


def run_8_areas(areas) -> dict:
    """Compute bug-fixed metrics for all 8 Tennessee comparison properties."""
    print("\n--- Computing bug-fixed 8-area metrics ---", flush=True)
    results = {}
    mtbs = ee.FeatureCollection("USFS/GTAC/MTBS/burned_area_boundaries/v1")

    for props, geom in areas:
        name = props["name"]
        print(f"\n=== {name} ({props['fire']}) ===", flush=True)
        total_ha = geom.area(30).divide(1e4).getInfo()
        plat_ha = P.masked_area_ha(P.plateau_mask(geom, use_srtm=False), geom, scale=30).getInfo()
        pct_plateau = round(100.0 * plat_ha / total_ha, 1) if total_ha > 0 else 0.0

        # Plateau-top (>=430 m) forest & loss
        fha_e, lha_e, grp = P.forest_and_loss(geom, plateau_only=True, use_srtm=False)
        fha = fha_e.getInfo()
        lha = lha_e.getInfo()
        by_year = {int(d["yr"]) + 2000: round(d["sum"], 1) for d in ee.List(grp).getInfo()} if grp else {}

        # Authoritative Annual NLCD 2021 composition on plateau-top stable forest
        comp = P.nlcd_forest_composition(geom, year=2021, plateau_only=True, use_srtm=False)
        dec_ha = comp["deciduous_ha"].getInfo()
        ev_ha = comp["evergreen_ha"].getInfo()
        mix_ha = comp["mixed_ha"].getInfo()
        stab_ha = comp["stable_ha"].getInfo()
        tot_nlcd = max(dec_ha + ev_ha + mix_ha, 1e-6)

        # Single-sensor + harmonized + unharmonized baseline trends on plateau-top pure deciduous (NLCD 41)
        pdec = P.pure_deciduous_mask(geom, plateau_only=True, use_srtm=False)
        stab = P.stable_forest_mask(geom, plateau_only=True, use_srtm=False)
        l5 = P.landsat_sensor_collection(geom, "L5")
        l7 = P.landsat_sensor_collection(geom, "L7")
        l89 = P.landsat_sensor_collection(geom, "L8").merge(P.landsat_sensor_collection(geom, "L9"))
        unharm = l5.merge(l7).merge(l89)
        harm_jf = P.empirical_harmonized_collection(geom, "jan_feb")

        t_unharm_dm = P.mean_value(P.sens_slope_decade(unharm, 1985, 2025, 12, 3, stab, "NDVI"), geom, scale=30).getInfo()
        t_harm_jf = P.mean_value(P.sens_slope_decade(harm_jf, 1985, 2025, 1, 2, pdec, "NDVI"), geom, scale=30).getInfo()
        t_l5_jf = P.mean_value(P.sens_slope_decade(l5, 1985, 2011, 1, 2, pdec, "NDVI"), geom, scale=30).getInfo()
        t_l7_jf = P.mean_value(P.sens_slope_decade(l7, 2000, 2021, 1, 2, pdec, "NDVI"), geom, scale=30).getInfo()
        t_l89_jf = P.mean_value(P.sens_slope_decade(l89, 2014, 2025, 1, 2, pdec, "NDVI"), geom, scale=30).getInfo()
        t_l89_grow = P.mean_value(P.sens_slope_decade(l89, 2014, 2025, 6, 8, pdec, "NDVI"), geom, scale=30).getInfo()

        # Check MTBS large fires intersecting polygon
        mtbs_sub = mtbs.filterBounds(geom)
        mtbs_n = mtbs_sub.size().getInfo()

        # MODIS burned area (2001-2024) fraction
        modis_ba = (
            ee.ImageCollection("MODIS/061/MCD64A1")
            .filterDate("2001-01-01", "2024-12-31")
            .select("BurnDate")
            .map(lambda im: im.gt(0))
            .sum()
            .gt(0)
            .clip(geom)
        )
        modis_ha = P.masked_area_ha(modis_ba.And(P.plateau_mask(geom, use_srtm=False)), geom, scale=500).getInfo()

        results[name] = {
            "fire": props["fire"],
            "kind": props["kind"],
            "source": props.get("source", ""),
            "total_ha": round(total_ha, 1),
            "plateau_ha": round(plat_ha, 1),
            "pct_plateau": pct_plateau,
            "forest_ha": round(fha, 1),
            "stable_forest_ha": round(stab_ha, 1),
            "loss_ha": round(lha, 1),
            "pct_loss": round(100.0 * lha / fha, 2) if fha > 0 else 0.0,
            "loss_by_year": by_year,
            "nlcd_pct_deciduous": round(100.0 * dec_ha / tot_nlcd, 1),
            "nlcd_pct_evergreen": round(100.0 * ev_ha / tot_nlcd, 1),
            "nlcd_pct_mixed": round(100.0 * mix_ha / tot_nlcd, 1),
            "evergreen_ha": round(ev_ha, 1),
            "pct_evergreen": round(100.0 * ev_ha / max(stab_ha, 1e-6), 2),
            "unharm_dec_mar_decade": round(t_unharm_dm, 4),
            "harm_jan_feb_decid_decade": round(t_harm_jf, 4),
            "l5_jan_feb_decid_decade": round(t_l5_jf, 4),
            "l7_jan_feb_decid_decade": round(t_l7_jf, 4),
            "l89_jan_feb_decid_decade": round(t_l89_jf, 4),
            "l89_jun_aug_decid_decade": round(t_l89_grow, 4),
            "mtbs_fire_count": int(mtbs_n),
            "modis_burned_ha": round(modis_ha, 1),
        }
        print(
            f"  total={total_ha:.0f} ha ({pct_plateau:.1f}% >=430m) | decid={results[name]['nlcd_pct_deciduous']:.1f}% ev={results[name]['nlcd_pct_evergreen']:.1f}% | "
            f"unharm={t_unharm_dm:+.4f} harm={t_harm_jf:+.4f} L5={t_l5_jf:+.4f} L7={t_l7_jf:+.4f} L89={t_l89_jf:+.4f} | MTBS={mtbs_n}",
            flush=True,
        )
    return results


def run_region(region: ee.Geometry) -> dict:
    """Compute bug-fixed regional metrics over the Greater Cumberland Plateau (>= 430 m)."""
    print("\n--- Computing bug-fixed regional metrics ---", flush=True)
    plateau = P.plateau_mask(region, use_srtm=True)
    region_km2 = region.area(30).divide(1e6).getInfo()
    plateau_km2 = P.masked_area_ha(plateau, region, scale=30, tile_scale=12).divide(100).getInfo()
    fha_e, lha_e, grp = P.forest_and_loss(region, plateau_only=True, use_srtm=True, scale=30, tile_scale=12)
    forest_top_ha = fha_e.getInfo()
    loss_ha = lha_e.getInfo()
    by_year_ha = {int(d["yr"]) + 2000: round(d["sum"], 1) for d in ee.List(grp).getInfo()} if grp else {}
    by_year_km2 = {k: round(v / 100.0, 2) for k, v in by_year_ha.items()}

    # Annual NLCD C1.0 consistent forest area (1985, 2001, 2021, 2024)
    nlcd_forest = {}
    for y in (1985, 2001, 2021, 2024):
        lc = P.annual_nlcd_image(y).clip(region)
        fmask = lc.gte(41).And(lc.lte(43)).And(plateau)
        nlcd_forest[str(y)] = round(P.masked_area_ha(fmask, region, scale=30, tile_scale=12).divide(100).getInfo(), 1)
    print("  Annual NLCD plateau-top forest (km2):", nlcd_forest, flush=True)

    # Composition on 2021 plateau-top stable forest (NLCD)
    comp = P.nlcd_forest_composition(region, year=2021, plateau_only=True, use_srtm=True, scale=30, tile_scale=12)
    dec_ha = comp["deciduous_ha"].getInfo()
    ev_ha = comp["evergreen_ha"].getInfo()
    mix_ha = comp["mixed_ha"].getInfo()
    tot_nlcd = max(dec_ha + ev_ha + mix_ha, 1e-6)

    # Regional trends (150 m reduce)
    RSCALE = 150
    stab = P.stable_forest_mask(region, plateau_only=True, use_srtm=True)
    pdec = P.pure_deciduous_mask(region, plateau_only=True, use_srtm=True)
    pev = P.pure_evergreen_mask(region, plateau_only=True, use_srtm=True)

    l5 = P.landsat_sensor_collection(region, "L5")
    l7 = P.landsat_sensor_collection(region, "L7")
    l89 = P.landsat_sensor_collection(region, "L8").merge(P.landsat_sensor_collection(region, "L9"))
    unharm = l5.merge(l7).merge(l89)
    roy = (
        P.landsat_sensor_collection(region, "L5", roy_rma=True)
        .merge(P.landsat_sensor_collection(region, "L7", roy_rma=True))
        .merge(l89)
    )
    harm_dm = P.empirical_harmonized_collection(region, "dec_mar")
    harm_jf = P.empirical_harmonized_collection(region, "jan_feb")

    trends = {
        "unharm_dec_mar_all": round(P.mean_value(P.sens_slope_decade(unharm, 1985, 2025, 12, 3, stab), region, scale=RSCALE).getInfo(), 4),
        "roy_dec_mar_all": round(P.mean_value(P.sens_slope_decade(roy, 1985, 2025, 12, 3, stab), region, scale=RSCALE).getInfo(), 4),
        "harm_dec_mar_all": round(P.mean_value(P.sens_slope_decade(harm_dm, 1985, 2025, 12, 3, stab), region, scale=RSCALE).getInfo(), 4),
        "harm_jan_feb_decid": round(P.mean_value(P.sens_slope_decade(harm_jf, 1985, 2025, 1, 2, pdec), region, scale=RSCALE).getInfo(), 4),
        "l5_jan_feb_all": round(P.mean_value(P.sens_slope_decade(l5, 1985, 2011, 1, 2, stab), region, scale=RSCALE).getInfo(), 4),
        "l5_jan_feb_decid": round(P.mean_value(P.sens_slope_decade(l5, 1985, 2011, 1, 2, pdec), region, scale=RSCALE).getInfo(), 4),
        "l5_jan_feb_evergreen": round(P.mean_value(P.sens_slope_decade(l5, 1985, 2011, 1, 2, pev), region, scale=RSCALE).getInfo(), 4),
        "l7_jan_feb_decid": round(P.mean_value(P.sens_slope_decade(l7, 2000, 2021, 1, 2, pdec), region, scale=RSCALE).getInfo(), 4),
        "l89_jan_feb_all": round(P.mean_value(P.sens_slope_decade(l89, 2014, 2025, 1, 2, stab), region, scale=RSCALE).getInfo(), 4),
        "l89_jan_feb_decid": round(P.mean_value(P.sens_slope_decade(l89, 2014, 2025, 1, 2, pdec), region, scale=RSCALE).getInfo(), 4),
        "l89_jan_feb_evergreen": round(P.mean_value(P.sens_slope_decade(l89, 2014, 2025, 1, 2, pev), region, scale=RSCALE).getInfo(), 4),
        "l89_jun_aug_decid": round(P.mean_value(P.sens_slope_decade(l89, 2014, 2025, 6, 8, pdec), region, scale=RSCALE).getInfo(), 4),
    }
    print("  Regional trends:", trends, flush=True)

    f01 = nlcd_forest["2001"]
    f21 = nlcd_forest["2021"]
    f24 = nlcd_forest["2024"]
    return {
        "region_km2": round(region_km2),
        "plateau_km2": round(plateau_km2),
        "forest_top_km2": round(forest_top_ha / 100.0, 1),
        "pct_forested_top": round(100.0 * (forest_top_ha / 100.0) / plateau_km2, 1),
        "nlcd_type_deciduous": round(100.0 * dec_ha / tot_nlcd, 1),
        "nlcd_type_mixed": round(100.0 * mix_ha / tot_nlcd, 1),
        "nlcd_type_evergreen": round(100.0 * ev_ha / tot_nlcd, 1),
        "loss_km2": round(loss_ha / 100.0, 1),
        "pct_loss": round(100.0 * loss_ha / forest_top_ha, 2),
        "loss_by_year_ha": by_year_ha,
        "loss_by_year_km2": by_year_km2,
        "annual_nlcd_forest_km2": nlcd_forest,
        "nlcd_forest_2001_km2": f01,
        "nlcd_forest_2021_km2": f21,
        "nlcd_forest_2024_km2": f24,
        "nlcd_net_2001_2021_km2": round(f21 - f01, 1),
        "nlcd_net_2001_2021_pct": round(100.0 * (f21 - f01) / f01, 2),
        "nlcd_net_2001_2024_km2": round(f24 - f01, 1),
        "nlcd_net_2001_2024_pct": round(100.0 * (f24 - f01) / f01, 2),
        "trends": trends,
    }


def main() -> None:
    P.init()
    areas = P.load_areas("data/areas.geojson")
    with open("data/region.geojson") as f:
        region = ee.Geometry(json.load(f)["features"][0]["geometry"])
    plateau = P.plateau_mask(region, use_srtm=True)

    sewanee_geom = next(g for p, g in areas if p["name"] == "Sewanee Domain")

    if "--skip-diag" not in sys.argv:
        print("=== 1. Measuring Sewanee inter-sensor overlaps ===", flush=True)
        sew_overlap = measure_sensor_overlap(sewanee_geom, P.stable_forest_mask(sewanee_geom, plateau_only=False), scale=30)

        print("=== 2. Measuring Regional plateau-top inter-sensor overlaps ===", flush=True)
        reg_overlap = measure_sensor_overlap(region, P.pure_deciduous_mask(region, plateau_only=True, use_srtm=True), scale=150)

        l5_diag = diagnose_l5_residual(region, plateau, scale=150)
        sew_persist = sewanee_burn_persistence(sewanee_geom)

        diag = {
            "sewanee_sensor_overlaps": sew_overlap,
            "regional_deciduous_sensor_overlaps": reg_overlap,
            "l5_residual_attribution": l5_diag,
            "sewanee_burn_persistence": sew_persist,
        }
        with open("data/diagnostic_optical.json", "w") as f:
            json.dump(diag, f, indent=2)
        print("Wrote data/diagnostic_optical.json", flush=True)

    area_res = run_8_areas(areas)
    with open("data/area_metrics.json", "w") as f:
        json.dump(area_res, f, indent=2)
    print("Wrote data/area_metrics.json", flush=True)

    reg_res = run_region(region)
    with open("data/region_metrics.json", "w") as f:
        json.dump(reg_res, f, indent=2)
    print("Wrote data/region_metrics.json", flush=True)


if __name__ == "__main__":
    main()
