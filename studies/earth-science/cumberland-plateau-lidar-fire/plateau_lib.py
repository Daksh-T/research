"""Standalone Earth Engine analysis library for the Cumberland Plateau Forest,
Fire, and Mesophication Study.

Fixes all confirmed methodology and implementation bugs from the v1 pipeline:
  1. Fully standalone (no broken import to ../analysis_lib.py).
  2. Uses USGS/3DEP/10m_collection (replacing deprecated USGS/3DEP/10m).
  3. Uses metric-radius TPI (450 m circle in meters) across all DEM resolutions
     so 10 m 3DEP and 30 m SRTM share identical spatial scale.
  4. Applies strict Landsat Collection 2 QA masking (Bits 1, 2, 3, 4, 5) and
     valid surface reflectance clamping (0 < SR < 1).
  5. Separates unharmonized baseline, Roy et al. (2016) RMA, empirical overlap
     harmonization (anchored to L7), and single-sensor era trends (L5 1985-2011,
     L7 2000-2021, L8+9 2014-2025) using annual median composites + Theil-Sen
     slopes rather than unharmonized scene-level OLS.
  6. Uses consistent Annual NLCD Collection 1.0 (1985-2024) instead of mixing
     2019-release and 2021-release NLCD products.
  7. Uses consistent plateau-top masking (>= 430 m) and stable-forest
     denominators for evergreen/deciduous shares.
"""

from __future__ import annotations

import json
import math
import os

import ee

EE_PROJECT = os.environ.get("EE_PROJECT", "")
HANSEN_ASSET = "UMD/hansen/global_forest_change_2025_v1_13"
ANNUAL_NLCD_COLLECTION = "projects/sat-io/open-datasets/USGS/ANNUAL_NLCD/LANDCOVER"
DEM_3DEP_COLLECTION = "USGS/3DEP/10m_collection"
DEM_SRTM = "USGS/SRTMGL1_003"
FOREST_THRESHOLD = 30
PLATEAU_ELEV_M = 430

LOSS_YEAR_PALETTE = [
    "#3b0f70", "#51127c", "#641a80", "#782281", "#8c2981",
    "#a1307e", "#b5367a", "#ca3e72", "#de4968", "#ed5a5f",
    "#f7705c", "#fc8961", "#fe9f6d", "#feb77e", "#fece91",
    "#fde725", "#c8e020", "#90d743", "#5ec962", "#35b779",
    "#21918c", "#287c8e", "#31688e", "#3b528b", "#440154",
]

# Empirically measured inter-sensor winter NDVI offsets on Cumberland Plateau
# stable forest (overlapping years: L7-L5 on 2000-2011, L8-L7 on 2013-2021).
# Default regional offsets are re-estimated per region in diagnostic scripts.
EMPIRICAL_OFFSETS_TO_L7 = {
    "dec_mar": {"L5": +0.0258, "L7": 0.0, "L8": -0.0601, "L9": -0.0601},
    "jan_feb": {"L5": +0.0331, "L7": 0.0, "L8": -0.0559, "L9": -0.0559},
    "jun_aug": {"L5": +0.0349, "L7": 0.0, "L8": +0.0027, "L9": +0.0027},
}


def init(project: str = EE_PROJECT) -> None:
    ee.Initialize(project=project)


def load_areas(path: str = "data/areas.geojson") -> list[tuple[dict, ee.Geometry]]:
    with open(path) as f:
        fc = json.load(f)
    return [(feat["properties"], ee.Geometry(feat["geometry"])) for feat in fc["features"]]


def masked_area_ha(mask_img: ee.Image, geom: ee.Geometry, scale: int = 30, tile_scale: int = 8) -> ee.Number:
    """Compute area in hectares of a binary mask inside geom."""
    area_img = ee.Image.pixelArea().divide(1e4).updateMask(mask_img)
    val = area_img.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geom,
        scale=scale,
        maxPixels=1e11,
        tileScale=tile_scale,
        bestEffort=True,
    ).get("area")
    return ee.Number(ee.Algorithms.If(val, val, 0))


def get_dem(use_srtm: bool = False) -> ee.Image:
    if use_srtm:
        return ee.Image(DEM_SRTM).select("elevation")
    col = ee.ImageCollection(DEM_3DEP_COLLECTION)
    proj = col.first().projection()
    return col.mosaic().setDefaultProjection(proj).select("elevation")


def plateau_mask(geom: ee.Geometry, min_elev_m: float = PLATEAU_ELEV_M, use_srtm: bool = False) -> ee.Image:
    return get_dem(use_srtm=use_srtm).clip(geom.buffer(500)).gte(min_elev_m)


def stable_forest_mask(geom: ee.Geometry, plateau_only: bool = True, use_srtm: bool = False) -> ee.Image:
    gfc = ee.Image(HANSEN_ASSET).clip(geom)
    m = gfc.select("treecover2000").gte(FOREST_THRESHOLD).And(gfc.select("loss").Not())
    if plateau_only:
        m = m.And(plateau_mask(geom, use_srtm=use_srtm))
    return m


def annual_nlcd_image(year: int) -> ee.Image:
    """Return Annual NLCD Collection 1.0 landcover image for a given year (1985-2024)."""
    col = ee.ImageCollection(ANNUAL_NLCD_COLLECTION)
    img = col.filterDate(f"{year}-01-01", f"{year}-12-31").first()
    return img.select("b1").rename("landcover")


def pure_deciduous_mask(geom: ee.Geometry, plateau_only: bool = True, use_srtm: bool = False) -> ee.Image:
    """Stable forest classified as Deciduous (NLCD 41) in both 2001 and 2021."""
    lc01 = annual_nlcd_image(2001)
    lc21 = annual_nlcd_image(2021)
    m = stable_forest_mask(geom, plateau_only=plateau_only, use_srtm=use_srtm)
    return m.And(lc01.eq(41)).And(lc21.eq(41))


def pure_evergreen_mask(geom: ee.Geometry, plateau_only: bool = True, use_srtm: bool = False) -> ee.Image:
    lc01 = annual_nlcd_image(2001)
    lc21 = annual_nlcd_image(2021)
    m = stable_forest_mask(geom, plateau_only=plateau_only, use_srtm=use_srtm)
    return m.And(lc01.eq(42)).And(lc21.eq(42))


def forest_and_loss(
    geom: ee.Geometry,
    plateau_only: bool = True,
    use_srtm: bool = False,
    scale: int = 30,
    tile_scale: int = 8,
) -> tuple[ee.Number, ee.Number, ee.List]:
    """Return (forest_ha, loss_ha, loss_by_year_groups) inside geom."""
    gfc = ee.Image(HANSEN_ASSET).clip(geom)
    forest = gfc.select("treecover2000").gte(FOREST_THRESHOLD)
    if plateau_only:
        forest = forest.And(plateau_mask(geom, use_srtm=use_srtm))
    loss = gfc.select("loss").And(forest)
    forest_ha = masked_area_ha(forest, geom, scale=scale, tile_scale=tile_scale)
    loss_ha = masked_area_ha(loss, geom, scale=scale, tile_scale=tile_scale)
    grp = (
        ee.Image.pixelArea()
        .divide(1e4)
        .addBands(gfc.select("lossyear"))
        .updateMask(loss)
        .reduceRegion(
            reducer=ee.Reducer.sum().group(1, "yr"),
            geometry=geom,
            scale=scale,
            maxPixels=1e11,
            tileScale=tile_scale,
            bestEffort=True,
        )
        .get("groups")
    )
    return forest_ha, loss_ha, grp


def nlcd_forest_composition(
    geom: ee.Geometry,
    year: int = 2021,
    plateau_only: bool = True,
    use_srtm: bool = False,
    scale: int = 30,
    tile_scale: int = 8,
) -> dict[str, ee.Number]:
    """Authoritative NLCD deciduous (41), evergreen (42), mixed (43) shares on stable forest."""
    lc = annual_nlcd_image(year).clip(geom)
    stable = stable_forest_mask(geom, plateau_only=plateau_only, use_srtm=use_srtm)
    return {
        "deciduous_ha": masked_area_ha(lc.eq(41).And(stable), geom, scale=scale, tile_scale=tile_scale),
        "evergreen_ha": masked_area_ha(lc.eq(42).And(stable), geom, scale=scale, tile_scale=tile_scale),
        "mixed_ha": masked_area_ha(lc.eq(43).And(stable), geom, scale=scale, tile_scale=tile_scale),
        "stable_ha": masked_area_ha(stable, geom, scale=scale, tile_scale=tile_scale),
    }


def mesic_index(geom: ee.Geometry, use_srtm: bool = False, tpi_radius_m: int = 450) -> ee.Image:
    """Continuous topographic moisture score (McCune-Keon HLI + metric-radius TPI).
    Uses a fixed metric radius (450 m) regardless of DEM resolution.
    """
    dem = get_dem(use_srtm=use_srtm).clip(geom.buffer(tpi_radius_m + 200))
    slope = ee.Terrain.slope(dem)
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
    tpi = dem.subtract(dem.focalMean(tpi_radius_m, "circle", "meters")).rename("tpi")

    def zscore(img: ee.Image, name: str) -> ee.Image:
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
            geometry=geom,
            scale=30,
            maxPixels=1e10,
            tileScale=8,
            bestEffort=True,
        )
        return img.subtract(ee.Number(stats.get(name + "_mean"))).divide(
            ee.Number(stats.get(name + "_stdDev"))
        )

    return zscore(hli, "hli").multiply(-1).add(zscore(tpi, "tpi").multiply(-1)).rename("mesic").clip(geom)


# ---- Strict Landsat Collection 2 preprocessing & harmonization -----------
def _prep_landsat_sr(img: ee.Image, nir: str, red: str, swir1: str, swir2: str, sensor: str,
                     roy_rma: bool = False) -> ee.Image:
    """Strict Landsat C2 L2 QA masking (bits 1,2,3,4,5) + valid SR bounds (0..1)."""
    qa = img.select("QA_PIXEL")
    bad = (
        qa.bitwiseAnd(1 << 1)
        .Or(qa.bitwiseAnd(1 << 2))
        .Or(qa.bitwiseAnd(1 << 3))
        .Or(qa.bitwiseAnd(1 << 4))
        .Or(qa.bitwiseAnd(1 << 5))
    )
    sr = img.select([nir, red, swir1, swir2], ["NIR", "RED", "SWIR1", "SWIR2"]).multiply(2.75e-5).add(-0.2)
    valid = (
        sr.select("NIR").gt(0).And(sr.select("NIR").lt(1))
        .And(sr.select("RED").gt(0)).And(sr.select("RED").lt(1))
        .And(sr.select("SWIR2").gt(0)).And(sr.select("SWIR2").lt(1))
    )
    sr = sr.updateMask(bad.Not()).updateMask(valid)
    ndvi = sr.normalizedDifference(["NIR", "RED"]).toFloat().rename("NDVI")
    if roy_rma and sensor in ("L5", "L7"):
        # Roy et al. (2016) Table 3 RMA transformation (ETM+/TM -> OLI)
        ndvi = ndvi.multiply(0.9589).add(0.0235).toFloat().rename("NDVI")
    nbr = sr.normalizedDifference(["NIR", "SWIR2"]).toFloat().rename("NBR")
    ndmi = sr.normalizedDifference(["NIR", "SWIR1"]).toFloat().rename("NDMI")
    return (
        ndvi.addBands([nbr, ndmi])
        .set("sensor", sensor)
        .copyProperties(img, ["system:time_start"])
    )


def landsat_sensor_collection(geom: ee.Geometry, sensor: str, roy_rma: bool = False) -> ee.ImageCollection:
    specs = {
        "L5": ("LANDSAT/LT05/C02/T1_L2", "SR_B4", "SR_B3", "SR_B5", "SR_B7"),
        "L7": ("LANDSAT/LE07/C02/T1_L2", "SR_B4", "SR_B3", "SR_B5", "SR_B7"),
        "L8": ("LANDSAT/LC08/C02/T1_L2", "SR_B5", "SR_B4", "SR_B6", "SR_B7"),
        "L9": ("LANDSAT/LC09/C02/T1_L2", "SR_B5", "SR_B4", "SR_B6", "SR_B7"),
    }
    coll_id, nir, red, sw1, sw2 = specs[sensor]
    return (
        ee.ImageCollection(coll_id)
        .filterBounds(geom)
        .map(lambda im: _prep_landsat_sr(im, nir, red, sw1, sw2, sensor, roy_rma=roy_rma))
    )


def annual_seasonal_composites(
    col: ee.ImageCollection,
    start_year: int,
    end_year: int,
    start_month: int,
    end_month: int,
    band: str = "NDVI",
) -> ee.ImageCollection:
    """Build annual median composites over [start_month, end_month] (supports wrap-around Dec-Mar)."""
    years = ee.List.sequence(start_year, end_year)

    def make_year(y):
        y = ee.Number(y).toInt()
        if start_month > end_month:
            # e.g., Dec of y-1 through Mar of y
            d1 = ee.Date.fromYMD(y.subtract(1), start_month, 1)
            d2 = ee.Date.fromYMD(y, end_month, 1).advance(1, "month")
        else:
            d1 = ee.Date.fromYMD(y, start_month, 1)
            d2 = ee.Date.fromYMD(y, end_month, 1).advance(1, "month")
        sub = col.filterDate(d1, d2).select(band)
        med = sub.median().toFloat().rename("val")
        t_img = ee.Image.constant(y.subtract(1985)).toFloat().rename("t")
        return t_img.addBands(med).set("year", y, "count", sub.size())

    return ee.ImageCollection(years.map(make_year)).filter(ee.Filter.gt("count", 0))


def sens_slope_decade(
    col: ee.ImageCollection,
    start_year: int,
    end_year: int,
    start_month: int,
    end_month: int,
    mask: ee.Image,
    band: str = "NDVI",
) -> ee.Image:
    """Compute annual-median Theil-Sen slope scaled to units per decade."""
    ann = annual_seasonal_composites(col, start_year, end_year, start_month, end_month, band=band)
    slope = ann.reduce(ee.Reducer.sensSlope()).select("slope").multiply(10.0).rename("trend")
    return slope.updateMask(mask)


def empirical_harmonized_collection(
    geom: ee.Geometry,
    season_key: str = "jan_feb",
    offsets: dict[str, float] | None = None,
) -> ee.ImageCollection:
    """Combine L5, L7, L8, L9 with empirical overlap offsets anchored to L7."""
    off = offsets or EMPIRICAL_OFFSETS_TO_L7[season_key]
    cols = []
    for s in ("L5", "L7", "L8", "L9"):
        delta = off[s]
        c = landsat_sensor_collection(geom, s).map(
            lambda im, d=delta: im.select("NDVI").add(d).toFloat().rename("NDVI").copyProperties(im, ["system:time_start", "sensor"])
        )
        cols.append(c)
    merged = cols[0].merge(cols[1]).merge(cols[2]).merge(cols[3])
    return merged


def mean_value(img: ee.Image, geom: ee.Geometry, band: str = "trend", scale: int = 30, tile_scale: int = 12) -> ee.Number:
    val = img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=scale,
        maxPixels=1e11,
        tileScale=tile_scale,
        bestEffort=True,
    ).get(band)
    return ee.Number(val)
