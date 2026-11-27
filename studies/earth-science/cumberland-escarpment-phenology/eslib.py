import os
"""Shared helpers for the escarpment study (AlphaEarth + Hopkins)."""
import ee, json, os, sys, urllib.request

PROJECT = os.environ.get('EE_PROJECT', '')
HERE = os.path.dirname(os.path.abspath(__file__))
A = lambda p: os.path.join(HERE, p)
sys.path.insert(0, A('../initial_study'))  # reuse parent analysis_lib
BANDS = ['A%02d' % i for i in range(64)]

def init():
    ee.Initialize(project=PROJECT)

def domain():
    g = json.load(open(A('../initial_study/domain_main.geojson')))
    return ee.Geometry(g['features'][0]['geometry'])

def embeddings():
    return ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')

def year_img(y, dom):
    return embeddings().filterDate('%d-01-01' % y, '%d-01-01' % (y + 1)).filterBounds(dom).mosaic().clip(dom)

def canonical_clusterer(emb, dom, k=7, seed=7, n=6000):
    """Deterministic k-means on embeddings (fixed seed/sample => stable cluster ids)."""
    train = emb.sample(region=dom, scale=10, numPixels=n, seed=seed, tileScale=16)
    return ee.Clusterer.wekaKMeans(k).train(train)

def save_thumb(img, vis, fname, region, dims=900):
    url = img.getThumbURL({**vis, 'region': region, 'dimensions': dims, 'format': 'png'})
    urllib.request.urlretrieve(url, A('assets/' + fname))
    print('  saved assets/' + fname)
    return A('assets/' + fname)

PALETTE7 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']


def build_sos(dom, years=('2018-01-01', '2025-01-01'), decid_only=True):
    """Per-pixel spring start-of-season (green-up DOY) by half-amplitude crossing of a
    biweekly Sentinel-2 NDVI climatology. Returns the SOS image (deciduous-masked)."""
    def _mask_ndvi(img):
        scl = img.select('SCL')
        good = scl.neq(1).And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
        return img.normalizedDifference(['B8', 'B4']).updateMask(good).copyProperties(img, ['system:time_start'])
    NDVI = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(dom)
            .filterDate(years[0], years[1])
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60)).map(_mask_ndvi))
    win = lambda d0, d1: NDVI.filter(ee.Filter.calendarRange(d0, d1, 'day_of_year')).median()
    comps, centers = [], []
    for d0 in range(40, 175, 14):
        comps.append(win(d0, d0 + 13)); centers.append(d0 + 6.5)
    winter = win(335, 365).max(win(1, 45)); summer = win(170, 240)
    amp = summer.subtract(winter)
    T = winter.add(amp.multiply(0.5))
    found = ee.Image(0); sos = ee.Image(0).rename('sos')
    for t in range(len(comps) - 1):
        c0, c1 = comps[t], comps[t + 1]; d0, d1 = centers[t], centers[t + 1]
        cross = c0.lt(T).And(c1.gte(T)).And(found.Not())
        interp = T.subtract(c0).divide(c1.subtract(c0).max(1e-4)).multiply(d1 - d0).add(d0)
        sos = sos.where(cross, interp); found = found.Or(cross)
    sos = sos.updateMask(found).updateMask(amp.gt(0.2)).rename('sos')
    if decid_only:
        nlcd = ee.ImageCollection('USGS/NLCD_RELEASES/2021_REL/NLCD').filter(
            ee.Filter.eq('system:index', '2021')).first().select('landcover')
        sos = sos.updateMask(nlcd.eq(41))
    return sos.clip(dom)
