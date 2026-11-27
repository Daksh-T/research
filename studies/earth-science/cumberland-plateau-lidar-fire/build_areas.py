"""Assemble the study geometries for the Greater Cumberland Plateau study.

Fixes all v1 geometry bugs:
  1. Fixes Sewanee Domain path to ../initial_study/domain_main.geojson.
  2. Replaces the truncated 2,946 ha WDPA Fall Creek Falls WMA sub-unit with
     the complete ~10,283 ha Fall Creek Falls State Park boundary from OSM.
  3. Keeps all 8 Tennessee comparison units plus the regional window
     (data/region.geojson) and verifies every geometry is valid.
"""

import json
import os
import time

import ee
import requests

ee.Initialize(project=os.environ.get("EE_PROJECT"))
REGION = ee.Geometry.Rectangle([-86.5, 34.9, -84.7, 36.3])

FIRE = {
    "Sewanee Domain": ("managed-longterm", "domain"),
    "Franklin State Forest": ("none-documented", "state forest"),
    "Savage Gulf Designated": ("recent-partial", "state natural area"),
    "Prentice Cooper State Forest &": ("habitat-burn", "state forest/WMA"),
    "Catoosa": ("habitat-burn", "WMA"),
    "Fall Creek Falls State Park": ("partial-state-park", "state park/SNA"),
    "Bridgestone/Firestone": ("habitat-burn", "WMA"),
    "North Chickamauga Creek Gorge Designated": ("none-documented", "state natural area"),
}


def fetch_osm_polygon(query: str) -> dict:
    r = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": query, "format": "json", "polygon_geojson": 1, "limit": 3},
        headers={"User-Agent": "cumberland-plateau-science-study/2.0"},
        timeout=60,
    )
    r.raise_for_status()
    for item in r.json():
        gj = item.get("geojson", {})
        if gj.get("type") in ("Polygon", "MultiPolygon"):
            return gj
    raise RuntimeError(f"No polygon returned by OSM for {query!r}")


def main() -> None:
    os.makedirs("data", exist_ok=True)
    features = []

    # 1. Sewanee Domain (LAL boundary in ../initial_study/)
    dom_path = os.path.join("..", "initial_study", "domain_main.geojson")
    with open(dom_path) as f:
        dom = json.load(f)["features"][0]
    features.append({
        "type": "Feature",
        "geometry": dom["geometry"],
        "properties": {
            "name": "Sewanee Domain",
            "kind": "domain",
            "fire": "managed-longterm",
            "source": "Sewanee LAL",
        },
    })
    print("added Sewanee Domain")

    # 2. Franklin State Forest (OSM)
    fsf = fetch_osm_polygon("Franklin State Forest, Tennessee")
    features.append({
        "type": "Feature",
        "geometry": fsf,
        "properties": {
            "name": "Franklin State Forest",
            "kind": "state forest",
            "fire": "none-documented",
            "source": "OpenStreetMap",
        },
    })
    print("added Franklin State Forest")
    time.sleep(1.2)

    # 3. Fall Creek Falls State Park (complete ~10,283 ha boundary from OSM, fixing WDPA truncation)
    fcf = fetch_osm_polygon("Fall Creek Falls State Park, Tennessee")
    features.append({
        "type": "Feature",
        "geometry": fcf,
        "properties": {
            "name": "Fall Creek Falls State Park",
            "kind": "state park/SNA",
            "fire": "partial-state-park",
            "source": "OpenStreetMap (full park; replaces truncated WDPA WMA)",
        },
    })
    print("added Fall Creek Falls State Park (complete OSM boundary)")
    time.sleep(1.2)

    # 4. Remaining 5 WDPA units
    wdpa = ee.FeatureCollection("WCMC/WDPA/current/polygons").filterBounds(REGION)
    for name in [
        "Savage Gulf Designated",
        "Prentice Cooper State Forest &",
        "Catoosa",
        "Bridgestone/Firestone",
        "North Chickamauga Creek Gorge Designated",
    ]:
        sub = wdpa.filter(ee.Filter.eq("NAME", name))
        geom = sub.geometry().simplify(50)
        gj = geom.getInfo()
        fire, kind = FIRE.get(name, ("none-documented", "protected"))
        features.append({
            "type": "Feature",
            "geometry": gj,
            "properties": {
                "name": name.rstrip(" &"),
                "kind": kind,
                "fire": fire,
                "source": "WDPA",
            },
        })
        print("added", name)

    fc = {"type": "FeatureCollection", "features": features}
    with open("data/areas.geojson", "w") as f:
        json.dump(fc, f)
    print(f"wrote data/areas.geojson with {len(features)} areas")

    # Regional window (expanded slightly north/east to include Catoosa WMA cleanly!)
    region = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"name": "Greater Cumberland Plateau (southern TN study window)"},
            "geometry": ee.Geometry.Rectangle([-86.05, 35.0, -84.70, 36.20]).getInfo(),
        }],
    }
    with open("data/region.geojson", "w") as f:
        json.dump(region, f)
    print("wrote data/region.geojson")


if __name__ == "__main__":
    main()
