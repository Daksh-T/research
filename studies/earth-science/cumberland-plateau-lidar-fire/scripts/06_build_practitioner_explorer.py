"""Phase 3 (Path 3): Build Standalone Interactive Cumberland Plateau Fire & LiDAR Explorer
(explorer.html) for land managers and curious public readers.

Key features:
  - File://-friendly keyless basemaps (USGS Topo, USGS Shaded Relief, Esri World Topo,
    Esri Satellite Imagery - zero CARTO, zero OSM) plus a vivid, embedded Google Earth
    Engine SRTM + Annual NLCD forest-and-relief overlay in its own Leaflet pane
    (works both online and completely offline)
  - Split interactive map + synchronized Plot & Protected-Area Inspector with a
    4-layer vertical canopy density chart using unbiased first-return >10 m canopy cover
    and marking the 4-7 m vertical crossover zone
  - All 689 deciduous forest plots across Alabama, Tennessee, and Kentucky
  - Accessible, plain-English Prescription & Monitoring Advisor powered directly by
    our measured statistical models
  - Tabbed publication figures & sortable calibrated Tennessee protected-area table.

Outputs: explorer.html
"""

from __future__ import annotations

import base64
import json
import os


def b64_img(path: str, mime: str = "image/png") -> str:
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode("ascii")


def simplify_geojson_coords(geom: dict) -> dict:
    """Lightly round coordinates to 4 decimals (~11 m), supporting GeometryCollection."""

    def rnd(obj):
        if isinstance(obj, list):
            if len(obj) >= 2 and isinstance(obj[0], (int, float)):
                return [round(obj[0], 4), round(obj[1], 4)]
            return [rnd(x) for x in obj]
        return obj

    if geom["type"] == "GeometryCollection":
        return {
            "type": "GeometryCollection",
            "geometries": [simplify_geojson_coords(g) for g in geom["geometries"]],
        }
    return {"type": geom["type"], "coordinates": rnd(geom["coordinates"])}


def _load_json(path: str):
    with open(path) as f:
        return json.load(f)


def _geom_center(geom: dict) -> tuple[float, float]:
    """Compute approximate centroid (lat, lon) of a GeoJSON geometry for row fly-to."""
    lons: list[float] = []
    lats: list[float] = []

    def walk(obj):
        if isinstance(obj, list):
            if len(obj) >= 2 and isinstance(obj[0], (int, float)):
                lons.append(float(obj[0]))
                lats.append(float(obj[1]))
            else:
                for x in obj:
                    walk(x)

    if geom["type"] == "GeometryCollection":
        for g in geom["geometries"]:
            walk(g["coordinates"])
    else:
        walk(geom["coordinates"])
    if not lons:
        return (35.55, -85.45)
    return (round((min(lats) + max(lats)) / 2, 4), round((min(lons) + max(lons)) / 2, 4))


def main() -> None:
    areas_fc = _load_json("data/areas.geojson")
    area_metrics = _load_json("data/area_metrics.json")
    reg_metrics = _load_json("data/region_metrics.json")
    asum = _load_json("data/lidar/analysis_summary.json")
    raw_plots = _load_json("data/lidar/plot_metrics.json")

    gee_overlay_uri = ""
    gee_bounds = {"west": -88.20, "south": 33.80, "east": -82.80, "north": 38.45}
    if os.path.exists("assets/gee_overlay_bounds.json"):
        gee_bounds = _load_json("assets/gee_overlay_bounds.json")
    if os.path.exists("assets/gee_plateau_overlay.jpg"):
        gee_overlay_uri = b64_img("assets/gee_plateau_overlay.jpg", "image/jpeg")
    elif os.path.exists("assets/gee_plateau_overlay.png"):
        gee_overlay_uri = b64_img("assets/gee_plateau_overlay.png", "image/png")

    # Compact plot payload for Leaflet + Inspector
    plot_records = []
    for pid, r in raw_plots.items():
        lp = r.get("lidar_primary")
        if not lp:
            continue
        lb = r.get("lidar_baseline")
        if lb and r["site"] == "DanielBoone_East_BACI" and not (lb.get("acq_date") or "").startswith("2017"):
            lb = None
        opt = r.get("optical", {})
        cat = (
            "mech_plus_fire"
            if r.get("has_mech")
            else (
                "0_burns"
                if r["n_burns_total_pre_lidar"] == 0
                else "1_burn"
                if r["n_burns_total_pre_lidar"] == 1
                else "2_burns"
                if r["n_burns_total_pre_lidar"] == 2
                else "3plus_burns"
            )
        )
        plot_records.append({
            "id": pid,
            "site": r["site"],
            "stratum": r["stratum"],
            "cat": cat,
            "lat": round(r["lat"], 5),
            "lon": round(r["lon"], 5),
            "nb": r["n_burns_total_pre_lidar"],
            "yrs": r.get("burn_years_pre_lidar", []),
            "mech": bool(r.get("has_mech")),
            "elev": round(float(r["elev_m"]), 1),
            "slope": round(float(r["slope_deg"]), 1),
            "hli": round(float(r["hli"]), 3),
            "und": round(lp["nrd_understory"], 3),
            "low": round(lp["nrd_low_midstory"], 3),
            "tall": round(lp["nrd_tall_midstory"], 3),
            "mid": round(lp["nrd_midstory"], 3),
            "can": round(lp.get("first_return_canopy_10m", lp["canopy_cover_10m"]), 3),
            "chm": round(float(lp["chm_p95"]), 1),
            "dens": round(float(lp["pulse_density"]), 1),
            "acq": lp.get("acq_date"),
            "acq_b": lb.get("acq_date") if lb else None,
            "dmid": round(lp["nrd_midstory"] - lb["nrd_midstory"], 3) if lb else None,
            "dund": round(lp["nrd_understory"] - lb["nrd_understory"], 3) if lb else None,
            "win_ndvi": round(opt["opt_win_ndvi"], 3) if opt.get("opt_win_ndvi") is not None else None,
            "spr_ndvi": round(opt["opt_spr_ndvi"], 3) if opt.get("opt_spr_ndvi") is not None else None,
        })

    # Compact TN comparison polygons with centroids for table-to-map navigation
    tn_polys = {"type": "FeatureCollection", "features": []}
    area_centers: dict[str, tuple[float, float]] = {}
    for f in areas_fc["features"]:
        name = f["properties"]["name"]
        m = area_metrics.get(name, {})
        c_lat, c_lon = _geom_center(f["geometry"])
        area_centers[name] = (c_lat, c_lon)
        tn_polys["features"].append({
            "type": "Feature",
            "geometry": simplify_geojson_coords(f["geometry"]),
            "properties": {
                "name": name,
                "kind": f["properties"].get("kind", ""),
                "fire": f["properties"].get("fire", ""),
                "clat": c_lat,
                "clon": c_lon,
                "total_ha": m.get("total_ha"),
                "pct_plateau": m.get("pct_plateau"),
                "pct_decid": m.get("nlcd_pct_deciduous"),
                "pct_ev": m.get("nlcd_pct_evergreen"),
                "unharm": m.get("unharm_dec_mar_decade"),
                "harm": m.get("harm_jan_feb_decid_decade"),
                "l5": m.get("l5_jan_feb_decid_decade"),
                "l89": m.get("l89_jan_feb_decid_decade"),
                "mtbs": m.get("mtbs_fire_count"),
            },
        })

    figs = {
        "fig1": b64_img("paper/figures/fig1_optical_illusion_vs_lidar.png"),
        "fig2": b64_img("paper/figures/fig2_baci_lidar_structural_change.png"),
        "fig3": b64_img("paper/figures/fig3_vertical_stratum_decoupling.png"),
        "fig4": b64_img("paper/figures/fig4_sensor_comparison_and_tn_reversal.png"),
    }

    ols = asum["pooled_fe_ols"]
    bh_baci = asum["baci"]["Bankhead_North_BACI"]["strata"]
    db_baci = asum["baci"]["DanielBoone_East_BACI"]["strata"]
    tr = reg_metrics["trends"]
    pct_collapse = round(100.0 * (1.0 - tr["harm_jan_feb_decid"] / tr["unharm_dec_mar_all"]))
    baci_win_r = asum["sensor_comparison"]["baci_deltas"]["d_opt_win_ndvi"]["pearson_r_d_midstory"]
    baci_win_n = asum["sensor_comparison"]["baci_deltas"]["d_opt_win_ndvi"]["n"]
    baci_spr_r = asum["sensor_comparison"]["baci_deltas"]["d_opt_spr_ndvi"]["pearson_r_d_midstory"]
    baci_spr_n = asum["sensor_comparison"]["baci_deltas"]["d_opt_spr_ndvi"]["n"]
    baci_spr_p = asum["sensor_comparison"]["baci_deltas"]["d_opt_spr_ndvi"]["p_d_midstory"]
    sc = asum["sensor_comparison"]["cross_sectional"]
    spr_r_pool = sc["opt_spr_ndvi"]["pearson_r_midstory"]
    spr_r_within = sc["opt_spr_ndvi"]["within_site_r_midstory"]
    win_r_pool = sc["opt_win_ndvi"]["pearson_r_midstory"]
    win_r_within = sc["opt_win_ndvi"]["within_site_r_midstory"]
    gedi_r_hi = sc["gedi_pai_5_10m"]["pearson_r_midstory"]
    gedi_r_hi_w = sc["gedi_pai_5_10m"]["within_site_r_midstory"]
    gedi_r_lo_w = sc["gedi_pai_0_5m"]["within_site_r_midstory"]
    pooled_3p = asum["dose_response_by_site"]["POOLED_ALL_SITES"]["3plus_burns"]["vs_0_matched_midstory"]
    bhs_3p = asum["dose_response_by_site"]["Bankhead_South_2017"]["3plus_burns"]["vs_0_matched_midstory"]
    bh_2p_did = bh_baci["baci_2plus_inter"]["did_vs_control"]["did_midstory"]
    bh_2p_d = bh_baci["baci_2plus_inter"]["did_vs_control"]["cohens_d"]
    bh_2p_ps = bh_baci["baci_2plus_inter"]["did_vs_control"]["p_cluster_suid"]
    bh_1b_did = bh_baci["baci_1_inter"]["did_vs_control"]["did_midstory"]
    db_pre_did = db_baci["baci_pre_only"]["did_vs_control"]["did_midstory"]
    db_pre_pw = db_baci["baci_pre_only"]["did_vs_control"]["p_welch"]
    db_pre_ps = db_baci["baci_pre_only"]["did_vs_control"]["p_cluster_suid"]
    db_pre_wcb = db_baci["baci_pre_only"]["did_vs_control"].get("p_wcb_suid", db_pre_ps)
    db_pre_anc = db_baci["baci_pre_only"]["did_vs_control"]["did_midstory_ancova"]
    db_pre_anc_p = db_baci["baci_pre_only"]["did_vs_control"]["p_ancova_suid"]
    db_pre_base = db_baci["baci_pre_only"]["baseline_nrd_midstory"]["mean"]
    db_pre_end = db_baci["baci_pre_only"]["primary_nrd_midstory"]["mean"]
    db_ctrl_base = db_baci["baci_0_control"]["baseline_nrd_midstory"]["mean"]
    db_ctrl_end = db_baci["baci_0_control"]["primary_nrd_midstory"]["mean"]
    db_1b_did = db_baci["baci_1_inter"]["did_vs_control"]["did_midstory"]
    db_1b_ps = db_baci["baci_1_inter"]["did_vs_control"]["p_cluster_suid"]
    tall_ct_p = ols["p_nrd_tall_midstory"]["contrasts"]["mech_minus_3plus"]["p_cluster_suid"]
    frc_mech = ols["p_first_return_canopy_10m"]["coefs"]["mech_plus_fire"]["coef"]
    frc_mech_p = ols["p_first_return_canopy_10m"]["coefs"]["mech_plus_fire"]["p_cluster_suid"]

    baci_summary_js = {
        "bh_2p_did": bh_2p_did,
        "bh_2p_d": bh_2p_d,
        "bh_2p_ps": bh_2p_ps,
        "bh_1b_did": bh_1b_did,
        "db_pre_did": db_pre_did,
        "db_pre_pw": db_pre_pw,
        "db_pre_ps": db_pre_ps,
        "db_pre_wcb": db_pre_wcb,
        "db_pre_anc": db_pre_anc,
        "db_pre_anc_p": db_pre_anc_p,
        "db_pre_base": db_pre_base,
        "db_pre_end": db_pre_end,
        "db_ctrl_base": db_ctrl_base,
        "db_ctrl_end": db_ctrl_end,
        "db_1b_did": db_1b_did,
        "db_1b_ps": db_1b_ps,
        "baci_win_r": baci_win_r,
        "baci_win_n": baci_win_n,
        "baci_spr_r": baci_spr_r,
        "baci_spr_n": baci_spr_n,
        "baci_spr_p": baci_spr_p,
        "tall_ct_p": tall_ct_p,
    }
    sensor_corr_js = {
        "spr_r_pool": spr_r_pool,
        "spr_r_within": spr_r_within,
        "win_r_pool": win_r_pool,
        "win_r_within": win_r_within,
        "gedi_r_hi": gedi_r_hi,
        "gedi_r_hi_w": gedi_r_hi_w,
        "gedi_r_lo_w": gedi_r_lo_w,
    }

    tn_table_records = []
    for name, d in sorted(area_metrics.items(), key=lambda kv: kv[1]["harm_jan_feb_decid_decade"]):
        c_lat, c_lon = area_centers.get(name, (35.55, -85.45))
        tn_table_records.append({
            "name": name,
            "kind": d["kind"],
            "total_ha": d["total_ha"],
            "pct_plateau": d["pct_plateau"],
            "decid": d["nlcd_pct_deciduous"],
            "ev": d["nlcd_pct_evergreen"],
            "unharm": d["unharm_dec_mar_decade"],
            "harm": d["harm_jan_feb_decid_decade"],
            "l5": d["l5_jan_feb_decid_decade"],
            "l89": d["l89_jan_feb_decid_decade"],
            "mtbs": d["mtbs_fire_count"],
            "clat": c_lat,
            "clon": c_lon,
        })

    ex_bh3 = next(
        (p["id"] for p in plot_records if p["site"] == "Bankhead_North_BACI" and p["nb"] >= 2 and p["dmid"] is not None),
        plot_records[0]["id"],
    )
    ex_db_ab = next(
        (p["id"] for p in plot_records if p["site"] == "DanielBoone_East_BACI" and p["stratum"] == "baci_pre_only" and p["dmid"] is not None),
        plot_records[0]["id"],
    )
    ex_mech = next(
        (p["id"] for p in plot_records if p["mech"] and p["site"] == "DanielBoone_Main_2017"),
        plot_records[0]["id"],
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Interactive explorer showing how prescribed fire reshapes eastern oak forests across Alabama, Tennessee, and Kentucky using airborne laser scanning and calibrated satellite imagery.">
<title>Cumberland Plateau Fire &amp; Forest Structure Atlas</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Instrument+Sans:wght@500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  :root {{
    /* Tailored Cumberland Plateau topographic & forest mineral palette */
    --ground: #f3f6f4;
    --surface: #ffffff;
    --surface-subtle: #e9efec;
    --surface-wash: #eef5f1;
    --topbar-bg: #0d1f19;
    --topbar-line: #1f3930;

    --ink: #0f201b;
    --ink-muted: #43544e;
    --line: #d4ded9;
    --line-strong: #b4c3bc;

    /* Single deliberate accent */
    --accent: #165943;
    --accent-hover: #0f4332;
    --accent-tint: #e3f0ea;

    /* Semantic states & data-encoding colors only */
    --alert: #b5421d;
    --alert-tint: #fbf0eb;
    --data-open: #166a50;
    --data-blue: #2374a6;
    --data-deep-blue: #0e4a7b;
    --data-amber: #c97614;
    --data-plum: #85315b;
    --data-slate: #5b6762;

    /* Strictly 2 font families (tabular-nums replaces monospace) */
    --font-display: "Instrument Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    --font-body: "IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;

    /* Strictly 3 body font-size steps below H1/masthead, zero <13px text */
    --text-h2: 1.5rem;
    --text-body: 0.9375rem;
    --text-sm: 0.8125rem;

    --ease-out: cubic-bezier(0.22, 1, 0.36, 1);
    --radius: 4px;
    --container: 1400px;
  }}

  * {{ box-sizing: border-box; }}
  ::selection {{ background: var(--accent); color: var(--surface); }}
  html {{ scroll-behavior: smooth; overflow-x: clip; }}
  body {{
    margin: 0;
    font-family: var(--font-body);
    font-size: var(--text-body);
    line-height: 1.56;
    color: var(--ink);
    background: var(--ground);
    -webkit-font-smoothing: antialiased;
    overflow-x: clip;
  }}
  .num, .bar-val, .f-num, .adv-metric-val, .kv-val {{
    font-variant-numeric: tabular-nums;
  }}

  .skip-link {{
    position: absolute; left: -9999px; top: 8px;
    background: var(--accent); color: var(--surface);
    padding: 8px 14px; border-radius: var(--radius);
    font-weight: 600; font-size: var(--text-sm); text-decoration: none; z-index: 2000;
  }}
  .skip-link:focus {{ left: 16px; }}
  :focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}

  /* Single-line sticky top navigation with strong masthead wordmark */
  .topbar {{
    position: sticky; top: 0; z-index: 1000;
    background: var(--topbar-bg);
    border-bottom: 1px solid var(--topbar-line);
    box-shadow: 0 1px 0 rgba(13, 31, 25, 0.25);
  }}
  .topbar-inner {{
    max-width: var(--container);
    margin: 0 auto;
    padding: 12px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
  }}
  .brand {{
    display: inline-flex; align-items: center; gap: 10px;
    color: #f4f7f5; text-decoration: none;
    white-space: nowrap; flex-shrink: 0;
  }}
  .brand-wordmark {{
    font-family: var(--font-display);
    font-size: 1.1875rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    color: #f6f9f7;
    line-height: 1.1;
  }}
  .nav-links {{
    display: flex; align-items: center; gap: 6px;
    flex-wrap: nowrap;
  }}
  .nav-links a {{
    color: #c3d6cd;
    text-decoration: none;
    font-size: var(--text-sm);
    font-weight: 500;
    padding: 6px 11px;
    border-radius: var(--radius);
    white-space: nowrap;
    transition: background-color 160ms var(--ease-out), color 160ms var(--ease-out);
  }}
  .nav-links a:hover {{
    background: #1b362d;
    color: #ffffff;
  }}

  /* Shared band container */
  .band {{
    padding: 40px 24px;
    scroll-margin-top: 58px;
  }}
  .band-white {{
    background: var(--surface);
    border-bottom: 1px solid var(--line);
  }}
  .band-tinted {{
    background: var(--surface-subtle);
    border-bottom: 1px solid var(--line);
  }}
  .band-ground {{
    background: var(--ground);
    border-bottom: 1px solid var(--line);
  }}
  .band-workspace {{
    background: var(--surface);
    border-bottom: 1px solid var(--line);
    padding: 20px 24px 28px;
  }}
  .inner {{
    max-width: var(--container);
    margin: 0 auto;
  }}

  /* Opening workspace title (direct to live map, no hero banner) */
  .workspace-header {{
    display: flex;
    flex-wrap: wrap;
    align-items: flex-end;
    justify-content: space-between;
    gap: 10px 28px;
    margin-bottom: 16px;
  }}
  .workspace-header h1 {{
    margin: 0;
    font-family: var(--font-display);
    font-size: clamp(1.625rem, 2.2vw, 2.05rem);
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.12;
    color: var(--ink);
  }}
  .workspace-header p {{
    margin: 0;
    font-size: var(--text-body);
    color: var(--ink-muted);
    max-width: 60ch;
    line-height: 1.48;
  }}

  /* Section headings (direct H2, zero eyebrows, zero numbering) */
  .sec-intro {{
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 12px 40px;
    align-items: end;
    margin-bottom: 28px;
  }}
  .sec-intro h2 {{
    margin: 0;
    font-family: var(--font-display);
    font-size: var(--text-h2);
    font-weight: 700;
    letter-spacing: -0.025em;
    line-height: 1.18;
    color: var(--ink);
  }}
  .sec-intro p {{
    margin: 0;
    font-size: var(--text-body);
    color: var(--ink-muted);
    max-width: 62ch;
  }}

  /* Live Map & Inspector Workbench */
  .workbench {{
    border: 1px solid var(--line-strong);
    border-radius: var(--radius);
    background: var(--surface);
    overflow: hidden;
  }}
  .toolbar {{
    background: var(--surface-subtle);
    border-bottom: 1px solid var(--line);
    padding: 10px 14px;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 10px 16px;
  }}
  .tool-cluster {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px 16px;
  }}
  .tool-group {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    max-width: 100%;
  }}
  .tool-label {{
    font-size: var(--text-sm);
    font-weight: 600;
    color: var(--ink-muted);
    white-space: nowrap;
  }}

  /* Connected segmented control for map color mode */
  .seg-group {{
    display: inline-flex;
    flex-wrap: wrap;
    background: var(--surface);
    border: 1px solid var(--line-strong);
    border-radius: var(--radius);
    padding: 2px;
    gap: 2px;
  }}
  .seg-btn {{
    appearance: none;
    background: transparent;
    border: none;
    color: var(--ink);
    padding: 6px 11px;
    border-radius: 3px;
    cursor: pointer;
    font-family: var(--font-body);
    font-size: var(--text-sm);
    font-weight: 500;
    line-height: 1.2;
    min-height: 30px;
    transition: background-color 160ms var(--ease-out), color 160ms var(--ease-out);
  }}
  .seg-btn:hover {{ background: var(--surface-subtle); }}
  .seg-btn.active {{
    background: var(--accent);
    color: var(--surface);
    font-weight: 600;
  }}
  .seg-btn.active:hover {{ background: var(--accent-hover); }}

  .tool-select {{
    font-family: var(--font-body);
    font-size: var(--text-sm);
    font-weight: 500;
    color: var(--ink);
    background: var(--surface);
    border: 1px solid var(--line-strong);
    border-radius: var(--radius);
    padding: 5px 10px;
    min-height: 34px;
    cursor: pointer;
    transition: border-color 160ms var(--ease-out);
  }}
  .tool-select:hover {{ border-color: var(--accent); }}

  .overlay-ctrl {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--surface);
    border: 1px solid var(--line-strong);
    border-radius: var(--radius);
    padding: 4px 10px;
    min-height: 34px;
    font-size: var(--text-sm);
    font-weight: 500;
    color: var(--ink);
  }}
  .overlay-ctrl input[type="checkbox"] {{
    accent-color: var(--accent);
    width: 15px; height: 15px;
    cursor: pointer; margin: 0;
  }}
  .overlay-ctrl input[type="range"] {{
    width: 68px;
    accent-color: var(--accent);
    cursor: pointer;
  }}

  .wb-grid {{
    display: grid;
    grid-template-columns: minmax(0, 1fr) 390px;
    min-height: 580px;
  }}
  .map-pane {{
    position: relative;
    display: flex;
    flex-direction: column;
    min-width: 0;
  }}
  #map {{
    flex: 1;
    min-height: 540px;
    background: #dde5e0;
  }}
  .legend-bar {{
    background: var(--surface);
    border-top: 1px solid var(--line);
    padding: 10px 14px;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px 18px;
    font-size: var(--text-sm);
    color: var(--ink-muted);
  }}
  .leg-item {{ display: inline-flex; align-items: center; gap: 7px; }}
  .dot {{
    display: inline-block; width: 10px; height: 10px; border-radius: 50%;
    border: 1px solid rgba(15, 32, 27, 0.45);
    flex-shrink: 0;
  }}

  /* Inspector Side Rail (zero nested boxes, pure hairline dividers) */
  .inspector {{
    border-left: 1px solid var(--line);
    background: var(--surface);
    padding: 18px 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    overflow-y: auto;
  }}
  .insp-head {{
    padding-bottom: 12px;
    border-bottom: 1px solid var(--line);
  }}
  .insp-head-row {{
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    justify-content: space-between;
    gap: 4px 12px;
  }}
  .insp-head h3 {{
    margin: 0;
    font-family: var(--font-display);
    font-size: var(--text-body);
    font-weight: 700;
    color: var(--ink);
    overflow-wrap: anywhere;
  }}
  .insp-meta {{
    font-size: var(--text-sm);
    font-weight: 600;
    color: var(--accent);
    white-space: nowrap;
  }}
  .insp-meta.alert {{ color: var(--alert); }}
  .insp-meta.neutral {{ color: var(--ink-muted); }}
  .insp-sub {{
    margin: 4px 0 0;
    font-size: var(--text-sm);
    color: var(--ink-muted);
    line-height: 1.4;
  }}

  .insp-section {{
    padding-bottom: 14px;
    border-bottom: 1px solid var(--line);
  }}
  .insp-section:last-child {{
    padding-bottom: 0;
    border-bottom: none;
  }}
  .insp-section-title {{
    margin: 0 0 10px;
    font-size: var(--text-sm);
    font-weight: 600;
    color: var(--ink);
  }}

  /* Clean hairline key-value grid (no mini card borders) */
  .kv-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px 16px;
  }}
  .kv-item {{ min-width: 0; }}
  .kv-key {{
    font-size: var(--text-sm);
    color: var(--ink-muted);
    line-height: 1.3;
  }}
  .kv-val {{
    font-size: var(--text-body);
    font-weight: 600;
    color: var(--ink);
    margin-top: 2px;
    line-height: 1.3;
    word-break: break-word;
  }}
  .kv-val.alert {{ color: var(--alert); }}

  /* Vertical canopy bars directly on subtle wash without card cage */
  .bar-row {{
    display: grid;
    grid-template-columns: 142px 1fr 46px;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
    font-size: var(--text-sm);
    color: var(--ink);
  }}
  .bar-row:last-child {{ margin-bottom: 0; }}
  .bar-track {{
    height: 10px;
    background: var(--surface-subtle);
    border-radius: 2px;
    overflow: hidden;
  }}
  .bar-fill {{
    height: 100%;
    border-radius: 2px;
    transition: width 200ms var(--ease-out);
  }}
  .bar-val {{
    text-align: right;
    font-size: var(--text-sm);
    font-weight: 600;
    color: var(--ink);
  }}
  .threshold-rule {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 9px 0;
    font-size: var(--text-sm);
    font-weight: 600;
    color: var(--alert);
  }}
  .threshold-rule::before, .threshold-rule::after {{
    content: ""; flex: 1; border-top: 1px dashed var(--alert);
  }}

  /* Hairline-divided interactive list for telling examples */
  .pick-list {{
    display: flex;
    flex-direction: column;
    border-top: 1px solid var(--line);
  }}
  .pick-item {{
    appearance: none;
    background: transparent;
    border: none;
    border-bottom: 1px solid var(--line);
    padding: 10px 4px;
    text-align: left;
    cursor: pointer;
    font-family: var(--font-body);
    color: var(--ink);
    transition: background-color 150ms var(--ease-out), padding-left 150ms var(--ease-out);
  }}
  .pick-item:hover {{
    background: var(--surface-wash);
    padding-left: 8px;
  }}
  .pick-item strong {{
    display: block;
    font-size: var(--text-sm);
    font-weight: 600;
    color: var(--accent);
    line-height: 1.35;
  }}
  .pick-item span {{
    display: block;
    font-size: var(--text-sm);
    color: var(--ink-muted);
    margin-top: 2px;
    line-height: 1.4;
  }}

  .back-link {{
    appearance: none;
    background: transparent;
    border: none;
    padding: 4px 0;
    font-family: var(--font-body);
    font-size: var(--text-sm);
    font-weight: 600;
    color: var(--accent);
    cursor: pointer;
    text-align: left;
    transition: color 150ms var(--ease-out);
  }}
  .back-link:hover {{
    color: var(--accent-hover);
    text-decoration: underline;
    text-underline-offset: 3px;
  }}
  .insp-note {{
    margin: 0;
    font-size: var(--text-sm);
    color: var(--ink-muted);
    line-height: 1.5;
  }}

  /* Editorial Findings 2x2 Hairline Grid (no boxed cards) */
  .findings-grid {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    border-top: 1px solid var(--line-strong);
  }}
  .finding {{
    padding: 24px 28px 24px 0;
    border-bottom: 1px solid var(--line);
    border-right: 1px solid var(--line);
  }}
  .finding:nth-child(even) {{
    padding: 24px 0 24px 28px;
    border-right: none;
  }}
  .finding:nth-last-child(-n+2) {{
    border-bottom: none;
    padding-bottom: 6px;
  }}
  .f-num {{
    font-family: var(--font-display);
    font-size: var(--text-h2);
    font-weight: 700;
    letter-spacing: -0.025em;
    line-height: 1.15;
    color: var(--ink);
  }}
  .f-num.accent {{ color: var(--accent); }}
  .f-num.alert {{ color: var(--alert); }}
  .finding h3 {{
    margin: 8px 0 8px;
    font-family: var(--font-display);
    font-size: var(--text-body);
    font-weight: 700;
    color: var(--ink);
    line-height: 1.3;
  }}
  .finding p {{
    margin: 0;
    font-size: var(--text-body);
    color: var(--ink-muted);
    max-width: 60ch;
    line-height: 1.56;
  }}

  /* Asymmetric Split Burn & Monitoring Advisor (zero nested boxes) */
  .advisor-split {{
    display: grid;
    grid-template-columns: 375px minmax(0, 1fr);
    gap: 0;
    border-top: 1px solid var(--line-strong);
  }}
  .advisor-controls {{
    padding: 24px 32px 12px 0;
    border-right: 1px solid var(--line);
  }}
  .field {{ margin-bottom: 18px; }}
  .field:last-child {{ margin-bottom: 0; }}
  .field label {{
    display: block;
    font-size: var(--text-sm);
    font-weight: 600;
    color: var(--ink);
    margin-bottom: 6px;
  }}
  .field select {{
    width: 100%;
    padding: 9px 11px;
    border: 1px solid var(--line-strong);
    border-radius: var(--radius);
    background: var(--surface);
    color: var(--ink);
    font-family: var(--font-body);
    font-size: var(--text-body);
    min-height: 40px;
    cursor: pointer;
    transition: border-color 160ms var(--ease-out);
  }}
  .field select:hover {{ border-color: var(--accent); }}

  .advisor-readout {{
    padding: 24px 0 8px 36px;
  }}
  .adv-top {{
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    justify-content: space-between;
    gap: 8px 18px;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--line);
  }}
  .adv-top h3 {{
    margin: 0;
    font-family: var(--font-display);
    font-size: var(--text-body);
    font-weight: 700;
    color: var(--ink);
    line-height: 1.35;
  }}
  .adv-status {{
    font-size: var(--text-sm);
    font-weight: 600;
    color: var(--accent);
    white-space: nowrap;
  }}
  .adv-status.warn {{ color: var(--alert); }}

  .adv-metrics {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 16px;
    padding: 16px 0;
    border-bottom: 1px solid var(--line);
    margin-bottom: 18px;
  }}
  .adv-metric-val {{
    font-family: var(--font-display);
    font-size: var(--text-body);
    font-weight: 700;
    color: var(--ink);
    line-height: 1.3;
  }}
  .adv-metric-lbl {{
    font-size: var(--text-sm);
    color: var(--ink-muted);
    margin-top: 3px;
    line-height: 1.35;
  }}
  .adv-text {{
    margin: 0 0 16px;
    font-size: var(--text-body);
    color: var(--ink);
    max-width: 65ch;
    line-height: 1.58;
  }}
  .adv-sensor {{
    margin: 0;
    padding: 14px 16px;
    background: var(--surface-wash);
    border-left: 3px solid var(--accent);
    font-size: var(--text-body);
    color: var(--ink-muted);
    max-width: 65ch;
    line-height: 1.55;
  }}
  .adv-sensor.warn {{
    background: var(--alert-tint);
    border-left-color: var(--alert);
  }}
  .adv-sensor strong {{ color: var(--ink); }}

  /* Underlined Editorial Tabs for Study Figures */
  .fig-tabs {{
    display: flex;
    flex-wrap: wrap;
    gap: 4px 20px;
    border-bottom: 1px solid var(--line-strong);
    margin-bottom: 20px;
  }}
  .tab-btn {{
    appearance: none;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
    padding: 10px 2px;
    font-family: var(--font-body);
    font-size: var(--text-body);
    font-weight: 500;
    color: var(--ink-muted);
    cursor: pointer;
    transition: color 160ms var(--ease-out), border-color 160ms var(--ease-out);
  }}
  .tab-btn:hover {{ color: var(--ink); }}
  .tab-btn.active {{
    color: var(--accent);
    font-weight: 600;
    border-bottom-color: var(--accent);
  }}
  .fig-stage {{
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: 18px;
  }}
  .fig-stage img {{
    width: 100%;
    height: auto;
    display: block;
    background: #ffffff;
  }}
  .fig-cap {{
    margin-top: 16px;
    padding-top: 14px;
    border-top: 1px solid var(--line);
    font-size: var(--text-body);
    color: var(--ink-muted);
    line-height: 1.56;
  }}
  .fig-cap p {{ margin: 0; max-width: 65ch; }}
  .fig-cap strong {{ color: var(--ink); }}
  .fig-grid-all {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 20px;
  }}
  .fig-mini {{
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: 14px;
  }}
  .fig-mini img {{
    width: 100%;
    height: auto;
    display: block;
  }}
  .fig-mini p {{
    margin: 10px 0 0;
    font-size: var(--text-sm);
    color: var(--ink-muted);
    line-height: 1.5;
  }}
  .fig-mini p strong {{ color: var(--ink); }}

  /* Full-width Calibrated Tennessee Table */
  .tn-context {{
    margin: 0 0 18px;
    font-size: var(--text-body);
    color: var(--ink-muted);
    max-width: 65ch;
  }}
  .table-wrap {{
    overflow-x: auto;
    border-top: 1px solid var(--line-strong);
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: var(--text-body);
  }}
  th, td {{
    padding: 12px 12px;
    border-bottom: 1px solid var(--line);
    text-align: left;
    vertical-align: middle;
  }}
  th {{
    background: var(--surface-subtle);
    font-size: var(--text-sm);
    font-weight: 600;
    color: var(--ink);
    white-space: nowrap;
  }}
  th button.sort-btn {{
    appearance: none;
    background: none;
    border: none;
    padding: 0;
    font: inherit;
    color: inherit;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    transition: color 150ms var(--ease-out);
  }}
  th button.sort-btn:hover {{ color: var(--accent); }}
  td.num, th.num {{
    text-align: right;
    font-variant-numeric: tabular-nums;
  }}
  tbody tr {{
    cursor: pointer;
    transition: background-color 140ms var(--ease-out);
  }}
  tbody tr:hover {{ background: var(--surface-wash); }}
  .area-kind {{
    display: block;
    font-size: var(--text-sm);
    color: var(--ink-muted);
    margin-top: 2px;
  }}
  .bad {{ color: var(--alert); font-weight: 600; }}
  .good {{ color: var(--accent); font-weight: 600; }}
  .muted {{ color: var(--ink-muted); font-size: var(--text-sm); }}

  /* Quiet Architectural Footer */
  .site-footer {{
    background: var(--ground);
    padding: 24px;
  }}
  .footer-inner {{
    max-width: var(--container);
    margin: 0 auto;
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    gap: 8px 24px;
    font-size: var(--text-sm);
    color: var(--ink-muted);
  }}
  .footer-inner strong {{ color: var(--ink); font-weight: 600; }}

  /* Leaflet popups & container at crisp 13px minimum */
  .leaflet-container {{ font-family: var(--font-body); font-size: var(--text-sm); }}
  .leaflet-popup-content-wrapper {{
    border-radius: var(--radius);
    box-shadow: 0 8px 24px rgba(15, 32, 27, 0.16);
    border: 1px solid var(--line);
  }}
  .leaflet-popup-content {{
    font-family: var(--font-body);
    font-size: var(--text-sm);
    line-height: 1.5;
    margin: 10px 12px;
    color: var(--ink);
  }}

  @media (max-width: 1060px) {{
    .wb-grid {{ grid-template-columns: 1fr; }}
    .inspector {{ border-left: none; border-top: 1px solid var(--line); }}
    .sec-intro {{ grid-template-columns: 1fr; gap: 8px; }}
    .advisor-split {{ grid-template-columns: 1fr; }}
    .advisor-controls {{
      padding: 20px 0;
      border-right: none;
      border-bottom: 1px solid var(--line);
    }}
    .advisor-readout {{ padding: 20px 0 0; }}
    .fig-grid-all {{ grid-template-columns: 1fr; }}
  }}
  @media (max-width: 760px) {{
    .topbar-inner {{
      padding: 10px 16px;
      flex-wrap: wrap;
      gap: 6px 12px;
    }}
    .nav-links {{
      width: 100%;
      flex-wrap: wrap;
      gap: 2px 4px;
    }}
    .nav-links a {{
      padding: 4px 8px;
    }}
    .band {{ padding: 28px 16px; scroll-margin-top: 108px; }}
    .band-workspace {{ padding: 16px 16px 24px; scroll-margin-top: 108px; }}
    .findings-grid {{ grid-template-columns: 1fr; }}
    .finding, .finding:nth-child(even) {{
      padding: 20px 0;
      border-right: none;
      border-bottom: 1px solid var(--line);
    }}
    .finding:last-child {{ border-bottom: none; padding-bottom: 4px; }}
    .adv-metrics {{ grid-template-columns: 1fr; gap: 12px; }}
    .toolbar {{ padding: 10px 12px; }}
    .tool-group {{ width: 100%; flex-wrap: wrap; }}
    .seg-group {{
      width: 100%;
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }}
    .seg-btn {{ padding: 6px 6px; text-align: center; }}
    .tool-select {{ flex: 1 1 180px; max-width: 100%; }}
    .overlay-ctrl {{ width: 100%; justify-content: space-between; }}
    .bar-row {{ grid-template-columns: 126px 1fr 44px; }}
    #map {{ min-height: 380px; }}
    .site-footer {{ padding: 20px 16px; }}
  }}
  @media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{ transition: none !important; scroll-behavior: auto !important; }}
  }}
</style>
</head>
<body>
<a class="skip-link" href="#main-content">Skip to map and analysis</a>

<header class="topbar">
  <div class="topbar-inner">
    <a class="brand" href="#workbench-sec">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#8ad4b8" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M3 17.5c3-2.5 6-2.5 9 0s6 2.5 9 0"/>
        <path d="M12 3.5 6.8 11h3.2L7.8 15h8.4L14 11h3.2L12 3.5Z"/>
      </svg>
      <span class="brand-wordmark">Cumberland Plateau Atlas</span>
    </a>
    <nav class="nav-links" aria-label="Primary sections">
      <a href="#workbench-sec">Map &amp; Inspector</a>
      <a href="#findings-sec">Key Findings</a>
      <a href="#advisor-sec">Burn Advisor</a>
      <a href="#figures-sec">Figures</a>
      <a href="#tn-sec">Tennessee Parks</a>
    </nav>
  </div>
</header>

<main id="main-content">
  <section class="band-workspace" id="workbench-sec" aria-label="Interactive regional map and forest canopy inspector">
    <div class="inner">
      <div class="workspace-header">
        <h1>See how prescribed fire reshapes Appalachian oak forests</h1>
        <p>Click any of the 689 hardwood plots or Tennessee park boundaries below to inspect subcanopy crowding from airborne laser scans.</p>
      </div>

      <div class="workbench">
        <div class="toolbar">
          <div class="tool-cluster">
            <div class="tool-group">
              <span class="tool-label">Color plots by</span>
              <div class="seg-group" role="group" aria-label="Plot color encoding">
                <button class="seg-btn active" id="mode-cat" type="button" aria-pressed="true" onclick="setMode('cat')">Burn history</button>
                <button class="seg-btn" id="mode-mid" type="button" aria-pressed="false" onclick="setMode('mid')">Midstory crowding (2&ndash;10 m)</button>
                <button class="seg-btn" id="mode-baci" type="button" aria-pressed="false" onclick="setMode('baci')">Multi-year change (283)</button>
              </div>
            </div>
          </div>

          <div class="tool-cluster">
            <div class="tool-group">
              <label class="tool-label" for="sel-basemap">Map layer</label>
              <select id="sel-basemap" class="tool-select" onchange="setBasemap(this.value)">
                <option value="usgs_topo" selected>USGS Topographic</option>
                <option value="usgs_relief">USGS Shaded Relief</option>
                <option value="esri_topo">Esri World Topographic</option>
                <option value="esri_imagery">Esri Satellite Imagery</option>
                <option value="gee_only">GEE Forest &amp; Relief Only (Offline)</option>
              </select>
            </div>

            <label class="overlay-ctrl" title="Overlay Google Earth Engine forest cover (Annual NLCD 2021) blended with shaded terrain relief">
              <input type="checkbox" id="chk-gee" checked onchange="toggleGeeOverlay()">
              <span>GEE forest relief</span>
              <input type="range" id="rng-gee" min="15" max="100" value="68" aria-label="GEE forest relief opacity" oninput="setGeeOpacity(this.value)">
            </label>

            <div class="tool-group">
              <label class="tool-label" for="sel-region">Region</label>
              <select id="sel-region" class="tool-select" onchange="jumpRegion(this.value)">
                <option value="all" selected>Whole plateau</option>
                <option value="bankhead">Bankhead NF (AL)</option>
                <option value="tennessee">Tennessee Plateau</option>
                <option value="daniel_boone">Daniel Boone NF (KY)</option>
              </select>
            </div>
          </div>
        </div>

        <div class="wb-grid">
          <div class="map-pane">
            <div id="map" role="application" aria-label="Interactive map of Cumberland Plateau forest plots and Tennessee protected areas"></div>
            <div class="legend-bar" id="map-legend"></div>
          </div>
          <aside class="inspector" id="inspector-panel" aria-live="polite"></aside>
        </div>
      </div>
    </div>
  </section>

  <section class="band band-tinted" id="findings-sec">
    <div class="inner">
      <div class="sec-intro">
        <h2>What repeat airborne laser scans reveal</h2>
        <p>Four headline takeaways comparing leaf-off USGS 3DEP LiDAR against 40 years of Landsat satellite imagery across Alabama, Tennessee, and Kentucky.</p>
      </div>

      <div class="findings-grid">
        <article class="finding">
          <div class="f-num alert">&minus;{pct_collapse}%</div>
          <h3>Standard winter satellite trends miss understory change</h3>
          <p>Removing the calibration jump between older and newer Landsat satellites cuts the 40-year winter trend from {tr['unharm_dec_mar_all']:+.3f} to {tr['harm_jan_feb_decid']:+.3f} per decade. Against repeat laser scans, neither winter (r = {baci_win_r:+.2f}) nor spring (r = {baci_spr_r:+.2f}) NDVI tracks midstory change.</p>
        </article>

        <article class="finding">
          <div class="f-num accent">{pooled_3p['pct_diff']:+.1f}%</div>
          <h3>Repeated burning clears nearly a third of midstory crowding</h3>
          <p>Across topographically matched hardwood stands, plots burned three or more times hold nearly a third less 2&ndash;10 m woody crowding than unburned controls (up to {bhs_3p['pct_diff']:+.1f}% in southern Bankhead NF; pooled site-fixed-effects slope &minus;0.064, p &lt; 0.001).</p>
        </article>

        <article class="finding">
          <div class="f-num accent">{bh_2p_did:+.3f} NRD</div>
          <h3>Repeat laser flights confirm fire works in Bankhead</h3>
          <p>In northern Bankhead NF (2012 to 2021), two or more burns lowered 2&ndash;10 m midstory change by {bh_2p_did:+.3f} relative to unburned controls (d = {bh_2p_d:+.2f}, burn-unit p = {bh_2p_ps:.4f}), unchanged after density-thinning sensitivity tests.</p>
        </article>

        <article class="finding">
          <div class="f-num">4&ndash;7 m</div>
          <h3>Surface fire works best below the midstory crossover zone</h3>
          <p>In 1 m cross-sectional height profiles, three or more burns outperform mechanical thinning plus fire at 1&ndash;4 m (p &le; 0.01) while leaving first-return canopy above 10 m intact; mechanical thinning plus fire is stronger at 8&ndash;10 m (p &le; 0.024) and opens the upper canopy ({frc_mech:+.3f}, p = {frc_mech_p:.3f}).</p>
        </article>
      </div>
    </div>
  </section>

  <section class="band band-white" id="advisor-sec">
    <div class="inner">
      <div class="sec-intro">
        <h2>Predict how a burn plan and sensor will perform</h2>
        <p>Pick how tall the encroaching understory trees have grown, the treatment you are weighing, and your monitoring sensor to read the measured evidence directly.</p>
      </div>

      <div class="advisor-split">
        <div class="advisor-controls">
          <div class="field">
            <label for="sel-stratum">Height of encroaching understory trees</label>
            <select id="sel-stratum" onchange="updateAdvisor()">
              <option value="under">Sprouts &amp; shrubs (0.5&ndash;2 m / 2&ndash;7 ft)</option>
              <option value="low" selected>Low midstory saplings (2&ndash;5 m / 7&ndash;16 ft)</option>
              <option value="tall">Tall escaped poles (5&ndash;10 m / 16&ndash;33 ft)</option>
            </select>
          </div>
          <div class="field">
            <label for="sel-rx">Planned management approach</label>
            <select id="sel-rx" onchange="updateAdvisor()">
              <option value="burn_1">Single winter prescribed burn (1 burn)</option>
              <option value="burn_2">Two winter prescribed burns (2 burns)</option>
              <option value="burn_3plus" selected>Repeated prescribed fire (3+ burns)</option>
              <option value="mech_plus_fire">Mechanical thinning + prescribed fire</option>
              <option value="abandoned">Burned once, then unburned 5+ years</option>
            </select>
          </div>
          <div class="field">
            <label for="sel-sensor">How you plan to track change</label>
            <select id="sel-sensor" onchange="updateAdvisor()">
              <option value="lidar" selected>Repeat airborne LiDAR (USGS 3DEP)</option>
              <option value="spring">Spring satellite NDVI (Mar&ndash;Apr)</option>
              <option value="winter">Winter satellite NDVI (Jan&ndash;Feb)</option>
              <option value="gedi">Space-station lidar (NASA GEDI L2B)</option>
            </select>
          </div>
        </div>
        <div class="advisor-readout" id="advisor-output" aria-live="polite"></div>
      </div>
    </div>
  </section>

  <section class="band band-ground" id="figures-sec">
    <div class="inner">
      <div class="sec-intro">
        <h2>Inspect the full evidence charts from the study</h2>
        <p>Examine each publication figure at full width or view all four side by side to see the exact vertical profiles, repeat-flight contrasts, and sensor comparisons.</p>
      </div>

      <div class="fig-tabs" role="tablist" aria-label="Study figures">
        <button class="tab-btn active" id="ftab-fig1" type="button" role="tab" aria-selected="true" onclick="showFig('fig1')">Optical audit &amp; deltas</button>
        <button class="tab-btn" id="ftab-fig2" type="button" role="tab" aria-selected="false" onclick="showFig('fig2')">Repeat laser scans (AL &amp; KY)</button>
        <button class="tab-btn" id="ftab-fig3" type="button" role="tab" aria-selected="false" onclick="showFig('fig3')">1 m vertical profile</button>
        <button class="tab-btn" id="ftab-fig4" type="button" role="tab" aria-selected="false" onclick="showFig('fig4')">Sensor skill &amp; TN parks</button>
        <button class="tab-btn" id="ftab-all" type="button" role="tab" aria-selected="false" onclick="showFig('all')">Compare all four</button>
      </div>
      <div id="fig-viewer"></div>
    </div>
  </section>

  <section class="band band-white" id="tn-sec">
    <div class="inner">
      <div class="sec-intro">
        <h2>Compare calibrated trends across Tennessee parks and forests</h2>
        <p>Evaluated strictly on the upland plateau surface (430 m elevation and above) using the repaired 10,282-hectare Fall Creek Falls boundary. Click any row to fly the map there.</p>
      </div>

      <p class="tn-context">Across the southern Tennessee plateau top as a whole, forest cover stepped down from 5,617 km&sup2; in 1985 to 5,135 km&sup2; in 2001 and has remained steady since (5,185 km&sup2; in 2024, +0.97% net since 2001 despite {reg_metrics['loss_km2']:,.0f} km&sup2; of rotational timber harvest that grew back).</p>

      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th><button class="sort-btn" type="button" onclick="sortTn('name')">Park or forest</button></th>
              <th class="num"><button class="sort-btn" type="button" onclick="sortTn('total_ha')">Total size (% on plateau)</button></th>
              <th class="num"><button class="sort-btn" type="button" onclick="sortTn('decid')">Hardwood / Pine</button></th>
              <th class="num"><button class="sort-btn" type="button" onclick="sortTn('unharm')">Uncorrected winter trend</button></th>
              <th class="num"><button class="sort-btn" type="button" onclick="sortTn('harm')">Corrected hardwood trend</button></th>
              <th class="num"><button class="sort-btn" type="button" onclick="sortTn('l5')">Landsat 5 (1985&ndash;2011)</button></th>
              <th class="num"><button class="sort-btn" type="button" onclick="sortTn('l89')">Landsat 8+9 (2014&ndash;25)</button></th>
              <th class="num"><button class="sort-btn" type="button" onclick="sortTn('mtbs')">Large fires</button></th>
            </tr>
          </thead>
          <tbody id="tn-tbody"></tbody>
        </table>
      </div>
    </div>
  </section>
</main>

<footer class="site-footer">
  <div class="footer-inner">
    <span><strong>Cumberland Plateau Forest Structure &amp; Prescribed Fire Study</strong> &bull; Daksh Tyagi &amp; Gemini 4 Neon</span>
    <span>Open data: USGS 3DEP Airborne LiDAR, USFS FACTS, Sewanee LAL, Annual NLCD, Landsat Collection 2, NASA GEDI, USGS National Map, and Esri.</span>
  </div>
</footer>

<script>
const PLOTS = {json.dumps(plot_records)};
const TN_POLYS = {json.dumps(tn_polys)};
const TN_TABLE = {json.dumps(tn_table_records)};
const OLS = {json.dumps(ols)};
const BACI = {json.dumps(baci_summary_js)};
const SENSOR_CORR = {json.dumps(sensor_corr_js)};
const GEE_OVERLAY_URI = {json.dumps(gee_overlay_uri)};
const GEE_BOUNDS = [[{gee_bounds['south']}, {gee_bounds['west']}], [{gee_bounds['north']}, {gee_bounds['east']}]];
const EXAMPLE_IDS = {{ bh3: {json.dumps(ex_bh3)}, dbAb: {json.dumps(ex_db_ab)}, mech: {json.dumps(ex_mech)} }};
const FIG_DATA = {{
  fig1: {{
    src: {json.dumps(figs['fig1'])},
    alt: 'Optical radiometric audit and BACI delta comparison against 3DEP LiDAR',
    cap: '<strong>Passive Landsat NDVI misses temporal deciduous midstory change.</strong> Removing the calibration jump between older and newer Landsat sensors cuts the 40-year winter trend by {pct_collapse}% (a). Across repeat-scanned BACI plots, neither winter NDVI (r = {baci_win_r:+.2f}, n = {baci_win_n}) nor spring NDVI (r = {baci_spr_r:+.2f}, n = {baci_spr_n}) tracks measured 3D midstory change (b). Cross-sectionally across 689 plots, airborne LiDAR and spring NDVI respond to burn regime while winter NDVI remains flat (c).'
  }},
  fig2: {{
    src: {json.dumps(figs['fig2'])},
    alt: 'Repeat airborne laser scans in Alabama and Kentucky',
    cap: '<strong>Identified burn-minus-control contrasts in repeat airborne LiDAR.</strong> Because newer flights also record more multiple returns per pulse, only the Difference-in-Differences (DiD) relative to unburned controls is causally identified. In Bankhead NF (2012&ndash;2021), 1 and 2+ inter-epoch burns lowered midstory change by {bh_1b_did:+.3f} and {bh_2p_did:+.3f} NRD (burn-unit p &lt; 0.001), invariant to density thinning (white diamonds). In Daniel Boone NF (2017&ndash;2023), pre-2017 burned stands (3 burn units) recovered from a lower starting baseline without exceeding controls (p<sub>u</sub> = {db_pre_ps:.3f}), and all mechanical thinning occurred prior to 2017.'
  }},
  fig3: {{
    src: {json.dumps(figs['fig3'])},
    alt: 'Continuous 1 m vertical profile and unbiased first-return canopy cover',
    cap: '<strong>Gradual vertical attenuation from 1 to 10 m and unbiased first-return canopy cover.</strong> In continuous 1 m height bins (a), 3+ surface burns reduce stem density more strongly than mechanical+fire from 1 to 4 m (burn-unit p &le; 0.01), both overlap across a 4&ndash;7 m crossover zone, and cross-sectional mechanical+fire is stronger at 8&ndash;10 m (p &le; 0.024). Using unbiased first-return cover above 10 m (b), surface fire leaves the upper canopy statistically unchanged while mechanical+fire opens it ({frc_mech:+.3f}, p<sub>u</sub> = {frc_mech_p:.3f}).'
  }},
  fig4: {{
    src: {json.dumps(figs['fig4'])},
    alt: 'Pooled versus within-site sensor correlations and calibrated Tennessee trends',
    cap: '<strong>Within-site optical correlations converge, temporal deltas fail, and calibration alters Tennessee trends.</strong> Spring NDVI&rsquo;s pooled cross-sectional advantage ({spr_r_pool:+.2f} vs. {win_r_pool:+.2f}) largely disappears within sites ({spr_r_within:+.2f} vs. {win_r_within:+.2f} for winter NDVI and {gedi_r_hi_w:+.2f} for GEDI 5&ndash;10 m), and neither winter nor spring tracks temporal BACI change (a). Calibrating Landsat on pure hardwood shifts seven of eight Tennessee parks slightly negative during 2014&ndash;2025 (b).'
  }}
}};

const map = L.map('map', {{ zoomControl: true }}).setView([36.05, -85.45], 7);

// Dedicated pane above base tiles (200) and below vector polygons & plot markers (400)
if (typeof map.createPane === 'function') {{
  const geePane = map.createPane('geePane');
  if (geePane && geePane.style) {{
    geePane.style.zIndex = 260;
    geePane.style.pointerEvents = 'none';
  }}
}}

// Keyless, file://-friendly basemaps (zero CARTO, zero OpenStreetMap referrer blocks)
const BASEMAPS = {{
  usgs_topo: L.tileLayer('https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
    attribution: 'USGS The National Map', maxZoom: 16
  }}),
  usgs_relief: L.tileLayer('https://basemap.nationalmap.gov/arcgis/rest/services/USGSShadedReliefOnly/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
    attribution: 'USGS The National Map', maxZoom: 16
  }}),
  esri_topo: L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
    attribution: 'Tiles &copy; Esri', maxZoom: 18
  }}),
  esri_imagery: L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
    attribution: 'Tiles &copy; Esri, Maxar, Earthstar Geographics', maxZoom: 18
  }}),
  gee_only: null
}};

let activeBaseKey = 'usgs_topo';
if (BASEMAPS[activeBaseKey]) BASEMAPS[activeBaseKey].addTo(map);

// Embedded Google Earth Engine SRTM Hillshade + Annual NLCD Forest Cover overlay
let geeOverlay = null;
if (GEE_OVERLAY_URI && typeof L.imageOverlay === 'function') {{
  geeOverlay = L.imageOverlay(GEE_OVERLAY_URI, GEE_BOUNDS, {{
    pane: 'geePane',
    opacity: 0.68,
    interactive: false,
    attribution: 'Google Earth Engine (SRTM Relief + Annual NLCD Forest)'
  }});
  geeOverlay.addTo(map);
}}

function setBasemap(key) {{
  if (BASEMAPS[activeBaseKey] && typeof map.removeLayer === 'function') {{
    map.removeLayer(BASEMAPS[activeBaseKey]);
  }}
  activeBaseKey = key;
  if (BASEMAPS[key]) BASEMAPS[key].addTo(map);
  if (key === 'gee_only' && geeOverlay) {{
    const chk = document.getElementById('chk-gee');
    const rng = document.getElementById('rng-gee');
    if (chk) chk.checked = true;
    if (rng) rng.value = '100';
    if (typeof map.hasLayer === 'function' && !map.hasLayer(geeOverlay)) geeOverlay.addTo(map);
    if (typeof geeOverlay.setOpacity === 'function') geeOverlay.setOpacity(1.0);
  }}
}}

function toggleGeeOverlay() {{
  if (!geeOverlay) return;
  const chk = document.getElementById('chk-gee');
  if (chk && chk.checked) {{
    geeOverlay.addTo(map);
  }} else if (typeof map.removeLayer === 'function') {{
    map.removeLayer(geeOverlay);
  }}
}}

function setGeeOpacity(val) {{
  if (geeOverlay && typeof geeOverlay.setOpacity === 'function') {{
    geeOverlay.setOpacity(Number(val) / 100);
  }}
}}

function flyTo(lat, lon, z) {{
  map.flyTo([lat, lon], z, {{ duration: 0.7 }});
}}

function jumpRegion(key) {{
  const views = {{
    all: [36.05, -85.45, 7],
    bankhead: [34.30, -87.34, 11],
    tennessee: [35.55, -85.45, 9],
    daniel_boone: [37.20, -84.15, 9],
  }};
  const v = views[key] || views.all;
  flyTo(v[0], v[1], v[2]);
}}

// TN protected area polygons
const tnLayersByName = {{}};
L.geoJSON(TN_POLYS, {{
  style: () => ({{ color: '#0f201b', weight: 1.6, fillColor: '#165943', fillOpacity: 0.16 }}),
  onEachFeature: (feat, layer) => {{
    const p = feat.properties;
    tnLayersByName[p.name] = layer;
    const fmtT = (v) => (v !== null && v !== undefined) ? (v >= 0 ? '+' : '') + Number(v).toFixed(4) : 'N/A';
    layer.bindPopup(
      `<strong>${{p.name}}</strong> (${{p.kind}})<br>` +
      `Size: ${{p.total_ha ? p.total_ha.toLocaleString() : 'N/A'}} ha (${{p.pct_plateau}}% on upland plateau)<br>` +
      `Hardwood cover: ${{p.pct_decid}}%<br>` +
      `Uncorrected winter trend: <b>${{fmtT(p.unharm)}}</b>/decade<br>` +
      `Corrected hardwood trend: <b>${{fmtT(p.harm)}}</b>/decade<br>` +
      `Recent decade (Landsat 8+9): <b>${{fmtT(p.l89)}}</b>/decade`
    );
    if (typeof layer.on === 'function') {{
      layer.on('click', () => inspectArea(p));
    }}
  }}
}}).addTo(map);

const plotLayer = L.layerGroup().addTo(map);
const CAT_COLORS = {{
  '0_burns': '#5b6762',
  '1_burn': '#c97614',
  '2_burns': '#2374a6',
  '3plus_burns': '#0e4a7b',
  'mech_plus_fire': '#85315b'
}};
const CAT_LABELS = {{
  '0_burns': 'Unburned control',
  '1_burn': '1 prescribed burn',
  '2_burns': '2 prescribed burns',
  '3plus_burns': '3+ prescribed burns',
  'mech_plus_fire': 'Mechanical thinning + fire'
}};

function midColor(v) {{
  if (v < 0.14) return '#166a50';
  if (v < 0.22) return '#2374a6';
  if (v < 0.30) return '#c97614';
  return '#b5421d';
}}
function baciColor(v) {{
  if (v === null) return '#cbd5e1';
  if (v < 0.0) return '#166a50';
  if (v < 0.04) return '#2374a6';
  if (v < 0.09) return '#c97614';
  return '#b5421d';
}}
const fmtOpt = (v) => (v !== null && v !== undefined) ? Number(v).toFixed(3) : 'N/A';

function renderOverviewInspector() {{
  const el = document.getElementById('inspector-panel');
  if (!el) return;
  const nShown = currentMode === 'baci' ? PLOTS.filter(p => p.dmid !== null).length : PLOTS.length;
  el.innerHTML = `
    <div class="insp-head">
      <div class="insp-head-row">
        <h3>Forest plot inspector</h3>
        <span class="insp-meta">${{nShown}} plots shown</span>
      </div>
      <p class="insp-sub">Click any colored dot or Tennessee boundary on the map to inspect its canopy profile.</p>
    </div>

    <div class="insp-section">
      <div class="kv-grid">
        <div class="kv-item"><div class="kv-key">National forests</div><div class="kv-val">Bankhead &amp; D. Boone</div></div>
        <div class="kv-item"><div class="kv-key">Tennessee sites</div><div class="kv-val">8 parks &amp; forests</div></div>
        <div class="kv-item"><div class="kv-key">Repeat laser pairs</div><div class="kv-val">283 matched plots</div></div>
        <div class="kv-item"><div class="kv-key">Forest filter</div><div class="kv-val">Intact hardwood oak</div></div>
      </div>
    </div>

    <div class="insp-section">
      <div class="insp-section-title">Explore three telling examples</div>
      <div class="pick-list">
        <button class="pick-item" type="button" onclick="selectPlotById(EXAMPLE_IDS.bh3)">
          <strong>Bankhead NF: repeatedly burned stand</strong>
          <span>Scanned in 2012 and 2021; recurring fire kept the understory open.</span>
        </button>
        <button class="pick-item" type="button" onclick="selectPlotById(EXAMPLE_IDS.dbAb)">
          <strong>Daniel Boone NF: pre-2017 burn only</strong>
          <span>Burned before 2017 with no follow-up; recovered toward (not past) unburned levels.</span>
        </button>
        <button class="pick-item" type="button" onclick="selectPlotById(EXAMPLE_IDS.mech)">
          <strong>Daniel Boone NF: mechanical thinning + fire</strong>
          <span>Cross-sectional stand with lower 5&ndash;10 m poles and a more open &gt;10 m canopy.</span>
        </button>
      </div>
    </div>

    <p class="insp-note">Airborne laser pulses pass through bare winter oak crowns to measure woody stem crowding across four height zones, from knee-high sprouts up to mature treetops.</p>
  `;
}}

function inspectPlot(p) {{
  const el = document.getElementById('inspector-panel');
  if (!el) return;
  const bar = (lbl, val, color, maxV) => {{
    const pct = Math.max(2, Math.min(100, Math.round((val / maxV) * 100)));
    return `<div class="bar-row"><span>${{lbl}}</span><div class="bar-track"><div class="bar-fill" style="width:${{pct}}%;background:${{color}}"></div></div><span class="bar-val">${{val.toFixed(3)}}</span></div>`;
  }};
  const epochTag = p.acq_b && p.acq ? `${{p.acq_b.slice(0, 4)}}&rarr;${{p.acq.slice(0, 4)}}` : 'repeat';
  const baciBlock = p.dmid !== null
    ? `<div class="insp-section">
         <div class="kv-grid">
           <div class="kv-item"><div class="kv-key">Raw midstory &Delta; (${{epochTag}})</div><div class="kv-val">${{p.dmid >= 0 ? '+' : ''}}${{p.dmid.toFixed(3)}}</div></div>
           <div class="kv-item"><div class="kv-key">Raw sprout &Delta; (vs. controls)</div><div class="kv-val">${{p.dund !== null ? ((p.dund >= 0 ? '+' : '') + p.dund.toFixed(3)) : 'N/A'}}</div></div>
         </div>
       </div>`
    : '';
  const statusClass = p.cat === '0_burns' ? 'neutral' : p.cat === 'mech_plus_fire' ? 'alert' : '';
  const displayId = p.id.replace(/_/g, '_<wbr>');
  el.innerHTML = `
    <div class="insp-head">
      <div class="insp-head-row">
        <h3>${{displayId}}</h3>
        <span class="insp-meta ${{statusClass}}">${{CAT_LABELS[p.cat]}}</span>
      </div>
      <p class="insp-sub">${{p.site.replace(/_/g, ' ')}} &bull; Laser scan ${{p.acq || 'N/A'}} (${{p.dens}} pts/m&sup2;)</p>
    </div>

    <div class="insp-section">
      <div class="insp-section-title">Vertical woody crowding (airborne LiDAR)</div>
      ${{bar('1st-ret. canopy (>10 m)', p.can, '#165943', 1.0)}}
      ${{bar('Tall poles (5&ndash;10 m)', p.tall, '#85315b', 0.55)}}
      <div class="threshold-rule">4&ndash;7 m vertical crossover zone</div>
      ${{bar('Saplings (2&ndash;5 m)', p.low, '#0e4a7b', 0.55)}}
      ${{bar('Sprouts (0.5&ndash;2 m)', p.und, '#c97614', 0.40)}}
    </div>

    ${{baciBlock}}

    <div class="insp-section">
      <div class="kv-grid">
        <div class="kv-item"><div class="kv-key">Burn history</div><div class="kv-val">${{p.nb}} burns${{p.mech ? ' + Thinning' : ''}}</div></div>
        <div class="kv-item"><div class="kv-key">Years burned</div><div class="kv-val">${{p.yrs.length ? p.yrs.join(', ') : 'None recorded'}}</div></div>
        <div class="kv-item"><div class="kv-key">Elevation / Slope</div><div class="kv-val">${{p.elev}} m / ${{p.slope}}&deg;</div></div>
        <div class="kv-item"><div class="kv-key">Canopy height (p95)</div><div class="kv-val">${{p.chm}} m</div></div>
        <div class="kv-item"><div class="kv-key">Winter satellite NDVI</div><div class="kv-val">${{fmtOpt(p.win_ndvi)}}</div></div>
        <div class="kv-item"><div class="kv-key">Spring satellite NDVI</div><div class="kv-val">${{fmtOpt(p.spr_ndvi)}}</div></div>
      </div>
    </div>

    <button class="back-link" type="button" onclick="renderOverviewInspector()">&larr; Back to regional overview</button>
  `;
}}

function inspectArea(p) {{
  const el = document.getElementById('inspector-panel');
  if (!el) return;
  const fmtT = (v) => (v !== null && v !== undefined) ? (v >= 0 ? '+' : '') + Number(v).toFixed(4) : 'N/A';
  el.innerHTML = `
    <div class="insp-head">
      <div class="insp-head-row">
        <h3>${{p.name}}</h3>
        <span class="insp-meta">Tennessee</span>
      </div>
      <p class="insp-sub">Protected ${{p.kind ? p.kind.toLowerCase() : 'area'}} on the southern plateau</p>
    </div>

    <div class="insp-section">
      <div class="kv-grid">
        <div class="kv-item"><div class="kv-key">Total size</div><div class="kv-val">${{p.total_ha ? p.total_ha.toLocaleString() + ' ha' : 'N/A'}}</div></div>
        <div class="kv-item"><div class="kv-key">On plateau (&ge;430 m)</div><div class="kv-val">${{p.pct_plateau}}%</div></div>
        <div class="kv-item"><div class="kv-key">Hardwood / Evergreen</div><div class="kv-val">${{p.pct_decid}}% / ${{p.pct_ev}}%</div></div>
        <div class="kv-item"><div class="kv-key">Recorded large fires</div><div class="kv-val">${{p.mtbs}}</div></div>
      </div>
    </div>

    <div class="insp-section">
      <div class="insp-section-title">Winter satellite trend check (NDVI per decade)</div>
      <div class="kv-grid">
        <div class="kv-item"><div class="kv-key">Uncorrected (Dec&ndash;Mar)</div><div class="kv-val alert">${{fmtT(p.unharm)}}</div></div>
        <div class="kv-item"><div class="kv-key">Corrected hardwood</div><div class="kv-val">${{fmtT(p.harm)}}</div></div>
        <div class="kv-item"><div class="kv-key">Landsat 5 (1985&ndash;2011)</div><div class="kv-val">${{fmtT(p.l5)}}</div></div>
        <div class="kv-item"><div class="kv-key">Landsat 8+9 (2014&ndash;25)</div><div class="kv-val">${{fmtT(p.l89)}}</div></div>
      </div>
    </div>

    <button class="back-link" type="button" onclick="renderOverviewInspector()">&larr; Back to regional overview</button>
  `;
}}

function selectPlotById(pid) {{
  const p = PLOTS.find(x => x.id === pid);
  if (!p) return;
  map.flyTo([p.lat, p.lon], 13, {{ duration: 0.7 }});
  inspectPlot(p);
}}

let currentMode = 'cat';
function renderPlots() {{
  plotLayer.clearLayers();
  PLOTS.forEach(p => {{
    if (currentMode === 'baci' && p.dmid === null) return;
    const col = currentMode === 'cat' ? CAT_COLORS[p.cat]
              : currentMode === 'mid' ? midColor(p.mid)
              : baciColor(p.dmid);
    const m = L.circleMarker([p.lat, p.lon], {{
      radius: currentMode === 'baci' ? 5.5 : 4.5,
      color: '#0f201b', weight: 0.6, fillColor: col, fillOpacity: 0.90
    }});
    const baciLine = p.dmid !== null
      ? `<br><b>Multi-year midstory change:</b> ${{p.dmid >= 0 ? '+' : ''}}${{p.dmid}} | Sprouts: ${{p.dund !== null ? ((p.dund >= 0 ? '+' : '') + p.dund) : 'N/A'}}`
      : '';
    m.bindPopup(
      `<strong>${{p.id}}</strong><br>` +
      `Forest: ${{p.site.replace(/_/g, ' ')}} (scan ${{p.acq || 'N/A'}})<br>` +
      `Fire history: <b>${{p.nb}} burns</b> ${{p.mech ? '+ Mechanical thinning' : '(Fire only)'}} ${{p.yrs.length ? '[' + p.yrs.join(', ') + ']' : ''}}<br>` +
      `Terrain: ${{p.elev}} m elevation | ${{p.slope}}&deg; slope<br>` +
      `<b>Woody crowding (LiDAR NRD):</b> 0.5&ndash;2 m = ${{p.und}} | 2&ndash;5 m = ${{p.low}} | 5&ndash;10 m = ${{p.tall}} | <b>2&ndash;10 m = ${{p.mid}}</b><br>` +
      `First-return canopy (&gt;10 m): ${{p.can}} | Treetop height: ${{p.chm}} m` + baciLine +
      `<br>Landsat NDVI: Winter = ${{fmtOpt(p.win_ndvi)}} | Spring = ${{fmtOpt(p.spr_ndvi)}}`
    );
    if (typeof m.on === 'function') {{
      m.on('click', () => inspectPlot(p));
    }}
    m.addTo(plotLayer);
  }});
  const leg = document.getElementById('map-legend');
  if (currentMode === 'cat') {{
    leg.innerHTML =
      `<span class="leg-item"><i class="dot" style="background:#5b6762"></i>Unburned control</span>` +
      `<span class="leg-item"><i class="dot" style="background:#c97614"></i>1 burn</span>` +
      `<span class="leg-item"><i class="dot" style="background:#2374a6"></i>2 burns</span>` +
      `<span class="leg-item"><i class="dot" style="background:#0e4a7b"></i>3+ burns</span>` +
      `<span class="leg-item"><i class="dot" style="background:#85315b"></i>Mechanical thinning + fire</span>`;
  }} else if (currentMode === 'mid') {{
    leg.innerHTML =
      `<span class="leg-item"><i class="dot" style="background:#166a50"></i>Open understory (&lt;0.14)</span>` +
      `<span class="leg-item"><i class="dot" style="background:#2374a6"></i>Moderate (0.14&ndash;0.22)</span>` +
      `<span class="leg-item"><i class="dot" style="background:#c97614"></i>Crowded (0.22&ndash;0.30)</span>` +
      `<span class="leg-item"><i class="dot" style="background:#b5421d"></i>Heavily crowded (&ge;0.30)</span>`;
  }} else {{
    leg.innerHTML =
      `<span class="leg-item"><i class="dot" style="background:#166a50"></i>Raw &Delta; &lt; 0</span>` +
      `<span class="leg-item"><i class="dot" style="background:#2374a6"></i>Raw &Delta; 0 to +0.04</span>` +
      `<span class="leg-item"><i class="dot" style="background:#c97614"></i>Raw &Delta; +0.04 to +0.09</span>` +
      `<span class="leg-item"><i class="dot" style="background:#b5421d"></i>Raw &Delta; &ge; +0.09</span>` +
      `<span class="leg-item">(raw &Delta; includes flight sensor shift; interpret against nearby controls)</span>`;
  }}
}}

function setMode(m) {{
  currentMode = m;
  ['cat', 'mid', 'baci'].forEach(k => {{
    const btn = document.getElementById('mode-' + k);
    if (btn) {{
      btn.classList.toggle('active', k === m);
      if (typeof btn.setAttribute === 'function') btn.setAttribute('aria-pressed', String(k === m));
    }}
  }});
  renderPlots();
  renderOverviewInspector();
}}
renderPlots();
renderOverviewInspector();

function updateAdvisor() {{
  const st = document.getElementById('sel-stratum').value;
  const rx = document.getElementById('sel-rx').value;
  const sn = document.getElementById('sel-sensor').value;

  const stratumMap = {{
    under: {{ key: 'p_nrd_understory', label: '0.5&ndash;2 m sprouts' }},
    low: {{ key: 'p_nrd_low_midstory', label: '2&ndash;5 m saplings' }},
    tall: {{ key: 'p_nrd_tall_midstory', label: '5&ndash;10 m tall poles' }},
  }}[st];

  const fmtP = (p) => p < 0.001 ? '&lt; 0.001' : '= ' + p.toFixed(3);
  let headline = '', effectStr = '', effectLabel = `Effect on ${{stratumMap.label}}`, canopyStr = '', guidance = '', statusText = 'Observed pattern', isWarn = false;
  if (rx === 'abandoned') {{
    headline = 'Pre-2017 burns recovered toward unburned levels, without overshooting them';
    effectStr = `DiD ${{BACI.db_pre_did >= 0 ? '+' : ''}}${{BACI.db_pre_did.toFixed(3)}} (unit p = ${{BACI.db_pre_ps.toFixed(3)}})`;
    effectLabel = '2&ndash;10 m midstory BACI DiD';
    canopyStr = 'No thinning';
    statusText = 'Recovery from lower baseline';
    isWarn = true;
    guidance = `In eastern Daniel Boone NF (2017&ndash;2023), 20 plots burned before 2017 (clustered in just 3 burn units) started with lower midstory density (${{BACI.db_pre_base.toFixed(3)}} vs. ${{BACI.db_ctrl_base.toFixed(3)}}) and recovered toward, but stayed below, unburned controls by 2023 (${{BACI.db_pre_end.toFixed(3)}} vs. ${{BACI.db_ctrl_end.toFixed(3)}}). Under burn-unit clustering (unit p = ${{BACI.db_pre_ps.toFixed(3)}}, wild cluster bootstrap p = ${{BACI.db_pre_wcb.toFixed(3)}}) and baseline-adjusted ANCOVA (${{BACI.db_pre_anc.toFixed(3)}}, unit p = ${{BACI.db_pre_anc_p.toFixed(2)}}), differential growth is indistinguishable from zero. These data indicate recovery from a suppressed baseline rather than an overshoot past never-burned stands.`;
  }} else {{
    const c = OLS[stratumMap.key].coefs[rx];
    const can = OLS['p_first_return_canopy_10m'].coefs[rx];
    const tallMech = OLS['p_nrd_tall_midstory'].coefs['mech_plus_fire'];
    const und3 = OLS['p_nrd_understory'].coefs['burn_3plus'];
    const low3 = OLS['p_nrd_low_midstory'].coefs['burn_3plus'];
    effectStr = `${{c.coef >= 0 ? '+' : ''}}${{c.coef.toFixed(3)}} (unit p ${{fmtP(c.p_cluster_suid)}})`;
    canopyStr = `${{can.coef >= 0 ? '+' : ''}}${{can.coef.toFixed(3)}} (unit p ${{fmtP(can.p_cluster_suid)}})`;
    if (st === 'tall' && rx !== 'mech_plus_fire') {{
      headline = 'Surface-fire reductions fade above 5&ndash;6 m in cross-sectional profiles';
      statusText = 'Attenuates with height';
      isWarn = true;
      guidance = `In pooled cross-sectional regressions, surface-fire reductions in the ${{stratumMap.label}} layer are smaller and non-monotone across burn counts (${{effectStr}}), becoming statistically flat above 6 m in 1 m height bins. Mechanical thinning paired with fire shows lower cross-sectional 5&ndash;10 m density (${{tallMech.coef.toFixed(3)}}, unit p ${{fmtP(tallMech.p_cluster_suid)}}; direct mech vs. 3+ contrast p = ${{BACI.tall_ct_p.toFixed(2)}} for 5&ndash;10 m and p &le; 0.024 at 8&ndash;10 m), though no inter-epoch BACI test of mechanical thinning is available in our flights.`;
    }} else if (rx === 'mech_plus_fire') {{
      headline = 'Lower 5&ndash;10 m pole density cross-sectionally, with a more open &gt;10 m canopy';
      statusText = 'Cross-sectional evidence';
      guidance = `Stands with mechanical thinning plus fire show lower 5&ndash;10 m density (${{tallMech.coef.toFixed(3)}}, unit p ${{fmtP(tallMech.p_cluster_suid)}}), which is mechanically expected when poles are cut, alongside a significant opening of unbiased first-return upper canopy (${{can.coef.toFixed(3)}}, unit p ${{fmtP(can.p_cluster_suid)}}). In Daniel Boone BACI, all mechanical thinning predated the 2017 baseline, so mechanical+fire efficacy is supported cross-sectionally rather than by before-and-after LiDAR.`;
    }} else if (rx === 'burn_3plus') {{
      headline = 'Strongest surface-fire reductions below 5 m while upper canopy stays intact';
      statusText = 'Well supported below 5 m';
      guidance = `Three or more burns are associated with large reductions in 0.5&ndash;2 m sprouts (${{und3.coef.toFixed(3)}}) and 2&ndash;5 m saplings (${{low3.coef.toFixed(3)}}, unit p &lt; 0.001) without significantly altering first-return canopy cover above 10 m (${{canopyStr}}). In Bankhead NF repeat LiDAR (2012&ndash;2021), 2+ inter-epoch burns produced an identified DiD reduction of ${{BACI.bh_2p_did.toFixed(3)}} NRD relative to unburned controls (d = ${{BACI.bh_2p_d.toFixed(2)}}, unit p = ${{BACI.bh_2p_ps.toFixed(4)}}).`;
    }} else {{
      headline = 'Moderate lower-stratum reduction; repeat BACI response varies by forest';
      statusText = 'Mixed across sites';
      guidance = `One to two burns show moderate cross-sectional reductions in ${{stratumMap.label}} (${{effectStr}}). In repeat airborne LiDAR, a single inter-epoch burn produced a clear reduction in Bankhead NF (DiD = ${{BACI.bh_1b_did.toFixed(3)}}, unit p &lt; 0.001) but no detectable 6-year reduction in steeper Daniel Boone NF (DiD = ${{BACI.db_1b_did >= 0 ? '+' : ''}}${{BACI.db_1b_did.toFixed(3)}}, unit p = ${{BACI.db_1b_ps.toFixed(2)}}).`;
    }}
  }}

  const sensorNote = {{
    lidar: '<strong>Monitoring note (Primary benchmark):</strong> Repeat leaf-off USGS 3DEP airborne LiDAR directly resolves subcanopy woody structure. Because flight hardware and multi-return sensitivity shift between epochs, evaluate fire impacts via burned-minus-unburned DiD rather than raw control trends.',
    spring: `<strong>Monitoring note (Cross-sectional only, not temporal change):</strong> March&ndash;April spring NDVI responds to burn regime in cross-sectional site-FE models, but its within-site correlation with LiDAR midstory (r = +${{SENSOR_CORR.spr_r_within.toFixed(2)}}) is nearly identical to winter NDVI (+${{SENSOR_CORR.win_r_within.toFixed(2)}}), and temporal &Delta;Spring NDVI fails to track BACI structural change (r = ${{BACI.baci_spr_r.toFixed(2)}}, p = ${{BACI.baci_spr_p.toFixed(2)}}, n = ${{BACI.baci_spr_n}}). Do not rely on it for annual change tracking.`,
    winter: `<strong>Monitoring note (Do not use for deciduous change):</strong> Winter Jan&ndash;Feb Landsat NDVI shows no correlation with multi-year 3D midstory change (BACI r = ${{BACI.baci_win_r.toFixed(2)}}, n = ${{BACI.baci_win_n}}), detects zero fire effect in pooled models, and contains a large Collection 2 inter-sensor step jump.`,
    gedi: `<strong>Monitoring note (Coarse regional context only):</strong> Within sites, GEDI L2B 5&ndash;10 m PAI correlates comparably to optical snapshots (within-site r = +${{SENSOR_CORR.gedi_r_hi_w.toFixed(2)}}, pooled r = +${{SENSOR_CORR.gedi_r_hi.toFixed(2)}}), yet sparse orbital tracks (~60 m spacing across mixed seasons) limit stand-level burn monitoring.`,
  }}[sn];
  const sensorWarn = sn === 'winter' || sn === 'spring';

  document.getElementById('advisor-output').innerHTML = `
    <div class="adv-top">
      <h3>${{headline}}</h3>
      <span class="adv-status ${{isWarn ? 'warn' : ''}}">${{statusText}}</span>
    </div>
    <div class="adv-metrics">
      <div><div class="adv-metric-val">${{effectStr}}</div><div class="adv-metric-lbl">${{effectLabel}}</div></div>
      <div><div class="adv-metric-val">${{canopyStr}}</div><div class="adv-metric-lbl">First-return canopy (&gt;10 m)</div></div>
      <div><div class="adv-metric-val">${{sn === 'lidar' ? 'Direct 3D DiD' : sn === 'spring' ? 'BACI r = ' + BACI.baci_spr_r.toFixed(2) : sn === 'gedi' ? 'Within r = +' + SENSOR_CORR.gedi_r_hi_w.toFixed(2) : 'BACI r = ' + BACI.baci_win_r.toFixed(2)}}</div><div class="adv-metric-lbl">Measured sensor skill</div></div>
    </div>
    <p class="adv-text">${{guidance}}</p>
    <p class="adv-sensor ${{sensorWarn ? 'warn' : ''}}">${{sensorNote}}</p>
  `;
}}
updateAdvisor();

// Tabbed publication figure viewer
function showFig(key) {{
  ['fig1', 'fig2', 'fig3', 'fig4', 'all'].forEach(k => {{
    const b = document.getElementById('ftab-' + k);
    if (b) {{
      b.classList.toggle('active', k === key);
      if (typeof b.setAttribute === 'function') b.setAttribute('aria-selected', String(k === key));
    }}
  }});
  const box = document.getElementById('fig-viewer');
  if (!box) return;
  if (key === 'all') {{
    box.innerHTML = `<div class="fig-grid-all">` +
      ['fig1', 'fig2', 'fig3', 'fig4'].map(k =>
        `<div class="fig-mini"><img src="${{FIG_DATA[k].src}}" alt="${{FIG_DATA[k].alt}}"><p>${{FIG_DATA[k].cap}}</p></div>`
      ).join('') + `</div>`;
  }} else {{
    const f = FIG_DATA[key];
    box.innerHTML = `<div class="fig-stage"><img src="${{f.src}}" alt="${{f.alt}}"><div class="fig-cap"><p>${{f.cap}}</p></div></div>`;
  }}
}}
showFig('fig1');

// Sortable TN protected area table
let tnSortKey = 'harm', tnSortAsc = true;
function renderTnTable() {{
  const tb = document.getElementById('tn-tbody');
  if (!tb) return;
  const rows = [...TN_TABLE].sort((a, b) => {{
    const av = a[tnSortKey], bv = b[tnSortKey];
    if (typeof av === 'string') return tnSortAsc ? av.localeCompare(bv) : bv.localeCompare(av);
    return tnSortAsc ? av - bv : bv - av;
  }});
  const fmtS = (v) => (v >= 0 ? '+' : '') + Number(v).toFixed(4);
  tb.innerHTML = rows.map(r => `
    <tr onclick="focusTnArea(${{JSON.stringify(r.name)}})">
      <td><strong>${{r.name}}</strong><span class="area-kind">${{r.kind}}</span></td>
      <td class="num">${{Math.round(r.total_ha).toLocaleString()}} ha <span class="muted">(${{Math.round(r.pct_plateau)}}%)</span></td>
      <td class="num">${{r.decid.toFixed(1)}}% / ${{r.ev.toFixed(1)}}%</td>
      <td class="num bad">${{fmtS(r.unharm)}}</td>
      <td class="num">${{fmtS(r.harm)}}</td>
      <td class="num">${{fmtS(r.l5)}}</td>
      <td class="num ${{r.l89 < 0 ? 'good' : ''}}">${{fmtS(r.l89)}}</td>
      <td class="num">${{r.mtbs}}</td>
    </tr>
  `).join('');
}}
function sortTn(key) {{
  if (tnSortKey === key) tnSortAsc = !tnSortAsc;
  else {{ tnSortKey = key; tnSortAsc = key === 'name' || key === 'harm'; }}
  renderTnTable();
}}
function focusTnArea(name) {{
  const row = TN_TABLE.find(r => r.name === name);
  const feat = TN_POLYS.features.find(f => f.properties.name === name);
  if (row) map.flyTo([row.clat, row.clon], 11, {{ duration: 0.7 }});
  if (feat) inspectArea(feat.properties);
  const sec = document.getElementById('workbench-sec');
  if (sec && typeof sec.scrollIntoView === 'function') sec.scrollIntoView({{ behavior: 'smooth' }});
}}
renderTnTable();
</script>
</body>
</html>
"""
    with open("explorer.html", "w") as f:
        f.write(html)
    print(f"Wrote explorer.html ({os.path.getsize('explorer.html') / 1e6:.2f} MB) with {len(plot_records)} interactive plots")


if __name__ == "__main__":
    main()
