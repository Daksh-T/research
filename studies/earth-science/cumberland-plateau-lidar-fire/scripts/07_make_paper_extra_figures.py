"""Generate two additional publication-grade figures following Remote Sensing of
Environment / Forest Ecology and Management best practices:

  1. paper/figures/fig0_study_area_map.{png,pdf}
     Multi-scale Study Region & Sampling Design Map (KY-TN-AL Cumberland Plateau)
     with GEE SRTM elevation + hillshade backdrop, 689 pure-deciduous 3DEP LiDAR
     plots, 8 repaired TN protected areas, scale bar, North arrow, CRS note, and
     Southeastern U.S. locator inset.

  2. paper/figures/fig_mechanism_schematic.{png,pdf}
     Accessible two-panel conceptual schematic explaining (a) why leaf-off
     airborne LiDAR resolves bare deciduous woody stems that winter passive
     optical satellites miss, and (b) gradual vertical attenuation across the
     4-7 m crossover zone with unbiased first-return >10 m canopy cover.
"""

from __future__ import annotations

import json
import math
import os

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import FancyBboxPatch, Polygon

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

DARK = "#18221f"
MUTED = "#4e5b56"
PINE = "#1d4d3e"
RUST = "#b94a22"
BLUE = "#0f4c81"
SKY = "#2e86b8"
AMBER = "#d98218"
MULBERRY = "#8c3460"
SLATE = "#5c6662"


def _load_json(path: str):
    with open(path) as f:
        return json.load(f)


def _extract_rings(geom: dict) -> list[list[list[float]]]:
    t = geom["type"]
    if t == "Polygon":
        return [geom["coordinates"][0]]
    if t == "MultiPolygon":
        return [poly[0] for poly in geom["coordinates"]]
    if t == "GeometryCollection":
        rings: list[list[list[float]]] = []
        for g in geom["geometries"]:
            rings.extend(_extract_rings(g))
        return rings
    return []


def make_study_area_map() -> None:
    bounds = _load_json("assets/gee_overlay_bounds.json")
    bg = plt.imread("assets/gee_study_elev_hillshade.png")
    areas = _load_json("data/areas.geojson")
    region = _load_json("data/region.geojson")
    plots = _load_json("data/lidar/plot_metrics.json")

    W, S, E, N = bounds["west"], bounds["south"], bounds["east"], bounds["north"]

    fig = plt.figure(figsize=(7.6, 5.5), dpi=300)
    ax = fig.add_axes([0.08, 0.10, 0.72, 0.82])

    # Background GEE SRTM elevation + hillshade
    ax.imshow(bg, extent=[W, E, S, N], aspect="auto", zorder=1)
    # Correct lon/lat aspect ratio at mid-latitude ~36.1 deg
    ax.set_aspect(1.0 / math.cos(math.radians(36.1)))

    # Subtle approximate state border reference lines (KY/TN ~36.50, TN/AL-GA ~34.99)
    ax.plot([W, E], [36.50, 36.50], color="#4a5752", lw=0.75, ls="--", alpha=0.65, zorder=2)
    ax.plot([W, E], [34.99, 34.99], color="#4a5752", lw=0.75, ls="--", alpha=0.65, zorder=2)
    ax.text(W + 0.14, 36.60, "KENTUCKY", fontsize=7.2, color="#3a4742", fontweight="bold", alpha=0.85, zorder=6)
    ax.text(W + 0.14, 36.28, "TENNESSEE", fontsize=7.2, color="#3a4742", fontweight="bold", alpha=0.85, zorder=6)
    ax.text(W + 0.14, 34.77, "ALABAMA", fontsize=7.2, color="#3a4742", fontweight="bold", alpha=0.85, zorder=6)

    # Regional Tennessee Plateau diagnostic window outline
    for ring in _extract_rings(region["features"][0]["geometry"]):
        xs, ys = zip(*ring, strict=False)
        ax.plot(xs, ys, color=PINE, lw=1.1, ls=":", zorder=3, label="TN regional audit window")

    # 8 Tennessee protected-area polygons
    poly_patches = []
    for feat in areas["features"]:
        for ring in _extract_rings(feat["geometry"]):
            poly_patches.append(Polygon(ring, closed=True))
    pc = PatchCollection(poly_patches, facecolor="#1d4d3e", edgecolor=DARK, alpha=0.28, linewidths=0.7, zorder=4)
    ax.add_collection(pc)

    # Plot all 689 3DEP LiDAR plots by burn regime
    cat_style = [
        ("0_burns", SLATE, "0 burns (Control)", 13),
        ("1_burn", AMBER, "1 burn (Fire only)", 15),
        ("2_burns", SKY, "2 burns (Fire only)", 15),
        ("3plus_burns", BLUE, "3+ burns (Fire only)", 17),
        ("mech_plus_fire", MULBERRY, "Mechanical + Fire", 18),
    ]
    pts_by_cat: dict[str, tuple[list[float], list[float]]] = {k: ([], []) for k, _, _, _ in cat_style}
    for r in plots.values():
        if not r.get("lidar_primary"):
            continue
        cat = (
            "mech_plus_fire"
            if r.get("has_mech")
            else "0_burns"
            if r["n_burns_total_pre_lidar"] == 0
            else "1_burn"
            if r["n_burns_total_pre_lidar"] == 1
            else "2_burns"
            if r["n_burns_total_pre_lidar"] == 2
            else "3plus_burns"
        )
        pts_by_cat[cat][0].append(r["lon"])
        pts_by_cat[cat][1].append(r["lat"])

    for k, col, lbl, sz in cat_style:
        xs, ys = pts_by_cat[k]
        ax.scatter(xs, ys, s=sz, c=col, edgecolors=DARK, linewidths=0.35, alpha=0.90, zorder=5, label=f"{lbl} (n={len(xs)})")

    # Crisp non-overlapping site annotations
    bbox_kw = {"boxstyle": "round,pad=0.28", "facecolor": "#ffffff", "edgecolor": "#b8c4bd", "alpha": 0.92, "lw": 0.7}
    ax.annotate(
        "Daniel Boone NF (KY)\n335 plots (144 BACI, 2017–23)",
        xy=(-83.82, 37.35), xytext=(-86.85, 37.82),
        fontsize=7.6, fontweight="bold", color=DARK, bbox=bbox_kw,
        arrowprops={"arrowstyle": "->", "color": DARK, "lw": 0.8, "connectionstyle": "arc3,rad=-0.08"},
        zorder=8,
    )
    ax.annotate(
        "8 TN Protected Areas\n(incl. 10,282 ha Fall Creek Falls)\n& Sewanee / Franklin SF (79 plots)",
        xy=(-85.55, 35.58), xytext=(-87.98, 35.55),
        fontsize=7.4, color=DARK, bbox=bbox_kw,
        arrowprops={"arrowstyle": "->", "color": DARK, "lw": 0.8, "connectionstyle": "arc3,rad=0.10"},
        zorder=8,
    )
    ax.annotate(
        "Bankhead NF (AL)\n275 plots (139 BACI, 2012–21)",
        xy=(-87.22, 34.31), xytext=(-86.55, 34.48),
        fontsize=7.6, fontweight="bold", color=DARK, bbox=bbox_kw,
        arrowprops={"arrowstyle": "->", "color": DARK, "lw": 0.8, "connectionstyle": "arc3,rad=-0.10"},
        zorder=8,
    )

    # Scale bar (100 km at lat ~34.1 deg: 1 deg lon ~= 92.2 km)
    sb_lon0, sb_lat = -84.35, 34.05
    deg_100km = 100.0 / (111.32 * math.cos(math.radians(36.0)))
    ax.plot([sb_lon0, sb_lon0 + deg_100km], [sb_lat, sb_lat], color=DARK, lw=2.2, zorder=9)
    ax.plot([sb_lon0, sb_lon0], [sb_lat - 0.04, sb_lat + 0.04], color=DARK, lw=1.2, zorder=9)
    ax.plot([sb_lon0 + deg_100km, sb_lon0 + deg_100km], [sb_lat - 0.04, sb_lat + 0.04], color=DARK, lw=1.2, zorder=9)
    ax.text(sb_lon0 + deg_100km / 2, sb_lat + 0.08, "100 km", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=DARK, zorder=9)

    # Minimal North arrow
    nx, ny = -83.12, 34.25
    ax.annotate("", xy=(nx, ny + 0.28), xytext=(nx, ny),
                arrowprops={"arrowstyle": "-|>", "color": DARK, "lw": 1.2, "mutation_scale": 11}, zorder=9)
    ax.text(nx, ny + 0.31, "N", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color=DARK, zorder=9)

    ax.set_xlim(W, E)
    ax.set_ylim(S, N)
    ax.set_xlabel("Longitude (°W, WGS 84 / EPSG:4326)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("Cumberland Plateau Study Transect: 689 Pure-Deciduous 3DEP LiDAR Plots (480 km)", pad=8)
    ax.xaxis.set_major_formatter(lambda x, _: f"{abs(x):.0f}°W")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0f}°N")

    # Right sidebar: Locator inset + Legend + Elevation Colorbar
    ax_in = fig.add_axes([0.815, 0.69, 0.165, 0.21])
    # Clean stylized Eastern US bounding outline
    us_x = [-95, -90, -85, -80, -75, -70, -71, -76, -80.5, -81.5, -84, -88, -90, -95, -95]
    us_y = [48, 47, 46, 44, 43, 45, 41, 35, 26, 30, 30, 30.2, 29.5, 29.5, 48]
    ax_in.fill(us_x, us_y, color="#e6ece8", edgecolor="#6e7c76", lw=0.7)
    rect_in = mpatches.Rectangle((W, S), E - W, N - S, facecolor=RUST, alpha=0.35, edgecolor=RUST, lw=1.1)
    ax_in.add_patch(rect_in)
    ax_in.text(-85.5, 41.8, "Eastern\nU.S.", ha="center", va="center", fontsize=6.5, color=MUTED)
    ax_in.set_xlim(-96, -69)
    ax_in.set_ylim(25, 49)
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    ax_in.set_title("Regional Context", fontsize=7.5, pad=3)

    # Elevation colorbar matching GEE palette
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "plateau_elev", ["#eef1eb", "#dce5d5", "#c3d4b8", "#a4bc97", "#85a179", "#698261", "#54684f"]
    )
    norm = mcolors.Normalize(vmin=150, vmax=750)
    cax = fig.add_axes([0.835, 0.43, 0.022, 0.20])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("Elevation (m a.s.l.)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7.0)

    # Compact legend below colorbar
    handles, labels = ax.get_legend_handles_labels()
    # Add TN protected areas patch
    tn_patch = mpatches.Patch(facecolor="#1d4d3e", edgecolor=DARK, alpha=0.35, lw=0.7, label="TN protected areas (8)")
    handles = [handles[0], tn_patch, *handles[1:]]
    labels = ["TN regional window", "TN protected areas (8)", *labels[1:]]
    fig.legend(
        handles, labels, loc="lower left", bbox_to_anchor=(0.802, 0.08),
        frameon=True, facecolor="#ffffff", edgecolor="#cfd8d3", fontsize=6.7,
        handletextpad=0.45, borderpad=0.45, labelspacing=0.38,
    )

    os.makedirs("paper/figures", exist_ok=True)
    fig.savefig("paper/figures/fig0_study_area_map.png", dpi=300)
    fig.savefig("paper/figures/fig0_study_area_map.pdf")
    plt.close(fig)
    print("Saved paper/figures/fig0_study_area_map.{png,pdf}")


def make_mechanism_schematic() -> None:
    asum = _load_json("data/lidar/analysis_summary.json")
    ols = asum["pooled_fe_ols"]
    bc = asum["sensor_comparison"]["baci_deltas"]
    r_win = bc["d_opt_win_ndvi"]["pearson_r_d_midstory"]
    r_spr = bc["d_opt_spr_ndvi"]["pearson_r_d_midstory"]
    low3 = ols["p_nrd_low_midstory"]["coefs"]["burn_3plus"]["coef"]
    tall_m = ols["p_nrd_tall_midstory"]["coefs"]["mech_plus_fire"]["coef"]
    frc_m = ols["p_first_return_canopy_10m"]["coefs"]["mech_plus_fire"]["coef"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.7), dpi=300, gridspec_kw={"width_ratios": [1.0, 1.22]})
    fig.subplots_adjust(left=0.03, right=0.97, top=0.87, bottom=0.07, wspace=0.12)

    # ---------------- Panel (a): Passive Optical Winter vs. Leaf-Off 3DEP LiDAR ----------------
    ax1.set_xlim(0, 10.6)
    ax1.set_ylim(0, 10)
    ax1.axis("off")
    ax1.set_title("(a) Sensing Physics in Leafless Deciduous Stands", fontsize=9.0, pad=8, color=DARK)

    # Ground line
    ax1.fill_between([0.2, 10.4], [0.5, 0.5], [1.15, 1.15], color="#d9d4c7", zorder=1)
    ax1.plot([0.2, 10.4], [1.15, 1.15], color=DARK, lw=1.0, zorder=2)
    ax1.text(5.3, 0.78, "Forest floor (leaf litter, bryophytes, occasional burn char)", ha="center", va="center", fontsize=6.7, color=MUTED)

    # Left half: Landsat winter optical
    ax1.add_patch(FancyBboxPatch((0.35, 8.3), 4.1, 1.1, boxstyle="round,pad=0.18", facecolor="#faeae4", edgecolor=RUST, lw=0.9))
    ax1.text(2.4, 8.85, "Landsat Optical NDVI\n(Passive Reflectance)", ha="center", va="center", fontsize=7.1, fontweight="bold", color=RUST)

    # Bare canopy + bare stems on left
    for cx in [1.3, 2.5, 3.6]:
        ax1.plot([cx, cx], [1.15, 5.8], color="#6f665d", lw=1.6, zorder=2)
        ax1.add_patch(mpatches.Ellipse((cx, 6.2), 1.35, 1.4, facecolor="none", edgecolor="#8c8378", ls="--", lw=0.9, zorder=2))
    # Dense leafless midstory stems (red maple / beech)
    for mx in [1.0, 1.65, 2.05, 2.85, 3.25, 3.9]:
        ax1.plot([mx, mx], [1.15, 3.6], color="#8f8377", lw=0.95, zorder=2)
        ax1.plot([mx, mx - 0.18], [2.6, 3.1], color="#8f8377", lw=0.7)
        ax1.plot([mx, mx + 0.18], [2.8, 3.3], color="#8f8377", lw=0.7)

    # Optical rays bouncing off litter / sensor jump
    ax1.annotate("", xy=(2.1, 1.25), xytext=(1.6, 8.2), arrowprops={"arrowstyle": "->", "color": RUST, "lw": 1.0, "ls": "--"})
    ax1.annotate("", xy=(3.1, 8.2), xytext=(2.2, 1.25), arrowprops={"arrowstyle": "->", "color": RUST, "lw": 1.1})
    ax1.text(2.4, 4.75, f"Optical BACI deltas fail\n(win r={r_win:+.2f}, spr r={r_spr:+.2f})", ha="center", va="center",
             fontsize=6.5, color=RUST, bbox={"boxstyle": "round,pad=0.22", "fc": "#fff8f5", "ec": RUST, "lw": 0.6})

    # Divider
    ax1.plot([4.95, 4.95], [1.3, 9.3], color="#cbd4cf", lw=0.9, ls=":")

    # Right half: Airborne 3DEP LiDAR
    ax1.add_patch(FancyBboxPatch((5.35, 8.3), 4.55, 1.1, boxstyle="round,pad=0.18", facecolor="#e1ede8", edgecolor=PINE, lw=0.9))
    ax1.text(7.62, 8.85, "USGS 3DEP Airborne LiDAR\n(Active 3D Laser Pulses)", ha="center", va="center", fontsize=7.0, fontweight="bold", color=PINE)

    # Canopy & stems on right
    for cx in [6.2, 7.45, 8.65]:
        ax1.plot([cx, cx], [1.15, 5.8], color="#4a4640", lw=1.6, zorder=2)
        ax1.add_patch(mpatches.Ellipse((cx, 6.2), 1.32, 1.4, facecolor="#e8eee9", edgecolor=PINE, lw=0.8, alpha=0.55, zorder=2))
    for mx in [5.85, 6.6, 7.05, 7.85, 8.25, 8.95]:
        ax1.plot([mx, mx], [1.15, 3.7], color=BLUE, lw=1.05, zorder=2)

    # Laser pulses & returns at multiple height strata
    for lx, hits in [(6.05, [6.4, 3.2, 1.15]), (7.25, [5.9, 4.1, 2.2, 1.15]), (8.45, [6.3, 3.5, 1.7, 1.15])]:
        ax1.plot([lx, lx], [8.2, 1.15], color=PINE, lw=0.8, alpha=0.65, zorder=3)
        for hy in hits:
            ax1.scatter([lx], [hy], s=18, color=PINE if hy > 5 else (BLUE if hy > 1.5 else DARK), zorder=5)

    # Stratum brackets on far right
    for y0, y1, txt, col in [
        (5.2, 7.1, ">10 m\n1st-ret.", PINE),
        (3.6, 5.1, "5–10 m", MULBERRY),
        (2.2, 3.5, "2–5 m", BLUE),
        (1.2, 2.1, "0.5–2 m", AMBER),
    ]:
        ax1.plot([9.35, 9.45, 9.45, 9.35], [y0, y0, y1, y1], color=col, lw=0.9)
        ax1.text(9.53, (y0 + y1) / 2, txt, va="center", ha="left", fontsize=6.1, fontweight="bold", color=col)

    ax1.text(7.30, 4.65, "Penetration-adjusted\nNRD + first-return\n>10 m canopy cover", ha="center", va="center",
             fontsize=6.5, color=PINE, bbox={"boxstyle": "round,pad=0.22", "fc": "#f3f9f6", "ec": PINE, "lw": 0.6})

    # ---------------- Panel (b): Vertical Attenuation Profile Across Height Zones ----------------
    ax2.set_xlim(0, 11.4)
    ax2.set_ylim(0, 10)
    ax2.axis("off")
    ax2.set_title("(b) Vertical Response Pattern Across Height Strata", fontsize=9.0, pad=8, color=DARK)

    # Ground
    ax2.fill_between([0.75, 11.15], [0.95, 0.95], [1.45, 1.45], color="#d9d4c7", zorder=1)
    ax2.plot([0.75, 11.15], [1.45, 1.45], color=DARK, lw=1.0, zorder=2)

    # Y-axis height markers
    ax2.plot([0.75, 0.75], [1.45, 8.9], color=MUTED, lw=0.8)
    for yv, lbl in [(1.45, "0 m"), (2.6, "2 m"), (4.3, "5 m"), (6.5, "10 m"), (8.4, "25 m")]:
        ax2.plot([0.60, 0.75], [yv, yv], color=MUTED, lw=0.8)
        ax2.text(0.52, yv, lbl, ha="right", va="center", fontsize=6.8, color=MUTED)

    # Gradual 4-7 m crossover band (reflecting continuous 1 m profile rather than a sharp cliff)
    ax2.axhspan(3.85, 5.20, xmin=0.07, xmax=0.98, color="#f7efe8", alpha=0.85, zorder=1)
    ax2.plot([0.75, 11.15], [4.3, 4.3], color=RUST, lw=0.9, ls=":", zorder=4)
    ax2.text(11.08, 4.42, "4–7 m crossover zone", ha="right", va="bottom",
             fontsize=6.6, fontweight="bold", color=RUST,
             bbox={"boxstyle": "round,pad=0.14", "fc": "#ffffff", "ec": "none", "alpha": 0.88}, zorder=7)

    # 10 m canopy line
    ax2.plot([0.75, 11.15], [6.5, 6.5], color="#9aa8a1", lw=0.7, ls=":", zorder=2)

    def draw_stand(x0: float, title: str, badge: str, badge_col: str, under_n: int, low_n: int, tall_n: int, open_canopy: bool = False):
        oak_xs = [x0 + 0.58, x0 + 1.92]
        for ox in oak_xs:
            ax2.plot([ox, ox], [1.45, 7.2], color="#594a3d", lw=2.2, zorder=2)
            w = 1.12 if open_canopy else 1.42
            ax2.add_patch(mpatches.Ellipse((ox, 7.7), w, 1.35, facecolor="#c8dfd3", edgecolor=PINE, lw=0.85, alpha=0.85, zorder=3))
        for i in range(tall_n):
            tx = x0 + 0.36 + (i + 0.5) * (1.80 / max(tall_n, 1))
            ax2.plot([tx, tx], [1.45, 5.7], color=MULBERRY, lw=1.25, zorder=3)
            ax2.add_patch(mpatches.Ellipse((tx, 5.5), 0.52, 0.88, facecolor="#ebd2de", edgecolor=MULBERRY, lw=0.7, alpha=0.85, zorder=3))
        for i in range(low_n):
            lx = x0 + 0.25 + (i + 0.5) * (2.00 / max(low_n, 1))
            ax2.plot([lx, lx], [1.45, 3.65], color=BLUE, lw=1.0, zorder=3)
            ax2.add_patch(mpatches.Ellipse((lx, 3.38), 0.42, 0.70, facecolor="#cde2f2", edgecolor=BLUE, lw=0.65, alpha=0.88, zorder=3))
        for i in range(under_n):
            ux = x0 + 0.20 + (i + 0.5) * (2.10 / max(under_n, 1))
            ax2.plot([ux, ux], [1.45, 2.22], color=AMBER, lw=0.95, zorder=3)
            ax2.scatter([ux], [2.18], s=14, color=AMBER, zorder=4)

        ax2.text(x0 + 1.25, 9.25, title, ha="center", va="center", fontsize=7.5, fontweight="bold", color=DARK)
        ax2.text(x0 + 1.25, 0.42, badge, ha="center", va="center", fontsize=6.0, color=badge_col, fontweight="bold")

    draw_stand(1.05, "Fire-Excluded\n(0 Burns)", "Dense 0.5–10 m\nmidstory packing", SLATE, under_n=7, low_n=5, tall_n=4)
    draw_stand(4.65, "Repeated Fire\n(3+ Burns)", f"2–5 m ({low3:+.3f}); fades >6 m\n1st-ret. >10 m n.s.", BLUE, under_n=2, low_n=1, tall_n=3)
    for fx in [4.95, 5.55, 6.15, 6.75]:
        ax2.scatter([fx], [1.60], marker="^", s=24, color=RUST, zorder=6)

    draw_stand(8.25, "Mechanical +\nPrescribed Fire", f"5–10 m ({tall_m:+.3f}, cross-sec.)\n1st-ret. >10 m ({frc_m:+.3f})", MULBERRY, under_n=3, low_n=2, tall_n=1, open_canopy=True)
    for fx in [8.55, 9.45, 10.35]:
        ax2.scatter([fx], [1.60], marker="^", s=24, color=RUST, zorder=6)

    fig.savefig("paper/figures/fig_mechanism_schematic.png", dpi=300)
    fig.savefig("paper/figures/fig_mechanism_schematic.pdf")
    plt.close(fig)
    print("Saved paper/figures/fig_mechanism_schematic.{png,pdf}")


if __name__ == "__main__":
    make_study_area_map()
    make_mechanism_schematic()
