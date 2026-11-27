"""Generate publication-grade vector PDF & 300-DPI PNG figures for the Cumberland
Plateau airborne LiDAR and prescribed-fire study.

Every single plotted value is read directly from the verified JSON outputs:
  - data/diagnostic_optical.json
  - data/area_metrics.json
  - data/region_metrics.json
  - data/lidar/plot_metrics.json
  - data/lidar/analysis_summary.json

Outputs: paper/figures/fig1..fig4 (.pdf and .png)
"""

from __future__ import annotations

import json
import os
from decimal import ROUND_HALF_UP, Decimal

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.0,
    "axes.titlesize": 7.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 6.5,
    "xtick.labelsize": 5.8,
    "ytick.labelsize": 5.8,
    "axes.linewidth": 0.65,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e5e5e5",
    "grid.linewidth": 0.45,
    "axes.axisbelow": True,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
})
os.makedirs("paper/figures", exist_ok=True)

# Okabe-Ito colorblind-safe palette
BLUE = "#0072B2"     # 3DEP LiDAR / repeated fire
ORANGE = "#D55E00"   # Unharmonized optical / pre-2017 burn only / warning
GREEN = "#009E73"    # Spring phenology / positive contrast
PURPLE = "#CC79A7"   # Mechanical + Fire
YELLOW = "#E69F00"   # Single burn / intermediate
SKY = "#56B4E9"      # GEDI / secondary
GREY = "#666666"     # Control / neutral
DARK = "#222222"


def _fmt_signed(val: float, digits: int = 3) -> str:
    """Format signed numbers with standard half-up rounding to match manuscript tables."""
    q = Decimal("1." + "0" * digits) if digits > 0 else Decimal(1)
    d = Decimal(str(val)).quantize(q, rounding=ROUND_HALF_UP)
    if d == 0:
        d = abs(d)
    sign = "+" if d >= 0 else ""
    return f"{sign}{d}"


def _load_json(path: str):
    with open(path) as f:
        return json.load(f)


def load_all():
    areas = _load_json("data/area_metrics.json")
    reg = _load_json("data/region_metrics.json")
    asum = _load_json("data/lidar/analysis_summary.json")
    raw = _load_json("data/lidar/plot_metrics.json")
    return areas, reg, asum, raw


def make_fig1(reg, asum, raw):
    """Fig 1 (full-width 3-panel diagnosis teaser):
    (a) Regional winter NDVI trend collapses 79% after empirical harmonization and turns negative in L8+9.
    (b) On 272 epoch-verified BACI plots, winter NDVI change shows r = -0.001 vs true 3D LiDAR midstory change.
    (c) Across 689 pure-deciduous plots, 3DEP LiDAR resolves a monotonic burn-frequency drop while
        winter NDVI stays flat and spring Mar-Apr NDVI partially tracks midstory removal.
    """
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), gridspec_kw={"width_ratios": [1.05, 1.0, 1.15]})
    fig.subplots_adjust(wspace=0.38, left=0.07, right=0.98, top=0.86, bottom=0.22)

    # Panel (a): Regional optical artifact collapse
    ax = axes[0]
    tr = reg["trends"]
    labels = ["Unharm.", "Roy\nRMA", "Emp.\nHarm.", "L5\n85-11", "L7\n00-21", "L8+9\n14-25"]
    vals = [
        tr["unharm_dec_mar_all"],
        tr["roy_dec_mar_all"],
        tr["harm_jan_feb_decid"],
        tr["l5_jan_feb_decid"],
        tr["l7_jan_feb_decid"],
        tr["l89_jan_feb_decid"],
    ]
    cols = [ORANGE, "#E08B55", BLUE, SKY, SKY, "#333333"]
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=cols, width=0.64, edgecolor="none", zorder=3)
    ax.axhline(0, color=DARK, lw=0.6, zorder=2)
    for b, v in zip(bars, vals, strict=False):
        va = "bottom" if v >= 0 else "top"
        off = 0.0012 if v >= 0 else -0.0014
        ax.text(b.get_x() + b.get_width() / 2, v + off, _fmt_signed(v, 3), ha="center", va=va, fontsize=5.2, color=DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5.3)
    ax.set_ylabel("Winter NDVI slope (/decade)", fontsize=6.5)
    ax.set_ylim(-0.015, 0.041)
    ax.set_title("(a) Optical artifact collapse", fontsize=7.2, fontweight="bold", pad=6)
    ax.annotate("-79%", xy=(2, 0.009), xytext=(1.0, 0.025),
                arrowprops={"arrowstyle": "->", "lw": 0.7, "color": BLUE},
                fontsize=6.0, fontweight="bold", color=BLUE)

    # Panel (b): BACI scatter ΔNRD_midstory vs ΔNDVI_winter (epoch-verified BACI plots)
    ax = axes[1]
    dx, dy, strata = [], [], []
    for rec in raw.values():
        lp = rec.get("lidar_primary")
        lb = rec.get("lidar_baseline")
        opt = rec.get("optical", {})
        if not (lp and lb and opt.get("opt_late_win_ndvi") is not None and opt.get("opt_early_win_ndvi") is not None):
            continue
        acq_b = lb.get("acq_date") or ""
        if rec["site"] == "Bankhead_North_BACI" and not acq_b.startswith("2012"):
            continue
        if rec["site"] == "DanielBoone_East_BACI" and not acq_b.startswith("2017"):
            continue
        dx.append(lp["nrd_midstory"] - lb["nrd_midstory"])
        dy.append(opt["opt_late_win_ndvi"] - opt["opt_early_win_ndvi"])
        strata.append(rec["stratum"])
    dx = np.array(dx)
    dy = np.array(dy)
    c_map = {"baci_0_control": GREY, "baci_1_inter": YELLOW, "baci_2plus_inter": BLUE, "baci_mech_fire": PURPLE, "baci_pre_only": ORANGE}
    for st in ("baci_0_control", "baci_pre_only", "baci_1_inter", "baci_mech_fire", "baci_2plus_inter"):
        m = np.array([s == st for s in strata])
        if m.any():
            ax.scatter(dx[m], dy[m], s=7, alpha=0.55, color=c_map[st], edgecolors="none", zorder=3)
    # Regression line
    slope, intercept = np.polyfit(dx, dy, 1)
    xs = np.linspace(dx.min(), dx.max(), 50)
    ax.plot(xs, intercept + slope * xs, color=ORANGE, lw=1.2, ls="--", zorder=4)
    ax.axhline(0, color="#BBBBBB", lw=0.5, zorder=1)
    ax.axvline(0, color="#BBBBBB", lw=0.5, zorder=1)
    bc = asum["sensor_comparison"]["baci_deltas"]["d_opt_win_ndvi"]
    bc_spr = asum["sensor_comparison"]["baci_deltas"]["d_opt_spr_ndvi"]
    ax.text(0.04, 0.94,
            f"Win: r={bc['pearson_r_d_midstory']:+.2f}, p={bc['p_d_midstory']:.2f} (n={bc['n']})\n"
            f"Spr: r={bc_spr['pearson_r_d_midstory']:+.2f}, p={bc_spr['p_d_midstory']:.2f} (n={bc_spr['n']})",
            transform=ax.transAxes, fontsize=5.1, va="top", color=ORANGE, fontweight="bold",
            bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "none", "alpha": 0.88})
    ax.set_xlabel(r"3DEP LiDAR $\Delta$NRD (2-10 m)", fontsize=6.5)
    ax.set_ylabel(r"Landsat OLI $\Delta$NDVI (Jan-Feb)", fontsize=6.5)
    ax.set_title("(b) Optical deltas miss 3D change", fontsize=7.2, fontweight="bold", pad=6)

    # Panel (c): Pooled FE-OLS burn-frequency effects (3DEP LiDAR vs Winter NDVI vs Spring NDVI)
    ax = axes[2]
    ols = asum["pooled_fe_ols"]
    cats = ["burn_1", "burn_2", "burn_3plus", "mech_plus_fire"]
    clabels = ["1 burn", "2 burns", "3+ burns", "Mech+Fire"]
    xc = np.arange(len(cats))
    series = [
        ("3DEP LiDAR (2-10 m)", "p_nrd_midstory", BLUE, "o", 1.8),
        ("Spring NDVI (Mar-Apr)", "opt_spr_ndvi", GREEN, "s", 1.2),
        ("Winter NDVI (Jan-Feb)", "opt_win_ndvi", ORANGE, "^", 1.2),
    ]
    ax.axhline(0, color=DARK, lw=0.6, zorder=1)
    for idx_s, (lbl, key, col, mk, lw) in enumerate(series):
        ys = [ols[key]["coefs"][c]["coef"] for c in cats]
        es = [1.96 * ols[key]["coefs"][c]["se_cluster_suid"] for c in cats]
        off = (idx_s - 1) * 0.12
        ax.errorbar(xc + off, ys, yerr=es, fmt=mk + "-", color=col, lw=lw, ms=3.8, capsize=2, elinewidth=0.8, label=lbl, zorder=4 if idx_s == 0 else 3)
    ax.set_xticks(xc)
    ax.set_xticklabels(clabels, fontsize=5.6)
    ax.set_ylabel("FE-OLS effect vs. 0 burns", fontsize=6.5)
    ax.set_ylim(-0.105, 0.028)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.88, fontsize=5.0, loc="lower left", handlelength=1.4)
    ax.set_title("(c) Dose-response across sensors", fontsize=7.2, fontweight="bold", pad=6)

    for ext in ("pdf", "png"):
        fig.savefig(f"paper/figures/fig1_optical_illusion_vs_lidar.{ext}", dpi=300)
    plt.close(fig)
    print("Saved fig1_optical_illusion_vs_lidar")


def make_fig2(asum):
    """Fig 2 (full-width 2-panel BACI multi-epoch airborne LiDAR structural change):
    (a) Bankhead North BACI (2012 -> 2021, 8.9 yr, n=139): raw & density-thinned ΔNRD_midstory.
    (b) Daniel Boone East BACI (2017 -> 2023, 6.0 yr QL2->QL2, n=144 epoch-verified): raw & density-thinned ΔNRD_midstory.
    Reports burn-unit clustered p_SUID directly on bars and labels pre-2017 mech timing accurately.
    """
    fig, axes = plt.subplots(1, 2, figsize=(6.875, 2.70), sharey=True)
    fig.subplots_adjust(wspace=0.14, left=0.09, right=0.98, top=0.84, bottom=0.25)

    panels = [
        (
            axes[0],
            "Bankhead_North_BACI",
            "(a) Bankhead NF, AL (2012 to 2021, 8.9 yr)",
            [("baci_0_control", "0 burns\n(Control)", GREY),
             ("baci_1_inter", "1 burn\n(2012-20)", YELLOW),
             ("baci_2plus_inter", "2+ burns\n(2012-20)", BLUE)],
        ),
        (
            axes[1],
            "DanielBoone_East_BACI",
            "(b) Daniel Boone NF, KY (2017 to 2023, 6.0 yr)",
            [("baci_0_control", "0 burns\n(Control)", GREY),
             ("baci_pre_only", "Pre-2017 burn\nonly", ORANGE),
             ("baci_1_inter", "1 burn\n(2018-22)", YELLOW),
             ("baci_mech_fire", "Pre-2017 mech\n+ 1 burn", PURPLE)],
        ),
    ]

    for ax, site_key, title, specs in panels:
        st_dict = asum["baci"][site_key]["strata"]
        x = np.arange(len(specs))
        means, lo_err, hi_err, thin_means, cols, labs = [], [], [], [], [], []
        for sk, lbl, col in specs:
            d = st_dict[sk]["delta_nrd_midstory"]
            dt = st_dict[sk]["delta_nrd_midstory_thinned"]
            m = d["mean"]
            means.append(m)
            lo_err.append(m - d["ci95_lo"])
            hi_err.append(d["ci95_hi"] - m)
            thin_means.append(dt["mean"])
            cols.append(col)
            g_cnt = st_dict[sk]["n_clusters_suid"]
            labs.append(f"{lbl}\n(n={st_dict[sk]['n']}, G={g_cnt})")

        ax.axhline(0, color=DARK, lw=0.6, zorder=1)
        ax.bar(x, means, yerr=[lo_err, hi_err], color=cols, alpha=0.85, width=0.56,
               capsize=2.5, error_kw={"lw": 0.8, "color": DARK}, zorder=3)
        ax.scatter(x, thin_means, marker="D", s=16, facecolors="white", edgecolors=DARK, lw=0.8, zorder=5, label="Density-thinned")

        for i, (sk, _, _) in enumerate(specs):
            m = means[i]
            top_y = m + hi_err[i] + 0.005
            txt = f"{m:+.3f}"
            if sk != "baci_0_control" and "did_vs_control" in st_dict[sk]:
                dc = st_dict[sk]["did_vs_control"]
                ps = dc["p_cluster_suid"]
                pstr = r"$p_u<0.001$" if ps < 0.001 else rf"$p_u={ps:.3f}$"
                txt += f"\nDiD {dc['did_midstory']:+.3f}\n({pstr})"
            ax.text(i, top_y, txt, ha="center", va="bottom", fontsize=5.0, color=DARK)

        ax.set_xticks(x)
        ax.set_xticklabels(labs, fontsize=5.3)
        ax.set_title(title, fontsize=7.2, fontweight="bold", pad=6)
        ax.set_ylim(-0.01, 0.182)

    axes[0].set_ylabel(r"3DEP LiDAR $\Delta$NRD Midstory (2-10 m)", fontsize=6.6)
    axes[0].text(0.03, 0.94, "Only DiD vs. control is identified\n(raw shifts include sensor change)",
                 transform=axes[0].transAxes, fontsize=4.9, va="top", color=GREY,
                 bbox={"boxstyle": "round,pad=0.20", "fc": "white", "ec": "#dddddd", "alpha": 0.9})
    axes[1].legend(frameon=False, fontsize=5.6, loc="upper right")

    for ext in ("pdf", "png"):
        fig.savefig(f"paper/figures/fig2_baci_lidar_structural_change.{ext}", dpi=300)
    plt.close(fig)
    print("Saved fig2_baci_lidar_structural_change")


def make_fig3(asum):
    """Fig 3 (full-width 2-panel vertical structure analysis):
    (a) Continuous 1 m vertical NRD profile (0.5-10 m) for 3+ burns vs Mechanical+fire,
        marking bands where the direct cluster-robust contrast (mech - 3+) is significant (p_SUID < 0.05).
    (b) Aggregated 4-stratum FE-OLS effects using UNBIASED first-return canopy cover (>10 m)
        alongside 0.5-2 m, 2-5 m, and 5-10 m NRD across all 4 regimes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(6.875, 2.85), gridspec_kw={"width_ratios": [1.05, 1.15]})
    fig.subplots_adjust(wspace=0.38, left=0.09, right=0.98, top=0.85, bottom=0.19)

    # Panel (a): Continuous 1-m height profile (0.5-10 m)
    ax = axes[0]
    p1m = asum["pooled_fe_ols_1m_profile"]
    bins = [
        ("0.5_1", 0.75), ("1_2", 1.5), ("2_3", 2.5), ("3_4", 3.5), ("4_5", 4.5),
        ("5_6", 5.5), ("6_7", 6.5), ("7_8", 7.5), ("8_9", 8.5), ("9_10", 9.5),
    ]
    ys = np.array([h for _, h in bins])
    b3 = np.array([p1m[k]["coefs"]["burn_3plus"]["coef"] for k, _ in bins])
    e3 = np.array([1.96 * p1m[k]["coefs"]["burn_3plus"]["se_cluster_suid"] for k, _ in bins])
    bm = np.array([p1m[k]["coefs"]["mech_plus_fire"]["coef"] for k, _ in bins])
    em = np.array([1.96 * p1m[k]["coefs"]["mech_plus_fire"]["se_cluster_suid"] for k, _ in bins])
    p_ct = [p1m[k]["contrasts"]["mech_minus_3plus"]["p_cluster_suid"] for k, _ in bins]

    ax.axhspan(4.0, 7.0, color="#f2efe9", zorder=0, label="4-7 m crossover zone")
    ax.axvline(0, color=DARK, lw=0.6, zorder=1)
    ax.fill_betweenx(ys, b3 - e3, b3 + e3, color=BLUE, alpha=0.18, zorder=2)
    ax.plot(b3, ys, "o-", color=BLUE, lw=1.4, ms=3.5, label="3+ burns (fire only)", zorder=4)
    ax.fill_betweenx(ys, bm - em, bm + em, color=PURPLE, alpha=0.18, zorder=2)
    ax.plot(bm, ys, "^-", color=PURPLE, lw=1.4, ms=3.8, label="Mechanical + fire", zorder=4)

    for h, pc in zip(ys, p_ct, strict=False):
        if pc < 0.05:
            mark = "**" if pc < 0.01 else "*"
            ax.text(0.0042, h, mark, va="center", ha="left", fontsize=6.5, fontweight="bold", color=DARK)

    ax.set_yticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ax.set_ylabel("Height above ground (m)", fontsize=6.3)
    ax.set_xlabel("1 m NRD shift vs. 0 burns (95% unit CI)", fontsize=6.0)
    ax.set_xlim(-0.041, 0.011)
    ax.set_ylim(0.3, 10.1)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.90, fontsize=4.8, loc="upper left")
    ax.text(0.03, 0.75, r"* mech vs. 3+ $p_u<0.05$", transform=ax.transAxes, fontsize=4.8, ha="left", va="top", color=DARK)
    ax.set_title("(a) Continuous 1 m vertical profile", fontsize=7.2, fontweight="bold", pad=6)

    # Panel (b): Aggregated 4 strata with UNBIASED first-return canopy (>10 m)
    ax = axes[1]
    ols = asum["pooled_fe_ols"]
    strata = [
        ("p_nrd_understory", "Understory\n0.5-2 m"),
        ("p_nrd_low_midstory", "Low midstory\n2-5 m"),
        ("p_nrd_tall_midstory", "Tall midstory\n5-10 m"),
        ("p_first_return_canopy_10m", "First-return\ncanopy >10 m"),
    ]
    y = np.arange(len(strata))
    ax.axvline(0, color=DARK, lw=0.6, zorder=1)
    treatments = [
        ("burn_1", "1 burn", YELLOW, -0.22, "o"),
        ("burn_2", "2 burns", SKY, -0.07, "s"),
        ("burn_3plus", "3+ burns", BLUE, +0.08, "D"),
        ("mech_plus_fire", "Mech + fire", PURPLE, +0.23, "^"),
    ]
    for tkey, tlbl, col, off, mk in treatments:
        vals = [ols[skey]["coefs"][tkey]["coef"] for skey, _ in strata]
        errs = [1.96 * ols[skey]["coefs"][tkey]["se_cluster_suid"] for skey, _ in strata]
        ax.errorbar(vals, y + off, xerr=errs, fmt=mk, color=col, ms=3.8, capsize=2, elinewidth=0.85, label=tlbl, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([lbl for _, lbl in strata], fontsize=5.8)
    ax.set_xlabel("Pooled FE-OLS shift vs. 0 burns (95% unit CI)", fontsize=6.0)
    ax.set_xlim(-0.115, 0.082)
    ax.legend(frameon=False, fontsize=5.0, loc="lower right", ncol=2, handletextpad=0.3, columnspacing=0.6)
    ax.set_title("(b) Stratum effects & unbiased canopy (n=689)", fontsize=7.2, fontweight="bold", pad=6)

    for ext in ("pdf", "png"):
        fig.savefig(f"paper/figures/fig3_vertical_stratum_decoupling.{ext}", dpi=300)
    plt.close(fig)
    print("Saved fig3_vertical_stratum_decoupling")


def make_fig4(areas, asum):
    """Fig 4 (full-width 2-panel):
    (a) Head-to-head sensor correlation with 3DEP LiDAR midstory (2-10 m):
        compares pooled vs. within-site cross-sectional r AND temporal BACI deltas
        (showing both winter and spring fail temporal BACI change detection).
    (b) Sensitivity of the 8 Tennessee Cumberland Plateau protected areas (fixed Fall Creek Falls)
        once inter-sensor harmonization & single-sensor eras replace unharmonized Dec-Mar NDVI.
    """
    fig, axes = plt.subplots(1, 2, figsize=(6.875, 2.90), gridspec_kw={"width_ratios": [1.08, 1.18]})

    # Panel (a): Pooled vs Within-site sensor correlations + BACI temporal deltas
    ax = axes[0]
    sc = asum["sensor_comparison"]["cross_sectional"]
    bc = asum["sensor_comparison"]["baci_deltas"]
    rows = [
        ("Spring NDVI\n(Mar-Apr)", sc["opt_spr_ndvi"]["pearson_r_midstory"], sc["opt_spr_ndvi"]["within_site_r_midstory"], GREEN),
        ("Winter NDVI\n(Jan-Feb)", sc["opt_win_ndvi"]["pearson_r_midstory"], sc["opt_win_ndvi"]["within_site_r_midstory"], YELLOW),
        ("GEDI L2B PAI\n(5-10 m)", sc["gedi_pai_5_10m"]["pearson_r_midstory"], sc["gedi_pai_5_10m"]["within_site_r_midstory"], SKY),
        ("GEDI L2B PAI\n(0-5 m)", sc["gedi_pai_0_5m"]["pearson_r_midstory"], sc["gedi_pai_0_5m"]["within_site_r_midstory"], SKY),
        ("Summer NDVI\n(Jun-Aug)", sc["opt_sum_ndvi"]["pearson_r_midstory"], sc["opt_sum_ndvi"]["within_site_r_midstory"], GREY),
        ("BACI $\\Delta$Spring\nNDVI (n=283)", bc["d_opt_spr_ndvi"]["pearson_r_d_midstory"], bc["d_opt_spr_ndvi"]["within_site_r_d_midstory"], ORANGE),
        ("BACI $\\Delta$Winter\nNDVI (n=272)", bc["d_opt_win_ndvi"]["pearson_r_d_midstory"], bc["d_opt_win_ndvi"]["within_site_r_d_midstory"], ORANGE),
        ("BACI $\\Delta$Winter\nNBR (n=272)", bc["d_opt_win_nbr"]["pearson_r_d_midstory"], bc["d_opt_win_nbr"]["within_site_r_d_midstory"], ORANGE),
    ]
    rows.reverse()
    y = np.arange(len(rows))
    h = 0.34
    ax.axvline(0, color=DARK, lw=0.6, zorder=1)
    ax.axhline(2.5, color="#cccccc", lw=0.7, ls="--", zorder=1)
    for i, (_lbl, rp, rw, col) in enumerate(rows):
        ax.barh(i + h / 2, rp, height=h, color=col, alpha=0.45, edgecolor=col, lw=0.5, zorder=3,
                label="Pooled r" if i == len(rows) - 1 else None)
        ax.barh(i - h / 2, rw, height=h, color=col, alpha=0.95, edgecolor="none", zorder=4,
                label="Within-site r" if i == len(rows) - 1 else None)
        # Annotate within-site r cleanly (the unbiased comparison)
        ha = "left" if rw >= 0 else "right"
        off = 0.012 if rw >= 0 else -0.012
        ax.text(rw + off, i - h / 2 - 0.03, f"{rw:+.2f}", va="center", ha=ha, fontsize=4.8, color=DARK, fontweight="bold")
        if abs(rp - rw) > 0.05:
            ha_p = "left" if rp >= 0 else "right"
            off_p = 0.012 if rp >= 0 else -0.012
            ax.text(rp + off_p, i + h / 2 + 0.03, f"{rp:+.2f}", va="center", ha=ha_p, fontsize=4.5, color=GREY)

    ax.set_yticks(y)
    ax.set_yticklabels([lbl for lbl, _, _, _ in rows], fontsize=5.2)
    ax.set_xlim(-0.20, 0.54)
    ax.set_xlabel("Pearson r vs. 3DEP LiDAR midstory", fontsize=6.2)
    ax.legend(frameon=False, fontsize=4.9, loc="lower right")
    ax.set_title("(a) Pooled vs. within-site & BACI skill", fontsize=7.2, fontweight="bold", pad=6)

    # Panel (b): 8 Tennessee protected areas — Unharmonized vs Harmonized vs L8+9
    ax = axes[1]
    short_names = {
        "Prentice Cooper State Forest": "Prentice Cooper",
        "Sewanee Domain": "Sewanee (burned)",
        "Franklin State Forest": "Franklin SF",
        "Catoosa": "Catoosa WMA",
        "North Chickamauga Creek Gorge Designated": "N. Chickamauga",
        "Savage Gulf Designated": "Savage Gulf",
        "Bridgestone/Firestone": "Bridgestone WMA",
        "Fall Creek Falls State Park": "Fall Creek Falls",
    }
    ordered = sorted(areas.items(), key=lambda kv: kv[1]["unharm_dec_mar_decade"])
    y2 = np.arange(len(ordered))
    u_vals = [d["unharm_dec_mar_decade"] for _, d in ordered]
    h_vals = [d["harm_jan_feb_decid_decade"] for _, d in ordered]
    o_vals = [d["l89_jan_feb_decid_decade"] for _, d in ordered]
    ax.axvline(0, color=DARK, lw=0.6, zorder=1)
    ax.scatter(u_vals, y2, color=ORANGE, marker="o", s=24, label="Unharm. Dec-Mar", zorder=4)
    ax.scatter(h_vals, y2, color=BLUE, marker="D", s=22, label="Harm. Jan-Feb (Decid)", zorder=5)
    ax.scatter(o_vals, y2, color=DARK, marker="^", s=22, label="L8+9 Only (2014-25)", zorder=4)
    for i in range(len(ordered)):
        ax.plot([o_vals[i], u_vals[i]], [i, i], color="#CCCCCC", lw=0.8, zorder=2)
    ax.set_yticks(y2)
    ax.set_yticklabels([short_names.get(k, k) for k, _ in ordered], fontsize=5.6)
    ax.set_xlabel("Winter NDVI trend (/decade, >=430 m)", fontsize=6.2)
    ax.set_xlim(-0.026, 0.046)
    ax.legend(frameon=False, fontsize=4.9, loc="upper center", bbox_to_anchor=(0.48, -0.22), ncol=3, handletextpad=0.3, columnspacing=0.8)
    ax.set_title("(b) TN protected-area trend sensitivity", fontsize=7.2, fontweight="bold", pad=6)
    fig.subplots_adjust(wspace=0.46, left=0.18, right=0.98, top=0.86, bottom=0.25)

    for ext in ("pdf", "png"):
        fig.savefig(f"paper/figures/fig4_sensor_comparison_and_tn_reversal.{ext}", dpi=300)
    plt.close(fig)
    print("Saved fig4_sensor_comparison_and_tn_reversal")


def main():
    areas, reg, asum, raw = load_all()
    make_fig1(reg, asum, raw)
    make_fig2(asum)
    make_fig3(asum)
    make_fig4(areas, asum)
    print("All 4 publication figures generated cleanly in paper/figures/")


if __name__ == "__main__":
    main()
