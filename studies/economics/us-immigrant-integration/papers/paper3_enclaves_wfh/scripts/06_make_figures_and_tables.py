#!/usr/bin/env python3
"""
Paper 3 — Script 06 (Revised): Generate 4 Publication-Grade PNG Figures & 5 LaTeX `booktabs` Tables.

Figures:
  - figures/fig1_cpuma_neighborhood_vs_msa.png
  - figures/fig2_telework_event_study_2015_2024.png
  - figures/fig3_enclave_arbitrage_vs_abandonment.png
  - figures/fig4_jrs_weak_iv_robust.png

LaTeX Tables (100% pure ASCII, zero em-dashes, compact for zero float-only pages):
  - tables/tab1_cpuma_housing_crowding.tex
  - tables/tab2_telework_did_retention.tex
  - tables/tab3_enclave_arbitrage_movers.tex
  - tables/tab3b_sensitivity_grid.tex
  - tables/tab4_jrs_sw_ar_inference.tex
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

ROOT = Path(__file__).resolve().parents[3]
P3 = ROOT / "papers" / "paper3_enclaves_wfh"
RES = P3 / "results"
FIG = P3 / "figures"
TAB = P3 / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

NAVY = "#1b4965"
CRIMSON = "#c1121f"
TEAL = "#2a9d8f"
AMBER = "#d97706"
SLATE = "#475569"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})


def stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


# ===========================================================================
# FIGURE 1: Sub-Metro Housing Prices (Structure-Only vs. Full Hedonic) & Crowding
# ===========================================================================
def make_fig1(m1: pd.DataFrame, msa_saiz: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.8))

    def get_row(df: pd.DataFrame, spec: str) -> tuple[float, float]:
        r = df[df["spec"] == spec].iloc[0]
        return float(r["coef"]), float(r["se"])

    panel_a_items = [
        ("MSA: Zillow ZORI Rent (2019-24, OLS)", *get_row(msa_saiz, "MSA_Zillow_ZORI_OLS"), TEAL),
        ("MSA: Zillow ZHVI Value (2000-24, IV)", *get_row(msa_saiz, "MSA_Zillow_ZHVI_BartikIV"), TEAL),
        ("CPUMA (07-21): Rent (Structure-Only)", *get_row(m1, "CPUMA_2007_2021_lnRent_StructOnly"), NAVY),
        ("CPUMA (07-21): Rent (Full Hedonic)", *get_row(m1, "CPUMA_2007_2021_lnRent_Hedonic"), NAVY),
        ("CPUMA (07-21): Value (Structure-Only)", *get_row(m1, "CPUMA_2007_2021_lnVal_StructOnly"), NAVY),
        ("Co-Eth CPUMA (07-21): Rent (Struct-Only)", *get_row(m1, "CoEth_CPUMA_2007_2021_lnRent_StructOnly"), AMBER),
        ("Co-Eth CPUMA (07-21): Value (Struct-Only)", *get_row(m1, "CoEth_CPUMA_2007_2021_lnVal_StructOnly"), AMBER),
        ("PUMA (22-24): Rent (Structure-Only)", *get_row(m1, "PUMA_2022_2024_lnRent_StructOnly"), CRIMSON),
        ("PUMA (22-24): Value (Structure-Only)", *get_row(m1, "PUMA_2022_2024_lnVal_StructOnly"), CRIMSON),
    ]

    y_a = np.arange(len(panel_a_items))
    ax1.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    for i, (lbl, c, s, col) in enumerate(panel_a_items):
        ax1.errorbar(c, i, xerr=1.96 * s, fmt="o", color=col, ecolor=col, elinewidth=2.0, capsize=4, markersize=6.2)
    ax1.set_yticks(y_a)
    ax1.set_yticklabels([x[0] for x in panel_a_items], fontsize=8.8)
    ax1.invert_yaxis()
    ax1.set_xlabel("Log Price / Rent Change per +1 pp Immigrant Share (95% CI)", fontsize=9.3)
    ax1.set_title("Panel A: Housing Prices (Structure-Only vs. Full Hedonic)", fontweight="bold", fontsize=10.8, pad=10)

    panel_b_items = [
        ("CPUMA (07-21): Crowding (All HHs, PERNUM=1)", *get_row(m1, "CPUMA_2007_2021_Crowding_All"), NAVY),
        ("CPUMA (07-21): Crowding (Native-Headed HHs)", *get_row(m1, "CPUMA_2007_2021_Crowding_Native"), SLATE),
        ("CPUMA (07-21): 3+ Gen Share (MULTGEN=3)", *get_row(m1, "CPUMA_2007_2021_Multigen_All"), NAVY),
        ("CPUMA (07-21): Homeownership (All HHs)", *get_row(m1, "CPUMA_2007_2021_Homeownership_All"), NAVY),
        ("Co-Eth Adults (07-21): Crowding (PERWT)", *get_row(m1, "CoEth_CPUMA_2007_2021_Crowding"), AMBER),
        ("Co-Eth Adults (07-21): 3+ Gen (MULTGEN=3)", *get_row(m1, "CoEth_CPUMA_2007_2021_Multigen"), AMBER),
        ("Co-Eth Adults (07-21): Homeownership", *get_row(m1, "CoEth_CPUMA_2007_2021_Homeownership"), AMBER),
        ("PUMA (22-24): Crowding (All HHs, PERNUM=1)", *get_row(m1, "PUMA_2022_2024_Crowding"), CRIMSON),
        ("PUMA (22-24): 3+ Gen Share (MULTGEN=3)", *get_row(m1, "PUMA_2022_2024_Multigen"), CRIMSON),
        ("PUMA (22-24): Homeownership (All HHs)", *get_row(m1, "PUMA_2022_2024_Homeownership"), CRIMSON),
    ]

    y_b = np.arange(len(panel_b_items))
    ax2.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    for i, (lbl, c, s, col) in enumerate(panel_b_items):
        ax2.errorbar(c, i, xerr=1.96 * s, fmt="s", color=col, ecolor=col, elinewidth=2.0, capsize=4, markersize=5.8)
    ax2.set_yticks(y_b)
    ax2.set_yticklabels([x[0] for x in panel_b_items], fontsize=8.8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Effect per +1 pp Neighborhood Immigrant Share (95% CI)", fontsize=9.3)
    ax2.set_title("Panel B: Household Crowding and Co-Residence Margins", fontweight="bold", fontsize=10.8, pad=10)

    fig.suptitle(
        "Figure 1: Within-Metro Housing Prices (No Room Bad Controls) and Crowding (2007-2024)",
        fontweight="bold", fontsize=11.8, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(FIG / "fig1_cpuma_neighborhood_vs_msa.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# FIGURE 2: Descriptive Event Study by Current Occupational Teleworkability
# ===========================================================================
def make_fig2(es: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8))
    panels = [
        ("GatewayOutMig_TeleXGateway", "A. Gateway Out-Migration (Teleworkable x Top-8 Gateway)", "Annual Out-Migration Prob. (pp)", CRIMSON, 100.0),
        ("TrueRetention_TeleXShare", "B. True Origin-MSA Retention (Teleworkable x Enclave Share)", "Retention Prob. per +1 pp Share (pp)", NAVY, 100.0),
        ("LogWage_TeleXShare", "C. Log Real Hourly Wage (Teleworkable x Enclave Share)", "Log Wage Points per +1 pp Share", TEAL, 1.0),
        ("RentBurden_TeleXShare", "D. Rent Burden (Teleworkable x Enclave Share; all yrs p>=0.17)", "Rent/Income Ratio per +1 pp Share (pp)", AMBER, 100.0),
    ]

    for idx, (ax, (outc, title, ylabel, col, scale)) in enumerate(zip(axes.flat, panels)):
        sub = es[es["outcome"] == outc].sort_values("year")
        yrs = sub["year"].to_numpy()
        c = sub["coef"].to_numpy() * scale
        se = sub["se"].to_numpy() * scale

        ax.axvspan(2019.5, 2020.5, color="#cbd5e1", alpha=0.45)
        ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.errorbar(
            yrs, c, yerr=1.96 * se, fmt="-o", color=col, ecolor=col,
            elinewidth=2.0, capsize=4, markersize=6.2, linewidth=2.0,
        )
        ax.plot([2019], [0], marker="D", markersize=7.5, color="black", zorder=5)
        if idx == 0:
            ax.annotate("2018 pre: +0.25 pp\n(p=0.28)", xy=(2018, 0.249), xytext=(2015.4, 0.56),
                        fontsize=8, color=SLATE, arrowprops={"arrowstyle": "->", "color": SLATE, "lw": 0.9})
            ax.annotate("2021 spike:\n+0.53 pp (p=0.001)", xy=(2021, 0.530), xytext=(2021.45, 0.66),
                        fontsize=8, color=CRIMSON, fontweight="bold", arrowprops={"arrowstyle": "->", "color": CRIMSON, "lw": 0.9})
        ax.set_xticks([2015, 2018, 2019, 2021, 2022, 2023, 2024])
        ax.set_title(title, fontweight="bold", fontsize=10.2, pad=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("ACS Survey Year (Reference = 2019)", fontsize=9)

    fig.suptitle(
        "Figure 2: Descriptive Event Study by Current Occupational Teleworkability (2015-2024)",
        fontweight="bold", fontsize=12.0, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(FIG / "fig2_telework_event_study_2015_2024.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# FIGURE 3: Inter-MSA Mover Destinations — Honest Reporting of Null/Negative Tests
# ===========================================================================
def make_fig3(m3_reg: pd.DataFrame, m3_means: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.8))

    gw = m3_means[m3_means["sample"] == "GatewayExits"].copy()
    labels = ["Rent Savings\n(-Delta ln ZORI)", "Secondary\nSunbelt Share", "Arbitrage Share\n(Cheap, >=0.5pp)", "Abandon. Share\n(Cheap, <0.5pp)"]
    x = np.arange(len(labels))
    width = 0.20

    configs = [
        ("Pre (2015-2019)", "Non-Teleworkable", "#94a3b8", "Pre: Non-Teleworkable"),
        ("Post (2021-2024)", "Non-Teleworkable", "#475569", "Post: Non-Teleworkable"),
        ("Pre (2015-2019)", "Teleworkable", "#7dd3fc", "Pre: Teleworkable"),
        ("Post (2021-2024)", "Teleworkable", NAVY, "Post: Teleworkable"),
    ]
    for k, (per, grp, col, leg) in enumerate(configs):
        row = gw[(gw["period"] == per) & (gw["group"] == grp)].iloc[0]
        vals = [
            -row["mean_d_ln_zori"] * 100.0,
            row["share_sunbelt"] * 100.0,
            row["share_arbitrage"] * 100.0,
            row["share_abandonment"] * 100.0,
        ]
        ax1.bar(x + (k - 1.5) * width, vals, width, label=leg, color=col, edgecolor="white")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.8)
    ax1.set_ylabel("Percentage Points / Log Points (x100)", fontsize=9.3)
    ax1.set_title("Panel A: Top-8 Gateway Exits (Descriptive Means)", fontweight="bold", fontsize=10.6, pad=8)
    ax1.legend(frameon=False, fontsize=8.3, loc="upper left")

    def grab(spec: str) -> tuple[float, float, float]:
        r = m3_reg[m3_reg["spec"] == spec].iloc[0]
        return float(r["coef"]), float(r["se"]), float(r["wcb_pval"])

    b_items = [
        ("GW Exits: P(Sunbelt) [Tele x Post]", *grab("GatewayExits_ITT_TeleXPost_Prob_SunbeltSecondaryEnclave"), SLATE),
        ("GW Exits: P(Sunbelt) [WFH x Post]", *grab("GatewayExits_Realized_WFHXPost_Prob_SunbeltSecondaryEnclave"), CRIMSON),
        ("GW Exits: Delta ln(Co-Eth) [Tele x Post]", *grab("GatewayExits_ITT_TeleXPost_Delta_ln_CoEthShare"), SLATE),
        ("GW Exits: Delta ln(Co-Eth) [WFH x Post]", *grab("GatewayExits_Realized_WFHXPost_Delta_ln_CoEthShare"), NAVY),
        ("High-Encl GW: P(Arbitrage) [Tele x Post]", *grab("HighEnclaveGatewayExits_ITT_TeleXPost_Prob_EnclaveArbitrage"), SLATE),
        ("High-Encl GW: P(Arbitrage) [WFH x Post]", *grab("HighEnclaveGatewayExits_Realized_WFHXPost_Prob_EnclaveArbitrage"), CRIMSON),
        ("High-Encl GW: P(Abandon.) [Tele x Post]", *grab("HighEnclaveGatewayExits_ITT_TeleXPost_Prob_EnclaveAbandonment"), SLATE),
        ("High-Encl GW: P(Abandon.) [WFH x Post]", *grab("HighEnclaveGatewayExits_Realized_WFHXPost_Prob_EnclaveAbandonment"), AMBER),
    ]
    yb = np.arange(len(b_items))
    ax2.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    for i, (lbl, c, s, pwcb, col) in enumerate(b_items):
        ax2.errorbar(c, i, xerr=1.96 * s, fmt="o", color=col, ecolor=col, elinewidth=2.0, capsize=4, markersize=6.0)
        ax2.text(c + 1.96 * s + 0.008, i, f"p_WCB={pwcb:.3f}", va="center", fontsize=7.8, color=SLATE)
    ax2.set_yticks(yb)
    ax2.set_yticklabels([z[0] for z in b_items], fontsize=8.5)
    ax2.invert_yaxis()
    xlim = ax2.get_xlim()
    ax2.set_xlim(xlim[0] - 0.01, xlim[1] + 0.075)
    ax2.set_xlabel("Post-2020 DiD Estimate (Asymptotic 95% CI & Webb WCB p-value, G=8)", fontsize=9.0)
    ax2.set_title("Panel B: Gateway-Exit Destination Tests (Null / Negative)", fontweight="bold", fontsize=10.6, pad=8)

    fig.suptitle(
        "Figure 3: Inter-MSA Mover Destinations (2015-2024) - No Systematic Sunbelt or Enclave Arbitrage",
        fontweight="bold", fontsize=11.6, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(FIG / "fig3_enclave_arbitrage_vs_abandonment.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# FIGURE 4: Uniform 10-Yr Decadal JRS (1980-2010) vs. Separate ACS (2010-2024)
# ===========================================================================
def make_fig4(m4: pd.DataFrame, contours: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.8))

    order = [
        ("Decadal_1980_2010_10yr", "Static_Bartik_IV", "ln_Residents", "cp_flow", "Decadal 1980-10 Static: ln(Res)"),
        ("Decadal_1980_2010_10yr", "Static_Bartik_IV", "ln_OOS_Residents", "cp_flow", "Decadal 1980-10 Static: ln(OOS Res)"),
        ("ACS_2010_2019_2024", "Static_Bartik_IV", "ln_Residents", "cp_flow", "ACS 2010-24 Static: ln(Res)"),
        ("ACS_2010_2019_2024", "Static_Bartik_IV", "ln_OOS_Residents", "cp_flow", "ACS 2010-24 Static: ln(OOS Res)"),
        ("Decadal_1980_2010_10yr", "JRS_Dynamic_2IV", "ln_Residents", "cp_flow", "Decadal JRS 2-IV: ln(Res) Contemp."),
        ("Decadal_1980_2010_10yr", "JRS_Dynamic_2IV", "ln_Residents", "cp_flow_lag", "Decadal JRS 2-IV: ln(Res) Lagged"),
        ("Decadal_1980_2010_10yr", "JRS_Dynamic_2IV", "ln_OOS_Residents", "cp_flow", "Decadal JRS 2-IV: ln(OOS) Contemp."),
        ("Decadal_1980_2010_10yr", "JRS_Dynamic_2IV", "ln_OOS_Residents", "cp_flow_lag", "Decadal JRS 2-IV: ln(OOS) Lagged"),
    ]

    ax1.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    for i, (pan, mod, outc, var, lbl) in enumerate(order):
        r = m4[(m4["panel"] == pan) & (m4["model"] == mod) & (m4["outcome"] == outc) & (m4["variable"] == var)].iloc[0]
        ax1.plot([r["ar_ci_lo"], r["ar_ci_hi"]], [i + 0.14, i + 0.14], color=TEAL, linewidth=3.4, alpha=0.85,
                 label="Anderson-Rubin 95% CI (Weak-IV Robust)" if i == 0 else "")
        ax1.errorbar(
            r["coef"], i - 0.12, xerr=1.96 * r["se_2way"], fmt="o", color=NAVY,
            ecolor=NAVY, elinewidth=1.8, capsize=3.5, markersize=5.5,
            label="2SLS Two-Way Clustered 95% Wald CI" if i == 0 else "",
        )
        sw_txt = f"SW F={r['sw_f_2way']:.1f}"
        ax1.text(max(r["ar_ci_hi"], r["wald_ci_hi"]) + 0.006, i, sw_txt, va="center", fontsize=8, color=SLATE)

    ax1.set_yticks(np.arange(len(order)))
    ax1.set_yticklabels([x[4] for x in order], fontsize=8.6)
    ax1.set_ylim(-0.6, len(order) + 0.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("2SLS Elasticity per +1 pp Co-Ethnic Inflow", fontsize=9.3)
    ax1.set_title("Panel A: Uniform 10-Yr Decadal & Separate ACS Estimates", fontweight="bold", fontsize=10.5, pad=8)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.0, loc="lower left")
    xlim = ax1.get_xlim()
    ax1.set_xlim(xlim[0] - 0.015, xlim[1] + 0.065)

    # Panel B: Exact 2D Joint Anderson-Rubin 95% Confidence Regions (Uniform 10-Yr Decadal Panel)
    ax2.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.6)
    ax2.axvline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.6)
    plot_specs = [
        ("ln_OOS_Residents", "ln(OOS Residents) [Decadal 2-IV]", TEAL),
        ("ln_Residents", "ln(Residents) [Decadal 2-IV]", NAVY),
        ("LogWage", "Log Hourly Wage [Decadal 2-IV]", CRIMSON),
    ]
    for key, lbl, col in plot_specs:
        cinfo = contours.get(key, {})
        pts = cinfo.get("ar_contour", [])
        b = cinfo.get("beta", {})
        if pts:
            poly = Polygon(pts, closed=True, facecolor=col, edgecolor=col, alpha=0.20, linewidth=2.0, label=lbl)
            ax2.add_patch(poly)
            ax2.plot([b["cp_flow"]], [b["cp_flow_lag"]], marker="o", color=col, markersize=6.5)

    ax2.plot([0], [0], marker="x", color="black", markersize=9, markeredgewidth=2, label="Null Hypothesis (0, 0)")
    ax2.autoscale_view()
    ax2.set_xlabel("Contemporaneous 10-Yr Inflow ($\\beta_{\\mathrm{contemp}}$)", fontsize=9.3)
    ax2.set_ylabel("Lagged 10-Yr Inflow ($\\beta_{\\mathrm{lag}}$)", fontsize=9.3)
    ax2.set_title("Panel B: Joint 2D AR 95% Sets (Uniform 10-Yr Panel, 2000-2010)", fontweight="bold", fontsize=10.5, pad=8)
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left")

    fig.suptitle(
        "Figure 4: Weak-IV-Robust Shift-Share Inference (Uniform 10-Yr Decadal vs. Separate ACS Panel)",
        fontweight="bold", fontsize=11.6, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(FIG / "fig4_jrs_weak_iv_robust.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# LATEX TABLES (booktabs, 100% pure ASCII, zero em-dashes, compact)
# ===========================================================================
def write_tables(
    m1: pd.DataFrame,
    nickell: pd.DataFrame,
    m2_did: pd.DataFrame,
    m3_reg: pd.DataFrame,
    m3_sens: pd.DataFrame,
    m4: pd.DataFrame,
) -> None:
    # --- Table 1: Sub-Metro Housing (Structure-Only vs. Full Hedonic), Crowding, and Nickell Demo ---
    def r1(spec: str, label: str, fe_txt: str) -> str:
        r = m1[m1["spec"] == spec].iloc[0]
        return (
            f"{label} & {r['coef']:+.4f}{stars(r['pval'])} & ({r['se']:.4f}) & "
            f"{r['tstat']:+.2f} & {int(r['n']):,} & {fe_txt} \\\\"
        )

    def rnick(spec: str, label: str) -> str:
        r = nickell[nickell["spec"] == spec].iloc[0]
        if pd.notna(r.get("implied_rho")):
            if r["estimator"] == "Song_WithLag_FE":
                lag_txt = f"$\\hat{{\\rho}}_{{\\mathrm{{FE}}}}={r['implied_rho']:+.2f}$ ($\\hat{{\\gamma}}={r['lag_coef']:+.2f}$)"
            else:
                lag_txt = f"$\\hat{{\\rho}}_{{\\mathrm{{AH}}}}={r['implied_rho']:+.2f}$ ($F_1={r['first_stage_f']:.0f}$)"
        else:
            lag_txt = "No dynamic lag"
        return (
            f"{label} & {r['coef']:+.4f}{stars(r['pval'])} & ({r['se']:.4f}) & "
            f"{r['tstat']:+.2f} & {int(r['n']):,} & {lag_txt} \\\\"
        )

    t1 = r"""\begin{table}[t!]
\centering
\caption{Within-Metro Housing Prices, Crowding, 3+ Generation Living, and Dynamic-Panel Nickell (1981) Bias}
\label{tab:cpuma_housing}
\scriptsize
\setlength{\tabcolsep}{4.0pt}
\renewcommand{\arraystretch}{1.04}
\begin{tabular}{lccccc}
\toprule
Outcome Specification & Coef. & (SE) & $t$-stat & $N$ & FE / Implied $\hat{\rho}$ \\
\midrule
\multicolumn{6}{l}{\textit{Panel A: Within-MSA \texttt{CPUMA0010} Unique Households (\texttt{PERNUM}=1, \texttt{HHWT}-Weighted, 2007--2021)}} \\
""" + "\n".join([
        r1("CPUMA_2007_2021_lnRent_Unadj", "Log Gross Rent (Unadjusted)", "MSA$\\times$Yr + CPUMA"),
        r1("CPUMA_2007_2021_lnRent_StructOnly", "Log Gross Rent (Structure-Only, No Rooms)", "MSA$\\times$Yr + CPUMA"),
        r1("CPUMA_2007_2021_lnRent_Hedonic", "Log Gross Rent (Full Hedonic)", "MSA$\\times$Yr + CPUMA"),
        r1("CPUMA_2007_2021_lnVal_StructOnly", "Log Home Value (Structure-Only, No Rooms)", "MSA$\\times$Yr + CPUMA"),
        r1("CPUMA_2007_2021_lnVal_Hedonic", "Log Home Value (Full Hedonic)", "MSA$\\times$Yr + CPUMA"),
        r1("CPUMA_2007_2021_Crowding_All", "Household Crowding (Persons/Room, All HHs)", "MSA$\\times$Yr + CPUMA"),
        r1("CPUMA_2007_2021_Multigen_All", "3+ Generation Households (\\texttt{MULTGEN}=3)", "MSA$\\times$Yr + CPUMA"),
    ]) + r"""
\midrule
\multicolumn{6}{l}{\textit{Panel B: U.S.-Born Co-Ethnic Adult Housing Experience in \texttt{CPUMA0010} (\texttt{PERWT}-Weighted, 2007--2021)}} \\
""" + "\n".join([
        r1("CoEth_CPUMA_2007_2021_lnRent_StructOnly", "Co-Ethnic Adult Log Rent (Structure-Only)", "3-Way FE"),
        r1("CoEth_CPUMA_2007_2021_lnRent_Hedonic", "Co-Ethnic Adult Log Rent (Full Hedonic)", "3-Way FE"),
        r1("CoEth_CPUMA_2007_2021_lnVal_StructOnly", "Co-Ethnic Adult Log Value (Structure-Only)", "3-Way FE"),
        r1("CoEth_CPUMA_2007_2021_Crowding", "Co-Ethnic Adult Crowding (Persons/Room)", "3-Way FE"),
        r1("CoEth_CPUMA_2007_2021_Multigen", "Co-Ethnic Adult 3+ Gen Living (\\texttt{MULTGEN}=3)", "3-Way FE"),
    ]) + r"""
\midrule
\multicolumn{6}{l}{\textit{Panel C: Post-COVID 2020-Census \texttt{PUMA} Unique Households (\texttt{PERNUM}=1, \texttt{HHWT}-Weighted, 2022--2024)}} \\
""" + "\n".join([
        r1("PUMA_2022_2024_lnRent_StructOnly", "Log Gross Rent (Structure-Only, No Rooms)", "MSA$\\times$Yr + PUMA"),
        r1("PUMA_2022_2024_lnVal_StructOnly", "Log Home Value (Structure-Only, No Rooms)", "MSA$\\times$Yr + PUMA"),
        r1("PUMA_2022_2024_Crowding", "Household Crowding (Persons/Room, All HHs)", "MSA$\\times$Yr + PUMA"),
        r1("PUMA_2022_2024_Multigen", "3+ Generation Households (\\texttt{MULTGEN}=3)", "MSA$\\times$Yr + PUMA"),
    ]) + r"""
\midrule
\multicolumn{6}{l}{\textit{Panel D: Common-Sample Nickell (1981) Bias Check on Song's Table 7 (1980--2024 Cell Panel)}} \\
""" + "\n".join([
        rnick("NickellDemo_WithLag_FE_Rent", "$\\Delta \\ln(\\text{Rent})$: Dynamic FE ($hp_{t-1}$ + 3-Way FE)"),
        rnick("NickellDemo_NoLag_FE_Rent", "$\\Delta \\ln(\\text{Rent})$: Static FD (No Lag, 3-Way FE)"),
        rnick("NickellDemo_AndersonHsiao_IV_Rent", "$\\Delta \\ln(\\text{Rent})$: Anderson--Hsiao IV ($hp_{t-2}$)"),
        rnick("NickellDemo_WithLag_FE_HouseValue", "$\\Delta \\ln(\\text{Value})$: Dynamic FE ($hp_{t-1}$ + 3-Way FE)"),
        rnick("NickellDemo_NoLag_FE_HouseValue", "$\\Delta \\ln(\\text{Value})$: Static FD (No Lag, 3-Way FE)"),
        rnick("NickellDemo_AndersonHsiao_IV_HouseValue", "$\\Delta \\ln(\\text{Value})$: Anderson--Hsiao IV ($hp_{t-2}$)"),
    ]) + r"""
\bottomrule
\end{tabular}
\begin{minipage}{0.96\linewidth}
\footnotesize\textit{Notes:} IPUMS ACS microdata (working-age 18--64, non-school, non-GQ), clustered by MSA. Panels A and C aggregate unique households (\texttt{PERNUM}=1, weighted by \texttt{HHWT}) and define three-or-more-generation households strictly as \texttt{MULTGEN}=3. Panel B averages housing conditions experienced by U.S.-born co-ethnic working-age adults using person weights (\texttt{PERWT}), absorbing Eth$\times$Yr, MSA$\times$Yr, and Eth$\times$CPUMA FEs. Structure-Only omits endogenous room/bedroom counts and controls only for \texttt{BUILTYR2}, \texttt{UNITSSTR}, homeownership, age, and education; Full Hedonic adds mean rooms and bedrooms. Panel D estimates $\Delta hp_{rct}$ on $\Delta \text{Share}_{rct}$ on the exact same non-singleton common cell sample with identical differenced controls ($\Delta\text{Educ}$, $\Delta\ln\text{Wage}$), comparing implied persistence $\hat{\rho}_{\text{FE}} = 1 + \hat{\gamma}$ under Dynamic FE against $\hat{\rho}_{\text{AH}}$ under Anderson--Hsiao IV. $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.
\end{minipage}
\end{table}
"""
    (TAB / "tab1_cpuma_housing_crowding.tex").write_text(t1)

    # --- Table 2: Descriptive Teleworkability DiD + Pre-2019 Origin-Metro Exposure Check ---
    def r2(spec: str, label: str, param: str) -> str:
        r = m2_did[m2_did["spec"] == spec].iloc[0]
        return (
            f"{label} & {param} & {r['coef']:+.5f}{stars(r['pval'])} & ({r['se']:.5f}) & "
            f"{r['tstat']:+.2f} & {int(r['raw_n']):,} \\\\"
        )

    t2 = r"""\begin{table}[t!]
\centering
\caption{Post-2020 Mobility, Wages, and Rent Burden by Occupational Teleworkability and Pre-2019 Origin-Metro Exposure (2015--2024)}
\label{tab:telework_did}
\scriptsize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.05}
\begin{tabular}{llcccc}
\toprule
Outcome & Interaction Term & Coef. & (SE) & $t$-stat & Micro $N$ \\
\midrule
\multicolumn{6}{l}{\textit{Panel A: Descriptive Associations by Current Occupational Teleworkability (\texttt{OCCSOC})}} \\
""" + "\n".join([
        r2("FirstStage_RealizedWFH_TeleXPost", "Realized WFH (\\texttt{TRANWORK}=80)", "$\\text{Tele} \\times \\text{Post2020}$"),
        r2("ITT_DiD_GatewayOutMig_TeleXPostXGateway", "Gateway Out-Migration Rate", "$\\text{Tele} \\times \\text{Post} \\times \\text{Top8Gateway}$"),
        r2("ITT_DiD_TrueRetention_AllMSAs", "True Origin-MSA Retention", "$\\text{Tele} \\times \\text{Post} \\times \\text{PreEnclaveShare}$"),
        r2("ITT_DiD_WithinGateways_OutMig_TeleXPostXShare", "Within-Gateway Out-Migration", "$\\text{Tele} \\times \\text{Post} \\times \\text{PreEnclaveShare}$"),
        r2("ITT_DiD_LogWage_TeleXPostXShare", "Log Real Hourly Wage", "$\\text{Tele} \\times \\text{Post} \\times \\text{PreEnclaveShare}$"),
        r2("ITT_DiD_RentBurden_TeleXPostXShare", "Rent Burden (Rent/Wage Income)", "$\\text{Tele} \\times \\text{Post} \\times \\text{PreEnclaveShare}$"),
    ]) + r"""
\midrule
\multicolumn{6}{l}{\textit{Panel B: Genuinely Pre-Determined Origin-Metro Pre-2019 Telework Exposure}} \\
""" + "\n".join([
        r2("Pre2019_OrigMetroTele_Post_OutMig", "All-MSA Out-Migration Rate", "$\\text{OrigMetroTele}_{15-19} \\times \\text{Post}$"),
        r2("Pre2019_OrigMetroTele_PostXGateway_OutMig", "Gateway Out-Migration (Metro Exp.)", "$\\text{OrigMetroTele}_{15-19} \\times \\text{Post} \\times \\text{GW}$"),
        r2("Pre2019_OrigEthMetroTele_PostXGateway_OutMig", "Gateway Out-Migration (Eth-Metro Exp.)", "$\\text{OrigEthTele}_{15-19} \\times \\text{Post} \\times \\text{GW}$"),
        r2("Pre2019_OrigEthMetroTele_PostXShare_Retention", "True Retention (Eth-Metro Exp.)", "$\\text{OrigEthTele}_{15-19} \\times \\text{Post} \\times \\text{Share}$"),
    ]) + r"""
\bottomrule
\end{tabular}
\begin{minipage}{0.96\linewidth}
\footnotesize\textit{Notes:} IPUMS ACS 2015, 2018, 2019 (Pre) vs.\ 2021--2024 (Post), U.S.-born co-ethnic working-age adults. Panel A classifies workers by current-year \texttt{OCCSOC} (observed after any 1-year move; hence descriptive, not causal), absorbing Eth$\times$Yr, MSA$\times$Yr, Eth$\times$MSA, and Tele$\times$Yr FEs. Panel B uses genuinely pre-determined 2015--2019 origin-metro (or origin-metro $\times$ ethnicity) teleworkable employment shares. SEs clustered by origin/destination MSA. $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.
\end{minipage}
\end{table}
"""
    (TAB / "tab2_telework_did_retention.tex").write_text(t2)

    # --- Table 3: Inter-MSA Mover Destinations with Webb 6-Point Wild Cluster Bootstrap p-values ---
    def r3(spec_itt: str, spec_wfh: str, label: str) -> str:
        ri = m3_reg[m3_reg["spec"] == spec_itt].iloc[0]
        rw = m3_reg[m3_reg["spec"] == spec_wfh].iloc[0]
        return (
            f"{label} & {ri['coef']:+.4f}{stars(ri['pval'])} & ({ri['se']:.4f}) & [{ri['wcb_pval']:.3f}] & "
            f"{rw['coef']:+.4f}{stars(rw['pval'])} & ({rw['se']:.4f}) & [{rw['wcb_pval']:.3f}] & {int(ri['n']):,} \\\\"
        )

    t3 = r"""\begin{table}[t!]
\centering
\caption{Inter-MSA Mover Destinations (\texttt{MIGMET131}, 2015--2024) with Webb 6-Point Wild Cluster Bootstrap $p$-Values}
\label{tab:enclave_arbitrage}
\scriptsize
\renewcommand{\arraystretch}{1.05}
\begin{tabular}{lccccccc}
\toprule
 & \multicolumn{3}{c}{$\text{Teleworkable} \times \text{Post}$} & \multicolumn{3}{c}{Realized $\text{WFH} \times \text{Post}$} & \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Destination Outcome ($\text{Dest} - \text{Orig}$) & Coef. & (SE) & [$p_{\text{WCB}}$] & Coef. & (SE) & [$p_{\text{WCB}}$] & $N$ \\
\midrule
\multicolumn{8}{l}{\textit{Panel A: All Inter-MSA Co-Ethnic Native Movers ($G = 259$--$266$ Origin MSAs)}} \\
""" + "\n".join([
        r3("AllMovers_ITT_TeleXPost_Delta_ln_ZORI", "AllMovers_Realized_WFHXPost_Delta_ln_ZORI", "$\\Delta \\ln(\\text{ZORI Rent})$"),
        r3("AllMovers_ITT_TeleXPost_Delta_ln_CoEthShare", "AllMovers_Realized_WFHXPost_Delta_ln_CoEthShare", "$\\Delta \\ln(\\text{Co-Ethnic Share})$"),
        r3("AllMovers_ITT_TeleXPost_Prob_SunbeltSecondaryEnclave", "AllMovers_Realized_WFHXPost_Prob_SunbeltSecondaryEnclave", "$P(\\text{Sunbelt Hub})$"),
        r3("AllMovers_ITT_TeleXPost_Prob_EnclaveArbitrage", "AllMovers_Realized_WFHXPost_Prob_EnclaveArbitrage", "$P(\\text{Enclave Arbitrage})$"),
        r3("AllMovers_ITT_TeleXPost_Prob_EnclaveAbandonment", "AllMovers_Realized_WFHXPost_Prob_EnclaveAbandonment", "$P(\\text{Enclave Abandonment})$"),
    ]) + r"""
\midrule
\multicolumn{8}{l}{\textit{Panel B: Movers Exiting Top-8 High-Cost Gateway MSAs ($G = 8$ Origin Clusters)}} \\
""" + "\n".join([
        r3("GatewayExits_ITT_TeleXPost_Delta_ln_ZORI", "GatewayExits_Realized_WFHXPost_Delta_ln_ZORI", "$\\Delta \\ln(\\text{ZORI Rent})$"),
        r3("GatewayExits_ITT_TeleXPost_Delta_CoEthShare_pp", "GatewayExits_Realized_WFHXPost_Delta_CoEthShare_pp", "$\\Delta \\text{Co-Ethnic Share}$ (pp)"),
        r3("GatewayExits_ITT_TeleXPost_Delta_ln_CoEthShare", "GatewayExits_Realized_WFHXPost_Delta_ln_CoEthShare", "$\\Delta \\ln(\\text{Co-Ethnic Share})$"),
        r3("GatewayExits_ITT_TeleXPost_Prob_SunbeltSecondaryEnclave", "GatewayExits_Realized_WFHXPost_Prob_SunbeltSecondaryEnclave", "$P(\\text{Sunbelt Hub})$"),
        r3("GatewayExits_ITT_TeleXPost_Prob_EnclaveArbitrage", "GatewayExits_Realized_WFHXPost_Prob_EnclaveArbitrage", "$P(\\text{Enclave Arbitrage})$"),
        r3("GatewayExits_ITT_TeleXPost_Prob_EnclaveAbandonment", "GatewayExits_Realized_WFHXPost_Prob_EnclaveAbandonment", "$P(\\text{Enclave Abandonment})$"),
    ]) + r"""
\midrule
\multicolumn{8}{l}{\textit{Panel C: High-Enclave Gateway Exits (Origin Co-Ethnic Share $\ge 1.0$ pp; $G = 8$ Clusters)}} \\
""" + "\n".join([
        r3("HighEnclaveGatewayExits_ITT_TeleXPost_Delta_CoEthShare_pp", "HighEnclaveGatewayExits_Realized_WFHXPost_Delta_CoEthShare_pp", "$\\Delta \\text{Co-Ethnic Share}$ (pp)"),
        r3("HighEnclaveGatewayExits_ITT_TeleXPost_Prob_SunbeltSecondaryEnclave", "HighEnclaveGatewayExits_Realized_WFHXPost_Prob_SunbeltSecondaryEnclave", "$P(\\text{Sunbelt Hub})$"),
        r3("HighEnclaveGatewayExits_ITT_TeleXPost_Prob_EnclaveArbitrage", "HighEnclaveGatewayExits_Realized_WFHXPost_Prob_EnclaveArbitrage", "$P(\\text{Enclave Arbitrage})$"),
        r3("HighEnclaveGatewayExits_ITT_TeleXPost_Prob_EnclaveAbandonment", "HighEnclaveGatewayExits_Realized_WFHXPost_Prob_EnclaveAbandonment", "$P(\\text{Enclave Abandonment})$"),
    ]) + r"""
\bottomrule
\end{tabular}
\begin{minipage}{0.98\linewidth}
\footnotesize\textit{Notes:} Inter-MSA native co-ethnic movers (\texttt{MIGMET131} $\ne$ \texttt{MET2013}, 2015--2024), controlling for Eth$\times$Yr and Origin-MSA FEs, age, education, and sex, weighted by \texttt{PERWT}. Standard errors clustered by Origin MSA; brackets report two-sided restricted wild cluster bootstrap $p$-values using Webb (2023) 6-point weights ($9{,}999$ draws; Cameron, Gelbach, and Miller 2008). Stars reflect asymptotic clustered $p$-values ($^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$). Under Webb WCB, high-enclave gateway abandonment ($-0.0494$) rises to $p_{\text{WCB}} = 0.102$, and both Sunbelt and enclave-arbitrage estimates are negative or null.
\end{minipage}
\end{table}
"""
    (TAB / "tab3_enclave_arbitrage_movers.tex").write_text(t3)

    # --- Table 3B: Pre-Specified Threshold & Gateway Sensitivity Grid (Webb WCB) ---
    sens_rows_tex = []
    for axis, cfg, label in [
        ("GatewaySet", "Top5", "Gateways: Top 5 ($G=5$)"),
        ("GatewaySet", "Top8 (Baseline)", "Gateways: Top 8 (Baseline, $G=8$)"),
        ("GatewaySet", "Top12", "Gateways: Top 12 ($G=12$)"),
        ("RentThreshold", "5% (d_ln_zori < -0.05)", "Rent Cutoff: $\\ge 5\\%$ Cheaper"),
        ("RentThreshold", "10% (Baseline)", "Rent Cutoff: $\\ge 10\\%$ (Baseline)"),
        ("RentThreshold", "15% (d_ln_zori < -0.15)", "Rent Cutoff: $\\ge 15\\%$ Cheaper"),
        ("CoEthThreshold", "0.25 pp", "Co-Ethnic Cutoff: $\\ge 0.25$ pp"),
        ("CoEthThreshold", "0.50 pp (Baseline)", "Co-Ethnic Cutoff: $\\ge 0.50$ pp (Base)"),
        ("CoEthThreshold", "1.00 pp", "Co-Ethnic Cutoff: $\\ge 1.00$ pp"),
    ]:
        sub = m3_sens[(m3_sens["axis"] == axis) & (m3_sens["config"] == cfg) & (m3_sens["sample"] == "HighEnclaveGatewayExits")]
        a_t = sub[(sub["treatment"] == "Tele_x_Post") & (sub["outcome"] == "Prob_EnclaveArbitrage")].iloc[0]
        a_w = sub[(sub["treatment"] == "WFH_x_Post") & (sub["outcome"] == "Prob_EnclaveArbitrage")].iloc[0]
        b_t = sub[(sub["treatment"] == "Tele_x_Post") & (sub["outcome"] == "Prob_EnclaveAbandonment")].iloc[0]
        b_w = sub[(sub["treatment"] == "WFH_x_Post") & (sub["outcome"] == "Prob_EnclaveAbandonment")].iloc[0]
        sens_rows_tex.append(
            f"{label} & "
            f"{a_t['coef']:+.4f} [{a_t['wcb_pval']:.2f}] & {a_w['coef']:+.4f} [{a_w['wcb_pval']:.2f}] & "
            f"{b_t['coef']:+.4f} [{b_t['wcb_pval']:.2f}] & {b_w['coef']:+.4f} [{b_w['wcb_pval']:.2f}] \\\\"
        )

    t3b = r"""\begin{table}[t!]
\centering
\caption{Pre-Specified Sensitivity Grid for High-Enclave Gateway Exits (Coefficients and Webb WCB $p$-Values)}
\label{tab:sensitivity_grid}
\footnotesize
\renewcommand{\arraystretch}{1.05}
\begin{tabular}{lcccc}
\toprule
 & \multicolumn{2}{c}{$P(\text{Enclave Arbitrage})$} & \multicolumn{2}{c}{$P(\text{Enclave Abandonment})$} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Sensitivity Specification & Tele $\times$ Post & WFH $\times$ Post & Tele $\times$ Post & WFH $\times$ Post \\
\midrule
""" + "\n".join(sens_rows_tex[:3]) + "\n\\midrule\n" + "\n".join(sens_rows_tex[3:6]) + "\n\\midrule\n" + "\n".join(sens_rows_tex[6:]) + r"""
\bottomrule
\end{tabular}
\begin{minipage}{0.96\linewidth}
\footnotesize\textit{Notes:} Each cell reports the DiD coefficient followed by its Webb 6-point wild cluster bootstrap $p$-value ($9{,}999$ draws) in brackets for high-enclave gateway exits (origin co-ethnic share $\ge 1.0$ pp). Top 5: NYC, LA, SF, Chicago, Miami. Top 8 adds DC, Boston, Seattle. Top 12 adds Houston, Dallas--Fort Worth, San Diego, San Jose. Across all nine configurations, no arbitrage estimate is positive and significant, and only Top-12 WFH abandonment reaches $p_{\text{WCB}} = 0.056$.
\end{minipage}
\end{table}
"""
    (TAB / "tab3b_sensitivity_grid.tex").write_text(t3b)

    # --- Table 4: Uniform 10-Yr Decadal JRS (1980-2010) vs. Separate ACS Panel (2010-2024) ---
    out_map = {
        "ln_Residents": "$\\ln(\\text{Residents})$",
        "ln_OOS_Residents": "$\\ln(\\text{OOS Residents})$",
        "TrueRetention_5yr": "True Retention (5-Yr)",
        "TrueRetention_1yr": "True Retention (1-Yr)",
        "LogWage": "$\\ln(\\text{Hourly Wage})$",
    }

    def fmt_t4_rows(sub_df: pd.DataFrame) -> list[str]:
        lines = []
        for _, r in sub_df.iterrows():
            mod = "Static 1-IV" if r["model"] == "Static_Bartik_IV" else "JRS 2-IV"
            var = "Contemp." if r["variable"] == "cp_flow" else "Lagged"
            ar_txt = str(r["ar_ci_tex"])
            lines.append(
                f"{out_map.get(r['outcome'], r['outcome'])} & {mod} & {var} & "
                f"{r['coef']:+.4f}{stars(r['pval_2way'])} & ({r['se_2way']:.4f}) & "
                f"{r['sw_f_2way']:.1f} & {ar_txt} & {int(r['n']):,} \\\\"
            )
        return lines

    dec_rows = fmt_t4_rows(m4[m4["panel"].isin(["Decadal_1980_2010_10yr", "Census5yr_1980_2000"])])
    acs_rows = fmt_t4_rows(m4[m4["panel"] == "ACS_2010_2019_2024"])

    t4 = r"""\begin{table}[t!]
\centering
\caption{Weak-IV-Robust Shift-Share Inference: Uniform 10-Year Decadal Panel vs.\ Separate ACS Extension}
\label{tab:jrs_weak_iv}
\scriptsize
\setlength{\tabcolsep}{3.1pt}
\renewcommand{\arraystretch}{1.05}
\begin{tabular}{lllccccc}
\toprule
Outcome & Model & Inflow & Coef. & (2-Way SE) & SW $F$ & Exact AR 95\% Set & $N$ \\
\midrule
\multicolumn{8}{l}{\textit{Panel A: Uniform 10-Year Decadal Panel (1980, 1990, 2000, 2010; Exact 10-Yr Lags; 5-Yr Retention 1980--2000)}} \\
""" + "\n".join(dec_rows) + r"""
\midrule
\multicolumn{8}{l}{\textit{Panel B: Modern ACS Extension Panel (2010, 2019, 2024; Uniform 1-Year ACS Migration Window)}} \\
""" + "\n".join(acs_rows) + r"""
\bottomrule
\end{tabular}
\begin{minipage}{0.98\linewidth}
\footnotesize\textit{Notes:} Balanced (\texttt{eth}, \texttt{met2013}, \texttt{year}) panels with Eth$\times$Yr, MSA$\times$Yr, and Eth$\times$MSA FEs absorbed via weighted FWL projections. Panel A uses exact 10-year decadal spacing (Static 1-IV on 1990--2010; JRS 2-IV on 2000--2010) and restricts 5-year retention strictly to 1980--2000 Census waves. Panel B reports the 2010--2019--2024 ACS extension separately (uniform 1-year migration window, Static 1-IV on 2019--2024). Two-way clustered SEs (MSA $\times$ Ethnicity) and Sanderson--Windmeijer (2016) conditional $F$-statistics. All 1-IV Anderson--Rubin (1949) 95\% confidence sets are solved analytically in closed form without finite grid clipping, yielding unbounded unions or $(-\infty, +\infty)$ whenever SW $F < 3.84$; all 2-IV projected sets are verified strictly interior to their 2D search grids. $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.
\end{minipage}
\end{table}
"""
    (TAB / "tab4_jrs_sw_ar_inference.tex").write_text(t4)


def main() -> None:
    print("[06] Generating 4 updated PNG figures and 5 LaTeX booktabs tables...")
    m1 = pd.read_csv(RES / "module1_cpuma_housing_crowding.csv")
    nickell = pd.read_csv(RES / "module1_nickell_demo.csv")
    msa_saiz = pd.read_csv(ROOT / "results" / "essay3" / "table2_msa_housing_saiz.csv")
    m2_did = pd.read_csv(RES / "module2_telework_did.csv")
    m2_es = pd.read_csv(RES / "module2_telework_event_study.csv")
    m3_reg = pd.read_csv(RES / "module3_enclave_arbitrage_regs.csv")
    m3_means = pd.read_csv(RES / "module3_enclave_arbitrage_means.csv")
    m3_sens = pd.read_csv(RES / "module3_sensitivity_grid.csv")
    m4 = pd.read_csv(RES / "module4_jrs_weak_iv_robust.csv")
    contours = json.loads((RES / "module4_ar_contours.json").read_text())

    make_fig1(m1, msa_saiz)
    make_fig2(m2_es)
    make_fig3(m3_reg, m3_means)
    make_fig4(m4, contours)
    write_tables(m1, nickell, m2_did, m3_reg, m3_sens, m4)
    print("   All 4 PNG figures and 5 LaTeX tables written cleanly!")


if __name__ == "__main__":
    main()
