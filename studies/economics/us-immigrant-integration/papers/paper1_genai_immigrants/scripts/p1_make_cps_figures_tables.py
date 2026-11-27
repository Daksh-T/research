#!/usr/bin/env python3
"""
Paper 1 — Generate Publication-Ready CPS & HonestDiD Figures (Fig 1 & Fig 4)
          and LaTeX Tables (Tab 2 & Tab 4)
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
P1_DIR = ROOT / "papers" / "paper1_genai_immigrants"
RES_DIR = P1_DIR / "results"
FIG_DIR = P1_DIR / "figures"
TAB_DIR = P1_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
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


def fmt_coef(c: float, p: float) -> str:
    return f"{c:.4f}{stars(p)}"


def fmt_se(s: float) -> str:
    return f"({s:.4f})"


def make_fig1_cps_quarterly(es_df: pd.DataFrame) -> None:
    """
    Fig 1: High-Frequency Quarterly CPS ORG Event Study (2020 Q1 – 2026 Q3, ref = 2022 Q3)
      Panel A: Non-English-Origin Immigrants vs. Natives — Overall LLM Beta vs. Orthogonalized LLM Writing
      Panel B: Non-English-Origin vs. English-Origin Placebo (Overall LLM Beta)
    """
    qtrs = sorted(es_df["yq_int"].unique().tolist())
    x_map = {q: i for i, q in enumerate(qtrs)}
    q_labels = [f"{q // 10}Q{q % 10}" for q in qtrs]
    ref_x = x_map[20223]
    launch_x = x_map[20223] + 0.5  # Between 2022Q3 and 2022Q4 (Nov 30, 2022)

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.2))

    # Panel A: Non-English-Origin — Overall LLM Beta vs. Orthogonalized Writing (clean shaded 95% CI bands)
    ax = axes[0]
    ax.axvspan(launch_x, len(qtrs) - 0.5, color="#f4ecf7", alpha=0.50, zorder=1)
    for task, label, col, mkr, ls, a_band in [
        ("llm_openai_beta", "Overall LLM Exposure (OpenAI β)", "#1b4f72", "o", "-", 0.22),
        ("llm_writing_orth", "Orthogonalized LLM Writing", "#148f77", "^", "--", 0.14),
    ]:
        sub = es_df[(es_df["comparison"] == "NonEng_vs_Native") & (es_df["task"] == task)].sort_values("yq_int")
        xs = np.array([x_map[q] for q in sub["yq_int"]])
        y = sub["coef"].to_numpy()
        ci = 1.96 * sub["se"].to_numpy()
        ax.fill_between(xs, y - ci, y + ci, color=col, alpha=a_band, zorder=2)
        ax.plot(xs, y, f"{ls}{mkr}", color=col, label=label, linewidth=1.8, markersize=4.8, zorder=4)
    ax.axhline(0, color="#2c3e50", linewidth=0.9, zorder=2)
    ax.axvline(launch_x, color="#7d3c98", linestyle=":", linewidth=1.6, zorder=3)
    ax.text(launch_x + 0.35, 0.056, "ChatGPT Launch\n(Nov 30, 2022)", color="#7d3c98", fontsize=8.3, fontweight="bold", va="top")

    tick_idx = [i for i, q in enumerate(qtrs) if q % 10 in (1, 3)]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([q_labels[i] for i in tick_idx], rotation=40, ha="right", fontsize=8.5)
    ax.set_ylim(-0.060, 0.062)
    ax.set_xlabel("Quarter (Reference = 2022 Q3)")
    ax.set_ylabel("Non-English Imm. × Task (Log Weekly Wage)")
    ax.set_title("A. Non-English Immigrants: Overall LLM vs. Pure Writing", loc="left", fontweight="bold", pad=10)
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, edgecolor="#dcdcdc", loc="lower left", fontsize=8.5)

    # Panel B: Non-English-Origin vs. English-Origin Placebo (Overall LLM Beta)
    ax = axes[1]
    ax.axvspan(launch_x, len(qtrs) - 0.5, color="#f4ecf7", alpha=0.50, zorder=1)
    for comp, label, col, mkr, ls, a_band in [
        ("NonEng_vs_Native", "Non-English-Origin Immigrants vs. Natives", "#1b4f72", "o", "-", 0.24),
        ("Eng_vs_Native", "English-Origin Immigrants vs. Natives (Comparison)", "#c0392b", "s", "--", 0.13),
    ]:
        sub = es_df[(es_df["comparison"] == comp) & (es_df["task"] == "llm_openai_beta")].sort_values("yq_int")
        xs = np.array([x_map[q] for q in sub["yq_int"]])
        y = sub["coef"].to_numpy()
        ci = 1.96 * sub["se"].to_numpy()
        ax.fill_between(xs, y - ci, y + ci, color=col, alpha=a_band, zorder=2)
        ax.plot(xs, y, f"{ls}{mkr}", color=col, label=label, linewidth=1.7, markersize=4.6, zorder=4)
    ax.axhline(0, color="#2c3e50", linewidth=0.9, zorder=2)
    ax.axvline(launch_x, color="#7d3c98", linestyle=":", linewidth=1.6, zorder=3)
    ax.text(launch_x + 0.35, 0.165, "ChatGPT Launch\n(Nov 30, 2022)", color="#7d3c98", fontsize=8.3, fontweight="bold", va="top")

    ax.set_xticks(tick_idx)
    ax.set_xticklabels([q_labels[i] for i in tick_idx], rotation=40, ha="right", fontsize=8.5)
    ax.set_ylim(-0.140, 0.180)
    ax.set_xlabel("Quarter (Reference = 2022 Q3)")
    ax.set_ylabel("Immigrant × OpenAI LLM β (Log Weekly Wage)")
    ax.set_title("B. Origin Comparison Through 2026 Q3 (OpenAI LLM Beta)", loc="left", fontweight="bold", pad=10)
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, edgecolor="#dcdcdc", loc="lower left", fontsize=8.5)

    fig.suptitle(
        "CPS ORG Quarterly Event Study: GenAI Exposure & Immigrant Relative Wages (2020 Q1 - 2026 Q3)",
        fontsize=12.5, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / "fig1_cps_quarterly_event_study.png", dpi=250)
    plt.close(fig)


def make_fig4_honestdid(acs_hd: pd.DataFrame, cps_hd: pd.DataFrame) -> None:
    """
    Fig 4: Rambachan & Roth (2023) HonestDiD Relative-Magnitude Sensitivity Bounds
      Panel A: ACS Annual Event Study (2015–2024, Avg 2023–24) — OpenAI LLM β vs. Orthogonalized Writing
      Panel B: CPS Quarterly Event Study (Pre 2021Q1–2022Q2) — Early (2023–24) vs. Full (2023–26) OpenAI LLM β
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))

    # Panel A: ACS HonestDiD
    ax = axes[0]
    for task, label, col, dx in [
        ("llm_openai_beta", "OpenAI LLM β (95% FLCI)", "#1b4f72", -0.055),
        ("llm_writing_orth", "Orthogonalized LLM Writing (95% FLCI)", "#c0392b", 0.055),
    ]:
        sub = acs_hd[(acs_hd["comparison"] == "NonEng_vs_Native") & (acs_hd["task"] == task)].sort_values("mbar")
        xs = sub["mbar"].to_numpy() + dx
        y_mid = sub["theta_hat"].to_numpy()
        y_lo = sub["flci_lo"].to_numpy()
        y_hi = sub["flci_hi"].to_numpy()
        ax.errorbar(
            xs, y_mid, yerr=[y_mid - y_lo, y_hi - y_mid],
            fmt="o", color=col, label=label, capsize=4.5, elinewidth=2.0, markersize=6, zorder=4
        )
        # Plot identified set [id_set_lo, id_set_hi] as thicker inner bar
        ax.vlines(xs, sub["id_set_lo"], sub["id_set_hi"], color=col, linewidth=4.2, alpha=0.45, zorder=3)

    ax.axhline(0, color="#2c3e50", linestyle="--", linewidth=1.0, zorder=2)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xlabel(r"Relative-Magnitude Pre-Trend Bound ($\bar{M}$)")
    ax.set_ylabel("Post-2022 Average Effect (Log Hourly Wage)")
    ax.set_title("A. ACS Annual Event Study (2015–2024, Avg 2023–24)", loc="left", fontweight="bold", pad=10)
    ax.set_ylim(-0.040, 0.045)
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, edgecolor="#dcdcdc", loc="upper left", fontsize=8.5)
    ax.text(0.50, -0.036, "Breakdown value: $\\bar{M}^* < 0.5$ for OpenAI $\\beta$;\n$\\bar{M}^* = 0$ for Orthogonalized Writing",
            fontsize=8.2, color="#34495e", bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8f9f9", edgecolor="#bdc3c7"))

    # Panel B: CPS Quarterly HonestDiD (Early 2023-24 vs Full 2023-26)
    ax = axes[1]
    for pwin, label, col, dx in [
        ("Post_2023Q1_2024Q4", "CPS Early GenAI: 2023Q1–2024Q4 (95% FLCI)", "#2471a3", -0.055),
        ("Post_2023Q1_2026Q3", "CPS Full Post: 2023Q1–2026Q3 (95% FLCI)", "#148f77", 0.055),
    ]:
        sub = cps_hd[
            (cps_hd["comparison"] == "NonEng_vs_Native")
            & (cps_hd["task"] == "llm_openai_beta")
            & (cps_hd["pre_window"] == "Pre_2021Q1_2022Q2")
            & (cps_hd["post_window"] == pwin)
        ].sort_values("mbar")
        xs = sub["mbar"].to_numpy() + dx
        y_mid = sub["theta_hat"].to_numpy()
        y_lo = sub["flci_lo"].to_numpy()
        y_hi = sub["flci_hi"].to_numpy()
        ax.errorbar(
            xs, y_mid, yerr=[y_mid - y_lo, y_hi - y_mid],
            fmt="s", color=col, label=label, capsize=4.5, elinewidth=2.0, markersize=5.8, zorder=4
        )
        ax.vlines(xs, sub["id_set_lo"], sub["id_set_hi"], color=col, linewidth=4.2, alpha=0.45, zorder=3)

    ax.axhline(0, color="#2c3e50", linestyle="--", linewidth=1.0, zorder=2)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xlabel(r"Relative-Magnitude Pre-Trend Bound ($\bar{M}$)")
    ax.set_ylabel("Post-ChatGPT Average Effect (Log Weekly Wage)")
    ax.set_title("B. CPS Quarterly Event Study (OpenAI LLM β, Pre 2021–22)", loc="left", fontweight="bold", pad=10)
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, edgecolor="#dcdcdc", loc="upper left", fontsize=8.5)

    fig.suptitle(
        "Exploratory Plug-In LP Pre-Trend Bounds (Time-Scaled Relative-Magnitude Delta^RM(Mbar))",
        fontsize=11.8, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / "fig4_honestdid_sensitivity.png", dpi=250)
    plt.close(fig)


def make_tab2_cps_did(did_df: pd.DataFrame, cps_counts: dict) -> None:
    """Write publication-grade LaTeX Table 2: CPS ORG Quarterly DiD, Post-2024 Persistence & MDEs."""
    se_types = sorted(did_df["se_type"].dropna().unique().tolist()) if "se_type" in did_df.columns else ["twoway_occ2018_statefip"]
    se_note = "two-way CRV1 (OCC2018 + state) in all rows" if se_types == ["twoway_occ2018_statefip"] else f"logged SE types: {', '.join(se_types)}"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{CPS ORG Quarterly Difference-in-Differences, Post-2024 Persistence, and 80\% MDEs (2020 Q1--2026 Q3)}",
        r"\label{tab:cps_quarterly_did}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Non-English Origin} & 80\% MDE & \multicolumn{2}{c}{English Origin (Comp.)} & \multicolumn{2}{c}{DDD (NonEng $-$ Eng)} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){5-6} \cmidrule(lr){7-8}",
        r"Task / Wage Harmonization & Early & Mature & (Mature & Early & Mature & Early & Mature \\",
        r" & (2023--24) & (2025--26) & Non-Eng) & (2023--24) & (2025--26) & (2023--24) & (2025--26) \\",
        r"\midrule",
        r"\multicolumn{8}{l}{\textit{Panel A: Task Decomposition (Harmonized Capped Weekly Wage, \$2,884.61 cap)}} \\",
        r"\addlinespace[3pt]",
    ]

    pA = did_df[did_df["outcome"] == "ln_earnweek_cap"].set_index("task")
    for tkey, tlab in [
        ("written_eng_only", "O*NET 31.0 Written-English"),
        ("llm_openai_beta", "OpenAI LLM Exposure ($E1+0.5\\,E2$)"),
        ("llm_writing_orth", "LLM Writing (Orthogonalized)"),
        ("llm_coding_math_raw", "LLM Coding \\& Math (Raw)"),
        ("face_to_face_social", "Face-to-Face Social Skills"),
    ]:
        if tkey not in pA.index:
            continue
        r = pA.loc[tkey]
        mde_m = r["mde80_ne_2526"] if "mde80_ne_2526" in r.index else 2.8016 * r["se_ne_2526"]
        lines.append(
            f"{tlab} & {fmt_coef(r['b_ne_2324'], r['p_ne_2324'])} & {fmt_coef(r['b_ne_2526'], r['p_ne_2526'])} & {mde_m:.4f} "
            f"& {fmt_coef(r['b_eng_2324'], r['p_eng_2324'])} & {fmt_coef(r['b_eng_2526'], r['p_eng_2526'])} "
            f"& {fmt_coef(r['ddd_2324'], r['ddd_p_2324'])} & {fmt_coef(r['ddd_2526'], r['ddd_p_2526'])} \\\\"
        )
        lines.append(
            f" & {fmt_se(r['se_ne_2324'])} & {fmt_se(r['se_ne_2526'])} & "
            f"& {fmt_se(r['se_eng_2324'])} & {fmt_se(r['se_eng_2526'])} "
            f"& {fmt_se(r['ddd_se_2324'])} & {fmt_se(r['ddd_se_2526'])} \\\\"
        )

    lines.extend([
        r"\midrule",
        r"\multicolumn{8}{l}{\textit{Panel B: April 2023 CPS Top-Code Harmonization Robustness (OpenAI LLM $\beta$)}} \\",
        r"\addlinespace[3pt]",
    ])
    for ykey, ylab in [
        ("ln_earnweek_q93", "Log Weekly Wage (Q93 Winsorized)"),
        ("ln_hrwage_cap", "Log Hourly Wage (Harmonized Cap)"),
        ("ln_earnweek_raw", "Log Weekly Wage (Uncapped Raw)"),
    ]:
        r = did_df[(did_df["outcome"] == ykey) & (did_df["task"] == "llm_openai_beta")].iloc[0]
        mde_m = r["mde80_ne_2526"] if "mde80_ne_2526" in r.index else 2.8016 * r["se_ne_2526"]
        lines.append(
            f"{ylab} & {fmt_coef(r['b_ne_2324'], r['p_ne_2324'])} & {fmt_coef(r['b_ne_2526'], r['p_ne_2526'])} & {mde_m:.4f} "
            f"& {fmt_coef(r['b_eng_2324'], r['p_eng_2324'])} & {fmt_coef(r['b_eng_2526'], r['p_eng_2526'])} "
            f"& {fmt_coef(r['ddd_2324'], r['ddd_p_2324'])} & {fmt_coef(r['ddd_2526'], r['ddd_p_2526'])} \\\\"
        )
        lines.append(
            f" & {fmt_se(r['se_ne_2324'])} & {fmt_se(r['se_ne_2526'])} & "
            f"& {fmt_se(r['se_eng_2324'])} & {fmt_se(r['se_eng_2526'])} "
            f"& {fmt_se(r['ddd_se_2324'])} & {fmt_se(r['ddd_se_2526'])} \\\\"
        )

    n_obs = int(pA.iloc[0]["n_obs"])
    r2_val = float(pA.iloc[0]["r2"])
    lines.extend([
        r"\midrule",
        f"Observations ($N$) & \\multicolumn{{7}}{{c}}{{{n_obs:,} FT wage earners (26 quarters, excl.\\ transition 2022 Q4); $R^2 = {r2_val:.3f}$}} \\\\",
        f"Clusters (OCC2018 / State) & \\multicolumn{{7}}{{c}}{{{cps_counts['n_occ2018_clusters']} 2018 Census occupations / {cps_counts['n_state_clusters']} states ({se_note})}} \\\\",
        r"Fixed Effects & \multicolumn{7}{c}{$\text{OCC2018}\times\text{YQ}$, $\text{State}\times\text{YQ}$, $\text{Educ4}\times\text{YQ}$, $\text{Origin}\times\text{YQ}$, $\text{Sex}\times\text{YQ}$} \\",
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\vspace{4pt}",
        r"\begin{minipage}{\textwidth}\scriptsize",
        r"\textit{Notes:} Sample is full-time wage earners aged 18--64 in 81 Census Basic Monthly CPS ORG files spanning Jan 2020--Oct 2026 (Oct 2025 is missing from Census releases due to the federal shutdown, leaving 2025 Q4 with two months; partial 2026 Q4 and transition quarter 2022 Q4 are excluded), weighted by \texttt{EARNWT}. Pre-period is 2020 Q1--2022 Q3; Early GenAI is 2023 Q1--2024 Q4; Mature GenAI is 2025 Q1--2026 Q3. Column ``80\% MDE'' reports $2.8016 \times \text{SE}$ for Mature Non-English-origin estimates. Note the positive CPS Face-to-Face Social estimates for Non-English-origin immigrants ($+0.0145^{***}$ in 2023--24, $+0.0115^{**}$ in 2025--26, DDD $+0.0393^{***}$), discussed in Section 5. Standard errors in parentheses. $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
        r"\end{minipage}",
        r"\end{table}",
    ])
    (TAB_DIR / "tab2_cps_quarterly_did.tex").write_text("\n".join(lines) + "\n")


def make_tab4_robustness_honestdid(acs_hd: pd.DataFrame, cps_hd: pd.DataFrame, het_df: pd.DataFrame) -> None:
    """Write publication-grade LaTeX Table 4: Rambachan-Roth HonestDiD & ACS Heterogeneity."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Exploratory Plug-In LP Pre-Trend Bounds and ACS Heterogeneity (2021--2024)}",
        r"\label{tab:robustness_honestdid}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\multicolumn{5}{l}{\textit{Panel A: Exploratory Plug-In LP Pre-Trend Bounds (Time-Scaled 95\% FLCIs, Non-English Immigrants vs.\ Natives)}} \\",
        r"\addlinespace[3pt]",
        r"Pre-Trend Bound ($\bar{M}$) & ACS OpenAI LLM $\beta$ & ACS LLM Writing (Orth.) & CPS OpenAI $\beta$ (2023--24) & CPS OpenAI $\beta$ (2023--26) \\",
        r"\midrule",
    ]

    a_beta = acs_hd[(acs_hd["comparison"] == "NonEng_vs_Native") & (acs_hd["task"] == "llm_openai_beta")].set_index("mbar")
    a_writ = acs_hd[(acs_hd["comparison"] == "NonEng_vs_Native") & (acs_hd["task"] == "llm_writing_orth")].set_index("mbar")
    c_e = cps_hd[(cps_hd["comparison"] == "NonEng_vs_Native") & (cps_hd["task"] == "llm_openai_beta") &
                 (cps_hd["pre_window"] == "Pre_2021Q1_2022Q2") & (cps_hd["post_window"] == "Post_2023Q1_2024Q4")].set_index("mbar")
    c_f = cps_hd[(cps_hd["comparison"] == "NonEng_vs_Native") & (cps_hd["task"] == "llm_openai_beta") &
                 (cps_hd["pre_window"] == "Pre_2021Q1_2022Q2") & (cps_hd["post_window"] == "Post_2023Q1_2026Q3")].set_index("mbar")

    lines.append(
        f"Point Est.\\ $\\hat{{\\theta}}$ (SE) & {a_beta.loc[0.0,'theta_hat']:.4f} ({a_beta.loc[0.0,'se_theta']:.4f}) "
        f"& {a_writ.loc[0.0,'theta_hat']:.4f} ({a_writ.loc[0.0,'se_theta']:.4f}) "
        f"& {c_e.loc[0.0,'theta_hat']:.4f} ({c_e.loc[0.0,'se_theta']:.4f}) "
        f"& {c_f.loc[0.0,'theta_hat']:.4f} ({c_f.loc[0.0,'se_theta']:.4f}) \\\\"
    )
    lines.append(r"\addlinespace[2pt]")
    for m in [0.0, 0.5, 1.0, 1.5, 2.0]:
        def rng(df_m):
            return f"[{df_m.loc[m,'flci_lo']:+.4f},\\; {df_m.loc[m,'flci_hi']:+.4f}]"
        lines.append(f"$\\bar{{M}} = {m:.1f}$ & {rng(a_beta)} & {rng(a_writ)} & {rng(c_e)} & {rng(c_f)} \\\\")

    lines.extend([
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Panel B: ACS 2021--2024 Heterogeneity (Non-English-Origin Immigrants vs.\ U.S.\ Natives)}} \\",
        r"\addlinespace[3pt]",
        r"Subgroup & \multicolumn{2}{c}{OpenAI LLM $\beta$ (SE)} & \multicolumn{2}{c}{LLM Writing Orth.\ (SE) / $N$ Workers} \\",
        r"\midrule",
    ])

    h21 = het_df[het_df["window"] == "2021-2024"]
    sub_order = [
        "Fluent English (SPEAKENG 3-4)",
        "Non-Fluent English (SPEAKENG 1,5,6)",
        "College STEM (DEGFIELD STEM)",
        "College Non-STEM",
        "Non-College (< BA)",
        "Men",
        "Women",
        "Young (18-34)",
        "Prime-Age (35-49)",
        "Senior (50-64)",
        "Recent Arrivals (<10 yrs)",
        "Established (>=10 yrs)",
    ]
    for s in sub_order:
        rb = h21[(h21["subgroup"] == s) & (h21["task"] == "llm_openai_beta")].iloc[0]
        rw = h21[(h21["subgroup"] == s) & (h21["task"] == "llm_writing_orth")].iloc[0]
        s_tex = s.replace("<", "$<$").replace(">=", "$\\ge$").replace("&", "\\&")
        lines.append(
            f"{s_tex} & {fmt_coef(rb['coef'], rb['pval'])} & {fmt_se(rb['se'])} "
            f"& {fmt_coef(rw['coef'], rw['pval'])} {fmt_se(rw['se'])} & {int(rb['n_workers']):,} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\vspace{4pt}",
        r"\begin{minipage}{\textwidth}\scriptsize",
        r"\textit{Notes:} Panel A reports exploratory plug-in pre-trend sensitivity bounds computed via a Python linear program (\texttt{scipy.optimize.linprog}, HiGHS) inspired by \citet{RambachanRoth2023} relative-magnitude restrictions ($\Delta^{\text{RM}}(\bar{M})$), using bias-aware folded-normal critical values $\text{cv}_{0.05}(b_{\max}/\hat{\sigma})$ and explicit per-period time-step scaling for uneven ACS calendar years and quarterly CPS steps (accounting for the omitted 2022 Q4 transition quarter). Because solving the LP at the plug-in maximum pre-trend slope does not account for sampling uncertainty in the estimated pre-trend, these intervals should be read as exploratory sensitivity diagnostics awaiting benchmarking against the official R \texttt{HonestDiD} package. $\bar{M}=0$ imposes exact parallel trends; $\bar{M}=1$ allows post-treatment per-period trend slopes as large as the maximum plug-in pre-treatment per-period slope. Panel B estimates the baseline ACS 2021--2024 specification separately by subgroup with two-way clustered SEs (\texttt{OCCSOC} + state). $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
        r"\end{minipage}",
        r"\end{table}",
    ])
    (TAB_DIR / "tab4_robustness_honestdid.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    cps_es = pd.read_csv(RES_DIR / "cps_quarterly_event_studies.csv")
    cps_did = pd.read_csv(RES_DIR / "cps_quarterly_did_persistence.csv")
    cps_hd = pd.read_csv(RES_DIR / "cps_honestdid_bounds.csv")
    acs_hd = pd.read_csv(RES_DIR / "acs_honestdid_bounds.csv")
    het_df = pd.read_csv(RES_DIR / "acs_heterogeneity.csv")
    cps_sum = json.loads((RES_DIR / "cps_summary.json").read_text())

    make_fig1_cps_quarterly(cps_es)
    make_fig4_honestdid(acs_hd, cps_hd)
    make_tab2_cps_did(cps_did, cps_sum["counts"])
    make_tab4_robustness_honestdid(acs_hd, cps_hd, het_df)
    print("Generated Fig 1, Fig 4, Tab 2, and Tab 4 successfully!")


if __name__ == "__main__":
    main()
