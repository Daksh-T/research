#!/usr/bin/env python3
"""
Paper 1 — Generate Publication-Ready 2026 Anthropic Economic Index Figure (Fig 5)
          and LaTeX Table (Tab 5, wrapped in \\resizebox{\\textwidth}{!}{...}, pure ASCII)
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

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
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


def fmt_c(c: float, p: float) -> str:
    return f"{c:.4f}{stars(p)}"


def fmt_se(s: float) -> str:
    return f"({s:.4f})"


def make_fig5_aei_2026() -> None:
    t6 = pd.read_parquet(RES_DIR / "aei_2026_tasks_occsoc.parquet")
    exact = t6[t6["aei_2026_exact_match"] == 1].copy()
    cps_es = pd.read_csv(RES_DIR / "aei_2026_cps_event_studies.csv")
    acs_p = pd.read_csv(RES_DIR / "aei_2026_acs_origin_placebo.csv")
    cps_d = pd.read_csv(RES_DIR / "aei_2026_cps_did_persistence.csv")

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.8))

    # --- Panel A: 2026 Observed Exposure vs. Legacy Pre-2025 OpenAI Beta ---
    ax = axes[0, 0]
    ax.scatter(
        exact["openai_dv_beta"], exact["aei_2026_obs_raw"],
        s=18, alpha=0.45, color="#1b4f72", edgecolors="none", zorder=2,
    )
    # Linear fit
    m, b = np.polyfit(exact["openai_dv_beta"], exact["aei_2026_obs_raw"], 1)
    xs_fit = np.linspace(0, 1.0, 100)
    ax.plot(xs_fit, m * xs_fit + b, color="#c0392b", linestyle="--", linewidth=1.6, zorder=3,
            label="OLS fit (r = 0.642, Spearman = 0.700)")
    ax.plot([0, 0.8], [0, 0.8], color="#7f8c8d", linestyle=":", linewidth=1.1, zorder=2, label="45-degree line")

    callouts = [
        ("151251", "Computer Programmers", (-105, -16)),
        ("434051", "Customer Service Reps", (10, 6)),
        ("131161", "Market Research Analysts", (-118, -14)),
        ("439081", "Proofreaders", (-74, 14)),
        ("432021", "Telephone Operators", (-95, 16)),
    ]
    for code, lbl, (dx, dy) in callouts:
        r = exact[exact["occsoc"] == code]
        if len(r) == 1:
            x0 = float(r["openai_dv_beta"].iloc[0])
            y0 = float(r["aei_2026_obs_raw"].iloc[0])
            ax.scatter([x0], [y0], s=42, color="#c0392b", zorder=4)
            ax.annotate(
                lbl, (x0, y0), xytext=(dx, dy), textcoords="offset points",
                fontsize=7.8, fontweight="bold", color="#2c3e50",
                arrowprops=dict(arrowstyle="-", color="#7f8c8d", lw=0.7),
            )
    ax.set_xlim(-0.02, 1.03)
    ax.set_ylim(-0.03, 0.82)
    ax.set_xlabel("GPT-4 Capability Rubric Exposure (Eloundou et al. 2024, OpenAI beta)")
    ax.set_ylabel("2026 Observed AI Exposure (Massenkoff & McCrory 2026)")
    ax.set_title("A. 2026 Observed vs. GPT-4 Capability Rubric Exposure (748 SOCs)", loc="left", fontweight="bold", pad=8)
    ax.legend(frameon=True, facecolor="white", framealpha=0.92, edgecolor="#dcdcdc", loc="upper left", fontsize=8.0)

    # --- Panel B: CPS ORG Quarterly Event Study — Non-English vs. English-Origin Placebo (AEI 2026 Observed) ---
    ax = axes[0, 1]
    qtrs = sorted(cps_es["yq_int"].unique().tolist())
    x_map = {q: i for i, q in enumerate(qtrs)}
    q_labels = [f"{q // 10}Q{q % 10}" for q in qtrs]
    launch_x = x_map[20223] + 0.5
    tick_idx = [i for i, q in enumerate(qtrs) if q % 10 in (1, 3)]

    ax.axvspan(launch_x, len(qtrs) - 0.5, color="#f4ecf7", alpha=0.50, zorder=1)
    for comp, lbl, col, mkr, ls, ab in [
        ("NonEng_vs_Native", "Non-English-Origin Immigrants vs. Natives", "#1b4f72", "o", "-", 0.24),
        ("Eng_vs_Native", "English-Origin Immigrants vs. Natives (Comparison)", "#c0392b", "s", "--", 0.13),
    ]:
        sub = cps_es[(cps_es["comparison"] == comp) & (cps_es["task"] == "aei_2026_observed")].sort_values("yq_int")
        xs = np.array([x_map[q] for q in sub["yq_int"]])
        y = sub["coef"].to_numpy()
        ci = 1.96 * sub["se"].to_numpy()
        ax.fill_between(xs, y - ci, y + ci, color=col, alpha=ab, zorder=2)
        ax.plot(xs, y, f"{ls}{mkr}", color=col, label=lbl, linewidth=1.7, markersize=4.4, zorder=4)
    ax.axhline(0, color="#2c3e50", linewidth=0.9, zorder=2)
    ax.axvline(launch_x, color="#7d3c98", linestyle=":", linewidth=1.5, zorder=3)
    ax.text(launch_x + 0.35, 0.115, "ChatGPT Launch\n(Nov 30, 2022)", color="#7d3c98", fontsize=8.0, fontweight="bold", va="top")
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([q_labels[i] for i in tick_idx], rotation=40, ha="right", fontsize=8.0)
    ax.set_ylim(-0.11, 0.13)
    ax.set_xlabel("Quarter (Reference = 2022 Q3)")
    ax.set_ylabel("Immigrant x AEI 2026 Observed (Log Weekly Wage)")
    ax.set_title("B. CPS ORG Origin Comparison: 2026 Observed Exposure (2020 Q1-2026 Q3)", loc="left", fontweight="bold", pad=8)
    ax.legend(frameon=True, facecolor="white", framealpha=0.94, edgecolor="#dcdcdc", loc="lower left", fontsize=8.0)

    # --- Panel C: CPS ORG Quarterly Event Study — AEI 2026 Observed vs. Orthogonalized Writing vs. Robot ---
    ax = axes[1, 0]
    ax.axvspan(launch_x, len(qtrs) - 0.5, color="#f4ecf7", alpha=0.50, zorder=1)
    for tcol, lbl, col, mkr, ls, ab in [
        ("aei_2026_observed", "AEI 2026 Observed Exposure", "#1b4f72", "o", "-", 0.22),
        ("aei_2026_writing_orth", "AEI 2026 Writing (Orthogonalized)", "#148f77", "^", "--", 0.12),
        ("aei_2026_robot", "AEI 2026 Robot Exposure (Physical)", "#b9770e", "D", ":", 0.10),
    ]:
        sub = cps_es[(cps_es["comparison"] == "NonEng_vs_Native") & (cps_es["task"] == tcol)].sort_values("yq_int")
        xs = np.array([x_map[q] for q in sub["yq_int"]])
        y = sub["coef"].to_numpy()
        ci = 1.96 * sub["se"].to_numpy()
        ax.fill_between(xs, y - ci, y + ci, color=col, alpha=ab, zorder=2)
        ax.plot(xs, y, f"{ls}{mkr}", color=col, label=lbl, linewidth=1.6, markersize=4.2, zorder=4)
    ax.axhline(0, color="#2c3e50", linewidth=0.9, zorder=2)
    ax.axvline(launch_x, color="#7d3c98", linestyle=":", linewidth=1.5, zorder=3)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([q_labels[i] for i in tick_idx], rotation=40, ha="right", fontsize=8.0)
    ax.set_ylim(-0.055, 0.055)
    ax.set_xlabel("Quarter (Reference = 2022 Q3)")
    ax.set_ylabel("Non-English Imm. x Task (Log Weekly Wage)")
    ax.set_title("C. CPS ORG Task Decomposition: Observed AI vs. Writing vs. Robots", loc="left", fontweight="bold", pad=8)
    ax.legend(frameon=True, facecolor="white", framealpha=0.94, edgecolor="#dcdcdc", loc="lower left", fontsize=8.0)

    # --- Panel D: Forest Plot — ACS 2021-24 & CPS 2025-26 Origin-Placebo Estimates ---
    ax = axes[1, 1]
    acs21 = acs_p[acs_p["window"] == "ACS_2021_2024"].set_index("task")
    cps_cap = cps_d[cps_d["outcome"] == "ln_earnweek_cap"].set_index("task")

    plot_items = [
        ("aei_2026_observed", "AEI 2026 Observed\n(Primary)"),
        ("aei_2026_writing_orth", "AEI 2026 Writing\n(Orthogonalized)"),
        ("aei_2026_coding_raw", "AEI 2026 Coding\n& Math"),
        ("aei_2025_auto_exp", "AEI 2025 Automation\n(Caveat: 2025)"),
        ("aei_2025_aug_exp", "AEI 2025 Augmentation\n(Caveat: 2025)"),
        ("aei_2026_robot", "AEI 2026 Robot\n(Physical)"),
        ("llm_openai_beta", "Eloundou (2024)\nOpenAI Beta"),
    ]
    y_pos = np.arange(len(plot_items))[::-1]

    for i, (tk, _) in enumerate(plot_items):
        yp = y_pos[i]
        # ACS 2021-24 Non-English
        c_acs_ne = acs21.loc[tk, "noneng_coef"]
        se_acs_ne = acs21.loc[tk, "noneng_se"]
        # ACS 2021-24 English Placebo
        c_acs_eng = acs21.loc[tk, "eng_coef"]
        se_acs_eng = acs21.loc[tk, "eng_se"]
        # CPS 2025-26 Non-English
        c_cps_ne = cps_cap.loc[tk, "ne_2526"]
        se_cps_ne = cps_cap.loc[tk, "ne_2526_se"]

        ax.errorbar(c_acs_ne, yp + 0.18, xerr=1.96 * se_acs_ne, fmt="o", color="#1b4f72",
                    markersize=5.5, capsize=3, linewidth=1.5,
                    label="ACS 2021-24: Non-English Imm." if i == 0 else None)
        ax.errorbar(c_acs_eng, yp, xerr=1.96 * se_acs_eng, fmt="s", color="#c0392b",
                    markersize=5.0, capsize=3, linewidth=1.3, alpha=0.85,
                    label="ACS 2021-24: English-Origin (Comparison)" if i == 0 else None)
        ax.errorbar(c_cps_ne, yp - 0.18, xerr=1.96 * se_cps_ne, fmt="^", color="#148f77",
                    markersize=5.5, capsize=3, linewidth=1.5,
                    label="CPS 2025-26: Non-English Imm." if i == 0 else None)

    ax.axvline(0, color="#2c3e50", linewidth=1.0, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([lbl for _, lbl in plot_items], fontsize=8.3)
    ax.set_xlim(-0.032, 0.032)
    ax.set_xlabel("Post-ChatGPT Relative Log Wage Effect per SD (95% CI)")
    ax.set_title("D. 2026 Observed AI vs. Legacy Benchmarks (ACS & CPS)", loc="left", fontweight="bold", pad=8)
    ax.legend(frameon=True, facecolor="white", framealpha=0.94, edgecolor="#dcdcdc", loc="upper left", fontsize=7.6)

    fig.suptitle(
        "2026 Anthropic Economic Index Observed AI Exposure: Divergence from Legacy Indices & Immigrant Wage Effects",
        fontsize=12.5, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = FIG_DIR / "fig5_aei_2026_observed.png"
    fig.savefig(out_path, dpi=250)
    plt.close(fig)
    print(f"Saved {out_path}")


def make_tab5_aei_2026() -> None:
    acs_p = pd.read_csv(RES_DIR / "aei_2026_acs_origin_placebo.csv")
    ind_df = pd.read_csv(RES_DIR / "acs_individual_vs_cell_comparison.csv")
    cps_d = pd.read_csv(RES_DIR / "aei_2026_cps_did_persistence.csv")
    acs_hr = pd.read_csv(RES_DIR / "aei_2026_acs_horserace.csv")

    a21 = acs_p[acs_p["window"] == "ACS_2021_2024"].set_index("task")
    ind21 = ind_df[ind_df["window"] == "2021-2024"].set_index("task")
    c_cap = cps_d[cps_d["outcome"] == "ln_earnweek_cap"].set_index("task")

    panel_a_rows = [
        ("written_eng_only", "O*NET 31.0 Written-English Intensity"),
        ("llm_openai_beta", "OpenAI $\\beta$ (Eloundou et al. 2024)"),
        ("llm_felten_lm", "Felten LM AIOE (2023)"),
        ("aei_2026_observed", "AEI 2026 Observed Exposure (Mar 2026)"),
        ("aei_2026_writing_orth", "AEI 2026 Writing (Orthogonalized)"),
        ("aei_2026_coding_raw", "AEI 2026 Coding \\& Math"),
        ("aei_2026_task_writing_orth", "AEI 2026 Task Writing (Orth.)$^\\ddagger$"),
        ("aei_2025_auto_exp", "AEI 2025 Automation Exposure$^{\\dagger\\ddagger}$"),
        ("aei_2025_aug_exp", "AEI 2025 Augmentation Exposure$^{\\dagger\\ddagger}$"),
        ("aei_2026_robot", "AEI 2026 Robot Exposure (Manual/Physical)"),
    ]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Capability-Rubric vs.\ Observed Usage (AEI 2026) Exposure: ACS Individual-Worker and CPS ORG Estimates with 80\% MDEs}",
        r"\label{tab:aei_2026_observed}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r" & \multicolumn{4}{c}{ACS Post-COVID (2021--2024; $N = 3.84\text{M}$ workers)} & \multicolumn{4}{c}{CPS ORG Mature GenAI (2025 Q1--2026 Q3)} \\",
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
        r"Occupation Exposure Index & Non-Eng & 80\% MDE & Eng (Comp.) & DDD & Non-Eng & 80\% MDE & Eng (Comp.) & DDD \\",
        r"(Standardized, Mean 0, SD 1) & (1) & (2) & (3) & (4) & (5) & (6) & (7) & (8) \\",
        r"\midrule",
        r"\multicolumn{9}{l}{\textit{Panel A: O*NET 31.0, Capability Rubrics, and March 2026 Anthropic Economic Index Observed Usage}} \\",
        r"\addlinespace[3pt]",
    ]

    for tk, lbl in panel_a_rows:
        if tk not in a21.index:
            continue
        if tk in ind21.index:
            ri = ind21.loc[tk]
            a_ne, a_ne_se, a_ne_p = ri["ind_b_noneng"], ri["ind_se_noneng"], ri["ind_p_noneng"]
            mde_a = ri["ind_mde80_noneng"]
            a_e, a_e_se, a_e_p = ri["ind_b_eng"], ri["ind_se_eng"], ri["ind_p_eng"]
            a_ddd, a_ddd_se, a_ddd_p = ri["ind_ddd_diff"], ri["ind_ddd_se"], ri["ind_ddd_p"]
        else:
            ra = a21.loc[tk]
            a_ne, a_ne_se, a_ne_p = ra["noneng_coef"], ra["noneng_se"], ra["noneng_p"]
            mde_a = ra["noneng_mde80"] if "noneng_mde80" in ra.index else 2.8016 * ra["noneng_se"]
            a_e, a_e_se, a_e_p = ra["eng_coef"], ra["eng_se"], ra["eng_p"]
            a_ddd, a_ddd_se, a_ddd_p = ra["ddd_coef"], ra["ddd_se"], ra["ddd_p"]
        if tk in c_cap.index:
            rc = c_cap.loc[tk]
            mde_c = rc["ne_2526_mde80"] if "ne_2526_mde80" in rc.index else 2.8016 * rc["ne_2526_se"]
            c5 = fmt_c(rc["ne_2526"], rc["ne_2526_p"])
            c6 = f"{mde_c:.4f}"
            c7 = fmt_c(rc["eng_2526"], rc["eng_2526_p"])
            c8 = fmt_c(rc["ddd_2526"], rc["ddd_2526_p"])
            s5 = fmt_se(rc["ne_2526_se"])
            s7 = fmt_se(rc["eng_2526_se"])
            s8 = fmt_se(rc["ddd_2526_se"])
        else:
            c5 = c6 = c7 = c8 = "---"
            s5 = s7 = s8 = ""
        lines.append(
            f"{lbl} & {fmt_c(a_ne, a_ne_p)} & {mde_a:.4f} & {fmt_c(a_e, a_e_p)} "
            f"& {fmt_c(a_ddd, a_ddd_p)} & {c5} & {c6} & {c7} & {c8} \\\\"
        )
        lines.append(
            f" & {fmt_se(a_ne_se)} & & {fmt_se(a_e_se)} & {fmt_se(a_ddd_se)} "
            f"& {s5} & & {s7} & {s8} \\\\"
        )

    # Panel B: CPS Top-Code Harmonization & Early vs. Mature for AEI 2026 Observed
    c_obs = cps_d[cps_d["task"] == "aei_2026_observed"].set_index("outcome")
    lines.extend([
        r"\midrule",
        r"\multicolumn{9}{l}{\textit{Panel B: CPS ORG 2026 Observed Exposure Across Wage Harmonizations (Early 2023--24 vs.\ Mature 2025--26)}} \\",
        r"\addlinespace[3pt]",
        r" & \multicolumn{2}{c}{Non-Eng Origin} & 80\% MDE & \multicolumn{2}{c}{Eng Origin (Comp.)} & \multicolumn{3}{c}{DDD (NonEng $-$ Eng)} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){5-6} \cmidrule(lr){7-9}",
        r"CPS Wage Harmonization & 2023--24 & 2025--26 & (25--26) & 2023--24 & 2025--26 & 2023--24 & \multicolumn{2}{c}{2025--26} \\",
        r"\addlinespace[2pt]",
    ])
    for oc, olbl in [
        ("ln_earnweek_cap", "Log Weekly Wage (Uniform \\$2,884.61 Cap)"),
        ("ln_earnweek_q93", "Log Weekly Wage (Q93 Winsorized)"),
        ("ln_hrwage_cap", "Log Hourly Wage (Harmonized Cap)"),
        ("ln_earnweek_raw", "Log Weekly Wage (Uncapped Raw)"),
    ]:
        r = c_obs.loc[oc]
        mde_c = r["ne_2526_mde80"] if "ne_2526_mde80" in r.index else 2.8016 * r["ne_2526_se"]
        lines.append(
            f"{olbl} & {fmt_c(r['ne_2324'], r['ne_2324_p'])} & {fmt_c(r['ne_2526'], r['ne_2526_p'])} & {mde_c:.4f} "
            f"& {fmt_c(r['eng_2324'], r['eng_2324_p'])} & {fmt_c(r['eng_2526'], r['eng_2526_p'])} "
            f"& {fmt_c(r['ddd_2324'], r['ddd_2324_p'])} & \\multicolumn{{2}}{{c}}{{{fmt_c(r['ddd_2526'], r['ddd_2526_p'])}}} \\\\"
        )
        lines.append(
            f" & {fmt_se(r['ne_2324_se'])} & {fmt_se(r['ne_2526_se'])} & "
            f"& {fmt_se(r['eng_2324_se'])} & {fmt_se(r['eng_2526_se'])} "
            f"& {fmt_se(r['ddd_2324_se'])} & \\multicolumn{{2}}{{c}}{{{fmt_se(r['ddd_2526_se'])}}} \\\\"
        )

    # Panel C: ACS Simultaneous Horse Race (HR2 & HR3)
    hr2 = acs_hr[acs_hr["spec"] == "HR2_AEI26_PostCOVID_21_24"].set_index("term")
    hr3 = acs_hr[acs_hr["spec"] == "HR3_AEI26_vs_LegacyOpenAI_21_24"].set_index("term")
    lines.extend([
        r"\midrule",
        r"\multicolumn{9}{l}{\textit{Panel C: ACS 2021--2024 Simultaneous Horse Races (Non-English Immigrants vs.\ U.S.\ Natives; $N = 3,788,812$)}} \\",
        r"\addlinespace[3pt]",
        f"Horse Race 1 (2026 Tasks): & \\multicolumn{{3}}{{l}}{{AEI 2026 Writing (Orth.): {fmt_c(hr2.loc['aei_2026_writing_orth','coef'], hr2.loc['aei_2026_writing_orth','pval'])} {fmt_se(hr2.loc['aei_2026_writing_orth','se'])}}} "
        f"& \\multicolumn{{2}}{{l}}{{AEI 2026 Coding: {fmt_c(hr2.loc['aei_2026_coding_raw','coef'], hr2.loc['aei_2026_coding_raw','pval'])} {fmt_se(hr2.loc['aei_2026_coding_raw','se'])}}} "
        f"& \\multicolumn{{3}}{{l}}{{AEI 2026 Robot: {fmt_c(hr2.loc['aei_2026_robot','coef'], hr2.loc['aei_2026_robot','pval'])} {fmt_se(hr2.loc['aei_2026_robot','se'])}}} \\\\",
        f"Horse Race 2 (2026 vs.\\ Rubric): & \\multicolumn{{3}}{{l}}{{AEI 2026 Observed: {fmt_c(hr3.loc['aei_2026_observed','coef'], hr3.loc['aei_2026_observed','pval'])} {fmt_se(hr3.loc['aei_2026_observed','se'])}}} "
        f"& \\multicolumn{{2}}{{l}}{{OpenAI $\\beta$: {fmt_c(hr3.loc['llm_openai_beta','coef'], hr3.loc['llm_openai_beta','pval'])} {fmt_se(hr3.loc['llm_openai_beta','se'])}}} "
        f"& \\multicolumn{{3}}{{l}}{{Face-to-Face: {fmt_c(hr3.loc['face_to_face_social','coef'], hr3.loc['face_to_face_social','pval'])} {fmt_se(hr3.loc['face_to_face_social','se'])}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"",
        r"\vspace{4pt}",
        r"\begin{minipage}{\textwidth}\scriptsize",
        r"\textit{Notes:} Columns (1)--(4) of Panel A report authoritative individual-level ACS worker regressions ($N = 3{,}840{,}043$) for all headline exposure indices ($^\ddagger$auxiliary rows use cell-grouped ACS regressions; side-by-side worker vs.\ cell-grouped comparisons are reported in Table~\ref{tab:acs_origin_placebo} and agree within $\le 0.00025$ log points across 2015--2024 and $\le 0.0001$ in 2021--2024). O*NET 31.0 (2026 release) task ratings are refreshed on rolling cycles; Eloundou et al.\ (2024) OpenAI $\beta$ was constructed in early 2023 based on GPT-4 capabilities; \citet{MassenkoffMcCrory2026} (Anthropic Economic Index, March 2026) measures post-treatment single-provider Claude usage (54.3\% zeros). Columns (2) and (6) report 80\%-power MDEs ($2.8016 \times \text{SE}$) for the occupational-exposure wage gradient among Non-English-origin immigrants. $^{\dagger}$AEI 2025 Automation and Augmentation shares \citep{Handa2025} are interim March 2025 snapshots. All specifications include occupation$\times$time, state$\times$time, education$\times$time, origin$\times$time, and sex$\times$time fixed effects, individual controls, and two-way clustered SEs (occupation + state). $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
        r"\end{minipage}",
        r"\end{table}",
    ])

    out_path = TAB_DIR / "tab5_aei_2026_observed.tex"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    make_fig5_aei_2026()
    make_tab5_aei_2026()
