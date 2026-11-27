#!/usr/bin/env python3
"""
Paper 1 — Generate Publication-Ready ACS Figures (Fig 2 & Fig 3) & LaTeX Tables (Tab 1 & Tab 3)
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


def make_fig2_origin_placebo(es_df: pd.DataFrame) -> None:
    """
    Fig 2: ACS Annual Event Study (2015-2024, ref=2022)
      Panel A: Overall LLM Exposure (OpenAI beta) — Non-English vs. English-Origin Placebo
      Panel B: Orthogonalized LLM_Writing — Non-English vs. English-Origin Placebo
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), sharey=False)

    panels = [
        (axes[0], "llm_openai_beta", "A. Overall LLM Exposure (OpenAI E1 + 0.5 E2)"),
        (axes[1], "llm_writing_orth", "B. Orthogonalized LLM Writing (Resid. on Coding/Math)"),
    ]

    for ax, task, title in panels:
        for grp, label, color, marker, ls, dx in [
            ("NonEng_vs_Native", "Non-English-Origin Immigrants vs. Natives", "#1b4f72", "o", "-", -0.10),
            ("Eng_vs_Native", "English-Origin Immigrants vs. Natives (Comparison)", "#c0392b", "s", "--", 0.10),
        ]:
            sub = es_df[(es_df["comparison"] == grp) & (es_df["task"] == task)].sort_values("year")
            ax.errorbar(
                sub["year"] + dx,
                sub["coef"],
                yerr=1.96 * sub["se"],
                fmt=f"{ls}{marker}",
                color=color,
                label=label,
                capsize=3.5,
                linewidth=1.8,
                markersize=6,
                zorder=4,
            )
        ax.axhline(0, color="#2c3e50", linestyle="-", linewidth=0.9, zorder=2)
        ax.axvline(2022.5, color="#7d3c98", linestyle=":", linewidth=1.6, zorder=3)
        ax.text(
            2022.58,
            ax.get_ylim()[1] * 0.88 if ax.get_ylim()[1] > 0 else 0.025,
            "ChatGPT\n(Nov 2022)",
            color="#7d3c98",
            fontsize=8.5,
            va="top",
            ha="left",
            fontweight="bold",
        )
        ax.set_xticks([2015, 2018, 2019, 2021, 2022, 2023, 2024])
        ax.set_xlabel("ACS Survey Year (Reference = 2022)")
        ax.set_ylabel("Immigrant × Task Exposure Effect on Log Hourly Wage")
        ax.set_title(title, loc="left", fontweight="bold", pad=10)
        ax.set_ylim(-0.033, 0.042)
        ax.legend(frameon=True, facecolor="white", framealpha=0.95, edgecolor="#dcdcdc", loc="upper left", fontsize=8.5)

    fig.suptitle(
        "ACS Event Study by Immigrant Origin: Non-English vs. English-Origin Immigrants (2015-2024)",
        fontsize=12.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / "fig2_acs_eng_vs_noneng_placebo.png", dpi=250)
    plt.close(fig)


def make_fig3_task_horserace(placebo_df: pd.DataFrame, hr_df: pd.DataFrame) -> None:
    """
    Fig 3: Task Decomposition & Origin-Placebo Comparison
      Panel A: Simultaneous Horse Race (Non-English Immigrants vs. Natives, 2015-2024 & 2021-2024)
      Panel B: Origin-Placebo Contrast across Tasks (2021-2024: Non-English vs. English-Origin)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))

    # Panel A: Simultaneous Horse Race (OrthWriting + CodingRaw + FaceToFace) + Overall OpenAI Beta
    ax = axes[0]
    cats = [
        ("OpenAI LLM β\n(Overall)", "OpenAI_Beta_vs_Social", "llm_openai_beta"),
        ("LLM Coding\n& Math", "OrthWriting_CodingRaw_Social", "llm_coding_math_raw"),
        ("LLM Writing\n(Orthogonalized)", "OrthWriting_CodingRaw_Social", "llm_writing_orth"),
        ("Face-to-Face\nSocial Skills", "OrthWriting_CodingRaw_Social", "face_to_face_social"),
    ]
    x = np.arange(len(cats))
    w = 0.34

    for idx_w, (win, col, offset) in enumerate([
        ("2015-2024", "#2471a3", -w / 2),
        ("2021-2024", "#17a589", w / 2),
    ]):
        vals, errs = [], []
        for _, sp, tc in cats:
            row = hr_df[(hr_df["window"] == win) & (hr_df["comparison"] == "NonEng_vs_Native") & (hr_df["spec"] == sp) & (hr_df["task"] == tc)].iloc[0]
            vals.append(row["coef"])
            errs.append(1.96 * row["se"])
        bars = ax.bar(x + offset, vals, width=w, yerr=errs, capsize=3.5, color=col, alpha=0.88, label=f"ACS {win}", zorder=3)
        for bx, v, e in zip(x + offset, vals, errs):
            y_txt = v + e + 0.0008 if v >= 0 else v - e - 0.0018
            ax.text(bx, y_txt, f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7.8, color="#2c3e50")

    ax.axhline(0, color="#2c3e50", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in cats], fontsize=9)
    ax.set_ylabel("Post-2022 × Non-English Imm. × Task (Log Wage)")
    ax.set_title("A. Simultaneous Task Horse Race (Non-English vs. Natives)", loc="left", fontweight="bold", pad=10)
    ax.set_ylim(-0.013, 0.0165)
    ax.legend(frameon=True, facecolor="white", edgecolor="#dcdcdc", loc="upper right", fontsize=8.5)

    # Panel B: Origin-Placebo Contrast across Tasks (2021-2024)
    ax = axes[1]
    p21 = placebo_df[placebo_df["window"] == "2021-2024"].set_index("task")
    cats_b = [
        ("OpenAI LLM β\n(Overall)", "llm_openai_beta"),
        ("LLM Coding\n& Math (Raw)", "llm_coding_math_raw"),
        ("LLM Writing\n(Orthogonalized)", "llm_writing_orth"),
        ("Face-to-Face\nSocial Skills", "face_to_face_social"),
    ]
    xb = np.arange(len(cats_b))
    ne_v = [p21.loc[t, "b_noneng"] for _, t in cats_b]
    ne_e = [1.96 * p21.loc[t, "se_noneng"] for _, t in cats_b]
    en_v = [p21.loc[t, "b_eng"] for _, t in cats_b]
    en_e = [1.96 * p21.loc[t, "se_eng"] for _, t in cats_b]

    ax.bar(xb - w / 2, ne_v, width=w, yerr=ne_e, capsize=3.5, color="#1b4f72", alpha=0.88, label="Non-English-Origin Immigrants", zorder=3)
    ax.bar(xb + w / 2, en_v, width=w, yerr=en_e, capsize=3.5, color="#c0392b", alpha=0.82, label="English-Origin Immigrants (Comparison)", zorder=3)

    for bx, v, e in zip(xb - w / 2, ne_v, ne_e):
        y_txt = v + e + 0.001 if v >= 0 else v - e - 0.0022
        ax.text(bx, y_txt, f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7.6, color="#1b4f72")
    for bx, v, e in zip(xb + w / 2, en_v, en_e):
        y_txt = v + e + 0.001 if v >= 0 else v - e - 0.0022
        ax.text(bx, y_txt, f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7.6, color="#c0392b")

    ax.axhline(0, color="#2c3e50", linewidth=0.9)
    ax.set_xticks(xb)
    ax.set_xticklabels([c[0] for c in cats_b], fontsize=9)
    ax.set_ylabel("Post-2022 × Immigrant × Task (Log Wage)")
    ax.set_title("B. Origin Comparison by Task (Post-COVID 2021-2024)", loc="left", fontweight="bold", pad=10)
    ax.set_ylim(-0.026, 0.036)
    ax.legend(frameon=True, facecolor="white", edgecolor="#dcdcdc", loc="upper right", fontsize=8.5)

    fig.suptitle(
        "Writing vs. Coding/Math vs. Face-to-Face Social Skills: Task Horse Race & Origin Comparison",
        fontsize=12.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / "fig3_writing_vs_coding_vs_social.png", dpi=250)
    plt.close(fig)


def make_tab1_origin_placebo(placebo_df: pd.DataFrame, ind_df: pd.DataFrame, counts: dict) -> None:
    """Write publication-grade LaTeX Table 1: ACS Individual-Worker (Authoritative) vs. Cell-Grouped Estimates."""
    order = [
        ("written_eng_only", "O*NET 31.0 Written-English Intensity"),
        ("llm_openai_beta", "OpenAI LLM Exposure ($E1 + 0.5\\,E2$, Eloundou 2024)"),
        ("llm_writing_raw", "LLM Writing (Raw)"),
        ("llm_writing_orth", "LLM Writing (Orthogonalized)"),
        ("llm_coding_math_raw", "LLM Coding \\& Math (Raw)"),
        ("llm_coding_math_orth", "LLM Coding \\& Math (Orthogonalized)"),
        ("face_to_face_social", "Face-to-Face Social Skills (O*NET 31.0)"),
        ("llm_felten_lm", "Felten Language-Modeling AIOE (2023)"),
    ]

    se_types = sorted(ind_df["se_type"].dropna().unique().tolist()) if "se_type" in ind_df.columns else ["twoway_occsoc_statefip"]
    se_note = "two-way CRV1 (OCCSOC + state) in all rows" if se_types == ["twoway_occsoc_statefip"] else f"logged SE types: {', '.join(se_types)}"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{ACS Occupational Exposure and Origin Comparison: Individual-Worker and Cell-Grouped Estimates (2015--2024)}",
        r"\label{tab:acs_origin_placebo}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4.0pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r" & (1) & (2) & (3) & (4) & (5) & (6) \\",
        r"Occupation Exposure Index & Non-Eng (Worker) & Cell-Grouped & 80\% MDE & English Origin & DDD Contrast & $p$-value \\",
        r"(Standardized, Mean 0, SD 1) & $\text{Post} \times \text{NonEng} \times T$ & (Non-Eng) & (Worker) & (Comparison) & (NonEng $-$ Eng) & (DDD $= 0$) \\",
        r"\midrule",
    ]

    for panel_idx, (win, _) in enumerate([
        ("2015-2024", ""),
        ("2021-2024", ""),
    ]):
        sub_cell = placebo_df[placebo_df["window"] == win].set_index("task")
        sub_ind = ind_df[ind_df["window"] == win].set_index("task")
        n_w = int(sub_cell.iloc[0]["n_workers"])
        n_c = int(sub_cell.iloc[0]["n_cells"])
        if panel_idx == 1:
            lines.append(r"\midrule")
        p_hdr = (
            f"\\textit{{Panel {'A' if panel_idx == 0 else 'B'}: "
            f"{'Full Sample (ACS 2015--2024' if panel_idx == 0 else 'Post-COVID Window (ACS 2021--2024'}; "
            f"$N = {n_w:,}$ individual workers; ${n_c:,}$ cells in col.\\ 2)}}"
        )
        lines.append(f"\\multicolumn{{7}}{{l}}{{{p_hdr}}} \\\\")
        lines.append(r"\addlinespace[3pt]")

        for tkey, tlab in order:
            if tkey not in sub_cell.index:
                continue
            rc = sub_cell.loc[tkey]
            if tkey in sub_ind.index:
                ri = sub_ind.loc[tkey]
                c_ne, se_ne, p_ne = ri["ind_b_noneng"], ri["ind_se_noneng"], ri["ind_p_noneng"]
                mde_val = ri["ind_mde80_noneng"]
                c_e, se_e, p_e = ri["ind_b_eng"], ri["ind_se_eng"], ri["ind_p_eng"]
                ddd, ddd_se, ddd_p = ri["ind_ddd_diff"], ri["ind_ddd_se"], ri["ind_ddd_p"]
                cell_c = fmt_coef(rc["b_noneng"], rc["p_noneng"])
                cell_se = fmt_se(rc["se_noneng"])
            else:
                # Non-headline derived rows (raw writing, orth coding): cell-grouped only
                c_ne, se_ne, p_ne = rc["b_noneng"], rc["se_noneng"], rc["p_noneng"]
                mde_val = rc["mde80_noneng"] if "mde80_noneng" in rc.index else 2.8016 * rc["se_noneng"]
                c_e, se_e, p_e = rc["b_eng"], rc["se_eng"], rc["p_eng"]
                ddd, ddd_se, ddd_p = rc["ddd_diff"], rc["ddd_se"], rc["ddd_p"]
                cell_c = "---"
                cell_se = ""
            lines.append(
                f"{tlab} & {fmt_coef(c_ne, p_ne)} & {cell_c} & {mde_val:.4f} & {fmt_coef(c_e, p_e)} "
                f"& {fmt_coef(ddd, ddd_p)} & {ddd_p:.3f} \\\\"
            )
            lines.append(
                f" & {fmt_se(se_ne)} & {cell_se} & & {fmt_se(se_e)} & {fmt_se(ddd_se)} & \\\\"
            )

    lines.extend([
        r"\midrule",
        r"Fixed Effects & \multicolumn{6}{c}{$\text{OCCSOC}\times\text{Year}$, $\text{State}\times\text{Year}$, $\text{Educ4}\times\text{Year}$, $\text{Origin}\times\text{Year}$, $\text{Sex}\times\text{Year}$} \\",
        r"Individual Controls & \multicolumn{6}{c}{Age, $\text{Age}^2$, Black, Asian, Hispanic} \\",
        f"Clusters (OCCSOC / State) & \\multicolumn{{6}}{{c}}{{{counts['n_occsoc_clusters']} occupations / {counts['n_state_clusters']} states ({se_note})}} \\\\",
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\vspace{4pt}",
        r"\begin{minipage}{\textwidth}\scriptsize",
        r"\textit{Notes:} Columns (1) and (3)--(6) report authoritative individual-level worker regressions estimated directly on ACS microdata rows ($N = 6{,}546{,}097$ in Panel A; $N = 3{,}840{,}043$ in Panel B) for all headline exposures; Column (2) reports the corresponding cell-grouped Non-English-origin estimate side by side (rows showing ``---'' in col.\ 2 are cell-grouped auxiliary decompositions). Because exact age, $\text{age}^2$, and race/ethnicity indicators vary within age-band cells, cell aggregation is not lossless, though individual-worker and cell-grouped estimates agree within $\le 0.00025$ log points (and $\le 0.0001$ in 2021--2024). Dependent variable is log real hourly wage (1999 dollars) for FTFY wage-and-salary workers aged 18--64, weighted by \texttt{PERWT}. O*NET 31.0 (2026 release) task scores are refreshed on rolling cycles; Eloundou et al.\ (2024) OpenAI $\beta$ was constructed in early 2023 based on GPT-4 capabilities. Column (3) reports the 80\%-power minimum detectable effect ($2.8016 \times \text{SE}$) for the occupational-exposure wage gradient. $\text{Post} = \mathbf{1}[\text{Year} \ge 2023]$ (note that 2023 ACS \texttt{INCWAGE} covers the preceding 12 months). Standard errors in parentheses. $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
        r"\end{minipage}",
        r"\end{table}",
    ])
    (TAB_DIR / "tab1_acs_origin_placebo.tex").write_text("\n".join(lines) + "\n")


def make_tab3_task_horserace(hr_df: pd.DataFrame, counts: dict) -> None:
    """Write publication-grade LaTeX Table 3: Simultaneous Task Horse Race."""
    cols = [
        ("2015-2024", "NonEng_vs_Native", "OrthWriting_CodingRaw_Social", "(1) Full\\\\2015--24"),
        ("2021-2024", "NonEng_vs_Native", "OrthWriting_CodingRaw_Social", "(2) Post-COVID\\\\2021--24"),
        ("2021-2024", "NonEng_vs_Native", "OpenAI_Beta_vs_Social", "(3) OpenAI $\\beta$\\\\2021--24"),
        ("2021-2024", "NonEng_vs_Native", "OpenAI_E1_vs_Social", "(4) OpenAI $E1$\\\\2021--24"),
        ("2021-2024", "NonEng_vs_Native", "Felten_LM_vs_Social", "(5) Felten LM\\\\2021--24"),
        ("2021-2024", "NonEng_vs_Eng_ImmiOnly", "OrthWriting_CodingRaw_Social", "(6) Immi-Only\\\\DDD 2021--24"),
    ]

    task_rows = [
        ("llm_writing_orth", "Post $\\times$ Treat $\\times$ LLM Writing (Orth.)"),
        ("llm_coding_math_raw", "Post $\\times$ Treat $\\times$ LLM Coding \\& Math"),
        ("llm_openai_beta", "Post $\\times$ Treat $\\times$ OpenAI LLM $\\beta$"),
        ("llm_openai_alpha", "Post $\\times$ Treat $\\times$ OpenAI LLM $E1$"),
        ("llm_felten_lm", "Post $\\times$ Treat $\\times$ Felten LM AIOE"),
        ("face_to_face_social", "Post $\\times$ Treat $\\times$ Face-to-Face Social"),
    ]

    def lookup(win, comp, spec, task):
        sub = hr_df[(hr_df["window"] == win) & (hr_df["comparison"] == comp) & (hr_df["spec"] == spec) & (hr_df["task"] == task)]
        if len(sub) == 0:
            return None
        return sub.iloc[0]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Simultaneous Task Horse Race: LLM Writing vs.\ LLM Coding/Math vs.\ Face-to-Face Social Skills}",
        r"\label{tab:task_horserace}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        " & " + " & ".join([f"\\shortstack{{{c[3]}}}" for c in cols]) + r" \\",
        r"\midrule",
    ]

    for tkey, tlab in task_rows:
        c_line = [tlab]
        s_line = [""]
        for win, comp, spec, _ in cols:
            r = lookup(win, comp, spec, tkey)
            if r is None:
                c_line.append("---")
                s_line.append("")
            else:
                c_line.append(fmt_coef(r["coef"], r["pval"]))
                s_line.append(fmt_se(r["se"]))
        lines.append(" & ".join(c_line) + r" \\")
        lines.append(" & ".join(s_line) + r" \\")

    # Summary stats rows
    n_w_row = ["Workers ($N$)"]
    n_c_row = ["Cells"]
    r2_row = ["$R^2$"]
    se_row = ["SE Clustering"]
    for win, comp, spec, _ in cols:
        sub = hr_df[(hr_df["window"] == win) & (hr_df["comparison"] == comp) & (hr_df["spec"] == spec)].iloc[0]
        n_w_row.append(f"{int(sub['n_workers']):,}")
        n_c_row.append(f"{int(sub['n_cells']):,}")
        r2_row.append(f"{sub['r2']:.3f}")
        st = str(sub["se_type"]) if "se_type" in sub.index else "twoway_occsoc_statefip"
        se_row.append("Two-way" if st.startswith("twoway") else "One-way (OCC)")

    lines.extend([
        r"\midrule",
        " & ".join(n_w_row) + r" \\",
        " & ".join(n_c_row) + r" \\",
        " & ".join(r2_row) + r" \\",
        " & ".join(se_row) + r" \\",
        r"Comparison Group & \multicolumn{5}{c}{Non-English-Origin Immigrants vs.\ U.S.\ Natives} & NonEng vs.\ Eng \\",
        r"Fixed Effects & \multicolumn{6}{c}{$\text{OCCSOC}\times\text{Year}$, $\text{State}\times\text{Year}$, $\text{Educ4}\times\text{Year}$, $\text{Origin}\times\text{Year}$, $\text{Sex}\times\text{Year}$} \\",
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\vspace{4pt}",
        r"\begin{minipage}{\textwidth}\scriptsize",
        r"\textit{Notes:} Each column estimates a single simultaneous regression competing multiple occupation task exposures interacted with $\text{Treat}$ and $\text{Post} = \mathbf{1}[\text{Year}\ge 2023]$. Columns (1)--(5) compare Non-English-origin immigrants to U.S.\ natives; Column (6) restricts the sample to foreign-born immigrants only (Non-English-origin vs.\ English-origin). Because \texttt{LLM Writing (Orth.)} is the OLS residual of \texttt{LLM Writing} on \texttt{LLM Coding \& Math}, it has exact zero correlation with \texttt{LLM Coding \& Math} across occupations. Exact logged SE clustering (\texttt{OCCSOC} + state two-way CRV1, or one-way \texttt{OCCSOC} if a finite-sample two-way diagonal is non-positive) is reported explicitly for each column. $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
        r"\end{minipage}",
        r"\end{table}",
    ])
    (TAB_DIR / "tab3_task_horserace.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    placebo_df = pd.read_csv(RES_DIR / "acs_origin_placebo.csv")
    ind_df = pd.read_csv(RES_DIR / "acs_individual_vs_cell_comparison.csv")
    es_df = pd.read_csv(RES_DIR / "acs_event_studies.csv")
    hr_df = pd.read_csv(RES_DIR / "acs_task_horserace.csv")
    summary = json.loads((RES_DIR / "acs_summary.json").read_text())

    make_fig2_origin_placebo(es_df)
    make_fig3_task_horserace(placebo_df, hr_df)
    make_tab1_origin_placebo(placebo_df, ind_df, summary["sample_counts"])
    make_tab3_task_horserace(hr_df, summary["sample_counts"])
    print("Generated Fig 2, Fig 3, Tab 1, and Tab 3 successfully!")


if __name__ == "__main__":
    main()
