#!/usr/bin/env python3
"""
Build harmonized occupation task, O*NET skill, and AI/LLM exposure lookup tables
for Essays 1 & 2.

Outputs:
  1. data/aux/occ1990dd_tasks.parquet       (330 occ1990dd occupations: ALM tasks, RTI, Song routine flag, offshorability)
  2. data/aux/dorn_occ_crosswalks.parquet   (Unified 1970/1980/1990/2000/2010 Census occ -> occ1990dd crosswalk)
  3. data/aux/occsoc_ai_onet_tasks.parquet  (6-digit SOC: OpenAI GPTs-are-GPTs, Felten AIOE & LM, O*NET 31.0 task indices)
  4. data/aux/clean_occ_tasks.parquet       (Alias/copy of occsoc_ai_onet_tasks.parquet for convenience)
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUX = ROOT / "data" / "aux"


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)


def build_occ1990dd_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    alm = pd.read_stata(AUX / "dorn" / "occ1990dd_task_alm.dta")
    off = pd.read_stata(AUX / "dorn" / "occ1990dd_task_offshore.dta")

    alm["occ1990dd"] = alm["occ1990dd"].round().astype(int)
    off["occ1990dd"] = off["occ1990dd"].round().astype(int)

    df = alm.merge(off, on="occ1990dd", how="left", validate="one_to_one")
    assert df.isna().sum().sum() == 0, f"Unexpected NaNs in occ1990dd merge: {df.isna().sum()}"

    # Exact Autor & Dorn (2013) RTI: ln(routine) - ln(manual) - ln(abstract)
    # Floor exact zeros / tiny values at 0.01 to prevent -inf / extreme log outliers
    floor = 0.01
    df["rti"] = (
        np.log(df["task_routine"].clip(lower=floor))
        - np.log(df["task_manual"].clip(lower=floor))
        - np.log(df["task_abstract"].clip(lower=floor))
    )
    df["rti_std"] = zscore(df["rti"])
    df["task_routine_std"] = zscore(df["task_routine"])
    df["task_abstract_std"] = zscore(df["task_abstract"])
    df["task_manual_std"] = zscore(df["task_manual"])
    df["task_offshorability_std"] = zscore(df["task_offshorability"])

    # Unweighted top-1/3 routine occupation indicator across 330 occ1990dd codes
    rti_p67 = df["rti"].quantile(2.0 / 3.0)
    df["is_routine_top33"] = (df["rti"] >= rti_p67).astype(int)

    # Song (2017) Footnote 7 coarse routine categorization:
    # Administrative support (303-389), retail sales (274-285),
    # precision production & craft (503-699), machine operators & assemblers (703-799)
    occ = df["occ1990dd"]
    df["is_routine_song"] = (
        occ.between(303, 389)
        | occ.between(274, 285)
        | occ.between(503, 699)
        | occ.between(703, 799)
    ).astype(int)

    df = df.sort_values("occ1990dd").reset_index(drop=True)

    # Unified Dorn crosswalk table (1970, 1980, 1990, 2000, 2010 -> occ1990dd)
    cw_frames = []
    for yr in [1970, 1980, 1990, 2000, 2010]:
        cw = pd.read_stata(AUX / "dorn" / f"occ{yr}_occ1990dd.dta").copy()
        if cw["occ"].dtype == object:
            cw["occ_str"] = cw["occ"].astype(str).str.strip().str.zfill(4)
            cw["occ_code"] = cw["occ_str"].astype(int)
        else:
            cw["occ_code"] = cw["occ"].round().astype(int)
            cw["occ_str"] = cw["occ_code"].astype(str).str.zfill(4)
        cw["occ1990dd"] = cw["occ1990dd"].round().astype(int)
        cw["source_year"] = yr
        cw["in_alm"] = cw["occ1990dd"].isin(df["occ1990dd"]).astype(int)
        cw_frames.append(cw[["source_year", "occ_code", "occ_str", "occ1990dd", "in_alm"]])

    cw_all = pd.concat(cw_frames, ignore_index=True)
    return df, cw_all


def build_occsoc_table() -> pd.DataFrame:
    # 1. O*NET Occupation Data (master titles & 6-digit SOC universe)
    occ_data = pd.read_csv(AUX / "onet" / "Occupation Data.txt", sep="\t")
    occ_data["occsoc_hyphen"] = occ_data["O*NET-SOC Code"].str.slice(0, 7)
    # Prefer the .00 base occupation title when multiple 8-digit codes map to one 6-digit SOC
    occ_data["is_base"] = occ_data["O*NET-SOC Code"].str.endswith(".00").astype(int)
    titles = (
        occ_data.sort_values(["occsoc_hyphen", "is_base", "O*NET-SOC Code"], ascending=[True, False, True])
        .drop_duplicates("occsoc_hyphen")[["occsoc_hyphen", "Title"]]
        .rename(columns={"Title": "occ_title"})
    )

    # 2. O*NET Work Activities & Abilities -> compute IM * LV composite per (O*NET-SOC, Element)
    wa = pd.read_csv(AUX / "onet" / "Work Activities.txt", sep="\t")
    ab = pd.read_csv(AUX / "onet" / "Abilities.txt", sep="\t")

    def pivot_im_lv(raw: pd.DataFrame) -> pd.DataFrame:
        sub = raw[raw["Scale ID"].isin(["IM", "LV"])][
            ["O*NET-SOC Code", "Element Name", "Scale ID", "Data Value"]
        ]
        piv = sub.pivot_table(
            index=["O*NET-SOC Code", "Element Name"],
            columns="Scale ID",
            values="Data Value",
            aggfunc="mean",
        ).reset_index()
        # Geometric-like importance * level score (standard in Felten et al. & O*NET literature)
        piv["score"] = piv["IM"] * piv["LV"]
        piv["occsoc_hyphen"] = piv["O*NET-SOC Code"].str.slice(0, 7)
        # Average across 8-digit O*NET-SOC into 6-digit SOC
        wide = piv.pivot_table(
            index="occsoc_hyphen",
            columns="Element Name",
            values="score",
            aggfunc="mean",
        )
        return wide

    wa_wide = pivot_im_lv(wa)
    ab_wide = pivot_im_lv(ab)
    onet_all = wa_wide.join(ab_wide, how="outer")

    # Define O*NET Deming-style & communication task groups
    social_cols = [
        "Communicating with Supervisors, Peers, or Subordinates",
        "Communicating with People Outside the Organization",
        "Establishing and Maintaining Interpersonal Relationships",
        "Selling or Influencing Others",
        "Resolving Conflicts and Negotiating with Others",
        "Performing for or Working Directly with the Public",
        "Coordinating the Work and Activities of Others",
        "Coaching and Developing Others",
        "Developing and Building Teams",
    ]
    english_written_cols = ["Written Expression", "Written Comprehension"]
    english_oral_cols = ["Oral Expression", "Oral Comprehension", "Speech Clarity", "Speech Recognition"]
    math_analytical_cols = [
        "Mathematical Reasoning",
        "Number Facility",
        "Deductive Reasoning",
        "Inductive Reasoning",
        "Analyzing Data or Information",
        "Processing Information",
    ]
    routine_physical_cols = [
        "Performing General Physical Activities",
        "Handling and Moving Objects",
        "Controlling Machines and Processes",
        "Operating Vehicles, Mechanized Devices, or Equipment",
        "Manual Dexterity",
        "Finger Dexterity",
        "Arm-Hand Steadiness",
        "Static Strength",
        "Stamina",
    ]

    onet_idx = pd.DataFrame(index=onet_all.index)
    onet_idx["onet_social_skill"] = zscore(onet_all[social_cols].mean(axis=1))
    onet_idx["onet_english_written"] = zscore(onet_all[english_written_cols].mean(axis=1))
    onet_idx["onet_english_oral"] = zscore(onet_all[english_oral_cols].mean(axis=1))
    onet_idx["onet_english_comm"] = zscore(
        onet_all[english_written_cols + english_oral_cols].mean(axis=1)
    )
    onet_idx["onet_math_analytical"] = zscore(onet_all[math_analytical_cols].mean(axis=1))
    onet_idx["onet_routine_physical"] = zscore(onet_all[routine_physical_cols].mean(axis=1))
    onet_idx["onet_computer_use"] = zscore(onet_all["Working with Computers"])

    # 3. Felten et al. (2023) Ability-Weighted Language Modeling Exposure from Appendix D
    app_d = pd.read_excel(AUX / "ai_exposure" / "AIOE_DataAppendix.xlsx", sheet_name="Appendix D")
    app_d = app_d.rename(columns={"Unnamed: 0": "Element Name"})
    app_d["Element Name"] = app_d["Element Name"].replace(
        {"Visual Color Determination": "Visual Color Discrimination"}
    )
    app_d = app_d.set_index("Element Name")
    common_ab = [c for c in ab_wide.columns if c in app_d.index]
    assert len(common_ab) == 52, f"Expected 52 O*NET abilities matching Appendix D, got {len(common_ab)}"

    ab_mat = ab_wide[common_ab]
    ab_denom = ab_mat.sum(axis=1).replace(0, np.nan)
    lm_weights = app_d.loc[common_ab, "Language Modeling"]
    lang_suite_weights = app_d.loc[
        common_ab, ["Language Modeling", "Reading Comprehension", "Translation", "Speech Recognition"]
    ].mean(axis=1)

    onet_idx["felten_aioe_lm"] = zscore((ab_mat @ lm_weights) / ab_denom)
    onet_idx["felten_aioe_lang_suite"] = zscore((ab_mat @ lang_suite_weights) / ab_denom)
    onet_idx = onet_idx.reset_index()

    # 4. Felten et al. Appendix A (6-digit SOC AIOE)
    felten_a = pd.read_excel(AUX / "ai_exposure" / "AIOE_DataAppendix.xlsx", sheet_name="Appendix A")
    felten_a = felten_a.rename(columns={"SOC Code": "occsoc_hyphen", "AIOE": "felten_aioe"})
    felten_a["occsoc_hyphen"] = felten_a["occsoc_hyphen"].astype(str).str.strip()
    felten_a = felten_a.groupby("occsoc_hyphen", as_index=False)["felten_aioe"].mean()

    # 5. OpenAI "GPTs are GPTs" (Eloundou et al. 2024) averaged from 8-digit O*NET-SOC to 6-digit SOC
    openai = pd.read_csv(AUX / "ai_exposure" / "occ_level.csv")
    openai["occsoc_hyphen"] = openai["O*NET-SOC Code"].astype(str).str.strip().str.slice(0, 7)
    oai_cols = [
        "dv_rating_alpha",
        "dv_rating_beta",
        "dv_rating_gamma",
        "human_rating_alpha",
        "human_rating_beta",
        "human_rating_gamma",
    ]
    oai_6 = openai.groupby("occsoc_hyphen", as_index=False)[oai_cols].mean()
    oai_6 = oai_6.rename(
        columns={
            "dv_rating_alpha": "openai_dv_alpha",
            "dv_rating_beta": "openai_dv_beta",
            "dv_rating_gamma": "openai_dv_gamma",
            "human_rating_alpha": "openai_human_alpha",
            "human_rating_beta": "openai_human_beta",
            "human_rating_gamma": "openai_human_gamma",
        }
    )
    oai_6["openai_dv_beta_std"] = zscore(oai_6["openai_dv_beta"])
    oai_6["openai_human_beta_std"] = zscore(oai_6["openai_human_beta"])

    # 6. Merge all 6-digit SOC sources
    soc = (
        titles.merge(onet_idx, on="occsoc_hyphen", how="outer")
        .merge(oai_6, on="occsoc_hyphen", how="outer")
        .merge(felten_a, on="occsoc_hyphen", how="outer")
    )

    # Backfill titles for any SOC only in OpenAI or Felten
    oai_titles = (
        openai.drop_duplicates("occsoc_hyphen")
        .set_index("occsoc_hyphen")["Title"]
    )
    felten_titles = (
        pd.read_excel(AUX / "ai_exposure" / "AIOE_DataAppendix.xlsx", sheet_name="Appendix A")
        .assign(occsoc_hyphen=lambda d: d["SOC Code"].astype(str).str.strip())
        .drop_duplicates("occsoc_hyphen")
        .set_index("occsoc_hyphen")["Occupation Title"]
    )
    soc["occ_title"] = (
        soc["occ_title"]
        .fillna(soc["occsoc_hyphen"].map(oai_titles))
        .fillna(soc["occsoc_hyphen"].map(felten_titles))
    )

    # Normalized IPUMS OCCSOC format (no hyphen, e.g. '111011') + 2-digit major group
    soc["occsoc"] = soc["occsoc_hyphen"].str.replace("-", "", regex=False)
    soc["soc_major_2d"] = soc["occsoc_hyphen"].str.slice(0, 2)

    # Reorder columns cleanly
    front = ["occsoc", "occsoc_hyphen", "soc_major_2d", "occ_title"]
    rest = [c for c in soc.columns if c not in front]
    soc = soc[front + rest].sort_values("occsoc").reset_index(drop=True)
    return soc


def main() -> None:
    occ1990dd_df, cw_df = build_occ1990dd_tables()
    soc_df = build_occsoc_table()

    # Save parquet outputs
    occ1990dd_df.to_parquet(AUX / "occ1990dd_tasks.parquet", index=False)
    cw_df.to_parquet(AUX / "dorn_occ_crosswalks.parquet", index=False)
    soc_df.to_parquet(AUX / "occsoc_ai_onet_tasks.parquet", index=False)
    soc_df.to_parquet(AUX / "clean_occ_tasks.parquet", index=False)

    print("=== 1. occ1990dd_tasks.parquet ===")
    print(f"Shape: {occ1990dd_df.shape}, NaNs: {occ1990dd_df.isna().sum().sum()}")
    print(occ1990dd_df[["task_routine", "task_abstract", "task_manual", "rti", "is_routine_top33", "is_routine_song", "task_offshorability"]].describe().round(3))
    print(f"\nAgreement between is_routine_top33 and is_routine_song:\n{pd.crosstab(occ1990dd_df['is_routine_top33'], occ1990dd_df['is_routine_song'])}")

    print("\n=== 2. dorn_occ_crosswalks.parquet ===")
    print(cw_df.groupby("source_year").agg(n_codes=("occ_code", "count"), n_in_alm=("in_alm", "sum")))

    print("\n=== 3. occsoc_ai_onet_tasks.parquet (+ clean_occ_tasks.parquet) ===")
    print(f"Shape: {soc_df.shape}")
    key_cols = [
        "openai_dv_beta", "openai_human_beta", "felten_aioe", "felten_aioe_lm",
        "onet_social_skill", "onet_english_written", "onet_english_oral",
        "onet_math_analytical", "onet_routine_physical", "onet_computer_use"
    ]
    coverage = pd.DataFrame({
        "non_null": soc_df[key_cols].notna().sum(),
        "pct_cov": (soc_df[key_cols].notna().mean() * 100).round(1),
        "mean": soc_df[key_cols].mean().round(3),
        "std": soc_df[key_cols].std().round(3),
    })
    print(coverage)

    print("\n=== Sanity Check: Top 5 & Bottom 5 Occupations per Index ===")
    for col in [
        "openai_dv_beta",
        "felten_aioe_lm",
        "onet_social_skill",
        "onet_english_written",
        "onet_math_analytical",
        "onet_routine_physical",
    ]:
        sub = soc_df[["occsoc_hyphen", "occ_title", col]].dropna()
        top5 = sub.nlargest(5, col)
        bot5 = sub.nsmallest(5, col)
        print(f"\n--- {col} ---")
        print("  TOP 5:")
        for _, r in top5.iterrows():
            print(f"    {r['occsoc_hyphen']} | {r[col]:+.3f} | {r['occ_title']}")
        print("  BOTTOM 5:")
        for _, r in bot5.iterrows():
            print(f"    {r['occsoc_hyphen']} | {r[col]:+.3f} | {r['occ_title']}")

    print("\n=== Correlation Matrix Across Key 6-Digit SOC Indices ===")
    corr_cols = [
        "openai_dv_beta", "openai_human_beta", "felten_aioe", "felten_aioe_lm",
        "onet_social_skill", "onet_english_written", "onet_english_oral",
        "onet_math_analytical", "onet_routine_physical"
    ]
    print(soc_df[corr_cols].corr().round(3))


if __name__ == "__main__":
    main()
