"""Build the two reproducible research notebooks for the revised Greater Cumberland
Plateau Forest, Fire, and 3DEP Airborne LiDAR study:
  notebooks/01_optical_audit_and_region.ipynb  - Optical harmonization audit, bug-fixed
      regional & 8-area Tennessee metrics (with complete Fall Creek Falls & km2 units)
  notebooks/02_lidar_fire_structure.ipynb      - Multi-epoch 3DEP Airborne LiDAR x USFS
      FACTS prescribed fire across 689 pure-deciduous plots (AL, TN, KY)

Reads cached JSON/PNG outputs so execution is fast and deterministic.
"""

from __future__ import annotations

import os

import nbformat as nbf


def md(cells, text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(cells, text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


PREAMBLE = r"""
import json, os
import pandas as pd
from IPython.display import Image, display

ROOT = ".." if os.path.basename(os.getcwd()) == "notebooks" else "."
def load(path):
    with open(os.path.join(ROOT, path)) as f:
        return json.load(f)
def show(path, w=780):
    fp = os.path.join(ROOT, path)
    if os.path.exists(fp):
        display(Image(fp, width=w))
    else:
        print("Missing figure:", fp)
"""


def build_nb1() -> nbf.NotebookNode:
    c = []
    md(c, r"""
# Notebook 1: Optical Audit & Calibrated Greater Cumberland Plateau Baseline

This notebook documents the methodological audit and full bug-fix rebuild of the
Greater Cumberland Plateau optical study:
1. Quantifying the Landsat Collection 2 inter-sensor winter step-jump (`L5 -> L7 -> L8/9`,
   LEDAPS vs. LaSRC) and its 79% collapse of apparent 40-year "mesophication".
2. Attributing the residual single-sensor Landsat 5 (1985-2011) deciduous winter slope
   using Annual NLCD stand-age cohorts, Daymet V4 winter climate, and SWIR indices (NBR/NDMI).
3. Rebuilding all 8 Tennessee comparison units with the complete Fall Creek Falls State Park
   boundary (10,282 ha, replacing the truncated 2,946 ha WDPA WMA sub-unit), consistent
   `>= 430 m` plateau-top masking, Annual NLCD Collection 1.0 (1985-2024), and corrected
   `km^2` loss units (fixing the v1 100x `ha` vs. `km^2` display bug).
""")
    code(c, PREAMBLE)

    md(c, "## 1. Inter-Sensor Overlap Offsets & Regional Optical Collapse")
    code(c, r"""
D = load("data/diagnostic_optical.json")
R = load("data/region_metrics.json")

print("=== Regional Pure-Deciduous Inter-Sensor Overlaps ===")
display(pd.DataFrame(D["regional_deciduous_sensor_overlaps"]).T)

print("\n=== Regional Plateau-Top (>=430 m) Trends (/decade) ===")
display(pd.Series(R["trends"]).rename("slope_per_decade").to_frame())

show("paper/figures/fig1_optical_illusion_vs_lidar.png")
""")

    md(c, "## 2. Residual Landsat 5 (1985-2011) Attribution & Sewanee Burn-Cohort Persistence")
    code(c, r"""
l5d = D["l5_residual_attribution"]
print("=== Stand-Age & Forest-Type Cohorts (Annual NLCD 1985 -> 2001 -> 2021) ===")
display(pd.DataFrame(l5d["cohorts"]).T)

print("\n=== Spectral Index Contrast on Old Deciduous (1985-2021) Jan-Feb ===")
display(pd.DataFrame(l5d["indices_old_decid"]).T)

print("\n=== Sewanee Burn-Cohort Persistence (Plateau-Top >=530 m Pure Deciduous) ===")
display(pd.DataFrame(D["sewanee_burn_persistence"]).T)
""")

    md(c, "## 3. Bug-Fixed Regional Forest Cover, Gross Loss (km^2/yr), and 8 Protected Areas")
    code(c, r"""
print(f"Regional window: {R['region_km2']:,} km^2 | Plateau top (>=430 m): {R['plateau_km2']:,} km^2")
print(f"Annual NLCD forest (km^2): {R['annual_nlcd_forest_km2']}")
print(f"Hansen gross loss 2001-2025: {R['loss_km2']:,.1f} km^2 ({R['pct_loss']}% of standing forest)")
print(f"NLCD net change 2001->2024: {R['nlcd_net_2001_2024_km2']:+.1f} km^2 ({R['nlcd_net_2001_2024_pct']:+.2f}%)")

# Fixed km^2/yr table (explicitly using loss_by_year_km2 = loss_by_year_ha / 100)
s_km2 = pd.Series({int(k): v for k, v in R["loss_by_year_km2"].items()}).sort_index()
print(f"Sum of annual loss_by_year_km2 = {s_km2.sum():.1f} km^2 (matches loss_km2 = {R['loss_km2']:.1f} km^2)")
display(s_km2.rename("km2_lost_per_yr").to_frame().T)

A = load("data/area_metrics.json")
cols = ["total_ha", "pct_plateau", "nlcd_pct_deciduous", "nlcd_pct_evergreen",
        "unharm_dec_mar_decade", "harm_jan_feb_decid_decade",
        "l5_jan_feb_decid_decade", "l7_jan_feb_decid_decade", "l89_jan_feb_decid_decade", "mtbs_fire_count"]
display(pd.DataFrame(A).T[cols].sort_values("harm_jan_feb_decid_decade"))
show("paper/figures/fig4_sensor_comparison_and_tn_reversal.png")
""")

    nb = nbf.v4.new_notebook()
    nb.cells = c
    return nb


def build_nb2() -> nbf.NotebookNode:
    c = []
    md(c, r"""
# Notebook 2: 3DEP Airborne LiDAR x USFS FACTS Prescribed Fire Across the Cumberland Plateau

This notebook reproduces the structural and sensor-validation core of the study:
1. **689 pure-deciduous upland plots** (Annual NLCD 41 in 2001, 2010, 2021; Hansen `treecover2000 >= 60%`, `loss == 0`) spanning **Bankhead NF (AL)**, **Sewanee / Franklin SF (TN)**, and **Daniel Boone NF (KY)**.
2. **Authoritative fire & mechanical records** from USFS FACTS Completed Hazardous Fuels Treatments (Layer 3 Fire & Layer 5 Mechanical) and Sewanee LAL `burns.geojson`.
3. **Multi-epoch BACI 3DEP Airborne LiDAR differencing** (`2012 -> 2021` on Bankhead North, `n = 139`; `2017 -> 2023` on Daniel Boone East, `n = 144` uniform-baseline pairs), with return-density thinning, flight `frac_first_returns` diagnostics, baseline-adjusted ANCOVA, and exact mechanical timing checks.
4. **Continuous 1 m vertical profiles (`0.5-10 m`), unbiased first-return canopy cover (`>10 m`), direct cluster-robust `mech - 3+` contrasts, and within-site + temporal BACI sensor validation** (Landsat winter & spring NDVI vs. GEDI L2B).
""")
    code(c, PREAMBLE)

    md(c, "## 1. Multi-Epoch BACI Structural Change, Flight Diagnostics & ANCOVA")
    code(c, r"""
S = load("data/lidar/analysis_summary.json")
print("Total pure-deciduous plots:", S["n_total_plots"])
print("Plots by site:", S["plots_by_site"])
print("Plots by burn category:", S["plots_by_burn_cat"])

for site, bdata in S["baci"].items():
    fd = bdata.get("flight_diagnostics", {})
    print(f"\n=== BACI: {site} (n={bdata['n_total']}, excluded={bdata.get('n_excluded_off_epoch', 0)}, G_suid={bdata.get('n_clusters_suid')}, frac_first={fd.get('baseline_mean_frac_first_returns')}->{fd.get('primary_mean_frac_first_returns')}) ===")
    rows = []
    for st, d in bdata["strata"].items():
        dc = d.get("did_vs_control", {})
        rows.append({
            "stratum": st,
            "n": d["n"],
            "G_suid": d.get("n_clusters_suid"),
            "inter_mech": d.get("n_inter_mech_plots"),
            "base_mid": d["baseline_nrd_midstory"]["mean"],
            "end_mid": d["primary_nrd_midstory"]["mean"],
            "delta_mid": d["delta_nrd_midstory"]["mean"],
            "DiD_mid": dc.get("did_midstory"),
            "DiD_p_suid": dc.get("p_cluster_suid"),
            "DiD_p_wcb": dc.get("p_wcb_suid"),
            "DiD_thinned": dc.get("did_midstory_thinned"),
            "DiD_ancova": dc.get("did_midstory_ancova"),
            "p_ancova_suid": dc.get("p_ancova_suid"),
            "DiD_winNDVI": dc.get("did_opt_win_ndvi"),
            "DiD_sprNDVI": dc.get("did_opt_spr_ndvi"),
            "DiD_sprDB": dc.get("did_opt_spr_ndvi_db_epoch"),
        })
    display(pd.DataFrame(rows).set_index("stratum"))

show("paper/figures/fig2_baci_lidar_structural_change.png")
""")

    md(c, "## 2. Continuous 1 m Vertical Profile, Unbiased First-Return Canopy & Direct Contrasts")
    code(c, r"""
ols = S["pooled_fe_ols"]
rows = []
for target in ["p_nrd_understory", "p_nrd_low_midstory", "p_nrd_tall_midstory",
               "p_nrd_midstory", "p_first_return_canopy_10m", "p_canopy_cover_10m",
               "p_chm_p95", "p_fhd", "opt_win_ndvi", "opt_spr_ndvi"]:
    m = ols[target]
    cdict = m["coefs"]
    ct = m["contrasts"]["mech_minus_3plus"]
    rows.append({
        "target": target,
        "R2": m["r2"],
        "1_burn": f"{cdict['burn_1']['coef']:+.3f} (p={cdict['burn_1']['p_cluster_suid']:.3f})",
        "2_burns": f"{cdict['burn_2']['coef']:+.3f} (p={cdict['burn_2']['p_cluster_suid']:.3f})",
        "3plus_burns": f"{cdict['burn_3plus']['coef']:+.3f} (p={cdict['burn_3plus']['p_cluster_suid']:.3f})",
        "mech_plus_fire": f"{cdict['mech_plus_fire']['coef']:+.3f} (p={cdict['mech_plus_fire']['p_cluster_suid']:.3f})",
        "mech_minus_3plus": f"{ct['diff']:+.3f} (p={ct['p_cluster_suid']:.3f})",
    })
display(pd.DataFrame(rows).set_index("target"))

print("\n=== Continuous 1 m Vertical NRD Profile (0.5-10 m) ===")
p1m_rows = []
for lbl, m in S["pooled_fe_ols_1m_profile"].items():
    c = m["coefs"]
    ct = m["contrasts"]["mech_minus_3plus"]
    p1m_rows.append({
        "band_m": lbl,
        "3plus": c["burn_3plus"]["coef"],
        "p_3plus": c["burn_3plus"]["p_cluster_suid"],
        "mech": c["mech_plus_fire"]["coef"],
        "p_mech": c["mech_plus_fire"]["p_cluster_suid"],
        "mech_minus_3plus": ct["diff"],
        "p_contrast_suid": ct["p_cluster_suid"],
    })
display(pd.DataFrame(p1m_rows).set_index("band_m"))
show("paper/figures/fig3_vertical_stratum_decoupling.png")
""")

    md(c, "## 3. Head-to-Head Sensor Validation (Pooled, Within-Site, and BACI Temporal Deltas)")
    code(c, r"""
sc = S["sensor_comparison"]
print("=== Cross-Sectional Pooled vs. Within-Site Correlations with 3DEP Midstory ===")
cs_df = pd.DataFrame({
    k: {
        "n": v["n"],
        "pooled_r": v["pearson_r_midstory"],
        "within_site_r": v["within_site_r_midstory"],
        "within_site_p": v["within_site_p_midstory"],
        "spearman_rho": v["spearman_rho_midstory"],
    }
    for k, v in sc["cross_sectional"].items()
}).T
display(cs_df)

print("\n=== Multi-Epoch BACI Temporal Delta Correlations (Winter & Spring Optical vs. 3DEP Midstory) ===")
bd_df = pd.DataFrame({
    k: {
        "n": v["n"],
        "pooled_r": v["pearson_r_d_midstory"],
        "pooled_p": v["p_d_midstory"],
        "within_site_r": v["within_site_r_d_midstory"],
        "within_site_p": v["within_site_p_d_midstory"],
    }
    for k, v in sc["baci_deltas"].items()
}).T
display(bd_df)
""")

    nb = nbf.v4.new_notebook()
    nb.cells = c
    return nb


def main() -> None:
    os.makedirs("notebooks", exist_ok=True)
    nbf.write(build_nb1(), "notebooks/01_optical_audit_and_region.ipynb")
    nbf.write(build_nb2(), "notebooks/02_lidar_fire_structure.ipynb")
    print("Wrote notebooks/01_optical_audit_and_region.ipynb and notebooks/02_lidar_fire_structure.ipynb")


if __name__ == "__main__":
    main()
