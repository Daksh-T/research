"""Build, execute, and export the revised Escarpment_Hopkins_Karst_v2.ipynb notebook.

Reads strictly from verified Stage-3 (profiling/p1..p6.json) and Stage-4 (data/v2_*.json)
outputs and embeds all 6 main publication figures plus the standalone supplementary AlphaEarth audit figure.
"""
import os
import subprocess
import sys
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
NB_PATH = os.path.join(HERE, 'Escarpment_Hopkins_Karst_v2.ipynb')
HTML_PATH = os.path.join(HERE, 'Escarpment_Hopkins_Karst_v2.html')


def md(text):
    return nbf.v4.new_markdown_cell(text.strip() + '\n')


def code(src):
    return nbf.v4.new_code_cell(src.strip() + '\n')


def build_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3',
    }
    cells = []

    # ── TITLE & ABSTRACT ────────────────────────────────────────────────────────
    cells.append(md("""
# Where Hopkins' 1918 Bioclimatic Law Bends: Closed-Karst Cold-Air Pooling and Plateau-Cap Phenology on the Southern Cumberland Escarpment

**Daksh Tyagi** — Reproducible Multi-Sensor Notebook (2000–2026)

*AI-assistance note: Code refactoring, statistical verification, and manuscript revision assistance were performed with Gemini 4 Neon under author direction.*

---

### Summary

Hopkins' (1918) bioclimatic law predicts that deciduous spring green-up lags by **3.28 days per 100 m of elevation**. Combining `3DEP` Priority-Flood basin-closure geometry, topographically corrected Sentinel-2 (`SCS+C`), BRDF-normalized NASA `HLS v2.0`, 25-year harmonized Landsat 5–9 (`2000–2024`), cloud-leak-filtered `ECOSTRESS v002 + v003` thermal infrared (`157` in-swath / `52` filtered clean nights across `7` karst basins in tile `16SED`), `POLARIS`/`gNATSGO` soils, public Sewanee Landscape Analysis Lab (`LAL`) timber-harvest polygons (`120` polygons), and raw `NOAA USC00408184` records yields four main conclusions with explicit caveats:

1. **MODIS 50% Mid-Greenup Matches Hopkins (1918), While 15% Onset Lapses Faster:** On `MODIS MCD12Q2 v061` (`2001–2023`, `QA_Overall_1 <= 1`, `NLCD-41`), half-amplitude `MidGreenup_1` (50%) averages **`3.19 ± 0.17 SE` (`0.82 SD`) d/100 m** (matching Hopkins' `3.28`), with no secular trend (`−0.05 d/100 m/dec, p = 0.86`). A 3-factor GEE decomposition shows that QA filtering (`0.00`) and deciduous masking (`+0.01`) contribute virtually nothing; switching from 15% early onset (`Greenup_1`, `3.98 ± 0.24 SE` d/100 m) to 50% `MidGreenup_1` accounts for `−0.80 d/100 m`. In unharmonized 1984–2024 Landsat Collection 2, the 2013 `LEDAPS → LaSRC` step (`−0.42 ± 3.69 SE d, p = 0.91`) and linear slope (`−1.58 ± 1.46 SE d/dec, p = 0.29`) are collinearly non-identifiable (`r = 0.80, VIF = 2.7`).
2. **Closed-Karst Growing-Season Compression (`−9.1 ± 1.8 SE d` Relief-Adjusted; `−8.7 ± 1.9 SE d` Rim-Free):** Across **11 forested closed-karst basins** paired 1:1 with open-drainage gorges across TN/AL (after excluding 6 non-forested/quarry basins including `Lost_Cove_TN` and `Sinking_Cove_TN` below the 60% `NLCD-41` cutoff), open gorges have `51.8 m` greater floor-to-rim relief (`178.3` vs. `126.5 m, p = 0.0043`). Adjusting 22-site NDRE regressions for relief with CR1 pair-clustered SEs (`df = 10`) yields **`ΔSOS = +4.28 ± 1.13 SE d (p = 0.0035)`**, **`ΔEOS = −4.79 ± 1.13 SE d (p = 0.0017)`**, and **`ΔGSL = −9.07 ± 1.81 SE d (p = 0.00053)`** (unadjusted paired DiD `−11.14 d`; no `is_closed × relief` interaction, `p = 0.66–0.86`). Rim-free `Δz_floor`-adjusted floor-vs-floor comparisons confirm **`ΔGSL = −8.75 ± 1.85 SE d (p = 0.0011)`** overall, **`−11.17 ± 2.61 SE d (p = 0.0027)`** inside dry oak (`EVT 3317`), and **`−6.78 ± 1.87 SE d (p = 0.0067)`** inside cove forest (`EVT 3321`). Binary closure (`is_closed`) adds **partial `R² = 54.4%`** over `Relief` (total `R² = 0.725`; `ΔAICc = −17.1` vs. `Relief + 300 m TPI`, where TPI adds `+0.004`), with no depth ramp above `22 m` and `0–22 m` unobserved. Internal gate `P1` is recorded as `PARTIAL_PASS`.
3. **Cloud-Leak-Filtered `ECOSTRESS` Thermal Support (With Swath Manski Caveat):** Applying a physical `|LST − ERA5_t2m| <= 10 °C` filter removes **8 unflagged cloud-contaminated/corrupted `ECOv002` scenes** (`0/14` clean `v003` dropped), leaving `n = 52` clean nights (`51` with ERA5 wind). Cooling is cleanly significant in **both** spring (`n = 19, −0.96 ± 0.22 SE °C, t p = 0.00042, Wilcoxon p = 0.00034`) and autumn (`n = 33, −1.49 ± 0.29 SE °C, p = 1.1 × 10⁻⁵`), replicates across **7/7 independent `16SED` coves** on Sewanee-verified clear granules (every cove individually `p <= 0.018`; cove-level calm mean **`−1.09 ± 0.22 SE °C, p = 0.0024`**), and collapses at Lost Cove over `2.5–3.0 m/s` (calm `<2.5 m/s, n = 44: −1.52 ± 0.20 SE °C` vs. breezy `2.5–3.0 m/s, n = 7: +0.33 ± 0.43 SE °C`, Welch `p = 0.0041`, Mann-Whitney `p = 0.0019`; `n = 0` clean nights `>= 3.0 m/s`, and wind collapse does not replicate across the 6 other `16SED` coves using Sewanee point wind, `p = 0.77`). **Key caveat:** leak-adjusted worst-case swath-level Manski bounds span zero (`v002: [−7.12, +5.34 °C]`, `v003: [−8.09, +6.05 °C]`).
4. **Plateau-Cap Spring Lag (`+3.53 d` Raw, `+2.62 d` Unharvested) & Falsified Predictions:** Across 25 multi-origin grid alignments at `1 km` and `2 km` spatial block clustering, the flat-vs-steep spring lag survives on well-drained soils (`+3.15 d, 2 km p <= 7.1 × 10⁻⁶`), unharvested `LAL` stands (`+2.62 d, 2 km p <= 0.0032`), 25/25 Landsat years (`+2.39 ± 1.61 SD d`), and 3 regional escarpment sites (`Sewanee +2.66`, `Fiery Gizzard +2.39`, `Savage Gulf +4.09 d`). Modeled fragipan soil inertia is falsified (`+0.32 d, 2 km med p = 0.66`; 25-yr precip `r = 0.20, p = 0.33`). Results are *consistent with* upland oak guild turnover (standardized `oct_mari_z β = +0.53 d/SD`, nested single-variable `ΔR² = +0.020`, though raw `oct_mari` `+0.120` is `2 km` grid-sensitive, `p = 0.017–0.094`, `36% < 0.05`). Internal decision criteria falsified S2 April-freeze vertical decoupling (`p = 0.76`) and the 25-yr 4-cove `CalmOctFreeze` panel (`β = +1.18 d, p = 0.035`).
"""))

    # ── SETUP CELL & MASTER LEDGER ──────────────────────────────────────────────
    cells.append(code("""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display, Image

ROOT = Path('.').resolve()
DATA = ROOT / 'data'
PROF = ROOT / 'profiling'
ASSETS = ROOT / 'assets'

master = json.loads((DATA / 'v2_master_summary.json').read_text())
v4a = json.loads((DATA / 'v2_p1_evt_dose_response.json').read_text())
v4b = json.loads((DATA / 'v2_ecostress_thermal_closure.json').read_text())
v4c = json.loads((DATA / 'v2_hopkins_climate_lal.json').read_text())
v4d = json.loads((DATA / 'v2_alphaearth_symmetric_dynamic.json').read_text())
v4e = json.loads((DATA / 'v2_signalb_audit.json').read_text())
p1 = json.loads((PROF / 'p1.json').read_text())
p2 = json.loads((PROF / 'p2.json').read_text())
p4 = json.loads((PROF / 'p4.json').read_text())
p5 = json.loads((PROF / 'p5.json').read_text())

pd.set_option('display.max_colwidth', 110)
ledger_df = pd.DataFrame(master['hypothesis_ledger'])[['id', 'status', 'claim']]
display(ledger_df)
"""))

    # ── SECTION 1: HOPKINS DECOMPOSITION & CLIMATE ──────────────────────────────
    cells.append(md("""
---
## 1. Decomposing Hopkins' (1918) Bioclimatic Law & Raw Station Climate Context

Internal round-1 checks (`03_hopkins_phenology.py`, `05_lapse_trend.py`, `06_landsat_trend.py`) produced steeper lapses (`4.10–4.27 d/100 m`) and a naive `−1.72 d/dec` 1984–2024 Landsat trend. Stage 4C decomposes both cleanly:

- **3-Factor MODIS `MCD12Q2 v061` Decomposition (`2001–2023`):** On this predominantly deciduous plateau, adding `QA_Overall_1 <= 1` (`Δ = 0.00 d/100 m`) and switching from `Hansen tc > 50%` to `NLCD-41` deciduous (`Δ = +0.01 d/100 m`) leave the 15% onset lapse (`Greenup_1`) unchanged at `3.98 ± 0.24 SE` (`1.16 SD`) d/100 m. Switching to 50% `MidGreenup_1` (matching Sentinel-2 `SOS50`) shifts the lapse by **`−0.80 d/100 m`** to **`3.19 ± 0.17 SE` (`0.82 SD`) d/100 m** — matching Hopkins' `3.28` — with no secular trend (`−0.05 d/100 m/dec, p = 0.86`) despite strong thermal coupling (`−2.40 ± 0.65 SE d/°C, r = −0.63, p = 0.0014`).
- **Unharmonized Landsat 1984–2024 Collinearity Non-Identifiability:** Because `year` and `I(year >= 2013)` are collinear (`r = 0.80, VIF = 2.7`), neither the 2013 `LEDAPS → LaSRC` step (`−0.42 ± 3.69 SE d, p = 0.91`) nor the secular trend (`−1.58 ± 1.46 SE d/dec, p = 0.29`) is individually identifiable without cross-sensor radiometric harmonization.
- **Raw `NOAA USC00408184` Station Context (`1895–1943` vs. `2000–2025`, `1944–1999` hiatus):** Raw GHCN-Daily records (unhomogenized for TOBS, instrument transitions, or micro-site moves) show `+0.83 °C` warmer Mar–Apr mean temperature (`p = 0.029`), `−7.5 d` earlier last spring freeze (`p = 0.036`), and `+10.1 d` longer frost-free season (`p = 0.021`), with flat autumn `Tmean` (`+0.00 °C, p = 0.995`).
"""))

    cells.append(code("""
m_clean = v4c['clean_modis_2001_2023']
m_dec = v4c['modis_3factor_decomposition']['configs']
era = v4c['climate_context']['noaa_modern_vs_hopkins_era_shift']
sens = v4c['climate_context']['modis_thermal_sensitivity']

tbl1 = pd.DataFrame([
    {'Metric': 'MODIS Greenup_1 (15% onset, Hansen >50%, no QA)',
     'Value': f"{m_dec['A_Greenup15_Hansen50_NoQA']['mean_d_per_100m']:.2f} ± {m_dec['A_Greenup15_Hansen50_NoQA']['se_d_per_100m']:.2f} SE ({m_dec['A_Greenup15_Hansen50_NoQA']['sd_d_per_100m']:.2f} SD) d/100 m",
     'Trend / Test': 'Baseline 15% onset metric'},
    {'Metric': 'MODIS Greenup_1 (15% onset, NLCD-41, QA<=1)',
     'Value': f"{m_dec['C_Greenup15_NLCD41_QA1']['mean_d_per_100m']:.2f} ± {m_dec['C_Greenup15_NLCD41_QA1']['se_d_per_100m']:.2f} SE d/100 m",
     'Trend / Test': 'QA + NLCD-41 delta = +0.01 d/100 m'},
    {'Metric': 'MODIS MidGreenup_1 (50%, NLCD-41, QA<=1)',
     'Value': f"{m_clean['lapse_rate_trend']['mean']:.2f} ± {m_clean['lapse_rate_trend']['se']:.2f} SE ({m_clean['lapse_rate_trend']['sd']:.2f} SD) d/100 m",
     'Trend / Test': f"50% vs 15% delta = -0.80; trend {m_clean['lapse_rate_trend']['ols_per_decade']:+.3f} (p={m_clean['lapse_rate_trend']['ols_p']})"},
    {'Metric': 'MODIS Thermal Sensitivity vs. Sewanee Mar-Apr Tmean',
     'Value': f"{sens['days_per_degC_spring']:.2f} ± {sens['se_days_per_degC']:.2f} SE d/°C",
     'Trend / Test': f"r = {sens['r']:.3f}, R² = {sens['r2']:.3f} (p={sens['p']})"},
    {'Metric': 'Raw GHCN-Daily Mar-Apr Tmean (unhomogenized)',
     'Value': f"{era['spr_tmean_C']['hopkins_era_mean']:.2f} → {era['spr_tmean_C']['modern_era_mean']:.2f} °C",
     'Trend / Test': f"Shift +{era['spr_tmean_C']['shift_modern_minus_hopkins']:.2f} °C (Welch p={era['spr_tmean_C']['welch_p']})"},
    {'Metric': 'Raw GHCN-Daily Frost-Free Season (unhomogenized)',
     'Value': f"{era['frost_free_days']['hopkins_era_mean']:.1f} → {era['frost_free_days']['modern_era_mean']:.1f} d",
     'Trend / Test': f"Shift +{era['frost_free_days']['shift_modern_minus_hopkins']:.1f} d (Welch p={era['frost_free_days']['welch_p']})"},
])
display(tbl1)
display(Image(filename=str(ASSETS / 'v2_fig1_hopkins_decomposition.png'), width=1050))
"""))

    # ── SECTION 2: BRDF / SCS+C / HLS & VERTICAL DECOUPLING ────────────────────
    cells.append(md("""
---
## 2. Optical & BRDF Artifact Immunity (`SCS+C` & `HLS v2.0`) and Internal-Criterion Vertical Decoupling Null

- **Signal A (Lost Cove vs. OpenN, `330–385 m`):** `SCS+C` topographic correction retains **`99.6%` of spring `ΔSOS` (`+3.84 d`)** and **`98.1%` of autumn `ΔEOS` (`−2.92 d`)** in Sentinel-2 NDRE; BRDF-normalized NASA `HLS v2.0` reproduces both signs (`NDRE +3.10 / −2.45 d`; `NDVI +3.33 / −2.19 d`). Note that internal gate `P2` is accurately recorded as **`PARTIAL_OR_FAIL`** (`sigA_pass = false` because `SCS+C` retention threshold and decoupling criteria were evaluated together).
- **Signal B (Flat Plateau Cap vs. Steep Rim):** Positive in all four cardinal aspect quadrants (`+2.30 to +2.61 d` `SCS+C NDRE`; `+2.77 to +3.32 d` `HLS NDVI`), ruling out solar-geometry/BRDF artifacts.
- **Falsified Internal Criterion — Vertical Canopy–Understory Decoupling (`ΔSOS_NDVI − ΔSOS_NDRE`):** On Sentinel-2 (`n = 7` yr), April-freeze decoupling failed (`freeze vs. mild p = 0.76, r = −0.20`). `HLS v2.0` (`r = −0.79, p = 0.036`) and post-DOY-100 frost splitting (`+1.70 vs. −0.84 d, p = 0.025`) are treated strictly as exploratory.
"""))

    cells.append(code("""
sigA = p2['signal_A_all']
vd_s = p2['vertical_decoupling']['scsc']
vd_h = p2['vertical_decoupling']['hls']
tbl2 = pd.DataFrame([
    {'Product': 'Raw S2 NDRE', 'SigA dSOS (d)': sigA['raw_ndre']['d_sos50_mean'], 'SigA dEOS (d)': sigA['raw_ndre']['d_eos50_mean'], 'SigA dGSL (d)': sigA['raw_ndre']['d_gsl_mean']},
    {'Product': 'SCS+C S2 NDRE', 'SigA dSOS (d)': sigA['scsc_ndre']['d_sos50_mean'], 'SigA dEOS (d)': sigA['scsc_ndre']['d_eos50_mean'], 'SigA dGSL (d)': sigA['scsc_ndre']['d_gsl_mean']},
    {'Product': 'HLS v2.0 NDRE', 'SigA dSOS (d)': sigA['hls_ndre']['d_sos50_mean'], 'SigA dEOS (d)': sigA['hls_ndre']['d_eos50_mean'], 'SigA dGSL (d)': sigA['hls_ndre']['d_gsl_mean']},
    {'Product': 'HLS v2.0 NDVI', 'SigA dSOS (d)': sigA['hls_ndvi']['d_sos50_mean'], 'SigA dEOS (d)': sigA['hls_ndvi']['d_eos50_mean'], 'SigA dGSL (d)': sigA['hls_ndvi']['d_gsl_mean']},
])
display(tbl2)
print(f"P2 internal gate: {p2['gate_check']['verdict']} | S2 decoupling p = {vd_s['freeze_vs_mild_p']} | HLS r = {vd_h['r_with_apr_tmin_min']} (p = {vd_h['p_with_apr_tmin_min']})")
display(Image(filename=str(ASSETS / 'v2_fig2_brdf_and_decoupling.png'), width=1050))
"""))

    # ── SECTION 3: REGIONAL KARST CENSUS, RELIEF ADJUSTMENT, & BINARY CLOSURE ──
    cells.append(md("""
---
## 3. Regional Closed-Karst Census (`N = 11` Pairs), Relief Adjustment, Rim-Free Floor Tests, & Binary Closure

Priority-Flood depression filling across the Southern Cumberland Plateau detected `17` closed depressions (`>= 22.5 m` depth, `>= 0.30 km²`). Exactly **6 basins were disqualified** before phenology extraction (`Karst_TN_04`, `Karst_TN_06`, `Karst_TN_13`, **`Lost_Cove_TN`** at `48.7%` 30 m `NLCD-41`, `Karst_AL_15` which collapsed to `0 m` at 30 m, and **`Sinking_Cove_TN`** at `57.1%` 30 m `NLCD-41`), leaving `11` forested closed basins (`9` passing strict 30 m `>= 25 m, >= 0.35 km²`) matched 1:1 with open-drainage gorges. Internal gate `P1` is recorded honestly as **`PARTIAL_PASS`** (open gorges showed `−3.03 d` NDVI floor–rim EOS and `r(σSOS, FillDepth) = 0.22`).

1. **Relief-Adjusted 22-Site CR1 Inference (`df = 10`):** Open-gorge controls have `51.8 m` greater floor-to-rim relief (`178.3` vs. `126.5 m, p = 0.0043`) and `7/11` open rims lie `>= 535 m` (vs. `2/11` closed), inflating the unadjusted paired DiD (`−11.14 d` NDRE; `−10.67 d` NDVI). Adjusting for relief with CR1 pair-clustered SEs (`df = 10`) yields **`ΔSOS = +4.28 ± 1.13 SE d (p = 0.0035)`**, **`ΔEOS = −4.79 ± 1.13 SE d (p = 0.0017)`**, and **`ΔGSL = −9.07 ± 1.81 SE d (p = 0.00053)`** in NDRE (`NDVI ΔGSL = −8.04 ± 2.10 SE d, p = 0.0034`), with no `is_closed × relief` interaction (`p = 0.66–0.86`).
2. **Rim-Free Direct Floor-vs-Floor (`Δz_floor`-Adjusted):** Bypassing rims entirely gives **`ΔSOS = +5.56 ± 1.15 SE d (p = 0.00095)`**, **`ΔEOS = −3.19 ± 1.38 SE d (p = 0.047)`**, and **`ΔGSL = −8.75 ± 1.85 SE d (p = 0.0011)`** across all 11 pairs; within identical `LANDFIRE EVT` classes (`n = 10` pairs), compression holds inside both **`EVT 3317` dry oak (`−11.17 ± 2.61 SE d, p = 0.0027`)** and **`EVT 3321` cove forest (`−6.78 ± 1.87 SE d, p = 0.0067`)**.
3. **Binary Topographic Closure vs. `300 m TPI` (Zero Depth Ramp Above 22 m):** Because all 11 open gorges have `FillDepth = 0 m` and all 11 closed basins exceed `22.2 m`, `min(FillDepth, 25 m)` restates binary `is_closed` on 21/22 sites (`ΔAICc = 0.03`). Controlling for relief, binary `is_closed` adds **partial `R² = 54.4%`** over `Relief` (`R² = 0.397 → 0.725`; `ΔAICc = −17.1` vs. `Relief + 300 m TPI`, where TPI adds `+0.004`). Within closed basins above 22 m, deeper sills add no extra compression (`r(dEOS, FillDepth) = +0.24, p = 0.47`), and the `0–22 m` regime is unobserved.
"""))

    cells.append(code("""
rf = v4a['relief_and_rim_free_analysis']
cr1 = rf['relief_adjusted_cr1_ols']['ndre']['all_11_pairs']
rf11 = rf['rim_free_floor_vs_floor']['ndre']['all_11_pairs']
rfe = rf['rim_free_floor_vs_floor']['ndre']['within_evt']

tbl3 = pd.DataFrame([
    {'Estimator (NDRE)': 'Relief-adjusted 22-site CR1 OLS (df=10)', 'n': 11,
     'dSOS (d ± SE)': f"{cr1['dSOS']['is_closed']['beta']:+.2f} ± {cr1['dSOS']['is_closed']['se']:.2f} (p={cr1['dSOS']['is_closed']['p']})",
     'dEOS (d ± SE)': f"{cr1['dEOS']['is_closed']['beta']:+.2f} ± {cr1['dEOS']['is_closed']['se']:.2f} (p={cr1['dEOS']['is_closed']['p']})",
     'dGSL (d ± SE)': f"{cr1['dGSL']['is_closed']['beta']:+.2f} ± {cr1['dGSL']['is_closed']['se']:.2f} (p={cr1['dGSL']['is_closed']['p']})"},
    {'Estimator (NDRE)': 'Rim-free floor-vs-floor (dz_floor-adj)', 'n': 11,
     'dSOS (d ± SE)': f"{rf11['dSOS']['dz_floor_adjusted']['dz_adj_mean']:+.2f} ± {rf11['dSOS']['dz_floor_adjusted']['dz_adj_se']:.2f} (p={rf11['dSOS']['dz_floor_adjusted']['dz_adj_p']})",
     'dEOS (d ± SE)': f"{rf11['dEOS']['dz_floor_adjusted']['dz_adj_mean']:+.2f} ± {rf11['dEOS']['dz_floor_adjusted']['dz_adj_se']:.2f} (p={rf11['dEOS']['dz_floor_adjusted']['dz_adj_p']})",
     'dGSL (d ± SE)': f"{rf11['dGSL']['dz_floor_adjusted']['dz_adj_mean']:+.2f} ± {rf11['dGSL']['dz_floor_adjusted']['dz_adj_se']:.2f} (p={rf11['dGSL']['dz_floor_adjusted']['dz_adj_p']})"},
    {'Estimator (NDRE)': 'Within-EVT 3317 dry oak (dz_floor-adj)', 'n': 10,
     'dSOS (d ± SE)': f"{rfe['EVT3317_oak']['dSOS']['dz_floor_adjusted']['dz_adj_mean']:+.2f} ± {rfe['EVT3317_oak']['dSOS']['dz_floor_adjusted']['dz_adj_se']:.2f}",
     'dEOS (d ± SE)': f"{rfe['EVT3317_oak']['dEOS']['dz_floor_adjusted']['dz_adj_mean']:+.2f} ± {rfe['EVT3317_oak']['dEOS']['dz_floor_adjusted']['dz_adj_se']:.2f}",
     'dGSL (d ± SE)': f"{rfe['EVT3317_oak']['dGSL']['dz_floor_adjusted']['dz_adj_mean']:+.2f} ± {rfe['EVT3317_oak']['dGSL']['dz_floor_adjusted']['dz_adj_se']:.2f} (p={rfe['EVT3317_oak']['dGSL']['dz_floor_adjusted']['dz_adj_p']})"},
    {'Estimator (NDRE)': 'Within-EVT 3321 cove forest (dz_floor-adj)', 'n': 10,
     'dSOS (d ± SE)': f"{rfe['EVT3321_cove']['dSOS']['dz_floor_adjusted']['dz_adj_mean']:+.2f} ± {rfe['EVT3321_cove']['dSOS']['dz_floor_adjusted']['dz_adj_se']:.2f}",
     'dEOS (d ± SE)': f"{rfe['EVT3321_cove']['dEOS']['dz_floor_adjusted']['dz_adj_mean']:+.2f} ± {rfe['EVT3321_cove']['dEOS']['dz_floor_adjusted']['dz_adj_se']:.2f}",
     'dGSL (d ± SE)': f"{rfe['EVT3321_cove']['dGSL']['dz_floor_adjusted']['dz_adj_mean']:+.2f} ± {rfe['EVT3321_cove']['dGSL']['dz_floor_adjusted']['dz_adj_se']:.2f} (p={rfe['EVT3321_cove']['dGSL']['dz_floor_adjusted']['dz_adj_p']})"},
])
display(tbl3)
display(pd.DataFrame(v4a['disqualified_basins_archive']['basins'])[['name', 'max_fill_30m', 'floor_nlcd41_30m_pct', 'status']])
display(Image(filename=str(ASSETS / 'v2_fig3_regional_karst_census.png'), width=1080))
"""))

    # ── SECTION 4: ECOSTRESS THERMAL CLOSURE ───────────────────────────────────
    cells.append(md("""
---
## 4. Cloud-Leak-Filtered `ECOSTRESS` (`v002 + v003`), Wind Collapse, Manski Bounds, & 7-Cove Replication

Applying a physical `|LST − ERA5_t2m| <= 10 °C` check to the `60` QA-passing nighttime overpasses removes **8 unflagged `ECOv002` cloud-contaminated or calibration-corrupted scenes** (`7` with LST `11–22 °C` below ERA5 `t2m` and `1` pre-dawn scene at `+46–48 °C`; `0/14` clean `v003` scenes flagged), leaving **`n = 52` filtered clean nights** (`51` ERA5-matched):

- **Seasonal Cooling in Lost Cove:** Spring (`Feb–Apr, n = 19`): **`−0.96 ± 0.22 SE °C`** (`t p = 0.00042, Wilcoxon p = 0.00034`; unfiltered `n = 23` t-test was `p = 0.42` due to 4 cloud-leaked `v002` spring scenes). Autumn (`Sep–Nov, n = 33`): **`−1.49 ± 0.29 SE °C`** (`t p = 1.1 × 10⁻⁵`; October `n = 7: −2.07 ± 0.65 SE °C, p = 0.019`).
- **Wind Collapse Scoped Strictly to Lost Cove (`2.5–3.0 m/s`):** Calm nights (`u10 < 2.5 m/s, n = 44`): **`−1.52 ± 0.20 SE °C`** (`p = 1.7 × 10⁻⁹, 88.6%` negative) vs. breezy nights (`2.5–3.0 m/s, n = 7`): **`+0.33 ± 0.43 SE °C`** (`p = 0.48`; **Welch `p = 0.0041`, Mann-Whitney `p = 0.0019`**; autumn alone `Welch p = 0.0038`, `MW p = 0.0015`). Zero clean nights remain at `>= 3.0 m/s` after cloud-leak filtering, and wind collapse does **not** replicate across the 6 other `16SED` coves using Sewanee's point ERA5 wind (`paired p = 0.77`).
- **7-Cove Thermal Replication (`MGRS 16SED`):** Restricting `v002` multi-cove granules to nights verified clear at Sewanee (`|LST − ERA5_t2m| <= 10 °C`), all **7/7 independent closed basins** (`Lost_Cove_TN`, `Sinking_Cove_TN`, `Karst_AL_05`, `Karst_AL_10`, `Karst_AL_12`, `Karst_TN_02`, `Karst_TN_17`) are individually significant on both t-test and Wilcoxon (`calm p <= 0.018`), yielding a cove-level calm mean of **`−1.09 ± 0.22 SE °C` (`N = 7, p = 0.0024`)**.
- **Separate Leak-Adjusted `v002` / `v003` Swath Manski Bounds Caveat:** While clean-scene pixel bounds remain negative (`[−1.79, −1.15 °C]` in autumn), properly counting unflagged cloud-leaked passes (`|LST − ERA5_t2m| > 10 °C`: `9` in `v002`, `3` non-clean in `v003`) as unobserved yields swath bounds spanning zero in both `v002` (`[−7.12, +5.34 °C]`, `29.8%` leak-adjusted missingness) and `v003` (`[−8.09, +6.05 °C]`, `23.8%` leak-adjusted missingness), requiring paired in-situ temperature loggers to close the fog MNAR bound.
"""))

    cells.append(code("""
mc_tbl = [{
    'Closed Basin': 'Lost_Cove_TN (10m)', 'FillDepth (m)': 30.4,
    'Calm n': v4b['wind_placebo']['calm_lt2p5ms']['n'],
    'Calm ΔLST (°C ± SE)': f"{v4b['wind_placebo']['calm_lt2p5ms']['mean_C']:+.2f} ± {v4b['wind_placebo']['calm_lt2p5ms']['se_C']:.2f}",
    'Calm t p': v4b['wind_placebo']['calm_lt2p5ms']['p'],
    'Calm Wilc p': v4b['wind_placebo']['calm_lt2p5ms']['wilcoxon_p'],
}]
for r in v4b['multicove_16sed_replication']['per_cove']:
    c = r['calm_lt2p5ms']
    mc_tbl.append({
        'Closed Basin': r['cove'], 'FillDepth (m)': r['max_fill_m'],
        'Calm n': c['n'],
        'Calm ΔLST (°C ± SE)': f"{c['mean_C']:+.2f} ± {c['se_C']:.2f}",
        'Calm t p': c['p'],
        'Calm Wilc p': c['wilcoxon_p'],
    })
display(pd.DataFrame(mc_tbl))
display(Image(filename=str(ASSETS / 'v2_fig4_ecostress_thermal.png'), width=1080))
"""))

    # ── SECTION 5: 25-YEAR LANDSAT PANEL & P4 FALSIFICATION ────────────────────
    cells.append(md("""
---
## 5. 25-Year Harmonized Landsat Panel (`2000–2024`): Persistence, Sensor Limits, & Falsified `P4` Panel

- Across 25 years of Roy-harmonized Landsat 5–9 (`2000–2024`), Lost Cove shows later spring in `88%` of years (`median +2.45 d`) and earlier autumn in `80%` (`median −2.53 d`). However, two sparse-observation outlier years (`2007` EOS `+23.6 d`, `2013` SOS `−30.8 d`) render unscreened OLS means non-significant (`+1.62 ± 1.42 SE d, p = 0.27`; `−2.22 ± 1.34 SE d, p = 0.11`).
- During the `2018–2024` overlap with `HLS v2.0`, 16-day Landsat tracks autumn `ΔEOS` closely (`r = 0.961, p = 0.00056`), **but fails to track interannual spring `ΔSOS` variability (`r = −0.041, p = 0.93`)**.
- **Falsification of `P4` (`4 coves × 25 yr, N = 100`):** `CalmOctFreeze_z` produced the *opposite* sign (`β = +1.18 ± 0.53 SE d, p = 0.035`) because hard regional freezes (`β = +1.85 ± 0.70 SE d, p = 0.014`) drop both rims and floors below freezing together. Nighttime October wind speed (`OctNightWspd`) is also non-significant in Lost Cove (`r = 0.374, p = 0.065`).
"""))

    cells.append(code("""
display(Image(filename=str(ASSETS / 'v2_fig5_25yr_weather_panel.png'), width=1020))
"""))

    # ── SECTION 6: SIGNAL B MECHANISM & SPATIAL BLOCK-CLUSTERED AUDIT ──────────
    cells.append(md("""
---
## 6. Signal B Mechanism: Spatial Block-Clustered Inference & Plateau-Cap Guild Evidence

All plateau-cap pixel tests are evaluated with **25 multi-origin grid alignments at `1 km` and `2 km` spatial block clustering** (`v2_signalb_audit.py`), replacing pixel-level p-values:

1. **Modeled Fragipan Soil Inertia Falsified:** Within the flat cap (`n = 700`), `gNATSGO`/SSURGO fragipan soils differ from well-drained soils by `+0.32 ± 0.71 SE d` (`2 km med p = 0.66`). Modeled `POLARIS`/`gNATSGO` layers add only `ΔR² = +0.0038` over `Elev + TPI`, and 25-yr flat–steep `ΔSOS` (`+2.39 ± 1.61 SD d`, `25/25 yr > 0`) does not correlate with Mar–Apr precipitation (`r = 0.20, p = 0.33`).
2. **Spatial-Clustered Plateau Lag Robust Across All Controls:** Raw flat vs. steep (`+3.53 d`, `2 km p <= 1.1 × 10⁻⁵`), well-drained soils only (`+3.15 d`, `2 km p <= 7.1 × 10⁻⁶`), and never-harvested `LAL` stands (`+2.62 d`, `2 km p <= 0.0032`) remain significant across 100% of 1 km and 2 km grid alignments, and replicate across 3 plateau sites (`Sewanee +2.66`, `Fiery Gizzard +2.39`, `Savage Gulf +4.09 d`).
3. **Consistent With (Not Definitive Proof of) Upland Oak Guild Turnover:** In strictly nested models, standardized `oct_mari_z` enters at **`+0.53 d/SD`** (`1 km med p = 0.0051`, `2 km med p = 0.013`; true single-variable `ΔR² = +0.020`), while the raw flat–steep `oct_mari` contrast (`+0.120`) is grid-sensitive at 2 km (`p = 0.017–0.094`, `36% < 0.05`). Because we lack species-level basal-area plots and `POLARIS` is modeled soil data, results are framed strictly as *consistent with* upland oak guild turnover.
"""))

    cells.append(code("""
sc = v4e['spatial_clustered_contrasts']
rows6 = []
for k, lbl in [
    ('raw_flat_vs_steep_sos', 'Raw Flat − Steep SOS (all n=1400)'),
    ('well_drained_only_flat_vs_steep_sos', 'Well-drained SSURGO only (n=1199)'),
    ('within_flat_fragipan_vs_welldrained_sos', 'Within-flat Fragipan − Well-drained (null)'),
    ('lal_excl_all_harvests_flat_vs_steep_sos', 'Excl. ALL 120 LAL harvests (n=435)'),
    ('flat_vs_steep_oct_mari', 'October mARI (Flat − Steep)'),
]:
    b1, b2 = sc[k]['block_1km'], sc[k]['block_2km']
    rows6.append({
        'Test': lbl, 'Beta': b1['beta'],
        '1km SE (med)': b1['se_median'], '1km p range': f"{b1['p_min']} – {b1['p_max']}",
        '2km SE (med)': b2['se_median'], '2km p range': f"{b2['p_min']} – {b2['p_max']}",
        '2km % <0.05': f"{b2['frac_origins_p_lt_0p05']*100:.0f}%",
    })
display(pd.DataFrame(rows6))
display(Image(filename=str(ASSETS / 'v2_fig6_plateau_guild_turnover.png'), width=1080))
"""))

    # ── SUPPLEMENTARY SECTION: ALPHAEARTH AUDIT ────────────────────────────────
    cells.append(md("""
---
## Supplementary Audit (Excluded From Main Manuscript): AlphaEarth vs. Raw Sentinel-2

Per peer review, the AlphaEarth embedding comparison (`v2_alphaearth_symmetric_dynamic.py`) has been **removed from the main manuscript** and archived here as a standalone supplementary benchmark. Comparing `AlphaEarth 64-d` against the **strongest** matched raw Sentinel-2 model (`Raw S2 40-band` beats `60-feature` on first-diff LOYO at `0.244` vs. `0.087`) across the 4 actual LOCO folds (`Fiery_Gizzard`, `Savage_Gulf`, `Sewanee_LostCove`, `Sinking_Cove`):
- Local 800 m buffered CV: `AE = 0.578` vs. best `Raw S2 + 3DEP = 0.611` (`−0.033`).
- 4-fold LOCO transfer: `AE + 3DEP = 0.681` vs. best `Raw S2 + 3DEP = 0.514` (`+0.167`).
- Non-overlapping first-diff LOYO: `AE = 0.352` vs. `Raw S2 40b = 0.244` (`60f = 0.087`); two-way first-diff LOYO: `AE = 0.139` vs. `Raw S2 = −0.020`.
- Two-way prior-year lag (`t−1 → t`): `AE = −0.024` (zero forecasting skill).
"""))

    cells.append(code("""
display(pd.DataFrame(v4d['best_raw_s2_comparison']).T)
display(Image(filename=str(ASSETS / 'v2_fig7_alphaearth_sidebar.png'), width=1050))
"""))

    nb.cells = cells
    with open(NB_PATH, 'w') as f:
        nbf.write(nb, f)
    print(f'WROTE {NB_PATH} ({len(cells)} cells)')


def execute_and_export():
    nbconv = os.path.join(os.path.dirname(sys.executable), 'jupyter-nbconvert')
    cmd_exec = [
        nbconv, '--to', 'notebook', '--execute', '--inplace',
        '--ExecutePreprocessor.timeout=120', NB_PATH,
    ]
    print('Executing notebook:', ' '.join(cmd_exec))
    subprocess.run(cmd_exec, check=True, cwd=HERE)

    cmd_html = [nbconv, '--to', 'html', '--output', HTML_PATH, NB_PATH]
    print('Exporting HTML:', ' '.join(cmd_html))
    subprocess.run(cmd_html, check=True, cwd=HERE)
    print(f'SUCCESS: executed {NB_PATH} and exported {HTML_PATH}')


def main():
    build_notebook()
    execute_and_export()


if __name__ == '__main__':
    main()
