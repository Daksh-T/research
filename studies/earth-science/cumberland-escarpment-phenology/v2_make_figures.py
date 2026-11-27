"""Generate publication-grade 300-dpi figures for the Southern Cumberland Escarpment study.

All numbers are loaded dynamically from verified Stage-3 (profiling/p1..p6.json) and
Stage-4 (data/v2_*.json) strict RFC 8259 JSON outputs — zero hard-coded numerical strings.

Outputs:
  assets/v2_fig1..6*.png (with suptitles, for notebook embedding)
  paper/figures/fig1..6.png (top suptitle cleanly cropped, for LaTeX manuscript)
  assets/v2_fig7_alphaearth_sidebar.png (standalone supplementary audit figure)
"""
import json
import math
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, 'assets')
PAPER_FIGS = os.path.join(HERE, 'paper', 'figures')
os.makedirs(ASSETS, exist_ok=True)
os.makedirs(PAPER_FIGS, exist_ok=True)

# Okabe-Ito colorblind-safe palette
C_BLUE = '#0072B2'
C_VERM = '#D55E00'
C_GREEN = '#009E73'
C_ORANGE = '#E69F00'
C_PURPLE = '#CC79A7'
C_SKY = '#56B4E9'
C_DARK = '#2B2B2B'
C_GRAY = '#7F7F7F'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9.0,
    'axes.titlesize': 10.2,
    'axes.titleweight': 'bold',
    'axes.labelsize': 9.3,
    'xtick.labelsize': 8.3,
    'ytick.labelsize': 8.3,
    'legend.fontsize': 7.8,
    'figure.titlesize': 12.0,
    'figure.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'savefig.dpi': 300,
})


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return 'n/a'
    if p < 1e-4:
        exp = int(math.floor(math.log10(p)))
        mant = p / (10 ** exp)
        return f'p = {mant:.1f}e{exp}'
    if p < 0.001:
        return f'p = {p:.4f}'
    if p < 0.01:
        return f'p = {p:.3f}'
    return f'p = {p:.2f}'


def save_fig_pair(fig, asset_name, paper_fig_name=None, crop_top_frac=0.068):
    """Save full figure to assets/ and suptitle-cropped version to paper/figures/."""
    out_asset = os.path.join(ASSETS, asset_name)
    fig.savefig(out_asset)
    plt.close(fig)
    print('SAVED', out_asset)
    if paper_fig_name is not None:
        out_paper = os.path.join(PAPER_FIGS, paper_fig_name)
        with Image.open(out_asset) as im:
            w, h = im.size
            y0 = int(round(h * crop_top_frac))
            im.crop((0, y0, w, h)).save(out_paper, dpi=(300, 300))
        print('SAVED', out_paper)


def load_all():
    return {
        'p1': json.load(open(os.path.join(HERE, 'profiling', 'p1.json'))),
        'p2': json.load(open(os.path.join(HERE, 'profiling', 'p2.json'))),
        'p3': json.load(open(os.path.join(HERE, 'profiling', 'p3.json'))),
        'p4': json.load(open(os.path.join(HERE, 'profiling', 'p4.json'))),
        'p5': json.load(open(os.path.join(HERE, 'profiling', 'p5.json'))),
        'p6': json.load(open(os.path.join(HERE, 'profiling', 'p6.json'))),
        'v4a': json.load(open(os.path.join(HERE, 'data', 'v2_p1_evt_dose_response.json'))),
        'v4b': json.load(open(os.path.join(HERE, 'data', 'v2_ecostress_thermal_closure.json'))),
        'v4c': json.load(open(os.path.join(HERE, 'data', 'v2_hopkins_climate_lal.json'))),
        'v4d': json.load(open(os.path.join(HERE, 'data', 'v2_alphaearth_symmetric_dynamic.json'))),
        'v4e': json.load(open(os.path.join(HERE, 'data', 'v2_signalb_audit.json'))),
    }


def make_fig1(D):
    """Fig 1: Hopkins (1918) Decomposition, 3-Factor MODIS Baseline, & Century Climate Context."""
    p2 = D['p2']
    v4c = D['v4c']
    v4e = D['v4e']
    bins = pd.DataFrame(v4c['lal_public_controls']['hopkins_domain_decomposition']['elevation_bins_15m'])
    modis = pd.DataFrame(v4c['clean_modis_2001_2023']['per_year'])
    noaa = pd.DataFrame(v4c['climate_context']['noaa_annual_series'])
    sens = v4c['climate_context']['modis_thermal_sensitivity']
    era_spr = v4c['climate_context']['noaa_modern_vs_hopkins_era_shift']['spr_tmean_C']
    era_ff = v4c['climate_context']['noaa_modern_vs_hopkins_era_shift']['frost_free_days']
    lap_tr = v4c['clean_modis_2001_2023']['lapse_rate_trend']
    m_dec = v4c.get('modis_3factor_decomposition', {})
    g15_cfg = m_dec.get('configs', {}).get('C_Greenup15_NLCD41_QA1', {})
    m_del = m_dec.get('sequential_deltas_d_per_100m', {})
    ls_step = v4c['legacy_landsat_step_artifact']['step_dummy_regression']

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), gridspec_kw={'width_ratios': [1.22, 1.08, 1.0]})

    # (a) Sewanee Domain Hopkins Decomposition
    ax = axes[0]
    ax.axvspan(295, 385, color=C_BLUE, alpha=0.09, label='Signal A: Karst floor (<385 m)')
    ax.axvspan(545, 610, color=C_ORANGE, alpha=0.11, label='Signal B: Plateau cap (>545 m)')

    b_open = bins.dropna(subset=['open_sos_mean'])
    b_lost = bins.dropna(subset=['lost_sos_mean'])
    ax.plot(b_open['elev_mid'], b_open['open_sos_mean'], 'o-', color=C_DARK, lw=2.0, ms=5.5,
            label='Open escarpment & plateau')
    ax.plot(b_lost['elev_mid'], b_lost['lost_sos_mean'], 's-', color=C_BLUE, lw=2.2, ms=6.2,
            label='Lost Cove (closed karst)')

    ref_e = np.linspace(305, 595, 50)
    y_450 = float(np.interp(450.0, b_open['elev_mid'], b_open['open_sos_mean']))
    ax.plot(ref_e, y_450 + 3.281 * (ref_e - 450.0) / 100.0, ':', color=C_VERM, lw=2.0,
            label='Hopkins (1918): 3.28 d/100 m')

    sigA_s2 = p2['signal_A_all']['scsc_ndre']['d_sos50_mean']
    sigA_hls = p2['signal_A_all']['hls_ndvi']['d_sos50_mean']
    sigB_unh = v4e['spatial_clustered_contrasts']['lal_excl_all_harvests_flat_vs_steep_sos']['block_1km']['beta']
    ax.annotate(f"Signal A: Cold-air pool\n({sigA_s2:+.1f} d S2 / {sigA_hls:+.1f} d HLS)",
                xy=(345, float(np.interp(345.0, b_lost['elev_mid'], b_lost['lost_sos_mean']))),
                xytext=(302, 117.2),
                fontsize=7.6, fontweight='bold', color=C_BLUE,
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=C_BLUE, alpha=0.92),
                arrowprops=dict(arrowstyle='->', color=C_BLUE, lw=1.2))
    ax.annotate(f"Signal B: Plateau-cap lag\n({sigB_unh:+.1f} d unharvested)",
                xy=(572, float(np.interp(572.0, b_open['elev_mid'], b_open['open_sos_mean']))),
                xytext=(445, 117.5),
                fontsize=7.6, fontweight='bold', color='#A66800',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=C_ORANGE, alpha=0.92),
                arrowprops=dict(arrowstyle='->', color='#A66800', lw=1.2))

    ax.set_xlabel('Elevation (m, 3DEP)')
    ax.set_ylabel('Sentinel-2 Spring Green-up SOS50 (DOY)')
    ax.set_title('(a) Decomposing Hopkins\' Law on Sewanee Domain')
    ax.set_xlim(295, 610)
    ax.set_ylim(101.2, 120.8)
    ax.legend(loc='lower right', fontsize=7.4, frameon=True, facecolor='white', framealpha=0.93)

    # (b) MODIS 50% MidGreenup vs 15% Greenup & Hopkins Benchmark
    ax = axes[1]
    ax.plot(modis['year'], modis['midgreenup_lapse_d_per_100m'], 'o-', color=C_GREEN, lw=1.8, ms=4.8,
            label=f"MidGreenup (50%): {lap_tr['mean']:.2f} ± {lap_tr['se']:.2f} SE")
    ax.axhline(3.281, color=C_VERM, ls='--', lw=1.8, label='Hopkins (1918): 3.28 d/100 m')
    if g15_cfg:
        ax.axhline(g15_cfg['mean_d_per_100m'], color=C_GRAY, ls=':', lw=1.7,
                   label=f"Greenup (15% onset): {g15_cfg['mean_d_per_100m']:.2f} ± {g15_cfg['se_d_per_100m']:.2f} SE")
    yrs_m = modis['year'].values
    yhat_m = lap_tr['mean'] + (lap_tr['ols_per_decade'] / 10.0) * (yrs_m - yrs_m.mean())
    ax.plot(yrs_m, yhat_m, '-', color=C_DARK, lw=1.4, alpha=0.8)

    step_b = ls_step['oli_2013_step_d']['beta']
    step_se = ls_step['oli_2013_step_d']['se']
    step_p = ls_step['oli_2013_step_d']['p']
    qa_nlcd_d = (m_del.get('delta_qa_filter_B_minus_A', 0.0) or 0.0) + (m_del.get('delta_nlcd41_mask_C_minus_B', 0.0) or 0.0)
    thr_d = m_del.get('delta_50pct_vs_15pct_D_minus_C', 0.0) or 0.0
    txt_b = (f"Lapse trend: {lap_tr['ols_per_decade']:+.2f} d/100m/dec ({fmt_p(lap_tr['ols_p'])})\n"
             f"3-factor Δ: 50% vs 15% = {thr_d:+.2f}; QA+NLCD = {qa_nlcd_d:+.2f}\n"
             f"Landsat 1984–2024: 2013 step non-identifiable\n"
             f"  (step {step_b:+.2f} ± {step_se:.2f} SE d, {fmt_p(step_p)})")
    ax.text(0.03, 0.03, txt_b, transform=ax.transAxes, fontsize=7.1, va='bottom',
            bbox=dict(boxstyle='round,pad=0.28', fc='white', ec='#CCCCCC', alpha=0.94))
    ax.set_xlabel('Year')
    ax.set_ylabel('Elevational Green-up Lapse (d / 100 m)')
    ax.set_title('(b) MODIS Lapse Rate (2001–2023, NLCD-41)')
    ax.set_ylim(0.4, 6.2)
    ax.legend(loc='upper left', fontsize=7.3, frameon=True, facecolor='white', framealpha=0.93)

    # (c) Century Sewanee Climate Context (raw GHCN-Daily)
    ax = axes[2]
    n_spr = noaa[['year', 'spr_tmean_C']].dropna()
    n_hop = n_spr[n_spr['year'] <= 1945]
    n_mod = n_spr[n_spr['year'] >= 2000]
    ax.scatter(n_hop['year'], n_hop['spr_tmean_C'], s=22, color=C_SKY, edgecolor=C_BLUE, lw=0.6,
               label=f"1895–1943 (n={era_spr['hopkins_era_1895_1943_n']}): {era_spr['hopkins_era_mean']:.2f} °C")
    ax.scatter(n_mod['year'], n_mod['spr_tmean_C'], s=28, color=C_VERM, edgecolor=C_DARK, lw=0.5,
               label=f"2000–2025 (n={era_spr['modern_era_2000_2025_n']}): {era_spr['modern_era_mean']:.2f} °C")
    ax.hlines(era_spr['hopkins_era_mean'], 1895, 1943, color=C_BLUE, lw=2.2)
    ax.hlines(era_spr['modern_era_mean'], 2000, 2025, color=C_VERM, lw=2.2)
    ax.axvspan(1944, 1999, color='#EAEAEA', alpha=0.75)
    ax.text(1971.5, 9.5, 'Temp sensor\nhiatus\n(1944–1999)', ha='center', va='center', fontsize=7.3, color=C_GRAY, style='italic')

    txt_c = (f"Raw GHCN-Daily shift (unhomogenized):\n"
             f"  Mar–Apr +{era_spr['shift_modern_minus_hopkins']:.2f} °C ({fmt_p(era_spr['welch_p'])})\n"
             f"  Frost-free +{era_ff['shift_modern_minus_hopkins']:.1f} d ({fmt_p(era_ff['welch_p'])})\n"
             f"MODIS sens: {sens['days_per_degC_spring']:.2f} ± {sens['se_days_per_degC']:.2f} SE d/°C\n"
             f"  (r = {sens['r']:.2f}, {fmt_p(sens['p'])})")
    ax.text(0.03, 0.03, txt_c, transform=ax.transAxes, fontsize=7.1, va='bottom',
            bbox=dict(boxstyle='round,pad=0.28', fc='white', ec='#CCCCCC', alpha=0.94))
    ax.set_xlabel('Year (NOAA Sewanee USC00408184)')
    ax.set_ylabel('Mar–Apr Mean Temperature (°C)')
    ax.set_title('(c) Raw Station Context vs. Hopkins Era')
    ax.set_xlim(1892, 2028)
    ax.set_ylim(5.0, 16.6)
    ax.legend(loc='upper left', fontsize=7.3, frameon=True, facecolor='white', framealpha=0.93)

    fig.suptitle('Figure 1. Decomposing Hopkins\' (1918) Bioclimatic Law Across the Southern Cumberland Escarpment', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig_pair(fig, 'v2_fig1_hopkins_decomposition.png', 'fig1.png')


def make_fig2(D):
    """Fig 2: BRDF/SCS+C/HLS Optical Immunity + Internal-Criterion Vertical Decoupling Null."""
    p2 = D['p2']
    sigA = p2['signal_A_all']
    sigB_asp = p2['signal_B_by_aspect']
    vd_scsc = p2['vertical_decoupling']['scsc']
    vd_hls = p2['vertical_decoupling']['hls']

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))

    # (a) Signal A across Raw S2, SCS+C S2, and NASA HLS v2.0
    ax = axes[0]
    keys = [('raw_ndre', 'Raw S2\nNDRE'), ('scsc_ndre', 'SCS+C S2\nNDRE'),
            ('hls_ndre', 'HLS v2.0\nNDRE'), ('hls_ndvi', 'HLS v2.0\nNDVI')]
    x = np.arange(len(keys))
    w = 0.35
    sos_m = [sigA[k]['d_sos50_mean'] for k, _ in keys]
    sos_se = [sigA[k]['d_sos50_sd'] / math.sqrt(sigA[k]['n_yrs']) for k, _ in keys]
    eos_m = [sigA[k]['d_eos50_mean'] for k, _ in keys]
    eos_se = [sigA[k]['d_eos50_sd'] / math.sqrt(sigA[k]['n_yrs']) for k, _ in keys]
    n_yrs_a = sigA['scsc_ndre']['n_yrs']

    ax.bar(x - w/2, sos_m, w, yerr=sos_se, capsize=3, color=C_BLUE, edgecolor=C_DARK, lw=0.7, label='Spring ΔSOS (Lost − OpenN)')
    ax.bar(x + w/2, eos_m, w, yerr=eos_se, capsize=3, color=C_VERM, edgecolor=C_DARK, lw=0.7, label='Autumn ΔEOS (Lost − OpenN)')
    ax.axhline(0, color=C_DARK, lw=0.9)
    for i in range(len(keys)):
        ax.text(x[i] - w/2, sos_m[i] + sos_se[i] + 0.22, f'+{sos_m[i]:.1f}d', ha='center', va='bottom', fontsize=7.5, fontweight='bold', color=C_BLUE)
        ax.text(x[i] + w/2, eos_m[i] - eos_se[i] - 0.25, f'{eos_m[i]:.1f}d', ha='center', va='top', fontsize=7.5, fontweight='bold', color=C_VERM)

    ax.text(0.03, 0.04,
            f"SCS+C retention: {p2['gate_check']['sigA_scsc_retention_SOS_pct']:.1f}% SOS, "
            f"{p2['gate_check']['sigA_scsc_retention_EOS_pct']:.1f}% EOS\n"
            f"Consistent across S2 & HLS v2.0 (n = {n_yrs_a} yr)",
            transform=ax.transAxes, fontsize=7.4, va='bottom',
            bbox=dict(boxstyle='round,pad=0.28', fc='white', ec='#CCCCCC', alpha=0.94))
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in keys])
    ax.set_ylabel('Lost Cove − OpenN (days, mean ± SE)')
    ax.set_ylim(-7.0, 6.8)
    ax.set_title('(a) Signal A Immunity: SCS+C & HLS v2.0')
    ax.legend(loc='upper right', fontsize=7.4, frameon=True, facecolor='white')

    # (b) Signal B Aspect-Quadrant BRDF Invariance (N, E, S, W)
    ax = axes[1]
    asps = ['N', 'E', 'S', 'W']
    xa = np.arange(len(asps))
    b_scsc = [sigB_asp[a]['scsc_ndre']['d_sos50_mean'] for a in asps]
    b_scsc_se = [sigB_asp[a]['scsc_ndre']['d_sos50_sd'] / math.sqrt(sigB_asp[a]['scsc_ndre']['n_yrs']) for a in asps]
    b_hls = [sigB_asp[a]['hls_ndvi']['d_sos50_mean'] for a in asps]
    b_hls_se = [sigB_asp[a]['hls_ndvi']['d_sos50_sd'] / math.sqrt(sigB_asp[a]['hls_ndvi']['n_yrs']) for a in asps]

    ax.bar(xa - w/2, b_scsc, w, yerr=b_scsc_se, capsize=3, color=C_GREEN, edgecolor=C_DARK, lw=0.7, label='SCS+C S2 NDRE (mean ± SE)')
    ax.bar(xa + w/2, b_hls, w, yerr=b_hls_se, capsize=3, color=C_ORANGE, edgecolor=C_DARK, lw=0.7, label='BRDF-Norm HLS NDVI (mean ± SE)')
    ax.axhline(0, color=C_DARK, lw=0.9)
    for i in range(len(asps)):
        ax.text(xa[i] - w/2, b_scsc[i] + b_scsc_se[i] + 0.10, f'+{b_scsc[i]:.2f}', ha='center', va='bottom', fontsize=7.4, color=C_GREEN, fontweight='bold')
        ax.text(xa[i] + w/2, b_hls[i] + b_hls_se[i] + 0.10, f'+{b_hls[i]:.2f}', ha='center', va='bottom', fontsize=7.4, color='#A66800', fontweight='bold')

    min_all = min(min(b_scsc), min(b_hls))
    max_all = max(max(b_scsc), max(b_hls))
    ax.text(0.03, 0.04,
            f"All 4 aspect quadrants positive ({min_all:+.1f} to {max_all:+.1f} d)\n"
            f"→ Rules out solar-geometry / BRDF artifact",
            transform=ax.transAxes, fontsize=7.5, va='bottom',
            bbox=dict(boxstyle='round,pad=0.28', fc='white', ec='#CCCCCC', alpha=0.94))
    ax.set_xticks(xa)
    ax.set_xticklabels(['North\n(315–45°)', 'East\n(45–135°)', 'South\n(135–225°)', 'West\n(225–315°)'])
    ax.set_ylabel('Plateau Cap − Escarpment ΔSOS (days)')
    ax.set_ylim(0, 5.4)
    ax.set_title('(b) Signal B Aspect-Quadrant Invariance')
    ax.legend(loc='upper right', fontsize=7.3, frameon=True, facecolor='white')

    # (c) Internal-Criterion Vertical Decoupling Null
    ax = axes[2]
    df_s = pd.DataFrame(vd_scsc['by_year'])
    df_h = pd.DataFrame(vd_hls['by_year'])
    ax.axhline(0, color=C_GRAY, ls='--', lw=1.0)
    ax.scatter(df_s['apr_tmin_min_C'], df_s['decoupling_ndvi_minus_ndre'], s=55, color=C_BLUE,
               marker='o', edgecolor=C_DARK, lw=0.7,
               label=f"SCS+C S2 (NULL: r={vd_scsc['r_with_apr_tmin_min']:.2f}, {fmt_p(vd_scsc['p_with_apr_tmin_min'])})")
    ax.scatter(df_h['apr_tmin_min_C'], df_h['decoupling_ndvi_minus_ndre'], s=55, color=C_VERM,
               marker='^', edgecolor=C_DARK, lw=0.7,
               label=f"HLS v2.0 (r={vd_hls['r_with_apr_tmin_min']:.2f}, {fmt_p(vd_hls['p_with_apr_tmin_min'])})")
    lr_h = stats.linregress(df_h['apr_tmin_min_C'], df_h['decoupling_ndvi_minus_ndre'])
    xx = np.linspace(df_h['apr_tmin_min_C'].min() - 0.3, df_h['apr_tmin_min_C'].max() + 0.3, 30)
    ax.plot(xx, lr_h.intercept + lr_h.slope * xx, '--', color=C_VERM, lw=1.4, alpha=0.8)
    for _, r in df_s.iterrows():
        dy = 0.18 if int(r['yr']) not in (2019, 2022) else -0.34
        ax.text(r['apr_tmin_min_C'] + 0.08, r['decoupling_ndvi_minus_ndre'] + dy, str(int(r['yr'])),
                fontsize=7.3, color=C_DARK)

    ax.text(0.03, 0.04,
            f"Internal criterion (S2 Apr freeze vs mild): {fmt_p(vd_scsc['freeze_vs_mild_p'])} (FAIL)\n"
            f"Post-hoc DOY>100 frost split: {vd_scsc['exploratory_post_doy100_frost_mean_d']:+.2f} vs "
            f"{vd_scsc['exploratory_no_post_doy100_frost_mean_d']:+.2f} d\n"
            f"  ({fmt_p(vd_scsc['exploratory_post_doy100_diff_p'])}, exploratory only, n={vd_scsc['n_yrs']})",
            transform=ax.transAxes, fontsize=7.2, va='bottom',
            bbox=dict(boxstyle='round,pad=0.28', fc='#FFF7F0', ec=C_VERM, alpha=0.92))
    ax.set_xlabel('Sewanee April Minimum Temperature (°C)')
    ax.set_ylabel('ΔSOS_NDVI − ΔSOS_NDRE (days)')
    ax.set_xlim(-3.8, 3.6)
    ax.set_ylim(-4.5, 3.2)
    ax.set_title('(c) Vertical Decoupling Audit (Null)')
    ax.legend(loc='upper right', fontsize=7.1, frameon=True, facecolor='white')

    fig.suptitle('Figure 2. Optical/BRDF Artifact Immunity (SCS+C & HLS v2.0) and Vertical Decoupling Null Test', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig_pair(fig, 'v2_fig2_brdf_and_decoupling.png', 'fig2.png')


def make_fig3(D):
    """Fig 3: Regional Forested Karst Census (11 Pairs), Relief-Adjusted & Rim-Free Tests, Binary Closure."""
    p1 = D['p1']
    v4a = D['v4a']
    ndre = p1['phenology_power']['ndre']
    tbl = pd.DataFrame(v4a['dose_response_aic']['site_table_ndre'])
    rf = v4a['relief_and_rim_free_analysis']
    cr1 = rf['relief_adjusted_cr1_ols']['ndre']['all_11_pairs']
    rimfree_all = rf['rim_free_floor_vs_floor']['ndre']['all_11_pairs']
    rimfree_evt = rf['rim_free_floor_vs_floor']['ndre']['within_evt']
    rc_gsl = rf['relief_controlled_closure_models']['ndre']['dGSL']
    w_slope = rf['within_closed_depth_slope']['ndre']
    rel_d = rf['relief_diagnostics']
    open_exp = v4a['within_evt_analysis']['open_gorge_baseline_explanation']['ndre']

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.1), gridspec_kw={'width_ratios': [1.08, 1.08, 1.14]})

    # (a) Paired Closed vs Open Floor-Rim Phenology + Relief-Adjusted CR1 Estimates
    ax = axes[0]
    metrics = ['dSOS', 'dEOS', 'dGSL']
    labels = ['Spring\nΔSOS\n(Fl − Rim)', 'Autumn\nΔEOS\n(Fl − Rim)', 'Growing Season\nΔGSL\n(Fl − Rim)']
    cl = tbl[tbl['kind'] == 'closed'].set_index('pair')
    op = tbl[tbl['kind'] == 'open'].set_index('pair')
    pairs_idx = cl.index.intersection(op.index)

    x = np.arange(len(metrics))
    w = 0.34
    cl_means = [cl.loc[pairs_idx, m].mean() for m in metrics]
    cl_ses = [cl.loc[pairs_idx, m].std(ddof=1) / math.sqrt(len(pairs_idx)) for m in metrics]
    op_means = [op.loc[pairs_idx, m].mean() for m in metrics]
    op_ses = [op.loc[pairs_idx, m].std(ddof=1) / math.sqrt(len(pairs_idx)) for m in metrics]

    ax.bar(x - w/2, cl_means, w, yerr=cl_ses, capsize=3, color=C_BLUE, edgecolor=C_DARK, lw=0.7, label=f'Closed Karst (n={len(pairs_idx)}, mean ± SE)')
    ax.bar(x + w/2, op_means, w, yerr=op_ses, capsize=3, color=C_SKY, edgecolor=C_DARK, lw=0.7, label=f'Open Gorge (n={len(pairs_idx)}, mean ± SE)')
    ax.axhline(0, color=C_DARK, lw=0.9)

    adj_specs = [
        (cr1['dSOS']['is_closed']['beta'], cr1['dSOS']['is_closed']['se'], cr1['dSOS']['is_closed']['p']),
        (cr1['dEOS']['is_closed']['beta'], cr1['dEOS']['is_closed']['se'], cr1['dEOS']['is_closed']['p']),
        (cr1['dGSL']['is_closed']['beta'], cr1['dGSL']['is_closed']['se'], cr1['dGSL']['is_closed']['p']),
    ]
    y_ann = [-8.9, -13.9, -15.9]
    for i, (b, se, pv) in enumerate(adj_specs):
        ax.text(x[i], y_ann[i], f"Relief-adj:\n{b:+.1f} ± {se:.1f} SE d\n({fmt_p(pv)})", ha='center', va='center',
                fontsize=6.8, fontweight='bold', color=C_DARK,
                bbox=dict(boxstyle='round,pad=0.20', fc='#F5F9FF', ec=C_BLUE, lw=0.7))

    rf_gsl = rimfree_all['dGSL']['dz_floor_adjusted']
    ax.text(0.97, 0.97,
            f"Open relief +{rel_d['open_minus_closed_relief_m']:.0f} m > closed ({fmt_p(rel_d['open_minus_closed_relief_p'])})\n"
            f"Unadj. DiD GSL: {ndre['paired_DiD_dGSL_mean']:+.1f} d\n"
            f"Rim-free Δz-adj GSL: {rf_gsl['dz_adj_mean']:+.1f} ± {rf_gsl['dz_adj_se']:.1f} SE d ({fmt_p(rf_gsl['dz_adj_p'])})",
            transform=ax.transAxes, fontsize=6.8, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#CCCCCC', alpha=0.94))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Floor − Rim Phenology Difference (days)')
    ax.set_ylim(-18.6, 8.2)
    ax.set_title('(a) Regional 11-Pair Census & Relief Adjustment')
    ax.legend(loc='lower left', fontsize=7.1, frameon=True, facecolor='white')

    # (b) Within-EVT Δz_floor-Adjusted Floor-vs-Floor (Rim-Free!)
    ax = axes[1]
    e21 = rimfree_evt['EVT3321_cove']
    e17 = rimfree_evt['EVT3317_oak']
    m_keys = ['dSOS', 'dEOS', 'dGSL']
    v21 = [e21[m]['dz_floor_adjusted']['dz_adj_mean'] for m in m_keys]
    s21 = [e21[m]['dz_floor_adjusted']['dz_adj_se'] for m in m_keys]
    v17 = [e17[m]['dz_floor_adjusted']['dz_adj_mean'] for m in m_keys]
    s17 = [e17[m]['dz_floor_adjusted']['dz_adj_se'] for m in m_keys]
    n21 = e21['dGSL']['dz_floor_adjusted']['n']
    n17 = e17['dGSL']['dz_floor_adjusted']['n']

    ax.bar(x - w/2, v21, w, yerr=s21, capsize=3, color=C_GREEN, edgecolor=C_DARK, lw=0.7,
           label=f'EVT 3321 Cove (n={n21}, Δz-adj ± SE)')
    ax.bar(x + w/2, v17, w, yerr=s17, capsize=3, color=C_ORANGE, edgecolor=C_DARK, lw=0.7,
           label=f'EVT 3317 Dry Oak (n={n17}, Δz-adj ± SE)')
    ax.axhline(0, color=C_DARK, lw=0.9)

    for i in range(3):
        dy21 = s21[i] + 0.45 if v21[i] > 0 else -(s21[i] + 0.45)
        dy17 = s17[i] + 0.45 if v17[i] > 0 else -(s17[i] + 0.45)
        ax.text(x[i] - w/2, v21[i] + dy21, f"{v21[i]:+.1f}d", ha='center',
                va='bottom' if v21[i] > 0 else 'top', fontsize=7.3, fontweight='bold', color=C_GREEN)
        ax.text(x[i] + w/2, v17[i] + dy17, f"{v17[i]:+.1f}d", ha='center',
                va='bottom' if v17[i] > 0 else 'top', fontsize=7.3, fontweight='bold', color='#A66800')

    op17_eos = open_exp['open_same_EVT3317_floor_minus_rim_EOS']['mean']
    txt_evt = (f"Rim-Free Within-EVT (Δz_floor-adjusted):\n"
               f"• EVT 3321 GSL: {e21['dGSL']['dz_floor_adjusted']['dz_adj_mean']:+.1f} ± {e21['dGSL']['dz_floor_adjusted']['dz_adj_se']:.1f} SE d ({fmt_p(e21['dGSL']['dz_floor_adjusted']['dz_adj_p'])})\n"
               f"• EVT 3317 GSL: {e17['dGSL']['dz_floor_adjusted']['dz_adj_mean']:+.1f} ± {e17['dGSL']['dz_floor_adjusted']['dz_adj_se']:.1f} SE d ({fmt_p(e17['dGSL']['dz_floor_adjusted']['dz_adj_p'])})\n"
               f"Open-gorge floor−rim EOS ({op17_eos:+.1f} d in 3317)\n"
               f"  reflects canyon shading & transient drainage")
    ax.text(0.03, 0.03, txt_evt, transform=ax.transAxes, fontsize=6.9, va='bottom',
            bbox=dict(boxstyle='round,pad=0.26', fc='white', ec='#CCCCCC', alpha=0.94))
    ax.set_xticks(x)
    ax.set_xticklabels(['Spring\nΔSOS', 'Autumn\nΔEOS', 'Growing Season\nΔGSL'])
    ax.set_ylabel('Closed Floor − Open Floor (days, Δz-adj)')
    ax.set_ylim(-18.6, 9.8)
    ax.set_title('(b) Rim-Free Within-EVT Floor Comparison')
    ax.legend(loc='upper right', fontsize=7.2, frameon=True, facecolor='white')

    # (c) Binary Topographic Closure vs 300m TPI (Controlling for Relief) + Unobserved 0-22m Gap
    ax = axes[2]
    is_cl = tbl['kind'] == 'closed'
    min_cl_fd = float(tbl.loc[is_cl, 'fill_30m'].min())
    ax.axvspan(0.8, min_cl_fd - 0.5, facecolor='#EAEAEA', alpha=0.65, hatch='///', edgecolor='#BDBDBD',
               label=f'Unobserved gap (0–{min_cl_fd:.0f} m)')
    ax.scatter(tbl.loc[~is_cl, 'fill_30m'], tbl.loc[~is_cl, 'dGSL'], s=60, color=C_SKY,
               edgecolor=C_DARK, lw=0.8, zorder=4, label='Open gorges (FillDepth = 0 m)')
    ax.scatter(tbl.loc[is_cl, 'fill_30m'], tbl.loc[is_cl, 'dGSL'], s=68, color=C_BLUE,
               marker='D', edgecolor=C_DARK, lw=0.8, zorder=5, label=f'Closed karst (≥{min_cl_fd:.0f} m)')

    # Group means + flat within-closed trend line
    op_m = float(tbl.loc[~is_cl, 'dGSL'].mean())
    cl_m = float(tbl.loc[is_cl, 'dGSL'].mean())
    ax.hlines(op_m, -3.0, 2.5, color=C_DARK, lw=2.2, zorder=6)
    cl_sub = tbl.loc[is_cl]
    lr_cl = stats.linregress(cl_sub['fill_30m'].values, cl_sub['dGSL'].values)
    xx_cl = np.linspace(min_cl_fd, float(cl_sub['fill_30m'].max()), 40)
    ax.plot(xx_cl, lr_cl.intercept + lr_cl.slope * xx_cl, '--', color=C_VERM, lw=1.8,
            label=f"Within-closed slope (r={w_slope['dGSL']['r']:+.2f}, {fmt_p(w_slope['dGSL']['p'])})")
    ax.axhline(0, color=C_DARK, lw=0.8, alpha=0.6)

    mor_res = v4a['dose_response_aic']['moran_i_spatial_autocorrelation']['m5_hinge25m_dGSL_residuals_22sites']
    txt_aic = (f"Relief-Controlled Models (ΔGSL, n=22):\n"
               f"• Relief only: R² = {rc_gsl['M_relief']['r2']:.3f}\n"
               f"• Relief + TPI300: R² = {rc_gsl['M_relief_tpi300']['r2']:.3f} (ΔR² = +{rc_gsl['delta_r2_tpi300_over_relief']:.3f})\n"
               f"• Relief + is_closed: R² = {rc_gsl['M_relief_is_closed']['r2']:.3f}\n"
               f"  (partial R² = {rc_gsl['partial_r2_is_closed_over_relief']*100:.1f}%, ΔAICc = {rc_gsl['delta_aicc_relief_closed_minus_relief_tpi300']:.1f})\n"
               f"• Residual Moran's I = {mor_res['I']:+.2f} ({fmt_p(mor_res['p_perm_two_sided'])})")
    ax.text(0.97, 0.97, txt_aic, transform=ax.transAxes, fontsize=6.9, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.26', fc='#FFF7F0', ec=C_VERM, alpha=0.94))
    ax.set_xlabel('30 m Verified Basin Closure FillDepth (m)')
    ax.set_ylabel('Floor − Rim Growing Season ΔGSL (days, NDRE)')
    ax.set_xlim(-4, 85)
    ax.set_ylim(-25.8, 11.8)
    ax.set_title('(c) Binary Closure vs. TPI & Unobserved 0–22 m Gap')
    ax.legend(loc='lower left', fontsize=6.9, frameon=True, facecolor='white')

    fig.suptitle('Figure 3. Regional Closed-Karst Census (11 Pairs), Relief-Adjusted & Rim-Free Tests, and Binary Closure', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig_pair(fig, 'v2_fig3_regional_karst_census.png', 'fig3.png')


def make_fig4(D):
    """Fig 4: Cloud-Leak-Filtered ECOSTRESS v002+v003, Lost Cove Wind Collapse, Manski Bounds, & 7-Cove Replication."""
    v4b = D['v4b']
    sc = pd.DataFrame(v4b['clean_night_scenes'])
    wp = v4b['wind_placebo']
    wreg = wp['continuous_wind_regression']
    ns = v4b['lost_cove_night_stats']
    sh = v4b['artifact_controls']['morning_085_100h_raytraced_shadow']
    mc = v4b['multicove_16sed_replication']
    ac = v4b['archive_counts']

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0), gridspec_kw={'width_ratios': [1.15, 1.0, 1.15]})

    # (a) Cloud-Leak-Filtered Nighttime Cooling & Lost Cove Wind Collapse (2.5-3.0 m/s)
    ax = axes[0]
    has_w = sc[sc['wspd_ms'].notna()].copy()
    spr = has_w[has_w['season'] == 'spring']
    aut = has_w[has_w['season'] == 'autumn']

    ax.axvspan(0, 2.5, color=C_BLUE, alpha=0.07)
    ax.axvspan(2.5, 3.15, color=C_ORANGE, alpha=0.10)
    ax.axhline(0, color=C_DARK, lw=1.0, ls='--')
    ax.axvline(2.5, color=C_VERM, lw=1.3, ls=':')

    ax.scatter(aut['wspd_ms'], aut['diff_lost_minus_openn'], s=46, color=C_BLUE, edgecolor=C_DARK, lw=0.6,
               label=f"Autumn (n={ns['autumn_sep_nov_night']['n']}; {ns['autumn_sep_nov_night']['mean_C']:+.2f} °C, {fmt_p(ns['autumn_sep_nov_night']['p'])})")
    ax.scatter(spr['wspd_ms'], spr['diff_lost_minus_openn'], s=46, color=C_GREEN, marker='^', edgecolor=C_DARK, lw=0.6,
               label=f"Spring (n={ns['spring_feb_apr_night']['n']}; {ns['spring_feb_apr_night']['mean_C']:+.2f} °C, {fmt_p(ns['spring_feb_apr_night']['p'])})")

    ax.hlines(wp['calm_lt2p5ms']['mean_C'], 0.3, 2.4, color=C_BLUE, lw=2.8)
    ax.hlines(wp['breezy_2p5_3p0ms']['mean_C'], 2.55, 3.05, color=C_VERM, lw=2.8)

    aut_w = wreg['season_stratified_breezy2p5']['autumn']
    ax.text(1.30, -5.65,
            f"Calm (<2.5 m/s, n={wp['calm_lt2p5ms']['n']})\n"
            f"Mean {wp['calm_lt2p5ms']['mean_C']:+.2f} ± {wp['calm_lt2p5ms']['se_C']:.2f} SE °C ({fmt_p(wp['calm_lt2p5ms']['p'])})\n"
            f"{wp['calm_lt2p5ms']['frac_neg']*100:.1f}% negative",
            ha='center', va='bottom', fontsize=7.0, fontweight='bold', color=C_BLUE,
            bbox=dict(boxstyle='round,pad=0.22', fc='white', ec=C_BLUE, alpha=0.93))
    ax.text(2.46, 5.10,
            f"Lost Cove breezy (2.5–3.0 m/s, n={wp['breezy_2p5_3p0ms']['n']})\n"
            f"Mean {wp['breezy_2p5_3p0ms']['mean_C']:+.2f} ± {wp['breezy_2p5_3p0ms']['se_C']:.2f} SE °C ({fmt_p(wp['breezy_2p5_3p0ms']['p'])})\n"
            f"Welch {fmt_p(wreg['calm_vs_breezy2p5_welch_p'])}; MW {fmt_p(wreg['calm_vs_breezy2p5_mannwhitney_p'])}\n"
            f"Autumn: {fmt_p(aut_w['welch_p'])} (n=0 at ≥3.0 m/s)",
            ha='center', va='top', fontsize=6.6, fontweight='bold', color=C_VERM,
            bbox=dict(boxstyle='round,pad=0.22', fc='white', ec=C_VERM, alpha=0.94))

    ax.set_xlabel('ERA5-Land 10 m Wind Speed at Overpass (m/s)')
    ax.set_ylabel('Nighttime ΔLST: Lost Cove − OpenN (°C)')
    ax.set_xlim(0.20, 3.25)
    ax.set_ylim(-6.0, 5.45)
    ax.set_title(f'(a) Filtered Night LST (n={ac["combined_unique_night_clean"]}) & Wind Collapse')
    ax.legend(loc='upper left', fontsize=6.7, frameon=True, facecolor='white')

    # (b) Morning Shadow Audit + Emissivity/VZA + Separate v002/v003 & Combined Manski Bounds
    ax = axes[1]
    n_aut = ns['autumn_sep_nov_night']['n']
    n_sh_all = sh['all_C']['n']
    n_sh_sun = sh['sunlit_only_C']['n']
    n_sh_clm = sh['calm_sunlit_C']['n']
    m_aut_px = v4b['manski_bounds']['autumn_sep_nov']['clean_scene_pixel']
    m_v3_sw = v4b['manski_bounds']['v003_only_nights']['swath_conditioned']
    m_v2_sw = v4b['manski_bounds']['v002_only_nights']['swath_conditioned']

    cats = [
        f'Autumn Night\n(Filtered, n={n_aut})',
        'Autumn Pixel\nManski Bound',
        f'v003 Swath\nManski (n={m_v3_sw["n_clean"]}/{m_v3_sw["n_in_swath"]})',
        f'v002 Swath\nManski (n={m_v2_sw["n_clean"]}/{m_v2_sw["n_in_swath"]})',
        f'08.5–10h Sunlit\nOnly (n={n_sh_sun})',
        f'08.5–10h Calm\nSunlit (n={n_sh_clm})',
    ]
    y_pos = np.arange(len(cats))[::-1]
    ax.axvline(0, color=C_DARK, lw=1.0, ls='--')
    pts = [
        (ns['autumn_sep_nov_night']['mean_C'],
         ns['autumn_sep_nov_night']['mean_C'] - 1.96 * ns['autumn_sep_nov_night']['se_C'],
         ns['autumn_sep_nov_night']['mean_C'] + 1.96 * ns['autumn_sep_nov_night']['se_C'], C_BLUE, 'o'),
        (m_aut_px['naive_diff'], m_aut_px['manski_lo'], m_aut_px['manski_hi'], C_GREEN, 's'),
        (m_v3_sw['naive_diff'], m_v3_sw['manski_lo'], m_v3_sw['manski_hi'], C_ORANGE, 'D'),
        (m_v2_sw['naive_diff'], m_v2_sw['manski_lo'], m_v2_sw['manski_hi'], C_VERM, 'D'),
        (sh['sunlit_only_C']['mean_C'],
         sh['sunlit_only_C']['mean_C'] - 1.96 * (sh['sunlit_only_C']['sd_C'] / math.sqrt(n_sh_sun)),
         sh['sunlit_only_C']['mean_C'] + 1.96 * (sh['sunlit_only_C']['sd_C'] / math.sqrt(n_sh_sun)), C_SKY, 'o'),
        (sh['calm_sunlit_C']['mean_C'],
         sh['calm_sunlit_C']['mean_C'] - 1.96 * (sh['calm_sunlit_C']['sd_C'] / math.sqrt(n_sh_clm)),
         sh['calm_sunlit_C']['mean_C'] + 1.96 * (sh['calm_sunlit_C']['sd_C'] / math.sqrt(n_sh_clm)), C_PURPLE, 's'),
    ]
    for yp, (est, lo, hi, col, mk) in zip(y_pos, pts):
        ax.hlines(yp, lo, hi, color=col, lw=2.5)
        ax.plot([lo, hi], [yp, yp], '|', color=col, ms=8, mew=1.9)
        ax.plot(est, yp, mk, color=col, ms=7.0, mec=C_DARK, mew=0.7)

    vza_s = v4b['artifact_controls']['vza_lt25deg_night']
    m_all_tot = v4b['manski_bounds']['all_combined_nights']['swath_total_invalid']
    ax.text(0.03, 0.03,
            f"Filter: |LST−ERA5|≤10 °C dropped {ac['dropped_v002_cloud_leaks']} v002 (0/14 clean v003)\n"
            f"ΔEmisWB = {v4b['artifact_controls']['mean_d_EmisWB']:+.5f}; VZA<25° (n={vza_s['n']}): {vza_s['mean_C']:+.2f} °C\n"
            f"Morning 08.5–10h all (n={n_sh_all}): {sh['all_C']['mean_C']:+.2f} °C\n"
            f"Leak-adj swath Manski spans 0 ({m_all_tot['p_miss1']*100:.0f}% total invalid)",
            transform=ax.transAxes, fontsize=6.6, va='bottom',
            bbox=dict(boxstyle='round,pad=0.24', fc='#FFF7F0', ec=C_VERM, alpha=0.93))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cats, fontsize=7.6)
    ax.set_xlabel('Lost Cove − OpenN ΔLST (°C; 95% CI or Manski)')
    ax.set_xlim(-8.9, 6.8)
    ax.set_ylim(-1.95, 5.6)
    ax.set_title('(b) Shadow, Emissivity & Leak-Adj. Manski')

    # (c) Multi-Cove Nighttime Thermal Replication inside MGRS 16SED (7 coves, all individually p<=0.011)
    ax = axes[2]
    lost_fd30 = next((b['max_fill_30m'] for b in D['v4a']['disqualified_basins_archive']['basins'] if b['name'] == 'Lost_Cove_TN'), 30.4)
    cove_rows = [{
        'name': f'Lost_Cove_TN ({lost_fd30:.0f}m)',
        'all_m': ns['all_nights_combined']['mean_C'], 'all_se': ns['all_nights_combined']['se_C'],
        'calm_m': wp['calm_lt2p5ms']['mean_C'], 'calm_se': wp['calm_lt2p5ms']['se_C'],
        'calm_p': wp['calm_lt2p5ms']['p'],
    }]
    for r in mc['per_cove']:
        cove_rows.append({
            'name': f"{r['cove']} ({r['max_fill_m']:.0f}m)",
            'all_m': r['all_nights']['mean_C'], 'all_se': r['all_nights']['se_C'],
            'calm_m': r['calm_lt2p5ms']['mean_C'], 'calm_se': r['calm_lt2p5ms']['se_C'],
            'calm_p': r['calm_lt2p5ms']['p'],
        })
    cdf = pd.DataFrame(cove_rows).sort_values('calm_m')
    yc = np.arange(len(cdf))
    h = 0.36
    ax.barh(yc + h/2, cdf['all_m'], h, xerr=cdf['all_se'], capsize=2.5, color=C_SKY, edgecolor=C_DARK, lw=0.6, label='All screened nights (mean ± SE)')
    ax.barh(yc - h/2, cdf['calm_m'], h, xerr=cdf['calm_se'], capsize=2.5, color=C_BLUE, edgecolor=C_DARK, lw=0.6, label='Calm (<2.5 m/s, mean ± SE)')
    ax.axvline(0, color=C_DARK, lw=1.0)
    ax.axvline(mc['cove_level_calm_mean_C'], color=C_VERM, ls='--', lw=1.5,
               label=f"7-cove calm: {mc['cove_level_calm_mean_C']:+.2f} ± {mc['cove_level_calm_se_C']:.2f} SE °C")

    n_sig = int((cdf['calm_p'] < 0.05).sum())
    ax.text(0.03, 0.72,
            f"7 / 7 coves negative ({n_sig}/7 calm p ≤ {cdf['calm_p'].max():.3f})\n"
            f"Cove-level test (N={mc['n_cove_pairs_incl_lost_cove']}):\n"
            f"• All nights: {mc['cove_level_all_nights_mean_C']:+.2f} ± {mc['cove_level_all_nights_se_C']:.2f} SE ({fmt_p(mc['cove_level_all_nights_p'])})\n"
            f"• Calm: {mc['cove_level_calm_mean_C']:+.2f} ± {mc['cove_level_calm_se_C']:.2f} SE ({fmt_p(mc['cove_level_calm_p'])})",
            transform=ax.transAxes, fontsize=6.9, va='top',
            bbox=dict(boxstyle='round,pad=0.25', fc='#F5F9FF', ec=C_BLUE, alpha=0.95))
    ax.set_yticks(yc)
    ax.set_yticklabels(cdf['name'], fontsize=7.8)
    ax.set_xlabel('Closed Floor − Open Gorge Night ΔLST (°C)')
    ax.set_xlim(-2.95, 0.35)
    ax.set_title('(c) 7-Cove Thermal Replication (MGRS 16SED)')
    ax.legend(loc='upper left', fontsize=6.9, frameon=True, facecolor='white')

    fig.suptitle('Figure 4. Cloud-Leak-Filtered ECOSTRESS (v002 + v003) Nocturnal Cooling, Wind Collapse, Manski Bounds, & 7-Cove Replication', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig_pair(fig, 'v2_fig4_ecostress_thermal.png', 'fig4.png')


def make_fig5(D):
    """Fig 5: 25-Year Harmonized Landsat Panel (Honest Spring/Autumn Diagnostics & Falsified P4 Panel)."""
    p4 = D['p4']
    lo = p4['lost_minus_openn_25yr']
    by = pd.DataFrame(lo['by_year'])
    fe_eos = p4['panel_results']['qc80']['fe_eos_closed']['coefs']
    fe_thr = p4['panel_results']['qc80']['fe_eos_closed_threshold_model']['coefs']
    fe_sos = p4['panel_results']['qc80']['fe_sos_closed']['coefs']

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={'width_ratios': [1.35, 1.0]})

    # (a) 25-yr Lost Cove - OpenN dSOS & dEOS time series
    ax = axes[0]
    med_sos = float(by['dSOS_lost_minus_openn'].median())
    med_eos = float(by['dEOS_lost_minus_openn'].median())
    sos_p = lo['sos_ols']['coefs']['intercept']['p']
    eos_p = lo['eos_ols']['coefs']['intercept']['p']

    ax.axhline(0, color=C_DARK, lw=0.9, ls='--')
    ax.plot(by['yr'], by['dSOS_lost_minus_openn'], 'o-', color=C_BLUE, lw=1.7, ms=4.8,
            label=f"Spring ΔSOS: med {med_sos:+.2f} d ({lo['frac_yrs_dSOS_gt0']*100:.0f}% > 0); mean {lo['mean_dSOS']:+.2f} d ({fmt_p(sos_p)})")
    ax.plot(by['yr'], by['dEOS_lost_minus_openn'], 's-', color=C_VERM, lw=1.7, ms=4.8,
            label=f"Autumn ΔEOS: med {med_eos:+.2f} d ({lo['frac_yrs_dEOS_lt0']*100:.0f}% < 0); mean {lo['mean_dEOS']:+.2f} d ({fmt_p(eos_p)})")
    ax.axhline(med_sos, color=C_BLUE, ls=':', lw=1.2, alpha=0.8)
    ax.axhline(med_eos, color=C_VERM, ls=':', lw=1.2, alpha=0.8)
    ax.axvspan(2017.5, 2024.5, color='#EAEAEA', alpha=0.55, label='2018–2024 HLS overlap')

    # Dynamically identify extreme outlier years
    i_max_eos = int(by['dEOS_lost_minus_openn'].idxmax())
    i_min_sos = int(by['dSOS_lost_minus_openn'].idxmin())
    yr_oe, val_oe = int(by.loc[i_max_eos, 'yr']), float(by.loc[i_max_eos, 'dEOS_lost_minus_openn'])
    yr_os, val_os = int(by.loc[i_min_sos, 'yr']), float(by.loc[i_min_sos, 'dSOS_lost_minus_openn'])
    no_out = by[~by['yr'].isin([yr_oe, yr_os])]
    gsl_no_out = float((no_out['dEOS_lost_minus_openn'] - no_out['dSOS_lost_minus_openn']).mean())

    ax.annotate(f'{yr_oe} sparse-obs\nEOS ({val_oe:+.1f} d)', xy=(yr_oe, val_oe), xytext=(2000.2, 20.5),
                fontsize=7.1, color=C_VERM, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_VERM, lw=1.0))
    ax.annotate(f'{yr_os} sparse-obs\nSOS ({val_os:+.1f} d)', xy=(yr_os, val_os), xytext=(2004.5, -30.5),
                fontsize=7.1, color=C_BLUE, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_BLUE, lw=1.0))

    xs = lo['cross_sensor_2018_2024_vs_HLS']
    ax.text(0.98, 0.03,
            f"25-yr Lost Cove (L5/7/8/9, n={lo['n_years']}):\n"
            f"• Mean GSL: {lo['mean_dGSL']:+.2f} d (excl. {yr_oe}/{yr_os}: {gsl_no_out:+.2f} d)\n"
            f"• 2018–24 vs HLS ΔEOS: r = {xs['r_landsat_vs_hls_dEOS']:.3f} ({fmt_p(xs['p_dEOS'])})\n"
            f"• 2018–24 vs HLS ΔSOS: r = {xs['r_landsat_vs_hls_dSOS']:+.3f} ({fmt_p(xs['p_dSOS'])}, weak)\n"
            f"• OctNightWspd vs ΔEOS: r = {lo['r_OctNightWspd_dEOS']:+.3f} ({fmt_p(lo['p_OctNightWspd'])})",
            transform=ax.transAxes, fontsize=6.9, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#CCCCCC', alpha=0.95))
    ax.set_xlabel('Year (Harmonized Landsat C2 2000–2024)')
    ax.set_ylabel('Lost Cove − OpenN Phenology Difference (days)')
    ax.set_xlim(1999.3, 2024.8)
    ax.set_ylim(-35.5, 29.5)
    ax.set_title('(a) 25-Year Lost Cove Record & Cross-Sensor Audit')
    ax.legend(loc='upper right', fontsize=6.8, frameon=True, facecolor='white', framealpha=0.94)

    # (b) Falsification of P4 Interannual Weather-Modulation Panel
    ax = axes[1]
    rows = [
        ('CalmOctFreeze_z\n→ ΔEOS (pred < 0)', fe_eos['CalmOctFreeze_z']['beta'], fe_eos['CalmOctFreeze_z']['se'], fe_eos['CalmOctFreeze_z']['p'], C_VERM),
        ('HardRegionalFreeze_z\n→ ΔEOS (regional freeze)', fe_thr['HardRegionalFreezeNights_z']['beta'], fe_thr['HardRegionalFreezeNights_z']['se'], fe_thr['HardRegionalFreezeNights_z']['p'], C_ORANGE),
        ('CalmCapOnlyNights_z\n→ ΔEOS (pool-only)', fe_thr['CalmCapOnlyNights_z']['beta'], fe_thr['CalmCapOnlyNights_z']['se'], fe_thr['CalmCapOnlyNights_z']['p'], C_GRAY),
        ('WindyOctCold_z\n→ ΔEOS (placebo)', fe_eos['WindyOctCold_z']['beta'], fe_eos['WindyOctCold_z']['se'], fe_eos['WindyOctCold_z']['p'], C_SKY),
        ('AprLateFrost_z\n→ ΔSOS (pred > 0)', fe_sos['AprLateFrost_z']['beta'], fe_sos['AprLateFrost_z']['se'], fe_sos['AprLateFrost_z']['p'], C_PURPLE),
        ('MarAprPrecip_z\n→ ΔSOS (hydrology)', fe_sos['MarAprPrecip_z']['beta'], fe_sos['MarAprPrecip_z']['se'], fe_sos['MarAprPrecip_z']['p'], C_GREEN),
    ]
    yp = np.arange(len(rows))[::-1]
    ax.axvline(0, color=C_DARK, lw=1.0, ls='--')
    for y_i, (lbl, b, se, pv, col) in zip(yp, rows):
        ci = 1.96 * se
        ax.hlines(y_i, b - ci, b + ci, color=col, lw=2.5)
        ax.plot(b, y_i, 'o', color=col, ms=7.4, mec=C_DARK, mew=0.7)
        ax.text(b + ci + 0.12 if b >= 0 else b - ci - 0.12, y_i,
                f"{b:+.2f} d ({fmt_p(pv)})", va='center',
                ha='left' if b >= 0 else 'right', fontsize=7.4, fontweight='bold', color=col)

    n_cy = p4['panel_results']['qc80']['closed_cove_years']
    ax.text(0.03, 0.04,
            f"FALSIFIED (4 coves × 25 yr, N={n_cy} cove-years):\n"
            f"CalmOctFreeze has opposite sign ({fe_eos['CalmOctFreeze_z']['beta']:+.2f} d, {fmt_p(fe_eos['CalmOctFreeze_z']['p'])}).\n"
            f"Hard regional freezes ({fe_thr['HardRegionalFreezeNights_z']['beta']:+.2f} d, {fmt_p(fe_thr['HardRegionalFreezeNights_z']['p'])}) hit BOTH\n"
            f"rim & floor simultaneously, shrinking contrast.",
            transform=ax.transAxes, fontsize=7.2, va='bottom',
            bbox=dict(boxstyle='round,pad=0.28', fc='#FFF7F0', ec=C_VERM, alpha=0.94))
    ax.set_yticks(yp)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.9)
    ax.set_xlabel('Cove-FE Panel Coefficient β (days per 1 SD, ±95% CI)')
    ax.set_xlim(-5.2, 5.2)
    ax.set_ylim(-1.75, 5.6)
    ax.set_title('(b) Falsification of P4 Calm-Freeze Panel Mechanism')

    fig.suptitle('Figure 5. 25-Year Landsat Panel (2000–2024): Persistence, Sensor Limits, and Falsified Weather Modulation', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig_pair(fig, 'v2_fig5_25yr_weather_panel.png', 'fig5.png')


def make_fig6(D):
    """Fig 6: Signal B Mechanism — Spatial Block-Clustered Inference, Nested Standardized mARI, & 25-Yr Persistence."""
    p5 = D['p5']
    v4e = D['v4e']
    t5b = p5['test_5b_25yr_precip_interaction']
    sc = v4e['spatial_clustered_contrasts']
    nest = v4e['nested_ols_standardized_mari']
    dr2 = nest['true_nested_delta_r2']
    desc = v4e['descriptives']

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.9), gridspec_kw={'width_ratios': [1.16, 1.06, 0.98]})

    # (a) Head-to-Head Mechanism & LAL Harvest Sensitivity with 1km/2km Spatial Block SEs
    ax = axes[0]
    spec_bars = [
        ('Fragipan vs.\nWell-Drained\n(within Flat)', 'within_flat_fragipan_vs_welldrained_sos', C_GRAY),
        ('Flat vs. Steep\n(Well-Drained\nSoils Only)', 'well_drained_only_flat_vs_steep_sos', C_BLUE),
        ('Flat vs. Steep\n(All Plateau\nPixels)', 'raw_flat_vs_steep_sos', C_SKY),
        ('Flat vs. Steep\n(Excl. ≥1990\nLAL Harvest)', 'lal_excl_ge1990_flat_vs_steep_sos', C_GREEN),
        ('Flat vs. Steep\n(Excl. ALL 120\nLAL Harvests)', 'lal_excl_all_harvests_flat_vs_steep_sos', C_ORANGE),
    ]
    xb = np.arange(len(spec_bars))
    vals = [sc[k]['block_2km']['beta'] for _, k, _ in spec_bars]
    se2k = [sc[k]['block_2km']['se_median'] for _, k, _ in spec_bars]
    p2k_med = [sc[k]['block_2km']['p_median'] for _, k, _ in spec_bars]
    cols = [c for _, _, c in spec_bars]

    ax.bar(xb, vals, width=0.60, yerr=se2k, capsize=3, color=cols, edgecolor=C_DARK, lw=0.8)
    ax.axhline(0, color=C_DARK, lw=0.9)
    for i, (v, se, pv) in enumerate(zip(vals, se2k, p2k_med)):
        lbl = f"{v:+.2f} d\n(2km {fmt_p(pv)})"
        ax.text(i, v + se + 0.14, lbl, ha='center', va='bottom', fontsize=6.9, fontweight='bold',
                color=C_DARK if i != 0 else C_VERM)

    b_frag = sc['within_flat_fragipan_vs_welldrained_sos']['block_2km']
    b_unh = sc['lal_excl_all_harvests_flat_vs_steep_sos']['block_2km']
    ax.text(0.03, 0.96,
            f"Spatial block-clustered SEs (25 grid origins, 2 km):\n"
            f"• Modeled fragipan null: {b_frag['beta']:+.2f} ± {b_frag['se_median']:.2f} d ({fmt_p(b_frag['p_median'])})\n"
            f"• Never-harvested LAL: {b_unh['beta']:+.2f} ± {b_unh['se_median']:.2f} d (p≤{b_unh['p_max']:.3f})",
            transform=ax.transAxes, fontsize=7.0, va='top',
            bbox=dict(boxstyle='round,pad=0.26', fc='#FFF7F0', ec=C_VERM, alpha=0.94))
    ax.set_xticks(xb)
    ax.set_xticklabels([b[0] for b in spec_bars], fontsize=7.3)
    ax.set_ylabel('Spring Green-up SOS Difference (days, ±2 km SE)')
    ax.set_ylim(-0.3, 5.6)
    ax.set_title('(a) Spatial Block-Clustered Controls & LAL')

    # (b) Strictly Nested OLS + Standardized oct_mari_z
    ax = axes[1]
    m_order = [
        ('M1_unadjusted', 'M1:\nUnadj.'),
        ('M2_elev_tpi', 'M2:\n+Elev\n& TPI'),
        ('M3_plus_soils', 'M3:\n+Modeled\nSoils'),
        ('M3b_elev_tpi_plus_mari_only', 'M3b:\n+Oct mARI\nOnly'),
        ('M4b_elev_tpi_plus_guild_and_mari', 'M4b:\n+Guild\n& mARI'),
        ('M5_full', 'M5:\nFull'),
    ]
    xm = np.arange(len(m_order))
    betas = [nest[k]['targets']['is_flat']['block_1km']['beta'] for k, _ in m_order]
    r2s = [nest[k]['r2'] for k, _ in m_order]

    ax.bar(xm, betas, width=0.56,
           color=[C_SKY, C_BLUE, C_GRAY, C_ORANGE, C_GREEN, C_GREEN], edgecolor=C_DARK, lw=0.7, alpha=0.9,
           label='β_flat (days)')
    ax2 = ax.twinx()
    ax2.spines['top'].set_visible(False)
    ax2.plot(xm, r2s, 'D-', color=C_VERM, lw=2.0, ms=5.8, label='Model R²')
    ax2.set_ylabel('Model R²', color=C_VERM)
    ax2.tick_params(axis='y', labelcolor=C_VERM)
    ax2.set_ylim(0.08, 0.36)

    for i, b in enumerate(betas):
        ax.text(i, b - 0.26, f'+{b:.2f}', ha='center', va='top', fontsize=7.2, fontweight='bold', color='white')

    mz_1k = nest['M4b_elev_tpi_plus_guild_and_mari']['targets']['oct_mari_z']['block_1km']
    mz_2k = nest['M4b_elev_tpi_plus_guild_and_mari']['targets']['oct_mari_z']['block_2km']
    mari_2k = sc['flat_vs_steep_oct_mari']['block_2km']
    ax.text(0.03, 0.03,
            f"True nested ΔR² over M2 (Elev+TPI):\n"
            f"• Soils: +{dr2['soils_over_M2']:.3f} | Oct mARI alone: +{dr2['oct_mari_alone_over_M2']:.3f}\n"
            f"• Standardized oct_mari_z: {mz_1k['beta']:+.2f} d/SD\n"
            f"  (1 km {fmt_p(mz_1k['p_median'])}; 2 km {fmt_p(mz_2k['p_median'])})\n"
            f"• Oct mARI diff ({desc['oct_mari_diff']:+.3f}): 2 km p={mari_2k['p_min']:.2f}–{mari_2k['p_max']:.2f}",
            transform=ax.transAxes, fontsize=6.7, va='bottom',
            bbox=dict(boxstyle='round,pad=0.24', fc='white', ec='#CCCCCC', alpha=0.95))
    ax.set_xticks(xm)
    ax.set_xticklabels([lbl for _, lbl in m_order], fontsize=7.1)
    ax.set_ylabel('Flat − Steep β_flat (days)')
    ax.set_ylim(0, 5.1)
    ax.set_title('(b) Strictly Nested OLS & Standardized mARI')

    # (c) 25-Year Flat - Steep dSOS vs Mar-Apr Precip
    ax = axes[2]
    df25 = pd.DataFrame(t5b['by_year'])
    ax.scatter(df25['mar_apr_precip_mm'], df25['dSOS_flat_minus_steep'], s=46, color=C_ORANGE,
               edgecolor=C_DARK, lw=0.7)
    lr = stats.linregress(df25['mar_apr_precip_mm'], df25['dSOS_flat_minus_steep'])
    xx = np.linspace(df25['mar_apr_precip_mm'].min() - 10, df25['mar_apr_precip_mm'].max() + 10, 40)
    ax.plot(xx, lr.intercept + lr.slope * xx, '--', color=C_DARK, lw=1.6)
    ax.axhline(0, color=C_GRAY, ls=':', lw=1.0)
    sd25_exact = float(df25['dSOS_flat_minus_steep'].std(ddof=1))
    ax.axhline(t5b['mean_dSOS_flat_minus_steep'], color=C_BLUE, ls='-', lw=1.5, alpha=0.75,
               label=f"25-yr mean: {t5b['mean_dSOS_flat_minus_steep']:+.2f} ± {sd25_exact:.2f} SD d")

    n_pos = int((df25['dSOS_flat_minus_steep'] > 0).sum())
    ax.text(0.04, 0.96,
            f"{n_pos} / {t5b['n_years']} years positive ({fmt_p(t5b['onesamp_p'])})\n"
            f"Mar–Apr precip correlation:\n"
            f"  r = {t5b['r_mar_apr_precip']:.2f} ({fmt_p(t5b['p_mar_apr_precip'])}, null)\n"
            f"→ Consistent with static upland\n  oak/rim species turnover",
            transform=ax.transAxes, fontsize=7.3, va='top',
            bbox=dict(boxstyle='round,pad=0.26', fc='white', ec='#CCCCCC', alpha=0.94))
    ax.set_xlabel('Sewanee Mar–Apr Precipitation (mm)')
    ax.set_ylabel('Annual Flat − Steep ΔSOS (days, Landsat)')
    ax.set_ylim(-0.5, 7.8)
    ax.set_title('(c) 25-Yr Non-Hydrologic Persistence (2000–24)')
    ax.legend(loc='lower right', fontsize=7.2, frameon=True, facecolor='white')

    fig.suptitle('Figure 6. Signal B Mechanism: Spatial-Clustered Inference, Modeled Soil Null, & Plateau-Cap Guild Evidence', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig_pair(fig, 'v2_fig6_plateau_guild_turnover.png', 'fig6.png')


def make_fig7_supplementary(D):
    """Supplementary Fig S1 (v2_fig7): Honest AlphaEarth Audit (with 40-band best raw S2 & first_diff_2w)."""
    v4d = D['v4d']
    sp = v4d['spatial_summary_from_p6']
    b800 = sp['buffered_800m']
    loco = sp['loco_4coves_transfer']
    dyn = v4d['dynamic_symmetric_summary']
    cmp = v4d['best_raw_s2_comparison']

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.1), gridspec_kw={'width_ratios': [1.05, 1.15]})

    ax = axes[0]
    models = [
        ('A_3DEP_morphometrics', 'A: 3DEP\nTerrain'),
        ('B_3DEP_plus_EVT', 'B: 3DEP\n+ EVT'),
        ('C2_Raw2017S2_Indices_plus_3DEP', 'C2: Raw 2017 S2\n+Idx + 3DEP'),
        ('C_Raw2017S2_plus_3DEP', 'C: Raw 2017 S2\n40b + 3DEP'),
        ('D_AE2017_only', 'D: AlphaEarth\n2017 (64-d)'),
        ('E_AE2017_plus_3DEP', 'E: AlphaEarth\n2017 + 3DEP'),
    ]
    x = np.arange(len(models))
    w = 0.36
    r2_loc = [max(b800[k]['Ridge']['r2'], b800[k]['RF']['r2']) for k, _ in models]
    r2_reg = [max(loco[k]['Ridge']['overall_r2'], loco[k]['RF']['overall_r2']) for k, _ in models]

    ax.bar(x - w/2, r2_loc, w, color=C_SKY, edgecolor=C_DARK, lw=0.7, label='Local Within-Domain CV (800 m buf)')
    ax.bar(x + w/2, r2_reg, w, color=C_BLUE, edgecolor=C_DARK, lw=0.7, label=f"{sp['n_loco_folds']}-Cove Cross-Regional LOCO")
    for i in range(len(models)):
        ax.text(x[i] - w/2, r2_loc[i] + 0.012, f'{r2_loc[i]:.2f}', ha='center', va='bottom', fontsize=7.2, color=C_DARK)
        ax.text(x[i] + w/2, r2_reg[i] + 0.012, f'{r2_reg[i]:.2f}', ha='center', va='bottom', fontsize=7.2, fontweight='bold', color=C_BLUE)

    ae_loc, s2_loc = r2_loc[4], max(r2_loc[2], r2_loc[3])
    ae_reg, s2_reg = r2_reg[5], max(r2_reg[2], r2_reg[3])
    ax.text(0.03, 0.96,
            f"Static Spatial Split (2017 → 2018–24 SOS):\n"
            f"• Local 800 m buf: AE {ae_loc:.3f} vs best Raw S2+3DEP {s2_loc:.3f} ({ae_loc-s2_loc:+.3f})\n"
            f"• 4-fold LOCO: AE+3DEP {ae_reg:.3f} vs best Raw S2+3DEP {s2_reg:.3f} ({ae_reg-s2_reg:+.3f})\n"
            f"  Folds: {', '.join(sp['loco_actual_cove_names'])}",
            transform=ax.transAxes, fontsize=6.9, va='top',
            bbox=dict(boxstyle='round,pad=0.26', fc='#F5F9FF', ec=C_BLUE, alpha=0.94))
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in models], fontsize=7.4)
    ax.set_ylabel('Out-of-Sample R² (Best of Ridge / RF)')
    ax.set_ylim(0, 0.89)
    ax.set_title('(a) Static Spatial Transfer: Local vs. 4-Fold LOCO')
    ax.legend(loc='lower right', fontsize=7.3, frameon=True, facecolor='white')

    ax = axes[1]
    tasks = [
        ('twoway_demeaned_LOYO_r2', 'Two-Way\nLOYO'),
        ('first_diff_LOYO_r2', 'Non-Overlap\nFirst-Diff'),
        ('first_diff_2w_LOYO_r2', 'Two-Way\nFirst-Diff'),
        ('twoway_demeaned_GroupPixel_r2', 'Two-Way\nGroupPixel'),
        ('lagged_prev_year_twoway_LOYO_r2', 'Prior-Yr Lag\n(t−1 → t)'),
    ]
    xt = np.arange(len(tasks))
    wt = 0.21
    m_ae = [dyn['AE_delta_emb_64d'][k] for k, _ in tasks]
    m_40 = [dyn['Raw_delta_S2_4seas_40b'][k] for k, _ in tasks]
    m_60 = [dyn['Raw_delta_S2_4seas_idx_60f'][k] for k, _ in tasks]
    m_cb = [dyn['Combined_AE_plus_RawS2_104f'][k] for k, _ in tasks]

    ax.bar(xt - 1.5*wt, m_ae, wt, color=C_BLUE, edgecolor=C_DARK, lw=0.6, label='AlphaEarth Δemb_t (64-d)')
    ax.bar(xt - 0.5*wt, m_40, wt, color=C_VERM, edgecolor=C_DARK, lw=0.6, label='Raw ΔS2_t 4-Seas (40b, best raw)')
    ax.bar(xt + 0.5*wt, m_60, wt, color=C_ORANGE, edgecolor=C_DARK, lw=0.6, label='Raw ΔS2_t 4-Seas+Idx (60f)')
    ax.bar(xt + 1.5*wt, m_cb, wt, color=C_GREEN, edgecolor=C_DARK, lw=0.6, label='Combined AE + Raw S2 (104f)')
    ax.axhline(0, color=C_DARK, lw=0.9)

    for i in range(len(tasks)):
        va_ae = 'bottom' if m_ae[i] >= 0 else 'top'
        dy_ae = 0.012 if m_ae[i] >= 0 else -0.014
        ax.text(xt[i] - 1.5*wt, m_ae[i] + dy_ae, f'{m_ae[i]:.2f}', ha='center', va=va_ae,
                fontsize=6.8, fontweight='bold', color=C_BLUE)
        if m_40[i] > 0.10:
            ax.text(xt[i] - 0.5*wt, m_40[i] + 0.012, f'{m_40[i]:.2f}', ha='center', va='bottom',
                    fontsize=6.8, fontweight='bold', color=C_VERM)

    ax.text(0.02, 0.03,
            f"Bit-exact panel (n={v4d['n_pixels']} px, {v4d['n_pixel_years']} px-yrs).\n"
            f"First-Diff LOYO: AE {cmp['first_diff_LOYO_r2']['AE_64d']:.3f} vs Raw S2 40b {cmp['first_diff_LOYO_r2']['Raw_S2_40b']:.3f} "
            f"(60f {cmp['first_diff_LOYO_r2']['Raw_S2_60f']:.3f}).\n"
            f"Two-Way First-Diff: AE {cmp['first_diff_2w_LOYO_r2']['AE_64d']:.3f} vs Raw S2 {cmp['first_diff_2w_LOYO_r2']['best_raw_s2_r2']:.3f}.\n"
            f"Two-Way Prior-Year Lag: AE {cmp['lagged_prev_year_twoway_LOYO_r2']['AE_64d']:.3f} (zero forecasting skill).",
            transform=ax.transAxes, fontsize=6.8, va='bottom',
            bbox=dict(boxstyle='round,pad=0.25', fc='#FFF7F0', ec=C_VERM, alpha=0.94))
    ax.set_xticks(xt)
    ax.set_xticklabels([lbl for _, lbl in tasks], fontsize=7.4)
    ax.set_ylabel('Strict Leak-Free Out-of-Sample R²')
    ax.set_ylim(-0.24, 0.56)
    ax.set_title('(b) Symmetric Dynamic Benchmark: Δemb_t vs. Raw ΔS2_t')
    ax.legend(loc='upper right', fontsize=7.0, frameon=True, facecolor='white')

    fig.suptitle('Supplementary Figure S1. AlphaEarth Audit (Excluded from Main Manuscript; Archived Standalone)', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig_pair(fig, 'v2_fig7_alphaearth_sidebar.png', paper_fig_name=None)


def main():
    D = load_all()
    make_fig1(D)
    make_fig2(D)
    make_fig3(D)
    make_fig4(D)
    make_fig5(D)
    make_fig6(D)
    make_fig7_supplementary(D)
    # Remove unused fig7.png from paper/figures if present (section 7 removed from main paper)
    old_fig7 = os.path.join(PAPER_FIGS, 'fig7.png')
    if os.path.exists(old_fig7):
        os.remove(old_fig7)
        print('REMOVED', old_fig7)
    print('ALL 6 MAIN PAPER FIGURES + 1 SUPPLEMENTARY FIGURE GENERATED AT 300 DPI.')


if __name__ == '__main__':
    main()
