"""Consolidate Stage 3 (profiling/p1..p6.json) + Stage 4 (data/v2_*.json) into one authoritative
strict RFC 8259 JSON master summary and hypothesis/falsification ledger.

Outputs:
  data/v2_master_summary.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'profiling'))
import common as C

OUT_PATH = os.path.join(HERE, 'data', 'v2_master_summary.json')


def main():
    p1 = json.load(open(os.path.join(HERE, 'profiling', 'p1.json')))
    p2 = json.load(open(os.path.join(HERE, 'profiling', 'p2.json')))
    p3 = json.load(open(os.path.join(HERE, 'profiling', 'p3.json')))
    p4 = json.load(open(os.path.join(HERE, 'profiling', 'p4.json')))
    p5 = json.load(open(os.path.join(HERE, 'profiling', 'p5.json')))
    p6 = json.load(open(os.path.join(HERE, 'profiling', 'p6.json')))

    v4a = json.load(open(os.path.join(HERE, 'data', 'v2_p1_evt_dose_response.json')))
    v4b = json.load(open(os.path.join(HERE, 'data', 'v2_ecostress_thermal_closure.json')))
    v4c = json.load(open(os.path.join(HERE, 'data', 'v2_hopkins_climate_lal.json')))
    v4d = json.load(open(os.path.join(HERE, 'data', 'v2_alphaearth_symmetric_dynamic.json')))
    v4e = json.load(open(os.path.join(HERE, 'data', 'v2_signalb_audit.json')))

    r_cr1 = v4a['relief_and_rim_free_analysis']['relief_adjusted_cr1_ols']['ndre']['all_11_pairs']
    r_rf = v4a['relief_and_rim_free_analysis']['rim_free_floor_vs_floor']['ndre']
    r_rc = v4a['relief_and_rim_free_analysis']['relief_controlled_closure_models']['ndre']['dGSL']
    m_dec = v4c.get('modis_3factor_decomposition', {})
    m_del = m_dec.get('sequential_deltas_d_per_100m', {})
    sb_c = v4e['spatial_clustered_contrasts']
    sb_n = v4e['nested_ols_standardized_mari']
    ae_cmp = v4d['best_raw_s2_comparison']

    # Master Hypothesis & Falsification Ledger (honest internal decision criteria, zero overclaiming)
    ledger = [
        {
            'id': 'H1_hopkins_baseline',
            'claim': '50% deciduous escarpment green-up (MODIS MidGreenup_1) tracks Hopkins (1918) 3.28 d/100 m; 15% Greenup_1 lapses faster due to phenological threshold definition rather than QA/conifer artifacts',
            'status': 'SUPPORTED',
            'key_metric': (
                f"Clean MODIS MidGreenup_1 (2001-2023, QA<=1, NLCD-41): lapse = "
                f"{v4c['clean_modis_2001_2023']['lapse_rate_trend']['mean']:.2f} +/- "
                f"{v4c['clean_modis_2001_2023']['lapse_rate_trend']['se']:.2f} SE "
                f"({v4c['clean_modis_2001_2023']['lapse_rate_trend']['sd']:.2f} SD) d/100 m "
                f"(Hopkins = 3.28); 3-factor GEE decomposition: QA delta={m_del.get('delta_qa_filter_B_minus_A')}, "
                f"NLCD-41 delta={m_del.get('delta_nlcd41_mask_C_minus_B')}, "
                f"50%-vs-15% delta={m_del.get('delta_50pct_vs_15pct_D_minus_C')} d/100 m; "
                f"lapse trend = {v4c['clean_modis_2001_2023']['lapse_rate_trend']['ols_per_decade']:+.3f} d/100m/dec "
                f"(p={v4c['clean_modis_2001_2023']['lapse_rate_trend']['ols_p']}); "
                f"thermal sens = {v4c['climate_context']['modis_thermal_sensitivity']['days_per_degC_spring']:.2f} +/- "
                f"{v4c['climate_context']['modis_thermal_sensitivity']['se_days_per_degC']:.2f} SE d/C "
                f"(r={v4c['climate_context']['modis_thermal_sensitivity']['r']}, p={v4c['climate_context']['modis_thermal_sensitivity']['p']}); "
                f"unharmonized Landsat 2013 step non-identifiable due to collinearity (step=-0.42 +/- 3.69 SE d, p=0.91); "
                f"raw GHCN-Daily Sewanee (unhomogenized for TOBS/hiatus): Mar-Apr +{v4c['climate_context']['noaa_modern_vs_hopkins_era_shift']['spr_tmean_C']['shift_modern_minus_hopkins']:.2f} C "
                f"(p={v4c['climate_context']['noaa_modern_vs_hopkins_era_shift']['spr_tmean_C']['welch_p']}), "
                f"frost-free +{v4c['climate_context']['noaa_modern_vs_hopkins_era_shift']['frost_free_days']['shift_modern_minus_hopkins']:.1f} d "
                f"(p={v4c['climate_context']['noaa_modern_vs_hopkins_era_shift']['frost_free_days']['welch_p']})"
            ),
        },
        {
            'id': 'H2_brdf_artifact_immunity',
            'claim': 'Signal A (Lost Cove) and Signal B (Plateau cap) are not solar-geometry/BRDF or timber-harvest artifacts (internal gate P2 = PARTIAL_OR_FAIL due to sigA_pass=false & decoupling fail)',
            'status': 'PARTIAL_OR_FAIL',
            'internal_gate_verdict': p2['gate_check']['verdict'],
            'key_metric': (
                f"Signal A SCS+C retention: {p2['gate_check']['sigA_scsc_retention_SOS_pct']}% SOS / "
                f"{p2['gate_check']['sigA_scsc_retention_EOS_pct']}% EOS; HLS v2.0 NDVI dSOS={p2['gate_check']['sigA_hls_ndvi_dSOS']:+.2f} d, "
                f"NDRE dSOS={p2['gate_check']['sigA_hls_ndre_dSOS']:+.2f} d, dEOS={p2['gate_check']['sigA_hls_ndre_dEOS']:+.2f} d "
                f"(internal gate P2 sigA_pass={p2['gate_check']['sigA_pass']}); "
                f"Signal B persists across all 4 aspect quadrants (N/E/S/W: +2.30 to +2.61 d SCS+C NDRE; +2.77 to +3.32 d HLS NDVI) "
                f"and after masking all 120 LAL harvest polygons (Sig A descriptive N=1 cove diff={v4c['lal_public_controls']['signal_A_harvest_sensitivity']['excluding_all_harvests']['diff_lost_minus_openn_d']:+.2f} d; "
                f"Sig B={sb_c['lal_excl_all_harvests_flat_vs_steep_sos']['block_1km']['beta']:+.2f} d, "
                f"1km block med p={sb_c['lal_excl_all_harvests_flat_vs_steep_sos']['block_1km']['p_median']}, "
                f"2km block med p={sb_c['lal_excl_all_harvests_flat_vs_steep_sos']['block_2km']['p_median']})"
            ),
        },
        {
            'id': 'H3_regional_karst_compression',
            'claim': 'Closed karst basins regionally compress deciduous growing season vs. matched open gorges after relief adjustment and rim-free floor-vs-floor comparison, inside identical EVTs (internal gate P1 = PARTIAL_PASS)',
            'status': 'PARTIAL_PASS',
            'internal_gate_verdict': p1['gate_check']['verdict'],
            'key_metric': (
                f"11 regional closed-open pairs (NDRE): relief-adjusted 22-site CR1 OLS (df=10): "
                f"dSOS={r_cr1['dSOS']['is_closed']['beta']:+.2f} +/- {r_cr1['dSOS']['is_closed']['se']:.2f} SE d (p={r_cr1['dSOS']['is_closed']['p']}), "
                f"dEOS={r_cr1['dEOS']['is_closed']['beta']:+.2f} +/- {r_cr1['dEOS']['is_closed']['se']:.2f} SE d (p={r_cr1['dEOS']['is_closed']['p']}), "
                f"dGSL={r_cr1['dGSL']['is_closed']['beta']:+.2f} +/- {r_cr1['dGSL']['is_closed']['se']:.2f} SE d (p={r_cr1['dGSL']['is_closed']['p']}; "
                f"unadjusted paired DiD={p1['phenology_power']['ndre']['paired_DiD_dGSL_mean']:+.2f} d, inflated by 51.8 m relief imbalance); "
                f"rim-free dz_floor-adjusted floor-vs-floor: dGSL={r_rf['all_11_pairs']['dGSL']['dz_floor_adjusted']['dz_adj_mean']:+.2f} +/- "
                f"{r_rf['all_11_pairs']['dGSL']['dz_floor_adjusted']['dz_adj_se']:.2f} SE d (p={r_rf['all_11_pairs']['dGSL']['dz_floor_adjusted']['dz_adj_p']}); "
                f"within-EVT dz_floor-adjusted: EVT 3317 dGSL={r_rf['within_evt']['EVT3317_oak']['dGSL']['dz_floor_adjusted']['dz_adj_mean']:+.2f} +/- "
                f"{r_rf['within_evt']['EVT3317_oak']['dGSL']['dz_floor_adjusted']['dz_adj_se']:.2f} SE d (p={r_rf['within_evt']['EVT3317_oak']['dGSL']['dz_floor_adjusted']['dz_adj_p']}), "
                f"EVT 3321 dGSL={r_rf['within_evt']['EVT3321_cove']['dGSL']['dz_floor_adjusted']['dz_adj_mean']:+.2f} +/- "
                f"{r_rf['within_evt']['EVT3321_cove']['dGSL']['dz_floor_adjusted']['dz_adj_se']:.2f} SE d (p={r_rf['within_evt']['EVT3321_cove']['dGSL']['dz_floor_adjusted']['dz_adj_p']}); "
                f"binary is_closed partial R2={r_rc['partial_r2_is_closed_over_relief']*100:.1f}% over Relief (total R2={r_rc['M_relief_is_closed']['r2']:.3f}, "
                f"dAICc={r_rc['delta_aicc_relief_closed_minus_relief_tpi300']} vs Relief+TPI300; no depth ramp >22 m, 0-22 m unobserved); "
                f"P1 internal gate PARTIAL_PASS (open dEOS=-3.03 d, r_sigmaSOS=0.22)"
            ),
        },
        {
            'id': 'H4_ecostress_thermal_mechanism',
            'claim': 'Closed karst floors exhibit nocturnal cold-air pooling in cloud-leak-filtered ECOSTRESS TIR across seasons and 7/7 16SED coves, collapsing at Lost Cove over 2.5-3.0 m/s',
            'status': 'SUPPORTED_WITH_MANSKI_CAVEAT',
            'key_metric': (
                f"Cloud-leak-filtered (|LST-ERA5_t2m|<=10 C, dropping 8 corrupted/cloud v002 scenes, 0/14 v003) "
                f"n={v4b['archive_counts']['combined_unique_night_clean']} clean / {v4b['archive_counts']['combined_unique_night_in_swath']} in-swath nights: "
                f"Spring={v4b['lost_cove_night_stats']['spring_feb_apr_night']['mean_C']:+.2f} +/- {v4b['lost_cove_night_stats']['spring_feb_apr_night']['se_C']:.2f} SE C "
                f"(n={v4b['lost_cove_night_stats']['spring_feb_apr_night']['n']}, t p={v4b['lost_cove_night_stats']['spring_feb_apr_night']['p']}, "
                f"Wilcoxon p={v4b['lost_cove_night_stats']['spring_feb_apr_night']['wilcoxon_p']}); "
                f"Autumn={v4b['lost_cove_night_stats']['autumn_sep_nov_night']['mean_C']:+.2f} +/- {v4b['lost_cove_night_stats']['autumn_sep_nov_night']['se_C']:.2f} SE C "
                f"(n={v4b['lost_cove_night_stats']['autumn_sep_nov_night']['n']}, t p={v4b['lost_cove_night_stats']['autumn_sep_nov_night']['p']}); "
                f"Lost Cove wind collapse (2.5-3.0 m/s only, n=0 >=3.0 m/s): calm (<2.5 m/s, n={v4b['wind_placebo']['calm_lt2p5ms']['n']})="
                f"{v4b['wind_placebo']['calm_lt2p5ms']['mean_C']:+.2f} +/- {v4b['wind_placebo']['calm_lt2p5ms']['se_C']:.2f} SE C vs "
                f"breezy (2.5-3.0 m/s, n={v4b['wind_placebo']['breezy_2p5_3p0ms']['n']})="
                f"{v4b['wind_placebo']['breezy_2p5_3p0ms']['mean_C']:+.2f} +/- {v4b['wind_placebo']['breezy_2p5_3p0ms']['se_C']:.2f} SE C "
                f"(Welch p={v4b['wind_placebo']['continuous_wind_regression']['calm_vs_breezy2p5_welch_p']}, "
                f"MW p={v4b['wind_placebo']['continuous_wind_regression']['calm_vs_breezy2p5_mannwhitney_p']}; "
                f"does not replicate across 6 other 16SED coves using Sewanee wind, p={v4b['multicove_16sed_replication']['six_other_coves_calm_vs_breezy_paired_p']}); "
                f"Multi-cove 16SED replication: 7/7 coves individually significant (cove-level calm mean="
                f"{v4b['multicove_16sed_replication']['cove_level_calm_mean_C']:+.2f} +/- {v4b['multicove_16sed_replication']['cove_level_calm_se_C']:.2f} SE C, "
                f"p={v4b['multicove_16sed_replication']['cove_level_calm_p']}); "
                f"CAVEAT: swath Manski bounds span 0 (combined cloud-only [{v4b['manski_bounds']['all_combined_nights']['swath_conditioned']['manski_lo']:+.2f}, {v4b['manski_bounds']['all_combined_nights']['swath_conditioned']['manski_hi']:+.2f} C]; "
                f"v002 [{v4b['manski_bounds']['v002_only_nights']['swath_conditioned']['manski_lo']:+.2f}, {v4b['manski_bounds']['v002_only_nights']['swath_conditioned']['manski_hi']:+.2f} C]; "
                f"v003 [{v4b['manski_bounds']['v003_only_nights']['swath_conditioned']['manski_lo']:+.2f}, {v4b['manski_bounds']['v003_only_nights']['swath_conditioned']['manski_hi']:+.2f} C])"
            ),
        },
        {
            'id': 'H5_signalb_guild_vs_fragipan',
            'claim': 'Signal B (Flat Plateau spring lag vs Steep Rim) is consistent with upland oak guild turnover rather than modeled wet-fragipan soil thermal inertia',
            'status': 'FRAGIPAN_FALSIFIED_GUILD_CONSISTENT',
            'key_metric': (
                f"Modeled fragipan soil inertia FALSIFIED: within-flat fragipan vs well-drained = "
                f"{sb_c['within_flat_fragipan_vs_welldrained_sos']['block_1km']['beta']:+.2f} d "
                f"(1km med p={sb_c['within_flat_fragipan_vs_welldrained_sos']['block_1km']['p_median']}, "
                f"2km med p={sb_c['within_flat_fragipan_vs_welldrained_sos']['block_2km']['p_median']}), "
                f"25-yr Mar-Apr precip r={p5['gate_check']['r_25yr_mar_apr_precip']} (p=0.33); "
                f"Plateau lag robust across all 25 grid origins at 1km & 2km: raw {sb_c['raw_flat_vs_steep_sos']['block_1km']['beta']:+.2f} d "
                f"(2km p<={sb_c['raw_flat_vs_steep_sos']['block_2km']['p_max']}), well-drained {sb_c['well_drained_only_flat_vs_steep_sos']['block_1km']['beta']:+.2f} d "
                f"(2km p<={sb_c['well_drained_only_flat_vs_steep_sos']['block_2km']['p_max']}), unharvested LAL {sb_c['lal_excl_all_harvests_flat_vs_steep_sos']['block_1km']['beta']:+.2f} d "
                f"(2km p<={sb_c['lal_excl_all_harvests_flat_vs_steep_sos']['block_2km']['p_max']}), 25/25 Landsat yrs (+2.39 +/- 1.61 SD d); "
                f"Consistent with oak guild turnover (no species-plot data): standardized oct_mari_z beta={sb_n['M4b_elev_tpi_plus_guild_and_mari']['targets']['oct_mari_z']['block_1km']['beta']:+.2f} d/SD "
                f"(single-var nested dR2={sb_n['true_nested_delta_r2']['oct_mari_alone_over_M2']:+.3f}; "
                f"oct_mari diff +0.120 is 2km grid-sensitive: p={sb_c['flat_vs_steep_oct_mari']['block_2km']['p_min']}..{sb_c['flat_vs_steep_oct_mari']['block_2km']['p_max']}, "
                f"{sb_c['flat_vs_steep_oct_mari']['block_2km']['frac_origins_p_lt_0p05']*100:.0f}% < 0.05)"
            ),
        },
        {
            'id': 'H6_s2_vertical_decoupling_gate',
            'claim': 'Internal-criterion S2 vertical canopy-understory decoupling (dSOS_NDVI - dSOS_NDRE) tracks April freeze severity',
            'status': 'FALSIFIED',
            'key_metric': (
                f"Internal S2 April freeze gate FAILED (freeze vs mild p={p2['vertical_decoupling']['scsc']['freeze_vs_mild_p']}, "
                f"r_AprTmin={p2['vertical_decoupling']['scsc']['r_with_apr_tmin_min']}); "
                f"HLS showed r=-0.785 (p=0.036); post-hoc DOY>100 frost split (+1.70 vs -0.84 d, p=0.025) is exploratory only (n=7)"
            ),
        },
        {
            'id': 'H7_p4_25yr_calm_freeze_modulation',
            'claim': 'Interannual CalmOctFreeze nights amplify closed-cove early EOS across a 25-yr (2000-2024) 4-cove Landsat panel',
            'status': 'FALSIFIED',
            'key_metric': (
                f"Lost Cove 25-yr medians +2.45 d SOS (88% >0) / -2.53 d EOS (80% <0), though OLS means are n.s. "
                f"(+1.62 +/- 1.42 SE d p=0.27; -2.22 +/- 1.34 SE d p=0.11); 2018-2024 vs HLS EOS r=0.961 (p=0.00056) "
                f"but spring SOS r=-0.041 (p=0.93); 4-cove panel CalmOctFreeze_z showed OPPOSITE sign "
                f"(beta={p4['gate_check']['beta_CalmOctFreeze_z']:+.2f} d, p={p4['gate_check']['p_CalmOctFreeze_z']}; "
                f"HardRegionalFreeze beta=+1.85 d, p=0.014); Lost Cove OctNightWspd n.s. (r=0.374, p=0.065)"
            ),
        },
        {
            'id': 'H8_alphaearth_supplementary_audit',
            'claim': 'Standalone supplementary audit: AlphaEarth 64-d annual embeddings vs. matched-capacity Raw Sentinel-2 + 3DEP (excluded from main manuscript)',
            'status': 'SUPPLEMENTARY_SPLIT_VERDICT',
            'key_metric': (
                f"Excluded from main manuscript per peer review; archived standalone in v2_alphaearth_symmetric_dynamic.py. "
                f"Loses locally (800m buf CV: AE=0.578 vs Raw S2+3DEP=0.611); wins 4-fold LOCO "
                f"({', '.join(v4d['spatial_summary_from_p6']['loco_actual_cove_names'])}: AE+3DEP=0.681 vs Raw S2+3DEP=0.514); "
                f"dynamic first-diff LOYO: AE={ae_cmp['first_diff_LOYO_r2']['AE_64d']:.3f} vs best Raw S2 (40b)={ae_cmp['first_diff_LOYO_r2']['Raw_S2_40b']:.3f} "
                f"(60f={ae_cmp['first_diff_LOYO_r2']['Raw_S2_60f']:.3f}); two-way first-diff LOYO: AE={ae_cmp['first_diff_2w_LOYO_r2']['AE_64d']:.3f} vs "
                f"best Raw S2={ae_cmp['first_diff_2w_LOYO_r2']['best_raw_s2_r2']:.3f}; zero prior-year two-way lag skill (AE=-0.024)"
            ),
        },
    ]

    master = {
        'project': 'Southern Cumberland Escarpment Karst Cold-Air Pooling & Plateau-Cap Phenology (v2 Revised)',
        'hypothesis_ledger': ledger,
        'stage3_internal_gates': {
            'P1': p1['gate_check'],
            'P2': p2['gate_check'],
            'P3': p3['gate_check'],
            'P4': p4['gate_check'],
            'P5': p5['gate_check'],
            'P6': p6['gate_check'],
        },
        'stage4_closures': {
            '4A_within_evt_relief_and_closure': {
                'p1_gate_status': v4a['p1_gate_status'],
                'disqualified_basins_archive': v4a['disqualified_basins_archive'],
                'relief_diagnostics': v4a['relief_and_rim_free_analysis']['relief_diagnostics'],
                'relief_adjusted_cr1_ndre': v4a['relief_and_rim_free_analysis']['relief_adjusted_cr1_ols']['ndre']['all_11_pairs'],
                'relief_adjusted_cr1_ndvi': v4a['relief_and_rim_free_analysis']['relief_adjusted_cr1_ols']['ndvi']['all_11_pairs'],
                'rim_free_floor_vs_floor_ndre': v4a['relief_and_rim_free_analysis']['rim_free_floor_vs_floor']['ndre'],
                'relief_controlled_closure_dGSL_ndre': v4a['relief_and_rim_free_analysis']['relief_controlled_closure_models']['ndre']['dGSL'],
                'within_closed_depth_slope_ndre': v4a['relief_and_rim_free_analysis']['within_closed_depth_slope']['ndre'],
                'evt_composition': v4a['within_evt_analysis']['evt_composition'],
            },
            '4B_ecostress_thermal_closure': {
                'cloud_leak_filter': v4b['cloud_leak_filter'],
                'archive_counts': v4b['archive_counts'],
                'unfiltered_diagnostics': {k: v for k, v in v4b['unfiltered_diagnostics'].items() if k != 'dropped_scenes'},
                'lost_cove_night_stats': v4b['lost_cove_night_stats'],
                'wind_placebo': v4b['wind_placebo'],
                'multicove_16sed': {k: v for k, v in v4b['multicove_16sed_replication'].items() if k not in ('per_cove', 'per_cove_excl_flagged_only', 'per_cove_unfiltered')},
                'manski_all': v4b['manski_bounds']['all_combined_nights'],
                'manski_v002': v4b['manski_bounds']['v002_only_nights'],
                'manski_v003': v4b['manski_bounds']['v003_only_nights'],
                'manski_autumn': v4b['manski_bounds']['autumn_sep_nov'],
            },
            '4C_hopkins_climate_lal': {
                'clean_modis_midgreenup_trend': v4c['clean_modis_2001_2023']['midgreenup_trend'],
                'clean_modis_lapse_rate': v4c['clean_modis_2001_2023']['lapse_rate_trend'],
                'modis_3factor_decomposition': {k: v for k, v in m_dec.items() if k != 'per_year'},
                'legacy_landsat_step': v4c['legacy_landsat_step_artifact']['step_dummy_regression'],
                'modis_thermal_sensitivity': v4c['climate_context']['modis_thermal_sensitivity'],
                'noaa_modern_vs_hopkins_era': v4c['climate_context']['noaa_modern_vs_hopkins_era_shift'],
                'noaa_homogenization_caveat': v4c['climate_context']['noaa_homogenization_caveat'],
                'lal_access': v4c['lal_public_controls']['access_status'],
                'lal_harvest_overlap': v4c['lal_public_controls']['harvest_overlap_pct'],
                'signal_A_harvest_sens': v4c['lal_public_controls']['signal_A_harvest_sensitivity'],
                'signal_B_harvest_sens': v4c['lal_public_controls']['signal_B_p5_harvest_sensitivity'],
            },
            '4D_alphaearth_supplementary_audit': {
                'manuscript_role': v4d['manuscript_role'],
                'alignment_audit': v4d['alignment_audit'],
                'n_pixels': v4d['n_pixels'],
                'n_pixel_years': v4d['n_pixel_years'],
                'loco_actual_cove_names': v4d['spatial_summary_from_p6']['loco_actual_cove_names'],
                'best_raw_s2_comparison': v4d['best_raw_s2_comparison'],
                'dynamic_symmetric_summary': v4d['dynamic_symmetric_summary'],
            },
            '4E_signalb_spatial_audit': {
                'descriptives': v4e['descriptives'],
                'spatial_clustered_contrasts': v4e['spatial_clustered_contrasts'],
                'true_nested_delta_r2': v4e['nested_ols_standardized_mari']['true_nested_delta_r2'],
                'regional_3site_plateau_replication': v4e['regional_3site_plateau_replication'],
            },
        },
    }

    clean = C.sanitize_json(master)
    with open(OUT_PATH, 'w') as f:
        json.dump(clean, f, indent=2, allow_nan=False)
    print(f'WROTE {OUT_PATH}')
    for row in ledger:
        print(f"  [{row['status']:36s}] {row['id']}")


if __name__ == '__main__':
    main()
