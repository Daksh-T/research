"""Aggregate profiling results (P1-P6) into profiling/scorecard.json and print crisp Pass/Fail table."""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import common as C


def load_p(name):
    p = os.path.join(HERE, name)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    p1 = load_p('p1.json')
    p2 = load_p('p2.json')
    p3 = load_p('p3.json')
    p4 = load_p('p4.json')
    p5 = load_p('p5.json')
    p6 = load_p('p6.json')

    scorecard = {
        'P1_karst_census': {
            'status': p1['gate_check']['verdict'] if p1 else 'MISSING',
            'key_metrics': p1['gate_check'] if p1 else None,
            'ndvi_summary': {k: v for k, v in p1['phenology_power']['ndvi'].items() if k != 'per_cove_table'} if p1 else None,
            'ndre_summary': {k: v for k, v in p1['phenology_power']['ndre'].items() if k != 'per_cove_table'} if p1 else None,
            'strict_threshold_sensitivity': p1.get('strict_threshold_sensitivity') if p1 else None,
            'census_counts': p1['census_counts'] if p1 else None,
        },
        'P2_brdf_decoupling': {
            'status': p2['gate_check']['verdict'] if p2 else 'MISSING',
            'key_metrics': p2['gate_check'] if p2 else None,
            'vertical_decoupling_scsc': p2['vertical_decoupling'].get('scsc') if p2 else None,
            'vertical_decoupling_raw': p2['vertical_decoupling'].get('raw') if p2 else None,
            'vertical_decoupling_hls': p2['vertical_decoupling'].get('hls') if p2 else None,
        },
        'P3_ecostress_shadow_v003': {
            'status': p3['gate_check']['whiteman_lag_verdict'] if p3 else 'MISSING',
            'key_metrics': p3['gate_check'] if p3 else None,
            'v002_predawn_all': p3['part_a_v002_shadow_and_synoptic']['predawn_02_06h_all'] if p3 else None,
            'v002_predawn_manski': p3['part_a_v002_shadow_and_synoptic']['predawn_cold_manski'] if p3 else None,
            'v003_summary': {
                k: v for k, v in p3['part_b_live_v003_spring_autumn'].items() if k != 'clean_scene_list'
            } if p3 else None,
        },
        'P4_25yr_weather_panel': {
            'status': p4['gate_check']['verdict'] if p4 else 'MISSING',
            'key_metrics': p4['gate_check'] if p4 else None,
            'weather_thresholds': p4.get('weather_thresholds') if p4 else None,
            'qc80_fe_eos_closed': p4['panel_results']['qc80']['fe_eos_closed'] if p4 else None,
            'qc80_fe_eos_closed_threshold': p4['panel_results']['qc80'].get('fe_eos_closed_threshold_model') if p4 else None,
            'qc80_fe_sos_closed': p4['panel_results']['qc80']['fe_sos_closed'] if p4 else None,
            'qc80_annual_closed_corr': p4['panel_results']['qc80']['annual_closed_corr'] if p4 else None,
            'lost_openn_25yr': {
                k: v for k, v in p4['lost_minus_openn_25yr'].items() if k != 'by_year'
            } if p4 else None,
        },
        'P5_signalb_mechanism': {
            'status': (
                'SOIL_INERTIA' if (p5 and p5['gate_check']['soil_inertia_verdict'] == 'PASS')
                else ('GUILD_TURNOVER' if (p5 and p5['gate_check']['guild_turnover_verdict'] == 'PASS') else 'MIXED_OR_INCONCLUSIVE')
            ) if p5 else 'MISSING',
            'key_metrics': p5['gate_check'] if p5 else None,
            'nested_ols': p5['test_5c_guild_and_pigment']['nested_ols_beta_is_flat'] if p5 else None,
            'precip_25yr': {
                k: v for k, v in p5['test_5b_25yr_precip_interaction'].items() if k != 'by_year'
            } if p5 else None,
        },
        'P6_alphaearth_gate': {
            'status': p6['gate_check']['verdict'] if p6 else 'MISSING',
            'key_metrics': p6['gate_check'] if p6 else None,
            'buffered_800m': p6['spatial_cv_buffered_800m'] if p6 else None,
            'buffered_eff': p6.get('spatial_cv_buffered_eff') if p6 else None,
            'loco_overall_r2': {
                k: {a: v['overall_r2'] for a, v in d.items()} for k, d in p6['loco_4coves_transfer'].items()
            } if p6 else None,
            'interannual_delta_emb': p6['interannual_delta_emb'] if p6 else None,
        }
    }

    out_path = os.path.join(HERE, 'scorecard.json')
    clean_sc = C.sanitize_json(scorecard)
    with open(out_path, 'w') as f:
        json.dump(clean_sc, f, indent=2, allow_nan=False)
    print(f'WROTE {out_path}')
    for k, v in scorecard.items():
        print(f'  {k:28s}: {v["status"]}')


if __name__ == '__main__':
    main()
