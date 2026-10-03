#!/usr/bin/env python3
"""Independently recompute delivered calibration/DCA and explanation summaries."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.diagnostics.uniform_result_scope import sha256
DATA=ROOT/'outputs/maomao_manuscript_figures_20260929/source_data'
def verify():
    dictionary=json.loads((DATA/'clinical_curve_dictionary.json').read_text())
    proof=json.loads((DATA/'clinical_risk_preparation.json').read_text())
    source=ROOT/'outputs/maomao_plot_sources/clinical_score_predictions_no_identifiers.csv'
    assert sha256(source)==proof['source_prediction_sha256']
    frame=pd.read_csv(source);risk=np.load(DATA/'clinical_risk_predictions.npz')
    cal=pd.read_csv(DATA/'clinical_calibration_curves.csv');dca=pd.read_csv(DATA/'clinical_dca_curves.csv')
    raw_cal=pd.read_csv(DATA/'clinical_raw_maomao_calibration_curves.csv');raw_dca=pd.read_csv(DATA/'clinical_raw_maomao_dca_curves.csv')
    series=0;bins=0;thresholds=0
    for endpoint,cohort in dictionary.items():
        columns=['label_'+endpoint]+[s['score_column'] for s in cohort['series']]
        keep=np.isfinite(frame[columns].to_numpy(float)).all(1);part=frame.loc[keep]
        y=part['label_'+endpoint].to_numpy(float)
        assert len(y)==cohort['complete_episodes'] and int(y.sum())==cohort['events']
        for spec in cohort['series']:
            p=risk[spec['prefix']+'__risk'];actual=risk[spec['prefix']+'__target']
            assert np.array_equal(actual,y) and np.isfinite(p).all() and np.all((p>=0)&(p<=1))
            fit=next(f for f in proof['score_mappings'] if f['endpoint']==endpoint and f['model']==spec['name'])
            x=part[spec['score_column']].to_numpy(float)
            assert fit['optimizer_success'] and fit['slope']>=0 and fit['common_training_cohort']
            x=x*spec['risk_direction']
            if fit['input_transform']=='logit':
                assert np.array_equal(risk[spec['prefix']+'__raw_risk'],x)
                x=np.log(np.clip(x,1e-7,1-1e-7)/np.clip(1-x,1e-7,1-1e-7))
            z=fit['intercept']+fit['slope']*(x-fit['mean'])/fit['scale']
            expected=1/(1+np.exp(-z))
            assert np.allclose(p,expected,atol=1e-12,rtol=1e-12)
            if spec['name']=='MAOMAO':
                raw=risk[spec['prefix']+'__raw_risk']
                native=part[spec['score_column']].to_numpy(float)
                assert np.array_equal(raw,native)
                cuts_raw=np.unique(np.quantile(raw,np.linspace(0,1,11)))[1:-1]
                group_raw=np.searchsorted(cuts_raw,raw,side='right')
                for row in raw_cal[raw_cal.endpoint==endpoint].itertuples():
                    take=group_raw==row.bin
                    assert row.n==int(take.sum()) and row.events==int(y[take].sum())
                    assert np.isclose(row.mean_risk,raw[take].mean(),atol=1e-12) and np.isclose(row.observed_fraction,y[take].mean(),atol=1e-12)
                for row in raw_dca[raw_dca.endpoint==endpoint].itertuples():
                    decision=raw>=row.threshold;tp=int(np.sum(decision&(y==1)));fp=int(np.sum(decision&(y==0)))
                    assert row.tp==tp and row.fp==fp and row.n==len(y)
                    assert np.isclose(row.net_benefit,(tp-fp*row.threshold/(1-row.threshold))/len(y),atol=1e-12)
            # Recover tied-value quantile membership independently from saved risks.
            cuts=np.unique(np.quantile(p,np.linspace(0,1,11)))[1:-1]
            membership=np.searchsorted(cuts,p,side='right')
            subset=cal[(cal.endpoint==endpoint)&(cal.model==spec['name'])]
            assert len(subset)==len(np.unique(membership)) and int(subset.n.sum())==len(y)
            for row in subset.itertuples():
                take=membership==row.bin
                assert row.n==int(take.sum()) and row.events==int(y[take].sum())
                assert np.isclose(row.mean_risk,p[take].mean(),atol=1e-12) and np.isclose(row.observed_fraction,y[take].mean(),atol=1e-12)
                bins+=1
            for row in dca[(dca.endpoint==endpoint)&(dca.model==spec['name'])].itertuples():
                positive=p>=row.threshold;tp=int(np.sum(positive&(y==1)));fp=int(np.sum(positive&(y==0)))
                assert tp==row.tp and fp==row.fp and row.n==len(y)
                assert np.isclose(row.net_benefit,(tp-fp*row.threshold/(1-row.threshold))/len(y),atol=1e-12)
                thresholds+=1
            series+=1
    assert series==52 and proof['test_patient_overlap']==0 and not proof['fitted_on_sealed_test']
    assert proof['maomao_calibrated'] and proof['protocol']=='common_training_monotone_risk_mapping_v2'
    for endpoint in dictionary:
        fits=[f for f in proof['score_mappings'] if f['endpoint']==endpoint]
        assert len({(f['train_rows'],f['train_patients'],f['train_events']) for f in fits})==1
    dep=np.load(DATA/'shap_dependence_observed_history.npz')
    assert dep['event_shap_to_log_death'].shape==(9989,210)
    assert np.allclose(dep['baseline_log_death']+dep['event_shap_to_log_death'].sum(1),dep['full_log_death'],atol=1e-4)
    selected=json.loads((DATA/'figure_5_dependency_selection.json').read_text())
    names=dep['event_names'].tolist()
    assert len(selected['selection'])==4
    for pair in selected['selection']:
        take=np.isfinite(dep['last_observed_value'][:,names.index(pair['event'])])&np.isfinite(dep['last_observed_value'][:,names.index(pair['paired_measurement'])])
        assert pair['patients']==int(take.sum()) and pair['all_pair_complete_patients_shown']
    waterfall=pd.read_csv(DATA/'supplementary_S5_waterfall_values.csv')
    assert set(waterfall.case)==set(range(9))
    for _,part in waterfall.groupby('case'):
        assert len(part)==5 and np.isclose(part.baseline.iloc[0]+part.value.sum(),part.full.iloc[0],atol=1e-4)
        assert np.allclose(part.start+part.value,part.end,atol=1e-12)
    result={'complete':True,'clinical_series':series,'calibration_bins':bins,'decision_thresholds':thresholds,
            'same_original_common_cases':True,'raw_maomao_scores_preserved':True,'maomao_training_risk_mapping_recomputed':True,
            'raw_maomao_series':10,'raw_maomao_decision_thresholds':len(raw_dca),
            'score_risk_mapping_predictions_recomputed':True,'clinical_fit_test_overlap':0,
            'shap_patients':9989,'dependence_pairs':4,'waterfall_cases':9,'shap_additivity_verified':True,
            'sources':{name:sha256(DATA/name) for name in ('clinical_risk_predictions.npz','clinical_calibration_curves.csv','clinical_dca_curves.csv','shap_dependence_observed_history.npz','supplementary_S5_waterfall_values.csv')}}
    (DATA/'revision_numerical_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
if __name__=='__main__':print(json.dumps(verify(),indent=2))
