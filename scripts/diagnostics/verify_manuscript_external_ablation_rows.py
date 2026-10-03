#!/usr/bin/env python3
"""Independently verify complete external ablation pairs against frozen rows.

Reads artifacts only. It neither scores models nor launches/resumes evaluators.
"""
import argparse,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.diagnostics.run_manuscript_external_ablations import SITES,MODELS,VERSION,checkpoint
from scripts.diagnostics.build_fullscale_external_rows import SOURCES
from scripts.diagnostics.uniform_result_scope import sha256
from maomao.evaluation.softmax_calibration import LEGACY_PROTOCOL
from maomao.evaluation.event_bias_calibration import validate_fit, PROTOCOL as BIAS_PROTOCOL
OUT=ROOT/'outputs/maomao_manuscript_figures_20260929'
ROWS=ROOT/'outputs/final_experiment_results_20260923/classical_full_scale/external'
METRICS=('micro_auprc','macro_auprc','micro_auroc','macro_auroc','mrr','brier','ece','hit_at_1','recall_at_5','recall_at_10')
def read(p):return json.loads(p.read_text())
def signatures(paths):return {str(p.relative_to(ROOT)):[p.stat().st_size,p.stat().st_mtime_ns] for p in paths}
def site_rows(site,recheck=False):
    folder=ROWS/site;manifest=read(folder/'manifest.json');source=ROOT/SOURCES[site]
    splitfile=Path(manifest['split']['split_file']);sp=np.load(splitfile)
    specs={k:read(ROOT/f'outputs/scale_ablations_richctx_20260928/specifications/vocab_{k}.json')['indices'] for k in (50,100,150)}
    paths=[folder/'manifest.json',splitfile,source/'admissions.csv',*[folder/f'{s}_{x}.npy' for s in ('calibration','test') for x in ('y','positions','window_indices')],*[ROOT/f'outputs/scale_ablations_richctx_20260928/specifications/vocab_{k}.json' for k in specs]]
    signature=signatures(paths);cache=OUT/f'source_data/external_row_audits/{site}.json'
    if cache.exists() and not recheck and read(cache).get('source_file_signatures')==signature:return read(cache)
    admission=pd.read_csv(source/'admissions.csv',dtype={'subject_id':'string'})
    group=manifest['split'].get('group_column','subject_id' if 'subject_id' in admission else 'admission_id')
    labels=admission[group].fillna('missing:').astype(str).to_numpy()
    calids=sp['validation_admissions'];testids=sp['test_admissions']
    calgroups=np.unique(labels[calids]);testgroups=np.unique(labels[testids]);allgroups=np.unique(labels)
    assert not np.intersect1d(calgroups,testgroups).size
    assert np.array_equal(np.sort(np.r_[calids,testids]),np.arange(len(labels)))
    shuffled=allgroups.copy();np.random.default_rng(42).shuffle(shuffled)
    assert np.array_equal(np.sort(shuffled[:max(1,round(.1*len(shuffled)))]),testgroups)
    assert len(calgroups)==manifest['split']['patients_validation'] and len(testgroups)==manifest['split']['patients_test']
    result={'site':site,'patient_or_record_proxy_split_reproduced':True,'patient_overlap':0,
            'record_proxy_ids':site=='surgical_pooled','source_file_signatures':signature,'splits':{}}
    for split in ('calibration','test'):
        yp=folder/f'{split}_y.npy';y=np.load(yp,mmap_mode='r');w=np.load(folder/f'{split}_window_indices.npy',mmap_mode='r');p=np.load(folder/f'{split}_positions.npy',mmap_mode='r')
        expected=int(manifest[f'{split}_target_rows']);assert y.shape==(expected,210) and len(w)==len(p)==expected
        expected_windows=sp['validation_windows' if split=='calibration' else 'test_windows']
        assert np.isin(np.unique(w),expected_windows).all()
        hashes={k:hashlib.sha256() for k in (210,*specs)};counts={k:0 for k in hashes};last=-1
        for lo in range(0,len(y),65536):
            chunk=np.asarray(y[lo:lo+65536]);positions=np.asarray(p[lo:lo+len(chunk)],dtype=np.int64);windows=np.asarray(w[lo:lo+len(chunk)],dtype=np.int64)
            assert (positions>=0).all() and (positions<256).all() and chunk.min()>=0 and chunk.max()<=1 and chunk.any(1).all()
            keys=windows*256+positions;assert keys[0]>last and (np.diff(keys)>0).all();last=int(keys[-1])
            for k in hashes:
                mask=chunk.any(1) if k==210 else chunk[:,specs[k]].any(1)
                ids=np.flatnonzero(mask).astype(np.int64)+lo;hashes[k].update(ids.tobytes());counts[k]+=len(ids)
        result['splits'][split]={'original_rows':len(y),'truth_sha256':sha256(yp),
            'window_coordinates_sha256':sha256(folder/f'{split}_window_indices.npy'),
            'positions_sha256':sha256(folder/f'{split}_positions.npy'),
            'tasks':{str(k):{'eligible_rows':counts[k],'excluded_empty_projected_target':len(y)-counts[k],
                       'eligible_position_sha256':hashes[k].hexdigest(),'indices':list(range(210)) if k==210 else specs[k]} for k in hashes}}
        print(f'{site}/{split}: {len(y):,} original rows; complete coordinate and projected-task hashes verified',flush=True)
    result['source_sha256']={str(p.relative_to(ROOT)):sha256(p) for p in paths if p.suffix!='.npy'}
    cache.parent.mkdir(parents=True,exist_ok=True);cache.write_text(json.dumps(result,indent=2)+'\n');return result
def verify(require_complete=False,recheck=False):
    pairs=[];done=[];hashes={}
    # Source integrity is required for all five planned cohorts, including
    # cohorts whose evaluator has not reached the first completed model yet.
    rowproofs={site:site_rows(site,recheck) for site in SITES}
    outcomes=read(ROOT/'data/perioperative_event_sequences_v5_richctx_static7/event_sequence_meta.json')['outcome_vocabulary']
    for site in SITES:
        for model in MODELS:
            folder=OUT/'external_ablations'/site/model;state=folder/'status.json'
            if not state.exists() or read(state).get('status')!='completed':continue
            status=read(state);done.append({'site':site,'model':model})
            if site not in rowproofs:rowproofs[site]=site_rows(site,recheck)
            cp=checkpoint(model)
            if cp not in hashes:hashes[cp]=sha256(cp)
            modelhash=hashes[cp];assert status['checkpoint_sha256']==modelhash
            for split in ('calibration','test'):
                pred=read(folder/f'{split}_prediction.json');signature=pred['signature']
                assert all(pred[k] for k in ('complete','all_original_rows_scored','target_projection_verified','prediction_only_bit_identical_verified'))
                assert signature['checkpoint_sha256']==modelhash and signature['row_manifest_sha256']==sha256(ROWS/site/'manifest.json')
                assert signature['rows']==rowproofs[site]['splits'][split]['original_rows'] and signature['protocol']==VERSION
            expected_labels=[f'reference_vocab_{k}' for k in (50,100,150)] if model=='reference_projected' else [model]
            assert status['report_labels']==expected_labels
            for label in expected_labels:
                target=folder/label;before=read(target/'metrics_before.json');after=read(target/'metrics_after.json');fit=read(target/'calibration_fit.json')
                count=len(before['output_indices']);assert count in (50,100,150,210)
                for split in ('calibration','test'):
                    saved=read(target/f'{split}_eligible_rows.json');actual=rowproofs[site]['splits'][split]['tasks'][str(count)]
                    assert all(saved[k]==v for k,v in actual.items()),f'Eligible rows mismatch: {site}/{label}/{split}'
                    assert saved['original_rows']==rowproofs[site]['splits'][split]['original_rows']
                    assert before[f'{split}_rows']==after[f'{split}_rows']==saved['eligible_rows']
                assert fit['calibration_rows']==before['calibration_rows'] and fit['test_labels_used_for_fit_or_selection'] is False
                assert before['temperature']==1 and after['temperature']==fit['temperature']
                assert .15<=after['temperature']<=6
                if fit['protocol']==BIAS_PROTOCOL:
                    validate_fit(fit,before['calibration_rows'],count)
                    assert after['bias']==fit['bias'] and after['calibration_family']==fit['family']
                    assert after['calibration_protocol']==BIAS_PROTOCOL
                    assert before['calibration_protocol']==BIAS_PROTOCOL
                    assert after['calibration_fit_sha256']==sha256(target/'calibration_fit.json')
                else:
                    assert fit['protocol']==LEGACY_PROTOCOL
                    if fit['fitted_temperature_accepted']:
                        assert fit['raw_calibration']['brier']-fit['fitted_calibration']['brier']>1e-8
                    else:assert after['temperature']==1
                for r in (before,after):
                    assert r['status']=='completed' and r['protocol']==VERSION and r['model_sha256']==modelhash
                    assert r['row_manifest_sha256']==sha256(ROWS/site/'manifest.json') and r['output_indices']==actual['indices']
                    assert r['patient_overlap']==0 and r['all_original_rows_scored'] and r['all_eligible_test_rows_evaluated'] and r['before_after_on_identical_rows']
                    assert r['test_labels_used_for_fit_or_selection'] is False
                    assert set(r['per_event'])=={outcomes[i] for i in r['output_indices']}
                    for metric in METRICS:assert np.isfinite([r[metric],*r[metric+'_95ci']]).all()
                    assert 0<=r['brier']<=2 and 0<=r['brier_95ci'][0]<=r['brier_95ci'][1]<=2
                    if fit['protocol']==BIAS_PROTOCOL and r['calibration_state']=='after' and r['test_rows']>30000:
                        assert r['brier_95ci'][1]!=1.,'Large-cohort Brier CI requires audit of inappropriate binary bound clipping'
                if after['temperature']==1 and not np.any(after.get('bias',[0.])):
                    for metric in METRICS:assert before[metric]==after[metric] and before[metric+'_95ci']==after[metric+'_95ci']
                pairs.append({'site':site,'model':label,'calibration_rows':before['calibration_rows'],'test_rows':before['test_rows'],
                    'output_classes':count,'model_sha256':modelhash,'before_sha256':sha256(target/'metrics_before.json'),
                    'after_sha256':sha256(target/'metrics_after.json'),'same_exact_eligible_row_hashes_verified':True})
    q=read(OUT/'external_ablations/queue_status.json');complete=len(done)==75 and len(pairs)==85 and q['status']=='completed'
    result={'complete':complete,'completed_jobs_audited':len(done),'calibration_pairs_audited':len(pairs),'jobs_total':75,'pairs_total':85,
            'completed_jobs':done,'pairs':pairs,'source_row_audit_files':{site:sha256(OUT/f'source_data/external_row_audits/{site}.json') for site in rowproofs},
            'patient_split_and_test_sealing_verified':True,'all_complete_pairs_pass':True,'updated_utc':datetime.now(timezone.utc).isoformat()}
    path=OUT/'external_ablation_row_verification.json'
    if path.exists():
        prior=read(path)
        if {k:v for k,v in prior.items() if k!='updated_utc'}=={k:v for k,v in result.items() if k!='updated_utc'}:
            result=prior
    temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(path)
    if require_complete and not complete:raise RuntimeError('Full external ablation queue is not yet complete')
    print(f'Completed-row audit: {len(done)}/75 jobs, {len(pairs)}/85 exact before/after pairs; overall complete={complete}',flush=True)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--require-complete',action='store_true');p.add_argument('--recheck-sources',action='store_true');a=p.parse_args();verify(a.require_complete,a.recheck_sources)
