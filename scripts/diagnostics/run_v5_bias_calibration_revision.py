#!/usr/bin/env python3
"""Locked, resumable restoration of V5 external calibration and all variants.

Stages complete results separately. Never fits on test data, retrains a model,
or promotes a mixed scalar/bias final delivery. Prediction caches are removed
only after their fit, full test metrics and plotting evidence are complete.
"""
import fcntl,gc,json,os,shutil,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from maomao.data.event_sequence import EventSequenceDataset
from maomao.evaluation.event_bias_calibration import PROTOCOL,fit_event_calibration,apply_calibration
from maomao.evaluation.event_metrics import event_metric_report
from maomao.evaluation.plot_sources import export_plot_sources
from scripts.diagnostics import score_uniform_external_five as scoring
from scripts.diagnostics import run_softmax_calibration_revision as scalar
from scripts.diagnostics import run_manuscript_external_ablations as variants
from scripts.diagnostics.complete_external_retrieval import extract
from scripts.diagnostics.maomao_display_family_groups import display_mapping
from scripts.diagnostics.uniform_result_scope import sha256

REV=ROOT/'outputs/calibration_v5_bias_revision_20260930'
PLOTS=REV/'plot_sources'
BACKUP=ROOT/'outputs/calibration_scalar_archive_20260930'
MODELS=('maomao','univariate','logistic_regression','xgboost','ann')
SITES=('ntuh','asac','uq','surgical_pooled','mimic','sicdb','mover','eicu')
read=lambda p:json.loads(p.read_text())
write=scoring.save_json


def row_groups(site):
    """Private anonymous group codes, aligned with original calibration rows."""
    p=REV/'private_group_inputs'/f'{site}.npy'
    if p.exists():return np.load(p,mmap_mode='r')
    source=ROOT/scoring.SOURCES[site]
    manifest=read(scoring.BASE/site/'manifest.json')
    admissions=pd.read_csv(source/'admissions.csv',dtype={'subject_id':'string'})
    column=manifest['split'].get('group_column','subject_id' if 'subject_id' in admissions else 'admission_id')
    codes,_=pd.factorize(admissions[column].fillna('missing:').astype(str),sort=True)
    dataset=EventSequenceDataset(source,256,128,dynamic_windows=False)
    windows=np.load(scoring.BASE/site/'calibration_window_indices.npy',mmap_mode='r')
    groups=codes[dataset.window_admission[windows]].astype(np.int64)
    if len(np.unique(groups))>manifest['split']['patients_validation']:raise RuntimeError('Calibration group count exceeds split')
    p.parent.mkdir(parents=True,exist_ok=True);np.save(p,groups)
    del dataset,admissions;gc.collect()
    return groups


def main_model(site,name,meta,hashes):
    folder=REV/'external'/site/name;folder.mkdir(parents=True,exist_ok=True)
    statusp=folder/'status.json'
    if statusp.exists() and read(statusp).get('status')=='completed':return
    manifest=read(scoring.BASE/site/'manifest.json')
    old=read(BACKUP/'external'/site/name/'metrics.json')
    if old['model_sha256']!=hashes[name]:raise RuntimeError('Frozen main checkpoint differs')
    write(statusp,dict(status='predicting_calibration',pid=os.getpid(),updated_utc=scoring.stamp()))
    scalar.PROTOCOL=PROTOCOL
    fitp=folder/'calibration_fit.json'
    cal=None
    if fitp.exists():
        fit=read(fitp)
        if fit['protocol']!=PROTOCOL or fit['model_sha256']!=hashes[name]:raise RuntimeError('Stale calibration fit')
    else:
        cal=scalar.scores(site,name,'calibration',folder,manifest,hashes)
        write(statusp,dict(status='fitting_v5_full90',pid=os.getpid(),updated_utc=scoring.stamp()))
        fit=fit_event_calibration(cal,scoring.BASE/site/'calibration_y.npy',row_groups(site),scoring.DEVICE)
        fit.update(site=site,model=name,model_sha256=hashes[name],created_utc=scoring.stamp())
        write(fitp,fit)
    write(statusp,dict(status='predicting_sealed_test',pid=os.getpid(),updated_utc=scoring.stamp()))
    test=scalar.scores(site,name,'test',folder,manifest,hashes)
    yp=scoring.BASE/site/'test_y.npy'
    logits=torch.from_numpy(np.array(np.load(test,mmap_mode='r'),dtype=np.float32))
    target=torch.from_numpy(np.array(np.load(yp,mmap_mode='r'),dtype=np.uint8))
    adapted=apply_calibration(logits,fit)
    write(statusp,dict(status='full_test_metrics_and_200_ci',pid=os.getpid(),updated_utc=scoring.stamp()))
    result=scalar.event_metric_report_with_subsample_ci(adapted,target,meta['outcome_vocabulary'])
    keys=('site','model','split','source_dataset','patients_calibration','patients_test','patient_overlap',
          'calibration_rows','test_rows','train_rows_internal','calibration_scope','test_scope')
    result.update({k:old[k] for k in keys})
    result.update(status='completed',temperature=fit['temperature'],bias=fit['bias'],calibration_family=fit['family'],
        model_sha256=hashes[name],calibration_protocol=PROTOCOL,calibration_state='after',
        metric_input='softmax((frozen model logits + 90%-fitted class bias) / 90%-fitted temperature)',
        calibration_fit_sha256=sha256(fitp),test_labels_used_for_fit_or_selection=False,created_utc=scoring.stamp())
    states=[('after',result,fit)]
    if name=='maomao':
        raw=read(BACKUP/'external'/site/name/'metrics_uncalibrated.json')
        check=event_metric_report(logits,target,meta['outcome_vocabulary'])
        for key in ('brier','ece','hit_at_1','mrr','micro_auroc','micro_auprc'):
            if abs(raw[key]-check[key])>5e-6:raise RuntimeError(f'Frozen raw result changed {site}/{key}')
        raw.update(calibration_protocol=PROTOCOL,fitted_temperature=fit['temperature'],
                   calibration_family='raw',bias=[0.]*210)
        states.append(('before',raw,dict(temperature=1.,bias=[0.]*210)))
    ids=torch.tensor(meta['outcome_to_family'],dtype=torch.long)
    mapping,_=display_mapping(meta['outcome_family_vocabulary'])
    broad=torch.tensor(mapping,dtype=torch.long)[ids]
    z=np.load(test,mmap_mode='r');y=np.load(yp,mmap_mode='r')
    for state,metrics,params in states:
        dest=PLOTS/f'external/{site}/{name}/{state}'
        plot=export_plot_sources(test,yp,dest,meta['outcome_vocabulary'],params['temperature'],
            dict(site=site,model=name,calibration_state=state,protocol=PROTOCOL,model_sha256=hashes[name],
                 calibration_family=params.get('family','raw'),calibration_fit_sha256=sha256(fitp)),bias=params['bias'])
        if abs(plot['brier_normalized_target']-metrics['brier'])>5e-6:raise RuntimeError('Plot/point Brier differs')
        metrics.update(extract(z,y,params['temperature'],ids,broad,dest,metrics,bias=params['bias']))
    write(folder/'metrics.json',result)
    if name=='maomao':
        raw['calibrated_reference_sha256']=sha256(folder/'metrics.json')
        write(folder/'metrics_uncalibrated.json',raw)
        write(folder/'uncalibrated_status.json',dict(status='completed',test_rows=raw['test_rows']))
    for state,metrics,_ in states:
        file=folder/('metrics.json' if state=='after' else 'metrics_uncalibrated.json')
        p=PLOTS/f'external/{site}/{name}/{state}/provenance.json'
        proof=read(p);proof.update(source_metric_sha256=sha256(file),retrieval_protocol=metrics['retrieval_protocol'],
                                  retrieval_vector_sha256=metrics['retrieval_vector_sha256']);write(p,proof)
    del logits,target,adapted,z,y;gc.collect();torch.cuda.empty_cache()
    for p in (folder/'calibration_logits.npy',test):p.unlink(missing_ok=True)
    write(statusp,dict(status='completed',site=site,model=name,protocol=PROTOCOL,
                      calibration_rows=manifest['calibration_target_rows'],test_rows=manifest['test_target_rows'],finished_utc=scoring.stamp()))
    print(f'COMPLETED main {site}/{name}: {fit["family"]}, T={fit["temperature"]:.5f}, Hit1={result["hit_at_1"]:.5f}',flush=True)


def variant_model(site,name,meta):
    folder=REV/'external_ablations'/site/name;folder.mkdir(parents=True,exist_ok=True)
    statusp=folder/'status.json'
    if statusp.exists() and read(statusp).get('status')=='completed':return
    state=dict(status='starting',site=site,model=name,pid=os.getpid(),started_utc=scoring.stamp())
    write(statusp,state)
    cal,proof=variants.score_logits(site,name,'calibration',folder,64,state)
    # A deterministic task projection is defined from calibration targets only.
    tasks=[(name,proof['indices'])]
    if name=='reference_projected':tasks=[(f'reference_vocab_{k}',read(ROOT/f'outputs/scale_ablations_richctx_20260928/specifications/vocab_{k}.json')['indices']) for k in (50,100,150)]
    fits={};calgroups=row_groups(site)
    for label,indices in tasks:
        dest=folder/label;dest.mkdir(parents=True,exist_ok=True)
        fitp=dest/'calibration_fit.json'
        if fitp.exists():fits[label]=read(fitp);continue
        cz,cy,_=variants.project_arrays(cal,scoring.BASE/site/'calibration_y.npy',dest,'calibration',indices,proof['indices'])
        original=np.load(scoring.BASE/site/'calibration_y.npy',mmap_mode='r')
        eligible=np.empty(len(original),bool)
        for lo in range(0,len(original),65536):eligible[lo:lo+65536]=original[lo:lo+65536][:,indices].any(1)
        fit=fit_event_calibration(cz,cy,np.asarray(calgroups)[eligible],scoring.DEVICE)
        fit.update(site=site,model=label,model_sha256=sha256(variants.checkpoint(name)),created_utc=scoring.stamp())
        write(fitp,fit);fits[label]=fit
        cz.unlink();cy.unlink();del original,eligible;gc.collect()
    test,testproof=variants.score_logits(site,name,'test',folder,64,state)
    for label,indices in tasks:
        dest=folder/label;fit=fits[label]
        tz,ty,n=variants.project_arrays(test,scoring.BASE/site/'test_y.npy',dest,'test',indices,proof['indices'])
        z=torch.from_numpy(np.array(np.load(tz,mmap_mode='r'),copy=True))
        y=torch.from_numpy(np.array(np.load(ty,mmap_mode='r'),copy=True))
        before=read(BACKUP/'external_ablations'/site/name/label/'metrics_before.json')
        names=[meta['outcome_vocabulary'][i] for i in indices]
        check=event_metric_report(z,y,names)
        for key in ('brier','ece','hit_at_1','mrr'):
            if abs(check[key]-before[key])>5e-6:raise RuntimeError(f'Frozen variant raw changed {site}/{label}/{key}')
        after=scalar.event_metric_report_with_subsample_ci(apply_calibration(z,fit),y,names,max_ci_rows=30000)
        # Preserve all row/checkpoint audit fields, replace every model statistic.
        template={k:v for k,v in before.items() if k not in check and not k.endswith('_95ci')}
        after.update(template,status='completed',calibration_state='after',temperature=fit['temperature'],bias=fit['bias'],
                     calibration_family=fit['family'],calibration_protocol=PROTOCOL,
                     calibration_fit_sha256=sha256(dest/'calibration_fit.json'),finished_utc=scoring.stamp())
        before.update(calibration_protocol=PROTOCOL,calibration_family='raw',bias=[0.]*len(indices))
        write(dest/'metrics_before.json',before);write(dest/'metrics_after.json',after)
        del z,y;tz.unlink();ty.unlink();gc.collect();torch.cuda.empty_cache()
        print(f'COMPLETED variant {site}/{label}: {fit["family"]}, Hit1 {before["hit_at_1"]:.5f}->{after["hit_at_1"]:.5f}',flush=True)
    state.update(status='completed',checkpoint_sha256=sha256(variants.checkpoint(name)),
                 report_labels=[label for label,_ in tasks],calibration_protocol=PROTOCOL,finished_utc=scoring.stamp())
    write(statusp,state);cal.unlink();test.unlink()


def main():
    torch.set_num_threads(8);REV.mkdir(parents=True,exist_ok=True)
    with (REV/'queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (BACKUP/'snapshot_complete.json').exists():
            BACKUP.mkdir(parents=True,exist_ok=True)
            shutil.copytree(scoring.RESULTS,BACKUP/'external',dirs_exist_ok=True)
            shutil.copytree(variants.OUT,BACKUP/'external_ablations',dirs_exist_ok=True)
            shutil.copy2(ROOT/'docs/MAOMAO_V5_FINAL_RESULTS.md',BACKUP/'MAOMAO_V5_FINAL_RESULTS.md')
            write(BACKUP/'snapshot_complete.json',dict(completed=True,created_utc=scoring.stamp()))
        meta=read(scoring.TRAINING/'event_sequence_meta.json')
        hashes={name:sha256(scalar.model_source(name)) for name in MODELS}
        specs=dict(protocol=PROTOCOL,models=list(MODELS),sites=list(SITES),main_jobs=40,external_variant_jobs=75,
                   family_selection='patient-disjoint 80:20 inside full external 90%; V5 positive-set NLL',
                   final_refit='all full eligible 90% rows',test='same sealed full 10% rows; evaluation only',
                   candidates=['raw','temperature_only','temperature_and_bias'],model_hashes=hashes,
                   model_training_repeated=False,optimizer=dict(method='bounded analytic-gradient full-row L-BFGS-B',maxiter=80,maxfun=120,ftol=1e-9,gtol=3e-6,maxls=15),created_utc=scoring.stamp())
        specpath=REV/'prespecified_protocol.json'
        if not specpath.exists():write(specpath,specs)
        elif read(specpath)['model_hashes']!=hashes:raise RuntimeError('Model changed during revision')
        completed=[];state=dict(status='running',pid=os.getpid(),protocol=PROTOCOL,jobs_total=115,started_utc=scoring.stamp())
        try:
            for name in MODELS:
                for site in SITES:
                    state.update(active=f'main/{site}/{name}',completed=completed,updated_utc=scoring.stamp());write(REV/'queue_status.json',state)
                    main_model(site,name,meta,hashes);completed.append(f'main/{site}/{name}')
            for site in variants.SITES:
                for name in variants.MODELS:
                    state.update(active=f'variant/{site}/{name}',completed=completed,updated_utc=scoring.stamp());write(REV/'queue_status.json',state)
                    variant_model(site,name,meta);completed.append(f'variant/{site}/{name}')
            if any(sha256(scalar.model_source(name))!=value for name,value in hashes.items()):raise RuntimeError('Frozen main model changed')
            state.update(status='evaluations_completed_awaiting_verified_promotion',completed=completed,finished_utc=scoring.stamp());write(REV/'queue_status.json',state)
        except BaseException as error:
            state.update(status='failed',completed=completed,error=repr(error),finished_utc=scoring.stamp());write(REV/'queue_status.json',state);raise
if __name__=='__main__':main()
