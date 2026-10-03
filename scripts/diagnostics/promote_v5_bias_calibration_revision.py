#!/usr/bin/env python3
"""Promote the complete, audited V5 event calibration restoration.

Never launches inference, changes fits, or copies private grouping inputs.
Publication plots/report/ZIP still require their separate final delivery QA.
"""
import fcntl,json,shutil,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.diagnostics import run_v5_bias_calibration_revision as worker
from scripts.diagnostics import verify_manuscript_external_ablation_rows as rowaudit
from scripts.diagnostics.verify_v5_calibration_selection import verify as verify_selection
from scripts.diagnostics.uniform_result_scope import sha256
from maomao.evaluation.event_bias_calibration import PROTOCOL,validate_fit
from maomao.evaluation.rank_statistics import RANK_VERSION

REV=worker.REV;read=worker.read;write=worker.write


def copy_json_tree(source,target):
    for path in source.rglob('*.json'):
        if path.name.endswith('.partial.json'):continue
        dest=target/path.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest)


def main():
    with (REV/'queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        queue=read(REV/'queue_status.json')
        if queue.get('promoted'):
            print('Already promoted; no inference or fitting repeated.');return
        assert queue['status']=='evaluations_completed_awaiting_verified_promotion'
        assert len(queue['completed'])==len(set(queue['completed']))==115
        gradient=read(REV/'analytic_gradient_verification.json')
        assert gradient['complete'] and not gradient['sealed_test_read']
        selection_proof=verify_selection(REV,require_complete=True)
        pairs=[]
        for site in worker.SITES:
            manifest=read(worker.scoring.BASE/site/'manifest.json')
            for name in worker.MODELS:
                folder=REV/'external'/site/name
                fit=read(folder/'calibration_fit.json');after=read(folder/'metrics.json')
                assert read(folder/'status.json')['status']=='completed'
                validate_fit(fit,manifest['calibration_target_rows'],210)
                assert after['status']=='completed' and after['rank_metric_definition']==RANK_VERSION
                assert after['model_sha256']==fit['model_sha256']==sha256(worker.scalar.model_source(name))
                assert after['test_rows']==manifest['test_target_rows'] and after['patient_overlap']==0
                assert after['calibration_protocol']==PROTOCOL and after['bias']==fit['bias']
                assert after['calibration_fit_sha256']==sha256(folder/'calibration_fit.json')
                groups=worker.row_groups(site)
                assert fit['selection_split']['row_group_sha256']==__import__('hashlib').sha256(np.asarray(groups,dtype=np.int64).tobytes()).hexdigest()
                assert after['retrieval_rows']==manifest['test_target_rows']
                assert 0<=after['brier']<=2 and 0<=after['brier_95ci'][0]<=after['brier_95ci'][1]<=2
                if after['test_rows']>100000 and after['brier_95ci'][1]==1.:
                    raise RuntimeError(f'{site}/{name}: verify Brier CI has not been clipped at the binary-score bound before promotion')
                if name=='maomao':
                    before=read(folder/'metrics_uncalibrated.json')
                    original=read(worker.BACKUP/'external'/site/name/'metrics_uncalibrated.json')
                    for key in ('micro_auroc','micro_auprc','brier','ece','hit_at_1','mrr'):
                        assert before[key]==original[key]
                    assert before['calibrated_reference_sha256']==sha256(folder/'metrics.json')
                pairs.append(dict(site=site,model=name,rows=after['test_rows'],calibration_rows=fit['calibration_rows'],
                                  selected_family=fit['family'],model_sha256=fit['model_sha256'],
                                  fit_sha256=sha256(folder/'calibration_fit.json'),metrics_sha256=sha256(folder/'metrics.json')))
        # Reuse authoritative source-coordinate audits, then run the complete
        # independent 75-job/85-pair verifier against the STAGED results.
        source=worker.variants.WORK/'source_data/external_row_audits'
        shutil.copytree(source,REV/'source_data/external_row_audits',dirs_exist_ok=True)
        jobs=[dict(site=site,model=name) for site in worker.variants.SITES for name in worker.variants.MODELS]
        write(REV/'external_ablations/queue_status.json',dict(status='completed',pid=queue['pid'],jobs_total=75,
              scope=worker.variants.VERSION,sites=list(worker.variants.SITES),completed=jobs,calibration_protocol=PROTOCOL))
        rowaudit.OUT=REV
        variants=rowaudit.verify(require_complete=True)
        proof=dict(complete=True,protocol=PROTOCOL,main_jobs=40,variant_jobs=75,variant_pairs=85,total_calibration_fits=125,before_after_pairs=93,
            main_pairs=pairs,variant_row_verification_sha256=sha256(REV/'external_ablation_row_verification.json'),
            selection_partition_verification_sha256=sha256(REV/'selection_partition_verification.json'),
            source_patient_selection_fits_verified=selection_proof['calibration_fits_checked'],
            fit_selection_no_test=True,all_full_calibration_and_test_rows=True,frozen_models_verified=True,
            changed_event_ranking_allowed_only_with_class_bias=True,created_utc=worker.scoring.stamp())
        write(REV/'promotion_verification.json',proof)
        # Only immutable completed result evidence is promoted. Large logits,
        # private grouping codes and model checkpoints remain outside delivery.
        copy_json_tree(REV/'external',worker.scoring.RESULTS)
        shutil.copytree(worker.PLOTS/'external',ROOT/'outputs/maomao_plot_sources/external',dirs_exist_ok=True)
        copy_json_tree(REV/'external_ablations',worker.variants.OUT)
        shutil.copy2(REV/'external_ablation_row_verification.json',worker.variants.WORK/'external_ablation_row_verification.json')
        for site in worker.SITES:
            write(worker.scoring.RESULTS/site/'summary.json',dict(site=site,status='completed',protocol=PROTOCOL,
                models={name:{key:read(REV/'external'/site/name/'metrics.json')[key] for key in ('micro_auprc','micro_auroc','test_rows')} for name in worker.MODELS}))
        # Switch active protocol LAST so consumers cannot treat old transforms
        # as new calibration while replacement is still in progress.
        write(worker.scoring.RESULTS/'active_calibration_protocol.json',dict(protocol=PROTOCOL,
              revision_directory=str(REV.relative_to(ROOT)),promotion_verification_sha256=sha256(REV/'promotion_verification.json'),
              promoted_utc=worker.scoring.stamp()))
        write(worker.scoring.RESULTS/'queue_status.json',dict(status='completed',phase='v5_event_calibration_restored',
              protocol=PROTOCOL,order=list(worker.SITES),pid=queue['pid'],
              sites={site:dict(status='completed',completed_models=5,maomao_raw_completed=True) for site in worker.SITES},
              maomao_raw_and_calibrated=True,pooled_sources=['ntuh','asac','uq'],
              updated_utc=worker.scoring.stamp(),calibration_revision_directory=str(REV.relative_to(ROOT))))
        queue.update(status='completed',promoted=True,finished_utc=worker.scoring.stamp());write(REV/'queue_status.json',queue)
        print('Promoted 40 full-row main models and 85 paired external variant tasks; report/figure delivery remains to be verified.')


if __name__=='__main__':main()
