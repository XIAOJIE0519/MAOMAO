#!/usr/bin/env python3
"""Serially finish the current bounded booster / raw-calibrated report revision."""
from __future__ import annotations
import fcntl
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.diagnostics.uniform_result_scope import XGB_CANDIDATE, current_xgboost_dir, xgboost_revision, sha256
from scripts.diagnostics.build_fullscale_external_rows import SOURCES

OUT = ROOT / 'outputs/external_validation_final_maomao_uniform'
ROWS = ROOT / 'outputs/final_experiment_results_20260923/classical_full_scale/external'
ORDER = ('ntuh', 'asac', 'uq', 'surgical_pooled', 'mimic', 'sicdb', 'mover', 'eicu')
MODELS = ('univariate', 'logistic_regression', 'xgboost', 'ann', 'maomao')
ARCHIVE = ROOT / 'outputs/archive_removed_experiments_20260923/before_three_rounds_and_calibration_revision_20260928'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def active(fragment):
    lines = subprocess.check_output(['ps', '-eo', 'pid=,args='], text=True).splitlines()
    return any('python' in line and fragment in line and int(line.split()[0]) != os.getpid() for line in lines)


def refresh():
    for name in ('build_uniform_results_report.py', 'package_uniform_final_results.py'):
        subprocess.run([sys.executable, str(ROOT / 'scripts/diagnostics' / name)], cwd=ROOT, check=True)


def run(script, args, log_name):
    logs = OUT / 'logs'
    logs.mkdir(exist_ok=True)
    with (logs / log_name).open('a') as log:
        log.write(f'\n[{stamp()}] {script} {args}\n')
        log.flush()
        env = dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
        subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/diagnostics' / script), *args],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)


def valid(site, model):
    path = OUT / site / model
    data, state = read(path / 'metrics.json'), read(path / 'status.json')
    manifest = read(ROWS / site / 'manifest.json')
    return (state.get('status') == 'completed' and data.get('test_rows') == manifest.get('test_target_rows')
            and data.get('calibration_rows') == manifest.get('calibration_target_rows')
            and bool(data) and (model != 'xgboost' or data.get('model_sha256') == xgboost_revision()))


def raw_valid(site):
    folder = OUT / site / 'maomao'
    raw, post = read(folder / 'metrics_uncalibrated.json'), folder / 'metrics.json'
    return (raw.get('status') == 'completed' and post.exists()
            and raw.get('calibrated_reference_sha256') == sha256(post))


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    with (OUT / '.current_revision.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {'status':'running', 'phase':'waiting_bounded_xgboost', 'pid':os.getpid(),
                 'started_utc':stamp(), 'order':list(ORDER), 'sites':{}, 'maomao_raw_and_calibrated':True,
                 'pooled_sources':['ntuh','asac','uq']}
        def save():
            state['updated_utc'] = stamp()
            write(OUT / 'queue_status.json', state)
            write(OUT / 'revision_status.json', state)
        save()
        try:
            while active('run_xgboost_conservative_continuation.py'):
                state['xgboost'] = read(XGB_CANDIDATE / 'supervisor_status.json')
                save()
                time.sleep(30)
            xgb = read(XGB_CANDIDATE / 'status.json')
            if not xgb or xgb.get('status') not in ('completed','stopped_no_extra_round','evaluation_failed'):
                raise RuntimeError('XGBoost stopped without a terminal guard/result record; no automatic training retry')
            state['xgboost'] = xgb
            new_xgb = current_xgboost_dir() == XGB_CANDIDATE
            if not new_xgb:
                state['limitation'] = 'Bounded continuation did not produce evaluated extra rounds; retain saved one-round model honestly.'
                # Attach hash to the already completed old-model result after preserving it.
                # These are exactly the prior frozen one-round booster results, never the failed candidate.
                for site in ORDER:
                    path = OUT / site / 'xgboost/metrics.json'
                    data = read(path)
                    if data and not data.get('model_sha256'):
                        target = ARCHIVE / path.relative_to(ROOT)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, target)
                        data['model_sha256'] = xgboost_revision()
                        data['provenance_note'] = 'Saved pre-continuation one-round model; no new round was promoted.'
                        write(path, data)
            state['phase'] = 'external_evaluation'
            save()
            refresh()
            for site in ORDER:
                while active('score_uniform_external_five.py') or active('run_uniform_external_queue.py') or active('run_current_results_revision.py --maomao-before-only'):
                    state['phase'] = 'waiting_existing_evaluator'
                    save()
                    time.sleep(30)
                manifest = read(ROWS / site / 'manifest.json')
                if manifest.get('status') != 'ready_for_full_scale_evaluation':
                    run('build_fullscale_external_rows.py', ['--sites',site], f'{site}_revision_rows.log')
                state.update(phase='external_evaluation', current_site=site)
                state['sites'][site] = {'status':'evaluating','completed_models':sum(valid(site,m) for m in MODELS),
                                        'maomao_raw_completed':raw_valid(site)}
                save()
                for model in MODELS:
                    if valid(site,model) and (model != 'maomao' or raw_valid(site)):
                        continue
                    folder = OUT / site / model
                    if model == 'xgboost' and folder.exists() and read(folder / 'metrics.json') and not valid(site,model):
                        target = ARCHIVE / "pre_label_serial_xgboost" / site / model
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if target.exists():
                            raise RuntimeError(f'Archive target exists for stale {site}/xgboost; inspect before replacing')
                        shutil.move(str(folder), target)
                    state['current_model'] = model
                    save()
                    run('score_uniform_external_five.py', ['--site',site,'--models',model],
                        f'{site}_{model}_revision_evaluation.log')
                    if not valid(site,model) or (model == 'maomao' and not raw_valid(site)):
                        raise RuntimeError(f'{site}/{model} exited without current full-row metrics')
                    state['sites'][site] = {'status':'evaluating','completed_models':sum(valid(site,m) for m in MODELS),
                                            'maomao_raw_completed':raw_valid(site)}
                    save()
                    refresh()
                state['sites'][site]['status'] = 'completed'
                save()
            state.update(status='completed' if new_xgb else 'completed_with_xgboost_limitation',
                         phase='final_reporting', finished_utc=stamp())
            state.pop('current_site',None)
            state.pop('current_model',None)
            save()
            refresh()
            command = [sys.executable, str(ROOT / 'scripts/diagnostics/verify_uniform_final_results.py')]
            if new_xgb:
                command.append('--require-complete')
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            write(OUT / 'revision_verification.json', {'returncode':result.returncode,
                  'stdout':result.stdout,'stderr':result.stderr,'finished_utc':stamp()})
            if result.returncode:
                raise RuntimeError(f'Final artifact verification failed: {result.stdout[-1500:]} {result.stderr[-500:]}')
            state['phase'] = 'finished'
            save()
            refresh()
        except Exception as exc:
            state.update(status='failed', error=repr(exc), failed_utc=stamp())
            save()
            raise


def before_after_only():
    # This queue uses one external evaluator and can coexist with the low-memory
    # label-serial trainer. The main revision supervisor waits for this queue.
    with (OUT / '.maomao_before_after.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {'status':'running','pid':os.getpid(),'started_utc':stamp(),'sites':{}}
        try:
            for site in ('mimic','sicdb','mover','eicu'):
                while active('score_uniform_external_five.py'):
                    time.sleep(30)
                if not raw_valid(site):
                    state['current_site'] = site
                    write(OUT / 'maomao_before_after_queue_status.json',state)
                    run('score_uniform_external_five.py',['--site',site,'--models','maomao'],
                        f'{site}_maomao_revision_evaluation.log')
                if not raw_valid(site):
                    raise RuntimeError(f'{site}: raw/calibrated pair missing after evaluator exit')
                state['sites'][site] = 'completed'
                write(OUT / 'maomao_before_after_queue_status.json',state)
                refresh()
            state.update(status='completed',finished_utc=stamp())
            state.pop('current_site',None)
            write(OUT / 'maomao_before_after_queue_status.json',state)
        except Exception as exc:
            state.update(status='failed',error=repr(exc),failed_utc=stamp())
            write(OUT / 'maomao_before_after_queue_status.json',state)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--maomao-before-only',action='store_true')
    arguments = parser.parse_args()
    before_after_only() if arguments.maomao_before_only else main()
