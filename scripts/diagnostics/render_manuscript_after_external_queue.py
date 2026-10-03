#!/usr/bin/env python3
"""Render missing Figure 3 after the existing evaluator exits successfully.

This watcher never launches evaluation or records human visual inspection.
"""
import fcntl,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/maomao_manuscript_figures_20260929'
def write(x):
    p=OUT/'figure_3_render_queue.json';t=p.with_suffix('.json.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False
def main():
    with (OUT/'figure_3_render.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status={'status':'waiting_for_existing_external_evaluator','pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat()};write(status)
        while True:
            q=json.loads((OUT/'external_ablations/queue_status.json').read_text())
            if q['status']=='completed' and not alive(q['pid']):break
            if q['status']=='failed' or not alive(q['pid']):
                status.update(status='needs_evaluator_recovery',evaluation_status=q);write(status);return
            time.sleep(30)
        try:
            status.update(status='rendering_final_figure_3');write(status)
            with (OUT/'render_3.log').open('w') as log:
                subprocess.run([sys.executable,str(ROOT/'scripts/diagnostics/build_manuscript_figures.py'),'--figures','3'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            status.update(status='checking_export_without_human_visual_claim');write(status)
            with (OUT/'verification_3.log').open('w') as log:
                subprocess.run([sys.executable,str(ROOT/'scripts/diagnostics/verify_manuscript_figures.py'),'--figures','3'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            status.update(status='awaiting_figure_3_visual_inspection_and_final_package',finished_utc=datetime.now(timezone.utc).isoformat());write(status)
        except BaseException as e:
            status.update(status='failed',error=f'{type(e).__name__}: {e}',finished_utc=datetime.now(timezone.utc).isoformat());write(status);raise
if __name__=='__main__':main()
