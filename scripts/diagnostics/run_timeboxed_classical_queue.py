#!/usr/bin/env python3
"""Run classical full-scale candidates serially with per-model 10-minute limits."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/final_experiment_results_20260923/classical_ml_fullscale'
MODELS=['decision_tree','random_forest','hist_gradient_boosting','gaussian_nb']
def available_gib():
 for line in Path('/proc/meminfo').read_text().splitlines():
  if line.startswith('MemAvailable:'):return int(line.split()[1])/1024/1024
 return 99
def write(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 previous=json.loads((OUT/'queue_status.json').read_text()) if (OUT/'queue_status.json').exists() else {}
 queue={'status':'running','per_model_timeout_seconds':600,'memory_guard_min_available_gib':6,
        'models':list(previous.get('models',[])),'current_model':None}
 prior={m['model']:m for m in queue['models']};write(OUT/'queue_status.json',queue)
 for name in MODELS:
  folder=OUT/name;folder.mkdir(exist_ok=True)
  if (folder/'metrics.json').exists():
   if name not in prior:queue['models'].append({'model':name,'status':'completed','reused':True})
   write(OUT/'queue_status.json',queue);continue
  old_status=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
  # A timed-out, memory-guarded, or failed candidate is terminal; do not
  # silently retry it. Resume one interrupted candidate at most once.
  if old_status.get('status') in ('timeout_no_result','memory_guard_stop','failed'):
   if name not in prior:queue['models'].append(old_status)
   write(OUT/'queue_status.json',queue);continue
  if old_status.get('status')=='interrupted_no_result' and old_status.get('resume_attempted'):
   if name not in prior:queue['models'].append(old_status)
   write(OUT/'queue_status.json',queue);continue
  queue['current_model']=name;write(OUT/'queue_status.json',queue)
  if old_status.get('status')=='interrupted_no_result':
   history=json.loads((folder/'attempt_history.json').read_text()) if (folder/'attempt_history.json').exists() else []
   history.append({'attempt':'resume_after_interruption','previous_status':old_status})
   write(folder/'attempt_history.json',history);old_status['resume_attempted']=True;write(folder/'status.json',old_status)
  log=(folder/'run.log').open('w')
  cmd=[sys.executable,str(ROOT/'scripts/diagnostics/run_fullscale_ml_candidate.py'),'--model',name]
  proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  start=time.monotonic();reason='completed';
  while proc.poll() is None:
   elapsed=time.monotonic()-start
   if elapsed>=600:reason='timeout_no_result';os.killpg(proc.pid,signal.SIGTERM);break
   if available_gib()<6:reason='memory_guard_stop';os.killpg(proc.pid,signal.SIGTERM);break
   time.sleep(5)
  try:rc=proc.wait(timeout=5)
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGKILL);rc=proc.wait()
  log.close();elapsed=round(time.monotonic()-start,1)
  if (folder/'metrics.json').exists():reason='completed'
  elif rc and reason=='completed':reason='failed'
  state={'model':name,'status':reason,'exit_code':rc,'elapsed_seconds':elapsed,'timeout_seconds':600,'memory_available_gib_at_end':round(available_gib(),2)}
  write(folder/'status.json',state)
  queue['models']=[m for m in queue['models'] if m.get('model')!=name]
  queue['models'].append(state);queue['current_model']=None;write(OUT/'queue_status.json',queue)
  print(json.dumps(state),flush=True)
 queue['status']='completed' if all(m['status']=='completed' for m in queue['models']) else 'partial';write(OUT/'queue_status.json',queue)
 print(json.dumps(queue,indent=2),flush=True)
if __name__=='__main__':main()
