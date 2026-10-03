#!/usr/bin/env python3
"""Four disjoint compute lanes for one frozen full-patient SHAP protocol."""
import os,sys,time,json,subprocess,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'outputs/maomao_family_shap_20260929'
def save(payload):
 p=WORK/'status.json';tmp=p.with_suffix('.supervisor.tmp');tmp.write_text(json.dumps(payload,indent=2)+'\n');tmp.replace(p)
def main():
 with (WORK/'parallel_supervisor.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  workers=[];started=time.time()
  for lane in range(4):
   status=WORK/f'lane_{lane}_status.json'
   if status.exists():
    prior=json.loads(status.read_text());pid=prior.get('pid')
    if prior.get('status')=='running' and pid and Path(f'/proc/{pid}').exists():raise RuntimeError('Refusing duplicate live SHAP lane')
   log=(WORK/f'fp32_lane_{lane}.log').open('a')
   workers.append(subprocess.Popen([sys.executable,str(ROOT/'scripts/diagnostics/run_maomao_family_shap.py'),'--lane-id',str(lane),'--lanes','4'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT))
   log.close()
  while any(p.poll() is None for p in workers):
   lanes=[]
   for lane in range(4):
    p=WORK/f'lane_{lane}_status.json'
    if p.exists():lanes.append(json.loads(p.read_text()))
   save(dict(status='running',pid=os.getpid(),worker_pids=[p.pid for p in workers],patients_completed=sum(p['patients_completed'] for p in lanes),eligible_patients=9989,
     elapsed_seconds=time.time()-started,lanes=lanes))
   if any(p.poll() is not None and p.returncode for p in workers):
    save(dict(status='failed',pid=os.getpid(),worker_exit_codes=[p.poll() for p in workers],patients_completed=sum(p['patients_completed'] for p in lanes),eligible_patients=9989,
      error='A SHAP lane exited unsuccessfully; other live lanes may continue. Check individual logs, do not duplicate.'))
    raise RuntimeError('SHAP lane failed')
   time.sleep(5)
  if any(p.returncode for p in workers):raise RuntimeError('SHAP workers failed')
  save(dict(status='aggregating',pid=os.getpid(),worker_exit_codes=[p.returncode for p in workers],patients_completed=9989,eligible_patients=9989,elapsed_seconds=time.time()-started))
  sys.path.insert(0,str(ROOT))
  from scripts.diagnostics.run_maomao_family_shap import aggregate
  aggregate()
  save(dict(status='completed',pid=os.getpid(),worker_exit_codes=[p.returncode for p in workers],patients_completed=9989,eligible_patients=9989,elapsed_seconds=time.time()-started))
if __name__=='__main__':main()
