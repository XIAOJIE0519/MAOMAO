#!/usr/bin/env python3
"""Prepare final source-verified figures once the existing SHAP lanes finish."""
import sys,os,json,time,subprocess,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT/'outputs/maomao_family_shap_20260929';OUT=ROOT/'outputs/maomao_plot_sources/shap'
def run(script,*args):subprocess.run([sys.executable,str(ROOT/'scripts/diagnostics'/script),*args],cwd=ROOT,check=True)
def main():
 with (WORK/'delivery_prepare.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  prior=-1
  while True:
   s=json.loads((WORK/'status.json').read_text())
   if s['status']=='completed':break
   if s['status']=='failed':raise RuntimeError('SHAP worker failure; do not claim complete')
   if not Path(f'/proc/{s["pid"]}').exists():raise RuntimeError('Existing supervisor no longer live')
   bucket=s['patients_completed']//1000
   if bucket!=prior:print(f'Waiting existing SHAP lanes: {s["patients_completed"]}/9989',flush=True);prior=bucket
   time.sleep(5)
  # This step recomputes only the prespecified budget diagnostic, no model fit.
  budget=OUT/'budget_sensitivity.json'
  while not budget.exists() or json.loads(budget.read_text()).get('status')!='completed':
   alive=False
   for entry in Path('/proc').iterdir():
    if entry.name.isdigit():
     try:
      cmd=(entry/'cmdline').read_bytes()
      if b'maomao_shap_budget_sensitivity.py' in cmd and b'python' in cmd:alive=True
     except (OSError,PermissionError):pass
   if not alive:raise RuntimeError('Budget diagnostic missing and no worker active')
   time.sleep(5)
  run('verify_maomao_shap.py')
  run('plot_maomao_delphi_shap.py')
  run('audit_maomao_shap_figures.py')
  run('build_maomao_shap_report.py')
  subprocess.run([sys.executable,str(OUT/'replot_maomao_shap.py'),'--output',str(WORK/'portable_replot')],cwd=OUT,check=True)
  from PIL import Image
  import numpy as np
  pairs=[]
  for p in sorted((OUT/'figures').glob('*.png')):
   other=WORK/'portable_replot'/p.name
   same=other.exists() and np.array_equal(np.asarray(Image.open(p)),np.asarray(Image.open(other)))
   pairs.append(dict(figure=p.name,identical_png_pixels=bool(same)))
  proof=dict(complete=all(x['identical_png_pixels'] for x in pairs) and len(pairs)==10,figures=pairs,
      uses_only_delivered_artifacts=True,checkpoint_or_original_data_required=False)
  (OUT/'portable_replot_verification.json').write_text(json.dumps(proof,indent=2)+'\n')
  if not proof['complete']:raise RuntimeError('Portable final rendering differs')
  print('Final full-cohort figures prepared. Visual inspection required before promotion/package.',flush=True)
if __name__=='__main__':main()
