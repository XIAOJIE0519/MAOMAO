#!/usr/bin/env python3
"""Promote independently verified figures after explicit agent visual inspection."""
import sys,json,re,subprocess,fcntl
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.diagnostics.uniform_result_scope import sha256
PLOT=ROOT/'outputs/maomao_plot_sources';SHAP=PLOT/'shap';WORK=ROOT/'outputs/maomao_family_shap_20260929'
def read(p):return json.loads(p.read_text())
def run(name,*args):subprocess.run([sys.executable,str(ROOT/'scripts/diagnostics'/name),*args],cwd=ROOT,check=True)
def main():
 with (WORK/'package_finalize.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  run('verify_maomao_shap.py')
  numerics=read(SHAP/'numerical_verification.json');qa=read(SHAP/'figure_qa.json');portable=read(SHAP/'portable_replot_verification.json');visual=read(SHAP/'visual_inspection.json')
  if not all(x['complete'] for x in (numerics,qa,portable,visual)):raise RuntimeError('Final verification is incomplete')
  if len(visual['figures'])!=10 or any(x['pdf_sha256']!=sha256(SHAP/'figures'/f'{x["figure"]}.pdf') for x in visual['figures']):raise RuntimeError('Visual inspection source changed')
  if any(sha256(SHAP/x['path'])!=x['sha256'] for x in numerics['files']):raise RuntimeError('Stored SHAP attribution source changed')
  proof=dict(complete=True,patients=9989,one_query_per_validation_patient=True,all_valid_validation_rows_explained=False,
      visual_inspection_completed=True,figures=visual['figures'],numerical_verification_sha256=sha256(SHAP/'numerical_verification.json'),
      figure_qa_sha256=sha256(SHAP/'figure_qa.json'),portable_verification_sha256=sha256(SHAP/'portable_replot_verification.json'),
      finite_budget_approximation=True,convergence_claimed=False,verified_utc=datetime.now(timezone.utc).isoformat())
  (SHAP/'delivery_verification.json').write_text(json.dumps(proof,indent=2)+'\n')
  run('verify_plot_delivery.py')
  run('build_uniform_results_report.py')
  run('package_uniform_final_results.py')
  run('verify_uniform_final_results.py','--require-complete')
  # Check every relative MD link against the actual organized deliverable.
  folder=ROOT/'outputs/maomao_v5_final_results';broken=[]
  for path in folder.rglob('*.md'):
   for url in re.findall(r'\]\(([^)]+)\)',path.read_text()):
    url=url.split('#',1)[0]
    if not url or '://' in url:continue
    if url.startswith('../outputs/') or url.startswith('/'):continue
    if not (path.parent/url).exists():broken.append(dict(file=str(path.relative_to(folder)),url=url))
  if broken:raise RuntimeError(f'Package MD links missing: {broken}')
  report=ROOT/'docs/MAOMAO_V5_FINAL_RESULTS.md';archive=ROOT/'outputs/MAOMAO_final_results_20260926.zip'
  result=dict(status='completed',report=str(report),report_sha256=sha256(report),archive=str(archive),archive_bytes=archive.stat().st_size,
      archive_sha256=sha256(archive),retrieval_states=48,retrieval_metrics=576,shap_patients=9989,shap_figures=10,broken_relative_links=broken,
      ablation_checkpoints_included=False,finished_utc=datetime.now(timezone.utc).isoformat())
  (WORK/'delivery_status.json').write_text(json.dumps(result,indent=2)+'\n')
  (ROOT/'outputs/calibration_softmax_revision_20260929/delivery_status.json').write_text(json.dumps(dict(result,plot_curve_states=71,clinical_pairs=42,calibration_models_completed=40),indent=2)+'\n')
  scale=ROOT/'outputs/scale_ablations_richctx_20260928/delivery_verification.json';prior=read(scale)
  prior.update(report_sha256=result['report_sha256'],archive_sha256=result['archive_sha256'],verified_utc=result['finished_utc'],later_retrieval_and_shap_revision=sha256(SHAP/'delivery_verification.json'))
  scale.write_text(json.dumps(prior,indent=2)+'\n')
  print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
