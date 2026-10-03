#!/usr/bin/env python3
"""Run mandatory PDF text, collision and rendered-alignment gates."""
import sys,json,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.diagnostics.uniform_result_scope import sha256
from scripts.diagnostics.maomao_display_family_groups import display_mapping
OUT=ROOT/'outputs/maomao_plot_sources/shap';SKILL=Path('/home/yunkunshi/.codex/skills/nature-figure/scripts')
def main():
    source=ROOT/'scripts/diagnostics/plot_maomao_delphi_shap.py'
    checks=[];errors=[]
    # Independently check that the plotted data really contain all three
    # stored lag layers with identical event axes and unmodified SHAP values.
    with np.load(OUT/'event_shap_matrices_by_family.npz') as full, np.load(OUT/'figure4c_display_matrix.npz') as shown:
        support=full['patients_with_feature'];families=full['family_names'].tolist()
        mapping,_=display_mapping(families)
        expected=np.flatnonzero(np.all(support>=6,axis=0))
        death=full['event_names'].tolist().index('inhospital_death')
        expected=expected[expected!=death]
        expected=expected[np.argsort(full['outcome_to_family'][expected],kind='stable')]
        expected=expected[np.argsort(np.array(mapping)[full['outcome_to_family'][expected]],kind='stable')]
        outputs=np.r_[expected,death]
        if not np.array_equal(shown['lag_ids'],np.arange(3)) or shown['lag_labels'].tolist()!=['<2 h','2–<24 h','≥24 h']:
            errors.append('Figure4c: three report-matched time scales missing')
        if not np.array_equal(shown['predictor_event_ids'],expected) or not np.array_equal(shown['predicted_event_ids'],outputs):
            errors.append('Figure4c: common supported event axes differ')
        if not np.array_equal(shown['predictor_support'],support[:,expected]) or not np.array_equal(shown['folds'],full['exp_mean_shap'][:,expected][:,:,outputs]):
            errors.append('Figure4c: display matrix differs from actual three-lag source')
        data_check=dict(panels=3,predictor_rows=len(expected),predicted_columns=len(outputs),
            lag_intervals_hours=['[0,2)','[2,24)','[24,infinity)'],same_event_axes=True,
            source_matrix_sha256=sha256(OUT/'event_shap_matrices_by_family.npz'),
            display_matrix_sha256=sha256(OUT/'figure4c_display_matrix.npz'))
    p=subprocess.run([sys.executable,str(SKILL/'validate_figure.py'),str(source),'--json'],capture_output=True,text=True)
    (OUT/'figure_source_preflight.json').write_text(p.stdout)
    data=json.loads(p.stdout)
    if data['summary']['counts']['FAIL']:errors.append('Source preflight FAIL')
    expected=('figure4a_maomao_embeddings','maomao_embeddings_63families','figure4c_maomao_shap_by_family','maomao_family_shap_full_matrix',
        'maomao_shap_family_importance','maomao_shap_death_family_distribution','maomao_shap_local_family_waterfalls',
        'maomao_shap_clinical_group_importance','maomao_shap_death_clinical_group_distribution','maomao_shap_local_clinical_group_waterfalls')
    for name in expected:
        folder=OUT/'figures';pdf=folder/f'{name}.pdf'
        if any(not (folder/f'{name}.{suffix}').exists() for suffix in ('pdf','svg','png','tiff','alignment.json')):
            errors.append(f'{name}: missing final format/alignment');continue
        p=subprocess.run([sys.executable,str(SKILL/'audit_pdf_text.py'),str(pdf),'--min-pt','5','--json'],capture_output=True,text=True)
        (folder/f'{name}.font.json').write_text(p.stdout)
        font=json.loads(p.stdout)
        if p.returncode:errors.append(f'{name}: font audit failed')
        p=subprocess.run([sys.executable,str(SKILL/'audit_figure_collisions.py'),str(pdf),'--json-out',str(folder/f'{name}.collision.json')],capture_output=True,text=True)
        collision=json.loads((folder/f'{name}.collision.json').read_text())
        alignment=json.loads((folder/f'{name}.alignment.json').read_text())
        if p.returncode or collision['summary']['fail']:errors.append(f'{name}: collision FAIL')
        single_panel=name in ('maomao_shap_family_importance','maomao_shap_death_family_distribution',
            'maomao_shap_clinical_group_importance','maomao_shap_death_clinical_group_distribution')
        if alignment['summary']['fail'] or (not single_panel and alignment.get('auditable') is not True):errors.append(f'{name}: alignment FAIL/not auditable')
        if single_panel and alignment.get('applicable') is False and alignment.get('verdict')!='NOT APPLICABLE':errors.append(f'{name}: single-panel alignment status invalid')
        checks.append(dict(figure=name,pdf_sha256=sha256(pdf),fonts=font,collision_summary=collision['summary'],alignment_verdict=alignment['verdict']))
        print(name,collision['verdict'],collision['summary'],flush=True)
    proof=dict(complete=not errors,errors=errors,figures=checks,source_sha256=sha256(source),figure4c_data_check=data_check,
        source_warnings_review='RNG is used only for patient bootstrap and vertical point jitter, not simulated clinical values. The three-panel Figure4c and 63-family companion use enlarged canvases to retain readable labels, not a final Nature production size. Rotated ticks are inspected in final PDFs.',
        final_visual_inspection_pending=True)
    (OUT/'figure_qa.json').write_text(json.dumps(proof,indent=2)+'\n')
    if errors:raise SystemExit(1)
if __name__=='__main__':main()
