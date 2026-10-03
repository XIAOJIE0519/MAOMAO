#!/usr/bin/env python3
"""Verify real PDF fonts, vector geometry, source hashes and rendered layout."""
import argparse,io,json,sys,subprocess
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pymupdf
from fontTools.ttLib import TTFont
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.diagnostics.uniform_result_scope import sha256
OUT=ROOT/'outputs/maomao_manuscript_figures_20260929';FIG=OUT/'figures';SKILL=Path('/home/yunkunshi/.codex/skills/nature-figure/scripts')
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def supplement_names():
    endpoints=read(OUT/'source_data/clinical_curve_dictionary.json')
    return ['Supplementary_S3_external',*[f'Supplementary_S4_{i+1:02d}_{key}' for i,key in enumerate(endpoints)],'Supplementary_S5_waterfalls']
def verify(numbers,record_visual=False):
    visual_path=OUT/'visual_inspection.json';visual=read(visual_path) if visual_path.exists() else {'figures':{}}
    results=[]
    supplement=supplement_names() if set(numbers)=={1,2,3,4,5} else []
    for n in [*numbers,*supplement]:
        stem=f'Figure_{n}' if isinstance(n,int) else n
        pdf=FIG/f'{stem}.pdf';png=FIG/f'{stem}.png'
        assert pdf.exists() and png.exists(),f'Missing {stem} export'
        assert not list(FIG.glob('*.svg')),'SVG images must not be delivered'
        doc=pymupdf.open(pdf);assert len(doc)==1;page=doc[0]
        assert abs(page.rect.width*25.4/72-210)<.001 and page.rect.height>page.rect.width
        assert not page.get_images(full=True) and not page.get_image_info(),f'Bitmap in PDF {n}'
        spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans'] if s['text'].strip()]
        assert all('Arial' in s['font'] and min(abs(s['size']-8),abs(s['size']-10))<.001 for s in spans)
        for f in page.get_fonts(full=True):
            assert 'Arial' in f[3],f'Unexpected font {f}'
            _,ext,typ,data=doc.extract_font(f[0]);assert ext=='ttf' and data,f'Nonembedded TrueType font {f}'
            tt=TTFont(io.BytesIO(data));assert tt['name'].getDebugName(1)=='Arial'
        layout=read(FIG/f'{stem}.alignment.json');assert layout['auditable'] and layout['verdict']=='PASS'
        command=[sys.executable,str(SKILL/'audit_figure_collisions.py'),str(pdf),'--json-out',str(FIG/f'{stem}.collision.json')]
        audit=subprocess.run(command,capture_output=True,text=True)
        if audit.returncode:raise RuntimeError(audit.stdout+audit.stderr)
        collision=read(FIG/f'{stem}.collision.json');assert collision['verdict']=='PASS' and not collision['summary']['fail'] and not collision['summary']['warn']
        glyph=subprocess.run([sys.executable,str(SKILL/'audit_pdf_text.py'),str(pdf),'--min-pt','8','--json'],capture_output=True,text=True)
        assert glyph.returncode==0,glyph.stdout+glyph.stderr;write(FIG/f'{stem}.glyphs.json',json.loads(glyph.stdout))
        source_name=f'figure_{n}_provenance.json' if isinstance(n,int) else ('supplementary_S3.provenance.json' if n.startswith('Supplementary_S3') else 'supplementary_S4.provenance.json' if n.startswith('Supplementary_S4') else 'supplementary_S5.provenance.json')
        provenance=read(OUT/f'source_data/{source_name}')
        for s in provenance['sources']:assert sha256(ROOT/s['path'])==s['sha256'],f'Stale source {s["path"]}'
        hashes={'pdf_sha256':sha256(pdf),'png_sha256':sha256(png)}
        if record_visual:
            prior=visual['figures'].get(str(n),{})
            # Identical exports retain their actual prior inspection date.
            # A new receipt is recorded only for newly inspected export bytes.
            if not (prior.get('whole_and_panels_inspected') and
                    all(prior.get(k)==v for k,v in hashes.items())):
                visual['figures'][str(n)]={**hashes,'whole_and_panels_inspected':True,'inspected_utc':datetime.now(timezone.utc).isoformat()}
        item=visual['figures'].get(str(n),{})
        seen=bool(item.get('whole_and_panels_inspected') and all(item.get(k)==v for k,v in hashes.items() if k!='svg_sha256'))
        result={'figure':n,**hashes,'font':'Embedded actual Arial','sizes_pt':sorted({round(s['size'],3) for s in spans}),
                'width_mm':page.rect.width*25.4/72,'height_mm':page.rect.height*25.4/72,'embedded_images':0,
                'vector_only':True,'alignment_pass':True,'collision_failures':0,'source_hashes_verified':True,'visual_inspection_current':seen}
        write(FIG/f'{stem}.export_verification.json',result);results.append(result)
        print(f'Figure {n}: real Arial 8/10, A4 width, vector-only, source/layout/collision verified; visual={seen}',flush=True)
    if record_visual:write(visual_path,visual)
    # Include prior verified exports only when their current bytes still match.
    by={r['figure']:r for r in results}
    for p in FIG.glob('Figure_*.export_verification.json'):
        r=read(p);n=r['figure']
        if n in by:continue
        if all(sha256(FIG/f'Figure_{n}.{ext}')==r[f'{ext}_sha256'] for ext in ('pdf','png')):by[n]=r
    queue=read(OUT/'external_ablations/queue_status.json')
    row_path=OUT/'external_ablation_row_verification.json'
    row_proof=read(row_path) if row_path.exists() else {}
    rows_complete=bool(row_proof.get('complete') and row_proof.get('completed_jobs_audited')==75 and row_proof.get('calibration_pairs_audited')==85)
    if 3 in by:
        assert rows_complete,'Figure 3 requires all exact before/after row audits'
        for pair in row_proof['pairs']:
            parent='reference_projected' if pair['model'].startswith('reference_vocab_') else pair['model']
            target=OUT/'external_ablations'/pair['site']/parent/pair['model']
            assert sha256(target/'metrics_before.json')==pair['before_sha256']
            assert sha256(target/'metrics_after.json')==pair['after_sha256']
        for site,digest in row_proof['source_row_audit_files'].items():
            assert sha256(OUT/f'source_data/external_row_audits/{site}.json')==digest
    main=[n for n in by if isinstance(n,int)];sup=[n for n in by if isinstance(n,str)]
    risk=read(OUT/'source_data/clinical_risk_preparation.json');assert risk['complete'] and risk['test_patient_overlap']==0 and not risk['fitted_on_sealed_test'] and not risk['internal_validation_patients_used_for_fitting']
    assert len(risk['score_mappings'])==52
    p2=read(OUT/'source_data/figure_2_provenance.json');assert p2['all_six_cohorts_per_metric'] and len(set(p2['cohort_marker_shapes'].values()))==6
    from maomao.evaluation.softmax_calibration import PROTOCOL,LEGACY_PROTOCOL
    if PROTOCOL!=LEGACY_PROTOCOL:
        assert p2['heatmap_metric_definitions_unchanged'] and p2['heatmap_values_from_current_full_row_results']
        assert p2['axis_limits_cover_all_current_point_estimates_and_cis']
    else:assert p2.get('heatmap_values_unchanged') or p2.get('heatmap_values_from_current_full_row_results')
    p3=read(OUT/'source_data/figure_3_provenance.json');assert p3['main_panels']==['a','d','g','j']
    p4=read(OUT/'source_data/figure_4_provenance.json');assert p4['main_roc_count']==6 and p4['excluded_original_panels']==['b','d','h','i'] and not p4['maomao_minus_score_panels']
    p5=read(OUT/'source_data/figure_5_provenance.json');assert p5['lag_labels']==['<2 h','2–<24 h','≥24 h'] and len(p5['layout_rows'])==4 and not p5['individual_waterfall_main']
    sw=read(OUT/'source_data/supplementary_S5.provenance.json');assert sw['cases']==list(range(9)) and sw['additivity_checked']
    complete=(set(main)=={1,2,3,4,5} and set(sup)==set(supplement_names()) and all(r['visual_inspection_current'] for r in by.values()) and queue['status']=='completed' and len(queue['completed'])==75 and rows_complete)
    state={'complete':complete,'verified_figures':[n for n in sorted(main) if by[n]['visual_inspection_current']],
           'verified_supplements':[n for n in sorted(sup) if by[n]['visual_inspection_current']],
           'pending_figures':[n for n in range(1,6) if n not in by or not by[n]['visual_inspection_current']],
           'external_jobs_completed':len(queue['completed']),'external_jobs_total':75,'external_queue_status':queue['status'],
           'external_exact_row_audits_complete':rows_complete,'external_calibration_pairs_audited':row_proof.get('calibration_pairs_audited',0),
           'clinical_exact_metrics_verified':read(OUT/'source_data/clinical_numerical_verification.json')['complete'],
           'figure_5_three_history_scales':['<2 h','2–<24 h','≥24 h'],
           'export_formats':['PDF','PNG'],'clinical_training_only_risk_mappings_verified':True,
           'figure_5_scale_panel':'d (formerly c); history lag, not future MAE',
           'plotting_sources_sha256':{name:sha256(ROOT/'scripts/diagnostics'/name) for name in ('build_manuscript_figures.py','maomao_figure_revision.py')},
           'figures':[by[n] for n in sorted(main)+sorted(sup)],'updated_utc':datetime.now(timezone.utc).isoformat()}
    write(OUT/'delivery_verification.json',state);return state
def main():
    p=argparse.ArgumentParser();p.add_argument('--figures',nargs='+',type=int,default=[1,2,4,5]);p.add_argument('--record-visual',action='store_true');a=p.parse_args()
    print(json.dumps(verify(a.figures,a.record_visual),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
