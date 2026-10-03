#!/usr/bin/env python3
"""Replot delivered figures using only files in the organized result bundle.

Work is isolated in a temporary directory; original figures remain untouched.
This checks reproducibility, real embedded fonts and vectors, not human review.
"""
import argparse,hashlib,io,json,os,shutil,subprocess,sys,tempfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pymupdf
from fontTools.ttLib import TTFont
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/maomao_manuscript_figures_20260929'
BUNDLE=ROOT/'outputs/maomao_v5_final_results'
def read(p):return json.loads(p.read_text())
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()
def copy_source(src,dst):
    try:os.link(src,dst)
    except OSError:shutil.copy2(src,dst)
    return str(dst)
def native_pdf(path):
    with pymupdf.open(path) as doc:
        assert len(doc)==1
        page=doc[0]
        assert abs(page.rect.width*25.4/72-210)<.001 and page.rect.height>page.rect.width
        assert not page.get_images(full=True) and not page.get_image_info()
        spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans'] if s['text'].strip()]
        assert spans and all('Arial' in s['font'] and min(abs(s['size']-8),abs(s['size']-10))<.001 for s in spans)
        for f in page.get_fonts(full=True):
            _,ext,_,data=doc.extract_font(f[0]);assert ext=='ttf' and 'Arial' in f[3]
            assert TTFont(io.BytesIO(data))['name'].getDebugName(1)=='Arial'
def verify(numbers):
    delivery=read(BUNDLE/'manuscript_figures/delivery_verification.json')
    assert set(numbers)<=set(delivery['verified_figures'])
    if 3 in numbers:assert delivery['complete'] and delivery['external_exact_row_audits_complete']
    results=[];inputs={}
    with tempfile.TemporaryDirectory(prefix='maomao-portable-replot-',dir=ROOT/'outputs') as directory:
        scratch=Path(directory)
        # Sources are read-only to the plotter; hard links avoid duplicating the
        # 1.1 GB of bundled data. Every file still comes from the delivered bundle.
        shutil.copytree(BUNDLE/'plot_sources',scratch/'plot_sources',copy_function=copy_source)
        shutil.copytree(BUNDLE/'internal/maomao',scratch/'internal/maomao',copy_function=copy_source)
        shutil.copytree(BUNDLE/'manuscript_figures',scratch/'manuscript_figures')
        work=scratch/'manuscript_figures'
        with (OUT/'portable_replot.log').open('w') as log:
            subprocess.run([sys.executable,str(work/'build_manuscript_figures.py'),'--figures',*map(str,numbers)],
                           cwd=work,stdout=log,stderr=subprocess.STDOUT,check=True)
        artifacts=[f'Figure_{n}' for n in numbers]+(delivery.get('verified_supplements',[]) if set(numbers)=={1,2,3,4,5} else [])
        for name in artifacts:
            reference=BUNDLE/f'manuscript_figures/figures/{name}.png'
            replot=work/f'figures/{name}.png'
            with Image.open(reference) as a,Image.open(replot) as b:
                assert a.mode==b.mode and a.size==b.size and np.array_equal(np.asarray(a),np.asarray(b)),f'PNG pixel mismatch: {name}'
            native_pdf(work/f'figures/{name}.pdf')
            assert not list((work/'figures').glob('*.svg'))
            layout=read(work/f'figures/{name}.alignment.json');assert layout['auditable'] and layout['verdict']=='PASS'
            prov='figure_'+name.split('_')[-1]+'_provenance.json' if name.startswith('Figure_') else 'supplementary_'+name.split('_')[1]+'.provenance.json'
            for source in read(work/f'source_data/{prov}')['sources']:
                assert digest(scratch/source['path'])==source['sha256']
                inputs[source['path']]=source['sha256']
            results.append({'figure':name,'identical_replotted_png':True,'standalone_plotting_script':True,
                            'pdf_native_vector_and_actual_arial_verified':True,'rendered_alignment_pass':True,
                            'reference_png_sha256':digest(reference),'replotted_png_sha256':digest(replot)})
            print(f'{name}: packaged-only replot pixel-identical; real Arial 8/10 and native PDF vectors verified',flush=True)
    result={'complete':True,'figures':results,'all_five_figures_complete':set(numbers)=={1,2,3,4,5},
            'scope':'All five delivered manuscript figures' if len(numbers)==5 else 'Currently delivered figures only; pending figures are not covered',
            'packaged_plotting_script_sha256':digest(BUNDLE/'manuscript_figures/build_manuscript_figures.py'),
            'revision_module_sha256':digest(BUNDLE/'manuscript_figures/maomao_figure_revision.py'),
            'all_twelve_supplements_complete':len([r for r in results if r['figure'].startswith('Supplementary_')])==12,
            'input_files':[{'path':path,'sha256':value} for path,value in sorted(inputs.items())],
            'updated_utc':datetime.now(timezone.utc).isoformat()}
    p=OUT/'portable_replot_verification.json';temporary=p.with_suffix('.json.tmp');temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(p)
    return result
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--figures',nargs='+',type=int,default=[1,2,4,5]);args=parser.parse_args()
    print(json.dumps(verify(args.figures),indent=2))
