#!/usr/bin/env python3
"""Replot three external examples from copied tables/bins and bundled helper.

No model, private row grouping, calibration fitting or original project imports
are used by the isolated plotter. This verifies reproducibility, not visual QA.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np
import pymupdf
from fontTools.ttLib import TTFont
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
NAMES=('external_maomao_brier','mimic_micro_roc_pr','mover_maomao_reliability')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def native_pdf(path):
    with pymupdf.open(path) as doc:
        assert len(doc)==1
        page=doc[0]
        assert abs(page.rect.width*25.4/72-210)<.001 and page.rect.height>page.rect.width
        assert not page.get_images(full=True) and not page.get_image_info()
        spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for line in b['lines'] for s in line['spans'] if s['text'].strip()]
        assert spans and all('Arial' in s['font'] and min(abs(s['size']-8),abs(s['size']-10))<.001 for s in spans)
        for font in page.get_fonts(full=True):
            _,ext,_,data=doc.extract_font(font[0])
            assert ext=='ttf' and TTFont(io.BytesIO(data))['name'].getDebugName(1)=='Arial'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=ROOT/'outputs/maomao_plot_sources')
    parser.add_argument('--reference',type=Path,default=ROOT/'outputs/maomao_plot_sources/example_figures')
    parser.add_argument('--proof',type=Path,default=ROOT/'outputs/maomao_plot_sources/external_example_portable_replot_verification.json')
    parser.add_argument('--plotter',type=Path,default=ROOT/'outputs/maomao_plot_sources/plot_examples.py')
    args=parser.parse_args()
    inputs=['metrics_long.csv']
    inputs += [f'external/mimic/{model}/after/full_row_curves.npz' for model in ('univariate','logistic_regression','xgboost','ann','maomao')]
    inputs += ['external/mimic/maomao/before/full_row_curves.npz']
    inputs += [f'external/mover/maomao/{state}/reliability.csv' for state in ('before','after')]
    records=[];sources=[]
    with tempfile.TemporaryDirectory(prefix='maomao-external-example-replot-') as temp:
        scratch=Path(temp)
        for relative in inputs:
            source=args.root/relative;target=scratch/relative
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
            sources.append(dict(path=relative,sha256=digest(source)))
        shutil.copy2(args.plotter,scratch/'plot_examples.py')
        shutil.copy2(args.plotter.parent/'audit_panel_alignment.py',scratch/'audit_panel_alignment.py')
        output=scratch/'figures'
        subprocess.run([sys.executable,str(scratch/'plot_examples.py'),'--root',str(scratch),
                        '--output',str(output),'--external-only'],cwd=scratch,check=True,
                        env={**os.environ,'OPENBLAS_NUM_THREADS':'1'})
        for name in NAMES:
            reference=args.reference/(name+'.png');replot=output/(name+'.png')
            with Image.open(reference) as a,Image.open(replot) as b:
                assert a.mode==b.mode and a.size==b.size and np.array_equal(np.asarray(a),np.asarray(b)),f'PNG pixel mismatch: {name}'
            native_pdf(output/(name+'.pdf'))
            if name=='mimic_micro_roc_pr':
                assert json.loads((output/(name+'.alignment.json')).read_text())['verdict']=='PASS'
            records.append(dict(name=name,identical_replotted_png=True,
                native_vector_embedded_arial_verified=True,
                reference_png_sha256=digest(reference),replotted_png_sha256=digest(replot)))
    result=dict(complete=True,scope='Three external examples from copied result tables/histograms and sibling alignment helper only; visual inspection is separate',
        examples=records,input_files=sources,plotter_sha256=digest(args.plotter),
        alignment_helper_sha256=digest(args.plotter.parent/'audit_panel_alignment.py'),
        checked_utc=datetime.now(timezone.utc).isoformat())
    args.proof.parent.mkdir(parents=True,exist_ok=True)
    args.proof.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('Three isolated external example replots pixel-identical; native Arial/vector PDFs verified.')


if __name__=='__main__':main()
