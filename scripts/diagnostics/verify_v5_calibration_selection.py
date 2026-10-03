#!/usr/bin/env python3
"""Reconstruct V5 calibration selection from source admissions and row windows.

Uses no prediction, fitting or test-label input. Anonymous row group codes are
reconstructed from sequence lengths instead of trusting the worker's cache.
Only group/row counts and digests are written; no patient identifiers or codes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.diagnostics.build_fullscale_external_rows import SOURCES
from scripts.diagnostics.uniform_result_scope import sha256
from maomao.evaluation.event_bias_calibration import PROTOCOL,validate_fit

ROWS=ROOT/'outputs/final_experiment_results_20260923/classical_full_scale/external'
REV=ROOT/'outputs/calibration_v5_bias_revision_20260930'


def read(path):return json.loads(path.read_text())


def partition(groups):
    """Independently implement the specified seed42 patient 80:20 selection."""
    groups=np.asarray(groups,dtype=np.int64);unique=np.unique(groups)
    assert len(unique)>=2
    permuted=np.random.default_rng(42).permutation(unique)
    n=min(len(unique)-1,max(1,round(.2*len(unique))))
    selected=np.sort(permuted[:n]);mask=np.isin(groups,selected)
    assert mask.any() and not mask.all()
    return dict(seed=42,fit_groups=int(len(unique)-n),selection_groups=int(n),
        fit_rows=int((~mask).sum()),selection_rows=int(mask.sum()),patient_or_record_overlap=0,
        row_group_sha256=hashlib.sha256(groups.tobytes()).hexdigest(),
        selection_group_sha256=hashlib.sha256(selected.tobytes()).hexdigest())


def source_partitions(site,stage,required_counts):
    source=ROOT/SOURCES[site];rows=ROWS/site;manifest=read(rows/'manifest.json')
    columns=pd.read_csv(source/'admissions.csv',nrows=0).columns
    column=manifest['split'].get('group_column','subject_id' if 'subject_id' in columns else 'admission_id')
    admissions=pd.read_csv(source/'admissions.csv',usecols=[column],dtype={'subject_id':'string'})
    codes,_=pd.factorize(admissions[column].fillna('missing:').astype(str),sort=True)
    ptr=np.load(source/'sequence_ptr.npy',mmap_mode='r')
    assert len(ptr)-1==len(admissions)
    lengths=np.diff(ptr);assert (lengths>=0).all()
    final=np.maximum(lengths-256,0)
    # Regular stride128 windows plus a final end-aligned window when needed.
    windows_per_admission=np.where(lengths<=256,1,final//128+1+(final%128!=0))
    ends=np.cumsum(windows_per_admission,dtype=np.int64)
    row_windows=np.load(rows/'calibration_window_indices.npy',mmap_mode='r')
    assert len(row_windows)==manifest['calibration_target_rows'] and row_windows.min()>=0 and row_windows.max()<ends[-1]
    group_codes=codes[np.searchsorted(ends,row_windows,side='right')].astype(np.int64)
    full=partition(group_codes)
    cached=stage/'private_group_inputs'/f'{site}.npy'
    if cached.exists():
        cached_codes=np.load(cached,mmap_mode='r')
        assert np.array_equal(cached_codes,group_codes),f'Worker grouping cache differs from source records: {site}'
    result={210:full}
    y=np.load(rows/'calibration_y.npy',mmap_mode='r');assert y.shape==(len(group_codes),210)
    for count in sorted(set(required_counts)-{210}):
        indices=read(ROOT/f'outputs/scale_ablations_richctx_20260928/specifications/vocab_{count}.json')['indices']
        mask=np.empty(len(y),dtype=bool)
        for lo in range(0,len(y),65536):mask[lo:lo+65536]=y[lo:lo+65536][:,indices].any(1)
        result[count]=partition(group_codes[mask])
    files=[source/'admissions.csv',source/'sequence_ptr.npy',rows/'manifest.json',rows/'calibration_window_indices.npy']
    proof=dict(site=site,source_files=[dict(path=str(p.relative_to(ROOT)),sha256=sha256(p)) for p in files],
        tasks={str(k):v for k,v in result.items()},worker_private_codes_match_source=True,
        definition='Source admission codes reconstructed using sequence-length window counts, original calibration window coordinates and per-vocabulary positive-target masks; no test labels read')
    return result,proof


def verify(stage=REV,require_complete=False):
    paths=sorted((stage/'external').glob('*/*/calibration_fit.json'))+sorted((stage/'external_ablations').glob('*/*/*/calibration_fit.json'))
    fits=[(path,read(path)) for path in paths]
    counts={}
    for _,fit in fits:counts.setdefault(fit['site'],set()).add(len(fit['bias']))
    by_site={};checks=[]
    for path,fit in fits:
        site=fit['site'];classes=len(fit['bias'])
        assert fit['protocol']==PROTOCOL and classes in (50,100,150,210)
        if site not in by_site:by_site[site]=source_partitions(site,stage,counts[site])
        expected=by_site[site][0][classes]
        for key,value in expected.items():
            assert fit['selection_split'][key]==value,f'Source selection differs: {site}/{fit["model"]}/{key}'
        validate_fit(fit,expected['fit_rows']+expected['selection_rows'],classes)
        checks.append(dict(site=site,model=fit['model'],classes=classes,fit_sha256=sha256(path),
            fit_path=str(path.relative_to(ROOT)),source_partition_verified=True))
    complete=len(checks)==125 and len({x['fit_path'] for x in checks})==125
    result=dict(complete=complete,protocol=PROTOCOL,calibration_fits_checked=len(checks),fits_total=125,
        source_patient_and_row_partition_reconstructed=True,no_test_labels_read=True,
        source_partitions=[item[1] for item in by_site.values()],checks=checks,
        checked_utc=datetime.now(timezone.utc).isoformat())
    path=stage/'selection_partition_verification.json';temp=path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
    if require_complete and not complete:raise RuntimeError('Every one of the 125 calibration fits must be checked before promotion')
    print(f'Source-based calibration selection verified: {len(checks)}/125 fits; complete={complete}',flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',type=Path,default=REV);parser.add_argument('--require-complete',action='store_true');args=parser.parse_args()
    verify(args.stage,args.require_complete)
