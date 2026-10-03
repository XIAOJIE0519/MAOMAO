#!/usr/bin/env python3
"""Independent integer-count reconstruction of all retrieval bootstrap CIs."""
import sys,json
from pathlib import Path
import numpy as np,torch,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.diagnostics.complete_external_retrieval import KEYS
from scripts.diagnostics.uniform_result_scope import sha256
PLOT=ROOT/'outputs/maomao_plot_sources';EXT=ROOT/'outputs/external_validation_final_maomao_uniform'
def main():
    torch.set_num_threads(2);errors=[];checks=[]
    for path in sorted((PLOT/'external').glob('*/*/*/retrieval_rows_no_identifiers.npz')):
        site,model,state=path.relative_to(PLOT/'external').parts[:3]
        source=EXT/site/model/('metrics.json' if state=='after' else 'metrics_uncalibrated.json')
        m=json.loads(source.read_text())
        with np.load(path) as a:v=a['hits'].astype(np.uint8)
        n=len(v);take=min(n,100000)
        index=torch.randperm(n,generator=torch.Generator().manual_seed(42))[:take].numpy() if n>take else np.arange(n)
        sample=v[index];center=sample.sum(0,dtype=np.int64)/take;full=v.sum(0,dtype=np.int64)/n
        generator=torch.Generator().manual_seed(43 if n>take else 42)
        draws=[]
        for _ in range(200):
            rows=torch.randint(take,(take,),generator=generator).numpy();draws.append(sample[rows].sum(0,dtype=np.int64)/take)
        quantiles=np.quantile(draws,[.025,.975],axis=0)
        if n>take:quantiles=full[None,:]+(quantiles-center[None,:])*np.sqrt(take/n)
        quantiles=np.clip(quantiles,0,1)
        err=max(abs(quantiles[side,i]-m[key+'_95ci'][side]) for i,key in enumerate(KEYS) for side in (0,1))
        if err>2e-7:errors.append(f'{site}/{model}/{state}: interval count reconstruction differs {err}')
        checks.append(dict(site=site,model=model,state=state,full_rows=n,ci_rows=take,draws=200,max_ci_difference=float(err),source_metric_sha256=sha256(source)))
        print(f'Retrieval CI verified {len(checks)}/48: {site}/{model}/{state}; max difference {err:.3g}',flush=True)
    result=dict(complete=not errors and len(checks)==48,errors=errors,states=len(checks),metrics_per_state=len(KEYS),checks=checks,
       method='Independent int64 hit counts, existing Torch seeded resampling indices, same subsample/scaled-width protocol; point estimates use all rows. No patient cluster CI claimed.')
    (PLOT/'external_retrieval_interval_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['complete']:raise SystemExit(1)
if __name__=='__main__':main()
