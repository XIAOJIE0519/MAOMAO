#!/usr/bin/env python3
"""Full sealed external rows: event and semantic-family Hit@1/5/10/all.

Only frozen test inference is repeated. No training/calibration fit is performed.
Family hits map the top-k EVENT predictions to semantic families, matching train.py.
"""
import sys, json, gc, fcntl
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.diagnostics import score_uniform_external_five as scoring
from scripts.diagnostics.run_softmax_calibration_revision import model_source
from scripts.diagnostics.uniform_result_scope import sha256
from maomao.evaluation.plot_sources import RETRIEVAL_TIE_PROTOCOL
from scripts.diagnostics.maomao_display_family_groups import display_mapping

WORK=ROOT/'outputs/maomao_retrieval_revision_20260929'
PLOTS=ROOT/'outputs/maomao_plot_sources'
KEYS=('hit_at_1','hit_at_5','hit_at_10','all_true_events_hit_at_10',
      'same_family_hit_at_1','same_family_hit_at_5','same_family_hit_at_10','all_true_families_hit_at_10',
      'clinical_group_hit_at_1','clinical_group_hit_at_5','clinical_group_hit_at_10','all_true_clinical_groups_hit_at_10')
VERSION='full_row_event_and_family_retrieval_cpu_logits_argsort_v2_clinical_groups'

def read(path):return json.loads(path.read_text())
def save(path,payload):scoring.save_json(path,payload)

def intervals(v):
    n=len(v);take=min(n,100000)
    idx=torch.randperm(n,generator=torch.Generator().manual_seed(42))[:take] if n>take else torch.arange(n)
    a=torch.from_numpy(v[idx.numpy()].astype(np.float32))
    center=a.mean(0).double().numpy();full=v.mean(0)
    rng=torch.Generator().manual_seed(43 if n>take else 42)
    draws=[]
    for _ in range(200):draws.append(a[torch.randint(take,(take,),generator=rng)].mean(0).double().numpy())
    q=np.quantile(draws,[.025,.975],axis=0)
    if n>take:q=full[None,:]+(q-center[None,:])*np.sqrt(take/n)
    return full,np.clip(q,0,1)

def extract(logits,labels,temperature,ids,broad_ids,folder,metric,bias=None):
    n=len(labels);v=np.empty((n,len(KEYS)),dtype=np.uint8)
    first=np.empty(n,dtype=np.uint16);last=first.copy();card=first.copy()
    member=torch.nn.functional.one_hot(ids,int(ids.max())+1).float()
    broad_member=torch.nn.functional.one_hot(broad_ids,int(broad_ids.max())+1).float()
    top10=np.empty((n,10),dtype=np.uint8)
    for lo in range(0,n,8192):
        hi=min(lo+8192,n)
        z=torch.from_numpy(np.array(logits[lo:hi],dtype=np.float32))
        if bias is not None:z=z+torch.as_tensor(bias,dtype=z.dtype)
        z=z/temperature
        t=torch.from_numpy(np.array(labels[lo:hi],dtype=np.uint8)).bool()
        order=z.argsort(-1,descending=True);ranked=t.gather(1,order)
        top10[lo:hi]=order[:,:10].numpy()
        if not bool(t.any(1).all()):raise RuntimeError('Empty positive target row')
        first[lo:hi]=(ranked.long().argmax(1)+1).numpy()
        last[lo:hi]=(ranked.shape[1]-ranked.flip(1).long().argmax(1)).numpy()
        card[lo:hi]=t.sum(1).numpy()
        true_f=(t.float()@member).bool()
        out=[]
        for k in (1,5,10):out.append(ranked[:,:k].any(1))
        out.append(ranked[:,:10].sum(1)==t.sum(1))
        for k in (1,5,10):
            pred=torch.zeros_like(true_f).scatter_(1,ids[order[:,:k]],True)
            out.append((pred&true_f).any(1))
        out.append((~true_f|pred).all(1))
        true_b=(t.float()@broad_member).bool()
        for k in (1,5,10):
            pred=torch.zeros_like(true_b).scatter_(1,broad_ids[order[:,:k]],True)
            out.append((pred&true_b).any(1))
        out.append((~true_b|pred).all(1))
        v[lo:hi]=torch.stack(out,1).numpy()
    # Exact integer rank histograms check ALL rows against delivered inference.
    ranks=pd.read_csv(folder/'first_positive_rank.csv')
    if not np.array_equal(np.bincount(first,minlength=211)[1:],ranks.rows.to_numpy()):
        raise RuntimeError(f'Frozen inference/tie layout differs: {folder}')
    point,ci=intervals(v)
    if abs(point[0]-metric['hit_at_1'])>2e-7:raise RuntimeError('Hit@1 differs')
    if np.max(np.abs(ci[:,0]-metric['hit_at_1_95ci']))>2e-7:raise RuntimeError('Bootstrap draws differ')
    np.savez_compressed(folder/'retrieval_rows_no_identifiers.npz',metric_names=np.array(KEYS),
        hits=v,first_positive_rank=first,last_positive_rank=last,positive_count=card,
        top10_prediction_event_ids=top10,packed_true_events=np.packbits(labels,axis=1,bitorder='little'))
    result={k:float(point[i]) for i,k in enumerate(KEYS)}
    result.update({k+'_95ci':ci[:,i].tolist() for i,k in enumerate(KEYS)})
    result.update(retrieval_protocol=VERSION,retrieval_tie_protocol=RETRIEVAL_TIE_PROTOCOL,
        retrieval_rows=n,retrieval_ci_repeats=200,retrieval_ci_rows=min(n,100000),
        family_hit_definition='Map top-k event predictions to existing semantic families; any true family hit. All covers every true family within top-10 events.',
        all_hit_definition='Every true event is among the top-10 predicted events.',
        clinical_group_hit_definition='Map top-k events and true events through unchanged 63 model families into 17 prespecified clinical display groups; any/all true group coverage.',
        retrieval_true_event_bitorder='little',retrieval_true_event_classes=210,
        retrieval_vector_sha256=sha256(folder/'retrieval_rows_no_identifiers.npz'))
    return result

def main():
    torch.set_num_threads(4)
    WORK.mkdir(exist_ok=True)
    with (WORK/'queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        meta=read(scoring.TRAINING/'event_sequence_meta.json')
        ids=torch.tensor(meta['outcome_to_family'],dtype=torch.long)
        mapping,groups=display_mapping(meta['outcome_family_vocabulary']);broad_ids=torch.tensor(mapping,dtype=torch.long)[ids]
        definitions=dict(protocol=VERSION,metrics=KEYS,family_vocabulary=meta['outcome_family_vocabulary'],
            outcome_to_family=meta['outcome_to_family'],tie_protocol=RETRIEVAL_TIE_PROTOCOL,
            family_to_clinical_group=mapping,clinical_group_vocabulary=groups,packed_true_event_bitorder='little',
            packed_true_event_classes=210,test_only=True,training_or_calibration_changed=False)
        save(WORK/'definitions.json',definitions)
        save(PLOTS/'external_retrieval_definitions.json',definitions)
        completed=[]
        for site in ('ntuh','asac','uq','surgical_pooled','mimic','sicdb','mover','eicu'):
            for name in scoring.MODELS:
                dest=WORK/site/name;dest.mkdir(parents=True,exist_ok=True)
                active=scoring.RESULTS/site/name
                states=['after','before'] if name=='maomao' else ['after']
                paths={s:active/('metrics.json' if s=='after' else 'metrics_uncalibrated.json') for s in states}
                hash_model=sha256(model_source(name))
                if all(read(paths[s]).get('retrieval_protocol')==VERSION and
                       (PLOTS/f'external/{site}/{name}/{s}/retrieval_rows_no_identifiers.npz').exists() for s in states):
                    completed.extend(f'{site}/{name}/{s}' for s in states);continue
                save(WORK/'status.json',dict(status='running',active=f'{site}/{name}',completed=completed,updated_utc=scoring.stamp()))
                metrics={s:read(paths[s]) for s in states}
                if any(m['model_sha256']!=hash_model for m in metrics.values()):raise RuntimeError('Wrong frozen model')
                y=np.load(scoring.BASE/site/'test_y.npy',mmap_mode='r')
                cache=dest/'test_logits.npy'
                if not cache.exists():
                    z=np.lib.format.open_memmap(dest/'test_logits.partial.npy',mode='w+',dtype=np.float32,shape=y.shape)
                    if name=='maomao':scoring.predict_maomao(site,scoring.BASE/site,'test',z,32)
                    else:scoring.predict_flat(name,scoring.BASE/site,'test',z)
                    z.flush();del z;(dest/'test_logits.partial.npy').replace(cache)
                z=np.load(cache,mmap_mode='r')
                for state in states:
                    folder=PLOTS/f'external/{site}/{name}/{state}'
                    update=extract(z,y,metrics[state]['temperature'],ids,broad_ids,folder,metrics[state],bias=metrics[state].get('bias'))
                    metrics[state].update(update)
                    save(dest/f'{state}_retrieval_metrics.json',update)
                save(paths['after'],metrics['after'])
                if name=='maomao':
                    metrics['before']['calibrated_reference_sha256']=sha256(paths['after'])
                    save(paths['before'],metrics['before'])
                for state in states:
                    proof=read(PLOTS/f'external/{site}/{name}/{state}/provenance.json')
                    proof.update(source_metric_sha256=sha256(paths[state]),retrieval_protocol=VERSION,
                        retrieval_vector_sha256=metrics[state]['retrieval_vector_sha256'])
                    save(PLOTS/f'external/{site}/{name}/{state}/provenance.json',proof)
                    completed.append(f'{site}/{name}/{state}')
                print(f'RETRIEVAL COMPLETED {site}/{name}: {len(y)} full test rows; {len(completed)}/48 states',flush=True)
                del z,y;cache.unlink();gc.collect()
        rows=[]
        for item in completed:
            site,name,state=item.split('/')
            m=read(scoring.RESULTS/site/name/('metrics.json' if state=='after' else 'metrics_uncalibrated.json'))
            for key in KEYS:
                rows.append(dict(site=site,model=name,calibration_state=state,test_rows=m['test_rows'],metric=key,value=m[key],
                    ci_lower=m[key+'_95ci'][0],ci_upper=m[key+'_95ci'][1],source_metric_sha256=sha256(scoring.RESULTS/site/name/('metrics.json' if state=='after' else 'metrics_uncalibrated.json'))))
        pd.DataFrame(rows).to_csv(PLOTS/'external_retrieval_metrics.csv',index=False)
        save(WORK/'status.json',dict(status='completed',completed=completed,states=48,updated_utc=scoring.stamp()))

if __name__=='__main__':
    try:main()
    except Exception as error:
        save(WORK/'status.json',dict(status='failed',error=repr(error),updated_utc=scoring.stamp()));raise
