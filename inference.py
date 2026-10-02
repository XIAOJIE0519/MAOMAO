"""MAOMAO: raw clinical-event JSON -> checkpoint-compatible tensors -> predictions."""
import json,math
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from model import EventMAOMAO, dual_timescale_expected_wait

MODEL_ID='luan0519/MAOMAO'
FILES=['config.json','vocabulary.json','calibrations.json','model.safetensors']

def assets(directory=None):
    if directory:return Path(directory)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(MODEL_ID,allow_patterns=FILES))

class Predictor:
    def __init__(self,directory=None,device='cpu'):
        self.path=assets(directory);self.device=device
        self.vocab=json.loads((self.path/'vocabulary.json').read_text())
        cfg=json.loads((self.path/'config.json').read_text())
        self.model=EventMAOMAO(**cfg,outcome_family_ids=torch.tensor(self.vocab['outcome_to_family'])).eval()
        self.model.load_state_dict(load_file(str(self.path/'model.safetensors')))
        self.model.to(device)
        self.calibrations=json.loads((self.path/'calibrations.json').read_text())

    def encode(self,payload,device=None):
        if isinstance(payload,str):payload=json.loads(payload)
        if not isinstance(payload,dict):raise ValueError('Input must be a JSON object / 输入必须是 JSON 对象')
        rows=payload.get('events',[])
        if not isinstance(rows,list) or not 1<=len(rows)<=10000:raise ValueError('events: 1–10000 rows / 事件行数须为 1–10000')
        vocab=self.vocab['token_vocabulary'];outcome={f'event:{n}':i for i,n in enumerate(self.vocab['outcome_vocabulary'])}
        records=[dict(token='<BOS>',time_min=0,value=None)]
        for r in rows:
            if r.get('token') not in vocab or r['token'] in ['<PAD>','<MASK>','<EPISODE_END>']:raise ValueError(f"Unsupported input token / 不支持的事件: {r.get('token')}")
            t=float(r['time_min']);v=r.get('value')
            if not math.isfinite(t) or not 0<=t<=43200:raise ValueError('time_min must be finite and within 0–43200 / 时间须为 0–43200 分钟')
            if v is not None and not math.isfinite(float(v)):raise ValueError('value must be finite / 数值须有限')
            records.append(dict(token=r['token'],time_min=t,value=None if v is None else float(v)))
        records=sorted(records,key=lambda r:r['time_min'])
        s=payload.get('static',{});ranges={'age_at_operation':(0,120),'male':(0,1),'asa':(0,6),'emergency':(0,1),'weight_kg':(0,300),'height_cm':(0,250),'bmi':(0,80)}
        for k,(lo,hi) in ranges.items():
            v=float(s.get(k,0) or 0)
            if not math.isfinite(v) or not lo<=v<=hi:raise ValueError(f'{k}: range {lo}–{hi}')
        weight=float(s.get('weight_kg',0) or 0);height=float(s.get('height_cm',0) or 0)
        bmi=float(s.get('bmi',0) or (weight/(height/100)**2 if height>0 else 0));bmi=min(80,bmi)
        static=[float(s.get('age_at_operation',0) or 0)/100,float(s.get('male',0) or 0),bmi/40,float(s.get('asa',0) or 0)/6,float(s.get('emergency',0) or 0),weight/150,height/200]
        phase_map={'event:or_entry':1,'event:anesthesia_start':2,'event:surgery_start':3,'event:surgery_end':4,'event:or_exit':5,'event:icu_transfer':6,'event:icu_discharge':7}
        phase=0;last={};left=0;phases=[];observations=[];fams=self.vocab['outcome_to_family'];count=np.zeros(63,dtype=np.float32)
        start=max(0,len(records)-256)
        for i,r in enumerate(records):
            phase=phase_map.get(r['token'],phase);phases.append(phase)
            while r['time_min']-records[left]['time_min']>360:left+=1
            obs=[math.log1p(r['time_min']-records[left]['time_min'])/math.log1p(360),math.log1p(i-left)/math.log1p(128),0.]
            oi=outcome.get(r['token'])
            if oi is not None:
                fam=fams[oi]
                if fam in last:obs[2]=math.log1p(r['time_min']-last[fam])/math.log1p(43200)
                last[fam]=r['time_min']
                if i<start:count[fam]+=1
            observations.append(obs)
        selected=records[start:];times=[r['time_min'] for r in selected]
        def kind(token):
            if token=='<CLOCK>':return 7
            if token.startswith('event:'):return 6
            if token.startswith(('med:','med_signal:')):return 4
            if token.startswith('diagnosis:'):return 5
            if token.startswith(('procedure:','phase_summary:','context:department:','context:asa:','context:antype:')):return 3
            if token.startswith(('context:','summary:')):return 2
            return 1
        batch={'token_id':[vocab[r['token']] for r in selected],'token_kind':[kind(r['token']) for r in selected],
            'time_min':times,'gap_min':[0]+[times[i]-times[i-1] for i in range(1,len(times))],
            'value':[r['value'] or 0 for r in selected],'has_value':[float(r['value'] is not None) for r in selected],
            'phase_id':phases[start:],'observation_features':observations[start:],'attention_mask':[True]*len(selected),
            'history_family_counts':np.log1p(count).tolist(),'static':static}
        integer={'token_id','token_kind','phase_id'}
        return {k:torch.tensor(v,dtype=torch.long if k in integer else torch.bool if k=='attention_mask' else torch.float32,device=device or self.device).unsqueeze(0) for k,v in batch.items()}

    @torch.inference_mode()
    def predict(self,payload,calibration='None',custom=None,top_k=10):
        batch=self.encode(payload);output=self.model(batch)
        logits=output.logits[0,-1].float();raw=torch.softmax(logits,-1)
        fit={'temperature':1.,'bias':[0.]*len(logits)}
        if calibration=='Custom':fit=json.loads(custom) if isinstance(custom,str) else custom
        elif calibration!='None':fit=self.calibrations[calibration]
        t=float(fit['temperature']);bias=torch.as_tensor(fit.get('bias',[0.]*len(logits)),device=self.device)
        if not math.isfinite(t) or t<=0 or bias.shape!=logits.shape or not torch.isfinite(bias).all():raise ValueError('Invalid temperature or 210-element bias / 温度或 210 项偏置无效')
        corrected=torch.softmax((logits+bias)/t,-1)
        wait=dual_timescale_expected_wait(output.fine_hazard_logits[0,-1],output.long_hazard_logits[0,-1],output.tail_mu[0,-1],output.tail_log_sigma[0,-1])
        horizon=torch.sigmoid(output.trajectory_logits[0,-1]);names=self.vocab['outcome_vocabulary'];families=self.vocab['outcome_family_vocabulary'];fids=self.vocab['outcome_to_family']
        ids=torch.argsort(corrected,descending=True)[:int(top_k)].tolist()
        result=[]
        for rank,i in enumerate(ids,1):result.append({'rank':rank,'event':names[i],'family':families[fids[i]],'raw':float(raw[i]),'calibrated':float(corrected[i]),'wait_hours':float(wait[i]),'risk_1h_raw':float(horizon[0,i]),'risk_6h_raw':float(horizon[1,i]),'risk_24h_raw':float(horizon[2,i])})
        return {'calibration':calibration,'temperature':t,'input_events':len(batch['token_id'][0]),'next_event':result,'event_logits':logits.cpu().tolist(),'query_time_min':float(batch['time_min'][0,-1])}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--calibration',default='None');p.add_argument('--model-dir');a=p.parse_args()
    print(json.dumps(Predictor(a.model_dir).predict(Path(a.input).read_text(),a.calibration),indent=2))
