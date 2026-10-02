"""Fit with calibration data only. Never pass test labels."""
import argparse,json
import numpy as np
from calibration import fit_event_calibration
p=argparse.ArgumentParser();p.add_argument('--logits',required=True);p.add_argument('--labels',required=True);p.add_argument('--groups',required=True);p.add_argument('--output',default='custom_calibration.json');p.add_argument('--device',default='cpu');a=p.parse_args()
z=np.load(a.logits,mmap_mode='r');y=np.load(a.labels,mmap_mode='r')
if z.shape!=y.shape or z.ndim!=2 or z.shape[1]!=210:raise ValueError('Expected logits and labels with shape [N,210] in vocabulary order')
f=fit_event_calibration(a.logits,a.labels,np.load(a.groups),a.device)
with open(a.output,'w') as out:json.dump(f,out,indent=2)
