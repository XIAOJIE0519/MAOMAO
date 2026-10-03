"""Threshold-grouped AP and midrank-equivalent AUROC, with float64 counts."""
import torch

RANK_VERSION="threshold_grouped_ap_midrank_auroc_float64_v2"


def binary_rank_metrics(scores,target):
    target=target.bool()
    positive=int(target.sum());negative=len(target)-positive
    if positive==0:return None,None
    sorted_scores,order=torch.sort(scores,descending=True)
    truth=target[order]
    cumulative=truth.cumsum(0,dtype=torch.float64)
    end=torch.ones(len(scores),dtype=torch.bool,device=scores.device)
    end[:-1]=sorted_scores[:-1]!=sorted_scores[1:]
    index=torch.nonzero(end,as_tuple=False).flatten()
    cp=cumulative[index]
    previous=torch.cat((cp.new_zeros(1),cp[:-1]))
    gp=cp-previous
    size=(index+1).to(torch.float64)
    ap=float((gp*cp/size).sum()/positive)
    auc=None
    if negative:
        group_size=size-torch.cat((size.new_zeros(1),size[:-1]))
        gn=group_size-gp
        below=negative-(size-cp)
        auc=float((gp*(below+.5*gn)).sum()/(positive*negative))
    return ap,auc


def batched_rank_metrics(scores,target):
    """Pair AP/AUC across the last axis; ties count as one score threshold."""
    n=scores.shape[-1]
    values,order=torch.sort(scores,dim=-1,descending=True)
    truth=target.gather(-1,order).bool()
    cp=truth.cumsum(-1,dtype=torch.float64)
    pos=truth.sum(-1).to(torch.float64);neg=n-pos
    end=torch.ones_like(truth)
    end[...,:-1]=values[...,:-1]!=values[...,1:]
    position=torch.arange(1,n+1,device=scores.device,dtype=torch.int64)
    last=torch.where(end,position,0).cummax(-1).values
    previous=torch.cat((torch.zeros_like(last[...,:1]),last[...,:-1]),dim=-1)
    previous_cp=cp.gather(-1,(previous-1).clamp_min(0)).masked_fill(previous==0,0.)
    gp=(cp-previous_cp).masked_fill(~end,0.)
    ap=(gp*cp/position).sum(-1)/pos.clamp_min(1)
    ap=ap.masked_fill(pos==0,float("nan"))
    group_neg=position-previous-gp
    auc=(gp*(neg[...,None]-(position-cp)+.5*group_neg)).sum(-1)/(pos*neg).clamp_min(1)
    auc=auc.masked_fill((pos==0)|(neg==0),float("nan"))
    return ap,auc
