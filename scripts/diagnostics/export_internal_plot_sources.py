#!/usr/bin/env python3
"""Rescore frozen checkpoints once to export full-cohort figure source summaries.

Training is unchanged; --statistics_revision refreshes metric definitions.
Time source vectors contain
only actual/predicted waits and errors, with no patient or episode identifiers.
"""
import sys
import json
import gc
import shutil
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from maomao.data.event_sequence import EventSequenceDataset,collate_event_sequences
from maomao.data.scale_ablation import ProjectedOutcomeDataset
from maomao.models.event_maomao import dual_timescale_expected_wait
from maomao.evaluation.plot_sources import export_plot_sources, RETRIEVAL_TIE_PROTOCOL
from scripts.diagnostics.scale_ablation_scope import model_for,NAMES,read
from scripts.diagnostics import score_uniform_external_five as original
from scripts.diagnostics.uniform_result_scope import sha256
from scripts.diagnostics.run_softmax_calibration_revision import event_metric_report_with_subsample_ci
from maomao.evaluation.rank_statistics import RANK_VERSION

BASE=original.INTERNAL
ROWS=BASE/"classical_full_scale"
SCALE=ROOT/"outputs/scale_ablations_richctx_20260928"
OUT=ROOT/"outputs/maomao_plot_sources"
CACHE=OUT/".temporary_internal"
MODULES=("no_block_causal","no_relative_time","no_family_head","no_clock_phase_summary",
         "no_measurement_intensity","no_masked_event_value","no_event_conditioned_time")
STATISTICS_ONLY="--statistics_revision" in sys.argv

def plot_ready(folder):
    p=folder/"provenance.json"
    return p.exists() and read(p).get("retrieval_tie_protocol")==RETRIEVAL_TIE_PROTOCOL

def metric_path(group,name):
    if group=="module_ablations":return BASE/f"module_metrics/{name}.json"
    if group=="scale_ablations":return SCALE/f"metrics/{name}.json"
    from scripts.diagnostics.uniform_result_scope import current_xgboost_dir
    return {"univariate":BASE/"baseline_metrics/univariate_fullscale_internal.json",
            "logistic_regression":BASE/"baseline_metrics/logistic_fullscale_internal.json",
            "xgboost":current_xgboost_dir()/"metrics.json",
            "ann":BASE/"baseline_metrics/common_full_validation/ann_fullscale_refined_metrics.json"}[name]


def update_metrics(group,name,logits,target,names,columns):
    path=metric_path(group,name);previous=read(path)
    if previous.get("rank_metric_definition")==RANK_VERSION:return
    y=np.asarray(target[:,columns],dtype=np.uint8)
    eligible=y.any(1)
    x=np.array(logits[eligible],dtype=np.float32)
    report=event_metric_report_with_subsample_ci(torch.from_numpy(x),torch.from_numpy(y[eligible]),names,
                 bootstrap_repeats=200,bootstrap_seed=4200 if group!="internal" or name=="maomao" else 42,
                 max_ci_rows=30000 if group!="internal" or name=="maomao" else 100000)
    if report["event_targets"]!=previous["event_targets"]:raise RuntimeError("Statistics revision changed row eligibility")
    for key in ("brier","ece","mrr","hit_at_1","recall_at_5","recall_at_10"):
        if abs(report[key]-previous[key])>5e-6:raise RuntimeError(f"Fixed prediction metric changed: {name}/{key}")
    previous.update(report)
    path.write_text(json.dumps(previous,ensure_ascii=False,indent=2)+"\n")
    print(f"STANDARD STATISTICS COMPLETED {group}/{name}",flush=True)


def time_source(destination,actual,predicted,checkpoint_hash):
    destination.mkdir(parents=True,exist_ok=True)
    error=np.abs(predicted-actual)
    np.savez_compressed(destination/"time_predictions_no_identifiers.npz",actual_wait_hours=actual,
                        predicted_positive_set_average_hours=predicted,absolute_error_hours=error)
    result={"rows":len(actual),"checkpoint_sha256":checkpoint_hash,"no_identifiers":True,
            "predicted_time_definition":"mean predicted waiting time over observed positive event set",
            "all_eligible_target_rows":True,"pooled_mae_reported":False,"strata":{}}
    for label,mask in (("less_than_2h",actual<2),("2_to_less_than_24h",(actual>=2)&(actual<24)),("at_least_24h",actual>=24)):
        a=error[mask]
        counts,edges=np.histogram(a,bins=np.r_[np.linspace(0,24,241),np.linspace(24.1,240,2160),np.linspace(241,2000,1760),np.inf])
        np.savez_compressed(destination/f"time_error_{label}_histogram.npz",edges_hours=edges,rows=counts)
        result["strata"][label]={"rows":len(a),"mae_hours":float(a.mean()) if len(a) else None,
                "quantile_probabilities":[0,.05,.25,.5,.75,.9,.95,.99,1],
                "absolute_error_quantiles_hours":np.quantile(a,[0,.05,.25,.5,.75,.9,.95,.99,1]).tolist() if len(a) else []}
    (destination/"time_provenance.json").write_text(json.dumps(result,indent=2)+"\n")


def projected_export(logits,target,indices,columns,names,destination,meta,actual,predicted):
    y=np.asarray(target[:,indices],dtype=np.uint8)
    eligible=y.any(1)
    ypath=CACHE/"selected_y.npy";zpath=CACHE/"selected_logits.npy"
    np.save(ypath,y[eligible]);np.save(zpath,np.asarray(logits[eligible][:,columns],dtype=np.float32))
    export_plot_sources(zpath,ypath,destination,[names[i] for i in indices],1.,meta)
    if actual is not None:
        time_source(destination,np.asarray(actual)[eligible],np.asarray(predicted)[eligible],meta["checkpoint_sha256"])
    ypath.unlink();zpath.unlink()


def neural(name,checkpoint,group):
    destination=OUT/group/name
    digest=sha256(checkpoint)
    done=destination/"complete.json"
    if done.exists() and read(done).get("checkpoint_sha256")==digest and not STATISTICS_ONLY:return
    if STATISTICS_ONLY and plot_ready(destination) and read(metric_path(group,name)).get("rank_metric_definition")==RANK_VERSION:
        if name!="reference" or all(plot_ready(OUT/f"scale_ablations/reference_vocab_{n}") and read(metric_path(group,f"reference_vocab_{n}")).get("rank_metric_definition")==RANK_VERSION for n in (50,100,150)):return
    base=EventSequenceDataset(original.TRAINING,256,128,dynamic_windows=False)
    names=base.meta["outcome_vocabulary"]
    payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
    checkpoint_epoch=int(payload["epoch"])
    cfg=payload["args"];indices=list(range(210));ds=base
    if cfg.get("outcome_projection_file"):
        indices=read(cfg["outcome_projection_file"])["indices"]
        ds=ProjectedOutcomeDataset(base,cfg["outcome_projection_file"])
    windows=np.load(ROWS/"validation_window_indices.npy",mmap_mode="r")
    positions=np.load(ROWS/"validation_positions.npy",mmap_mode="r")
    targets=np.load(ROWS/"validation_y.npy",mmap_mode="r")
    batch_size=64 if name in NAMES else 32
    unique=np.load(BASE/"full_maomao_reference/patient_validation_split.npz")["validation_window_indices"] if name in NAMES else np.unique(windows)
    path=CACHE/"neural_logits.npy"
    logits=np.lib.format.open_memmap(path,mode="w+",dtype=np.float32,shape=(len(targets),len(indices)))
    actual=np.empty(len(targets),dtype=np.float32);predicted=np.empty_like(actual)
    reference_sets={n:read(SCALE/f"specifications/vocab_{n}.json")["indices"] for n in (50,100,150)} if name=="reference" else {}
    reference_times={n:np.empty(len(targets),dtype=np.float32) for n in reference_sets}
    model=model_for(ds,payload,original.DEVICE);del payload
    loader=DataLoader(Subset(ds,unique),batch_size=batch_size,shuffle=False,num_workers=0,collate_fn=collate_event_sequences)
    time_available=True
    with torch.inference_mode():
        for batch_no,batch in enumerate(loader):
            rows=[];local=[]
            for b,window in enumerate(unique[batch_no*batch_size:(batch_no+1)*batch_size]):
                lo=np.searchsorted(windows,window,side="left");hi=np.searchsorted(windows,window,side="right")
                rows.extend(range(lo,hi));local.extend([b]*(hi-lo))
            rows=np.asarray(rows,dtype=np.int64);pos=np.asarray(positions[rows],dtype=np.int64)
            target=batch["target_set"][local,pos].numpy().astype(np.uint8)
            if not np.array_equal(target,np.asarray(targets[rows][:,indices],dtype=np.uint8)):
                raise RuntimeError("Plot target coordinate mismatch")
            actual[rows]=batch["target_dt_hours"][local,pos].numpy()
            batch={k:v.to(original.DEVICE) if torch.is_tensor(v) else v for k,v in batch.items()}
            with torch.autocast(original.DEVICE.type,dtype=torch.bfloat16,enabled=original.DEVICE.type=="cuda"):
                result=model(batch)
            li=torch.tensor(local,device=original.DEVICE);pi=torch.tensor(pos,device=original.DEVICE)
            logits[rows]=result.logits[li,pi].float().cpu().numpy()
            if result.fine_hazard_logits is not None:
                wait=dual_timescale_expected_wait(result.fine_hazard_logits[li,pi],result.long_hazard_logits[li,pi],result.tail_mu[li,pi],result.tail_log_sigma[li,pi]).cpu().numpy()
                predicted[rows]=(wait*target).sum(1)/np.maximum(1,target.sum(1))
                for n,selected in reference_sets.items():
                    ref_target=target[:,selected]
                    reference_times[n][rows]=(wait[:,selected]*ref_target).sum(1)/np.maximum(1,ref_target.sum(1))
            else:time_available=False
            if (batch_no+1)%100==0 or batch_no+1==len(loader):print(f"plot {group}/{name}: {batch_no+1}/{len(loader)} batches",flush=True)
    logits.flush()
    meta=dict(group=group,model=name,checkpoint_sha256=digest,checkpoint_epoch=checkpoint_epoch,
              temperature=1.,split="internal full patient-disjoint 10% validation",output_indices=indices,
              inference_batch_size=batch_size,inference_window_count=len(unique),inference_precision="bf16")
    if STATISTICS_ONLY:
        update_metrics(group,name,logits,targets,[names[i] for i in indices],indices)
        if name=="reference":
            report=read(metric_path(group,name))
            primary=BASE/"model_metrics/maomao_internal.json";old=read(primary)
            for key in ("event_targets","per_event","frequency_strata","rank_metric_definition","ci_method","ci_rows","ci_cohort_rows","ci_bootstrap_repeats","ci_arithmetic_device","ci_random_row_draws"):
                if key in report:old[key]=report[key]
            for key in ("micro_auprc","macro_auprc","micro_auroc","macro_auroc","mrr","brier","ece","hit_at_1","recall_at_5","recall_at_10"):
                old[key]=report[key];old[key+"_95ci"]=report[key+"_95ci"]
            primary.write_text(json.dumps(old,ensure_ascii=False,indent=2)+"\n")
    if not plot_ready(destination):
        projected_export(logits,targets,indices,list(range(len(indices))),names,destination,meta,
                         actual if time_available else None,predicted if time_available else None)
    if name=="reference":
        for n in (50,100,150):
            selected=read(SCALE/f"specifications/vocab_{n}.json")["indices"]
            if STATISTICS_ONLY:
                update_metrics(group,f"reference_vocab_{n}",np.asarray(logits[:,selected]),targets,[names[i] for i in selected],selected)
            if not plot_ready(OUT/f"scale_ablations/reference_vocab_{n}"):
                projected_export(logits,targets,selected,selected,names,OUT/f"scale_ablations/reference_vocab_{n}",
                                 dict(meta,model=f"reference_vocab_{n}",output_indices=selected),actual,reference_times[n])
    (destination/"complete.json").write_text(json.dumps(dict(checkpoint_sha256=digest,all_rows=True,rows=len(targets)))+"\n")
    del model,ds,base,loader,logits,actual,predicted
    path.unlink();gc.collect();torch.cuda.empty_cache()


def main():
    torch.set_num_threads(8);CACHE.mkdir(parents=True,exist_ok=True)
    original.RESULTS=OUT/".unused"
    names=read(original.TRAINING/"event_sequence_meta.json")["outcome_vocabulary"]
    for name in ("univariate","logistic_regression","xgboost","ann"):
        destination=OUT/"internal"/name
        if (destination/"provenance.json").exists() and not STATISTICS_ONLY:continue
        if STATISTICS_ONLY and plot_ready(destination) and read(metric_path("internal",name)).get("rank_metric_definition")==RANK_VERSION:continue
        y=np.load(ROWS/"validation_y.npy",mmap_mode="r")
        path=CACHE/f"{name}.npy"
        out=np.lib.format.open_memmap(path,mode="w+",dtype=np.float32,shape=y.shape)
        original.predict_flat(name,ROWS,"validation",out);out.flush();del out
        if STATISTICS_ONLY:
            update_metrics("internal",name,np.load(path,mmap_mode="r"),y,names,list(range(210)))
        if not plot_ready(destination):
            export_plot_sources(path,ROWS/"validation_y.npy",destination,names,metadata=dict(model=name,group="internal",split="full patient-disjoint 10% validation"))
        path.unlink()
        print(f"COMPLETED full internal plot source {name}",flush=True)
    neural("reference",BASE/"full_maomao_reference/best_model.pt","scale_ablations")
    shutil.copytree(OUT/"scale_ablations/reference",OUT/"internal/maomao",dirs_exist_ok=True)
    for name in MODULES:neural(name,BASE/f"module_ablation_runs/{name}/best_model.pt","module_ablations")
    for name in NAMES:neural(name,SCALE/f"runs/{name}/best_model.pt","scale_ablations")
    shutil.rmtree(CACHE)
    if STATISTICS_ONLY:
        common_path=BASE/"baseline_metrics/common_full_validation/current_five_internal_metrics.json"
        common=read(common_path)
        for name,path in (("univariate",metric_path("internal","univariate")),("logistic_regression",metric_path("internal","logistic_regression")),("ann_fullscale_refined",metric_path("internal","ann")),("maomao",BASE/"model_metrics/maomao_internal.json")):
            common["models"][name].update(read(path))
        common_path.write_text(json.dumps(common,ensure_ascii=False,indent=2)+"\n")
    (OUT/"internal_export_status.json").write_text(json.dumps(dict(status="completed",internal_models=5,module_ablations=7,scale_states=11,full_rows=True))+"\n")


if __name__=="__main__":main()
