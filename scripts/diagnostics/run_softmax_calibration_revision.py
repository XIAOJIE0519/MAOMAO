#!/usr/bin/env python3
"""One locked serial revision: frozen models, unchanged external sealed rows.

All model/site items are completed and checked before updating active results.
Old calibration artifacts remain in a separate audit archive, outside delivery.
"""
import sys
import json
import gc
import fcntl
import shutil
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.diagnostics import score_uniform_external_five as original
from scripts.diagnostics.uniform_result_scope import sha256, current_xgboost_model
from maomao.evaluation.softmax_calibration import fit_temperature, PROTOCOL
from maomao.evaluation.plot_sources import export_plot_sources, RETRIEVAL_TIE_PROTOCOL
from maomao.evaluation.event_metrics import event_metric_report
from maomao.evaluation.gpu_bootstrap import bootstrap_scalar_cis
from maomao.evaluation.rank_statistics import RANK_VERSION

REV = ROOT / "outputs/calibration_softmax_revision_20260929"
PLOTS = ROOT / "outputs/maomao_plot_sources"
OLD = ROOT / "outputs/calibration_audit_original_20260929"
MODELS = ("maomao", "univariate", "logistic_regression", "xgboost", "ann")
SITES = ("ntuh", "asac", "uq", "surgical_pooled", "mimic", "sicdb", "mover", "eicu")

def plot_ready(site,name,state):
    p=PLOTS/f"external/{site}/{name}/{state}/provenance.json"
    return p.exists() and json.loads(p.read_text()).get("retrieval_tie_protocol")==RETRIEVAL_TIE_PROTOCOL

def event_metric_report_with_subsample_ci(logits,targets,outcomes,bootstrap_repeats=200,bootstrap_seed=42,max_ci_rows=100000):
    """Same full CPU point metrics and seeded CI row draws; GPU CI arithmetic."""
    full=event_metric_report(logits,targets,outcomes)
    n=len(logits)
    if n<=max_ci_rows:
        idx=torch.arange(n);seed=bootstrap_seed
        center=full
    else:
        idx=torch.randperm(n,generator=torch.Generator(device="cpu").manual_seed(bootstrap_seed))[:max_ci_rows]
        seed=bootstrap_seed+1
        center=event_metric_report(logits[idx],targets[idx],outcomes)
    cis=bootstrap_scalar_cis(logits[idx].to(original.DEVICE),targets[idx].to(original.DEVICE),repeats=bootstrap_repeats,seed=seed)
    scale=(len(idx)/n)**.5
    for metric,ci in cis.items():
        if ci is not None and full.get(metric) is not None and center.get(metric) is not None:
            # Squared distance between two categorical distributions ranges
            # from 0 to 2; unlike Hit/AUC/ECE, multiclass Brier is not bounded by 1.
            upper_bound=2. if metric=='brier' else 1.
            full[metric+"_95ci"]=[max(0.,min(upper_bound,full[metric]+(bound-center[metric])*scale)) for bound in ci] if n>max_ci_rows else ci
    full.update(ci_method="row bootstrap on full test cohort" if n<=max_ci_rows else "uniform row subsample bootstrap; interval deviations scaled by sqrt(sample_n/cohort_n)",
                ci_rows=len(idx),ci_cohort_rows=n,ci_bootstrap_repeats=bootstrap_repeats,
                ci_arithmetic_device=str(original.DEVICE),ci_random_row_draws="original CPU generator; identical seed and indices")
    torch.cuda.empty_cache()
    return full


def model_source(name):
    return {"maomao":original.INTERNAL/"full_maomao_reference/best_model.pt",
            "univariate":original.INTERNAL/"baseline_metrics/common_full_validation/univariate_last_token_model.npz",
            "logistic_regression":original.INTERNAL/"baseline_metrics/logistic_fullscale.pt",
            "ann":original.INTERNAL/"baseline_metrics/common_full_validation/ann_fullscale_refined.pt",
            "xgboost":current_xgboost_model()}[name]


def scores(site,name,split,dest,manifest,hashes):
    sidecar = dest/f"{split}_fingerprint.json"
    fingerprint = dict(model_sha256=hashes[name], manifest_sha256=sha256(original.BASE/site/"manifest.json"),
                       protocol=PROTOCOL, split=split, precision="float32",
                       source_dataset=manifest["dataset"], batch_size=32)
    path = dest/f"{split}_logits.npy"
    if path.exists():
        if not sidecar.exists() or json.loads(sidecar.read_text()) != fingerprint:
            raise RuntimeError(f"Stale logits cache {path}")
        saved = np.load(path,mmap_mode="r")
        if saved.shape != (manifest[f'{split}_target_rows'],210):
            raise RuntimeError("Wrong cache shape")
        return path
    partial = dest/f"{split}_logits.partial.npy"
    output = np.lib.format.open_memmap(partial,mode="w+",dtype=np.float32,
                                      shape=(manifest[f'{split}_target_rows'],210))
    if name == "maomao":
        original.predict_maomao(site,original.BASE/site,split,output,32)
    else:
        original.predict_flat(name,original.BASE/site,split,output)
    output.flush()
    del output
    partial.replace(path)
    original.save_json(sidecar,fingerprint)
    return path


def evaluate(site,name,outcomes,hashes):
    dest=REV/site/name
    dest.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((original.BASE/site/"manifest.json").read_text())
    old=json.loads((OLD/site/name/"metrics.json").read_text())
    status=dest/"status.json"
    metrics_ready=(dest/"metrics.json").exists() and json.loads((dest/"metrics.json").read_text()).get("rank_metric_definition")==RANK_VERSION
    if metrics_ready and status.exists() and json.loads(status.read_text()).get("status")=="completed" and plot_ready(site,name,"after") and (name!="maomao" or plot_ready(site,name,"before")):
        return
    original.check_contract(ROOT/original.SOURCES[site],original.TRAINING)
    if manifest["split"]["patient_overlap"] != 0:
        raise RuntimeError("Patient overlap")
    original.save_json(status,dict(status="scoring_calibration",site=site,model=name,updated_utc=original.stamp()))
    cal=None
    fitpath=dest/"calibration_fit.json"
    if fitpath.exists():
        fit=json.loads(fitpath.read_text())
        if fit["protocol"]!=PROTOCOL or fit["model_sha256"]!=hashes[name]:
            raise RuntimeError("Stale calibration fit")
    else:
        cal=scores(site,name,"calibration",dest,manifest,hashes)
        original.save_json(status,dict(status="fitting_full_90_percent_softmax_temperature",site=site,model=name,updated_utc=original.stamp()))
        fit=fit_temperature(cal,original.BASE/site/"calibration_y.npy",original.DEVICE)
        fit.update(site=site,model=name,model_sha256=hashes[name],created_utc=original.stamp())
        original.save_json(fitpath,fit)
    # Fit is sealed before any new test prediction or test label scoring.
    original.save_json(status,dict(status="scoring_sealed_10_percent",site=site,model=name,temperature=fit["temperature"],updated_utc=original.stamp()))
    test=scores(site,name,"test",dest,manifest,hashes)
    ypath=original.BASE/site/"test_y.npy"
    logits=torch.from_numpy(np.array(np.load(test,mmap_mode="r"),dtype=np.float32))
    targets=torch.from_numpy(np.array(np.load(ypath,mmap_mode="r"),dtype=np.uint8))
    report=json.loads((dest/"metrics.json").read_text()) if metrics_ready else event_metric_report_with_subsample_ci(logits/fit["temperature"],targets,outcomes,
                         bootstrap_repeats=200,bootstrap_seed=42,max_ci_rows=100000)
    for key in ("site","model","split","source_dataset","patients_calibration","patients_test",
                "patient_overlap","calibration_rows","test_rows","train_rows_internal","calibration_scope","test_scope"):
        report[key]=old[key]
    report.update(status="completed",temperature=fit["temperature"],model_sha256=hashes[name],
                  calibration_protocol=PROTOCOL,calibration_state="after",
                  metric_input="softmax of logits / temperature fitted and selected exclusively on external 90%",
                  calibration_precision="float32",created_utc=original.stamp())
    original.save_json(dest/"metrics.json",report)
    plot=export_plot_sources(test,ypath,PLOTS/f"external/{site}/{name}/after",outcomes,
                            fit["temperature"],dict(site=site,model=name,calibration_state="after",protocol=PROTOCOL,model_sha256=hashes[name]))
    if abs(plot["brier_normalized_target"]-report["brier"]) > 5e-6:
        raise RuntimeError("Plot Brier verification failed")
    if name=="maomao":
        # Current raw point results must reproduce. No model weights changed.
        rawpath=dest/"metrics_uncalibrated.json"
        raw_ready=rawpath.exists() and json.loads(rawpath.read_text()).get("rank_metric_definition")==RANK_VERSION
        raw=json.loads(rawpath.read_text()) if raw_ready else event_metric_report_with_subsample_ci(logits,targets,outcomes,bootstrap_repeats=200,
                                                bootstrap_seed=42,max_ci_rows=100000)
        previous=json.loads((OLD/site/name/"metrics_uncalibrated.json").read_text())
        for key in ("brier","ece","hit_at_1","mrr"):
            if abs(raw[key]-previous[key])>5e-6:
                raise RuntimeError(f"{site}: raw {key} changed")
        for key in ("site","model","split","source_dataset","patients_calibration","patients_test",
                    "patient_overlap","calibration_rows","test_rows","train_rows_internal","calibration_scope","test_scope"):
            raw[key]=report[key]
        raw.update(status="completed",calibration_state="before",temperature=1.,fitted_temperature=fit["temperature"],
                   model_sha256=hashes[name],calibrated_reference_sha256=sha256(dest/"metrics.json"),
                   same_test_targets_verified=True,calibrated_point_metrics_reproduced=True,
                   calibration_protocol=PROTOCOL,created_utc=original.stamp())
        original.save_json(dest/"metrics_uncalibrated.json",raw)
        original.save_json(dest/"uncalibrated_status.json",dict(status="completed",test_rows=raw["test_rows"]))
        export_plot_sources(test,ypath,PLOTS/f"external/{site}/{name}/before",outcomes,1.,
                            dict(site=site,model=name,calibration_state="before",protocol=PROTOCOL,model_sha256=hashes[name]))
        if raw["hit_at_1"] != report["hit_at_1"] or abs(raw["mrr"]-report["mrr"])>1e-7:
            raise RuntimeError("Temperature changed within-row ranking")
    del logits,targets
    gc.collect()
    if cal is not None:cal.unlink(missing_ok=True)
    # Scores from an interrupted pre-version evaluation may have been kept.
    (dest/"calibration_logits.npy").unlink(missing_ok=True)
    test.unlink(missing_ok=True)
    original.save_json(status,dict(status="completed",site=site,model=name,calibration_rows=manifest["calibration_target_rows"],
                                  test_rows=manifest["test_target_rows"],protocol=PROTOCOL,finished_utc=original.stamp()))
    print(f"COMPLETED {site}/{name}: T={fit['temperature']:.6g} Brier={report['brier']:.6g}",flush=True)


def main():
    torch.set_num_threads(8)
    REV.mkdir(parents=True,exist_ok=True)
    with (REV/"queue.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not OLD.exists():
            shutil.copytree(original.RESULTS,OLD)
            shutil.copy2(ROOT/"outputs/MAOMAO_final_results_20260926.zip",OLD/"original_delivery.zip")
        outcomes=json.loads((original.TRAINING/"event_sequence_meta.json").read_text())["outcome_vocabulary"]
        hashes={name:sha256(model_source(name)) for name in MODELS}
        original.save_json(REV/"prespecified_protocol.json",dict(protocol=PROTOCOL,model_hashes=hashes,
                 fit="full 90% normalized-target softmax cross entropy",selection="full 90% Brier improvement otherwise identity",
                 test_use="sealed 10% evaluation only",sources=list(SITES),models=list(MODELS),training_repeated=False))
        completed=[]
        # MAOMAO lane first, then remaining frozen baseline calibrators.
        for name in MODELS:
            for site in SITES:
                original.save_json(REV/"queue_status.json",dict(status="running",active=f"{site}/{name}",completed=completed,updated_utc=original.stamp()))
                evaluate(site,name,outcomes,hashes)
                completed.append(f"{site}/{name}")
        for name in MODELS:
            if hashes[name]!=sha256(model_source(name)):
                raise RuntimeError("Model changed during revision")
        for site in SITES:
            for name in MODELS:
                for file in ("metrics.json","status.json","calibration_fit.json","metrics_uncalibrated.json","uncalibrated_status.json"):
                    source=REV/site/name/file
                    if source.exists():
                        shutil.copy2(source,original.RESULTS/site/name/file)
            summaries={name:{key:json.loads((REV/site/name/"metrics.json").read_text())[key] for key in ("micro_auprc","micro_auroc","test_rows")} for name in MODELS}
            original.save_json(original.RESULTS/site/"summary.json",dict(site=site,status="completed",models=summaries,protocol=PROTOCOL))
        original.save_json(REV/"queue_status.json",dict(status="completed",completed=completed,promoted=True,finished_utc=original.stamp()))
        print("All 40 full-row calibration results promoted; originals retained separately for audit.",flush=True)


if __name__=="__main__":
    main()
