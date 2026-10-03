#!/usr/bin/env python3
"""Wait for existing workers, then verify, render and atomically package delivery."""
import sys
import os
import time
import json
import subprocess
import fcntl
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.diagnostics.uniform_result_scope import sha256
from scripts.diagnostics.uniform_result_scope import current_xgboost_dir
from maomao.evaluation.rank_statistics import RANK_VERSION
from maomao.evaluation.plot_sources import RETRIEVAL_TIE_PROTOCOL

REV=ROOT/"outputs/calibration_softmax_revision_20260929"
PLOTS=ROOT/"outputs/maomao_plot_sources"


def save(obj):
    path=REV/"delivery_status.json";temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n");temp.replace(path)


def run(script,*args):
    subprocess.run([sys.executable,str(ROOT/"scripts/diagnostics"/script),*args],cwd=ROOT,check=True)


def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False


def statistics_ready():
    base=ROOT/"outputs/final_experiment_results_20260923"
    paths=[base/"baseline_metrics/univariate_fullscale_internal.json",base/"baseline_metrics/logistic_fullscale_internal.json",
           current_xgboost_dir()/"metrics.json",base/"baseline_metrics/common_full_validation/ann_fullscale_refined_metrics.json",
           base/"model_metrics/maomao_internal.json",*list((base/"module_metrics").glob("no_*.json")),
           *[p for p in (ROOT/"outputs/scale_ablations_richctx_20260928/metrics").glob("*.json") if not p.stem.endswith("_time_scales")]]
    if len(paths)!=23 or any(json.loads(p.read_text()).get("rank_metric_definition")!=RANK_VERSION for p in paths):return False
    plot_proofs=[*list((PLOTS/"internal").glob("*/provenance.json")),*list((PLOTS/"module_ablations").glob("*/provenance.json")),*list((PLOTS/"scale_ablations").glob("*/provenance.json"))]
    if len(plot_proofs)!=23 or any(json.loads(p.read_text()).get("retrieval_tie_protocol")!=RETRIEVAL_TIE_PROTOCOL for p in plot_proofs):return False
    common=json.loads((base/"baseline_metrics/common_full_validation/current_five_internal_metrics.json").read_text())
    return all(m.get("rank_metric_definition")==RANK_VERSION for m in common["models"].values())


def main():
    pid_cal,pid_plots=map(int,sys.argv[1:3])
    with (REV/"delivery.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            c=json.loads((REV/"queue_status.json").read_text())
            p=json.loads((PLOTS/"internal_export_status.json").read_text()) if (PLOTS/"internal_export_status.json").exists() else {}
            try:ready=statistics_ready()
            except (FileNotFoundError,json.JSONDecodeError):ready=False
            if c.get("status")=="completed" and p.get("status")=="completed" and ready:break
            save(dict(status="waiting_for_existing_workers",calibration_completed=len(c.get("completed",[])),active=c.get("active"),plot_completed=p.get("status")=="completed",standard_statistics_completed=ready,updated_utc=datetime.now(timezone.utc).isoformat()))
            if c.get("status")!="completed" and not alive(pid_cal):raise RuntimeError("Calibration worker stopped; preserved cache requires audit before resuming")
            if (p.get("status")!="completed" or not ready) and not alive(pid_plots):raise RuntimeError("Internal statistics exporter stopped; audit its log")
            time.sleep(10)
        save(dict(status="building_and_verifying_full_delivery",updated_utc=datetime.now(timezone.utc).isoformat()))
        run("write_data_cleaning_report.py")
        run("verify_scale_ablations.py","--require-complete")
        run("build_scale_ablation_report.py")
        run("build_plot_tables_and_audit.py")
        plot_env=os.environ.copy()
        try:import matplotlib
        except ImportError:
            isolated=os.environ.get("MAOMAO_PLOT_PYTHONPATH","/tmp/lsj_plot_runtime_20260929")
            plot_env["PYTHONPATH"]=isolated+os.pathsep+plot_env.get("PYTHONPATH","")
        subprocess.run([sys.executable,str(PLOTS/"plot_examples.py"),"--output",str(PLOTS/"example_figures")],cwd=ROOT,env=plot_env,check=True)
        run("verify_plot_delivery.py")
        run("build_uniform_results_report.py")
        run("package_uniform_final_results.py")
        run("verify_uniform_final_results.py","--require-complete")
        scale=ROOT/"outputs/scale_ablations_richctx_20260928/delivery_verification.json"
        proof=json.loads(scale.read_text())
        proof.update(report_sha256=sha256(ROOT/"docs/MAOMAO_V5_FINAL_RESULTS.md"),
                     archive_sha256=sha256(ROOT/"outputs/MAOMAO_final_results_20260926.zip"),
                     verified_utc=datetime.now(timezone.utc).isoformat(),
                     later_calibration_and_plot_revision=sha256(PLOTS/"delivery_verification.json"))
        scale.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+"\n")
        archive=ROOT/"outputs/MAOMAO_final_results_20260926.zip"
        save(dict(status="completed",report=str(ROOT/"docs/MAOMAO_V5_FINAL_RESULTS.md"),archive=str(archive),
                  archive_bytes=archive.stat().st_size,archive_sha256=sha256(archive),
                  plot_curve_states=71,clinical_pairs=42,calibration_models_completed=40,
                  ablation_checkpoints_included=False,finished_utc=datetime.now(timezone.utc).isoformat()))
        print("Calibration audit, cleaning account, full-row plot sources, report and ZIP verified.",flush=True)


if __name__=="__main__":
    try:main()
    except Exception as error:
        save(dict(status="failed",error=str(error),updated_utc=datetime.now(timezone.utc).isoformat()))
        raise
