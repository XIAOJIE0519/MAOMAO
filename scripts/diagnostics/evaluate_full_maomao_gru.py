#!/usr/bin/env python3
"""Serially score the full-cohort MAOMAO and GRU internal checkpoints."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/final_experiment_results_20260923"
DATA = ROOT / "data/perioperative_event_sequences_v5_richctx_static7"
MODELS = (
    ("maomao_internal", "maomao", OUT / "full_maomao_reference/best_model.pt"),
    ("gru_internal", "gru", ROOT / "outputs/baseline_gru_richctx/gru_best.pt"),
)
REQUIRED = (
    "micro_auprc", "macro_auprc", "micro_auroc", "macro_auroc", "mrr",
    "brier", "ece", "hit_at_1", "recall_at_5", "recall_at_10",
)


def valid_full_result(path: Path) -> bool:
    try:
        data = json.loads(path.read_text())
        return (data.get("evaluation_rows_full_patient_validation") is True
                and data.get("evaluation_rows") == 1_563_972
                and all(f"{metric}_95ci" in data for metric in REQUIRED))
    except (OSError, json.JSONDecodeError):
        return False


def main() -> None:
    status_path = OUT / "logs/full_model_evaluation_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    statuses = {}
    for name, kind, checkpoint in MODELS:
        metrics = OUT / f"model_metrics/{name}.json"
        log = OUT / f"logs/{name}_evaluation.log"
        if valid_full_result(metrics):
            statuses[name] = {"status": "evaluated", "metrics": str(metrics), "reused_existing": True}
            status_path.write_text(json.dumps({
                "updated_utc": datetime.now(timezone.utc).isoformat(), "statuses": statuses,
            }, ensure_ascii=False, indent=2))
            continue
        if not checkpoint.exists():
            statuses[name] = {"status": "checkpoint_missing", "checkpoint": str(checkpoint)}
        else:
            cmd = [sys.executable, str(ROOT / "scripts/diagnostics/evaluate_internal_checkpoints.py"),
                   "--kind", kind, "--checkpoint", str(checkpoint), "--data_dir", str(DATA),
                   "--output", str(metrics), "--model_name", name, "--batch_size", "64",
                   "--bootstrap_repeats", "200", "--full_validation"]
            with log.open("w") as handle:
                proc = subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            statuses[name] = {"status": "evaluated" if proc.returncode == 0 else "evaluation_failed",
                              "returncode": proc.returncode, "metrics": str(metrics),
                              "checkpoint": str(checkpoint)}
        status_path.write_text(json.dumps({
            "updated_utc": datetime.now(timezone.utc).isoformat(), "statuses": statuses,
        }, ensure_ascii=False, indent=2))
        print(json.dumps({name: statuses[name]}, ensure_ascii=False), flush=True)
        if statuses[name]["status"] != "evaluated":
            raise RuntimeError(f"{name} did not finish successfully: {statuses[name]}")
    subprocess.run([sys.executable, str(ROOT / "scripts/diagnostics/build_requested_results_report.py")],
                   cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
