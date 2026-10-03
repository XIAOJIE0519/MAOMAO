#!/usr/bin/env python3
"""Evaluate the seven requested full-validation MAOMAO module ablations."""
from __future__ import annotations

import concurrent.futures
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "outputs/module_ablations_richctx_20260923"
RESULT_DIR = ROOT / "outputs/final_experiment_results_20260923"
DATA_DIR = ROOT / "data/perioperative_event_sequences_v5_richctx_static7"
MODELS = (
    "no_block_causal", "no_relative_time", "no_family_head",
    "no_clock_phase_summary", "no_measurement_intensity",
    "no_masked_event_value", "no_event_conditioned_time",
)


def evaluate(name: str) -> dict[str, object]:
    checkpoint = RUN_DIR / name / "best_model.pt"
    train_log = RUN_DIR / name / "train.log"
    metrics = RESULT_DIR / "module_metrics" / f"{name}.json"
    log_path = RESULT_DIR / "logs" / f"{name}_evaluation.log"
    if metrics.exists():
        try:
            saved = json.loads(metrics.read_text())
            if saved.get("evaluation_rows_full_patient_validation") is True and all(
                f"{metric}_95ci" in saved for metric in (
                    "micro_auprc", "macro_auprc", "micro_auroc", "macro_auroc",
                    "mrr", "brier", "ece", "hit_at_1", "recall_at_5", "recall_at_10",
                )
            ):
                return {"name": name, "status": "evaluated", "metrics": str(metrics),
                        "reused_existing": True}
        except (OSError, json.JSONDecodeError):
            pass
    if not train_log.exists() or "Training finished" not in train_log.read_text(errors="ignore"):
        return {"name": name, "status": "training_incomplete"}
    if not checkpoint.exists():
        return {"name": name, "status": "checkpoint_missing"}
    metrics.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(ROOT / "scripts/diagnostics/evaluate_internal_checkpoints.py"),
        "--kind", "maomao", "--checkpoint", str(checkpoint), "--data_dir", str(DATA_DIR),
        "--output", str(metrics), "--model_name", name, "--batch_size", "64",
        "--bootstrap_repeats", "200", "--full_validation",
    ]
    with log_path.open("w") as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    return {"name": name, "status": "evaluated" if result.returncode == 0 else "evaluation_failed",
            "returncode": result.returncode, "metrics": str(metrics)}


def main() -> None:
    statuses: list[dict[str, object]] = []
    status_path = RESULT_DIR / "module_completion.json"
    if status_path.exists():
        try:
            previous = json.loads(status_path.read_text())
            prior = {row.get("name"): row for row in previous.get("statuses", [])}
            for name in MODELS:
                existing = prior.get(name, {})
                metric_file = RESULT_DIR / "module_metrics" / f"{name}.json"
                if existing.get("status") == "evaluated" and metric_file.exists():
                    statuses.append(existing)
        except (OSError, json.JSONDecodeError):
            statuses = []
    # Run one full cohort evaluation at a time to avoid the observed global OOM
    # when four validation and bootstrap jobs execute concurrently.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        futures = {pool.submit(evaluate, name): name for name in MODELS}
        for future in concurrent.futures.as_completed(futures):
            statuses.append(future.result())
            statuses.sort(key=lambda row: MODELS.index(str(row["name"])))
            status_path.write_text(json.dumps({
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "evaluation_concurrency": 1,
                "statuses": statuses,
            }, ensure_ascii=False, indent=2))
            print(json.dumps(statuses[-1], ensure_ascii=False), flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/diagnostics/build_requested_results_report.py")],
                   cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
