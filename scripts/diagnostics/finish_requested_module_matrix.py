#!/usr/bin/env python3
"""Wait for the seven MAOMAO module runs, then score and package them."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RUN_DIR = ROOT / "outputs/module_ablations_richctx_20260923"
RESULT_DIR = ROOT / "outputs/final_experiment_results_20260923"
DATA_DIR = ROOT / "data/perioperative_event_sequences_v5_richctx_static7"
MODELS = (
    "no_block_causal", "no_relative_time", "no_family_head",
    "no_clock_phase_summary", "no_measurement_intensity",
    "no_masked_event_value", "no_event_conditioned_time",
)


def runner_active() -> bool:
    result = subprocess.run(["pgrep", "-af", "^python3 scripts/run_module_ablation_matrix.py"],
                            capture_output=True, text=True)
    return bool(result.stdout.strip())


def main() -> None:
    log_dir = RESULT_DIR / "logs"
    metric_dir = RESULT_DIR / "module_metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    while runner_active():
        time.sleep(60)

    full_manifest = RESULT_DIR / "classical_full_scale/manifest.json"
    while not full_manifest.exists() or json.loads(full_manifest.read_text()).get("status") != "ready_for_full_scale_fit":
        time.sleep(30)

    # Refit the Full MAOMAO comparison reference with the exact fixed-window
    # exposure used by this ablation matrix. The older reference checkpoint
    # used dynamic crops, so it is not the correct comparator for these runs.
    full_reference_dir = RESULT_DIR / "full_maomao_reference"
    full_reference_dir.mkdir(parents=True, exist_ok=True)
    reference_log = log_dir / "full_maomao_reference_training.log"
    reference_train_log = full_reference_dir / "train.log"
    reference_finished = reference_train_log.exists() and "Training finished" in reference_train_log.read_text(errors="ignore")
    if not reference_finished:
        command = [
            sys.executable, str(ROOT / "scripts/train.py"),
            "--data_dir", str(DATA_DIR), "--output_dir", str(full_reference_dir),
            "--epochs", "7", "--batch_size", "64", "--num_workers", "0",
            "--device", "cuda", "--require_cuda", "--dual_timescale_time_head",
            "--fine_time_bins", "24", "--long_time_bins", "44",
            "--validation_fraction", "0.1", "--early_stopping_patience", "25",
            "--min_lr", "1e-6", "--lr_plateau_patience", "8",
            "--lr_plateau_factor", "0.3", "--no-dynamic_windows",
            "--same_time_block_causal", "--relative_time_attention",
            "--event_conditioned_time_head", "--phase_memory",
            "--clock_phase_context", "--observation_intensity",
            "--masked_event_loss_weight", "0.2", "--masked_value_loss_weight", "0.1",
            "--log_interval", "100", "--save_interval", "500",
        ]
        if (full_reference_dir / "checkpoint_latest.pt").exists():
            command += ["--resume", "auto", "--reset_early_stopping"]
        with reference_log.open("w") as handle:
            subprocess.run(command, cwd=ROOT, stdout=handle,
                           stderr=subprocess.STDOUT, check=True)
    reference_checkpoint = full_reference_dir / "best_model.pt"
    if not reference_checkpoint.exists():
        raise FileNotFoundError(reference_checkpoint)

    for name, kind, checkpoint in [
        ("maomao_internal", "maomao", reference_checkpoint),
        ("gru_internal", "gru", ROOT / "outputs/baseline_gru_richctx/gru_best.pt"),
    ]:
        command = [
            sys.executable, str(ROOT / "scripts/diagnostics/evaluate_internal_checkpoints.py"),
            "--kind", kind, "--checkpoint", str(checkpoint), "--data_dir", str(DATA_DIR),
            "--output", str(RESULT_DIR / f"model_metrics/{name}.json"),
            "--model_name", name, "--batch_size", "64", "--bootstrap_repeats", "200",
            "--full_validation",
        ]
        with (log_dir / f"{name}_evaluation.log").open("w") as handle:
            subprocess.run(command, cwd=ROOT, stdout=handle,
                           stderr=subprocess.STDOUT, check=True)

    statuses = []
    for name in MODELS:
        model_dir = RUN_DIR / name
        log = model_dir / "train.log"
        completed = log.exists() and "Training finished" in log.read_text(errors="ignore")
        checkpoint = model_dir / "best_model.pt"
        if completed and checkpoint.exists():
            command = [
                sys.executable, str(ROOT / "scripts/diagnostics/evaluate_internal_checkpoints.py"),
                "--kind", "maomao", "--checkpoint", str(checkpoint),
                "--data_dir", str(DATA_DIR), "--output",
                str(metric_dir / f"{name}.json"), "--model_name", name,
                "--batch_size", "64", "--bootstrap_repeats", "200",
                "--full_validation",
            ]
            with (log_dir / f"{name}_evaluation.log").open("w") as handle:
                result = subprocess.run(command, cwd=ROOT, stdout=handle,
                                        stderr=subprocess.STDOUT, check=False)
            statuses.append({"name": name, "status": "evaluated" if result.returncode == 0 else "evaluation_failed",
                             "returncode": result.returncode,
                             "metrics": str(metric_dir / f"{name}.json")})
        else:
            statuses.append({"name": name, "status": "training_incomplete" if not completed else "checkpoint_missing"})
    for source, dest in [
        (RUN_DIR / "module_ablation_manifest.json", log_dir / "module_ablation_manifest.json"),
        (ROOT / "outputs/module_ablations_richctx_20260923_execution.log",
         log_dir / "module_ablation_execution.log"),
    ]:
        if source.exists():
            dest.write_bytes(source.read_bytes())
    (RESULT_DIR / "module_completion.json").write_text(json.dumps({
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "statuses": statuses,
    }, ensure_ascii=False, indent=2))
    subprocess.run([sys.executable, str(ROOT / "scripts/diagnostics/build_requested_results_report.py")],
                   cwd=ROOT, check=True)
    print(json.dumps(statuses, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
