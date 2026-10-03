#!/usr/bin/env python3
"""Run the frozen external protocol for every completed ablation."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path("outputs/ablations_v5_final")
    destination = Path("outputs/ablation_external_v5")
    destination.mkdir(parents=True, exist_ok=True)
    summary = []
    for model_dir in sorted(root.glob("model_*")):
        checkpoint = model_dir / "best_model.pt"
        config_path = model_dir / "run_config.json"
        log_path = model_dir / "train.log"
        if not checkpoint.exists() or not config_path.exists():
            continue
        if not any("Training finished" in line for line in log_path.read_text().splitlines()):
            summary.append({"model": model_dir.name, "status": "skipped_partial"})
            continue
        config = json.loads(config_path.read_text())
        output_dir = destination / model_dir.name
        result_file = output_dir / "summary.json"
        if result_file.exists():
            summary.append({"model": model_dir.name, "status": "already_done",
                            "output": str(result_file.resolve())})
            continue
        vocab = int(config.get("vocabulary_size", 0))
        command = [sys.executable, "scripts/diagnostics/evaluate_external_validation.py",
                   "--checkpoint", str(checkpoint),
                   "--training_dir", "data/perioperative_event_sequences_v5_full",
                   "--data_dirs", "data/val_mimic_v5", "data/val_mover_v5",
                   "--output_dir", str(output_dir), "--test_fraction", "0.30",
                   "--vocabulary_size", str(vocab), "--device", "cuda"]
        print(f"Starting external evaluation: {model_dir.name}", flush=True)
        try:
            subprocess.run(command, check=True)
            status = "completed"
        except subprocess.CalledProcessError as exc:
            status = f"failed:{exc.returncode}"
        summary.append({"model": model_dir.name, "status": status,
                        "output": str((output_dir / "summary.json").resolve())})
    (destination / "matrix_status.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
