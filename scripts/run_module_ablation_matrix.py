#!/usr/bin/env python3
"""Build or execute the V5 MAOMAO module-ablation matrix.

Every row keeps the data contract, patient-level 9:1 validation split, model
size, optimizer budget, and dual-timescale head fixed.  Only one requested
module is removed at a time. The Full MAOMAO reference is reported in the separate
five-model comparison, not trained or listed as an ablation row.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = {
    "no_block_causal": {"--no-same_time_block_causal": True},
    "no_relative_time": {"--no-relative_time_attention": True},
    "no_family_head": {"--no-use_family_head": True},
    "no_clock_phase_summary": {
        "--no-clock_phase_context": True,
        "--no-phase_memory": True,
    },
    "no_measurement_intensity": {"--no-observation_intensity": True},
    "no_masked_event_value": {
        "--masked_event_loss_weight": "0",
        "--masked_value_loss_weight": "0",
    },
    "no_event_conditioned_time": {"--no-event_conditioned_time_head": True},
}


def build_command(data_dir: Path, output_dir: Path, name: str,
                  epochs: int, batch_size: int) -> list[str]:
    command = [
        sys.executable, "scripts/train.py",
        "--data_dir", str(data_dir), "--output_dir", str(output_dir / name),
        "--epochs", str(epochs), "--batch_size", str(batch_size),
        "--num_workers", "0", "--device", "cuda", "--require_cuda",
        "--dual_timescale_time_head", "--fine_time_bins", "24",
        "--long_time_bins", "44", "--validation_fraction", "0.1",
        "--early_stopping_patience", "25", "--min_lr", "1e-6",
        "--lr_plateau_patience", "8", "--lr_plateau_factor", "0.3",
        "--no-dynamic_windows",
        "--same_time_block_causal", "--relative_time_attention",
        "--event_conditioned_time_head", "--phase_memory",
        "--clock_phase_context", "--observation_intensity",
        "--masked_event_loss_weight", "0.2",
        "--masked_value_loss_weight", "0.1",
        "--log_interval", "100", "--save_interval", "500",
    ]
    for flag, value in EXPERIMENTS[name].items():
        command.append(flag)
        if value is not True:
            command.append(str(value))
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path,
                        default=Path("data/perioperative_event_sequences_v5_static7"))
    parser.add_argument("--output_dir", type=Path,
                        default=Path("outputs/module_ablations_v5"))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    manifest = []
    for name in EXPERIMENTS:
        experiment_dir = args.output_dir / name
        log_path = experiment_dir / "train.log"
        completed = log_path.exists() and any(
            "Training finished" in line for line in log_path.read_text().splitlines())
        command = build_command(args.data_dir, args.output_dir, name,
                                args.epochs, args.batch_size)
        if (experiment_dir / "checkpoint_latest.pt").exists() and not completed:
            command += ["--resume", "auto", "--reset_early_stopping"]
        manifest.append({"name": name, "completed_before_run": completed,
                         "command": command})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "module_ablation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.execute:
        for row in manifest:
            if row["completed_before_run"]:
                continue
            subprocess.run(row["command"], check=True)


if __name__ == "__main__":
    main()
