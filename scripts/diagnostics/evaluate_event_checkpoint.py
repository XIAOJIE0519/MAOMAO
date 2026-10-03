#!/usr/bin/env python3
"""Evaluate an EventMAOMAO checkpoint on a fixed training-fit window sample.

There is intentionally no validation claim: the project currently uses every
operation episode for gradient training.  Event-time MAE is calculated only
where an event was observed; right-censored episode ends contribute to the
waiting-time likelihood but not to MAE.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from maomao.data.event_sequence import EventSequenceDataset, collate_event_sequences  # noqa: E402
from maomao.data.outcome_families import family_ids  # noqa: E402
from maomao.models.event_maomao import EventMAOMAO  # noqa: E402
from scripts.train import monitor  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/event_maomao_multitask_100/best_model.pt")
    parser.add_argument("--data_dir", default="data/perioperative_event_sequences_v4")
    parser.add_argument("--windows", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output", default="outputs/event_maomao_multitask_100/train_fit_evaluation.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["args"]
    dataset = EventSequenceDataset(
        args.data_dir, config.get("block_size", 256), config.get("window_stride", 128))
    window_count = min(args.windows, len(dataset))
    indices = (list(range(len(dataset))) if window_count == len(dataset) else
               torch.linspace(0, len(dataset) - 1, window_count,
                              dtype=torch.long).tolist())
    loader = DataLoader(
        Subset(dataset, indices), batch_size=config.get("batch_size", 256),
        shuffle=False, num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_event_sequences,
    )
    model = EventMAOMAO(
        dataset.num_tokens, dataset.num_outcomes, dataset.num_static,
        config.get("hidden_dim", 256), config.get("num_layers", 8),
        config.get("num_heads", 8), config.get("ffn_dim", 1024),
        config.get("dropout", 0.1), config.get("initial_event_interval_hours", 24.0),
        config.get("decoupled_time_head", False),
        config.get("enhanced_time_encoding", False),
        len(dataset.trajectory_horizons_hours),
        config.get("lognormal_time_head", False),
        outcome_family_ids=torch.as_tensor(
            dataset.meta.get("outcome_to_family", family_ids(
                dataset.meta["outcome_vocabulary"]).tolist()), dtype=torch.long),
        use_family_head=config.get("use_family_head", any(
            key.startswith("family_head.") for key in checkpoint["model"])),
        same_time_block_causal=config.get("same_time_block_causal", False),
        relative_time_attention=config.get("relative_time_attention", False),
        event_conditioned_time_head=config.get("event_conditioned_time_head", False),
        phase_memory=config.get("phase_memory", False),
        clock_phase_context=config.get("clock_phase_context", True),
        observation_intensity=config.get("observation_intensity", False),
        value_reconstruction=config.get("masked_value_loss_weight", 0.0) > 0,
        dual_timescale_time_head=config.get("dual_timescale_time_head", False),
        fine_time_bins=config.get("fine_time_bins", 24),
        long_time_bins=config.get("long_time_bins", 44),
    ).to(device)
    clock_phase_ids = [int(token_id) for token, token_id in dataset.meta["token_vocabulary"].items()
                       if token == "<CLOCK>" or token.startswith("phase_summary:")]
    model.clock_phase_token_ids = torch.as_tensor(clock_phase_ids, dtype=torch.long, device=device)
    model.load_state_dict(checkpoint["model"])
    amp_enabled = device.type == "cuda" and config.get("precision", "bf16") != "fp32"
    amp_dtype = (torch.bfloat16 if config.get("precision", "bf16") == "bf16"
                 else torch.float16)
    outcome_names = dataset.meta["outcome_vocabulary"]
    counts = torch.tensor([
        max(1, int(dataset.meta.get("audit_counts", {}).get(name, 1)))
        for name in outcome_names
    ], dtype=torch.float32, device=device)
    class_weights = torch.sqrt(counts.sum() / (len(counts) * counts)).clamp(0.5, 5.0)
    class_weights = class_weights / class_weights.mean()
    stable_index = outcome_names.index("stable_interval") if "stable_interval" in outcome_names else -1
    metrics = monitor(
        model, loader, device, amp_enabled, amp_dtype,
        config.get("time_loss_weight", 1.0), outcome_names,
        config.get("max_wait_hours", 24.0 * 38), stable_index,
        config.get("trajectory_loss_weight", 0.5), class_weights,
        config.get("family_loss_weight", 0.0),
    )
    report = {
        "scope": "training-fit fixed monitor; not independent validation",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "optimizer_step": int(checkpoint["global_step"]),
        "windows": len(indices),
        **metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
