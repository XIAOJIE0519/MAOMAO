#!/usr/bin/env python3
"""A lightweight GRU baseline on the same sparse event-sequence contract."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from maomao.data.event_sequence import EventSequenceDataset, collate_event_sequences
from maomao.evaluation.event_metrics import event_metric_report_with_subsample_ci
from scripts.diagnostics.run_baseline_comparison import patient_windows
from scripts.diagnostics.evaluate_external_validation import patient_split


class GRUBaseline(nn.Module):
    def __init__(self, num_tokens: int, num_outcomes: int, num_static: int,
                 hidden: int = 256):
        super().__init__()
        self.token = nn.Embedding(num_tokens, hidden, padding_idx=0)
        self.kind = nn.Embedding(16, hidden, padding_idx=0)
        self.value = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.time = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.static = nn.Sequential(nn.Linear(num_static, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, num_outcomes)

    def forward(self, batch):
        value = torch.sign(batch["value"]) * torch.log1p(batch["value"].abs())
        value = value.clamp(-12, 12)
        vf = torch.stack((value, batch["has_value"]), -1)
        timing = torch.stack((torch.log1p(batch["time_min"].clamp_min(0)) / math.log1p(43200),
                              torch.log1p(batch["gap_min"].clamp_min(0)) / math.log1p(43200)), -1)
        x = self.token(batch["token_id"]) + self.kind(batch["token_kind"])
        x = x + self.value(vf) + self.time(timing) + self.static(batch["static"])[:, None, :]
        x, _ = self.gru(x)
        return self.head(x).float()


def move(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()}


def run_loader(model, loader, device):
    logits, targets = [], []
    model.eval()
    with torch.no_grad():
        for batch_no, batch in enumerate(loader, 1):
            batch = move(batch, device)
            pred = model(batch)
            valid = batch["loss_mask"].bool() & (batch["target_set"].sum(-1) > 0)
            if valid.any():
                logits.append(pred[valid].cpu())
                targets.append(batch["target_set"][valid].cpu())
            if batch_no == 1 or batch_no % 10 == 0:
                print(f"[gru:eval] batch={batch_no} valid_rows={sum(x.shape[0] for x in targets)}", flush=True)
    if not logits:
        raise RuntimeError("No valid GRU baseline rows")
    return torch.cat(logits), torch.cat(targets)


@torch.no_grad()
def validation_bce(model, loader, device):
    """Cheap validation objective for checkpoint selection on the large split."""
    model.eval()
    total = 0.0
    count = 0
    for batch_no, batch in enumerate(loader, 1):
        batch = move(batch, device)
        pred = model(batch)
        valid = batch["loss_mask"].bool() & (batch["target_set"].sum(-1) > 0)
        if valid.any():
            loss = F.binary_cross_entropy_with_logits(
                pred[valid], batch["target_set"][valid], reduction="sum")
            total += float(loss)
            count += int(valid.sum())
        if batch_no == 1 or batch_no % 50 == 0:
            print(f"[gru:val] batch={batch_no} valid_rows={count}", flush=True)
    if not count:
        raise RuntimeError("No valid GRU validation rows")
    return total / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_dir", type=Path, required=True)
    parser.add_argument("--external_dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--external_split_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--bootstrap_repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # patient_split writes one split artifact per external source; create the
    # parent explicitly so an external-only resume can reuse the checkpoint.
    args.external_split_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = EventSequenceDataset(args.training_dir, 256, 128, dynamic_windows=False)
    train_idx, val_idx, split = patient_windows(dataset, 0.1, args.seed)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=64, shuffle=True,
                              num_workers=2, pin_memory=device.type == "cuda",
                              collate_fn=collate_event_sequences)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=64, shuffle=False,
                            num_workers=0, pin_memory=device.type == "cuda",
                            collate_fn=collate_event_sequences)
    model = GRUBaseline(dataset.num_tokens, dataset.num_outcomes, dataset.num_static).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    best, stale, best_epoch = None, 0, 0
    for epoch in range(args.epochs):
        model.train()
        total_train = len(train_loader)
        for batch_no, batch in enumerate(train_loader, 1):
            batch = move(batch, device)
            pred = model(batch)
            valid = batch["loss_mask"].bool() & (batch["target_set"].sum(-1) > 0)
            if not valid.any():
                continue
            loss = F.binary_cross_entropy_with_logits(pred[valid], batch["target_set"][valid])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if batch_no == 1 or batch_no % 10 == 0 or batch_no == total_train:
                print(f"[gru:train] epoch={epoch + 1}/{args.epochs} "
                      f"batch={batch_no}/{total_train} loss={float(loss):.6f}", flush=True)
        val_bce = validation_bce(model, val_loader, device)
        report = {"model": "gru_baseline", "epoch": epoch + 1,
                  "validation_bce": val_bce}
        score = -val_bce
        if best is None or score > best.get("selection_score", float("-inf")):
            best, best_epoch, stale = {**report, "selection_score": score}, epoch + 1, 0
            torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                        "metrics": report, "args": vars(args)}, args.output_dir / "gru_best.pt")
        else:
            stale += 1
        print(f"gru epoch={epoch + 1}/{args.epochs} validation_bce={val_bce:.6f} stale={stale}", flush=True)
        if stale >= args.patience:
            break
    result = {"protocol": "patient-disjoint sparse next-event comparison",
              "split": split, "model": best, "best_epoch": best_epoch,
              "outcomes": len(dataset.meta["outcome_vocabulary"]), "external": {}}
    for data_dir in args.external_dirs:
        print(f"[gru:external] starting {data_dir.name}", flush=True)
        ext = EventSequenceDataset(data_dir, 256, 128, dynamic_windows=False)
        cal_idx, test_idx, ext_split = patient_split(ext, data_dir, args.external_split_dir, 0.1, args.seed + 200)
        cal_loader = DataLoader(Subset(ext, cal_idx), batch_size=128, shuffle=False,
                                num_workers=0, collate_fn=collate_event_sequences)
        test_loader = DataLoader(Subset(ext, test_idx), batch_size=128, shuffle=False,
                                 num_workers=0, collate_fn=collate_event_sequences)
        cache_path = args.output_dir / f"{data_dir.name}_gru_external_logits.pt"
        if cache_path.exists():
            print(f"[gru:external] loading cached logits {cache_path.name}", flush=True)
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            cal_logits, cal_targets = cached["cal_logits"], cached["cal_targets"]
            test_logits, test_targets = cached["test_logits"], cached["test_targets"]
            del cached
        else:
            checkpoint = torch.load(args.output_dir / "gru_best.pt", map_location=device, weights_only=False)
            ext_model = GRUBaseline(ext.num_tokens, ext.num_outcomes, ext.num_static).to(device)
            ext_model.load_state_dict(checkpoint["model"])
            cal_logits, cal_targets = run_loader(ext_model, cal_loader, device)
            test_logits, test_targets = run_loader(ext_model, test_loader, device)
            torch.save({"cal_logits": cal_logits, "cal_targets": cal_targets,
                        "test_logits": test_logits, "test_targets": test_targets}, cache_path)
            print(f"[gru:external] cached logits to {cache_path.name}", flush=True)
        calibration_count = len(cal_targets)
        test_count = len(test_targets)
        # Calibration rows are not used by this raw baseline report.  Release
        # them before bootstrap scoring; the separate calibration script loads
        # the cached file when it needs them.
        del cal_logits, cal_targets
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[gru:external] scoring {data_dir.name} calibration_rows={calibration_count} "
              f"test_rows={test_count}", flush=True)
        raw = event_metric_report_with_subsample_ci(
            test_logits, test_targets, ext.meta["outcome_vocabulary"],
            bootstrap_repeats=args.bootstrap_repeats, bootstrap_seed=args.seed + 1)
        result["external"][data_dir.name] = {"split": ext_split, "raw": raw,
                                               "calibration_rows": int(calibration_count),
                                               "test_rows": int(test_count)}
    (args.output_dir / "gru_baseline.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({"output": str(args.output_dir / "gru_baseline.json"), "best_epoch": best_epoch}), flush=True)


if __name__ == "__main__":
    main()
