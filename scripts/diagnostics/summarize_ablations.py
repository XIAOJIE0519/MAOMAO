#!/usr/bin/env python3
"""Summarize completed MAOMAO size/vocabulary ablations."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def main() -> None:
    root = Path("outputs/ablations_v5_final")
    rows = []
    for history in sorted(root.glob("model_*/validation_history.jsonl")):
        records = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
        if not records:
            continue
        best = min(records, key=lambda item: float(item.get("validation_loss", item.get("loss", float("inf")))))
        config_path = history.parent / "run_config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        rows.append({
            "model": history.parent.name,
            "status": "completed" if any("Training finished" in line for line in (history.parent / "train.log").read_text().splitlines()) else "partial",
            "epochs_observed": len(records),
            "best_epoch": best.get("epoch"),
            "validation_loss": best.get("validation_loss", best.get("loss")),
            "hit_at_5": best.get("next_event_hit_at_5"),
            "hit_at_10": best.get("next_event_hit_at_10"),
            "family_hit_at_5": best.get("same_family_hit_at_5", best.get("family_hit_at_5")),
            "family_hit_at_10": best.get("same_family_hit_at_10", best.get("family_hit_at_10")),
            "time_mae_hours": best.get("time_mae_hours"),
            "hidden_dim": config.get("hidden_dim"),
            "num_layers": config.get("num_layers"),
            "num_heads": config.get("num_heads"),
            "ffn_dim": config.get("ffn_dim"),
            "vocabulary_size": config.get("vocabulary_size"),
            "checkpoint": str((history.parent / "best_model.pt").resolve()),
        })
    root.mkdir(parents=True, exist_ok=True)
    (root / "ablation_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    if rows:
        with (root / "ablation_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
