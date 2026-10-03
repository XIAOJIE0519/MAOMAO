#!/usr/bin/env python3
"""Plot validation loss and metric history emitted by scripts/train.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    rows = [json.loads(line) for line in args.history.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No history rows found in {args.history}")
    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    plots = [
        ("loss", "Validation total loss", axes[0, 0]),
        ("event_loss", "Exact event loss", axes[0, 1]),
        ("time_loss", "Waiting-time loss", axes[1, 0]),
        ("next_event_hit_at_10", "Exact Hit@10", axes[1, 1]),
    ]
    for key, title, axis in plots:
        values = [row.get(key) for row in rows]
        axis.plot(epochs, values, linewidth=1.5)
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
