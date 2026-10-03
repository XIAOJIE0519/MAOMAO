#!/usr/bin/env python3
"""Refresh the comparison report as soon as all full-scale internal baselines finish."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MARKER = ROOT / "outputs/final_experiment_results_20260923/baseline_metrics/fullscale_classical_completion.json"


def main() -> None:
    while not MARKER.exists():
        time.sleep(30)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/diagnostics/build_requested_results_report.py")],
        cwd=ROOT, check=True,
    )


if __name__ == "__main__":
    main()
