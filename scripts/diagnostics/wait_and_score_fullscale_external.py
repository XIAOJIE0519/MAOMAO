#!/usr/bin/env python3
"""Wait for full-scale fits/splits, score both external cohorts, refresh report."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "outputs/final_experiment_results_20260923"


def main() -> None:
    required = [
        RESULT / "baseline_metrics/xgboost_fullscale_internal.json",
        RESULT / "model_metrics/maomao_internal.json",
        RESULT / "full_maomao_reference/train.log",
    ]
    manifests = [
        RESULT / "classical_full_scale/external/mimic/manifest.json",
        RESULT / "classical_full_scale/external/mover/manifest.json",
    ]
    while True:
        ready = all(path.exists() for path in required)
        ready = ready and all(
            path.exists() and json.loads(path.read_text()).get("status") == "ready_for_full_scale_evaluation"
            for path in manifests
        )
        if ready:
            log_dir = RESULT / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            with (log_dir / "fullscale_external_scoring.log").open("w") as handle:
                subprocess.run(
                    [sys.executable, str(ROOT / "scripts/diagnostics/score_fullscale_classical_external.py")],
                    cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True,
                )
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/diagnostics/build_requested_results_report.py")],
                cwd=ROOT, check=True,
            )
            (RESULT / "fullscale_external_completion.json").write_text(json.dumps({
                "status": "complete",
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "sites": ["mimic", "mover"],
                "reference_checkpoint": str(RESULT / "full_maomao_reference/best_model.pt"),
            }, ensure_ascii=False, indent=2) + "\n")
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
