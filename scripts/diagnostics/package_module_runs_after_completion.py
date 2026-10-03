#!/usr/bin/env python3
"""Move finished module-ablation artifacts into the requested result folder."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "outputs/final_experiment_results_20260923"
SOURCE = ROOT / "outputs/module_ablations_richctx_20260923"
DEST = RESULT / "module_ablation_runs"


def process_active(pattern: str) -> bool:
    result = subprocess.run(["pgrep", "-af", pattern], capture_output=True, text=True)
    return any(line.split(maxsplit=1)[0] != str(os.getpid())
               for line in result.stdout.splitlines() if line.strip())


def main() -> None:
    completion = RESULT / "module_completion.json"
    while not completion.exists():
        time.sleep(30)
    # The evaluator copies the manifest/log and writes the report immediately
    # after module_completion.json. Wait for it to exit before moving the tree.
    while process_active(r"^python3 scripts/diagnostics/finish_requested_module_matrix.py"):
        time.sleep(10)
    if DEST.is_symlink():
        DEST.unlink()
    if not DEST.exists() and SOURCE.exists():
        shutil.copytree(SOURCE, DEST)
    package = {
        "status": "complete",
        "source_before_move": str(SOURCE),
        "result_directory": str(DEST),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (RESULT / "module_artifact_packaging.json").write_text(
        json.dumps(package, ensure_ascii=False, indent=2) + "\n"
    )
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/diagnostics/build_requested_results_report.py")],
        cwd=ROOT, check=True,
    )


if __name__ == "__main__":
    main()
