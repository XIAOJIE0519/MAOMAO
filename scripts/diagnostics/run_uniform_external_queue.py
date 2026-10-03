#!/usr/bin/env python3
"""Resume the eight-cohort five-model external queue and refresh artifacts."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.diagnostics.build_fullscale_external_rows import SOURCES

OUT = ROOT / "outputs/external_validation_final_maomao_uniform"
ROWS = ROOT / "outputs/final_experiment_results_20260923/classical_full_scale/external"
MODELS = ("univariate", "logistic_regression", "xgboost", "ann", "maomao")
ORDER = ("mimic", "ntuh", "asac", "uq",
         "surgical_pooled", "sicdb", "mover", "eicu")


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(payload: dict) -> None:
    path = OUT / "queue_status.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def active(command_fragment: str) -> bool:
    output = subprocess.check_output(["ps", "-eo", "args="], text=True)
    return any(command_fragment in line and "python" in line for line in output.splitlines())


def ready(site: str) -> bool:
    path = ROWS / site / "manifest.json"
    return path.exists() and json.loads(path.read_text()).get("status") == "ready_for_full_scale_evaluation"


def completed_count(site: str) -> int:
    count = 0
    for model in MODELS:
        folder = OUT / site / model
        status = folder / "status.json"
        metrics = folder / "metrics.json"
        if status.exists() and metrics.exists() and json.loads(status.read_text()).get("status") == "completed":
            count += 1
    return count


def refresh() -> None:
    for script in ("build_uniform_results_report.py", "package_uniform_final_results.py"):
        subprocess.run([sys.executable, str(ROOT / "scripts/diagnostics" / script)],
                       cwd=ROOT, check=True)


def run_logged(script: str, args: list[str], log_name: str) -> None:
    log_dir = OUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / log_name).open("a") as handle:
        handle.write(f"\n[{stamp()}] starting {script} {' '.join(args)}\n")
        handle.flush()
        subprocess.run([sys.executable, "-u", str(ROOT / "scripts/diagnostics" / script), *args],
                       cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    state = {"status": "running", "order": list(ORDER), "started_utc": stamp(), "sites": {}}
    write(state)
    try:
        for site in ORDER:
            state["current_site"] = site
            state["sites"][site] = {"status": "preparing", "completed_models": completed_count(site)}
            write(state)
            if not ready(site):
                source = ROOT / SOURCES[site] / "event_sequence_meta.json"
                while not source.exists() or not json.loads(source.read_text()).get("complete"):
                    if site in ("eicu", "sicdb") and not active(f"preprocess_icu_validation.py {site}"):
                        raise RuntimeError(f"{site}: aligned source missing and no active preprocessing")
                    time.sleep(30)
                run_logged("build_fullscale_external_rows.py", ["--sites", site],
                           f"{site}_row_materialization.log")
            fragment = f"score_uniform_external_five.py --site {site}"
            last_count = -1
            while active(fragment):
                count = completed_count(site)
                if count != last_count:
                    state["sites"][site] = {"status": "evaluating", "completed_models": count}
                    write(state)
                    if count > 0:
                        refresh()
                    last_count = count
                time.sleep(30)
            if completed_count(site) < len(MODELS):
                state["sites"][site] = {"status": "evaluating", "completed_models": completed_count(site)}
                write(state)
                run_logged("score_uniform_external_five.py", ["--site", site], f"{site}_evaluation.log")
            if completed_count(site) != len(MODELS):
                raise RuntimeError(f"{site}: evaluator exited without all five metric files")
            state["sites"][site] = {"status": "completed", "completed_models": len(MODELS)}
            write(state)
            refresh()
        state["status"] = "completed"
        state["finished_utc"] = stamp()
        state.pop("current_site", None)
        write(state)
        refresh()
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = repr(exc)
        state["failed_utc"] = stamp()
        write(state)
        raise


if __name__ == "__main__":
    main()
