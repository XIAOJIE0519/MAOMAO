#!/usr/bin/env python3
"""Run full-scale classical ML candidates serially with a hard per-model cap."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/final_experiment_results_20260923/classical_ml_fullscale"
MODELS = ("gaussian_nb", "decision_tree", "random_forest", "hist_gradient_boosting",
          "lightgbm", "catboost", "xgboost", "adaboost")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def history_append(path: Path, record: dict) -> None:
    rows = json.loads(path.read_text()) if path.exists() else []
    rows.append(record)
    write_json(path, rows if isinstance(rows, dict) else {"attempts": rows})


def live_candidate_pids(name: str) -> list[int]:
    """Find matching candidate workers through /proc argv, without shell false positives."""
    found=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv=(entry/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except OSError:
            continue
        if any(arg.endswith('run_fullscale_ml_candidate.py') for arg in argv) and any(
            argv[i]=='--model' and i+1<len(argv) and argv[i+1]==name
            for i in range(len(argv))
        ):
            found.append(int(entry.name))
    return found


def run_one(name: str, timeout_s: int) -> dict:
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    status_path = folder / "status.json"
    metrics_path = folder / "metrics.json"
    if metrics_path.exists():
        try:
            saved = json.loads(metrics_path.read_text())
            if saved.get("status") == "completed":
                return {"model": name, "status": "already_completed", "metrics": str(metrics_path)}
        except Exception:
            pass
    running_pids=live_candidate_pids(name)
    if running_pids:
        return {"model":name,"status":"already_running","active_pids":running_pids,
                "detail":"Refused duplicate fit because a matching worker PID is active."}
    if status_path.exists():
        try:
            saved_status = json.loads(status_path.read_text())
            if saved_status.get("status") in ("timeout_20min_no_result", "failed_no_result"):
                return {**saved_status, "status": "already_terminal",
                        "terminal_status": saved_status.get("status")}
        except Exception:
            pass

    history_path = folder / "attempt_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else {"attempts": []}
    if isinstance(history, list):
        history = {"attempts": history}
    attempts = history.setdefault("attempts", [])
    today = datetime.now(timezone.utc).date().isoformat()
    # Recover the remaining wall-time budget after an interrupted supervisor.
    if status_path.exists():
        try:
            saved_status=json.loads(status_path.read_text())
            if saved_status.get("status")=="running" and str(saved_status.get("started_utc","")).startswith(today):
                stale_start=datetime.fromisoformat(saved_status['started_utc'])
                stale_elapsed=max(0,(datetime.now(timezone.utc)-stale_start).total_seconds())
                stale={"model":name,"status":"interrupted_no_result",
                       "attempt":saved_status.get('attempt'),
                       "started_utc":saved_status['started_utc'],
                       "timeout_seconds":timeout_s,"elapsed_seconds":round(stale_elapsed,2),
                       "finished_utc":now(),"result_file_present":metrics_path.exists(),
                       "detail":"Recovered stale running status after confirming no matching candidate PID."}
                attempts.append(stale)
                write_json(history_path,history)
                write_json(status_path,stale)
        except (ValueError,TypeError,KeyError):
            pass
    today_attempts = [r for r in attempts if str(r.get("started_utc", "")).startswith(today)]
    used_seconds = sum(float(r.get("elapsed_seconds", 0) or 0) for r in today_attempts)
    remaining_s = max(0, int(timeout_s - used_seconds))
    attempt_no = max([int(r.get("attempt", 0) or 0) for r in today_attempts] + [0]) + 1
    if remaining_s <= 0:
        record = {"model": name, "status": "timeout_20min_no_result", "attempt": attempt_no,
                  "started_utc": now(), "timeout_seconds": timeout_s,
                  "elapsed_seconds": used_seconds, "finished_utc": now(),
                  "detail": "Cumulative per-model wall-time cap exhausted; no valid metrics JSON."}
        attempts.append(record)
        write_json(history_path, history)
        write_json(status_path, record)
        return record

    started = time.monotonic()
    record = {"model": name, "status": "running", "started_utc": now(),
              "timeout_seconds": timeout_s, "attempt_timeout_seconds": remaining_s,
              "previous_attempt_seconds": used_seconds, "attempt": attempt_no}
    write_json(status_path, record)
    log_path = folder / f"run_20min_attempt_{attempt_no}.log"
    command = [sys.executable, str(ROOT / "scripts/diagnostics/run_fullscale_ml_candidate.py"),
               "--model", name]
    with log_path.open("w", buffering=1) as log:
        log.write(json.dumps({"queue": "serial_full_scale_ml", "command": command,
                              "started_utc": record["started_utc"],
                              "hard_timeout_seconds": timeout_s,
                              "attempt_timeout_seconds": remaining_s,
                              "previous_attempt_seconds": used_seconds}) + "\n")
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=remaining_s)
            elapsed = round(time.monotonic() - started, 2)
            if code == 0 and metrics_path.exists():
                metrics = json.loads(metrics_path.read_text())
                metrics['attempt_wall_time_seconds']=elapsed
                metrics['cumulative_wall_time_seconds']=round(used_seconds+elapsed,2)
                write_json(metrics_path,metrics)
                record.update({"status": "completed", "exit_code": 0,
                           "elapsed_seconds": elapsed,
                           "cumulative_elapsed_seconds": round(used_seconds+elapsed,2),
                           "finished_utc": now(),
                               "metrics_file": str(metrics_path.relative_to(ROOT)),
                               "micro_auprc": metrics.get("micro_auprc"),
                               "micro_auroc": metrics.get("micro_auroc")})
            else:
                record.update({"status": "failed_no_result", "exit_code": code,
                               "elapsed_seconds": elapsed,
                               "cumulative_elapsed_seconds": round(used_seconds+elapsed,2),
                               "finished_utc": now()})
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            attempt_elapsed = round(time.monotonic() - started, 2)
            record.update({"status": "timeout_20min_no_result", "exit_code": -signal.SIGTERM,
                           "elapsed_seconds": attempt_elapsed,
                           "cumulative_elapsed_seconds": round(used_seconds+attempt_elapsed,2),
                           "finished_utc": now()})
        log.write(json.dumps(record, ensure_ascii=False) + "\n")
    attempts.append(record)
    write_json(history_path, history)
    write_json(status_path, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--models", nargs="*", choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    queue_path = OUT / "queue_status.json"
    previous = json.loads(queue_path.read_text()) if queue_path.exists() else {}
    queue = {"status": "running", "started_utc": now(),
             "per_model_timeout_seconds": args.timeout_seconds,
             "models": [], "previous_queue": previous.get("status")}
    write_json(queue_path, queue)

    for name in args.models:
        active=live_candidate_pids(name)
        if active:
            queue.update({"status":"already_running","current_model":name,
                          "active_pids":active,"updated_utc":now()})
            write_json(queue_path,queue)
            print(json.dumps({"model":name,"status":"already_running","active_pids":active},ensure_ascii=False),flush=True)
            return
        queue["current_model"] = name
        queue["models"] = [m for m in queue["models"] if m.get("model") != name]
        queue["models"].append({"model": name, "status": "running",
                                 "started_utc": now(),
                                 "timeout_seconds": args.timeout_seconds})
        write_json(queue_path, queue)
        result = run_one(name, args.timeout_seconds)
        queue["models"] = [m for m in queue["models"] if m.get("model") != name] + [result]
        write_json(queue_path, queue)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        # Refresh the report after every terminal candidate so completed
        # results survive an interruption of the remaining queue.
        subprocess.run([sys.executable, str(ROOT / "scripts/diagnostics/integrate_requested_baselines.py")],
                       cwd=ROOT, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([sys.executable, str(ROOT / "scripts/diagnostics/package_final_results.py")],
                       cwd=ROOT, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    queue["current_model"] = None
    queue["finished_utc"] = now()
    queue["status"] = "completed" if all(
        m.get("status") in ("completed", "already_completed", "already_terminal",
                             "timeout_20min_no_result", "failed_no_result")
        for m in queue["models"]
    ) else "partial"
    write_json(queue_path, queue)
    print(json.dumps(queue, ensure_ascii=False, indent=2), flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/diagnostics/integrate_requested_baselines.py")],
                   cwd=ROOT, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([sys.executable, str(ROOT / "scripts/diagnostics/package_final_results.py")],
                   cwd=ROOT, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
