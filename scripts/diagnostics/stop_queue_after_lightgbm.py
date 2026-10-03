#!/usr/bin/env python3
"""Allow one LightGBM attempt to finish, then stop the queued candidates."""
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/final_experiment_results_20260923/classical_ml_fullscale"
STATUS = OUT / "lightgbm/status.json"
QUEUE = OUT / "queue_status.json"

while True:
    try:
        status = json.loads(STATUS.read_text()).get("status")
    except (OSError, ValueError):
        status = None
    if status == "running":
        try:
            started = datetime.fromisoformat(json.loads(STATUS.read_text())["started_utc"])
            elapsed = max(0, int((datetime.now(timezone.utc) - started).total_seconds()))
            unit_state = subprocess.run(["systemctl", "--user", "is-active",
                                         "maomao-fullscale-ml-queue.service"],
                                        check=False, capture_output=True, text=True).stdout.strip()
            candidate_live = any(
                Path(f"/proc/{pid}/cmdline").exists() and
                b"run_fullscale_ml_candidate.py" in Path(f"/proc/{pid}/cmdline").read_bytes() and
                b"lightgbm" in Path(f"/proc/{pid}/cmdline").read_bytes()
                for pid in (int(p.name) for p in Path("/proc").iterdir() if p.name.isdigit())
            )
            if elapsed > 5 and unit_state != "active" and not candidate_live:
                result = {"model": "lightgbm", "status": "failed_no_result",
                          "attempt": 1, "started_utc": started.isoformat(),
                          "finished_utc": datetime.now(timezone.utc).isoformat(),
                          "elapsed_seconds": elapsed, "timeout_seconds": 1200,
                          "result_file_present": (OUT / "lightgbm/metrics.json").exists(),
                          "detail": "The LightGBM worker and queue stopped before writing terminal status; no valid metrics or exact exit cause are available, and no restart is allowed under the single-attempt scope."}
                tmp_status = STATUS.with_suffix(".json.tmp")
                tmp_status.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
                tmp_status.replace(STATUS)
                history_path = OUT / "lightgbm/attempt_history.json"
                history = json.loads(history_path.read_text()) if history_path.exists() else {"attempts": []}
                history.setdefault("attempts", []).append(result)
                tmp_history = history_path.with_suffix(".json.tmp")
                tmp_history.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n")
                tmp_history.replace(history_path)
                status = result["status"]
        except (OSError, ValueError, KeyError, TypeError):
            pass
    if status and status != "running":
        subprocess.run(["systemctl", "--user", "stop", "maomao-fullscale-ml-queue.service"],
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            queue = json.loads(QUEUE.read_text())
        except (OSError, ValueError):
            queue = {}
        candidate = json.loads(STATUS.read_text())
        candidate["user_scope"] = "single LightGBM run; no other candidate should launch"
        queue.update({
            "status": "completed" if status == "completed" else "partial",
            "current_model": None,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "models": [candidate],
            "scope": "LightGBM only; one attempt, 20-minute hard cap",
        })
        tmp = QUEUE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(queue, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(QUEUE)
        subprocess.run([sys.executable, str(ROOT / "scripts/diagnostics/integrate_requested_baselines.py")],
                       cwd=ROOT, check=False)
        subprocess.run([sys.executable, str(ROOT / "scripts/diagnostics/package_final_results.py")],
                       cwd=ROOT, check=False)
        break
    time.sleep(0.25)
