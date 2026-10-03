#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/outputs/final_experiment_results_20260923/classical_ml_fullscale/linear_svm"
mkdir -p "$OUT"
LOG="$OUT/run.log"
START="$(date +%s)"
timeout --signal=TERM --kill-after=5s 600s python3 "$ROOT/scripts/diagnostics/run_fullscale_linear_svm.py" >"$LOG" 2>&1
RC=$?
END="$(date +%s)"
STATUS="failed"
if [ "$RC" -eq 0 ] && [ -s "$OUT/metrics.json" ]; then STATUS="completed"; fi
if [ "$RC" -eq 124 ] || [ "$RC" -eq 137 ]; then STATUS="timeout_no_result"; fi
python3 - "$OUT/status.json" "$STATUS" "$RC" "$((END-START))" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); p.write_text(json.dumps({"status":sys.argv[2],"exit_code":int(sys.argv[3]),"elapsed_seconds":int(sys.argv[4]),"hard_timeout_seconds":600},indent=2))
PY
exit 0
