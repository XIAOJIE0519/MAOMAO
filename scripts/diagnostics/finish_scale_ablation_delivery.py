"""Finalize only verified new results, including residual legacy summaries.

    A live training/evaluation supervisor is never restarted or interrupted.
    This command can be used after the serial queue exits, even when a prior
    supervisor loaded an older cleanup function before it was updated.
"""
from __future__ import annotations

import fcntl
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.diagnostics.scale_ablation_scope import *
from scripts.diagnostics.run_scale_ablation_queue import delete_superseded, live_worker, utc
from scripts.diagnostics.verify_scale_ablations import audit as experiment_audit


def run(script):
    command = [sys.executable, str(ROOT / "scripts/diagnostics" / script)]
    if script.startswith("verify_"):
        command.append("--require-complete")
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / ".queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Live experiment supervisor; finalization deferred.")
            return
        worker = live_worker()
        if worker:
            print(f"Live experiment worker {worker['pid']}; finalization deferred.")
            return
        result = experiment_audit()
        if not result["complete"]:
            raise RuntimeError(f"New experiments are incomplete; old results preserved: {result}")
        write(OUT / "verification.json", result)
        # The complete replacement must be published and independently checked
        # before removing even a remaining standalone historical summary.
        for script in ("build_scale_ablation_report.py", "build_uniform_results_report.py",
                       "package_uniform_final_results.py", "verify_uniform_final_results.py"):
            run(script)
        state = read(OUT / "queue_status.json")
        delete_superseded(state)
        receipt = read(OUT / "legacy_removal.json")
        if not receipt["all_directories_absent"]:
            raise RuntimeError("Legacy result removal is incomplete")
        if not receipt.get("all_edited_reports_clean"):
            raise RuntimeError("Legacy numerical chapters remain in historical report snapshots")
        for item in receipt.get("edited_reports", []):
            if sha256(Path(item["path"])) != item["after_sha256"]:
                raise RuntimeError(f"Cleaned historical report changed: {item['path']}")
        for script in ("build_scale_ablation_report.py", "build_uniform_results_report.py",
                       "package_uniform_final_results.py", "verify_uniform_final_results.py"):
            run(script)
        # Confirm the ZIP carries the exact deletion receipt, in addition to
        # the verifier's complete model/row/metric/provenance/CRC checks.
        from zipfile import ZipFile
        archive_path = ROOT / "outputs/MAOMAO_final_results_20260926.zip"
        with ZipFile(archive_path) as archive:
            member = "MAOMAO_uniform_90_10/scale_ablations/legacy_removal.json"
            if archive.read(member) != (OUT / "legacy_removal.json").read_bytes():
                raise RuntimeError("Final ZIP has a stale removal record")
        proof = {"status": "completed", "verified_utc": utc(),
                 "experiments": result, "legacy_removal": receipt,
                 "report_sha256": sha256(ROOT / "docs/MAOMAO_V5_FINAL_RESULTS.md"),
                 "archive_sha256": sha256(archive_path),
                 "legacy_removal_in_archive_verified": True}
        write(OUT / "delivery_verification.json", proof)
        state.update(status="completed", stage="all_experiments_and_delivery_verified",
                     supervisor_pid=None, child_pid=None, delivery_verified_utc=utc())
        write(OUT / "queue_status.json", state)
        print("Verified all new comparisons, final report/ZIP, and removal of superseded results.")


if __name__ == "__main__":
    main()
