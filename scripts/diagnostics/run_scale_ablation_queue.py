"""Serial, resumable full-data experiment queue and final result promotion."""
from __future__ import annotations

import fcntl
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.diagnostics.scale_ablation_scope import *


def utc():
    return datetime.now(timezone.utc).isoformat()


def evaluation_complete(name):
    names = ("reference", "reference_vocab_50", "reference_vocab_100", "reference_vocab_150") if name == "reference" else (name,)
    manifest_hash = sha256(OUT / "manifest.json")
    checkpoint = REFERENCE / "best_model.pt" if name == "reference" else OUT / "runs" / name / "best_model.pt"
    if not checkpoint.exists():
        return False
    checkpoint_hash = sha256(checkpoint)
    for item in names:
        metrics_path = OUT / "metrics" / f"{item}.json"
        time_path = OUT / "metrics" / f"{item}_time_scales.json"
        if not metrics_path.exists() or not time_path.exists():
            return False
        metrics, time = read(metrics_path), read(time_path)
        if (metrics.get("status") != "completed" or metrics.get("checkpoint_sha256") != checkpoint_hash or
                metrics.get("manifest_sha256") != manifest_hash or
                time.get("checkpoint_sha256") != checkpoint_hash or
                time.get("manifest_sha256") != manifest_hash or
                time.get("output_indices") != metrics.get("output_indices") or
                time.get("target_time_sha256") != metrics.get("target_time_sha256") or
                time.get("bootstrap_repeats") != 200 or
                time.get("pooled_time_mae_reported") is not False or
                not all(key + "_95ci" in metrics for key in METRICS)):
            return False
    return True


def live_worker():
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except OSError:
            continue
        if (str(ROOT / "scripts/diagnostics/evaluate_scale_ablation.py") in args or
                (str(OUT / "runs") in args and "scripts/train.py" in args)):
            return {"pid": int(proc.name), "command": args}
    return None


def run_stage(command, label, state):
    log = OUT / "logs" / f"{label}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", PYTHONUNBUFFERED="1")
    with log.open("a") as handle:
        handle.write(f"\nSTART {utc()} {command}\n"); handle.flush()
        child = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=environment)
        state.update(status="running", stage=label, child_pid=child.pid, child_command=command, stage_started_utc=utc())
        write(OUT / "queue_status.json", state)
        print(f"START {label} pid={child.pid} log={log}", flush=True)
        result = child.wait()
        handle.write(f"\nEND {utc()} returncode={result}\n")
    state.update(child_pid=None, stage_returncode=result, updated_utc=utc())
    write(OUT / "queue_status.json", state)
    print(f"END {label} returncode={result}", flush=True)
    if result:
        state.update(status="failed", failed_stage=label)
        write(OUT / "queue_status.json", state)
        raise RuntimeError(f"{label} failed; inspect {log} before resuming")


def legacy_report_edits(names):
    """Plan removal of obsolete size/vocabulary chapters, retaining other work."""
    root = ROOT / "outputs/archive_removed_experiments_20260923/ordinary_transformer_baseline"
    pattern = re.compile("|".join(re.escape(name) for name in sorted(names)))
    headings = {"### 模型规模与词表消融", "### 消融模型外部验证",
                "### 已完成 full-vocabulary 外部指标", "### 外部验证扩展指标（Full event set）",
                "### 外部验证扩展指标（Core event set）"}
    plans = []
    for filename in ("MAOMAO_V5_FINAL_RESULTS.md", "MAOMAO_V5_FINAL_RESULTS_snapshot.md", "MAOMAO_V5_TODO.md"):
        path = root / filename
        if not path.exists():
            continue
        original = path.read_text()
        if not pattern.search(original):
            continue
        kept, removed = [], []
        for section in re.split(r"(?m)(?=^#{1,6} )", original):
            heading = section.splitlines()[0] if section.splitlines() else ""
            if filename == "MAOMAO_V5_TODO.md":
                if heading in {"### P0：完成当前消融训练", "### P0：统一外部验证矩阵"}:
                    removed.append(heading)
                    continue
                section = "".join(line for line in section.splitlines(keepends=True)
                                  if not pattern.search(line) and
                                  "outputs/ablations_v5_final" not in line and
                                  "outputs/ablation_external_v5" not in line and
                                  not line.startswith("三个词表实验由独立进程并行运行"))
            elif pattern.search(section):
                if heading not in headings:
                    raise RuntimeError(f"Unrecognized legacy report chapter; preserve it for review: {path}: {heading}")
                # Only the six obsolete models may have data rows in these chapters.
                for line in section.splitlines():
                    if not line.startswith("|"):
                        continue
                    first = line.split("|")[1].strip()
                    if first not in names and first != "模型" and not re.fullmatch(r"[- :]+", first):
                        raise RuntimeError(f"Unrelated result in legacy chapter; preserve it: {path}: {first}")
                removed.append(heading)
                continue
            kept.append(section)
        replacement = "".join(kept)
        if pattern.search(replacement):
            raise RuntimeError(f"Legacy model rows remain in report: {path}")
        plans.append({"path": path, "text": replacement, "before_sha256": sha256(path),
                      "removed_sections": removed})
    return plans


def delete_superseded(state):
    """Only remove the old size/vocabulary result folders after promotion."""
    # A named allowlist prevents broad deletion of unrelated clinical or module
    # experiments. Keep a small inventory of paths/digests proving replacement.
    names = {"model_small_vocab_full", "model_current_vocab_full", "model_large_vocab_full",
             "model_current_vocab_50", "model_current_vocab_100", "model_current_vocab_150"}
    report_plans = legacy_report_edits(names)
    candidates = [ROOT / "outputs/ablations_v5_final", ROOT / "outputs/ablation_external_v5"]
    # The earliest one-epoch plan predates the final ten-epoch matrix. Its
    # directory now contains only that obsolete six-model manifest. Refuse
    # removal if unrelated work has appeared there before final promotion.
    legacy_plan = ROOT / "outputs/ablations_v5"
    if legacy_plan.exists():
        if ({path.name for path in legacy_plan.iterdir()} != {"ablation_manifest.json"} or
                {item.get("name") for item in read(legacy_plan / "ablation_manifest.json")} != names):
            raise RuntimeError(f"Unexpected contents in legacy plan; preserve for review: {legacy_plan}")
        candidates.append(legacy_plan)
    archive = ROOT / "outputs/archive_removed_experiments_20260923"
    for path in archive.rglob("*"):
        if path.is_dir() and (path.name in names or path.name in {"existing_ablation_runs", "existing_ablation_summary"}):
            candidates.append(path)
        elif path.is_file() and path.name in {"ablation_summary.json", "ablation_summary.csv"}:
            if any(name in path.read_text(errors="ignore") for name in names):
                candidates.append(path)
    roots = []
    for path in sorted(set(candidates), key=lambda p: len(p.parts)):
        if path.exists() and not any(path.is_relative_to(parent) for parent in roots):
            roots.append(path)
    inventory = []
    receipt_path = OUT / "legacy_removal.json"
    edited_reports = []
    if receipt_path.exists():
        prior = read(receipt_path)
        inventory.extend(prior.get("directories", []))
        edited_reports.extend(prior.get("edited_reports", []))
    for path in roots:
        records = []
        items = list(path.rglob("*")) if path.is_dir() else [path]
        for item in items:
            if item.is_file() and item.suffix in {".json", ".csv"}:
                records.append({"path": str(item.relative_to(path)) if path.is_dir() else item.name, "sha256": sha256(item)})
        inventory.append({"directory": str(path), "type": "directory" if path.is_dir() else "file",
                          "files": sum(1 for p in items if p.is_file()), "metadata": records})
    inventory = list({record["directory"]: record for record in inventory}.values())
    write(OUT / "legacy_removal.json", {"status": "deleting", "verified_replacement": read(OUT / "verification.json"),
                                        "directories": inventory, "edited_reports": edited_reports})
    for path in roots:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    for plan in report_plans:
        path = plan["path"]
        if sha256(path) != plan["before_sha256"]:
            raise RuntimeError(f"Historical report changed during cleanup: {path}")
        temporary = path.with_name(path.name + ".cleanup.tmp")
        temporary.write_text(plan["text"])
        temporary.replace(path)
        edited_reports.append({"path": str(path), "before_sha256": plan["before_sha256"],
                               "after_sha256": sha256(path), "removed_sections": plan["removed_sections"]})
    edited_reports = list({record["path"]: record for record in edited_reports}.values())
    receipt = read(OUT / "legacy_removal.json")
    receipt.update(status="completed", finished_utc=utc(), edited_reports=edited_reports,
                   all_directories_absent=all(not Path(item["directory"]).exists() for item in inventory),
                   all_edited_reports_clean=all(Path(item["path"]).exists() and sha256(Path(item["path"])) == item["after_sha256"]
                                               and not any(name in Path(item["path"]).read_text() for name in names)
                                               for item in edited_reports))
    write(OUT / "legacy_removal.json", receipt)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / ".queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("A live supervisor holds this experiment queue; no duplicate launched.")
            return
        worker = live_worker()
        if worker:
            print(f"Existing worker still live: {worker}; wait before resuming queue.")
            return
        state = read(OUT / "queue_status.json") if (OUT / "queue_status.json").exists() else {"models": {}}
        state.update(supervisor_pid=os.getpid(), updated_utc=utc())
        run_stage([sys.executable, str(ROOT / "scripts/diagnostics/prepare_scale_ablations.py")], "prepare", state)
        manifest = read(OUT / "manifest.json")
        for path, evidence in manifest["source_evidence"].items():
            p = Path(path)
            if p.stat().st_mtime_ns != evidence["mtime_ns"] or p.stat().st_size != evidence["size_bytes"]:
                raise RuntimeError(f"Frozen experiment source changed: {p}")
        # Train one smaller variant first; reusing the reference never retrains it.
        for name in NAMES:
            log = OUT / "runs" / name / "train.log"
            complete_training = log.exists() and "Training finished" in log.read_text()
            if not complete_training:
                run_stage(training_command(name, resume=(log.parent / "checkpoint_latest.pt").exists()), f"{name}_training", state)
            state["models"][name] = {"training": "completed", "evaluation": "running"}
            write(OUT / "queue_status.json", state)
            if name == "vocab_50" and not evaluation_complete("reference"):
                run_stage([sys.executable, str(ROOT / "scripts/diagnostics/evaluate_scale_ablation.py"), "--name", "reference"], "reference_evaluation", state)
            if not evaluation_complete(name):
                run_stage([sys.executable, str(ROOT / "scripts/diagnostics/evaluate_scale_ablation.py"), "--name", name], f"{name}_evaluation", state)
            state["models"][name] = {"training": "completed", "evaluation": "completed", "finished_utc": utc()}
            write(OUT / "queue_status.json", state)
            run_stage([sys.executable, str(ROOT / "scripts/diagnostics/build_scale_ablation_report.py")], f"{name}_report", state)
        if not evaluation_complete("reference"):
            run_stage([sys.executable, str(ROOT / "scripts/diagnostics/evaluate_scale_ablation.py"), "--name", "reference"], "reference_evaluation", state)
        run_stage([sys.executable, str(ROOT / "scripts/diagnostics/verify_scale_ablations.py"), "--require-complete"], "experiment_verification", state)
        for script in ("build_scale_ablation_report.py", "build_uniform_results_report.py", "package_uniform_final_results.py", "verify_uniform_final_results.py"):
            command = [sys.executable, str(ROOT / "scripts/diagnostics" / script)]
            if script.startswith("verify_"):
                command.append("--require-complete")
            run_stage(command, script.removesuffix(".py"), state)
        delete_superseded(state)
        run_stage([sys.executable, str(ROOT / "scripts/diagnostics/package_uniform_final_results.py")], "package_after_old_removal", state)
        run_stage([sys.executable, str(ROOT / "scripts/diagnostics/verify_uniform_final_results.py"), "--require-complete"], "verify_after_old_removal", state)
        shutil.rmtree(OUT / "cache", ignore_errors=True)
        state.update(status="completed", finished_utc=utc(), supervisor_pid=None, stage="all_experiments_and_report_complete")
        write(OUT / "queue_status.json", state)
        print("COMPLETED: all seven experiments verified, old comparison removed, final MD and ZIP rebuilt.", flush=True)


if __name__ == "__main__":
    main()
