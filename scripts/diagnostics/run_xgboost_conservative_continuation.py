#!/usr/bin/env python3
"""Continue the saved full-row booster to at most three rounds, under guards."""
from __future__ import annotations
import argparse
import copy
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.diagnostics.uniform_result_scope import BASE, XGB_ROOT, XGB_CANDIDATE as OUT, current_xgboost_model, sha256
def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def DiskMemmapIter(x, y, batch_rows, cache_prefix):
    import xgboost as xgb
    class Iterator(xgb.DataIter):
        def __init__(self):
            self.offset = 0
            super().__init__(cache_prefix=str(cache_prefix), release_data=True,
                             on_host=False, min_cache_page_bytes=1 << 28)
        def reset(self):
            self.offset = 0
        def next(self, input_data):
            if self.offset >= len(x):
                return 0
            end = min(len(x), self.offset + batch_rows)
            input_data(data=np.asarray(x[self.offset:end], dtype=np.float32),
                       label=np.asarray(y[self.offset:end], dtype=np.float32))
            self.offset = end
            if end == len(x) or end % (batch_rows * 100) == 0:
                print(f"external-cache rows={end}/{len(x)}", flush=True)
            return 1
    return Iterator()

DATA = BASE / "classical_full_scale"
PARAMS = dict(objective="binary:logistic", tree_method="hist", multi_strategy="one_output_per_tree",
              max_depth=1, max_bin=16, eta=0.1, subsample=1.0, colsample_bytree=0.01,
              nthread=8, seed=42, num_target=210)


def state(**updates):
    path = OUT / "status.json"
    value = json.loads(path.read_text()) if path.exists() else {}
    value.update(updates)
    write_json(path, value)


def single_target_model(full, target):
    result = copy.deepcopy(full)
    learner = result["learner"]
    parameters = learner["learner_model_param"]
    bases = json.loads(parameters["base_score"])
    parameters["base_score"] = json.dumps([bases[target]])
    parameters["num_target"] = "1"
    model = learner["gradient_booster"]["model"]
    model["trees"] = [tree for tree, group in zip(model["trees"], model["tree_info"]) if group == target]
    for index, tree in enumerate(model["trees"]):
        tree["id"] = index
    model["tree_info"] = [0] * len(model["trees"])
    model["iteration_indptr"] = list(range(len(model["trees"]) + 1))
    model["gbtree_model_param"]["num_trees"] = str(len(model["trees"]))
    return result


def merge_targets(original, targets, rounds):
    result = copy.deepcopy(original)
    model = result["learner"]["gradient_booster"]["model"]
    trees, groups = [], []
    for round_index in range(rounds):
        for target in range(210):
            tree = copy.deepcopy(targets[target]["learner"]["gradient_booster"]["model"]["trees"][round_index])
            tree["id"] = len(trees)
            trees.append(tree)
            groups.append(target)
    model.update(trees=trees, tree_info=groups, iteration_indptr=[210*i for i in range(rounds+1)])
    model["gbtree_model_param"]["num_trees"] = str(len(trees))
    # Native saved trees from round one must be preserved verbatim.
    if model["trees"][:210] != original["learner"]["gradient_booster"]["model"]["trees"]:
        raise RuntimeError("Warm-start trees changed while combining independent target models")
    return result


def train():
    import xgboost as xgb
    old = current_xgboost_model()
    original = json.loads(old.read_text())
    initial = xgb.Booster(params={"nthread":8})
    initial.load_model(str(old))
    start_rounds = initial.num_boosted_rounds()
    if start_rounds != 1:
        raise RuntimeError("This label-serial continuation expects exactly one saved round")
    del initial
    state(status="building_train_matrix", rounds=start_rounds, target_rounds=3,
          training_rows=14_128_539, validation_rows=1_563_972,
          parameters=PARAMS, source_model=str(old), source_model_sha256=sha256(old),
          split="patient-disjoint 90:10, seed 42", row_subsampling=False,
          training_limit_seconds=3600, per_round_limit_seconds=2700,
          anonymous_memory_limit_gib=40, minimum_available_memory_gib=8,
          training_implementation="210 independent binary targets in series sharing one external quantile matrix; native trees combined in original target order",
          target_seed_rule="42 + target_index * 101 + boosting_round * 1009; chosen before validation",
          memory_guard_note="RssAnon + RssShmem plus system MemAvailable; reclaimable file pages are logged separately")
    x = np.load(DATA / "train_X.npy", mmap_mode="r")
    y = np.load(DATA / "train_y.npy", mmap_mode="r")
    if x.shape != (14_128_539,1549) or y.shape != (14_128_539,210):
        raise RuntimeError("Full training array contract mismatch")
    cache = OUT / "cache"
    cache.mkdir(exist_ok=True)
    iterator = DiskMemmapIter(x, y[:,0], 16_384, cache / "train")
    matrix = xgb.ExtMemQuantileDMatrix(iterator,max_bin=16,nthread=8)
    # Only the target vector and its gradients occupy memory; the full feature matrix is reused.
    targets = [single_target_model(original,target) for target in range(210)]
    probe = xgb.DMatrix(np.asarray(x[:8],dtype=np.float32),nthread=8)
    for round_number in range(2,4):
        started = time.time()
        state(status="training", fitting_round=round_number,
              round_started_unix=started, completed_targets_in_round=0, total_targets=210)
        reference_margins = []
        for target in range(210):
            matrix.set_label(np.asarray(y[:,target],dtype=np.float32))
            booster = xgb.Booster(params={"nthread":8})
            booster.load_model(bytearray(json.dumps(targets[target]).encode()))
            parameters = dict(PARAMS,num_target=1,seed=42+target*101+round_number*1009)
            booster = xgb.train(parameters,matrix,num_boost_round=1,xgb_model=booster,verbose_eval=False)
            if booster.num_boosted_rounds() != round_number:
                raise RuntimeError("Single target did not add exactly one native round")
            targets[target] = json.loads(booster.save_raw(raw_format="json"))
            reference_margins.append(booster.predict(probe,output_margin=True))
            partial = OUT / "partial_target_models"
            partial.mkdir(exist_ok=True)
            write_json(partial / f"target_{target:03d}.json",targets[target])
            state(status="training", completed_targets_in_round=target+1,
                  round_elapsed_seconds=round(time.time()-started,1))
            print(f"round {round_number}/3: target {target+1}/210 uses all {len(y)} training rows",flush=True)
            del booster
            gc.collect()
        combined = merge_targets(original,targets,round_number)
        path = OUT / f"checkpoint_round_{round_number:03d}.json"
        temporary = OUT / f"checkpoint_round_{round_number:03d}.tmp.json"
        write_json(temporary,combined)
        native = xgb.Booster(params={"nthread":8})
        native.load_model(str(temporary))
        if native.num_boosted_rounds() != round_number:
            raise RuntimeError("Combined native model has incorrect round count")
        # Integrity check on training rows only; no holdout-guided model selection.
        if not np.allclose(native.predict(probe,output_margin=True),
                           np.stack(reference_margins,axis=1),rtol=1e-6,atol=1e-6):
            raise RuntimeError("Combined native model margins differ from per-target native margins")
        temporary.replace(path)
        state(status="round_saved", rounds=round_number,model_file=path.name,
              native_merge_margin_integrity_verified=True)
        print(f"saved full-row XGBoost round {round_number}/3",flush=True)
        del native
    state(status="trained",rounds=3)


def score():
    import xgboost as xgb
    import torch
    from scripts.diagnostics.fit_fullscale_classical_baselines import report_logits
    torch.set_num_threads(8)
    candidates = sorted(OUT.glob("checkpoint_round_[0-9][0-9][0-9].json"))
    if not candidates:
        raise RuntimeError("No additional complete round is available")
    source = candidates[-1]
    booster = xgb.Booster(params={"nthread": 8})
    booster.load_model(str(source))
    rounds = booster.num_boosted_rounds()
    if rounds < 2 or rounds > 3:
        raise RuntimeError("Unexpected continued round count")
    model = OUT / "xgboost.json"
    shutil.copy2(source, model)
    state(status="scoring_full_validation", rounds=rounds, model_file=model.name)
    x = np.load(DATA / "validation_X.npy", mmap_mode="r")
    y = np.load(DATA / "validation_y.npy", mmap_mode="r")
    if x.shape != (1_563_972, 1549) or y.shape != (1_563_972, 210):
        raise RuntimeError("Full validation array contract mismatch")
    path = OUT / "validation_logits.npy"
    logits = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=y.shape)
    for begin in range(0, len(x), 16_384):
        end = min(len(x), begin + 16_384)
        dm = xgb.DMatrix(np.asarray(x[begin:end], dtype=np.float32), nthread=8)
        logits[begin:end] = booster.predict(dm, output_margin=True)
        if end == len(x) or end % (16_384 * 25) == 0:
            print(f"validation rows={end}/{len(x)}", flush=True)
    logits.flush()
    del booster
    gc.collect()
    state(status="calculating_metrics")
    outcomes = json.loads((ROOT / "data/perioperative_event_sequences_v5_richctx_static7/event_sequence_meta.json").read_text())["outcome_vocabulary"]
    metrics = report_logits(logits, y, outcomes, "xgboost", 200, 4290)
    metrics.update(train_rows_full=14_128_539, validation_rows=1_563_972,
                   epochs_or_rounds=rounds, parameters=PARAMS,
                   model_sha256=sha256(model), split="patient-disjoint 90:10, seed 42",
                   feature_sampling_note="All valid training rows and full feature matrix used; each independent target retains colsample_bytree=0.01.",
                   training_implementation="label-serial native binary boosters combined as 210-target XGBoost; initial saved round retained",
                   target_seed_rule="42 + target_index * 101 + boosting_round * 1009",
                   binary_target_training_parameters=dict(PARAMS,num_target=1,seed="42 + target_index * 101 + boosting_round * 1009"),
                   native_merge_margin_integrity_verified=True)
    write_json(OUT / "metrics.json", metrics)
    state(status="completed", rounds=rounds, metrics_file="metrics.json",
          model_file=model.name, finished_unix=time.time())
    del logits
    path.unlink(missing_ok=True)


def available_gib():
    values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(values["MemAvailable"].strip().split()[0]) / 1024**2


def supervise(phase, limit, adopt_pid=None):
    with (OUT / f"{phase}.log").open("a") as log:
        env = dict(os.environ, OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
        previous = json.loads((OUT / "supervisor_status.json").read_text()) if adopt_pid else {}
        if adopt_pid:
            command = Path(f"/proc/{adopt_pid}/cmdline").read_bytes().replace(b"\0",b" ").decode()
            if "run_xgboost_conservative_continuation.py" not in command or "--phase train" not in command:
                raise RuntimeError("Refusing to adopt a process that is not the live training worker")
            worker_stat = Path(f"/proc/{adopt_pid}/stat").read_text().rsplit(")",1)[1].split()
            start_ticks = int(worker_stat[19])
            elapsed_before = float(Path("/proc/uptime").read_text().split()[0]) - start_ticks/os.sysconf("SC_CLK_TCK")
            child = None
            pid = adopt_pid
            state(per_round_limit_seconds=2700,
                  policy_adjustment_note="Measured label-serial throughput needs roughly 38 minutes per complete round; per-round cap raised to 45 minutes while total training remains 60 minutes. Live worker adopted, never restarted.")
        else:
            child = subprocess.Popen([sys.executable, "-u", __file__, "--phase", phase],
                                     cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     env=env, start_new_session=True)
            pid = child.pid
            elapsed_before = 0
            start_ticks = None
        def running():
            if child is not None:
                return child.poll() is None
            try:
                entries = Path(f"/proc/{pid}/stat").read_text().rsplit(")",1)[1].split()
                return entries[0] != "Z" and int(entries[19]) == start_ticks
            except FileNotFoundError:
                return False
        start = time.monotonic() - elapsed_before
        reason = None
        peak = previous.get("peak_rss_gib",0.0)
        peak_anon = previous.get("peak_anonymous_gib",0.0)
        while running():
            try:
                proc = Path(f"/proc/{pid}/status").read_text()
                rss = int(next(line for line in proc.splitlines() if line.startswith("VmRSS:")).split()[1]) / 1024**2
                peak = max(peak, rss)
                entries = {line.split(":")[0]: line.split()[1] for line in proc.splitlines() if line.startswith(("RssAnon:", "RssShmem:", "RssFile:"))}
                anon = (int(entries.get("RssAnon", 0)) + int(entries.get("RssShmem", 0))) / 1024**2
                peak_anon = max(peak_anon, anon)
            except (FileNotFoundError, StopIteration):
                rss = anon = 0
            current = json.loads((OUT / "status.json").read_text()) if (OUT / "status.json").exists() else {}
            if time.monotonic() - start > limit:
                reason = "time_limit"
            elif anon > 40 or available_gib() < 8:
                reason = "memory_guard"
            elif phase == "train" and current.get("status") == "training" and time.time() - current.get("round_started_unix", time.time()) > 2700:
                reason = "per_round_time_limit"
            write_json(OUT / "supervisor_status.json", dict(status="running", phase=phase,
                       pid=pid, elapsed_seconds=round(time.monotonic()-start, 1),
                       rss_gib=round(rss, 2), peak_rss_gib=round(peak, 2),
                       anonymous_gib=round(anon, 2), peak_anonymous_gib=round(peak_anon, 2),
                       available_memory_gib=round(available_gib(), 2)))
            if reason:
                os.killpg(pid, signal.SIGTERM)
                deadline = time.monotonic() + 10
                while running() and time.monotonic() < deadline:
                    time.sleep(0.5)
                if running():
                    os.killpg(pid, signal.SIGKILL)
                break
            time.sleep(2)
        if child is not None:
            code = child.wait()
        else:
            final_state = json.loads((OUT / "status.json").read_text())
            code = -15 if reason else (0 if final_state.get("status") == "trained" else None)
        record = dict(phase=phase, returncode=code, stop_reason=reason,
                      elapsed_seconds=round(time.monotonic()-start, 1), peak_rss_gib=round(peak, 2), peak_anonymous_gib=round(peak_anon, 2),
                      adopted_live_process=bool(adopt_pid), runtime_clock="original worker start from /proc; no budget reset" if adopt_pid else "monotonic since original launch")
        write_json(OUT / f"{phase}_guard_record.json", record)
        return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "score"))
    parser.add_argument("--adopt-train-pid", type=int)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase:
        return train() if args.phase == "train" else score()
    with (XGB_ROOT / ".continuation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUT / "metrics.json").exists():
            raise SystemExit("Completed metrics exist; refusing a duplicate run")
        if (OUT / "train_guard_record.json").exists():
            raise SystemExit("This bounded attempt already ran; refusing to repeat it")
        # The label-serial algorithm has bounded target buffers. Require generous
        # free memory at startup; guards remain active throughout training.
        while not args.adopt_train_pid and available_gib() < 40:
            state(status="waiting_available_memory", target_rounds=3, rounds=1)
            write_json(OUT / "supervisor_status.json", {"status":"waiting_available_memory"})
            time.sleep(30)
        train_record = supervise("train", 3600, args.adopt_train_pid)
        shutil.rmtree(OUT / "cache", ignore_errors=True)
        if not list(OUT.glob("checkpoint_round_[0-9][0-9][0-9].json")):
            state(status="stopped_no_extra_round", guard_record=train_record)
            write_json(OUT / "supervisor_status.json", {"status":"stopped_no_extra_round", **train_record})
            return
        score_record = supervise("score", 2700)
        if score_record["returncode"] or not (OUT / "metrics.json").exists():
            state(status="evaluation_failed", guard_record=score_record)
            write_json(OUT / "supervisor_status.json", {"status":"evaluation_failed", **score_record})
            return
        write_json(XGB_ROOT / "current_model.json", {"directory":str(OUT.relative_to(ROOT)),
                   "model_sha256":sha256(OUT / "xgboost.json"), "promoted_unix":time.time()})
        write_json(OUT / "supervisor_status.json", {"status":"completed", "rounds":json.loads((OUT / "status.json").read_text())["rounds"]})


if __name__ == "__main__":
    main()
