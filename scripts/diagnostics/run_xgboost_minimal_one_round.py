#!/usr/bin/env python3
"""Try one deliberately tiny XGBoost round on the full patient-disjoint train split."""
from __future__ import annotations

import gc
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.diagnostics.fit_fullscale_classical_baselines import report_logits

DATA = ROOT / "outputs/final_experiment_results_20260923/classical_full_scale"
OUT = ROOT / "outputs/final_experiment_results_20260923/classical_ml_fullscale/xgboost/minimal_one_round_20260927"


def DiskMemmapIter(x, y, batch_rows, cache_prefix):
    class _Iterator(xgb.DataIter):
        def __init__(self):
            self.x, self.y, self.batch_rows, self.offset = x, y, batch_rows, 0
            super().__init__(cache_prefix=str(cache_prefix), release_data=True,
                             on_host=False, min_cache_page_bytes=1 << 28)

        def reset(self):
            self.offset = 0

        def next(self, input_data):
            if self.offset >= len(self.x):
                return 0
            end = min(len(self.x), self.offset + self.batch_rows)
            input_data(data=np.asarray(self.x[self.offset:end], dtype=np.float32),
                       label=np.asarray(self.y[self.offset:end], dtype=np.float32))
            self.offset = end
            if end == len(self.x) or end % (self.batch_rows * 100) == 0:
                print(f"external-cache rows={end}/{len(self.x)}", flush=True)
            return 1
    return _Iterator()


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    history_path = OUT / "attempt_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else {"attempts": []}
    attempt = len(history.get("attempts", [])) + 1
    attempt_record = {"attempt": attempt, "status": "starting", "fit_started": False,
                      "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "external_cache": "disk"}
    history.setdefault("attempts", []).append(attempt_record)
    write_json(history_path, history)
    state = {"model": "xgboost_minimal_one_round", "status": "starting",
             "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "timeout_seconds": None, "attempt": attempt, "rounds": 1,
             "parameters": {"objective": "binary:logistic", "tree_method": "hist",
                            "multi_strategy": "one_output_per_tree", "max_depth": 1,
                            "max_bin": 16, "eta": 0.1, "subsample": 1.0,
                            "colsample_bytree": 0.01, "nthread": 4, "seed": 42},
             "data_scope": "all valid train target rows; all patient-disjoint validation target rows",
             "external_cache": "disk"}
    status_path = OUT / "status.json"
    write_json(status_path, state)
    cache = OUT / "cache"
    cache.mkdir(exist_ok=True)
    logits_path = OUT / "validation_logits.npy"
    try:
        x_train = np.load(DATA / "train_X.npy", mmap_mode="r")
        y_train = np.load(DATA / "train_y.npy", mmap_mode="r")
        x_val = np.load(DATA / "validation_X.npy", mmap_mode="r")
        y_val = np.load(DATA / "validation_y.npy", mmap_mode="r")
        metadata = json.loads((ROOT / "data/perioperative_event_sequences_v5_richctx_static7/event_sequence_meta.json").read_text())
        outcomes = metadata["outcome_vocabulary"]
        if (x_train.shape[0], y_train.shape[0], x_val.shape[0], y_val.shape[0]) != (14_128_539, 14_128_539, 1_563_972, 1_563_972):
            raise RuntimeError("Unexpected split row counts; refusing to run on a different scope")
        if x_train.shape[1] != x_val.shape[1] or y_train.shape[1] != y_val.shape[1] or y_train.shape[1] != len(outcomes):
            raise RuntimeError("Feature or outcome dimensions do not match")
        state.update({"status": "building_train_matrix", "train_rows": len(x_train),
                      "validation_rows": len(x_val), "feature_count": x_train.shape[1],
                      "outcome_count": y_train.shape[1], "xgboost_version": xgb.__version__,
                      "split": "patient-disjoint 90:10, seed 42"})
        write_json(status_path, state)
        train_iter = DiskMemmapIter(x_train, y_train, 16_384, cache / "train")
        dtrain = xgb.ExtMemQuantileDMatrix(train_iter, max_bin=16, nthread=4)
        state.update({"status": "building_validation_matrix", "elapsed_seconds": round(time.time() - started, 2)})
        write_json(status_path, state)
        valid_iter = DiskMemmapIter(x_val, y_val, 16_384, cache / "validation")
        dvalid = xgb.ExtMemQuantileDMatrix(valid_iter, max_bin=16, ref=dtrain, nthread=4)
        params = {"objective": "binary:logistic", "tree_method": "hist",
                  "multi_strategy": "one_output_per_tree", "max_depth": 1,
                  "eta": 0.1, "subsample": 1.0, "colsample_bytree": 0.01,
                  "eval_metric": "logloss", "max_bin": 16, "nthread": 4,
                  "seed": 42, "num_target": int(y_train.shape[1])}
        state.update({"status": "fitting_one_round", "elapsed_seconds": round(time.time() - started, 2)})
        attempt_record.update({"status": "fitting_one_round", "fit_started": True,
                               "fit_started_elapsed_seconds": round(time.time() - started, 2)})
        write_json(history_path, history)
        write_json(status_path, state)
        booster = xgb.train(params, dtrain, num_boost_round=1,
                            evals=[(dvalid, "validation")], verbose_eval=True)
        model_path = OUT / "xgboost_minimal_one_round.json"
        booster.save_model(str(model_path))
        state.update({"status": "scoring_full_validation", "elapsed_seconds": round(time.time() - started, 2)})
        write_json(status_path, state)
        logits = np.lib.format.open_memmap(logits_path, mode="w+", dtype="float32",
                                           shape=(len(x_val), y_train.shape[1]))
        for begin in range(0, len(x_val), 16_384):
            end = min(len(x_val), begin + 16_384)
            dm = xgb.DMatrix(np.asarray(x_val[begin:end], dtype=np.float32), nthread=4)
            logits[begin:end] = booster.predict(dm, output_margin=True)
            print(f"validation rows={end}/{len(x_val)}", flush=True)
        logits.flush()
        state.update({"status": "calculating_metrics", "elapsed_seconds": round(time.time() - started, 2)})
        write_json(status_path, state)
        report = report_logits(logits, y_val, outcomes, "xgboost_minimal_one_round_fullscale", 200, 4290)
        report.update({"train_rows_full": len(x_train), "validation_rows": len(x_val),
                       "epochs_or_rounds": 1, "xgboost_version": xgb.__version__,
                       "split": "patient-disjoint 90:10, seed 42", "parameters": params,
                       "feature_sampling_note": "colsample_bytree=0.01; row subsampling disabled; training data includes every valid training row"})
        write_json(OUT / "metrics.json", report)
        state.update({"status": "completed", "elapsed_seconds": round(time.time() - started, 2),
                      "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "metrics_file": "metrics.json", "model_file": model_path.name})
        attempt_record.update({"status": "completed", "elapsed_seconds": state["elapsed_seconds"],
                               "metrics_file": "metrics.json"})
        write_json(history_path, history)
        write_json(status_path, state)
        del logits
        logits_path.unlink(missing_ok=True)
        del booster, dtrain, dvalid, train_iter, valid_iter
        gc.collect()
        shutil.rmtree(cache, ignore_errors=True)
        print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
    except BaseException as exc:
        state.update({"status": "failed", "elapsed_seconds": round(time.time() - started, 2),
                      "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "error": f"{type(exc).__name__}: {exc}"})
        attempt_record.update({"status": "failed", "elapsed_seconds": state["elapsed_seconds"],
                               "error": state["error"]})
        write_json(history_path, history)
        write_json(status_path, state)
        raise


if __name__ == "__main__":
    main()
