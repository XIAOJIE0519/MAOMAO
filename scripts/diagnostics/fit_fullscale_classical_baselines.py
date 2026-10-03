#!/usr/bin/env python3
"""Fit the classical baselines on all internal train rows, then score all val rows."""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from maomao.evaluation.event_metrics import event_metric_report_with_subsample_ci


class MemmapIter:
    """Lazy XGBoost iterator over disk-backed dense multi-label arrays."""
    def __new__(cls, x, y, batch_rows, cache_prefix, xgb):
        class _Iterator(xgb.DataIter):
            def __init__(self):
                self.x, self.y, self.batch_rows = x, y, batch_rows
                self.offset = 0
                super().__init__(cache_prefix=str(cache_prefix), release_data=True,
                                 on_host=True, min_cache_page_bytes=1 << 28)

            def reset(self):
                self.offset = 0

            def next(self, input_data):
                if self.offset >= len(self.x):
                    return 0
                end = min(len(self.x), self.offset + self.batch_rows)
                input_data(data=np.asarray(self.x[self.offset:end], dtype=np.float32),
                           label=np.asarray(self.y[self.offset:end], dtype=np.float32))
                self.offset = end
                return 1
        return _Iterator()


def report_logits(logits: np.ndarray, targets: np.ndarray, outcomes: list[str],
                  model_name: str, repeats: int, seed: int) -> dict:
    report = event_metric_report_with_subsample_ci(
        torch.from_numpy(np.asarray(logits, dtype=np.float32)),
        torch.from_numpy(np.array(targets, dtype=np.uint8, copy=True)), outcomes,
        bootstrap_repeats=repeats, bootstrap_seed=seed, max_ci_rows=100_000)
    report["model"] = model_name
    report["rows"] = int(len(targets))
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=Path("outputs/final_experiment_results_20260923/classical_full_scale"))
    ap.add_argument("--output_dir", type=Path, default=Path("outputs/final_experiment_results_20260923/baseline_metrics"))
    ap.add_argument("--epochs", type=int, default=7)
    ap.add_argument("--batch_rows", type=int, default=8192)
    ap.add_argument("--xgb_batch_rows", type=int, default=65536)
    ap.add_argument("--bootstrap_repeats", type=int, default=200)
    ap.add_argument("--xgb_rounds", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = args.data_dir
    x_train = np.load(data / "train_X.npy", mmap_mode="r")
    y_train = np.load(data / "train_y.npy", mmap_mode="r")
    x_val = np.load(data / "validation_X.npy", mmap_mode="r")
    y_val = np.load(data / "validation_y.npy", mmap_mode="r")
    outcomes = json.loads((ROOT / "data/perioperative_event_sequences_v5_richctx_static7/event_sequence_meta.json").read_text())["outcome_vocabulary"]
    if len(outcomes) != y_train.shape[1] or y_val.shape[1] != y_train.shape[1]:
        raise RuntimeError("Outcome dimensions do not match the full-scale arrays")
    protocol = {
        "training_rows": int(len(x_train)), "validation_rows": int(len(x_val)),
        "outcomes": int(y_train.shape[1]), "feature_count": int(x_train.shape[1]),
        "training_epochs": args.epochs, "internal_split": "patient-disjoint 90:10, seed 42",
        "row_sampling": "none; every valid target position in both patient splits",
        "ci": "full-cohort point estimates; uniform row-subset bootstrap CIs, details per metric JSON",
    }
    (args.output_dir / "fullscale_classical_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    # The deliberately simple single-variable baseline also uses all rows.
    # This is the sole baseline allowed to reduce each row to its current token.
    token_vocabulary = json.loads((ROOT / "data/perioperative_event_sequences_v5_richctx_static7/token_vocabulary.json").read_text())
    num_tokens = len(token_vocabulary)
    global_counts = np.zeros(y_train.shape[1], dtype=np.float64)
    token_counts = np.zeros(num_tokens, dtype=np.float64)
    token_targets = np.zeros((num_tokens, y_train.shape[1]), dtype=np.float64)
    for start in range(0, len(x_train), args.xgb_batch_rows):
        end = min(len(x_train), start + args.xgb_batch_rows)
        ids = np.clip(np.rint(np.asarray(x_train[start:end, 0]) * (num_tokens - 1)).astype(np.int64),
                      0, num_tokens - 1)
        labels = np.asarray(y_train[start:end], dtype=np.uint8)
        global_counts += labels.sum(0)
        token_counts += np.bincount(ids, minlength=num_tokens)
        row_ids, col_ids = np.nonzero(labels)
        np.add.at(token_targets, (ids[row_ids], col_ids), 1.0)
    global_rate = (global_counts + 1.0) / (len(y_train) + 2.0)
    token_rates = (token_targets + global_rate[None, :]) / (token_counts[:, None] + 1.0)
    test_ids = np.clip(np.rint(np.asarray(x_val[:, 0]) * (num_tokens - 1)).astype(np.int64),
                       0, num_tokens - 1)
    uni_probs = np.clip(token_rates[test_ids], 1e-6, 1 - 1e-6)
    uni_logits = np.log(uni_probs) - np.log1p(-uni_probs)
    uni_report = report_logits(uni_logits, y_val, outcomes, "univariate_last_token_fullscale",
                               args.bootstrap_repeats, 4210)
    (args.output_dir / "univariate_fullscale_internal.json").write_text(json.dumps(uni_report, ensure_ascii=False, indent=2) + "\n")
    del uni_probs, uni_logits, token_rates, token_targets

    # Full-scale one-vs-rest logistic regression, optimized in batches over the
    # complete 14M-row matrix. Moments are computed from every training row.
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    scaler = StandardScaler(copy=False)
    for start in range(0, len(x_train), args.xgb_batch_rows):
        scaler.partial_fit(np.asarray(x_train[start:start + args.xgb_batch_rows], dtype=np.float32))
    np.savez(args.output_dir / "logistic_scaler.npz", mean=scaler.mean_, scale=scaler.scale_)
    torch.manual_seed(42)
    linear = torch.nn.Linear(x_train.shape[1], y_train.shape[1]).to(device)
    torch.nn.init.zeros_(linear.weight)
    torch.nn.init.zeros_(linear.bias)
    optimizer = torch.optim.AdamW(linear.parameters(), lr=3e-4, weight_decay=0.1)
    scaler_mean = torch.as_tensor(scaler.mean_, device=device, dtype=torch.float32)
    scaler_scale = torch.as_tensor(scaler.scale_, device=device, dtype=torch.float32).clamp_min(1e-8)
    losses = []
    start_time = time.time()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for start in range(0, len(x_train), args.batch_rows):
            end = min(len(x_train), start + args.batch_rows)
            xb = torch.tensor(np.asarray(x_train[start:end], dtype=np.float32), device=device)
            yb = torch.tensor(np.asarray(y_train[start:end], dtype=np.float32), device=device)
            xb.sub_(scaler_mean).div_(scaler_scale)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                linear(xb), yb, reduction="sum") / (end - start)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * (end - start)
            if (start // args.batch_rows) % 200 == 0:
                print(f"logistic epoch={epoch + 1}/{args.epochs} rows={end}/{len(x_train)} loss={float(loss):.6f}", flush=True)
        losses.append(total_loss / len(x_train))
        print(f"logistic epoch={epoch + 1} mean_loss={losses[-1]:.6f} elapsed_s={time.time()-start_time:.1f}", flush=True)
    torch.save({"weight": linear.weight.detach().cpu(), "bias": linear.bias.detach().cpu(),
                "training_rows": len(x_train), "epochs": args.epochs, "losses": losses},
               args.output_dir / "logistic_fullscale.pt")
    logistic_logits = np.lib.format.open_memmap(args.output_dir / "logistic_validation_logits.npy",
                                                mode="w+", dtype="float32",
                                                shape=(len(x_val), y_val.shape[1]))
    linear.eval()
    with torch.inference_mode():
        for begin in range(0, len(x_val), args.batch_rows):
            end = min(len(x_val), begin + args.batch_rows)
            xb = torch.tensor(np.asarray(x_val[begin:end], dtype=np.float32), device=device)
            xb.sub_(scaler_mean).div_(scaler_scale)
            logistic_logits[begin:end] = linear(xb).float().cpu().numpy()
    logistic_logits.flush()
    logistic_report = report_logits(logistic_logits, y_val, outcomes, "logistic_regression_fullscale",
                                    args.bootstrap_repeats, 4211)
    (args.output_dir / "logistic_fullscale_internal.json").write_text(json.dumps(logistic_report, ensure_ascii=False, indent=2) + "\n")
    print("logistic full-cohort validation scored", flush=True)

    # Native XGBoost multi-label classification uses binary relevance and one
    # output tree per label. The external-memory quantile matrix avoids loading
    # the full feature matrix into RAM.
    import xgboost as xgb
    train_iter = MemmapIter(x_train, y_train, args.xgb_batch_rows,
                            args.data_dir / "xgb_train_cache", xgb)
    dtrain = xgb.ExtMemQuantileDMatrix(train_iter, max_bin=256, nthread=16)
    valid_iter = MemmapIter(x_val, y_val, args.xgb_batch_rows,
                            args.data_dir / "xgb_validation_cache", xgb)
    dvalid = xgb.ExtMemQuantileDMatrix(valid_iter, max_bin=256, ref=dtrain, nthread=16)
    params = {
        "objective": "binary:logistic", "tree_method": "hist",
        "multi_strategy": "one_output_per_tree", "max_depth": 4,
        "eta": 0.08, "subsample": 1.0, "colsample_bytree": 1.0,
        "eval_metric": "logloss", "max_bin": 256, "nthread": 16,
        "seed": 42, "num_target": y_train.shape[1],
    }
    booster = xgb.train(params, dtrain, num_boost_round=args.xgb_rounds,
                        evals=[(dvalid, "validation")], verbose_eval=1)
    booster.save_model(str(args.output_dir / "xgboost_fullscale.json"))
    xgb_logits = np.lib.format.open_memmap(args.output_dir / "xgboost_validation_logits.npy",
                                          mode="w+", dtype="float32",
                                          shape=(len(x_val), y_val.shape[1]))
    for begin in range(0, len(x_val), args.xgb_batch_rows):
        end = min(len(x_val), begin + args.xgb_batch_rows)
        xgb_logits[begin:end] = booster.predict(xgb.DMatrix(
            np.asarray(x_val[begin:end], dtype=np.float32), nthread=16),
            output_margin=True)
        print(f"xgboost validation rows={end}/{len(x_val)}", flush=True)
    xgb_logits.flush()
    xgb_report = report_logits(xgb_logits, y_val, outcomes, "xgboost_fullscale",
                               args.bootstrap_repeats, 4212)
    (args.output_dir / "xgboost_fullscale_internal.json").write_text(json.dumps(xgb_report, ensure_ascii=False, indent=2) + "\n")
    manifest = {**protocol, "status": "complete", "logistic_losses": losses,
                "xgboost_rounds": args.xgb_rounds, "elapsed_s": time.time() - start_time}
    (args.output_dir / "fullscale_classical_completion.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
