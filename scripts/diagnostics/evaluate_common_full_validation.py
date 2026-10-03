#!/usr/bin/env python3
"""Evaluate saved baselines on every row in the common patient-held-out 10%.

The original full-scale single-variable/logistic/MAOMAO reports are reused only
after verifying their evaluated row counts. Saved SVM and ANN checkpoints are
scored on the full cohort. One additional training-only pass creates modestly
refined SVM and ANN variants; the final 10% holdout is never used for tuning.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from maomao.evaluation.event_metrics import event_metric_report_with_subsample_ci
from scripts.diagnostics.run_requested_50k_models import SequenceNet

BASE = ROOT / "outputs/final_experiment_results_20260923/classical_full_scale"
RESULTS = ROOT / "outputs/final_experiment_results_20260923/baseline_metrics/common_full_validation"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 4096
ROWS_EXPECTED = 1_563_972


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def save_logits(name: str, predictor) -> Path:
    xval = np.load(BASE / "validation_X.npy", mmap_mode="r")
    yval = np.load(BASE / "validation_y.npy", mmap_mode="r")
    path = RESULTS / f"{name}_validation_logits.npy"
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                    shape=(len(yval), yval.shape[1]))
    for start in range(0, len(yval), BATCH):
        end = min(start + BATCH, len(yval))
        xb = torch.as_tensor(np.array(xval[start:end], dtype=np.float32, copy=True), device=DEVICE)
        out[start:end] = predictor(xb).detach().float().cpu().numpy()
        if end % (BATCH * 100) == 0 or end == len(yval):
            print(f"score {name}: {end}/{len(yval)}", flush=True)
    out.flush()
    del out, xval, yval
    gc.collect()
    return path


def score_linear_checkpoint(name: str, source: Path, out_checkpoint: Path | None = None,
                            refine: bool = False) -> Path:
    xtrain = np.load(BASE / "train_X.npy", mmap_mode="r")
    ytrain = np.load(BASE / "train_y.npy", mmap_mode="r")
    use_existing_refinement = bool(refine and out_checkpoint and out_checkpoint.exists())
    checkpoint_to_load = out_checkpoint if use_existing_refinement else source
    payload = torch.load(checkpoint_to_load, map_location="cpu", weights_only=False)
    model = torch.nn.Linear(xtrain.shape[1], ytrain.shape[1]).to(DEVICE)
    model.load_state_dict(payload["model"])
    model.train(refine)
    if refine and not use_existing_refinement:
        # One lower-rate hinge-loss continuation epoch, fit only on the 90% train split.
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-5)
        order = np.random.default_rng(4210).permutation(len(xtrain))
        for start in range(0, len(order), BATCH):
            ids = order[start:start + BATCH]
            xb = torch.as_tensor(np.array(xtrain[ids], dtype=np.float32, copy=True), device=DEVICE)
            yb = torch.as_tensor(np.array(ytrain[ids], dtype=np.float32, copy=True), device=DEVICE)
            signed = yb.mul(2).sub(1)
            loss = F.relu(1 - signed * model(xb)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if start // BATCH % 200 == 0:
                print(f"refine {name}: rows={min(start + BATCH, len(order))}/{len(order)} loss={float(loss):.6f}", flush=True)
            del xb, yb, signed, loss
        if out_checkpoint:
            torch.save({"model": model.state_dict(), "train_rows_full": len(xtrain),
                        "refinement_epochs": 1, "optimizer": "SGD lr=0.01, weight_decay=1e-5"},
                       out_checkpoint)
    model.eval()
    path = save_logits(name, lambda xb: model(xb))
    del model, xtrain, ytrain
    gc.collect()
    return path


def ann_checkpoint(source: Path, refined_checkpoint: Path | None = None) -> SequenceNet:
    reuse_refined = bool(refined_checkpoint and refined_checkpoint.exists())
    payload = torch.load(refined_checkpoint if reuse_refined else source,
                         map_location="cpu", weights_only=False)
    model = SequenceNet("ann", 0, 210).to(DEVICE)
    model.load_state_dict(payload["model"])
    if refined_checkpoint is not None and not reuse_refined:
        xtrain = np.load(BASE / "train_X.npy", mmap_mode="r")
        ytrain = np.load(BASE / "train_y.npy", mmap_mode="r")
        order = np.random.default_rng(4211).permutation(len(xtrain))
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        model.train()
        for start in range(0, len(order), BATCH):
            ids = order[start:start + BATCH]
            xb = torch.as_tensor(np.array(xtrain[ids], dtype=np.float32, copy=True), device=DEVICE)
            yb = torch.as_tensor(np.array(ytrain[ids], dtype=np.float32, copy=True), device=DEVICE)
            logits, _ = model(xb, None, None)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if start // BATCH % 200 == 0:
                print(f"refine ann: rows={min(start + BATCH, len(order))}/{len(order)} loss={float(loss):.6f}", flush=True)
            del xb, yb, logits, loss
        torch.save({"model": model.state_dict(), "model_name": "ann_fullscale_refined",
                    "base_checkpoint": str(source.relative_to(ROOT)), "refinement_epochs": 1,
                    "train_rows_full": len(xtrain), "optimizer": "AdamW lr=0.0002, weight_decay=0.0001"},
                   refined_checkpoint)
        model.eval()
        del xtrain, ytrain, order
        gc.collect()
    model.eval()
    return model


def evaluate_logits(name: str, path: Path, targets: np.ndarray, outcomes: list[str],
                    train_rows: int, note: str) -> dict:
    status_path = RESULTS / f"{name}_status.json"
    metrics_path = RESULTS / f"{name}_metrics.json"
    if metrics_path.exists():
        saved = json.loads(metrics_path.read_text())
        if saved.get("status") == "completed" and saved.get("validation_rows") == len(targets):
            return saved
    write_json(status_path, {"model": name, "status": "scoring", "train_rows": train_rows,
                             "validation_rows": len(targets), "started_utc": time.time()})
    logits = np.load(path, mmap_mode="r")
    report = event_metric_report_with_subsample_ci(
        torch.from_numpy(logits), torch.from_numpy(np.array(targets, dtype=np.uint8, copy=True)),
        outcomes, bootstrap_repeats=200, bootstrap_seed=4300, max_ci_rows=100_000)
    report.update({"model": name, "status": "completed", "train_rows": train_rows,
                   "validation_rows": len(targets), "split": "patient-disjoint 90:10, seed 42",
                   "fit_scope": "all valid training target rows" if train_rows == 14_128_539 else "saved 50k-row historical fit",
                   "method_note": note})
    write_json(metrics_path, report)
    write_json(status_path, {"model": name, "status": "completed", "train_rows": train_rows,
                             "validation_rows": len(targets), "metrics": metrics_path.name})
    del logits
    gc.collect()
    return report


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    xtrain = np.load(BASE / "train_X.npy", mmap_mode="r")
    ytrain = np.load(BASE / "train_y.npy", mmap_mode="r")
    xval = np.load(BASE / "validation_X.npy", mmap_mode="r")
    yval = np.load(BASE / "validation_y.npy", mmap_mode="r")
    if len(ytrain) != 14_128_539 or len(yval) != ROWS_EXPECTED:
        raise RuntimeError(f"Unexpected patient 90:10 row counts: train={len(ytrain)} validation={len(yval)}")
    outcomes = json.loads((ROOT / "data/perioperative_event_sequences_v5_richctx_static7/event_sequence_meta.json").read_text())["outcome_vocabulary"]
    existing = {
        "univariate": ROOT / "outputs/final_experiment_results_20260923/baseline_metrics/univariate_fullscale_internal.json",
        "logistic_regression": ROOT / "outputs/final_experiment_results_20260923/baseline_metrics/logistic_fullscale_internal.json",
        "maomao": ROOT / "outputs/final_experiment_results_20260923/model_metrics/maomao_internal.json",
    }
    reports = {}
    for name, path in existing.items():
        report = json.loads(path.read_text())
        rows = report.get("rows", report.get("evaluation_rows"))
        if rows != len(yval):
            raise RuntimeError(f"{name} metrics use {rows} validation rows, expected {len(yval)}")
        report.setdefault("validation_rows", rows)
        report.setdefault("train_rows", len(ytrain))
        reports[name] = report

    # Materialize the small single-variable model weights from all training rows;
    # the upstream run saved its full-validation metrics but not the token rates.
    univariate_model_path = RESULTS / "univariate_last_token_model.npz"
    if not univariate_model_path.exists():
        token_vocab = json.loads((ROOT / "data/perioperative_event_sequences_v5_richctx_static7/token_vocabulary.json").read_text())
        num_tokens = len(token_vocab)
        token_counts = np.zeros(num_tokens, dtype=np.int64)
        token_targets = np.zeros((num_tokens, ytrain.shape[1]), dtype=np.int64)
        global_counts = np.zeros(ytrain.shape[1], dtype=np.int64)
        for start in range(0, len(xtrain), 65_536):
            end = min(start + 65_536, len(xtrain))
            ids = np.clip(np.rint(np.asarray(xtrain[start:end, 0]) * (num_tokens - 1)).astype(np.int64),
                          0, num_tokens - 1)
            labels = np.asarray(ytrain[start:end], dtype=np.uint8)
            # NumPy promotes uint8 reductions to uint64; mixing uint64 and
            # int64 in-place can route through float64 and reject the cast.
            global_counts += labels.sum(0, dtype=np.int64)
            token_counts += np.bincount(ids, minlength=num_tokens)
            row_ids, col_ids = np.nonzero(labels)
            np.add.at(token_targets, (ids[row_ids], col_ids), 1)
            if end % (65_536 * 100) == 0 or end == len(xtrain):
                print(f"fit univariate model: {end}/{len(xtrain)}", flush=True)
        global_rate = (global_counts + 1.0) / (len(ytrain) + 2.0)
        token_rates = (token_targets + global_rate[None, :]) / (token_counts[:, None] + 1.0)
        np.savez(univariate_model_path, token_rates=token_rates, global_rate=global_rate,
                 token_counts=token_counts, training_rows=len(ytrain), feature_index=0)

    svm_path = ROOT / "outputs/final_experiment_results_20260923/classical_ml_fullscale/linear_svm/linear_svm.pt"
    svm_logits = score_linear_checkpoint("linear_svm_saved", svm_path)
    reports["linear_svm_saved"] = evaluate_logits("linear_svm_saved", svm_logits, yval, outcomes,
        len(ytrain), "Saved full-scale hinge-SGD checkpoint; evaluated on every patient-held-out validation target row.")
    svm_refined = RESULTS / "linear_svm_refined.pt"
    svm_refined_logits = score_linear_checkpoint("linear_svm_refined", svm_path, svm_refined, refine=True)
    reports["linear_svm_refined"] = evaluate_logits("linear_svm_refined", svm_refined_logits, yval, outcomes,
        len(ytrain), "Saved SVM followed by one lower-learning-rate hinge-loss pass over all training rows; no validation-based tuning.")

    ann_source = ROOT / "outputs/final_experiment_results_20260923/requested_50k_models/ann.pt"
    ann0 = ann_checkpoint(ann_source)
    ann0_logits = save_logits("ann_saved_50k", lambda xb: ann0(xb, None, None)[0])
    del ann0
    reports["ann_saved_50k"] = evaluate_logits("ann_saved_50k", ann0_logits, yval, outcomes,
        50_000, "Saved one-epoch 50k-row ANN checkpoint; evaluated on all full-scale validation rows.")
    ann_refined_path = RESULTS / "ann_fullscale_refined.pt"
    ann_refined = ann_checkpoint(ann_source, ann_refined_path)
    ann_refined_logits = save_logits("ann_fullscale_refined", lambda xb: ann_refined(xb, None, None)[0])
    del ann_refined
    reports["ann_fullscale_refined"] = evaluate_logits("ann_fullscale_refined", ann_refined_logits,
        yval, outcomes, len(ytrain), "Saved ANN followed by one AdamW pass over every training row; no validation-based tuning.")

    report = {
        "status": "completed", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split": "patient-disjoint 90:10, seed 42", "patients_train": 89_897,
        "patients_validation": 9_989, "train_target_rows": len(ytrain),
        "validation_target_rows": len(yval), "outcomes": len(outcomes),
        "feature_count": int(xtrain.shape[1]), "confidence_intervals": "95%; full-cohort point estimates; 200 row-bootstrap replicates with 100,000-row sample and documented scaling",
        "models": reports,
    }
    write_json(RESULTS / "common_full_validation_metrics.json", report)
    print(json.dumps({name: {k: r.get(k) for k in ("train_rows", "validation_rows", "micro_auprc", "micro_auroc", "mrr", "micro_auprc_95ci", "micro_auroc_95ci")}
                      for name, r in reports.items()}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
