"""Bounded-memory plotting summaries using every supplied prediction row.

Threshold curves are histogram approximations, never substitutes for point metrics.
No patient identifiers or individual clinical observations are exported.
"""
from pathlib import Path
import json
import numpy as np
import torch

RETRIEVAL_TIE_PROTOCOL = "torch_cpu_argsort_on_logits_v1"

def export_plot_sources(logits_path, labels_path, destination, outcomes, temperature=1., metadata=None, bias=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    z = np.load(logits_path, mmap_mode="r")
    y = np.load(labels_path, mmap_mode="r")
    if z.shape != y.shape or z.shape[1] != len(outcomes):
        raise ValueError("Prediction/label/outcome mismatch")
    bins = 4096
    edges = np.linspace(-40., 0., bins+1, dtype=np.float64)
    positives = np.zeros((len(outcomes), bins), dtype=np.int64)
    totals = np.zeros_like(positives)
    reliability = np.zeros((20, 4), dtype=np.float64)
    ranks = np.zeros(len(outcomes), dtype=np.int64)
    cardinality = np.zeros(len(outcomes)+1, dtype=np.int64)
    confusion = np.zeros((len(outcomes), len(outcomes)), dtype=np.float64)
    brier = 0.
    clipped = 0
    for start in range(0, len(y), 8192):
        scores = torch.from_numpy(np.array(z[start:start+8192], dtype=np.float32))
        if bias is not None:
            scores = scores + torch.as_tensor(bias, dtype=scores.dtype)
        scores = scores / temperature
        target = np.asarray(y[start:start+8192], dtype=np.uint8)
        lp = torch.log_softmax(scores, -1).numpy()
        p = torch.softmax(scores, -1).numpy().astype(np.float64)
        idx = np.clip(((lp+40.)*(bins/40.)).astype(np.int64), 0, bins-1)
        clipped += int((lp < -40.).sum())
        for event in range(len(outcomes)):
            totals[event] += np.bincount(idx[:,event], minlength=bins)
            positives[event] += np.bincount(idx[target[:,event] > 0,event], minlength=bins)
        count = target.sum(1)
        if np.any(count == 0):
            raise ValueError("Empty target set")
        cardinality += np.bincount(count.astype(np.int64), minlength=len(cardinality))
        q = target / count[:,None]
        brier += float(np.square(p-q).sum())
        order = scores.argsort(dim=-1, descending=True).numpy()
        rank = np.argmax(np.take_along_axis(target, order, axis=1) > 0, axis=1)
        ranks += np.bincount(rank, minlength=len(ranks))
        pred = order[:,0]
        conf = p[np.arange(len(p)),pred]
        correct = target[np.arange(len(p)),pred]
        bi = np.minimum((conf*20).astype(np.int64), 19)
        for col, weights in enumerate((np.ones(len(p)), conf, correct, conf*conf)):
            reliability[:,col] += np.bincount(bi, weights=weights, minlength=20)
        np.add.at(confusion, pred, q)
    if not np.array_equal(totals.sum(1), np.full(len(outcomes),len(y))):
        raise RuntimeError("Histogram mass verification failed")
    np.savez_compressed(destination / "full_row_curves.npz", log_probability_edges=edges,
                        positive_histogram=positives, negative_histogram=totals-positives,
                        outcome_names=np.asarray(outcomes))
    np.savetxt(destination / "reliability.csv", np.column_stack((np.arange(20)/20,
               (np.arange(20)+1)/20, reliability)), delimiter=",", comments="",
               header="lower_confidence,upper_confidence,rows,sum_confidence,sum_any_positive_correct,sum_squared_confidence")
    np.savetxt(destination / "first_positive_rank.csv", np.column_stack((np.arange(1,len(ranks)+1),ranks)),
               delimiter=",", comments="", header="rank,rows", fmt="%d")
    np.savetxt(destination / "target_cardinality.csv", np.column_stack((np.arange(len(cardinality)),cardinality)),
               delimiter=",", comments="", header="positive_classes,rows", fmt="%d")
    np.savez_compressed(destination / "top1_fractional_confusion.npz", matrix=confusion,
                        outcome_names=np.asarray(outcomes))
    selected = np.arange(len(y)) if len(y) <= 10000 else np.sort(np.random.default_rng(42).choice(len(y),10000,replace=False))
    display_logits = torch.from_numpy(np.array(z[selected],dtype=np.float32))/temperature
    np.savez_compressed(destination / "prediction_display_sample.npz",
                        probabilities=torch.softmax(display_logits,-1).numpy(),
                        targets=np.asarray(y[selected],dtype=np.uint8),
                        outcome_names=np.asarray(outcomes))
    result = dict(metadata or {}, rows=int(len(y)), classes=len(outcomes), temperature=float(temperature),
                  curve_bins=bins, curve_grid="log probability [-40, 0], upper cumulative thresholds",
                  curve_approximate=True, below_grid_values=clipped,
                  brier_normalized_target=brier/len(y),
                  reliability_definition="max softmax confidence vs any-positive top1 correctness",
                  confusion_definition="predicted class rows; normalized positive target class columns",
                  identifiers_included=False, all_prediction_rows_used=True)
    result.update(display_sample_rows=len(selected),display_sample_seed=42,
                  retrieval_tie_protocol=RETRIEVAL_TIE_PROTOCOL,
                  display_sample_used_for_full_metrics_or_curves=False,
                  reliability_bins=20,reported_ece_bins=15)
    (destination / "provenance.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    return result
