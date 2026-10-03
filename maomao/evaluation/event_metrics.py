"""Exact, calibration and frequency-stratified next-event metrics."""
from __future__ import annotations

import math
from typing import Callable, Sequence

import torch
from .rank_statistics import binary_rank_metrics, batched_rank_metrics, RANK_VERSION


def average_precision(scores: torch.Tensor, target: torch.Tensor) -> float | None:
    return binary_rank_metrics(scores, target)[0]


def roc_auc(scores: torch.Tensor, target: torch.Tensor) -> float | None:
    return binary_rank_metrics(scores, target)[1]


def _auroc_pair(scores: torch.Tensor, targets: torch.Tensor) -> tuple[float | None, float | None]:
    micro = roc_auc(scores.flatten(), targets.flatten())
    values = [roc_auc(scores[:, i], targets[:, i]) for i in range(scores.shape[1])]
    values = [value for value in values if value is not None]
    return micro, (float(sum(values) / len(values)) if values else None)


def bootstrap_ci(scores: torch.Tensor, targets: torch.Tensor,
                 metric: Callable[[torch.Tensor, torch.Tensor], float | None],
                 repeats: int = 1000, seed: int = 42,
                 alpha: float = 0.05) -> list[float] | None:
    """Row-bootstrap confidence interval for one scalar metric."""
    if repeats <= 0 or len(scores) < 2:
        return None
    generator = torch.Generator(device=scores.device).manual_seed(seed)
    values = []
    for _ in range(int(repeats)):
        indices = torch.randint(len(scores), (len(scores),), generator=generator,
                                device=scores.device)
        value = metric(scores[indices], targets[indices])
        if value is not None and math.isfinite(value):
            values.append(value)
    if not values:
        return None
    quantiles = torch.tensor(values, dtype=torch.float64).quantile(
        torch.tensor([alpha / 2.0, 1.0 - alpha / 2.0], dtype=torch.float64))
    return [float(quantiles[0]), float(quantiles[1])]


def _scalar_metric_values(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float | None]:
    """Return aggregate metrics for one bootstrap sample."""
    probabilities = torch.softmax(logits.float(), dim=-1)
    order = logits.argsort(dim=-1, descending=True)
    ranked_true = targets.bool().gather(1, order)
    first_rank = ranked_true.float().argmax(1) + 1
    has_true = ranked_true.any(1)
    reciprocal_rank = torch.where(
        has_true, first_rank.float().reciprocal(), torch.zeros_like(first_rank, dtype=torch.float))
    top = order[:, 0]
    top_correct = targets.bool().gather(1, top[:, None]).squeeze(1)
    confidence = probabilities.gather(1, top[:, None]).squeeze(1)
    normalized = targets.float() / targets.sum(-1, keepdim=True).clamp_min(1)
    micro_ap = average_precision(probabilities.flatten(), targets.flatten())
    per_ap = [average_precision(probabilities[:, i], targets[:, i])
              for i in range(targets.shape[1])]
    per_ap = [value for value in per_ap if value is not None]
    micro_auroc, macro_auroc = _auroc_pair(probabilities, targets)
    return {
        "micro_auprc": micro_ap,
        "macro_auprc": (float(sum(per_ap) / len(per_ap)) if per_ap else None),
        "micro_auroc": micro_auroc,
        "macro_auroc": macro_auroc,
        "mrr": float(reciprocal_rank.mean()),
        "brier": float((probabilities - normalized).square().sum(-1).mean()),
        "ece": expected_calibration_error(confidence, top_correct),
        "hit_at_1": float(top_correct.float().mean()),
        "recall_at_5": float(
            ranked_true[:, :min(5, logits.shape[1])].sum(1).float().div(
                targets.sum(1).clamp_min(1)).mean()),
        "recall_at_10": float(
            ranked_true[:, :min(10, logits.shape[1])].sum(1).float().div(
                targets.sum(1).clamp_min(1)).mean()),
    }


def bootstrap_scalar_cis(logits: torch.Tensor, targets: torch.Tensor,
                         repeats: int = 1000, seed: int = 42,
                         alpha: float = 0.05) -> dict[str, list[float] | None]:
    """Compute shared row-bootstrap CIs with batched tensor operations.

    The previous implementation called ``_scalar_metric_values`` once per
    replicate, which made a 1000-replicate external report spend most of its
    time in Python loops.  The bootstrap distribution is unchanged in meaning:
    rows are sampled with replacement and every aggregate metric uses the same
    sampled rows.  Replicates are now processed in chunks so sorting, ranking,
    and reductions happen in PyTorch batches.
    """
    if repeats <= 0 or len(logits) < 2:
        return {}
    generator = torch.Generator(device=logits.device).manual_seed(seed)
    values: dict[str, list[torch.Tensor]] = {}
    n = len(logits)
    classes = logits.shape[1]
    # Keep one row-bootstrap replicate at a time.  Large external cohorts can
    # have millions of rows; batching 16 replicates materializes an enormous
    # [replicate, row, event] tensor and can OOM even when the base tensors fit.
    chunk_size = 1
    probabilities = torch.softmax(logits.float().cpu(), dim=-1).to(logits.device)
    base_order = logits.float().cpu().argsort(dim=-1, descending=True).to(logits.device)
    target_bool = targets.bool()

    def add(name: str, value: torch.Tensor) -> None:
        value = value.detach().float().cpu()
        finite = torch.isfinite(value)
        if finite.any():
            values.setdefault(name, []).append(value[finite])

    for start in range(0, int(repeats), chunk_size):
        batch_repeats = min(chunk_size, int(repeats) - start)
        if start == 0 or (start + batch_repeats) % (chunk_size * 4) == 0 or start + batch_repeats == int(repeats):
            print(f"[bootstrap] progress={min(start + batch_repeats, int(repeats))}/{int(repeats)}", flush=True)
        indices = torch.randint(n, (batch_repeats, n), generator=generator,
                                device=logits.device)
        p = probabilities[indices]
        t = target_bool[indices]
        order = base_order[indices]
        ranked_true = t.gather(2, order)
        first_rank = ranked_true.float().argmax(2) + 1
        has_true = ranked_true.any(2)
        rr = torch.where(has_true, first_rank.float().reciprocal(),
                         torch.zeros_like(first_rank, dtype=torch.float)).mean(1)
        normalized = t.float() / t.sum(-1, keepdim=True).clamp_min(1)
        top = order[:, :, 0]
        top_correct = t.gather(2, top[:, :, None]).squeeze(2)
        confidence = p.gather(2, top[:, :, None]).squeeze(2)
        add("mrr", rr)
        add("brier", (p - normalized).square().sum(-1).mean(1))
        add("hit_at_1", top_correct.float().mean(1))
        add("recall_at_5", ranked_true[:, :, :min(5, classes)].sum(2).float().div(
            t.sum(2).clamp_min(1)).mean(1))
        add("recall_at_10", ranked_true[:, :, :min(10, classes)].sum(2).float().div(
            t.sum(2).clamp_min(1)).mean(1))

        # ECE is kept identical to expected_calibration_error, but evaluated
        # over the bootstrap and sample dimensions together per bin.
        ece = confidence.new_zeros(batch_repeats)
        boundaries = torch.linspace(0, 1, 16, device=confidence.device)
        for bin_index in range(15):
            if bin_index == 14:
                selected = (confidence >= boundaries[bin_index]) & (confidence <= boundaries[bin_index + 1])
            else:
                selected = (confidence >= boundaries[bin_index]) & (confidence < boundaries[bin_index + 1])
            count = selected.sum(1)
            safe = count.clamp_min(1)
            ece += count.float().div(n) * (
                (confidence * selected).sum(1).div(safe) -
                (top_correct.float() * selected).sum(1).div(safe)).abs()
        add("ece", ece)

        per_ap, per_auc = batched_rank_metrics(p.transpose(1, 2), t.transpose(1, 2))
        add("macro_auprc", torch.nanmean(per_ap, dim=1))
        add("macro_auroc", torch.nanmean(per_auc, dim=1))
        micro_ap, micro_auc = batched_rank_metrics(p.reshape(batch_repeats, -1), t.reshape(batch_repeats, -1))
        add("micro_auprc", micro_ap)
        add("micro_auroc", micro_auc)

    result: dict[str, list[float] | None] = {}
    for name, chunks in values.items():
        sample = torch.cat(chunks)
        quantiles = sample.to(torch.float64).quantile(
            torch.tensor([alpha / 2.0, 1.0 - alpha / 2.0], dtype=torch.float64))
        result[name] = [float(quantiles[0]), float(quantiles[1])]
    return result


def expected_calibration_error(confidence: torch.Tensor, correct: torch.Tensor,
                               bins: int = 15) -> float:
    boundaries = torch.linspace(0, 1, bins + 1, device=confidence.device)
    error = confidence.new_zeros(())
    for index in range(bins):
        if index == bins - 1:
            selected = (confidence >= boundaries[index]) & (confidence <= boundaries[index + 1])
        else:
            selected = (confidence >= boundaries[index]) & (confidence < boundaries[index + 1])
        if selected.any():
            error += selected.float().mean() * (
                confidence[selected].mean() - correct[selected].float().mean()).abs()
    return float(error)


def event_metric_report(logits: torch.Tensor, targets: torch.Tensor,
                        outcome_names: Sequence[str],
                        core_mask: torch.Tensor | None = None,
                        bootstrap_repeats: int = 0,
                        bootstrap_seed: int = 42) -> dict:
    """Compute multi-target next-event metrics from uncalibrated logits."""
    logits = logits.float()
    targets = targets.bool()
    if core_mask is not None:
        core_mask = core_mask.bool().to(logits.device)
        keep_rows = targets[:, core_mask].any(1)
        logits = logits[keep_rows][:, core_mask]
        targets = targets[keep_rows][:, core_mask]
        outcome_names = [name for name, keep in zip(outcome_names, core_mask.tolist()) if keep]
    if not len(logits):
        return {"event_targets": 0}
    probabilities = torch.softmax(logits, dim=-1)
    order = logits.argsort(dim=-1, descending=True)
    ranked_true = targets.gather(1, order)
    first_rank = ranked_true.float().argmax(1) + 1
    has_true = ranked_true.any(1)
    reciprocal_rank = torch.where(
        has_true, first_rank.float().reciprocal(), torch.zeros_like(first_rank, dtype=torch.float))
    normalized_target = targets.float() / targets.sum(-1, keepdim=True).clamp_min(1)
    support = targets.sum(0)
    per_event = {}
    ap_values = []
    for index, name in enumerate(outcome_names):
        ap, auc = binary_rank_metrics(probabilities[:, index], targets[:, index])
        if ap is not None:
            ap_values.append(ap)
        top5 = order[:, :min(5, logits.shape[1])]
        hit5 = (top5 == index).any(1) & targets[:, index]
        per_event[name] = {
            "support": int(support[index]),
            "auroc": auc,
            "auprc": ap,
            "recall_at_5": (float(hit5.sum() / support[index])
                            if support[index] else None),
        }
    top = order[:, 0]
    top_correct = targets.gather(1, top[:, None]).squeeze(1)
    confidence = probabilities.gather(1, top[:, None]).squeeze(1)
    micro_ap, micro_auroc = binary_rank_metrics(probabilities.flatten(), targets.flatten())
    auc_values = [item["auroc"] for item in per_event.values() if item["auroc"] is not None]
    macro_auroc = sum(auc_values)/len(auc_values) if auc_values else None
    frequency = support.float()
    active = frequency > 0
    strata = {}
    if active.any():
        quantiles = torch.quantile(frequency[active], torch.tensor(
            [0.25, 0.75], device=frequency.device))
        labels = {
            "rare": active & (frequency <= quantiles[0]),
            "medium": active & (frequency > quantiles[0]) & (frequency < quantiles[1]),
            "frequent": active & (frequency >= quantiles[1]),
        }
        for label, mask in labels.items():
            values = [per_event[name]["auprc"] for name, keep in zip(outcome_names, mask.tolist())
                      if keep and per_event[name]["auprc"] is not None]
            strata[label] = {
                "event_classes": int(mask.sum()),
                "macro_auprc": float(sum(values) / len(values)) if values else None,
                "support_range": ([int(frequency[mask].min()), int(frequency[mask].max())]
                                  if mask.any() else None),
            }
    report = {
        "event_targets": len(logits),
        "rank_metric_definition": RANK_VERSION,
        "micro_auprc": micro_ap,
        "macro_auprc": float(sum(ap_values) / len(ap_values)) if ap_values else None,
        "micro_auroc": micro_auroc,
        "macro_auroc": macro_auroc,
        "mrr": float(reciprocal_rank.mean()),
        "brier": float((probabilities - normalized_target).square().sum(-1).mean()),
        "ece": expected_calibration_error(confidence, top_correct),
        "hit_at_1": float(top_correct.float().mean()),
        "recall_at_5": float(
            ranked_true[:, :min(5, logits.shape[1])].sum(1).float().div(
                targets.sum(1).clamp_min(1)).mean()),
        "recall_at_10": float(
            ranked_true[:, :min(10, logits.shape[1])].sum(1).float().div(
                targets.sum(1).clamp_min(1)).mean()),
        "frequency_strata": strata,
        "per_event": per_event,
    }
    if bootstrap_repeats > 0:
        for metric_name, interval in bootstrap_scalar_cis(
                logits, targets, bootstrap_repeats, bootstrap_seed).items():
            report[f"{metric_name}_95ci"] = interval
    return report


def event_metric_report_with_subsample_ci(
        logits: torch.Tensor, targets: torch.Tensor, outcome_names: Sequence[str],
        bootstrap_repeats: int = 20, bootstrap_seed: int = 42,
        max_ci_rows: int = 100_000) -> dict:
    """Full-cohort point metrics plus low-memory, sample-size-scaled row CIs.

    The exact point estimates use every row. For very large cohorts, bootstrap
    replicates on a bounded uniform row sample, then scale deviations from the
    sample estimate by sqrt(sample_n / cohort_n). This normal-approximation
    correction avoids materializing [replicate, million_rows, outcomes] arrays.
    The method and effective sample size are recorded in the returned report.
    """
    logits = logits.float()
    targets = targets.bool()
    if len(logits) <= max_ci_rows:
        report = event_metric_report(
            logits, targets, outcome_names,
            bootstrap_repeats=bootstrap_repeats, bootstrap_seed=bootstrap_seed)
        report["ci_method"] = "row bootstrap on full test cohort"
        report["ci_rows"] = int(len(logits))
        return report

    full_n = len(logits)
    full = event_metric_report(logits, targets, outcome_names)
    generator = torch.Generator(device="cpu").manual_seed(bootstrap_seed)
    sample_idx = torch.randperm(full_n, generator=generator)[:max_ci_rows]
    sample_logits = logits[sample_idx.to(logits.device)]
    sample_targets = targets[sample_idx.to(targets.device)]
    sample = event_metric_report(
        sample_logits, sample_targets, outcome_names,
        bootstrap_repeats=bootstrap_repeats, bootstrap_seed=bootstrap_seed + 1)
    width_scale = (len(sample_idx) / full_n) ** 0.5
    bounded_metrics = {
        "micro_auprc", "macro_auprc", "micro_auroc", "macro_auroc",
        "mrr", "brier", "ece", "hit_at_1", "recall_at_5", "recall_at_10",
    }
    for metric in bounded_metrics:
        ci = sample.get(f"{metric}_95ci")
        center_sample = sample.get(metric)
        center_full = full.get(metric)
        if ci is None or center_sample is None or center_full is None:
            continue
        full[f"{metric}_95ci"] = [
            max(0.0, min(1.0, center_full + (float(bound) - center_sample) * width_scale))
            for bound in ci
        ]
    full["ci_method"] = "uniform row subsample bootstrap; interval deviations scaled by sqrt(sample_n/cohort_n)"
    full["ci_rows"] = int(len(sample_idx))
    full["ci_cohort_rows"] = int(full_n)
    full["ci_bootstrap_repeats"] = int(bootstrap_repeats)
    del sample_logits, sample_targets, sample
    return full
