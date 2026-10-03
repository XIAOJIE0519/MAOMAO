"""GPU bootstrap with original CPU random row draws and standard tie statistics."""
import math
import torch
from .rank_statistics import batched_rank_metrics

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
    generator = torch.Generator(device="cpu").manual_seed(seed)
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
                                device="cpu").to(logits.device)
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
