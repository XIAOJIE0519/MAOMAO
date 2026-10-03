"""Autoregressive sparse event + time trajectory generation."""
from __future__ import annotations

import math
from typing import Dict, Sequence

import torch


@torch.no_grad()
def generate_trajectory(model, context: Dict[str, torch.Tensor],
                        outcome_token_ids: Sequence[int], steps: int = 20,
                        temperature: float = 1.0,
                        max_context: int = 256) -> list[dict]:
    """Generate ``next event -> elapsed time -> append`` trajectories for one patient."""
    if context["token_id"].shape[0] != 1:
        raise ValueError("Autoregressive generation currently accepts one patient at a time")
    outcome_token_ids = torch.as_tensor(
        outcome_token_ids, dtype=torch.long, device=context["token_id"].device)
    generated = []
    batch = {key: value.clone() for key, value in context.items()}
    for _ in range(int(steps)):
        output = model(batch)
        position = int(batch["attention_mask"][0].sum()) - 1
        logits = output.logits[0, position] / max(float(temperature), 1e-4)
        event_id = int(logits.argmax())
        if output.time_mu is not None:
            sigma = output.time_log_sigma[0, position].exp()
            delta_hours = float(torch.exp(
                output.time_mu[0, position] + 0.5 * sigma.square()).clamp(1 / 60, 24 * 38))
        elif output.log_total_rate is not None:
            delta_hours = float(torch.exp(-output.log_total_rate[0, position]).clamp(1 / 60, 24 * 38))
        else:
            delta_hours = float(torch.exp(-torch.logsumexp(logits, -1)).clamp(1 / 60, 24 * 38))
        next_time = float(batch["time_min"][0, position]) + delta_hours * 60.0
        generated.append({"event_index": event_id, "delta_hours": delta_hours,
                          "time_min": next_time})

        sequence_values = {
            "token_id": outcome_token_ids[event_id],
            "time_min": torch.tensor(next_time, device=logits.device),
            "gap_min": torch.tensor(delta_hours * 60.0, device=logits.device),
            "value": torch.tensor(0.0, device=logits.device),
            "has_value": torch.tensor(0.0, device=logits.device),
            "token_kind": torch.tensor(6, dtype=torch.long, device=logits.device),
            "phase_id": batch["phase_id"][0, position] if "phase_id" in batch else None,
        }
        for key, value in sequence_values.items():
            if value is None or key not in batch:
                continue
            batch[key] = torch.cat((batch[key], value.reshape(1, 1)), dim=1)[:, -max_context:]
        if "observation_features" in batch:
            feature = torch.zeros((1, 1, 3), device=logits.device)
            batch["observation_features"] = torch.cat(
                (batch["observation_features"], feature), dim=1)[:, -max_context:]
        length = batch["token_id"].shape[1]
        batch["attention_mask"] = torch.ones(
            (1, length), dtype=torch.bool, device=logits.device)
    return generated
