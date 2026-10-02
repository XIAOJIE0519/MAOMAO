"""MAOMAO event-time transformer for sparse perioperative trajectories."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class EventMAOMAOOutput:
    logits: Optional[torch.Tensor]
    family_logits: Optional[torch.Tensor] = None
    log_total_rate: Optional[torch.Tensor] = None
    time_mu: Optional[torch.Tensor] = None
    time_log_sigma: Optional[torch.Tensor] = None
    family_time_mu: Optional[torch.Tensor] = None
    family_time_log_sigma: Optional[torch.Tensor] = None
    fine_hazard_logits: Optional[torch.Tensor] = None
    long_hazard_logits: Optional[torch.Tensor] = None
    tail_mu: Optional[torch.Tensor] = None
    tail_log_sigma: Optional[torch.Tensor] = None
    trajectory_logits: Optional[torch.Tensor] = None
    token_logits: Optional[torch.Tensor] = None
    value_prediction: Optional[torch.Tensor] = None
    hidden_state: Optional[torch.Tensor] = None


class ContinuousTimeEncoding(nn.Module):
    def __init__(self, hidden_dim: int, enhanced: bool = False):
        super().__init__()
        half = hidden_dim // 2
        frequencies = torch.exp(torch.linspace(math.log(1 / 5), math.log(1 / 43200), half))
        self.register_buffer("frequencies", frequencies)
        self.projection = nn.Linear(half * 4, hidden_dim, bias=False)
        self.scale_projection = (nn.Sequential(
            nn.Linear(4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim, bias=False)
        ) if enhanced else None)

    def forward(self, absolute_minutes: torch.Tensor, gap_minutes: torch.Tensor) -> torch.Tensor:
        absolute = absolute_minutes[..., None] * self.frequencies
        gap = gap_minutes[..., None] * self.frequencies
        features = torch.cat((absolute.sin(), absolute.cos(), gap.sin(), gap.cos()), dim=-1)
        periodic = self.projection(features)
        if self.scale_projection is None:
            return periodic
        scale = math.log1p(43200.0)
        monotonic = torch.stack((
            torch.log1p(absolute_minutes.clamp_min(0)) / scale,
            torch.log1p(gap_minutes.clamp_min(0)) / scale,
            (gap_minutes > 0).to(absolute_minutes.dtype),
            (absolute_minutes / 43200.0).clamp(0, 2),
        ), dim=-1)
        return periodic + self.scale_projection(monotonic)


def dual_timescale_expected_wait(fine_hazard_logits: torch.Tensor,
                                 long_hazard_logits: torch.Tensor,
                                 tail_mu: torch.Tensor,
                                 tail_log_sigma: torch.Tensor) -> torch.Tensor:
    """Decode event-specific expected waiting hours from the dual hazard head.

    The output shape is ``[..., num_outcomes]``. Fine hazards cover 0--2 h in
    five-minute bins, long hazards cover the configurable 2 h onward interval
    in thirty-minute bins, and the log-normal tail starts after the long head.
    """
    fine_probability = torch.sigmoid(fine_hazard_logits.float())
    fine_log_survival = F.logsigmoid(-fine_hazard_logits.float())
    fine_previous = torch.exp(torch.cat((
        torch.zeros_like(fine_log_survival[..., :1]),
        fine_log_survival[..., :-1].cumsum(-1)), dim=-1))
    fine_edges = (5.0 / 60.0) * torch.arange(
        fine_hazard_logits.shape[-1] + 1, device=fine_hazard_logits.device,
        dtype=fine_hazard_logits.dtype)
    fine_midpoints = (fine_edges[:-1] + fine_edges[1:]) / 2.0
    fine_mean = (fine_previous * fine_probability * fine_midpoints).sum(-1)
    survival_after_fine = torch.exp(fine_log_survival.sum(-1))

    long_probability = torch.sigmoid(long_hazard_logits.float())
    long_log_survival = F.logsigmoid(-long_hazard_logits.float())
    long_previous = torch.exp(torch.cat((
        torch.zeros_like(long_log_survival[..., :1]),
        long_log_survival[..., :-1].cumsum(-1)), dim=-1))
    long_edges = 2.0 + 0.5 * torch.arange(
        long_hazard_logits.shape[-1] + 1, device=long_hazard_logits.device,
        dtype=long_hazard_logits.dtype)
    long_midpoints = (long_edges[:-1] + long_edges[1:]) / 2.0
    long_mean = (long_previous * long_probability * long_midpoints).sum(-1)
    survival_after_long = torch.exp(long_log_survival.sum(-1))
    long_end = 2.0 + 0.5 * long_hazard_logits.shape[-1]
    # Keep diagnostic decoding numerically consistent with the bounded
    # log-normal tail used by the training loss.  An unconstrained sigma can
    # otherwise make exp(mu + sigma^2 / 2) overflow and produce meaningless
    # multi-million-hour MAE values even though the hazard logits are finite.
    tail_sigma = tail_log_sigma.float().clamp(-3.0, 2.0).exp()
    tail_mu_safe = tail_mu.float().clamp(-10.0, 10.0)
    tail_mean = long_end + torch.exp(
        tail_mu_safe + 0.5 * tail_sigma.square()).clamp_max(24.0 * 38.0)
    return (fine_mean + survival_after_fine * (
        long_mean + survival_after_long * tail_mean)
            ).clamp_min(1.0 / 60.0).clamp_max(24.0 * 38.0)


class EventMAOMAO(nn.Module):
    """Causal transformer predicting the next outcome set and waiting time.

    Output logits are interpreted as log cause-specific rates.  Their softmax
    predicts event identity; their log-sum-exp predicts the total event rate,
    following Delphi's competing-exponentials formulation.
    """

    def __init__(self, num_tokens: int, num_outcomes: int, num_static: int = 2,
                 hidden_dim: int = 256, num_layers: int = 8, num_heads: int = 8,
                 ffn_dim: int = 1024, dropout: float = 0.1,
                 initial_event_interval_hours: float = 24.0,
                 decoupled_time_head: bool = False,
                 enhanced_time_encoding: bool = False,
                 num_trajectory_horizons: int = 3,
                 lognormal_time_head: bool = False,
                 outcome_family_ids: Optional[torch.Tensor] = None,
                 use_family_head: bool = True,
                 same_time_block_causal: bool = False,
                 relative_time_attention: bool = False,
                 event_conditioned_time_head: bool = False,
                 phase_memory: bool = False,
                 clock_phase_context: bool = True,
                 observation_intensity: bool = False,
                 value_reconstruction: bool = False,
                 dual_timescale_time_head: bool = False,
                 fine_time_bins: int = 24,
                 long_time_bins: int = 8,
                 context_segment_length: int = 0,
                 model_family_count: int | None = None):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_outcomes = int(num_outcomes)
        self.num_heads = int(num_heads)
        self.decoupled_time_head = bool(decoupled_time_head)
        self.use_family_head = bool(use_family_head)
        self.lognormal_time_head = bool(lognormal_time_head)
        self.same_time_block_causal = bool(same_time_block_causal)
        self.relative_time_attention = bool(relative_time_attention)
        self.event_conditioned_time_head = bool(event_conditioned_time_head)
        self.phase_memory = bool(phase_memory)
        self.clock_phase_context = bool(clock_phase_context)
        self.observation_intensity = bool(observation_intensity)
        self.value_reconstruction = bool(value_reconstruction)
        self.dual_timescale_time_head = bool(dual_timescale_time_head)
        self.fine_time_bins = int(fine_time_bins)
        self.long_time_bins = int(long_time_bins)
        self.context_segment_length = int(context_segment_length)
        if self.context_segment_length < 0:
            raise ValueError("context_segment_length must be non-negative")
        if self.dual_timescale_time_head and (self.fine_time_bins <= 0 or self.long_time_bins <= 0):
            raise ValueError("dual-timescale hazard heads require positive bin counts")
        if outcome_family_ids is None:
            family_ids = torch.empty(0, dtype=torch.long)
            self.num_event_families = 0
        else:
            family_ids = torch.as_tensor(outcome_family_ids, dtype=torch.long)
            if family_ids.shape != (self.num_outcomes,):
                raise ValueError("outcome_family_ids must contain one ID per outcome")
            if family_ids.min().item() < 0:
                raise ValueError("outcome family IDs must be non-negative")
            self.num_event_families = int(family_ids.max().item()) + 1
        if model_family_count is not None:
            if int(model_family_count) < self.num_event_families:
                raise ValueError("model_family_count excludes an outcome family")
            self.num_event_families = int(model_family_count)
        if self.event_conditioned_time_head and not self.num_event_families:
            raise ValueError("event-conditioned time requires outcome_family_ids")
        self.register_buffer("outcome_family_ids", family_ids, persistent=False)
        self.register_buffer("clock_phase_token_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.num_trajectory_horizons = int(num_trajectory_horizons)
        self.token_embedding = nn.Embedding(num_tokens, hidden_dim, padding_idx=0)
        self.kind_embedding = nn.Embedding(16, hidden_dim, padding_idx=0)
        self.phase_embedding = (nn.Embedding(8, hidden_dim) if self.phase_memory else None)
        self.value_projection = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.observation_projection = (nn.Sequential(
            nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim, bias=False)
        ) if self.observation_intensity else None)
        self.static_projection = nn.Sequential(
            nn.Linear(num_static, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.history_projection = (nn.Sequential(
            nn.Linear(self.num_event_families, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        ) if self.num_event_families and self.phase_memory else None)
        self.time_encoding = ContinuousTimeEncoding(hidden_dim, enhanced_time_encoding)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            bias=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                             norm=nn.LayerNorm(hidden_dim))
        self.outcome_head = nn.Linear(hidden_dim, num_outcomes, bias=True)
        self.family_head = (nn.Linear(hidden_dim, self.num_event_families, bias=True)
                            if self.num_event_families and self.use_family_head else None)
        self.relative_time_bias = (nn.Sequential(
            nn.Linear(2, max(16, num_heads * 2)), nn.GELU(),
            nn.Linear(max(16, num_heads * 2), num_heads, bias=False),
        ) if self.relative_time_attention else None)
        self.time_head = (nn.Linear(hidden_dim, 1, bias=True)
                          if self.decoupled_time_head and not self.lognormal_time_head else None)
        time_output_dim = 2 * (self.num_event_families
                               if self.event_conditioned_time_head else 1)
        self.time_distribution_head = (nn.Linear(hidden_dim, time_output_dim, bias=True)
                                       if self.lognormal_time_head else None)
        self.dual_hazard_head = (nn.Linear(
            hidden_dim, self.num_outcomes * (self.fine_time_bins + self.long_time_bins), bias=True)
                                 if self.dual_timescale_time_head else None)
        self.tail_distribution_head = (nn.Linear(hidden_dim, self.num_outcomes * 2, bias=True)
                                       if self.dual_timescale_time_head else None)
        self.trajectory_head = nn.Linear(
            hidden_dim, self.num_trajectory_horizons * self.num_outcomes, bias=True)
        self.masked_token_bias = nn.Parameter(torch.zeros(num_tokens))
        self.value_head = (nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)) if self.value_reconstruction else None)
        self.apply(self._init_weights)
        # Initialise the aggregate rate to a clinically plausible sparse-event
        # interval.  A low starting rate prevents long event-free episodes from
        # producing an explosive waiting-time likelihood at step zero.
        interval = max(float(initial_event_interval_hours), 1.0 / 60.0)
        if self.lognormal_time_head:
            nn.init.zeros_(self.outcome_head.bias)
            nn.init.zeros_(self.time_distribution_head.weight)
            with torch.no_grad():
                parameters = self.time_distribution_head.bias.view(-1, 2)
                parameters[:, 0] = math.log(interval)
                parameters[:, 1] = 0.0
        elif self.decoupled_time_head:
            nn.init.zeros_(self.outcome_head.bias)
            nn.init.constant_(self.time_head.bias, math.log(1.0 / interval))
        else:
            nn.init.constant_(self.outcome_head.bias, math.log(1.0 / (interval * num_outcomes)))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(self, batch: Dict[str, torch.Tensor], *, causal: bool = True,
                return_token_logits: bool = False,
                masked_only: bool = False) -> EventMAOMAOOutput:
        x = self.token_embedding(batch["token_id"])
        x = x + self.kind_embedding(batch["token_kind"])
        if not self.clock_phase_context:
            # Remove the actual clock and phase-summary token signal for the
            # corresponding module ablation. Time encodings remain available,
            # so this does not accidentally ablate absolute/relative time.
            drop = batch["token_kind"].eq(7)
            if self.clock_phase_token_ids.numel():
                drop = drop | torch.isin(batch["token_id"], self.clock_phase_token_ids)
            x = x.masked_fill(drop[..., None], 0.0)
        # Raw clinical values span several orders of magnitude.  A signed-log
        # transform prevents EBL/enzymes from dominating bounded severity and
        # binary signals while preserving direction and ordering.
        robust_value = torch.sign(batch["value"]) * torch.log1p(batch["value"].abs())
        robust_value = robust_value.clamp(-12.0, 12.0)
        value_features = torch.stack((robust_value, batch["has_value"]), dim=-1)
        x = x + self.value_projection(value_features)
        if self.observation_projection is not None and "observation_features" in batch:
            x = x + self.observation_projection(batch["observation_features"])
        x = x + self.time_encoding(batch["time_min"], batch["gap_min"])
        x = x + self.static_projection(batch["static"])[:, None, :]
        if self.phase_embedding is not None and "phase_id" in batch:
            x = x + self.phase_embedding(batch["phase_id"])
        if (self.history_projection is not None and "history_family_counts" in batch and
                batch["history_family_counts"].shape[-1] == self.num_event_families):
            x = x + self.history_projection(batch["history_family_counts"])[:, None, :]

        length = x.shape[1]
        causal_mask = self._attention_mask(batch["time_min"], causal)
        padding = ~batch["attention_mask"].bool()
        if causal_mask is not None and causal_mask.dtype.is_floating_point:
            float_padding = torch.zeros_like(padding, dtype=causal_mask.dtype)
            padding = float_padding.masked_fill(padding, -torch.inf)
        encoded = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding)
        token_logits = (F.linear(encoded.float(), self.token_embedding.weight.float(),
                                 self.masked_token_bias.float())
                        if return_token_logits else None)
        value_prediction = (self.value_head(encoded).squeeze(-1).float()
                            if self.value_head is not None else None)
        if masked_only:
            return EventMAOMAOOutput(
                logits=None, token_logits=token_logits,
                value_prediction=value_prediction, hidden_state=encoded)
        log_total_rate = (self.time_head(encoded).squeeze(-1).float()
                          if self.time_head is not None else None)
        event_logits = self.outcome_head(encoded).float()
        family_logits = (self.family_head(encoded).float()
                         if self.family_head is not None else None)
        if family_logits is not None:
            event_logits = event_logits + family_logits[..., self.outcome_family_ids]
        time_mu = time_log_sigma = None
        family_time_mu = family_time_log_sigma = None
        fine_hazard_logits = long_hazard_logits = tail_mu = tail_log_sigma = None
        if self.time_distribution_head is not None:
            time_parameters = self.time_distribution_head(encoded).float()
            if self.event_conditioned_time_head:
                time_parameters = time_parameters.reshape(
                    *encoded.shape[:2], self.num_event_families, 2)
                family_time_mu = time_parameters[..., 0]
                family_time_log_sigma = time_parameters[..., 1].clamp(-3.0, 2.0)
                if family_logits is None:
                    # In the family-head ablation the family-specific time
                    # parameters remain available, but no family classifier is
                    # allowed to route probability mass.  Use a uniform prior
                    # so this ablation removes only the hierarchical family
                    # prediction pathway rather than silently changing the
                    # time-head tensor contract.
                    family_probability = torch.full_like(
                        family_time_mu, 1.0 / self.num_event_families)
                else:
                    family_probability = torch.softmax(family_logits, dim=-1)
                time_mu = (family_probability * family_time_mu).sum(-1)
                time_log_sigma = (family_probability * family_time_log_sigma).sum(-1)
            else:
                time_mu = time_parameters[..., 0]
                time_log_sigma = time_parameters[..., 1].clamp(-3.0, 2.0)
        if self.dual_hazard_head is not None:
            hazard = self.dual_hazard_head(encoded).float().reshape(
                *encoded.shape[:2], self.num_outcomes,
                self.fine_time_bins + self.long_time_bins)
            fine_hazard_logits = hazard[..., :self.fine_time_bins]
            long_hazard_logits = hazard[..., self.fine_time_bins:]
            tail = self.tail_distribution_head(encoded).float().reshape(
                *encoded.shape[:2], self.num_outcomes, 2)
            tail_mu = tail[..., 0]
            tail_log_sigma = tail[..., 1].clamp(-3.0, 2.0)
        trajectory_logits = self.trajectory_head(encoded).float().reshape(
            *encoded.shape[:2], self.num_trajectory_horizons, self.num_outcomes)
        return EventMAOMAOOutput(
            logits=event_logits, family_logits=family_logits,
            log_total_rate=log_total_rate,
            time_mu=time_mu, time_log_sigma=time_log_sigma,
            family_time_mu=family_time_mu,
            family_time_log_sigma=family_time_log_sigma,
            fine_hazard_logits=fine_hazard_logits,
            long_hazard_logits=long_hazard_logits,
            tail_mu=tail_mu,
            tail_log_sigma=tail_log_sigma,
            trajectory_logits=trajectory_logits, token_logits=token_logits,
            value_prediction=value_prediction, hidden_state=encoded)

    def _attention_mask(self, time_min: torch.Tensor,
                        causal: bool) -> Optional[torch.Tensor]:
        """Build a temporal bias and optionally hide all concurrent siblings."""
        batch_size, length = time_min.shape
        if not causal and self.relative_time_bias is None and not self.context_segment_length:
            return None
        query_time = time_min[:, :, None]
        key_time = time_min[:, None, :]
        delta = query_time - key_time
        mask = torch.zeros(
            batch_size, self.num_heads, length, length,
            device=time_min.device, dtype=torch.float32)
        if self.relative_time_bias is not None:
            scale = math.log1p(43200.0)
            relative_features = torch.stack((
                torch.sign(delta) * torch.log1p(delta.abs()) / scale,
                (delta == 0).to(delta.dtype),
            ), dim=-1)
            bias = self.relative_time_bias(relative_features).permute(0, 3, 1, 2)
            mask = mask + bias.float()
        if causal:
            positions = torch.arange(length, device=time_min.device)
            future = positions[None, :] > positions[:, None]
            forbidden = future[None, :, :].expand(batch_size, -1, -1)
            if self.same_time_block_causal:
                same_time = query_time == key_time
                diagonal = torch.eye(length, dtype=torch.bool, device=time_min.device)
                forbidden = forbidden | (same_time & ~diagonal[None, :, :])
            mask = mask.masked_fill(forbidden[:, None, :, :], -torch.inf)
        if self.context_segment_length:
            # Isolate contiguous segments at every encoder layer, including the
            # masked auxiliary pass. A sliding per-layer band would leak older
            # events through intermediate states and is not a strict length cap.
            segment = torch.arange(length, device=time_min.device) // self.context_segment_length
            separate = segment[:, None] != segment[None, :]
            mask = mask.masked_fill(separate[None, None, :, :], -torch.inf)
        return mask.reshape(batch_size * self.num_heads, length, length)


def event_time_loss(logits: torch.Tensor, target_set: torch.Tensor,
                    target_dt_hours: torch.Tensor, loss_mask: torch.Tensor,
                    time_mask: torch.Tensor | None = None, time_weight: float = 1.0,
                    max_wait_hours: float = 24.0 * 38,
                    stable_index: int = -1,
                    log_total_rate: torch.Tensor | None = None,
                    time_mu: torch.Tensor | None = None,
                    time_log_sigma: torch.Tensor | None = None,
                    trajectory_logits: torch.Tensor | None = None,
                    trajectory_target: torch.Tensor | None = None,
                    trajectory_mask: torch.Tensor | None = None,
                    trajectory_weight: float = 0.0,
                    class_weights: torch.Tensor | None = None,
                    family_logits: torch.Tensor | None = None,
                    outcome_family_ids: torch.Tensor | None = None,
                    family_weight: float = 0.0,
                    family_time_mu: torch.Tensor | None = None,
                    family_time_log_sigma: torch.Tensor | None = None,
                    fine_hazard_logits: torch.Tensor | None = None,
                    long_hazard_logits: torch.Tensor | None = None,
                    tail_mu: torch.Tensor | None = None,
                    tail_log_sigma: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
    """Event CE plus competing-exponential event/censor waiting-time NLL."""
    event_valid = loss_mask.bool() & (target_set.sum(-1) > 0)
    time_valid = event_valid if time_mask is None else time_mask.bool()
    has_trajectory = (trajectory_logits is not None and trajectory_target is not None and
                      trajectory_mask is not None and trajectory_mask.any())
    if not event_valid.any() and not time_valid.any() and not has_trajectory:
        zero = logits.sum() * 0.0
        if log_total_rate is not None:
            zero = zero + log_total_rate.sum() * 0.0
        return {"loss": zero, "event_loss": zero, "family_loss": zero,
                "time_loss": zero,
                "trajectory_loss": zero,
                "event_accuracy": zero.detach(), "time_mae_hours": zero.detach(),
                "nonstable_accuracy": zero.detach(),
                "stable_prediction_fraction": zero.detach(),
                "num_targets": event_valid.sum().detach(),
                "num_time_targets": time_valid.sum().detach(),
                "num_time_mae_targets": event_valid.sum().detach(),
                "num_nonstable_targets": event_valid.sum().detach()}

    zero = logits.sum() * 0.0
    if log_total_rate is not None:
        zero = zero + log_total_rate.sum() * 0.0
    if event_valid.any():
        selected_logits = logits[event_valid]
        selected_targets = target_set[event_valid]
        # Probability mass of the complete tied event set.  Unlike averaging
        # independent cross-entropies, this does not force simultaneous events
        # to compete as if only one could be correct.
        positive_logits = selected_logits.masked_fill(selected_targets <= 0, -torch.inf)
        event_nll = torch.logsumexp(selected_logits, dim=-1) - torch.logsumexp(
            positive_logits, dim=-1)
        if class_weights is not None:
            sample_weight = (selected_targets * class_weights).sum(-1) / selected_targets.sum(-1).clamp_min(1)
            event_nll = event_nll * sample_weight
        event_loss = event_nll.mean()
    else:
        selected_logits = logits.new_zeros((0, logits.shape[-1]))
        selected_targets = target_set.new_zeros((0, target_set.shape[-1]))
        event_loss = zero

    family_loss = zero
    if family_logits is not None and outcome_family_ids is not None and event_valid.any():
        num_families = family_logits.shape[-1]
        membership = F.one_hot(
            outcome_family_ids.to(logits.device), num_classes=num_families).to(target_set.dtype)
        family_targets = ((target_set[event_valid] @ membership) > 0).to(target_set.dtype)
        selected_family_logits = family_logits[event_valid]
        positive_family_logits = selected_family_logits.masked_fill(
            family_targets <= 0, -torch.inf)
        family_loss = (
            torch.logsumexp(selected_family_logits, dim=-1)
            - torch.logsumexp(positive_family_logits, dim=-1)
        ).mean()

    dt = target_dt_hours[time_valid].float().clamp(
        min=1.0 / 60.0, max=float(max_wait_hours))
    is_event = event_valid[time_valid]
    hazard_expected_wait = None
    if (fine_hazard_logits is not None and long_hazard_logits is not None and
            tail_mu is not None and tail_log_sigma is not None):
        # Event-specific dual-timescale hazard.  Fine bins cover 0-120 min in
        # five-minute intervals; long bins cover 2 h onward in 30-minute
        # intervals. The continuous tail models the residual wait after the
        # last long bin (24 h with the default 44-bin configuration).
        fine_width = 5.0 / 60.0
        long_width = 30.0 / 60.0
        fine_edges = fine_width * torch.arange(
            fine_hazard_logits.shape[-1] + 1, device=dt.device, dtype=dt.dtype)
        long_edges = 2.0 + long_width * torch.arange(
            long_hazard_logits.shape[-1] + 1, device=dt.device, dtype=dt.dtype)
        long_end = float(long_edges[-1])
        targets_for_time = target_set[time_valid].float()
        class_count = targets_for_time.sum(-1).clamp_min(1.0)
        fine_logits = fine_hazard_logits[time_valid].float()
        long_logits = long_hazard_logits[time_valid].float()
        fine_bin = torch.bucketize(dt, fine_edges[1:-1]).clamp_max(fine_logits.shape[-1] - 1)
        long_bin = torch.bucketize(dt, long_edges[1:-1]).clamp_max(long_logits.shape[-1] - 1)

        fine_log_survival = F.logsigmoid(-fine_logits)
        fine_cumulative = fine_log_survival.cumsum(-1)
        fine_index = fine_bin[:, None, None].expand(-1, fine_logits.shape[1], 1)
        fine_log_hazard = F.logsigmoid(torch.gather(fine_logits, -1, fine_index).squeeze(-1))
        fine_previous = torch.gather(
            fine_cumulative, -1, (fine_bin - 1).clamp_min(0)[:, None, None]
            .expand(-1, fine_logits.shape[1], 1)).squeeze(-1)
        fine_previous = torch.where(fine_bin[:, None] > 0, fine_previous,
                                    torch.zeros_like(fine_previous))
        fine_event_nll = -((fine_log_hazard + fine_previous) * targets_for_time).sum(-1) / class_count

        long_log_survival = F.logsigmoid(-long_logits)
        long_cumulative = long_log_survival.cumsum(-1)
        long_index = long_bin[:, None, None].expand(-1, long_logits.shape[1], 1)
        long_log_hazard = F.logsigmoid(torch.gather(long_logits, -1, long_index).squeeze(-1))
        long_previous = torch.gather(
            long_cumulative, -1, (long_bin - 1).clamp_min(0)[:, None, None]
            .expand(-1, long_logits.shape[1], 1)).squeeze(-1)
        long_previous = torch.where(long_bin[:, None] > 0, long_previous,
                                    torch.zeros_like(long_previous))
        fine_to_long = fine_log_survival.sum(-1)
        long_event_nll = -((long_log_hazard + long_previous + fine_to_long) * targets_for_time).sum(-1) / class_count

        tail_dt = (dt - long_end).clamp_min(1.0 / 60.0)
        tail_sigma = tail_log_sigma[time_valid].float().exp()
        tail_mu_selected = tail_mu[time_valid].float()
        tail_z = (tail_dt[:, None].log() - tail_mu_selected) / tail_sigma
        tail_event_nll = ((0.5 * tail_z.square() + tail_log_sigma[time_valid].float() +
                           tail_dt[:, None].log() + 0.5 * math.log(2 * math.pi)) *
                          targets_for_time).sum(-1) / class_count
        long_to_tail = long_log_survival.sum(-1)
        tail_log_survival = torch.log((0.5 * torch.erfc(tail_z / math.sqrt(2.0))).clamp_min(1e-7))
        tail_event_nll = tail_event_nll - ((long_to_tail + fine_to_long + tail_log_survival) *
                                           targets_for_time).sum(-1) / class_count

        fine_rows = dt < 2.0
        long_rows = (dt >= 2.0) & (dt < long_end)
        tail_rows = dt >= long_end
        # Censored rows contribute survival-only terms across all event types.
        censor = ~is_event
        censor_nll = torch.zeros_like(dt)
        censor_nll = torch.where(fine_rows, -fine_previous.mean(-1), censor_nll)
        censor_nll = torch.where(long_rows, -(fine_to_long.mean(-1) + long_previous.mean(-1)), censor_nll)
        censor_nll = torch.where(tail_rows, -(fine_to_long.mean(-1) + long_to_tail.mean(-1) + tail_log_survival.mean(-1)), censor_nll)
        time_nll = torch.where(is_event & fine_rows, fine_event_nll,
                               torch.where(is_event & long_rows, long_event_nll,
                                           torch.where(is_event & tail_rows, tail_event_nll, censor_nll)))
        time_loss = time_nll.mean() if time_nll.numel() else zero
        log_rate = None
        mu = dt.new_zeros(dt.shape)
        sigma = dt.new_ones(dt.shape)
        # Decode an event-specific expected waiting time for diagnostics.  It
        # is not used to replace the hazard likelihood: each event keeps its
        # own discrete survival curve and continuous tail.
        fine_probability = torch.sigmoid(fine_logits)
        fine_previous_survival = torch.exp(torch.cat((
            torch.zeros_like(fine_log_survival[..., :1]),
            fine_log_survival[..., :-1].cumsum(-1)), dim=-1))
        fine_midpoints = (fine_edges[:-1] + fine_edges[1:]) / 2.0
        fine_mean = (fine_previous_survival * fine_probability * fine_midpoints).sum(-1)
        survival_after_fine = torch.exp(fine_log_survival.sum(-1))
        long_probability = torch.sigmoid(long_logits)
        long_previous_survival = torch.exp(torch.cat((
            torch.zeros_like(long_log_survival[..., :1]),
            long_log_survival[..., :-1].cumsum(-1)), dim=-1))
        long_midpoints = (long_edges[:-1] + long_edges[1:]) / 2.0
        long_mean = (long_previous_survival * long_probability * long_midpoints).sum(-1)
        survival_after_long = torch.exp(long_log_survival.sum(-1))
        tail_mean = 6.0 + torch.exp(tail_mu[time_valid].float() +
                                    0.5 * tail_log_sigma[time_valid].float().exp().square())
        expected_by_event = fine_mean + survival_after_fine * (
            long_mean + survival_after_long * tail_mean)
        hazard_expected_wait = torch.where(
            is_event,
            (expected_by_event * targets_for_time).sum(-1) / class_count,
            expected_by_event.mean(-1),
        ).clamp_max(max_wait_hours)
    elif time_mu is not None and time_log_sigma is not None:
        mu = time_mu[time_valid].float()
        log_sigma = time_log_sigma[time_valid].float().clamp(-3.0, 2.0)
        if (family_time_mu is not None and family_time_log_sigma is not None and
                outcome_family_ids is not None and is_event.any()):
            membership = F.one_hot(
                outcome_family_ids.to(logits.device),
                num_classes=family_time_mu.shape[-1]).to(target_set.dtype)
            time_family_targets = (target_set[time_valid] @ membership).clamp_max(1.0)
            weights = time_family_targets / time_family_targets.sum(-1, keepdim=True).clamp_min(1.0)
            conditional_mu = (family_time_mu[time_valid].float() * weights).sum(-1)
            conditional_log_sigma = (
                family_time_log_sigma[time_valid].float() * weights).sum(-1).clamp(-3.0, 2.0)
            mu = torch.where(is_event, conditional_mu, mu)
            log_sigma = torch.where(is_event, conditional_log_sigma, log_sigma)
        sigma = log_sigma.exp()
        log_dt = dt.log()
        z = (log_dt - mu) / sigma
        event_time_nll = (0.5 * z.square() + log_sigma + log_dt +
                          0.5 * math.log(2 * math.pi))
        survival = (0.5 * torch.erfc(z / math.sqrt(2.0))).clamp_min(1e-7)
        time_nll = torch.where(is_event, event_time_nll, -survival.log())
        log_rate = None
    elif log_total_rate is None:
        time_logits = logits[time_valid]
        log_rate = torch.logsumexp(time_logits, dim=-1)
    else:
        log_rate = log_total_rate[time_valid]
    if log_rate is not None:
        log_rate = log_rate.clamp(min=-16.0, max=8.0)
        time_nll = torch.exp(log_rate) * dt - log_rate * is_event.float()
    time_loss = time_nll.mean() if time_nll.numel() else zero

    trajectory_loss = zero
    if (trajectory_logits is not None and trajectory_target is not None and
            trajectory_mask is not None and trajectory_mask.any()):
        valid = trajectory_mask.bool().unsqueeze(-1).expand_as(trajectory_target)
        raw = F.binary_cross_entropy_with_logits(
            trajectory_logits[valid], trajectory_target[valid], reduction="none")
        probability = torch.sigmoid(trajectory_logits[valid])
        target = trajectory_target[valid]
        pt = torch.where(target > 0, probability, 1.0 - probability)
        alpha = torch.where(target > 0, 0.75, 0.25)
        trajectory_loss = (alpha * (1.0 - pt).square() * raw).mean()
    total = (event_loss + float(family_weight) * family_loss +
             float(time_weight) * time_loss +
             float(trajectory_weight) * trajectory_loss)

    top = selected_logits.argmax(-1) if len(selected_logits) else torch.empty(
        0, dtype=torch.long, device=logits.device)
    hit = (selected_targets.gather(1, top[:, None]).squeeze(1) > 0
           if len(selected_logits) else torch.empty(0, dtype=torch.bool, device=logits.device))
    if stable_index >= 0:
        nonstable = selected_targets.sum(-1).bool() & ~selected_targets[:, stable_index].bool()
        stable_fraction = (top == stable_index).float().mean() if len(top) else zero.detach()
    else:
        nonstable = torch.ones(len(selected_targets), dtype=torch.bool, device=logits.device)
        stable_fraction = zero.detach()
    nonstable_accuracy = hit[nonstable].float().mean() if nonstable.any() else hit.float().sum() * 0.0
    if hazard_expected_wait is not None:
        expected_wait = hazard_expected_wait
    elif time_mu is not None and time_log_sigma is not None:
        expected_wait = torch.exp(mu + 0.5 * sigma.square()).clamp_max(max_wait_hours)
    else:
        expected_wait = torch.exp(-log_rate)
    event_wait_mae = ((expected_wait[is_event] - dt[is_event]).abs().mean()
                      if is_event.any() else zero.detach())
    return {
        "loss": total,
        "event_loss": event_loss,
        "family_loss": family_loss,
        "time_loss": time_loss,
        "trajectory_loss": trajectory_loss,
        "event_accuracy": hit.float().mean().detach() if len(hit) else zero.detach(),
        "nonstable_accuracy": nonstable_accuracy.detach(),
        "stable_prediction_fraction": stable_fraction.detach(),
        "time_mae_hours": event_wait_mae.detach(),
        "num_targets": event_valid.sum().detach(),
        "num_time_targets": time_valid.sum().detach(),
        "num_time_mae_targets": is_event.sum().detach(),
        "num_nonstable_targets": nonstable.sum().detach(),
    }


def masked_event_loss(token_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross entropy over positions selected for bidirectional event masking."""
    valid = labels >= 0
    if not valid.any():
        return token_logits.sum() * 0.0
    return F.cross_entropy(token_logits[valid], labels[valid])


def masked_value_loss(value_prediction: torch.Tensor, labels: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
    """Robustly reconstruct signed-log clinical values hidden from the encoder."""
    if not mask.any():
        return value_prediction.sum() * 0.0
    target = torch.sign(labels[mask]) * torch.log1p(labels[mask].abs())
    target = target.clamp(-12.0, 12.0)
    return F.smooth_l1_loss(value_prediction[mask], target)
