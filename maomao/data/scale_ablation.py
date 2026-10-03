"""Output projection without changing the next-event time or input features."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from maomao.data.event_sequence import EventSequenceDataset


class ProjectedOutcomeDataset(Dataset):
    """Keep every source window; supervise retained original next-event labels.

    Dropping an outcome must not silently change the target to a later event.
    Rows whose original next-event set has no retained label have no event or
    observed-time loss. They remain input/auxiliary/trajectory training rows.
    True episode-end censoring is retained, unlike an excluded observed event.
    All original family-history and observation input features are preserved.
    """

    def __init__(self, source: EventSequenceDataset, specification: str | Path):
        self.source = source
        spec = json.loads(Path(specification).read_text())
        names = list(spec["outcome_names"])
        original = source.meta["outcome_vocabulary"]
        self.indices = torch.tensor([original.index(name) for name in names])
        if len(set(names)) != len(names):
            raise ValueError("Duplicate projected outcomes")
        self.num_outcomes = len(names)
        self.meta = dict(source.meta)
        self.meta["outcome_vocabulary"] = names
        self.meta["outcome_to_family"] = [source.meta["outcome_to_family"][i] for i in self.indices.tolist()]
        self.meta["model_family_count"] = source.num_event_families
        self.meta["output_projection"] = spec

    def __getattr__(self, name):
        return getattr(self.source, name)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        sample = self.source[index]
        original_observed = sample["loss_mask"] & sample["target_set"].sum(-1).gt(0)
        sample["target_set"] = sample["target_set"].index_select(-1, self.indices)
        retained_observed = original_observed & sample["target_set"].sum(-1).gt(0)
        sample["loss_mask"] = retained_observed
        sample["time_mask"] &= ~original_observed | retained_observed
        sample["trajectory_target"] = sample["trajectory_target"].index_select(-1, self.indices)
        return sample
