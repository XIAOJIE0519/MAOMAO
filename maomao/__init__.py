"""Sparse perioperative event identity and event-time prediction."""

MODEL_NAME = "MAOMAO"
MODEL_FULL_NAME = "Multi-horizon Anticipatory Outcome Model for Anesthesia and Operations"

__version__ = "4.0.0"
__author__ = "Clinical AI Research Team"

from .data.event_sequence import EventSequenceDataset, collate_event_sequences
from .models.event_maomao import EventMAOMAO, event_time_loss, masked_event_loss

__all__ = [
    "EventMAOMAO",
    "event_time_loss",
    "masked_event_loss",
    "EventSequenceDataset",
    "collate_event_sequences",
]
