"""Canonical sparse perioperative event-time MAOMAO pipeline.

Preprocess::

    python3 scripts/preprocess_event_sequences.py \
        --input data/admission_timeline_v6 \
        --source_root data/train_data \
        --output data/perioperative_event_sequences_v4 \
        --max_outcomes 100

Train or resume::

    python3 scripts/train.py \
        --data_dir data/perioperative_event_sequences_v4 \
        --output_dir outputs/event_maomao_multitask_100 \
        --epochs 100 \
        --resume auto --require_cuda
"""

__version__ = "4.0.0"
__author__ = "Clinical AI Research Team"
