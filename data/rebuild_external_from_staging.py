#!/usr/bin/env python3
"""Rewrite an external event artifact after a frozen-contract mapping update."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.external_validation_common import write_validation_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing", type=Path, required=True)
    parser.add_argument("--stage_db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training_dir", type=Path,
                        default=Path("data/perioperative_event_sequences_v5_full"))
    args = parser.parse_args()
    old_meta = json.loads((args.existing / "event_sequence_meta.json").read_text())
    episodes = pd.read_csv(args.existing / "admissions.csv", low_memory=False)
    result = write_validation_dataset(
        episodes=episodes,
        store_path=args.stage_db,
        output_dir=args.output,
        training_dir=args.training_dir,
        source_name=old_meta["source_name"],
        source_manifest=old_meta.get("source_manifest", []),
        source_audit=old_meta.get("source_audit", {}),
    )
    print(json.dumps({
        "complete": result["complete"],
        "output": str(args.output),
        "episodes": result["num_admissions"],
        "tokens": result["num_tokens"],
        "medication_mapping_audit": result["medication_mapping_audit"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
