#!/usr/bin/env python3
"""Combine contract-compatible external surgical cohorts for one frozen-model run."""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ARRAYS = {
    "token_id.bin": "int32", "time_min.bin": "float32", "value.bin": "float32",
    "has_value.bin": "uint8", "token_kind.bin": "uint8", "outcome_class.bin": "int16",
}
CONTRACT_KEYS = ("token_vocabulary", "outcome_vocabulary", "stored_outcome_vocabulary",
                 "outcome_class_remap", "num_static", "trajectory_horizons_hours")


def combine(inputs: list[Path], output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    metas = [json.loads((path / "event_sequence_meta.json").read_text()) for path in inputs]
    for meta in metas:
        if not meta.get("complete") or meta.get("split") != "external_validation":
            raise ValueError("Every input must be a complete external_validation dataset")
    for key in CONTRACT_KEYS:
        if any(meta.get(key) != metas[0].get(key) for meta in metas[1:]):
            raise ValueError(f"Input training contracts differ at {key}")

    output.mkdir(parents=True)
    pointers = [0]
    static_rows, admissions = [], []
    audit, contexts, dropped = Counter(), Counter(), Counter()
    source_manifest, source_audit = [], {}
    total_tokens = 0
    writers = {name: (output / name).open("wb") for name in ARRAYS}
    try:
        for path, meta in zip(inputs, metas):
            ptr = np.load(path / "sequence_ptr.npy", mmap_mode="r")
            static = np.load(path / "static_baseline.npy", mmap_mode="r")
            if len(ptr) != meta["num_admissions"] + 1 or static.shape != (
                    meta["num_admissions"], meta["num_static"]):
                raise ValueError(f"Sequence pointer/static alignment failed: {path}")
            local_tokens = int(ptr[-1])
            for name in ARRAYS:
                with (path / name).open("rb") as src:
                    shutil.copyfileobj(src, writers[name], length=8 * 1024 * 1024)
            pointers.extend((ptr[1:] + total_tokens).tolist())
            total_tokens += local_tokens
            static_rows.append(static)
            frame = pd.read_csv(path / "admissions.csv", dtype={"subject_id": "string"})
            frame["validation_source"] = meta["source_name"]
            admissions.append(frame)
            audit.update(meta.get("audit_counts", {}))
            contexts.update(meta.get("context_counts", {}))
            dropped.update(meta.get("dropped_context_counts", {}))
            source_manifest.append({"source_name": meta["source_name"],
                                    "path": str(path.resolve()),
                                    "num_admissions": meta["num_admissions"],
                                    "num_tokens": local_tokens})
            source_audit[meta["source_name"]] = meta.get("source_audit", {})
    finally:
        for file in writers.values():
            file.close()

    np.save(output / "sequence_ptr.npy", np.asarray(pointers, dtype=np.int64))
    np.save(output / "static_baseline.npy", np.concatenate(static_rows, axis=0).astype(np.float32))
    pd.concat(admissions, ignore_index=True).to_csv(output / "admissions.csv", index=False)
    (output / "token_vocabulary.json").write_text(
        json.dumps(metas[0]["token_vocabulary"], ensure_ascii=False, indent=2))
    meta = dict(metas[0])
    meta.update({
        "source_name": "combined_surgery_part1_part2",
        "split": "external_validation", "complete": True,
        "num_admissions": int(sum(item["num_admissions"] for item in metas)),
        "num_operation_episodes": int(sum(item.get("num_operation_episodes", item["num_admissions"])
                                           for item in metas)),
        "num_unique_admissions": int(pd.concat(admissions, ignore_index=True)["admission_id"].nunique()),
        "num_tokens": total_tokens, "source_manifest": source_manifest,
        "source_audit": source_audit, "audit_counts": dict(audit),
        "context_counts": dict(contexts), "dropped_context_counts": dict(dropped),
        "frozen_training_contract": metas[0]["frozen_training_contract"],
    })
    (output / "event_sequence_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    meta = combine(args.inputs, args.output)
    print(json.dumps({"output": str(args.output.resolve()),
                      "episodes": meta["num_admissions"], "tokens": meta["num_tokens"],
                      "sources": meta["source_manifest"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
