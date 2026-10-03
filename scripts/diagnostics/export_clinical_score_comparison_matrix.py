#!/usr/bin/env python3
"""Flatten the current INSPIRE-only score-comparison JSON into one CSV."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/classic_score_comparison/independent_15pct_test"
SOURCES = {
    "INSPIRE sealed patient-disjoint test": BASE / "expanded_results/expanded_comparison_summary.json",
}


def add_metric(row: dict, prefix: str, item: dict) -> None:
    row[f"{prefix}_estimate"] = item["estimate"]
    row[f"{prefix}_ci95_low"] = item["ci95"][0]
    row[f"{prefix}_ci95_high"] = item["ci95"][1]


def main() -> None:
    rows: list[dict] = []
    for dataset_name, path in SOURCES.items():
        if not path.exists():
            raise FileNotFoundError(path)
        obj = json.loads(path.read_text())
        for endpoint, comparisons in obj["comparisons_by_endpoint"].items():
            for result in comparisons:
                name = result["comparator"]
                metrics = result["metrics_patient_cluster_bootstrap"]
                row = {
                    "dataset": dataset_name,
                    "endpoint": endpoint,
                    "comparator": name,
                    "n_episodes": result["n_episodes"],
                    "n_patients": result["n_patients"],
                    "events": result["events"],
                    "event_rate": result["event_rate"],
                }
                add_metric(row, "comparator_auroc", metrics[name]["auroc"])
                add_metric(row, "maomao_auroc", metrics["MAOMAO"]["auroc"])
                add_metric(row, "delta_auroc_comparator_minus_maomao",
                           metrics["score_minus_maomao"][name]["auroc"])
                add_metric(row, "comparator_auprc", metrics[name]["auprc"])
                add_metric(row, "maomao_auprc", metrics["MAOMAO"]["auprc"])
                add_metric(row, "delta_auprc_comparator_minus_maomao",
                           metrics["score_minus_maomao"][name]["auprc"])
                rows.append(row)
    out = pd.DataFrame(rows)
    target = BASE / "clinical_score_comparison_full_matrix.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False)
    print(f"wrote {len(out)} paired score-endpoint rows: {target}")


if __name__ == "__main__":
    main()
