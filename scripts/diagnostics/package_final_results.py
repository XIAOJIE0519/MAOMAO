#!/usr/bin/env python3
"""Build a portable archive of final MAOMAO results and their provenance."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/final_experiment_results_20260923"
PACKAGE_ROOT = "MAOMAO_final_results_20260926"
ARCHIVE = ROOT / "outputs/MAOMAO_final_results_20260926.zip"


def gather() -> dict[str, Path]:
    files: dict[str, Path] = {}

    def add(path: Path, arc: str | None = None) -> None:
        if path.is_file():
            files[arc or path.relative_to(ROOT).as_posix()] = path

    def add_tree(directory: Path, suffixes: set[str] | None = None) -> None:
        if not directory.exists():
            return
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or (suffixes is not None and path.suffix.lower() not in suffixes):
                continue
            add(path)

    add(ROOT / "docs/MAOMAO_V5_FINAL_RESULTS.md")
    add(OUT / "README.md")
    add(OUT / "protocol.json")
    add(OUT / "module_completion.json")
    add_tree(OUT / "baseline_metrics", {".json", ".npz", ".pt"})
    add_tree(OUT / "baseline_metrics/common_full_validation", {".json", ".pt"})
    add_tree(OUT / "model_metrics", {".json"})
    add_tree(OUT / "module_metrics", {".json"})
    add_tree(OUT / "existing_ablation_summary")
    add_tree(OUT / "requested_50k_models", {".json", ".pt"})
    # Preserve the existing full-scale multilabel SVM and add artifacts for
    # every explicitly requested full-scale candidate.
    add_tree(OUT / "classical_ml_fullscale/linear_svm", {".json", ".pt", ".log"})
    for candidate in ("random_forest", "decision_tree", "hist_gradient_boosting", "lightgbm",
                      "catboost", "xgboost", "adaboost", "gaussian_nb"):
        add_tree(OUT / "classical_ml_fullscale" / candidate, {".json", ".log"})
    add(OUT / "classical_ml_fullscale/queue_status.json")
    add(OUT / "classical_ml_fullscale/queue_supervisor.log")
    add(OUT / "model_metrics/maomao_dual_timescale_mae_internal.json")
    add_tree(OUT / "logs", {".log", ".json", ".md", ".txt"})
    add(OUT / "classical_full_scale/manifest.json")
    add(OUT / "classical_full_scale/external/manifest.json")
    add(OUT / "full_maomao_reference/best_model.pt")
    add(OUT / "full_maomao_reference/run_config.json")
    add(OUT / "full_maomao_reference/train.log")

    module_root = ROOT / "outputs/module_ablations_richctx_20260923"
    for config in json.loads((OUT / "module_completion.json").read_text())["statuses"]:
        name = config["name"]
        run = module_root / name
        for filename in ("run_config.json", "train.log",
                         "validation_history.jsonl", "patient_validation_split.npz"):
            add(run / filename)

    # Include the fitted GRU checkpoint used in the full internal comparison.
    add(ROOT / "outputs/baseline_gru_richctx/gru_best.pt")
    # Include evaluator/report-builder sources for reproducibility.
    for filename in ("evaluate_internal_checkpoints.py", "evaluate_requested_ablation_matrix.py",
                     "evaluate_full_maomao_gru.py", "build_requested_results_report.py",
                     "run_requested_50k_models.py", "run_fullscale_linear_svm.py",
                     "run_fullscale_ml_candidate.py", "run_fullscale_ml_queue.py",
                     "integrate_requested_baselines.py", "evaluate_dual_timescale_internal.py",
                     "evaluate_common_full_validation.py", "run_xgboost_minimal_one_round.py",
                     "stop_queue_after_lightgbm.py"):
        add(ROOT / "scripts/diagnostics" / filename)

    # Include the patient-disjoint external split artifacts referenced by the
    # saved 90:10 calibration/test reports.
    for path in (
        ROOT / "outputs/external_validation_richctx_gru/val_mimic_richctx_static7_patient_split.npz",
        ROOT / "outputs/external_validation_richctx_gru/val_mover_richctx_static7_patient_split.npz",
        ROOT / "outputs/external_validation_richctx_maomao/mimic/val_mimic_richctx_static7_patient_split.npz",
        ROOT / "outputs/external_validation_richctx_maomao/mover/val_mover_richctx_static7_patient_split.npz",
    ):
        add(path)
    return files


def main() -> None:
    # Historical entry point; the current package contains only uniform 90:10 results.
    from package_uniform_final_results import main as package_current_results
    package_current_results()
    return
    files = gather()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "archive": ARCHIVE.name,
        "file_count": len(files) + 2,
        "included_files": [
            {"path": f"{PACKAGE_ROOT}/{arc}", "size_bytes": path.stat().st_size}
            for arc, path in sorted(files.items())
        ],
        "excluded_large_intermediates": [
            "raw dataset files",
            "classical_full_scale train_X.npy and validation_X.npy arrays",
            "XGBoost gradient-index cache pages",
            "large cached validation logits",
            "best_model.pt checkpoints for the seven module ablations",
            "redundant checkpoint_latest.pt and checkpoint_epoch_*.pt files",
            "large raw full-scale ML model dumps (the explicitly requested 115 KB one-round XGBoost diagnostic booster is included)",
        ],
        "xgboost_note": "The original 30-round full-scale attempt was OOM-killed after round 4. A separately requested minimal one-round fit completed on all 14,128,539 training rows and all 1,563,972 validation rows in 4,413.27 seconds; its metrics and small booster are included as a diagnostic, not as a substitute for the full 30-round baseline. Saved 50k/20k historical metrics are included separately.",
    }
    readme = """# MAOMAO final results package\n\nThis archive contains the final report, result JSON/CSV files, evaluation and training logs, split/protocol provenance, and selected non-ablation model checkpoints. It does not include checkpoints for the seven module-ablation runs.\n\nThe report distinguishes full-cohort internal results from the saved historical XGBoost result (50,000 train rows / 20,000 validation rows). The original full-scale 30-round XGBoost attempt was OOM-killed after round 4; a later minimal depth-1, max_bin-16 single-round diagnostic completed on all training and validation rows in 4,413 seconds. Its metrics and 115 KB booster are included separately and do not stand in for the full 30-round baseline. LightGBM was stopped after its 20-minute cap without metrics. The package also contains the common full patient-disjoint 90:10 validation comparison for the saved univariate, logistic, linear-SVM, ANN, and MAOMAO models, including training-only SVM/ANN refinements. The 50k-row one-epoch LSTM/CNN/RNN/GAN/AutoEncoder/ANN sweep and MAOMAO time MAE by dual-timescale range remain separate historical sections.\n\nRaw datasets, very large feature matrices, XGBoost external-memory cache pages, large cached validation logits, ablation checkpoints, large raw candidate model dumps, and redundant latest/epoch checkpoints are excluded to keep the ZIP focused on results and provenance.\n"""
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(ARCHIVE, "w", compression=ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        archive.writestr(f"{PACKAGE_ROOT}/README_PACKAGE.md", readme)
        archive.writestr(f"{PACKAGE_ROOT}/package_manifest.json",
                         json.dumps(manifest, ensure_ascii=False, indent=2))
        for arc, path in sorted(files.items()):
            archive.write(path, f"{PACKAGE_ROOT}/{arc}")
    print(json.dumps({"archive": str(ARCHIVE), "files": len(files) + 2,
                      "bytes": ARCHIVE.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
