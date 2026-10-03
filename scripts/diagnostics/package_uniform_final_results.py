#!/usr/bin/env python3
"""Refresh the current 90:10 result folder and ZIP without retired experiments."""
from __future__ import annotations

import hashlib
import argparse
import json
import re
import sys
import shutil
import fcntl
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.diagnostics.uniform_result_scope import current_xgboost_dir, current_xgboost_model, xgboost_revision, sha256
from scripts.diagnostics.build_fullscale_external_rows import SOURCES
from maomao.evaluation.softmax_calibration import PROTOCOL, LEGACY_PROTOCOL, revision_directory
OUT = ROOT / "outputs/final_experiment_results_20260923"
SOURCE_EXTERNAL = ROOT / "outputs/external_validation_final_maomao_uniform"
DEST = ROOT / "outputs/maomao_v5_final_results"
STAGE = ROOT / "outputs/maomao_v5_final_results.next"
ZIP = ROOT / "outputs/MAOMAO_final_results_20260926.zip"
MODELS = ("univariate", "logistic_regression", "xgboost", "ann", "maomao")
SITES = tuple(SOURCES)
MODULES = ("no_block_causal", "no_relative_time", "no_family_head",
           "no_clock_phase_summary", "no_measurement_intensity",
           "no_masked_event_value", "no_event_conditioned_time")


def copy(source: Path, destination: str, required: bool = True) -> None:
    if source.suffix.lower()=='.svg':return
    if not source.is_file():
        if required:
            raise FileNotFoundError(source)
        return
    target = STAGE / destination
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def package(write_zip: bool = True) -> None:
    xgb_dir = current_xgboost_dir()
    xgb_model = xgb_dir / json.loads((xgb_dir / "status.json").read_text())["model_file"]
    xgb_hash = sha256(xgb_model)
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    copy(ROOT / "docs/MAOMAO_V5_FINAL_RESULTS.md", "MAOMAO_V5_FINAL_RESULTS.md")
    for name in ("MAOMAO_DATA_CLEANING_FLOW.md", "MAOMAO_CALIBRATION_AUDIT.md"):
        text = (ROOT / "docs" / name).read_text()
        text = text.replace("../outputs/maomao_plot_sources/", "plot_sources/")
        (STAGE / name).write_text(text)
    shap_delivery=ROOT/'outputs/maomao_plot_sources/shap/delivery_verification.json'
    shap_complete=shap_delivery.exists() and json.loads(shap_delivery.read_text()).get('complete') is True
    if shap_complete:
        copy(ROOT/'docs/MAOMAO_FAMILY_SHAP.md','MAOMAO_FAMILY_SHAP.md')
    clinical_base = ROOT / "outputs/classic_score_comparison/independent_15pct_test"
    clinical_doc = (ROOT / "docs/MAOMAO_CLINICAL_SCORE_COMPARISON.md").read_text()
    clinical_doc = clinical_doc.replace("(MAOMAO_V5_FINAL_RESULTS.md)", "(../MAOMAO_V5_FINAL_RESULTS.md)")
    clinical_doc = clinical_doc.replace("../outputs/classic_score_comparison/independent_15pct_test/", "")
    clinical_doc = re.sub(r"\[([^\]]+)\]\(MAOMAO_CLASSIC_SCORE_COMPARISON_PLAN.md\)", r"\1（项目中的历史记录，未纳入本结果包）", clinical_doc)
    (STAGE / "clinical_scores").mkdir()
    (STAGE / "clinical_scores/MAOMAO_CLINICAL_SCORE_COMPARISON.md").write_text(clinical_doc)
    for source in ("clinical_score_comparison_full_matrix.csv", "split_summary.json",
                   "expanded_results/expanded_comparison_summary.json"):
        copy(clinical_base / source, f"clinical_scores/{source}")
    checkpoint = Path(json.loads((clinical_base / "expanded_results/expanded_comparison_summary.json").read_text())["checkpoint"])
    clinical_provenance = {"protocol": "patient-disjoint 85:15; sealed independent test",
                           "checkpoint_sha256": sha256(checkpoint),
                           "checkpoint_epoch": 7,
                           "checkpoint_included": False,
                           "report_source_sha256": sha256(ROOT / "docs/MAOMAO_CLINICAL_SCORE_COMPARISON.md")}
    (STAGE / "clinical_scores/provenance.json").write_text(json.dumps(clinical_provenance, indent=2) + "\n")
    copy(ROOT / "data/perioperative_event_sequences_v5_richctx_static7/event_sequence_meta.json",
         "provenance/training_event_sequence_meta.json")
    copy(ROOT / "data/perioperative_event_sequences_v5_richctx_static7/token_vocabulary.json",
         "provenance/training_token_vocabulary.json")
    copy(OUT / "full_maomao_reference/patient_validation_split.npz",
         "provenance/internal_patient_split.npz")
    for name, source in (
        ("univariate", OUT / "baseline_metrics/univariate_fullscale_internal.json"),
        ("logistic_regression", OUT / "baseline_metrics/logistic_fullscale_internal.json"),
        ("xgboost", xgb_dir / "metrics.json"),
        ("ann", OUT / "baseline_metrics/common_full_validation/ann_fullscale_refined_metrics.json"),
        ("maomao", OUT / "model_metrics/maomao_internal.json"),
    ):
        copy(source, f"internal/{name}/metrics.json")
    for name, source in (
        ("univariate", OUT / "baseline_metrics/common_full_validation/univariate_last_token_model.npz"),
        ("logistic_regression", OUT / "baseline_metrics/logistic_fullscale.pt"),
        ("xgboost", xgb_model),
        ("ann", OUT / "baseline_metrics/common_full_validation/ann_fullscale_refined.pt"),
        ("maomao", OUT / "full_maomao_reference/best_model.pt"),
    ):
        copy(source, f"models/{name}{source.suffix}")
    copy(OUT / "baseline_metrics/logistic_scaler.npz", "models/logistic_scaler.npz")
    copy(xgb_dir / "status.json",
         "internal/xgboost/status.json")
    for file in ("train_guard_record.json", "score_guard_record.json", "supervisor_status.json"):
        copy(xgb_dir / file, f"internal/xgboost/{file}", required=False)
    copy(OUT / "baseline_metrics/common_full_validation/ann_fullscale_refined_status.json",
         "internal/ann/status.json")
    copy(OUT / "full_maomao_reference/run_config.json", "internal/maomao/run_config.json")
    copy(OUT / "module_completion.json", "ablation/completion.json")
    for module in MODULES:
        copy(OUT / f"module_metrics/{module}.json", f"ablation/{module}/metrics.json")
        run = OUT / "module_ablation_runs" / module
        for source_name in ("run_config.json",):
            copy(run / source_name, f"ablation/{module}/{source_name}", required=False)
    copy(OUT / "model_metrics/maomao_dual_timescale_mae_internal.json",
         "time_scales/maomao_mae_by_scale.json")
    external_complete = True
    completed = {}
    calibration_pairs = {}
    for site in SITES:
        manifest = OUT / f"classical_full_scale/external/{site}/manifest.json"
        copy(manifest, f"external/{site}/manifest.json", required=False)
        if manifest.exists():
            payload = json.loads(manifest.read_text())
            split_path = Path(payload.get("split", {}).get("split_file", ""))
            if split_path.is_file():
                copy(split_path, f"external/{site}/patient_split.npz")
        count = 0
        for model in MODELS:
            path = SOURCE_EXTERNAL / site / model
            status = path / "status.json"
            result = path / "metrics.json"
            if (status.exists() and result.exists() and json.loads(status.read_text()).get("status") == "completed"
                    and (model != "xgboost" or json.loads(result.read_text()).get("model_sha256") == xgb_hash)):
                copy(status, f"external/{site}/{model}/status.json")
                copy(result, f"external/{site}/{model}/metrics.json")
                copy(path / "calibration_fit.json", f"external/{site}/{model}/calibration_fit.json")
                count += 1
        completed[site] = count
        if count != len(MODELS):
            external_complete = False
        raw = SOURCE_EXTERNAL / site / "maomao/metrics_uncalibrated.json"
        post = SOURCE_EXTERNAL / site / "maomao/metrics.json"
        if (raw.exists() and post.exists() and
                json.loads(raw.read_text()).get("calibrated_reference_sha256") == sha256(post)):
            calibration_pairs[site] = True
            copy(raw, f"external/{site}/maomao/metrics_uncalibrated.json")
            copy(SOURCE_EXTERNAL / site / "maomao/uncalibrated_status.json",
                 f"external/{site}/maomao/uncalibrated_status.json", required=False)
        else:
            calibration_pairs[site] = False
            external_complete = False
        # Summaries are regenerated from the packaged current revision only.
        summary = {"site":site,"status":"completed" if count == len(MODELS) else "partial","models":{}}
        for model in MODELS:
            packed = STAGE / f"external/{site}/{model}/metrics.json"
            if packed.exists():
                data = json.loads(packed.read_text())
                summary["models"][model] = {key:data[key] for key in ("micro_auprc", "micro_auroc", "test_rows")}
        target = STAGE / f"external/{site}/summary.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    copy(SOURCE_EXTERNAL / "queue_status.json", "external/queue_status.json", required=False)
    copy(ROOT / SOURCES["surgical_pooled"] / "event_sequence_meta.json",
         "provenance/three_source_pooled_meta.json")
    for source in (
        "build_fullscale_baseline_rows.py",
        "build_fullscale_external_rows.py",
        "score_uniform_external_five.py",
        "run_uniform_external_queue.py",
        "verify_uniform_final_results.py",
        "verify_external_example_portable_replot.py",
        "build_uniform_results_report.py",
        "package_uniform_final_results.py",
        "uniform_result_scope.py",
        "run_xgboost_conservative_continuation.py",
        "run_current_results_revision.py",
        "evaluate_common_full_validation.py",
        "run_baseline_comparison.py",
        "compare_clinical_scores_expanded.py",
        "export_clinical_score_comparison_matrix.py",
    ):
        copy(ROOT / "scripts/diagnostics" / source, f"scripts/{source}")
    copy(ROOT / "maomao/evaluation/event_metrics.py", "scripts/event_metrics.py")
    copy(ROOT / "maomao/data/event_sequence.py", "scripts/event_sequence.py")
    for name in ("run_softmax_calibration_revision.py", "audit_data_cleaning_flow.py", "write_data_cleaning_report.py",
                 "export_internal_plot_sources.py", "build_plot_tables_and_audit.py", "verify_plot_delivery.py"):
        copy(ROOT / "scripts/diagnostics" / name, f"scripts/{name}")
    copy(ROOT/'scripts/diagnostics/complete_external_retrieval.py','scripts/complete_external_retrieval.py')
    copy(ROOT/'scripts/diagnostics/verify_external_retrieval_intervals.py','scripts/verify_external_retrieval_intervals.py')
    for name in ('maomao_display_family_groups.py','run_maomao_family_shap.py','verify_maomao_shap.py',
                 'plot_maomao_delphi_shap.py','build_maomao_shap_report.py','maomao_shap_budget_sensitivity.py',
                 'maomao_prediction_only.py','verify_maomao_prediction_only.py','run_maomao_shap_parallel.py',
                 'prepare_maomao_shap_delivery.py','audit_maomao_shap_figures.py','verify_maomao_fixed_time_cache.py','finalize_maomao_shap_package.py'):
        copy(ROOT/'scripts/diagnostics'/name,f'scripts/{name}')
    for name in ("softmax_calibration.py", "event_bias_calibration.py", "gpu_bootstrap.py", "rank_statistics.py", "plot_sources.py"):
        copy(ROOT / "maomao/evaluation" / name, f"scripts/{name}")
    for source in ("data/preprocess_mimic_validation.py", "data/preprocess_mover_validation.py", "data/preprocess_icu_validation.py",
                   "data/preprocess_surgery_external_validation.py", "data/external_validation_common.py",
                   "maomao/data/preprocess_timeline.py", "maomao/data/clinical_events.py", "maomao/data/static_features.py", "scripts/preprocess_event_sequences.py"):
        copy(ROOT / source, f"preprocessing/{Path(source).name}")
    plot_root = ROOT / "outputs/maomao_plot_sources"
    if not (plot_root / "delivery_verification.json").exists() or not json.loads((plot_root / "delivery_verification.json").read_text()).get("complete"):
        raise RuntimeError("Plot source full-row verification is incomplete")
    for path in sorted(plot_root.rglob("*")):
        if path.is_file() and not any(part.startswith(".") or part=='__pycache__' for part in path.relative_to(plot_root).parts) and path.suffix!='.pyc':
            if path.relative_to(plot_root).parts[0]=='shap':
                if not shap_complete:continue
            if path.suffix in (".pt", ".bin") or "logits" in path.name:
                raise RuntimeError(f"Unexpected raw checkpoint/cache in plot sources: {path}")
            copy(path, f"plot_sources/{path.relative_to(plot_root).as_posix()}")
    revision=revision_directory()
    for name in ("prespecified_protocol.json", "queue_status.json"):
        copy(revision / name, f"calibration_revision/{name}")
    for name in ("ci_acceleration_record.json", "statistics_revision_record.json"):
        copy(ROOT / "outputs/calibration_softmax_revision_20260929" / name, f"calibration_revision/{name}")
    if PROTOCOL!=LEGACY_PROTOCOL:
        copy(revision/'analytic_gradient_verification.json','calibration_revision/analytic_gradient_verification.json')
        copy(revision/'promotion_verification.json','calibration_revision/promotion_verification.json')
        copy(revision/'selection_partition_verification.json','calibration_revision/selection_partition_verification.json')
        copy(SOURCE_EXTERNAL/'active_calibration_protocol.json','calibration_revision/active_calibration_protocol.json')
        for name in ('run_v5_bias_calibration_revision.py','promote_v5_bias_calibration_revision.py','verify_v5_calibration_selection.py'):
            copy(ROOT/'scripts/diagnostics'/name,f'scripts/{name}')
    scale_dir = ROOT / "outputs/scale_ablations_richctx_20260928"
    if (scale_dir / "verification.json").exists() and json.loads((scale_dir / "verification.json").read_text()).get("complete") is True:
        copy(ROOT / "docs/MAOMAO_SCALE_ABLATIONS_CURRENT.md", "scale_ablations/MAOMAO_SCALE_ABLATIONS_CURRENT.md")
        for path in sorted((scale_dir / "metrics").glob("*.json")):
            copy(path, f"scale_ablations/metrics/{path.name}")
        for path in sorted((scale_dir / "specifications").glob("*.json")):
            copy(path, f"scale_ablations/specifications/{path.name}")
        for source in ("manifest.json", "verification.json", "legacy_removal.json"):
            copy(scale_dir / source, f"scale_ablations/{source}", required=source != "legacy_removal.json")
        for name in ("model_small", "model_large", "vocab_50", "vocab_100", "vocab_150", "context_64", "context_128"):
            copy(scale_dir / f"runs/{name}/run_config.json", f"scale_ablations/runs/{name}/run_config.json")
            # No model state, optimizer state, or ablation checkpoint is packaged.
        for source in ("prepare_scale_ablations.py", "scale_ablation_scope.py", "evaluate_scale_ablation.py",
                       "verify_scale_ablations.py", "build_scale_ablation_report.py", "run_scale_ablation_queue.py",
                       "finish_scale_ablation_delivery.py", "evaluate_dual_timescale_internal.py"):
            copy(ROOT / "scripts/diagnostics" / source, f"scripts/{source}")
        copy(ROOT / "maomao/data/scale_ablation.py", "scripts/scale_ablation.py")
        copy(ROOT / "maomao/models/event_maomao.py", "scripts/event_maomao.py")
        copy(ROOT / "scripts/train.py", "scripts/train.py")

    manuscript_root=ROOT/'outputs/maomao_manuscript_figures_20260929'
    manuscript_proof=manuscript_root/'delivery_verification.json'
    manuscript=None
    if manuscript_proof.exists():
        manuscript=json.loads(manuscript_proof.read_text())
        copy(ROOT/'docs/MAOMAO_MANUSCRIPT_FIGURES.md','MAOMAO_MANUSCRIPT_FIGURES.md')
        copy(ROOT/'docs/MAOMAO_CLINICAL_COMMON_COHORT_CURVES.md','MAOMAO_CLINICAL_COMMON_COHORT_CURVES.md')
        copy(ROOT/'docs/MAOMAO_CLINICAL_CALIBRATION_DCA_AUDIT.md','MAOMAO_CLINICAL_CALIBRATION_DCA_AUDIT.md')
        if manuscript.get('external_exact_row_audits_complete'):
            copy(ROOT/'docs/MAOMAO_MANUSCRIPT_EXTERNAL_ABLATIONS.md','MAOMAO_MANUSCRIPT_EXTERNAL_ABLATIONS.md')
        for name in ('delivery_verification.json','visual_inspection.json','font_provenance.json'):
            copy(manuscript_root/name,f'manuscript_figures/{name}')
        for name in ('portable_replot_verification.json','source_preflight.json','source_preflight_review.json','external_ablation_row_verification.json','revision_requirements_verification.json'):
            copy(manuscript_root/name,f'manuscript_figures/{name}',required=False)
        copy(manuscript_root/'reference/gallery_selected.json','manuscript_figures/reference/gallery_selected.json')
        for number in manuscript['verified_figures']:
            proof=json.loads((manuscript_root/f'figures/Figure_{number}.export_verification.json').read_text())
            for ext in ('pdf','png'):
                if sha256(manuscript_root/f'figures/Figure_{number}.{ext}')!=proof[f'{ext}_sha256']:
                    raise RuntimeError(f'Manuscript Figure {number} changed after verification')
            for path in sorted((manuscript_root/'figures').glob(f'Figure_{number}.*')):
                copy(path,f'manuscript_figures/figures/{path.name}')
        for name in manuscript.get('verified_supplements',[]):
            proof=json.loads((manuscript_root/f'figures/{name}.export_verification.json').read_text())
            for ext in ('pdf','png'):
                if sha256(manuscript_root/f'figures/{name}.{ext}')!=proof[f'{ext}_sha256']:raise RuntimeError(f'{name} changed after verification')
            for path in sorted((manuscript_root/'figures').glob(f'{name}.*')):
                copy(path,f'manuscript_figures/figures/{path.name}')
        for path in sorted((manuscript_root/'source_data').rglob('*')):
            if path.is_file() and path.suffix in ('.csv','.json','.npz'):
                copy(path,f'manuscript_figures/source_data/{path.relative_to(manuscript_root/"source_data").as_posix()}')
        # Snapshot only complete before/after result pairs; omit every score cache
        # and every module/size/vocabulary checkpoint.
        copy(manuscript_root/'external_ablations/queue_status.json','manuscript_figures/external_ablations/queue_status.json')
        for status in sorted((manuscript_root/'external_ablations').glob('*/*/status.json')):
            payload=json.loads(status.read_text())
            if payload.get('status')!='completed':continue
            for path in sorted(status.parent.rglob('*.json')):
                if '.partial.' in path.name:continue
                copy(path,f'manuscript_figures/external_ablations/{path.relative_to(manuscript_root/"external_ablations").as_posix()}')
        copy(ROOT/'scripts/diagnostics/build_manuscript_figures.py','manuscript_figures/build_manuscript_figures.py')
        copy(ROOT/'scripts/diagnostics/maomao_figure_revision.py','manuscript_figures/maomao_figure_revision.py')
        copy(ROOT/'scripts/diagnostics/maomao_display_family_groups.py','manuscript_figures/maomao_display_family_groups.py')
        copy(Path('/home/yunkunshi/.codex/skills/nature-figure/scripts/audit_panel_alignment.py'),'manuscript_figures/audit_panel_alignment.py')
        copy(ROOT/'maomao/models/event_maomao.py','manuscript_figures/provenance/event_maomao.py')
        for name in ('run_manuscript_external_ablations.py','prepare_manuscript_clinical_curves.py',
                     'build_manuscript_clinical_report.py','verify_manuscript_figures.py','render_manuscript_after_external_queue.py',
                     'verify_manuscript_external_ablation_rows.py','build_manuscript_external_ablation_report.py',
                     'verify_manuscript_portable_replot.py','prepare_manuscript_revision_sources.py','verify_manuscript_revision_sources.py',
                     'prepare_clinical_maomao_calibration_inputs.py','audit_clinical_calibration_dca.py'):
            copy(ROOT/'scripts/diagnostics'/name,f'scripts/{name}')
        (STAGE/'manuscript_figures/requirements.txt').write_text('numpy==2.2.6\npandas==3.0.5\nmatplotlib==3.10.6\npymupdf\nfonttools\n')
        (STAGE/'manuscript_figures/README.md').write_text(
            '# A4-width MAOMAO manuscript figures\n\n'
            'Run python build_manuscript_figures.py --figures 1 2 3 4 5 from this directory to replot all delivered panels. '
            'Figure 3 requires delivery_verification.json to confirm all external jobs and final figure checks are complete. '
            'Install genuine Arial Regular/Bold from a licensed source; the script refuses font substitution. '
            'TTF font files are not redistributed. PDF retains vector marks and text; PNG is 300 dpi; SVG images are excluded. '
            'The command also generates S3, ten S4 endpoint pages and the 3×3 S5 waterfall supplement.\n\n'
            'source_data/ includes exact clinical curves on common complete cases and paired patient confidence intervals. '
            'external_ablations/ is a snapshot of completed before/after pairs only, with row-count, model-hash and prediction-equivalence evidence. '
            'external_ablation_row_verification.json and source_data/external_row_audits/ independently check frozen patient splits, full original row coordinates and matched projected-task row hashes. '
            'No ablation checkpoints or intermediate logits are included. See ../MAOMAO_MANUSCRIPT_FIGURES.md for distinct cohort protocols.\n')

    readme = (
        "# MAOMAO 当前统一 9:1 结果\n\n"
        "内部五模型使用完整训练目标行和完整患者级 10% 验证目标行。"
        "七个独立外部数据集和三源合并集各使用患者级 90% 校准与 10% 封存测试，"
        "每个已完成的模型均对完整封存测试目标行评分。小型NTUH/ASAC/UQ按源记录代理ID划分，未提供跨记录自然人链接。\n\n"
        f"外部完成数：{json.dumps(completed, ensure_ascii=False)}。\n"
        + ("八个外部评估队列的五模型和 MAOMAO 校准前后结果已全部完成。\n" if external_complete
           else "外部结果仍在计算；本包会随新结果更新。\n")
        + "\nMAOMAO 校准前和校准后直接显示在外部模型比较表中。"
          "clinical_scores/ 包含临床评分文档、42 组配对矩阵、汇总 JSON、划分记录与模型来源摘要；"
          "该独立研究使用 INSPIRE 85:15 封存测试协议和独立 MAOMAO checkpoint。\n"
          "\n本包包含当前五模型、七项模块消融和分尺度时间误差，"
          "不含先前小样本实验、其他模型候选、消融 checkpoint、训练过程日志或大型中间特征矩阵。\n"
    )
    if (STAGE / "scale_ablations/manifest.json").exists():
        readme += "\nscale_ablations/ 包含当前数据下的模型规模、50/100/150 类输出、64/128 事件局部上下文段实验，以及复用 MAOMAO 的相同任务参照、95% CI 和分尺度 MAE。\n"
    readme += "\n新增MAOMAO_DATA_CLEANING_FLOW.md和MAOMAO_CALIBRATION_AUDIT.md。plot_sources/含完整评估行的图源汇总、分尺度时间向量、临床去标识预测和可独立运行的绘图脚本，详见其README。ROC/PR直方图曲线积分是近似值，报告点估计以精确JSON为准。\n"
    readme += "\n外部大表显示精确事件与63-family的Hit@1/5/10、All-hit@10及95%CI，MAOMAO校正前后同时列出；不展示17临床大类的Hit表。完整top10预测/真实多标签与已有12项命中向量均保留在图源中。\n"
    if shap_complete:
        readme += "\nMAOMAO_FAMILY_SHAP.md及plot_sources/shap/包含全部9,989验证患者各一个查询的真实SHAP、Figure4a嵌入图、三个时间尺度（<2h、2–<24h、≥24h）的Figure4c、17临床大类/63-family补充图与完整贡献源数据。无需checkpoint即可独立重绘；Partition贡献为有限预算近似，局部预算敏感性如实记录。\n"
    if manuscript:
        readme += (f'\n新增 manuscript_figures/：已核查 Figure {manuscript["verified_figures"]} 的 Arial 8/10 pt、210 mm 宽、严格矢量 PDF 及 PNG；不含 SVG 图片。'
                   + ('Figure 1–5 全部完成。\n' if manuscript['complete'] else 'Figure 3 新增外部消融及最终图尚未完成，当前包不代表五张稿件图全部完成。\n'))
    (STAGE / "README.md").write_text(readme)
    entries = []
    for path in sorted(STAGE.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            entries.append({"path": path.relative_to(STAGE).as_posix(),
                            "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "protocol": "full-row 90:10; patient IDs where provided, source record proxies otherwise",
                "small_source_record_proxy_cohorts": ["ntuh", "asac", "uq", "surgical_pooled"],
                "calibration_protocol": PROTOCOL,
                "rank_metric_definition": "threshold_grouped_ap_midrank_auroc_float64_v2",
                "clinical_score_protocol": "patient-disjoint 85:15; sealed independent test",
                "external_complete": external_complete,
                "completed_models_by_site": completed,
                "maomao_before_after_pairs_by_site": calibration_pairs,
                "external_retrieval_metrics": "48 states × 12 metrics; exact events, 63 semantic families, 17 clinical display groups",
                "shap_full_eligible_patient_complete": shap_complete,
                "manuscript_figures_complete": bool(manuscript and manuscript['complete']),
                "manuscript_verified_figures": manuscript['verified_figures'] if manuscript else [],
                "manuscript_verified_supplements": manuscript.get('verified_supplements',[]) if manuscript else [],
                "svg_images_excluded": True,
                "files": entries}
    (STAGE / "package_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    previous = DEST.with_suffix(".previous")
    if previous.exists():
        shutil.rmtree(previous)
    if DEST.exists():
        DEST.replace(previous)
    STAGE.replace(DEST)
    if previous.exists():
        shutil.rmtree(previous)
    if not write_zip:
        print(json.dumps({'folder':str(DEST),'folder_only':True,'external_complete':external_complete},ensure_ascii=False))
        return
    temp_zip = ZIP.with_suffix(".zip.tmp")
    with ZipFile(temp_zip, "w", compression=ZIP_DEFLATED,
                 compresslevel=1, allowZip64=True) as archive:
        for path in sorted(DEST.rglob("*")):
            if path.is_file():
                archive.write(path, f"MAOMAO_uniform_90_10/{path.relative_to(DEST).as_posix()}")
    temp_zip.replace(ZIP)
    print(json.dumps({"folder": str(DEST), "archive": str(ZIP),
                      "external_complete": external_complete,
                      "completed_models_by_site": completed,
                "maomao_before_after_pairs_by_site": calibration_pairs,
                      "archive_bytes": ZIP.stat().st_size}, ensure_ascii=False))


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--folder-only',action='store_true');args=parser.parse_args()
    lock_path = ROOT / "outputs/.maomao_uniform_package.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            package(write_zip=not args.folder_only)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
