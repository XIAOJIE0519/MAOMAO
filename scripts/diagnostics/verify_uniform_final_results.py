#!/usr/bin/env python3
"""Audit the current five-model 90:10 result set and its deliverables.

This is deliberately read-only.  A running queue is reported as incomplete;
only --require-complete turns incompleteness into a nonzero exit status.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.diagnostics.build_fullscale_external_rows import SOURCES
from scripts.diagnostics.uniform_result_scope import current_xgboost_dir, xgboost_revision, XGB_CANDIDATE, sha256
from maomao.evaluation.rank_statistics import RANK_VERSION
from maomao.evaluation.softmax_calibration import PROTOCOL, LEGACY_PROTOCOL
from maomao.evaluation.event_bias_calibration import validate_fit

BASE = ROOT / "outputs/final_experiment_results_20260923"
EXTERNAL = ROOT / "outputs/external_validation_final_maomao_uniform"
ROWS = BASE / "classical_full_scale/external"
FOLDER = ROOT / "outputs/maomao_v5_final_results"
ZIP = ROOT / "outputs/MAOMAO_final_results_20260926.zip"
REPORT = ROOT / "docs/MAOMAO_V5_FINAL_RESULTS.md"
MODELS = ("univariate", "logistic_regression", "xgboost", "ann", "maomao")
MODEL_LABELS = ("单变量", "逻辑回归", "XGBoost", "ANN", "MAOMAO")
SITE_LABELS = {"mimic": "MIMIC", "mover": "MOVER", "eicu": "eICU",
               "sicdb": "SICdb", "ntuh": "NTUH",
               "asac": "Auckland ASAC EDS", "uq": "UQ Vital Signs",
               "surgical_pooled": "三源合并汇总"}
METRICS = ("micro_auprc", "macro_auprc", "micro_auroc", "macro_auroc",
           "mrr", "brier", "ece", "hit_at_1", "recall_at_5", "recall_at_10")
MODULES = ("no_block_causal", "no_relative_time", "no_family_head",
           "no_clock_phase_summary", "no_measurement_intensity",
           "no_masked_event_value", "no_event_conditioned_time")
ABLATION_CONFIG_CHANGES = {
    "no_block_causal": {"same_time_block_causal": False},
    "no_relative_time": {"relative_time_attention": False},
    "no_family_head": {"use_family_head": False},
    "no_clock_phase_summary": {"phase_memory": False, "clock_phase_context": False},
    "no_measurement_intensity": {"observation_intensity": False},
    "no_masked_event_value": {"masked_event_loss_weight": 0.0,
                              "masked_value_loss_weight": 0.0},
    "no_event_conditioned_time": {"event_conditioned_time_head": False},
}
TRAIN_ROWS = 14_128_539
VALID_ROWS = 1_563_972


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def patient_ids(source: Path, indices: np.ndarray) -> set[str]:
    with (source / "admissions.csv").open(newline="") as stream:
        admissions = [row["subject_id"] for row in csv.DictReader(stream)]
    return {admissions[int(index)] for index in indices}


def check_metrics(data: dict, description: str, errors: list[str],
                  require_ci: bool, expected_rows: int,
                  allow_undefined_macro_auroc: bool = False,
                  large_ci_rows: int = 100_000) -> None:
    if data.get("rank_metric_definition") != RANK_VERSION:
        errors.append(f"{description}: old or missing AP/AUROC tie definition")
    if data.get("event_targets") != expected_rows:
        errors.append(f"{description}: event target count mismatch")
    if require_ci:
        large_cohort_ci = (data.get("ci_bootstrap_repeats") == 200 and
                           data.get("ci_cohort_rows") == expected_rows and
                           data.get("ci_rows") == large_ci_rows and
                           data.get("ci_method") ==
                           "uniform row subsample bootstrap; interval deviations scaled by sqrt(sample_n/cohort_n)")
        small_cohort_ci = (data.get("ci_rows") == expected_rows and
                           data.get("ci_method") == "row bootstrap on full test cohort")
        if not (large_cohort_ci or small_cohort_ci):
            errors.append(f"{description}: 95% CI cohort/repeat count mismatch")
    for key in METRICS:
        value = data.get(key)
        if value is None:
            if not (allow_undefined_macro_auroc and key == "macro_auroc"):
                errors.append(f"{description}: missing {key}")
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            errors.append(f"{description}: invalid {key}")
            continue
        ci = data.get(key + "_95ci")
        if require_ci and (not isinstance(ci, list) or len(ci) != 2 or
                           not all(isinstance(x, (int, float)) and math.isfinite(x)
                                   for x in ci) or ci[0] > ci[1]):
            errors.append(f"{description}: missing/invalid {key} 95% CI")


def check_external_retrieval(data,description,errors,n):
    from scripts.diagnostics.complete_external_retrieval import KEYS,VERSION
    if data.get('retrieval_protocol')!=VERSION or data.get('retrieval_rows')!=n:
        errors.append(f'{description}: full-row event/family retrieval missing')
    for key in KEYS:
        value=data.get(key);ci=data.get(key+'_95ci')
        if not isinstance(value,(int,float)) or not 0<=value<=1 or not isinstance(ci,list) or len(ci)!=2 or not 0<=ci[0]<=ci[1]<=1:
            errors.append(f'{description}: missing {key}/CI')
    if all(data.get(k) is not None for k in KEYS):
        if not data['hit_at_1']<=data['hit_at_5']<=data['hit_at_10'] or data['all_true_events_hit_at_10']>data['hit_at_10']:
            errors.append(f'{description}: event retrieval ordering invalid')
        if not data['same_family_hit_at_1']<=data['same_family_hit_at_5']<=data['same_family_hit_at_10'] or data['all_true_families_hit_at_10']>data['same_family_hit_at_10']:
            errors.append(f'{description}: family retrieval ordering invalid')
        if any(data[f'same_family_hit_at_{k}']+2e-7<data[f'hit_at_{k}'] for k in (1,5,10)) or data['all_true_families_hit_at_10']<data['all_true_events_hit_at_10']:
            errors.append(f'{description}: family covers fewer rows than exact events')
        if any(data[f'clinical_group_hit_at_{k}']<data[f'same_family_hit_at_{k}'] for k in (1,5,10)) or data['all_true_clinical_groups_hit_at_10']<data['all_true_families_hit_at_10']:
            errors.append(f'{description}: coarse clinical grouping covers fewer rows than finer family')

def audit() -> dict:
    errors: list[str] = []
    pending: list[str] = []
    complete: dict[str, int] = {}
    raw_complete: dict[str, bool] = {}
    xgb_dir = current_xgboost_dir()
    if xgb_dir != XGB_CANDIDATE:
        pending.append("xgboost: additional bounded rounds not yet promoted")
    pooled_meta = read(ROOT / SOURCES["surgical_pooled"] / "event_sequence_meta.json")
    names = {item["source_name"] for item in pooled_meta["source_manifest"]}
    if (names != {"ntuh110_ecg_derived_hr", "auckland_asac_eds25_surgical", "uq_vital_signs32_surgical"}
            or pooled_meta.get("num_admissions") != 165):
        errors.append("pooled cohort: source membership differs from NTUH/ASAC/UQ")
    if (EXTERNAL / "evaluation").exists():
        errors.append("external: non-90:10 legacy evaluation folder is in the uniform result tree")

    split_path = BASE / "full_maomao_reference/patient_validation_split.npz"
    with np.load(split_path) as saved:
        train_patients = set(saved["train_patients"].tolist())
        valid_patients = set(saved["validation_patients"].tolist())
        train_windows = saved["train_window_indices"]
        valid_windows = saved["validation_window_indices"]
    if train_patients & valid_patients or len(train_patients) != 89_897 or len(valid_patients) != 9_989:
        errors.append("internal: patient split is not disjoint 90:10")
    if len(set(train_windows.tolist()) & set(valid_windows.tolist())):
        errors.append("internal: training and validation windows overlap")
    common = read(BASE / "baseline_metrics/common_full_validation/current_five_internal_metrics.json")
    if (common.get("status") != "completed" or common.get("train_target_rows") != TRAIN_ROWS or
            common.get("validation_target_rows") != VALID_ROWS):
        errors.append("internal: common full-row manifest mismatch")
    if set(common.get("models", {})) != {"univariate", "logistic_regression", "ann_fullscale_refined", "maomao"}:
        errors.append("internal: current manifest includes missing or retired models")
    internal = {key: common.get("models", {}).get(key) for key in
                ("univariate", "logistic_regression", "ann_fullscale_refined", "maomao")}
    xgb = read(xgb_dir / "metrics.json")
    internal["xgboost"] = xgb
    for key, data in internal.items():
        if not data:
            errors.append(f"internal: missing {key} metrics")
            continue
        rows = data.get("rows") or data.get("evaluation_rows") or data.get("validation_rows") or data.get("event_targets")
        if rows != VALID_ROWS or data.get("train_rows", data.get("train_rows_full")) != TRAIN_ROWS:
            errors.append(f"internal/{key}: incomplete target rows")
        check_metrics(data, f"internal/{key}", errors, require_ci=True,
                      expected_rows=VALID_ROWS,
                      large_ci_rows=30_000 if key == "maomao" else 100_000)
    if internal.get("maomao") and (internal["maomao"].get("evaluation_rows_match_classical") is not True or
                                 internal["maomao"].get("evaluation_rows_full_patient_validation") is not True or
                                 internal["maomao"].get("evaluation_row_selection") != "full_validation"):
        errors.append("maomao: full-validation row alignment metadata mismatch")
    maomao_source = read(BASE / "model_metrics/maomao_internal.json")
    if internal.get("maomao"):
        for key in ("event_targets", *METRICS, *(metric + "_95ci" for metric in METRICS)):
            if maomao_source.get(key) != internal["maomao"].get(key):
                errors.append(f"internal/maomao: common manifest and source differ on {key}")
    xgb_status = read(xgb_dir / "status.json")
    if xgb_status.get("status") != "completed" or not 1 <= xgb_status.get("rounds", 0) <= 3:
        errors.append("internal/xgboost: current model status mismatch")
    if xgb_dir == XGB_CANDIDATE:
        current_xgb = read(xgb_dir / "metrics.json")
        if (xgb_status.get("rounds", 0) < 2 or
                current_xgb.get("epochs_or_rounds") != xgb_status.get("rounds") or
                current_xgb.get("model_sha256") != xgboost_revision() or
                current_xgb.get("native_merge_margin_integrity_verified") is not True or
                xgb_status.get("native_merge_margin_integrity_verified") is not True):
            errors.append("internal/xgboost: extra-round native model integrity/provenance missing")
        guard = read(xgb_dir / "train_guard_record.json")
        if (guard.get("elapsed_seconds",float("inf")) > 3615 or
                guard.get("peak_anonymous_gib",float("inf")) > 41):
            errors.append("internal/xgboost: bounded training time/memory evidence mismatch")
    full_config = read(BASE / "full_maomao_reference/run_config.json")
    full_log = (BASE / "full_maomao_reference/train.log").read_text()
    if (full_config.get("dynamic_windows") is not False or
            full_config.get("validation_fraction") != 0.1 or
            full_config.get("validation_max_windows") != 0 or
            full_config.get("max_steps") != 0 or full_config.get("epochs") != 7 or
            f"train_windows={len(train_windows)} validation_windows={len(valid_windows)}" not in full_log or
            any(f"epoch={epoch} VALIDATION" not in full_log for epoch in range(1, 8))):
        errors.append("internal/maomao: full-window seven-epoch training evidence missing")
    for module in MODULES:
        path = BASE / "module_metrics" / f"{module}.json"
        if not path.exists():
            errors.append(f"ablation/{module}: missing")
            continue
        data = read(path)
        if data.get("event_targets") != VALID_ROWS:
            errors.append(f"ablation/{module}: not full validation")
        if (data.get("evaluation_rows_match_classical") is not True or
                data.get("evaluation_rows_full_patient_validation") is not True or
                data.get("evaluation_row_selection") != "full_validation"):
            errors.append(f"ablation/{module}: full-validation row alignment metadata mismatch")
        check_metrics(data, f"ablation/{module}", errors, require_ci=True,
                      expected_rows=VALID_ROWS, large_ci_rows=30_000)
        run_dir = ROOT / "outputs/module_ablations_richctx_20260923" / module
        config = read(run_dir / "run_config.json")
        differences = {key: config.get(key) for key in set(full_config) | set(config)
                       if full_config.get(key) != config.get(key) and key not in {"data_dir", "output_dir"}}
        if (differences != ABLATION_CONFIG_CHANGES[module] or
                config.get("dynamic_windows") is not False or
                config.get("validation_fraction") != 0.1 or
                config.get("validation_max_windows") != 0 or
                config.get("max_steps") != 0 or config.get("epochs") != 7):
            errors.append(f"ablation/{module}: configuration differs beyond specified removal")
        log = (run_dir / "train.log").read_text()
        if (f"train_windows={len(train_windows)} validation_windows={len(valid_windows)}" not in log or
                any(f"epoch={epoch} VALIDATION" not in log for epoch in range(1, 8))):
            errors.append(f"ablation/{module}: full-window seven-epoch training evidence missing")

    scales = read(BASE / "model_metrics/maomao_dual_timescale_mae_internal.json")
    time_scales = scales.get("time_mae_by_scale", {})
    expected_scales = ("fine_0_to_2h", "long_2h_to_tail_start", "tail_from_tail_start")
    if set(time_scales) != set(expected_scales) or sum(
            int(time_scales.get(key, {}).get("event_target_rows", 0)) for key in expected_scales
    ) != VALID_ROWS:
        errors.append("time scales: target rows do not partition full validation")
    for key in expected_scales:
        item = time_scales.get(key, {})
        ci = item.get("mae_hours_95ci")
        if (not isinstance(item.get("mae_hours"), (int, float)) or
                not isinstance(ci, list) or len(ci) != 2 or
                not all(isinstance(x, (int, float)) and math.isfinite(x) for x in ci)):
            errors.append(f"time scales/{key}: MAE or 95% CI missing")

    for site, source in SOURCES.items():
        manifest_path = ROWS / site / "manifest.json"
        if not manifest_path.exists():
            pending.append(f"{site}: no row manifest")
            complete[site] = 0
            continue
        manifest = read(manifest_path)
        split = manifest["split"]
        with np.load(split["split_file"]) as saved:
            cal_ids = patient_ids(ROOT / source, saved["validation_admissions"])
            test_ids = patient_ids(ROOT / source, saved["test_admissions"])
        total = len(cal_ids) + len(test_ids)
        if (cal_ids & test_ids or len(cal_ids) != split["patients_validation"] or
                len(test_ids) != split["patients_test"] or
                len(test_ids) != round(total * 0.1)):
            errors.append(f"{site}: source patient IDs contradict 90:10 manifest")
        if manifest.get("status") != "ready_for_full_scale_evaluation":
            pending.append(f"{site}: row materialization {manifest.get('status')}")
            complete[site] = 0
            continue
        ncal = manifest["calibration_target_rows"]
        ntest = manifest["test_target_rows"]
        for split_name, expected in (("calibration", ncal), ("test", ntest)):
            x = np.load(ROWS / site / f"{split_name}_X.npy", mmap_mode="r")
            y = np.load(ROWS / site / f"{split_name}_y.npy", mmap_mode="r")
            if x.shape != (expected, 1549) or y.shape != (expected, 210):
                errors.append(f"{site}/{split_name}: full-row array shape mismatch")
        count = 0
        for model in MODELS:
            path = EXTERNAL / site / model
            metrics_path, status_path = path / "metrics.json", path / "status.json"
            if not metrics_path.exists() or not status_path.exists():
                pending.append(f"{site}/{model}: evaluation pending")
                continue
            data, status = read(metrics_path), read(status_path)
            if status.get("status") != "completed" or data.get("status") != "completed":
                pending.append(f"{site}/{model}: evaluation not terminal")
                continue
            fit=read(path/'calibration_fit.json')
            if data.get('calibration_protocol')!=PROTOCOL or fit.get('protocol')!=PROTOCOL:
                errors.append(f'{site}/{model}: obsolete calibration protocol')
            if PROTOCOL!=LEGACY_PROTOCOL:
                try:validate_fit(fit,ncal,210)
                except AssertionError:errors.append(f'{site}/{model}: full-row V5 calibration proof invalid')
                if data.get('bias')!=fit['bias'] or data['temperature']!=fit['temperature'] or data.get('calibration_fit_sha256')!=sha256(path/'calibration_fit.json'):
                    errors.append(f'{site}/{model}: calibrated transform provenance invalid')
            if model == "xgboost" and data.get("model_sha256") != xgboost_revision():
                pending.append(f"{site}/xgboost: current model revision not evaluated")
                continue
            if (data.get("calibration_rows") != ncal or data.get("test_rows") != ntest or
                    data.get("train_rows_internal") != TRAIN_ROWS or data.get("patient_overlap") != 0 or
                    data.get("patients_calibration") != len(cal_ids) or
                    data.get("patients_test") != len(test_ids)):
                errors.append(f"{site}/{model}: scale/row/patient mismatch")
            check_metrics(data, f"{site}/{model}", errors, require_ci=ntest > 1,
                          expected_rows=ntest,
                          allow_undefined_macro_auroc=ntest == 1)
            check_external_retrieval(data,f"{site}/{model}",errors,ntest)
            count += 1
        complete[site] = count
        raw_path = EXTERNAL / site / "maomao/metrics_uncalibrated.json"
        post_path = EXTERNAL / site / "maomao/metrics.json"
        raw_complete[site] = False
        if raw_path.exists() and post_path.exists():
            raw, post = read(raw_path), read(post_path)
            if (raw.get("status") == "completed" and
                    raw.get("calibrated_reference_sha256") == sha256(post_path)):
                check_metrics(raw, f"{site}/maomao_uncalibrated", errors, require_ci=ntest > 1,
                              expected_rows=ntest)
                check_external_retrieval(raw,f"{site}/maomao_uncalibrated",errors,ntest)
                if (raw.get("calibration_rows") != ncal or raw.get("test_rows") != ntest or
                        raw.get("temperature") != 1.0 or raw.get("fitted_temperature") != post.get("temperature") or
                        raw.get("patient_overlap") != 0 or raw.get("same_test_targets_verified") is not True or
                        raw.get("calibrated_point_metrics_reproduced") is not True or
                        raw.get("model_sha256") != sha256(BASE / "full_maomao_reference/best_model.pt")):
                    errors.append(f"{site}/maomao: pre/post calibration provenance mismatch")
                raw_complete[site] = True
        if not raw_complete[site]:
            pending.append(f"{site}/maomao: raw full-test metrics pending")

    all_external = (len(complete) == len(SOURCES) and all(n == len(MODELS) for n in complete.values())
                    and all(raw_complete.get(site) for site in SOURCES))
    scale_dir = ROOT / "outputs/scale_ablations_richctx_20260928"
    scale_started = (scale_dir / "manifest.json").exists()
    scale_ready = (scale_dir / "verification.json").exists() and read(scale_dir / "verification.json").get("complete") is True
    if scale_started and not scale_ready:
        pending.append("scale experiments: full-row size/output/context comparisons not all verified")
    if scale_ready:
        from scripts.diagnostics.verify_scale_ablations import audit as scale_audit
        scale_result = scale_audit()
        if not scale_result["complete"]:
            errors.append(f"new scale experiments: {scale_result}")
    shap_root=ROOT/'outputs/maomao_plot_sources/shap'
    shap_ready=False
    if (shap_root/'protocol.json').exists():
        shap_delivery=read(shap_root/'delivery_verification.json') if (shap_root/'delivery_verification.json').exists() else {}
        shap_ready=shap_delivery.get('complete') is True and shap_delivery.get('visual_inspection_completed') is True
        if not shap_ready:pending.append('MAOMAO SHAP: full-patient numerical, figure and visual verification pending')
        else:
            shap_numerics=read(shap_root/'numerical_verification.json')
            if not shap_numerics['complete'] or shap_numerics['patients_verified']!=9989 or len(shap_numerics['files'])!=9989:
                errors.append('SHAP: numerical verification or full patient count mismatch')
            if read(shap_root/'protocol.json')['protocol']!='maomao_family_lag_partition_shap_conditional_content_v2_float32_probability':
                errors.append('SHAP: obsolete bf16 family probability calculation')
            for item in shap_numerics['files']:
                if sha256(shap_root/item['path'])!=item['sha256']:errors.append(f'SHAP: attribution file changed {item["path"]}')
            if not read(shap_root/'portable_replot_verification.json')['complete'] or len(shap_delivery['figures'])!=10:
                errors.append('SHAP: standalone figure reproduction/10-figure coverage missing')
            figure_qa=read(shap_root/'figure_qa.json')
            plot_data=figure_qa.get('figure4c_data_check',{})
            if not figure_qa.get('complete') or plot_data.get('panels')!=3 or plot_data.get('lag_intervals_hours')!=['[0,2)','[2,24)','[24,infinity)']:
                errors.append('SHAP: three report-matched Figure4c panels not verified')
            if plot_data.get('display_matrix_sha256')!=sha256(shap_root/'figure4c_display_matrix.npz'):
                errors.append('SHAP: verified three-lag display matrix changed')
            shap_report=(ROOT/'docs/MAOMAO_FAMILY_SHAP.md').read_text()
            if '三个时间尺度的既往事件贡献热图' not in shap_report or '从左至右分别为<2h、2–<24h、≥24h' not in shap_report:
                errors.append('SHAP report: three-lag caption missing')
            if '两张热图的预测事件列相同' in shap_report or '这两层是既往信息' in shap_report:
                errors.append('SHAP report: obsolete two-lag caption remains')
    if all_external:
        report = REPORT.read_text()
        if "待评估" in report or "待生成" in report or "统计中" in report:
            errors.append("report: pending placeholders remain")
        if "总体 time MAE" in report or "time_MAE_h" in report:
            errors.append("report: pooled time MAE reappeared")
        for old in ("|GRU|", "|GRN|", "50k", "50,000", "CapnoBase", "四源"):
            if old in report:
                errors.append(f"report: retired result marker {old!r}")
        if "七个独立外部数据集及三源合并集的五模型 9:1 全量评估已完成" not in report:
            errors.append("report: final completion statement missing")
        internal_section = report.split("## 五模型内部全量验证\n", 1)[-1].split("\n## ", 1)[0]
        for label in MODEL_LABELS:
            prefix = f"|{label}|{TRAIN_ROWS:,}|{VALID_ROWS:,}|"
            if not any(line.startswith(prefix) and line.count("|") == 14
                       for line in internal_section.splitlines()):
                errors.append(f"report: missing full-row internal table entry for {label}")
        for site in SOURCES:
            heading = f"### {SITE_LABELS[site]}\n"
            if heading not in report:
                errors.append(f"report: missing {site} section")
                continue
            section = report.split(heading, 1)[1].split("\n### ", 1)[0].split("\n## ", 1)[0]
            manifest = read(ROWS / site / "manifest.json")
            ncal, ntest = manifest["calibration_target_rows"], manifest["test_target_rows"]
            external_labels = (*MODEL_LABELS[:-1], "MAOMAO（校准前）", "MAOMAO（校准后）")
            from scripts.diagnostics.build_uniform_results_report import metric_row,EXTERNAL_METRICS,FAMILY_METRICS
            for label,model in zip(external_labels,(*MODELS[:-1],'maomao','maomao')):
                file='metrics_uncalibrated.json' if label=='MAOMAO（校准前）' else 'metrics.json'
                data=read(EXTERNAL/site/model/file)
                for metrics in (EXTERNAL_METRICS,FAMILY_METRICS):
                    expected_row=metric_row([label,f'{ncal:,}',f'{ntest:,}'],data,metrics)
                    if expected_row not in section.splitlines():
                        errors.append(f'report: missing or stale full-row {site}/{label} {metrics} entry')
        if "## 最终指标比较汇总" not in report:
            errors.append("report: final metric comparison summary missing")
        if shap_ready and ('## MAOMAO family SHAP 与学习到的事件嵌入' not in report or '(MAOMAO_FAMILY_SHAP.md)' not in report):
            errors.append('report: full-cohort family SHAP methods/figures missing')
        if scale_ready:
            if "## 当前框架全量规模、输出类别与上下文比较" not in report:
                errors.append("report: current scale/context/output experiment comparison missing")
            for heading in ("### 模型规模", "### 输出类别数：与同任务 MAOMAO 配对比较", "### 局部上下文段长度"):
                if heading not in report:
                    errors.append(f"report: missing updated experiment table {heading}")
        for removed in ("### XGBoost 全量续训前后比较", "## MAOMAO 外部校准前与校准后", "### MAOMAO 校准指标比较", "较大的临床 family", "Clinical family Hit", "Clinical family All-hit"):
            if removed in report:
                errors.append(f"report: removed standalone section reappeared {removed}")
        summary_header = "|数据集|评估目标行|单变量|逻辑回归|XGBoost|ANN|MAOMAO（校准前）|MAOMAO（校准后）|"
        if report.count(summary_header) != 2:
            errors.append("report: MAOMAO pre/post columns absent from AUPRC/AUROC model summaries")
        clinical_base = ROOT / "outputs/classic_score_comparison/independent_15pct_test"
        clinical_summary = read(clinical_base / "expanded_results/expanded_comparison_summary.json")
        clinical_split = read(clinical_base / "split_summary.json")
        with (clinical_base / "clinical_score_comparison_full_matrix.csv").open(newline="") as stream:
            matrix = list(csv.DictReader(stream))
        pairs = {(endpoint, item["comparator"]): item
                 for endpoint, comparisons in clinical_summary["comparisons_by_endpoint"].items()
                 for item in comparisons}
        if (len(matrix) != 42 or len(pairs) != 42 or
                clinical_summary.get("status") != "sealed_independent_test" or
                clinical_summary.get("patient_overlap_with_development") != 0 or
                clinical_split.get("patient_overlap") != 0 or
                clinical_split.get("test_fraction_requested") != 0.15 or
                clinical_split["test"]["patients"] != 14_983):
            errors.append("clinical scores: sealed test protocol or 42-pair matrix mismatch")
        for row in matrix:
            item = pairs.get((row["endpoint"], row["comparator"]))
            if not item:
                errors.append("clinical scores: matrix pair missing from summary")
                continue
            metrics = item["metrics_patient_cluster_bootstrap"]
            for metric in ("auroc", "auprc"):
                for prefix, source in (("comparator", metrics[row["comparator"]]),
                                       ("maomao", metrics["MAOMAO"]),
                                       (f"delta_{metric}_comparator_minus_maomao", metrics["score_minus_maomao"][row["comparator"]])):
                    column = prefix if prefix.startswith("delta_") else f"{prefix}_{metric}"
                    values = source[metric]
                    for suffix, value in (("estimate", values["estimate"]),
                                          ("ci95_low", values["ci95"][0]),
                                          ("ci95_high", values["ci95"][1])):
                        if not math.isclose(float(row[f"{column}_{suffix}"]), value, abs_tol=1e-12):
                            errors.append(f"clinical scores: matrix metric differs for {row['endpoint']}/{row['comparator']}")
        if ("## MAOMAO 与临床评分预测能力比较" not in report or
                "### 代表性结果" not in report or "### 解释边界" not in report or
                "85% development / 15% 封存测试" not in report):
            errors.append("report: clinical score results or separate protocol missing")
        manifest_path = FOLDER / "package_manifest.json"
        if not manifest_path.exists() or not read(manifest_path).get("external_complete"):
            errors.append("package: final manifest not complete")
        if not ZIP.exists():
            errors.append("package: ZIP missing")
        else:
            with ZipFile(ZIP) as archive:
                names = archive.namelist()
                report_member = "MAOMAO_uniform_90_10/MAOMAO_V5_FINAL_RESULTS.md"
                if report_member not in names or archive.read(report_member) != REPORT.read_bytes():
                    errors.append("package: report is stale or missing")
                if any(name.lower().endswith('.svg') for name in names):
                    errors.append('package: user-excluded SVG image included')
                if PROTOCOL!=LEGACY_PROTOCOL:
                    plot_root=ROOT/'outputs/maomao_plot_sources'
                    example_proof_path=plot_root/'external_example_portable_replot_verification.json'
                    example_proof_member='MAOMAO_uniform_90_10/plot_sources/external_example_portable_replot_verification.json'
                    if (not example_proof_path.exists() or example_proof_member not in names or
                            archive.read(example_proof_member)!=example_proof_path.read_bytes()):
                        errors.append('package: current independent external example replot proof missing')
                    else:
                        example_proof=read(example_proof_path)
                        expected_examples={'external_maomao_brier','mimic_micro_roc_pr','mover_maomao_reliability'}
                        if not example_proof.get('complete') or {x['name'] for x in example_proof.get('examples',[])}!=expected_examples:
                            errors.append('package: external example replot scope incomplete')
                        if (example_proof.get('plotter_sha256')!=sha256(plot_root/'plot_examples.py') or
                                example_proof.get('alignment_helper_sha256')!=sha256(plot_root/'audit_panel_alignment.py')):
                            errors.append('package: obsolete external example plotting script/helper')
                        for item in example_proof.get('input_files',[]):
                            if sha256(plot_root/item['path'])!=item['sha256']:
                                errors.append(f"package: stale external example input {item['path']}")
                        for item in example_proof.get('examples',[]):
                            image_path=plot_root/'example_figures'/(item['name']+'.png')
                            member='MAOMAO_uniform_90_10/plot_sources/example_figures/'+image_path.name
                            if (not item.get('identical_replotted_png') or not item.get('native_vector_embedded_arial_verified') or
                                    item['reference_png_sha256']!=sha256(image_path) or
                                    item['reference_png_sha256']!=item['replotted_png_sha256'] or
                                    member not in names or hashlib.sha256(archive.read(member)).hexdigest()!=item['reference_png_sha256']):
                                errors.append(f"package: stale/unverified external example {item['name']}")
                manuscript_root=ROOT/'outputs/maomao_manuscript_figures_20260929'
                if PROTOCOL!=LEGACY_PROTOCOL:
                    preflight_path=manuscript_root/'source_preflight.json'
                    review_path=manuscript_root/'source_preflight_review.json'
                    preflight=read(preflight_path);review=read(review_path)
                    if (not review.get('all_findings_resolved_under_user_contract') or
                            review.get('source_preflight_sha256')!=sha256(preflight_path)):
                        errors.append('package: current plotting source preflight review missing/stale')
                    source_names=('scripts/diagnostics/build_manuscript_figures.py',
                                  'scripts/diagnostics/maomao_figure_revision.py')
                    if set(review.get('source_files',{}))!=set(source_names):
                        errors.append('package: plotting source preflight scope incomplete')
                    for source_name in source_names:
                        if review.get('source_files',{}).get(source_name)!=sha256(ROOT/source_name):
                            errors.append(f'package: obsolete plotting source preflight {source_name}')
                    combined='\n\n'.join((ROOT/source_name).read_text() for source_name in source_names)
                    if review.get('combined_source_sha256')!=hashlib.sha256(combined.encode()).hexdigest():
                        errors.append('package: plotting preflight did not cover current combined render sources')
                    unresolved=[(x['check_id'],x['level']) for x in preflight['findings'] if x['level']!='PASS']
                    reviewed=[(x['check_id'],x['level']) for x in review.get('reviewed_findings',[])]
                    if unresolved!=reviewed or any(not x.get('resolution') for x in review.get('reviewed_findings',[])):
                        errors.append('package: plotting preflight findings not explicitly reviewed')
                    for name in ('source_preflight.json','source_preflight_review.json'):
                        member='MAOMAO_uniform_90_10/manuscript_figures/'+name
                        if member not in names or archive.read(member)!=(manuscript_root/name).read_bytes():
                            errors.append(f'package: missing/stale current plotting preflight {name}')
                manuscript_proof=read(manuscript_root/'delivery_verification.json')
                if not manuscript_proof.get('complete') or len(manuscript_proof.get('verified_supplements',[]))!=12:
                    errors.append('package: revised five main and twelve supplementary figures incomplete')
                artifacts=[f'Figure_{n}' for n in range(1,6)]+manuscript_proof.get('verified_supplements',[])
                for artifact in artifacts:
                    proof=read(manuscript_root/f'figures/{artifact}.export_verification.json')
                    for extension in ('pdf','png'):
                        member=f'MAOMAO_uniform_90_10/manuscript_figures/figures/{artifact}.{extension}'
                        if member not in names or hashlib.sha256(archive.read(member)).hexdigest()!=proof[f'{extension}_sha256']:
                            errors.append(f'package: missing/stale revised figure {artifact}.{extension}')
                portable=read(manuscript_root/'portable_replot_verification.json')
                if not portable.get('complete') or not portable.get('all_twelve_supplements_complete'):
                    errors.append('package: all revised figures/supplements portable reproduction missing')
                data_dir=manuscript_root/'source_data'
                audit=read(data_dir/'clinical_calibration_dca_audit.json')
                if (not audit.get('complete') or audit.get('test_set_fitted') or
                    audit.get('protocol')!='common_training_monotone_risk_mapping_v2' or
                    not audit.get('maomao_auroc_ap_unchanged') or not audit.get('training_comparator_cases_identical')):
                    errors.append('package: clinical probability calibration/DCA revision not verified')
                for relative,digest in audit['source_hashes'].items():
                    member=f'MAOMAO_uniform_90_10/manuscript_figures/source_data/{relative}'
                    if member not in names or hashlib.sha256(archive.read(member)).hexdigest()!=digest:
                        errors.append(f'package: stale clinical probability audit source {relative}')
                for relative in ('MAOMAO_CLINICAL_CALIBRATION_DCA_AUDIT.md','MAOMAO_CLINICAL_COMMON_COHORT_CURVES.md'):
                    member=f'MAOMAO_uniform_90_10/{relative}'
                    if member not in names or archive.read(member)!=(ROOT/'docs'/relative).read_bytes():
                        errors.append(f'package: clinical calibration/DCA report missing or stale {relative}')
                if any('/private_risk_training_inputs/' in name for name in names):
                    errors.append('package: private calibration identifiers included')
                if any(name.startswith("MAOMAO_uniform_90_10/ablation/") and
                       name.endswith((".pt", ".pth", ".ckpt")) for name in names):
                    errors.append("package: ablation checkpoint included")
                if any("/scale_ablations/" in name and name.endswith((".pt", ".pth", ".ckpt", ".npy")) for name in names):
                    errors.append("package: scale ablation checkpoint or raw cache included")
                if scale_ready:
                    for item in (scale_dir / "metrics").glob("*.json"):
                        member = f"MAOMAO_uniform_90_10/scale_ablations/metrics/{item.name}"
                        if member not in names or archive.read(member) != item.read_bytes():
                            errors.append(f"package: missing/stale current scale metric {item.name}")
                    if "MAOMAO_uniform_90_10/scale_ablations/MAOMAO_SCALE_ABLATIONS_CURRENT.md" not in names:
                        errors.append("package: current scale comparison report missing")
                if any(name.endswith((".log", ".jsonl")) for name in names):
                    errors.append("package: training trace with pooled time MAE included")
                if any("50k" in name.lower() or "gru" in name.lower() or "capnobase" in name.lower() for name in names):
                    errors.append("package: retired result included")
                if any(name.endswith("/attempt_history.json") for name in names):
                    errors.append("package: superseded attempt history included")
                if any("/continuation_reference_" in name for name in names):
                    errors.append("package: removed XGBoost reference comparison included")
                for relative in ("MAOMAO_CLINICAL_SCORE_COMPARISON.md", "clinical_score_comparison_full_matrix.csv",
                                 "split_summary.json", "expanded_results/expanded_comparison_summary.json", "provenance.json"):
                    member = f"MAOMAO_uniform_90_10/clinical_scores/{relative}"
                    if member not in names:
                        errors.append(f"package: clinical score artifact missing {relative}")
                    elif relative not in ("MAOMAO_CLINICAL_SCORE_COMPARISON.md", "provenance.json") and archive.read(member) != (clinical_base / relative).read_bytes():
                        errors.append(f"package: stale clinical score artifact {relative}")
                for site in SOURCES:
                    for model in MODELS:
                        member = f"MAOMAO_uniform_90_10/external/{site}/{model}/metrics.json"
                        if member not in names:
                            errors.append(f"package: missing {site}/{model} metrics")
                    if f"MAOMAO_uniform_90_10/external/{site}/maomao/metrics_uncalibrated.json" not in names:
                        errors.append(f"package: missing {site}/maomao pre-calibration metrics")
                manifest = read(manifest_path) if manifest_path.exists() else {"files": []}
                if manifest.get('calibration_protocol')!=PROTOCOL:
                    errors.append('package: manifest has an obsolete calibration protocol')
                if PROTOCOL!=LEGACY_PROTOCOL:
                    optimizer_path=ROOT/'outputs/maomao_plot_sources/calibration_optimizer_audit.json'
                    optimizer_member='MAOMAO_uniform_90_10/plot_sources/calibration_optimizer_audit.json'
                    if (not optimizer_path.exists() or optimizer_member not in names or
                            archive.read(optimizer_member)!=optimizer_path.read_bytes()):
                        errors.append('package: missing/stale full calibration optimizer audit')
                    else:
                        optimizer=read(optimizer_path)
                        fits=optimizer.get('fits',[])
                        if (not optimizer.get('complete') or optimizer.get('protocol')!=PROTOCOL or
                                optimizer.get('calibration_fits')!=125 or len(fits)!=125 or
                                len({x['path'] for x in fits})!=125):
                            errors.append('package: optimizer audit does not cover every actual fit')
                        selected_steps=[]
                        for item in fits:
                            path=ROOT/item['path']
                            if not path.exists() or sha256(path)!=item['sha256']:
                                errors.append(f"package: obsolete optimizer input {item['path']}")
                                continue
                            fit=read(path)
                            selected_steps.extend(x for x in fit['optimization_history']
                                if x['scope']=='all' and x['family']==fit['family'])
                        if (optimizer.get('selected_full_refits')!=len(selected_steps) or
                                optimizer.get('selected_full_refits_converged')!=sum(x['converged'] for x in selected_steps) or
                                optimizer.get('selected_full_refits_not_converged')!=sum(not x['converged'] for x in selected_steps)):
                            errors.append('package: optimizer convergence claims disagree with saved fits')
                    record='MAOMAO_uniform_90_10/calibration_revision/promotion_verification.json'
                    if record not in names:
                        errors.append('package: full V5 restoration promotion audit missing')
                    else:
                        promoted=json.loads(archive.read(record))
                        if not (promoted.get('complete') and promoted.get('total_calibration_fits')==125 and promoted.get('before_after_pairs')==93 and promoted.get('protocol')==PROTOCOL):
                            errors.append('package: V5 restoration coverage not complete')
                        selection_member='MAOMAO_uniform_90_10/calibration_revision/selection_partition_verification.json'
                        if (selection_member not in names or
                                hashlib.sha256(archive.read(selection_member)).hexdigest()!=promoted.get('selection_partition_verification_sha256')):
                            errors.append('package: source patient-selection reconstruction proof missing/stale')
                        else:
                            selection=json.loads(archive.read(selection_member))
                            if (not selection.get('complete') or selection.get('calibration_fits_checked')!=125 or
                                    promoted.get('source_patient_selection_fits_verified')!=125 or
                                    not selection.get('no_test_labels_read') or
                                    len(selection.get('checks',[]))!=125 or
                                    len({x['fit_path'] for x in selection['checks']})!=125 or
                                    {x['site'] for x in selection.get('source_partitions',[])}!=set(SOURCES)):
                                errors.append('package: source calibration-selection coverage incomplete')
                            for item in selection.get('checks',[]):
                                if sha256(ROOT/item['fit_path'])!=item['fit_sha256']:
                                    errors.append(f"package: obsolete calibration-selection fit {item['fit_path']}")
                            for site_proof in selection.get('source_partitions',[]):
                                for source in site_proof['source_files']:
                                    if sha256(ROOT/source['path'])!=source['sha256']:
                                        errors.append(f"package: changed calibration patient/window source {source['path']}")
                    for site in SOURCES:
                        for model in MODELS:
                            relative=f'external/{site}/{model}/calibration_fit.json'
                            member='MAOMAO_uniform_90_10/'+relative
                            current=EXTERNAL/site/model/'calibration_fit.json'
                            if member not in names or archive.read(member)!=current.read_bytes():
                                errors.append(f'package: missing/stale V5 parameters {site}/{model}')
                if shap_ready:
                    if not manifest.get('shap_full_eligible_patient_complete'):errors.append('package: complete SHAP not declared')
                    shap_prefix='MAOMAO_uniform_90_10/plot_sources/shap/'
                    if len([name for name in names if name.startswith(shap_prefix+'patients/case_') and name.endswith('.npz')])!=9989:
                        errors.append('package: full 9,989 SHAP case files missing')
                    for relative in ('MAOMAO_FAMILY_SHAP.md','plot_sources/shap/delivery_verification.json','plot_sources/shap/replot_maomao_shap.py'):
                        member='MAOMAO_uniform_90_10/'+relative
                        if member not in names:errors.append(f'package: missing SHAP artifact {relative}')
                    if any('superseded' in name or 'private_source_window_query_plan' in name or '/preview/' in name for name in names):
                        errors.append('package: superseded SHAP/private source plan/preview included')
                expected_members = {"MAOMAO_uniform_90_10/package_manifest.json"}
                manifest_member = "MAOMAO_uniform_90_10/package_manifest.json"
                if (manifest_path.exists() and manifest_member in names and
                        archive.read(manifest_member) != manifest_path.read_bytes()):
                    errors.append("package: manifest in ZIP differs from organized folder")
                for item in manifest.get("files", []):
                    member = f"MAOMAO_uniform_90_10/{item['path']}"
                    expected_members.add(member)
                    if member not in names:
                        errors.append(f"package: manifest member missing {item['path']}")
                        continue
                    digest = hashlib.sha256()
                    size = 0
                    with archive.open(member) as stream:
                        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                            digest.update(block)
                            size += len(block)
                    if digest.hexdigest() != item["sha256"] or size != item["size_bytes"]:
                        errors.append(f"package: manifest checksum/size mismatch {item['path']}")
                if set(names) != expected_members:
                    errors.append("package: ZIP members differ from manifest")
                corrupt = archive.testzip()
                if corrupt:
                    errors.append(f"package: corrupt ZIP member {corrupt}")
        if (FOLDER / "MAOMAO_V5_FINAL_RESULTS.md").exists():
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            if digest(FOLDER / "MAOMAO_V5_FINAL_RESULTS.md") != digest(REPORT):
                errors.append("package: organized folder report is stale")

    return {"all_external_complete": all_external, "completed_models_by_site": complete,
            "scale_experiments_complete": scale_ready if scale_started else None,
            "pending": pending, "errors": errors, "verified": all_external and not pending and not errors}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    result = audit()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["errors"] or (args.require_complete and not result["verified"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
