#!/usr/bin/env python3
"""Build the requested four-baseline + MAOMAO and seven-module report."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("outputs/final_experiment_results_20260923")
REPORT = Path("docs/MAOMAO_V5_FINAL_RESULTS.md")
MODULE_RUNS = (OUT / "module_ablation_runs" if (OUT / "module_ablation_runs").exists()
               else Path("outputs/module_ablations_richctx_20260923"))


def read(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def metric_cell(data: dict | None, name: str) -> str:
    if not data or data.get("status") == "failed":
        return "未完成"
    value = data.get(name)
    ci = data.get(f"{name}_95ci")
    if value is None:
        return "NA"
    if ci and len(ci) == 2:
        return f"{value:.4f} ({ci[0]:.4f}–{ci[1]:.4f})"
    return f"{value:.4f} (95% CI NA)"


def best_validation_summary(history_path: Path) -> tuple[str, str, str, dict]:
    if not history_path.exists():
        return "NA", "NA", "待训练", {}
    rows = [json.loads(x) for x in history_path.read_text().splitlines() if x.strip()]
    if not rows:
        return "NA", "NA", "训练中/待评估", {}
    best = min(rows, key=lambda row: float(row.get("validation_loss", row.get("loss", float("inf")))) )
    epoch = str(best.get("epoch", "NA"))
    loss = best.get("validation_loss", best.get("loss"))
    loss = f"{float(loss):.4f}" if loss is not None else "NA"
    log = history_path.parent / "train.log"
    finished = log.exists() and "Training finished" in log.read_text(errors="ignore")
    return epoch, loss, "训练完成/待全量指标评估" if finished else "训练中/部分结果", best


def full_internal_result(data: dict | None) -> dict | None:
    """Exclude sampled historical metrics from the full-cohort comparison."""
    if not data:
        return None
    rows = data.get("rows", data.get("evaluation_rows"))
    if rows != 1_563_972 or data.get("evaluation_rows_full_patient_validation") is False:
        return None
    return data


def main() -> None:
    # Keep historical callers pointed at the current uniform 90:10 report.
    from build_uniform_results_report import main as build_current_report
    build_current_report()
    return
    fullscale = Path("outputs/final_experiment_results_20260923/baseline_metrics")
    full_uni = full_internal_result(read(fullscale / "univariate_fullscale_internal.json"))
    full_logistic = full_internal_result(read(fullscale / "logistic_fullscale_internal.json"))
    full_xgb = full_internal_result(read(fullscale / "xgboost_fullscale_internal.json"))
    xgb_one_round_dir = OUT / "classical_ml_fullscale/xgboost/minimal_one_round_20260927"
    xgb_one_round = read(xgb_one_round_dir / "metrics.json")
    xgb_one_round_status = read(xgb_one_round_dir / "status.json") or {}
    historical_classical = read(fullscale / "baseline_classical_internal_external.json") or {}
    historical_xgb = historical_classical.get("models", {}).get("xgboost")
    full_external = read(fullscale / "fullscale_classical_external.json") or {}
    saved_maomao_external = {
        "mimic": read(fullscale / "maomao_mimic_external_90_10.json"),
        "mover": read(fullscale / "maomao_mover_external_90_10.json"),
    }
    saved_gru_external = read(fullscale / "gru_baseline_calibrated.json") or {}
    # A stale completion marker must not make the report claim that full-scale
    # XGBoost is part of the primary comparison. Require all three validated
    # full-scale classical result files before marking that group complete.
    fullscale_done = bool(full_uni and full_logistic and full_xgb)
    maomao = full_internal_result(read(OUT / "model_metrics/maomao_internal.json"))
    gru = full_internal_result(read(OUT / "model_metrics/gru_internal.json"))
    module_metric_files = {
        name: read(OUT / f"module_metrics/{name}.json")
        for name in (
            "no_block_causal", "no_relative_time", "no_family_head",
            "no_clock_phase_summary", "no_measurement_intensity",
            "no_masked_event_value", "no_event_conditioned_time",
        )
    }
    fullscale_classical_outputs = {
        label: row is not None
        for label, row in (("单变量", full_uni), ("逻辑回归", full_logistic), ("XGBoost", full_xgb))
    }
    models = [
        ("单变量", full_uni),
        ("逻辑回归", full_logistic),
        ("XGBoost（30轮完整主基线）", full_xgb),
        ("GRU/RNN", gru),
        ("MAOMAO", maomao),
    ]
    lines = [
        "# MAOMAO 模块消融与五模型评估结果",
        "",
        "",
        "## 统一协议",
        "",
        "- 训练数据：`data/perioperative_event_sequences_v5_richctx_static7`；外部 MIMIC/MOVER 不进入训练。",
        "- 内部拆分：按患者 90% train / 10% validation，seed=42，患者无交叉；外部校准与封存测试也固定 90% / 10%。",
        "- 比较任务：相同事件词表与 next-event 多标签目标。MAOMAO 与 GRU 使用序列模型输入；逻辑回归/XGBoost 使用因果原始序列通道定长展平及相同静态协变量，不添加专属手工临床汇总特征。",
        "- 报告 AUPRC、AUROC、MRR、Brier、ECE、Hit@1、Recall@5、Recall@10 的点估计和 95% CI；CI 方法及样本量保存在各结果 JSON。",
        "- 全量经典模型预算：逻辑回归为 7 次完整训练集遍历；原计划的完整 XGBoost 基线为 `hist`、max_depth=4、max_bin=256、eta=0.08、30 个 boosting rounds，固定轮数且未设置 early stopping（boosting round 不等同于 epoch）。",
        ("- 单变量、逻辑回归和 XGBoost 均使用全部 14,128,539 个有效训练目标行与全部 1,563,972 个验证目标行。" if fullscale_done else
         "- 全量经典结果状态：当前完整基线指标已核验为" + ("、".join(k for k, v in fullscale_classical_outputs.items() if v) or "暂无") + "。原计划的 30 轮 XGBoost 在第 4 轮后因系统 OOM 终止。随后单独完成了一个 max_depth=1、max_bin=16 的极简单轮试跑，结果另列如下；它不替代 30 轮完整基线，也不填入五模型完整主表。"),
        ("- MAOMAO 与 GRU/RNN 已在同一患者级 90:10 划分的完整验证目标行上重新评估。" if maomao and gru else
         "- MAOMAO 与 GRU/RNN 的全量验证复评尚未完成；此前 20,000 行结果不放入主比较表。"),
        "",
        "## 五模型内部验证对比",
        "",
        "|模型|micro-AUPRC (95% CI)|macro-AUPRC (95% CI)|micro-AUROC (95% CI)|macro-AUROC (95% CI)|MRR (95% CI)|Brier (95% CI)|ECE (95% CI)|Hit@1 (95% CI)|Recall@5 (95% CI)|Recall@10 (95% CI)|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    keys = ["micro_auprc", "macro_auprc", "micro_auroc", "macro_auroc", "mrr", "brier", "ece", "hit_at_1", "recall_at_5", "recall_at_10"]
    for label, row in models:
        lines.append("|" + label + "|" + "|".join(metric_cell(row, key) for key in keys) + "|")
    lines += ["", "全量经典基线数据、训练日志、模型权重与指标位于 `outputs/final_experiment_results_20260923/classical_full_scale/` 和 `baseline_metrics/`；历史结果文件见 `baseline_metrics/baseline_classical_internal_external.json`。GRU/MAOMAO 内部指标位于 `outputs/final_experiment_results_20260923/model_metrics/`。", ""]
    if xgb_one_round and xgb_one_round_status.get("status") == "completed":
        elapsed = xgb_one_round_status.get("elapsed_seconds", "NA")
        params = xgb_one_round.get("parameters", {})
        lines += [
            "## XGBoost 全量极简单轮诊断（不替代完整主基线）", "",
            f"该试跑使用全部 {xgb_one_round.get('train_rows_full', 0):,} 个训练目标行，在全部 {xgb_one_round.get('validation_rows', 0):,} 个患者隔离验证目标行评分；完成 1 个 boosting round，实际耗时 {elapsed} 秒。参数为 `max_depth={params.get('max_depth')}`、`max_bin={params.get('max_bin')}`、`eta={params.get('eta')}`、`colsample_bytree={params.get('colsample_bytree')}`、`subsample={params.get('subsample')}`、`nthread={params.get('nthread')}`。所有训练行参与该轮（关闭 row subsampling），但仅抽样 1% 特征；因此这是低容量诊断，不能等同于原计划的 30 轮完整 XGBoost 基线。指标使用全量验证集点估计和 200 次 row-bootstrap 95% CI。第一次以内存占用过高停止于建矩阵阶段，未开始拟合；最终使用磁盘外存缓存完成。",
            "",
            "|指标|单轮全量 XGBoost (95% CI)|", "|---|---:|",
        ]
        for key in keys:
            lines.append(f"|{key}|{metric_cell(xgb_one_round, key)}|")
        lines += ["", "训练状态、单轮 booster、完整指标和尝试记录：`outputs/final_experiment_results_20260923/classical_ml_fullscale/xgboost/minimal_one_round_20260927/`。", ""]
    if historical_xgb:
        lines += ["## 已保存的历史 XGBoost 结果（非全量主比较）", "",
                  f"该次已保存评估使用 {historical_classical.get('train_rows', 'NA'):,} 个训练目标行和 {historical_classical.get('validation_rows', 'NA'):,} 个验证目标行。以下结果可供参考，但训练规模与本轮全量 MAOMAO/基线不同，因此不填入全量主表，也不用于声称全量公平比较。原始结果保留在 `outputs/final_experiment_results_20260923/baseline_metrics/baseline_classical_internal_external.json`。", "",
                  "|指标|历史结果 (95% CI)|", "|---|---:|"]
        for key in keys:
            lines.append(f"|{key}|{metric_cell(historical_xgb, key)}|")
        lines.append("")

    lines += ["## 外部验证（患者级 90% calibration / 10% sealed test）", "",
              ("外部集按既有固定患者划分：90% 患者用于温度校准，10% 用作封存测试；五个模型对校准集与测试集的所有有效目标行评分。" if full_external else
               "外部集使用患者级 90% calibration / 10% sealed test。当前五模型匹配检查点的外部全量对比未完成；已有旧检查点 90:10 结果另列如下，不与本轮内部全量主表混用。"), "",
              "|模型|数据集|模式|micro-AUPRC (95% CI)|micro-AUROC (95% CI)|结果文件|",
              "|---|---|---|---:|---:|---|"]
    for dataset, label in (("val_mimic_richctx_static7", "MIMIC"),
                           ("val_mover_richctx_static7", "MOVER")):
        site = "mimic" if label == "MIMIC" else "mover"
        ext_full = full_external.get(site, {})
        for name, model_label in (("univariate_last_token", "单变量"),
                                  ("logistic_regression", "逻辑回归"),
                                  ("xgboost", "XGBoost")):
            for mode, key in (("raw", "models"), ("calibrated", "calibrated_models")):
                row = ext_full.get(key, {}).get(name)
                if row:
                    lines.append(f"|{model_label}|{label}|{mode}|{metric_cell(row,'micro_auprc')}|{metric_cell(row,'micro_auroc')}|`fullscale_classical_external.json`|")
        full_site = full_external.get(site, {})
        for mode in ("raw", "calibrated"):
            full_mode = "models" if mode == "raw" else "calibrated_models"
            metrics = full_site.get(full_mode, {}).get("gru")
            if metrics:
                lines.append(f"|GRU/RNN|{label}|{mode}|{metric_cell(metrics,'micro_auprc')}|{metric_cell(metrics,'micro_auroc')}|`fullscale_classical_external.json`|")
        for mode in ("raw", "calibrated"):
            full_mode = "models" if mode == "raw" else "calibrated_models"
            metrics = full_site.get(full_mode, {}).get("maomao")
            if metrics:
                lines.append(f"|MAOMAO|{label}|{mode}|{metric_cell(metrics,'micro_auprc')}|{metric_cell(metrics,'micro_auroc')}|`fullscale_classical_external.json`|")
    lines.append("")

    lines += ["### 已保存的历史 90:10 外部校准/测试结果", "",
              "这些记录按患者拆分校准集和封存测试集，patient overlap 为 0。MAOMAO 文件来自旧检查点且每站点以 20,000 个测试目标行估计指标；GRU 使用其已保存检查点，报告全部测试目标行。它们证明 90:10 校准流程已有产物，但并非本轮五模型使用统一全量检查点的外部比较。", "",
              "|模型|站点|模式|校准患者/测试患者|测试目标行|micro-AUPRC (95% CI)|micro-AUROC (95% CI)|结果文件|",
              "|---|---|---|---:|---:|---:|---:|---|"]
    for dataset, site_label, maomao_site in (("val_mimic_richctx_static7", "MIMIC", "mimic"),
                                           ("val_mover_richctx_static7", "MOVER", "mover")):
        saved = saved_maomao_external.get(maomao_site) or {}
        reports = saved.get("reports", [])
        if reports:
            item = reports[0]
            split = item.get("split", {})
            patient_counts = f"{split.get('patients_validation','NA')}/{split.get('patients_test','NA')}"
            for mode, field in (("raw", "test_raw"), ("calibrated", "test_calibrated")):
                metrics = item.get(field)
                if metrics:
                    lines.append(f"|MAOMAO（旧检查点）|{site_label}|{mode}|{patient_counts}|{metrics.get('event_targets','NA')}|{metric_cell(metrics,'micro_auprc')}|{metric_cell(metrics,'micro_auroc')}|`maomao_{maomao_site}_external_90_10.json`|")
        gru_external_site = saved_gru_external.get("external", {}).get(dataset)
        if gru_external_site:
            split = gru_external_site.get("split", {})
            patient_counts = f"{split.get('patients_validation','NA')}/{split.get('patients_test','NA')}"
            metrics = gru_external_site.get("calibrated")
            if metrics:
                lines.append(f"|GRU/RNN|{site_label}|calibrated|{patient_counts}|{gru_external_site.get('test_rows','NA')}|{metric_cell(metrics,'micro_auprc')}|{metric_cell(metrics,'micro_auroc')}|`gru_baseline_calibrated.json`|")
    lines.append("")

    lines += ["## 七项模块消融（相对 Full MAOMAO）", "",
              "仅报告点名的七个模块移除实验。Full MAOMAO 的结果只出现在五模型对比中，作为计算相对变化的参照，不在消融表中重复列行或重复训练。七个变体分别只关闭目标模块；其他模型宽度、数据、划分、优化器与训练轮数保持一致。每轮遍历完整固定窗口训练集（208,303 个窗口，14,128,539 个有效目标行），不做窗口裁剪或行抽样。", "",
              "|配置|训练状态|最佳 epoch|验证 loss|micro-AUPRC (95% CI)|micro-AUROC (95% CI)|macro-AUPRC (95% CI)|macro-AUROC (95% CI)|MRR (95% CI)|Brier (95% CI)|ECE (95% CI)|Hit@1 (95% CI)|Recall@5 (95% CI)|Recall@10 (95% CI)|验证 Hit@5|验证 family Hit@5|验证 Hit@10|验证 family Hit@10|",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    module_rows = []
    for name, module in [
        ("移除 block-causal", "no_block_causal"),
        ("移除 relative-time bias", "no_relative_time"),
        ("移除 family hierarchical head", "no_family_head"),
        ("移除 clock/phase-summary", "no_clock_phase_summary"),
        ("移除 measurement-intensity", "no_measurement_intensity"),
        ("移除 masked-event/value auxiliary loss", "no_masked_event_value"),
        ("移除 event-conditioned time", "no_event_conditioned_time"),
    ]:
        module_rows.append((name, read(OUT / f"module_metrics/{module}.json"), MODULE_RUNS / module / "validation_history.jsonl"))
    for label, data, history in module_rows:
        hist_epoch, hist_loss, hist_status, history_metrics = best_validation_summary(history)
        status = "已评估" if data else hist_status
        epoch = data.get("checkpoint_epoch", hist_epoch) if data else hist_epoch
        val_loss = data.get("validation_loss", hist_loss) if data else hist_loss
        if isinstance(val_loss, float): val_loss = f"{val_loss:.4f}"
        module_keys = ["micro_auprc", "micro_auroc", "macro_auprc", "macro_auroc",
                       "mrr", "brier", "ece", "hit_at_1", "recall_at_5", "recall_at_10"]
        metrics = [metric_cell(data, key) for key in module_keys]
        monitor_keys = ["next_event_hit_at_5", "same_family_hit_at_5",
                        "next_event_hit_at_10", "same_family_hit_at_10"]
        monitor_metrics = [
            f"{float(history_metrics[key]):.4f}" if history_metrics.get(key) is not None else "NA"
            for key in monitor_keys
        ]
        lines.append("|" + "|".join([label, status, str(epoch), str(val_loss), *metrics, *monitor_metrics]) + "|")
    lines += ["", f"训练命令清单、日志、配置和 checkpoint 位于 `{MODULE_RUNS.as_posix()}/`；评估 JSON 位于 `outputs/final_experiment_results_20260923/module_metrics/`。", "",
              "## 已有模型规模/词表消融", "",
              "上一轮已完成 50/100/150 输出词表规模，以及 small/current/large 模型规模实验（各 10 个 epoch）；配置、checkpoint 和汇总保留在 `outputs/final_experiment_results_20260923/existing_ablation_runs/` 与 `outputs/final_experiment_results_20260923/existing_ablation_summary/`。50/100/150 指输出词表上限；模型规模组比较 hidden/layer/head/FFN 维度。", "",
              "这些实验使用上一轮 `perioperative_event_sequences_v5_full` 数据，和本轮 `richctx_static7` 不是同一输入数据版本，因此作为历史实验单独报告，不能与本轮七项模块消融直接作同条件数值比较。已核实的旧配置均为 `block_size=256`、`window_stride=128`、`minimum_context_fraction=0.75`，未发现改变上下文长度/含量的独立消融；上下文含量实验目前缺失。", "",
              "## 实验完成度与边界", "",
              ("七项指定模块消融均已完成训练和全量验证评估，使用 9:1 patient-level validation、全部 1,563,972 个验证目标行，并为表中十项报告指标提供 200 次 bootstrap 95% CI。" if all(
                  module_metric_files[name] and module_metric_files[name].get("evaluation_rows_full_patient_validation")
                  and module_metric_files[name].get("evaluation_rows") == 1_563_972
                  and all(f"{metric}_95ci" in module_metric_files[name] for metric in keys)
                  for name in module_metric_files
              ) else "模块消融状态按结果文件核对；缺少有效全量验证结果的行明确标记待完成，不以旧数据、训练拟合指标或部分 checkpoint 替代。"), "",
              "AUROC/AUPRC 的逐事件支持数和指标、所有聚合事件指标的置信区间及 bootstrap 细节保存在原始 JSON。表中的 Hit@5/family Hit 是完整 patient-validation monitor 点估计，尚无 bootstrap CI。置信区间是 row-level，不是 patient-cluster bootstrap；样本行数、bootstrap 重复数和区间方法以每份结果 JSON 为准。", ""]
    internal_rows = [maomao, gru]
    expected_rows = 1_563_972
    all_internal_full = all(
        row and row.get("evaluation_rows_full_patient_validation")
        and row.get("evaluation_rows") == expected_rows
        for row in internal_rows
    )
    if all_internal_full:
        lines.insert(-1, "MAOMAO 与 GRU/RNN 的结果均来自同一 patient-level 90:10 划分的全部 1,563,972 个验证目标行；各报告指标带 95% CI。")
    else:
        lines.insert(-1, "MAOMAO 与 GRU/RNN 的全量内部评估尚未完成；此前 20,000 行结果不进入五模型全量主表。七项模块消融已独立完成全量验证，其结果已列于上表。")
    REPORT.write_text("\n".join(lines))
    print(REPORT.resolve())


if __name__ == "__main__":
    main()
