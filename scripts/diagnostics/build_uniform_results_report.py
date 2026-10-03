#!/usr/bin/env python3
"""Build the current full-row patient-disjoint 90:10 MAOMAO results report."""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.diagnostics.uniform_result_scope import current_xgboost_dir, xgboost_revision, XGB_CANDIDATE, sha256
from maomao.evaluation.softmax_calibration import PROTOCOL, LEGACY_PROTOCOL
OUT = ROOT / "outputs/final_experiment_results_20260923"
EXTERNAL = ROOT / "outputs/external_validation_final_maomao_uniform"
REPORT = ROOT / "docs/MAOMAO_V5_FINAL_RESULTS.md"
METRICS = ("micro_auprc", "macro_auprc", "micro_auroc", "macro_auroc",
           "mrr", "brier", "ece", "hit_at_1", "recall_at_5", "recall_at_10")
LABELS = (("单变量", "univariate"), ("逻辑回归", "logistic_regression"),
          ("XGBoost", "xgboost"), ("ANN", "ann"), ("MAOMAO", "maomao"))
SITES = (("MIMIC", "mimic"), ("MOVER", "mover"),
         ("eICU", "eicu"), ("SICdb", "sicdb"),
         ("NTUH", "ntuh"),
         ("Auckland ASAC EDS", "asac"), ("UQ Vital Signs", "uq"),
         ("三源合并汇总", "surgical_pooled"))
MODULES = (("移除 block-causal", "no_block_causal"),
           ("移除 relative-time bias", "no_relative_time"),
           ("移除 family hierarchical head", "no_family_head"),
           ("移除 clock/phase-summary", "no_clock_phase_summary"),
           ("移除 measurement-intensity", "no_measurement_intensity"),
           ("移除 masked-event/value auxiliary loss", "no_masked_event_value"),
           ("移除 event-conditioned time", "no_event_conditioned_time"))


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def cell(data: dict | None, metric: str) -> str:
    if data is None:
        return "待评估"
    if data.get(metric) is None:
        return "不可估计"
    ci = data.get(metric + "_95ci")
    if not isinstance(ci, list) or len(ci) != 2:
        return f"{float(data[metric]):.4f}（CI 不可估计）"
    lower, upper = float(ci[0]), float(ci[1])
    precision = 4
    while lower < upper and precision < 10 and f"{lower:.{precision}f}" == f"{upper:.{precision}f}":
        precision += 1
    return (f"{float(data[metric]):.{precision}f} "
            f"({lower:.{precision}f}–{upper:.{precision}f})")


EXTERNAL_METRICS=METRICS[:8]+('hit_at_5','hit_at_10','all_true_events_hit_at_10')+METRICS[8:]
FAMILY_METRICS=('same_family_hit_at_1','same_family_hit_at_5','same_family_hit_at_10','all_true_families_hit_at_10')
CLINICAL_METRICS=('clinical_group_hit_at_1','clinical_group_hit_at_5','clinical_group_hit_at_10','all_true_clinical_groups_hit_at_10')

def header(prefix: str, metrics=METRICS) -> list[str]:
    names = ("micro-AUPRC", "macro-AUPRC", "micro-AUROC", "macro-AUROC",
             "MRR", "Brier", "ECE", "Hit@1", "Recall@5", "Recall@10")
    labels=dict(zip(METRICS,names),hit_at_5='Hit@5',hit_at_10='Hit@10',all_true_events_hit_at_10='All-hit@10',
                same_family_hit_at_1='Family Hit@1',same_family_hit_at_5='Family Hit@5',
                same_family_hit_at_10='Family Hit@10',all_true_families_hit_at_10='Family All-hit@10')
    labels.update(clinical_group_hit_at_1='Clinical family Hit@1',clinical_group_hit_at_5='Clinical family Hit@5',
                  clinical_group_hit_at_10='Clinical family Hit@10',all_true_clinical_groups_hit_at_10='Clinical family All-hit@10')
    columns = prefix.split("|") + [f"{labels[key]} (95% CI)" for key in metrics]
    return ["|" + "|".join(columns) + "|",
            "|" + "|".join(["---"] + ["---:" for _ in columns[1:]]) + "|"]


def metric_row(prefix: list[str], data: dict, metrics=METRICS) -> str:
    return "|" + "|".join(prefix + [cell(data, key) for key in metrics]) + "|"


def validate(data: dict, name: str, rows: int) -> None:
    count = (data.get("rows") or data.get("evaluation_rows") or
             data.get("validation_rows") or data.get("test_rows"))
    if count != rows:
        raise RuntimeError(f"{name}: evaluated {count} rows, expected {rows}")
    if data.get("evaluation_rows_full_patient_validation") is False:
        raise RuntimeError(f"{name}: sampled validation is ineligible for this report")
    for key in METRICS:
        cell(data, key)


def main() -> None:
    common_path = OUT / "baseline_metrics/common_full_validation/current_five_internal_metrics.json"
    common = load(common_path)
    if not common or common.get("status") != "completed":
        raise RuntimeError("The common internal full-validation manifest is incomplete")
    train = int(common["train_target_rows"])
    valid = int(common["validation_target_rows"])
    if (train, valid) != (14_128_539, 1_563_972):
        raise RuntimeError(f"Unexpected internal 90:10 rows: {train}, {valid}")
    internal = {}
    for key in ("univariate", "logistic_regression", "ann_fullscale_refined", "maomao"):
        data = common["models"][key]
        validate(data, key, valid)
        if int(data["train_rows"]) != train:
            raise RuntimeError(f"{key}: incomplete internal training target rows")
        internal[key] = data
    xgb_dir = current_xgboost_dir()
    xgb = load(xgb_dir / "metrics.json")
    xgb_status = load(xgb_dir / "status.json")
    if (not xgb or not xgb_status or xgb_status.get("status") != "completed" or
            int(xgb_status.get("rounds", 0)) < 1 or int(xgb.get("train_rows_full", 0)) != train):
        raise RuntimeError("Current full-row XGBoost result is incomplete")
    validate(xgb, "xgboost", valid)
    internal["xgboost"] = xgb
    xgb_hash = sha256(xgb_dir / xgb_status["model_file"])
    internal["ann"] = internal.pop("ann_fullscale_refined")
    guard = load(xgb_dir / "train_guard_record.json")
    xgb_budget_note = ""
    if guard:
        xgb_budget_note = (f"本次训练实际用时 {guard['elapsed_seconds']:.1f} 秒；"
                           f"训练阶段匿名/共享驻留内存峰值 {guard['peak_anonymous_gib']:.2f} GiB，"
                           f"进程 RSS 峰值 {guard['peak_rss_gib']:.2f} GiB（含文件映射驻留页）；"
                           "内存保护同时检查系统可用内存至少 8 GiB。")
        if guard.get("stop_reason") == "time_limit":
            xgb_budget_note += (f"目标 {xgb_status.get('target_rounds',3)} 轮，"
                                f"已完整保存 {xgb_status['rounds']} 轮；"
                                f"第 {xgb_status.get('fitting_round')} 轮完成 "
                                f"{xgb_status.get('completed_targets_in_round')}/210 个标签时达到训练总时间上限。"
                                "未完成的轮次未计入本报告模型，使用最高完整轮次进行内部与外部评分。")
    xgb_note = ("新增轮次按 210 个独立二元标签串行训练，复用同一个全量外存分位数矩阵；"
                "保存原始第一轮的全部树，按原始标签顺序组合 XGBoost 原生树，并核对组合前后训练行的 margin。"
                "标签/轮次种子为 42 + 标签序号×101 + 轮次×1009，在验证前固定。"
                if xgb.get("native_merge_margin_integrity_verified") else "当前表引用此前保存的一轮模型。")

    lines = ["# MAOMAO 统一 9:1 全量评估结果", "", "**MAOMAO**：Multi-horizon Anticipatory Outcome Model for Anesthesia and Operations。", "",
             f"更新：{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}。"
             "本报告汇总当前五模型、七个独立外部数据集、一个三源合并汇总集、MAOMAO 外部校准前后比较、七项模块消融、分尺度时间误差和临床评分预测能力比较。", "",
             "## 统一评估协议", "",
             f"- 内部 INSPIRE 患者级划分：seed 42，训练 {common['patients_train']:,} 名患者，"
             f"验证 {common['patients_validation']:,} 名患者；训练目标行 {train:,}，"
             f"验证目标行 {valid:,}。患者无交叉，内部点估计使用完整验证目标行。",
             "- 五模型顺序为单变量、逻辑回归、XGBoost、ANN、MAOMAO。"
             "单变量基于当前 token；其他定长输入模型使用相同的 256 事件通道和七项静态输入；"
             "MAOMAO 使用相同原始事件序列及静态输入。",
             "- 内部 MAOMAO 与七项模块消融使用经典模型全量验证数组中的相同窗口和位置坐标评分，"
             "并在计算指标前逐行核对目标标签张量。",
             f"- XGBoost 使用当前保存的全量训练模型：{xgb_status['rounds']} 个 boosting round、"
             "max_depth=1、max_bin=16、colsample_bytree=0.01、subsample=1.0。"
             "ANN 在完整内部训练目标行上完成一轮续训。各模型的训练预算按实际配置记录。" + xgb_note + xgb_budget_note,
             "- 每个外部评估队列各自按患者 90% 校准、10% 封存测试划分，seed 42。"
             "小型队列的 10% 患者数取最接近整数，精确人数和有效目标行见下表。"+
             ("五个内部训练模型的权重冻结；恢复 V5 的原始输出、仅温度、温度＋类别偏置三候选。"
              "只在外部90%内部按患者/记录代理划分80%拟合、20%选择，按正事件集合softmax负对数似然选择方法；"
              "随后在全部90%有效校准行重新拟合。温度范围0.15–6，类别偏置范围−4至4，偏置均方正则系数0.002。"
              "所有五模型、模块/规模/输出/上下文变体采用同一协议。" if PROTOCOL!=LEGACY_PROTOCOL else
              "五个内部训练模型的权重冻结；各模型在该数据集全部校准目标行上，按softmax对归一化正事件集合的交叉熵拟合温度。"
              "只在90%校准集上检查Brier是否优于T=1，否则保留T=1；温度范围固定0.15–6。")+
             "拟合与选择均不使用封存测试行，再在全部原10%封存测试目标行上计算指标。"
             "旧sigmoid拟合/softmax评分不一致已修正，详情见[校准核查](MAOMAO_CALIBRATION_AUDIT.md)。",
             "本次为已报告测试集上的实现修正复算，不是新增未见过的独立确认性验证；校准方法选择和参数拟合仅使用校准行。",
             "- AUROC、AUPRC、MRR、Brier、ECE、Hit@1、Recall@5、Recall@10 点估计使用对应完整评估集。"
             "本轮统一修正并列概率处理：AUROC使用平均并列秩等价的正负配对计数，AP（表中AUPRC）按相同概率阈值分组，计数累积使用float64；"
             "当前指标JSON标记threshold_grouped_ap_midrank_auroc_float64_v2。"
             "95% CI 基于 200 次目标行 bootstrap；大样本区间在固定种子的均匀目标行样本上计算，"
             "内部 MAOMAO 与七项消融使用 30,000 行，其余大样本评估使用 100,000 行，"
             "再按样本量缩放区间宽度，因此属于近似区间。每份 JSON 记录实际行数和 CI 方法。"
             "目标行相关性与校准方法选择、温度及类别偏置拟合的不确定性尚未通过患者聚类bootstrap完整传播。", "",
             "- Brier对q=y/sum(y)评分；ECE按15桶比较max-softmax confidence与top1是否属于任一正事件。"
             "多正事件时二者目标不同，后者不能直接视为归一化事件分布的完全校准误差；"
             "校准前后不保证两项同时改善，实际变化见外部大表和校准核查。", "",
             "- 各数据库的源病例、排除阶段、最终患者/记录数和实际清洗见[数据清洗病例流](MAOMAO_DATA_CLEANING_FLOW.md)。"
             "小型麻醉数据以记录ID作为患者代理；外部桶起点时间、基础normalizer拟合范围和术前静态字段可用性限制在该文档明确列出。", "",
             "- 完整图源见[画图数据说明](plot_sources/README.md)：指标/CI长表、逐事件/家族与频率分组、全行ROC/PR桶计数、可靠性、排名/混淆、分尺度时间预测/误差、学习曲线和42组临床评分曲线。", "",
             "### 当前五模型拟合记录", "",
             "|模型|内部训练目标行|实际拟合记录|",
             "|---|---:|---|",
             f"|单变量|{train:,}|在完整训练目标行上估计当前 token 条件频率|",
             f"|逻辑回归|{train:,}|7 次完整训练集遍历|",
             f"|XGBoost|{train:,}|{xgb_status['rounds']} 个 boosting round|",
             f"|ANN|{train:,}|在保存的初始化权重上完成 1 次完整训练集续训|",
             f"|MAOMAO|{train:,}|完整训练 7 个 epoch；当前 checkpoint 选自第 {internal['maomao']['checkpoint_epoch']} 个 epoch|", "",
             "## 五模型内部全量验证", ""]
    lines += header("模型|训练目标行|验证目标行")
    for label, key in LABELS:
        lines.append(metric_row([label, f"{train:,}", f"{valid:,}"], internal[key]))
    lines += ["", f"当前内部汇总指标：{common_path.relative_to(ROOT)}；"
              f"XGBoost 指标：{(xgb_dir / 'metrics.json').relative_to(ROOT)}。", "",
              "## 外部七源及三源合并集：患者级 9:1 校准与封存测试", "",
              "下表逐站点列出患者及有效目标行。每站点全部五模型使用该站点相同的校准和测试行；"
              "正在计算的行只显示状态，不填写尚未产生的指标。三源合并集由 NTUH、ASAC、UQ 构成，"
              "与这三个独立来源重叠，单列为汇总视角；合并后重新进行患者级 9:1 划分。", "",
              "|数据集|校准患者|测试患者|校准目标行|测试目标行|五模型完成数|",
              "|---|---:|---:|---:|---:|---:|"]
    site_data = {}
    site_test_rows = {}
    external_done = True
    for label, site in SITES:
        manifest = load(OUT / f"classical_full_scale/external/{site}/manifest.json")
        if manifest and isinstance(manifest.get("test_target_rows"), int):
            site_test_rows[site] = manifest["test_target_rows"]
        if manifest and manifest.get("status") == "ready_for_full_scale_evaluation":
            split = manifest["split"]
            if (split.get("patient_overlap") != 0 or
                    split["patients_test"] != round(split["patients_total"] * 0.1)):
                raise RuntimeError(f"{site}: patient-level 90:10 split is not verified")
            reports = {}
            for _, key in LABELS:
                report = load(EXTERNAL / site / key / "metrics.json")
                status = load(EXTERNAL / site / key / "status.json")
                if report and status and status.get("status") == "completed":
                    if report.get("calibration_protocol") != PROTOCOL:
                        raise RuntimeError(f"{site}/{key}: old calibration protocol cannot enter revised final report")
                    if key == "xgboost" and report.get("model_sha256") != xgb_hash:
                        continue
                    validate(report, site + "/" + key, manifest["test_target_rows"])
                    if (report.get("calibration_rows") != manifest["calibration_target_rows"] or
                            report.get("patient_overlap") != 0 or
                            report.get("train_rows_internal") != train):
                        raise RuntimeError(f"{site}/{key}: external comparison contract mismatch")
                    reports[key] = report
            site_data[site] = (label, manifest, reports)
            lines.append(f"|{label}|{split['patients_validation']:,}|{split['patients_test']:,}|"
                         f"{manifest['calibration_target_rows']:,}|{manifest['test_target_rows']:,}|"
                         f"{len(reports)}/5|")
            external_done = external_done and len(reports) == 5
        else:
            if manifest and manifest.get("split"):
                split = manifest["split"]
                cal_rows = manifest.get("calibration_target_rows", "统计中")
                test_rows = manifest.get("test_target_rows", "统计中")
                cal_rows = f"{cal_rows:,}" if isinstance(cal_rows, int) else cal_rows
                test_rows = f"{test_rows:,}" if isinstance(test_rows, int) else test_rows
                lines.append(f"|{label}|{split['patients_validation']:,}|{split['patients_test']:,}|"
                             f"{cal_rows}|{test_rows}|0/5|")
            else:
                lines.append(f"|{label}|待预处理|待预处理|待生成|待生成|0/5|")
            external_done = False
    raw_by_site = {}
    for label, site in SITES:
        entry = site_data.get(site)
        if not entry:
            external_done = False
            continue
        _, manifest, reports = entry
        calibrated = reports.get("maomao")
        raw = load(EXTERNAL / site / "maomao/metrics_uncalibrated.json")
        if (raw and calibrated and raw.get("status") == "completed" and
                raw.get("calibrated_reference_sha256") == sha256(EXTERNAL / site / "maomao/metrics.json")):
            validate(raw, site + "/maomao_raw", manifest["test_target_rows"])
            raw_by_site[site] = raw
        else:
            raw = None
            external_done = False
    lines.append("")
    for metric, title in (("micro_auprc", "micro-AUPRC"),
                          ("micro_auroc", "micro-AUROC")):
        lines += [f"### 内部与外部统一汇总：{title}", "",
                  "同一行的五个模型使用完全相同的目标行。内部行来自训练集患者级 10% 验证集；"
                  "外部行来自该队列患者级 10% 封存测试集，校准参数仅用其 90% 校准集拟合。"
                  "MAOMAO 校准前与校准后列使用同一冻结 checkpoint 和相同测试行；"
                  "校准前温度为 1，校准后温度见各数据集表。内部结果未作外部温度校准。", "",
                  "|数据集|评估目标行|单变量|逻辑回归|XGBoost|ANN|MAOMAO（校准前）|MAOMAO（校准后）|",
                  "|---|---:|---:|---:|---:|---:|---:|---:|",
                  "|INSPIRE 内部验证|" + f"{valid:,}|" +
                  "|".join(cell(internal[key], metric) for _, key in LABELS) + "|不适用（内部未校准）|"]
        for label, site in SITES:
            site_entry = site_data.get(site)
            if site_entry:
                _, manifest, reports = site_entry
                count = f"{manifest['test_target_rows']:,}"
            else:
                reports = {}
                count = (f"{site_test_rows[site]:,}" if site in site_test_rows
                         else "待生成")
            lines.append("|" + "|".join([label, count] +
                         [cell(reports.get(key), metric) for _, key in LABELS if key != "maomao"] +
                         [cell(raw_by_site.get(site), metric), cell(reports.get("maomao"), metric)]) + "|")
        lines.append("")
    for label, site in SITES:
        if site not in site_data:
            continue
        _, manifest, reports = site_data[site]
        lines += [f"### {label}", ""]
        lines += ["精确事件命中：Hit@k 表示前 k 个预测中至少包含一个真实事件；All-hit@10 表示覆盖该行全部真实事件。Recall@k 是真实事件的覆盖比例，含义不同。点估计均用完整封存测试目标行；95% CI 沿用本报告的 200 次目标行 bootstrap 协议。", ""]
        lines += header("模型|校准目标行|测试目标行",EXTERNAL_METRICS)
        external_states = [(name, reports.get(key)) for name, key in LABELS if key != "maomao"]
        external_states += [("MAOMAO（校准前）", raw_by_site.get(site)),
                            ("MAOMAO（校准后）", reports.get("maomao"))]
        for model_label, report in external_states:
            if report:
                lines.append(metric_row([model_label,
                                         f"{manifest['calibration_target_rows']:,}",
                                         f"{manifest['test_target_rows']:,}"], report,EXTERNAL_METRICS))
            else:
                lines.append("|" + "|".join([model_label,
                                            f"{manifest['calibration_target_rows']:,}",
                                            f"{manifest['test_target_rows']:,}"] +
                                           ["待评估"] * len(EXTERNAL_METRICS)) + "|")
        lines += ["", "**63 小类 Family 命中（同一批测试行）**：将前 k 个事件预测映射到模型已有的 63 个语义 family，再判断是否命中真实 family。Family All-hit@10 表示前 10 个事件的 family 覆盖全部真实 family；没有把 63 个 family 重新排名取 top-k。", ""]
        lines += header("模型|校准目标行|测试目标行",FAMILY_METRICS)
        for model_label,report in external_states:
            if report:
                lines.append(metric_row([model_label,f"{manifest['calibration_target_rows']:,}",f"{manifest['test_target_rows']:,}"],report,FAMILY_METRICS))
        if manifest["test_target_rows"] < 2:
            lines += ["", "此队列的封存测试集只有 1 个有效目标行；逐标签宏 AUROC 与目标行 bootstrap 95% CI 在统计上不可计算，表中按实际情况标记。"]
        post = reports.get("maomao")
        if post:
            lines += ["", f"MAOMAO：校准前为原始输出，校准后方法为 {post.get('calibration_family','temperature_only')}，温度为 {post['temperature']:.6f}；"
                      "两行在相同完整封存测试目标行上评分。校准不保证所有测试指标改善。"]
            fitted = load(EXTERNAL / site / "maomao/calibration_fit.json")
            if fitted and PROTOCOL!=LEGACY_PROTOCOL:
                split=fitted['selection_split']
                lines += [f"90%校准数据内部：{split['fit_groups']:,}个有有效目标的患者/记录代理（{split['fit_rows']:,}行）拟合，"
                          f"{split['selection_groups']:,}个独立患者/记录代理（{split['selection_rows']:,}行）选择；"
                          f"选定方法最终使用全部{fitted['calibration_rows']:,}校准行重新拟合。"
                          f"全校准集正事件集合NLL由{fitted['raw_calibration']['positive_set_nll']:.6f}变为{fitted['fitted_calibration']['positive_set_nll']:.6f}。"
                          "参数、候选选择分数和完整优化记录见calibration_fit.json；封存测试结果未用于选择。"]
                final_steps=[x for x in fitted['optimization_history'] if x['scope']=='all' and x['family']==fitted['family']]
                if final_steps:
                    step=final_steps[-1]
                    lines += [f"全90%重拟合优化器状态：{'报告收敛' if step['converged'] else '未报告收敛'}；"
                              f"{step['iterations']}次迭代、{step['full_row_passes']}次全行目标/梯度评估；停止原因：{step['message']}。"
                              "采用预设预算内参数，不根据封存测试结果延长优化；不能据此宣称校准达到全局最优。"]
            elif fitted and fitted.get("fitted_temperature_accepted") is False:
                lines += [f"完整90%校准集拟合候选T={fitted['fitted_temperature']:.6f}，"
                          f"Brier由{fitted['raw_calibration']['brier']:.6f}变为{fitted['fitted_calibration']['brier']:.6f}，"
                          "未达到预设改善条件，因此校准后采用T=1，两个状态相同。选择没有使用10%测试结果。"]
        lines += ["", f"结果目录：{(EXTERNAL / site).relative_to(ROOT)}。", ""]
    if not external_done:
        lines += ["当前状态：部分外部队列仍在评估；全部五模型结果齐备后统一更新本节。", ""]

    lines += ["", "## 最终指标比较汇总", "",
              "完整五模型指标见前述内部表和各外部数据集表。下表汇总外部主要指标；差值为点估计差，不作为显著性检验："
              "基线最高值取单变量、逻辑回归、XGBoost、ANN 中已完成结果的最大点估计，"
              "用于展示差值，不用于训练或调整超参数。", "",
              "|数据集|测试目标行|基线最高 micro-AUPRC（模型；95% CI）|MAOMAO 校准前 micro-AUPRC（95% CI）|MAOMAO 校准后 micro-AUPRC（95% CI）|校准后减基线|基线最高 micro-AUROC（模型；95% CI）|MAOMAO 校准前 micro-AUROC（95% CI）|MAOMAO 校准后 micro-AUROC（95% CI）|校准后减基线|",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, site in SITES:
        entry = site_data.get(site)
        if not entry:
            continue
        _, manifest, reports = entry
        raw, post = raw_by_site.get(site), reports.get("maomao")
        row = [label, f"{manifest['test_target_rows']:,}"]
        for metric in ("micro_auprc", "micro_auroc"):
            candidates = [(name, reports[key]) for name, key in LABELS if key != "maomao" and key in reports]
            best = max(candidates, key=lambda item: item[1][metric]) if len(candidates) == 4 else None
            row += [f"{best[0]}；{cell(best[1], metric)}" if best else "待评估",
                    cell(raw, metric), cell(post, metric),
                    f"{post[metric] - best[1][metric]:+.4f}" if best and post else "待评估"]
        lines.append("|" + "|".join(row) + "|")
    lines.append("")

    lines += ["## 外部来源筛选与指标解释", "",
              "下列来源说明采用与冻结 MAOMAO 输入契约一致的实际映射；缺失静态变量按训练契约填零。"
              "当前对比与校准前后表使用患者级 90:10 拆分。", "",
              "|来源|当前队列及输入范围|",
              "|---|---|",
              "|MIMIC|手术室转入 ICU 的手术 episode；映射有时间戳的生命体征、实验室、药物与阶段事件。|",
              "|MOVER|手术 episode；映射 EPIC flowsheet、实验室、药物与阶段事件。ICU_ADMIN_FLAG 只有 Yes/No，不能视为 ICU 病区类型。|",
              "|eICU|明确 APACHE Operative 分支或入 ICU 后 4 小时内手术室证据；输入 ICU 内的观测与药物事件。|",
              "|SICdb|Elective Surgery 或 Urgent Surgery；映射生命体征、实验室与可匹配药物，排除 No Surgery/Unknown。|",
              "|NTUH|明确手术记录，当前动态输入为 ECG 派生心率。|",
              "|ASAC、UQ|明确手术病例；输入可映射的 HR、动脉/袖带血压、RR、SpO₂、EtCO₂、体温。|", "",
              "事件指标使用 210 类结局中的有效多标签目标行。micro 指标合并目标行与类别；"
              "macro 指标按可估计类别平均，正负样本不足的类别不计算 AUROC。"
              "AUPRC/AUROC、MRR、Hit@1、Recall@5/10 越高越好，Brier/ECE 越低越好。"
              "Brier 与 ECE 的具体定义以本包 event_metrics.py 和各指标 JSON 为准。", "",
              ("外部校准恢复 V5 温度＋类别偏置及校准内部候选选择，最终拟合使用全部90%有效校准行，MAOMAO权重冻结；" if PROTOCOL!=LEGACY_PROTOCOL else
               "外部温度校准使用全部 90% 校准行拟合一个标量，MAOMAO 权重冻结；")+
              "这里的校准前后比较属于事件概率校准。时间结果仅保留独立的分尺度 MAE 表。"+
              ("类别偏置能够改变同一目标行内的事件排序，Hit/Recall/MRR须实际重新评估；只有零偏置的正温度缩放保持排序。" if PROTOCOL!=LEGACY_PROTOCOL else
               "温度缩放不改变同一目标行内的事件排序，因此 Hit/Recall/MRR 可能完全不变；")+
              "跨目标行的概率与校准误差按实际结果报告。", "",
              "原始 MAOMAO 的外部结果衡量冻结模型直接迁移的表现；使用当地90%带标签校准数据后的结果衡量当地适配后的表现。"
              "两种状态在相同封存10%测试行上并列展示，校准后结果不能替代原始结果作为无需当地标签的迁移证据。", "",
              "校准需求取决于输出用途：事件排序可以直接评估原始模型；需要解释绝对发生概率、采用风险阈值或计算临床DCA时，必须另行核查对应结局的概率校准。"
              "本节next-event softmax表示事件间归一化分布，不能解释为各事件独立发生的绝对风险；临床风险校准见后述独立临床评分研究。", "",
              "小型手术队列的封存患者与目标行较少，点估计和 CI 仅作描述；"
              "三源合并集与 NTUH、ASAC、UQ 重叠，不与这三个来源共同计作四次独立外部验证。", ""]

    lines += ["## 七项模块消融", "",
              "七项变体使用相同 INSPIRE 患者级 9:1 划分。每轮训练遍历完整固定窗口训练集"
              "（208,303 个窗口，对应 14,128,539 个有效目标行），"
              "点估计均来自完整 1,563,972 个验证目标行。", ""]
    lines += header("移除模块|最佳 epoch|验证目标行")
    for label, key in MODULES:
        data = load(OUT / f"module_metrics/{key}.json")
        if not data:
            raise RuntimeError(f"Missing current full-row ablation: {key}")
        validate(data, key, valid)
        lines.append(metric_row([label, str(data["checkpoint_epoch"]), f"{valid:,}"], data))
    lines += ["", f"消融评估 JSON：{(OUT / 'module_metrics').relative_to(ROOT)}。", "",
              "## MAOMAO 分尺度时间误差", ""]
    timescale = load(OUT / "model_metrics/maomao_dual_timescale_mae_internal.json")
    if not timescale:
        raise RuntimeError("Missing current dual-timescale metric artifact")
    lines += ["|时间尺度|有效目标行|MAE（小时；95% CI）|",
              "|---|---:|---:|"]
    for key in ("fine_0_to_2h", "long_2h_to_tail_start", "tail_from_tail_start"):
        scale = timescale["time_mae_by_scale"][key]
        ci = scale["mae_hours_95ci"]
        label={"fine_0_to_2h":"<2 h","long_2h_to_tail_start":"2–<24 h","tail_from_tail_start":"≥24 h"}[key]
        lines.append(f"|{label}|{int(scale['event_target_rows']):,}|"
                     f"{scale['mae_hours']:.4f} ({ci[0]:.4f}–{ci[1]:.4f})|")
    lines += ["", f"分尺度指标 JSON："
              f"{(OUT / 'model_metrics/maomao_dual_timescale_mae_internal.json').relative_to(ROOT)}。", "",
              "## 结果状态", "",
              "五模型内部验证和七项模块消融已完成。"
              + ("七个独立外部数据集及三源合并集的五模型 9:1 全量评估已完成。"
                 if external_done else "外部五模型全量评估正在继续，尚未汇总为最终完成。"), ""]
    scale_root = ROOT / "outputs/scale_ablations_richctx_20260928"
    scale_state = load(scale_root / "verification.json")
    if (scale_root / "manifest.json").exists() and not (scale_state and scale_state.get("complete")):
        lines += ["新增模型规模、输出类别和上下文实验正在全量训练/评估；其最终数值尚未纳入本报告，完成核验后更新。", ""]
    progress = load(XGB_CANDIDATE / "status.json")
    if progress and current_xgboost_dir() != XGB_CANDIDATE:
        lines += [f"本轮 XGBoost 续训状态：{progress.get('status')}；目标总计 3 轮，"
                  "保留逐轮 checkpoint；训练阶段上限 60 分钟、单轮上限 45 分钟，匿名/共享驻留内存上限 40 GiB，系统可用内存至少 8 GiB。"
                  "当前表格引用已完成的模型；续训结果完成评估后更新。"
                  "当前尝试按标签串行训练，共用全量特征矩阵，只有同一轮的 210 个标签全部完成后才保存该轮。", ""]
    if progress and current_xgboost_dir() != XGB_CANDIDATE:
        saved_rounds = sorted(int(path.stem.rsplit("_",1)[1])
                              for path in XGB_CANDIDATE.glob("checkpoint_round_[0-9][0-9][0-9].json"))
        if saved_rounds:
            lines += [f"已完整保存总计 {saved_rounds[-1]} 轮的 XGBoost checkpoint；"
                      "新增模型尚待完整内部验证及外部重新评分，当前指标表暂未替换。", ""]
        if progress.get("status") == "training":
            lines += [f"当前第 {progress.get('fitting_round')} 轮已完成 "
                      f"{progress.get('completed_targets_in_round',0)}/210 个标签；"
                      "该轮未全部完成前不计为完整轮次。", ""]
    clinical = (ROOT / "docs/MAOMAO_CLINICAL_SCORE_COMPARISON.md").read_text()
    clinical = clinical.split("\n", 1)[1].strip()
    clinical = re.sub(r"\[([^\]]+)\]\((?!https?://)[^\)]+\)", r"\1", clinical)
    clinical = re.sub(r"(?m)^(#{2,5}) ", r"#\1 ", clinical)
    lines += ["## MAOMAO 与临床评分预测能力比较", "",
              "本节合入 MAOMAO_CLINICAL_SCORE_COMPARISON.md 的当前结果。"
              "该独立研究使用 INSPIRE 患者级 85% development / 15% 封存测试、单独训练的冻结 MAOMAO；"
              "其手术结束时点的临床结局和患者级配对 bootstrap 与前述 9:1 多事件评估不同，指标分别解释。"
              "完整评分文档、42 组配对矩阵、汇总 JSON 和划分记录位于结果包 clinical_scores/ 目录。", "",
              clinical, ""]
    scale_dir = ROOT / "outputs/scale_ablations_richctx_20260928"
    scale_verification = load(scale_dir / "verification.json")
    if scale_verification and scale_verification.get("complete") is True:
        scale = (ROOT / "docs/MAOMAO_SCALE_ABLATIONS_CURRENT.md").read_text().split("\n", 1)[1].strip()
        scale = re.sub(r"(?m)^(#{2,5}) ", r"#\1 ", scale)
        lines += ["## 当前框架全量规模、输出类别与上下文比较", "", scale, ""]
    shap_root=ROOT/'outputs/maomao_plot_sources/shap'
    shap_proof=load(shap_root/'delivery_verification.json')
    if shap_proof and shap_proof.get('complete'):
        numerics=load(shap_root/'numerical_verification.json')
        lines += ["## MAOMAO family SHAP 与学习到的事件嵌入", "",
                  f"完成全部 {numerics['patients_verified']:,} 位内部验证患者的解释，每人一个seed42预先固定查询；并非全部验证目标行。"
                  "使用冻结全量训练MAOMAO、真实Partition SHAP与真实embedding。主图按17个临床大类显示，63-family细节和完整逐case273输出贡献同时保留。", "",
                  "[完整SHAP方法、临床大类字典、预算敏感性与补充图](MAOMAO_FAMILY_SHAP.md)。"
                  "Figure 4a式面板为UMAP嵌入和四个局部放大框；Figure 4c式面板分别展示<2h、2–<24h、≥24h三个尺度的历史事件贡献，共用相同事件行列与色阶。"
                  "分界与本报告分尺度时间误差一致；SHAP图表示既往信息滞后，时间误差表按未来真实等待时间分层，二者统计含义不同。exp(φ)为概率贡献倍数。", "",
                  "![MAOMAO event embedding](plot_sources/shap/figures/figure4a_maomao_embeddings.png)", "",
                  "![MAOMAO three-timescale SHAP](plot_sources/shap/figures/figure4c_maomao_shap_by_family.png)", "",
                  "![MAOMAO clinical family importance](plot_sources/shap/figures/maomao_shap_clinical_group_importance.png)", "",
                  "SHAP为max_evals500的分层有限预算近似；25例500→1000预算对照仍有局部差异，不能声称已经收敛或存在因果机制。"
                  "贡献条件于固定时间网格、intensity、phase、static和其它上下文，mask baseline并非健康人。"
                  "PDF/PNG/TIFF、矩阵、逐case贡献、支持数、来源与可独立重绘脚本全部纳入结果包plot_sources/shap/，不交付 SVG 图片。", ""]
    manuscript=load(ROOT/'outputs/maomao_manuscript_figures_20260929/delivery_verification.json')
    if manuscript:
        lines += ["## 当前稿件 Figure 1–5", "",
                  "正文 Arial 8 pt、标题 Arial 10 pt，最终宽度 210 mm；PDF 严格矢量。"
                  "[逐图方法、口径和来源说明](MAOMAO_MANUSCRIPT_FIGURES.md)。"
                  "Figure 5 原 c 的三尺度热图在新布局改为 d：<2 h、2–<24 h、≥24 h，与上述未来时间 MAE 表统一分界。"
                  "热图衡量既往信息的滞后，MAE 表按未来真实等待时间分层，二者不混为同一指标。", "",
                  "|图|内容|当前状态|文件|", "|---|---|---|---|"]
        figure_names={1:'MAOMAO 架构与数据流程',2:'内部/外部五模型对比',3:'模块、大小、输出数、上下文消融',
                      4:'六个临床结局的共同病例 ROC',5:'UMAP、重要性、蜂巢、三尺度 SHAP 与四个依赖图'}
        for number in range(1,6):
            if number in manuscript['verified_figures']:
                files=f'[PDF](manuscript_figures/figures/Figure_{number}.pdf) / [PNG](manuscript_figures/figures/Figure_{number}.png)'
                state='图源、字体/字号、矢量、对齐、碰撞及人工查看已核查'
            else:
                files='尚未作为完成文件交付';state='外部全量评估/最终图核查尚未完成'
            lines.append(f'|Figure {number}|{figure_names[number]}|{state}|{files}|')
        if not manuscript['complete']:
            lines += ['', '修订图的最终核查尚未全部完成；本节仅交付已通过核查的当前版本。']
        if manuscript.get('verified_supplements'):
            lines += ['', '补充图：S3 展示六组外部消融指标（空心蓝圆＝校准前，实心玫红菱形＝校准后）；S4.1–S4.10 每个结局包含原始 ROC/PR、统一训练映射后的校准/DCA 和 MAOMAO 原始未校准状态；S5 为九个固定匿名病例的 3×3 箭头瀑布图。',
                      '', '2026-09-30 校准/DCA 修正：此前临床评分已映射概率而 MAOMAO 未校准，比较口径不一致。现对所有模型在同一结局训练完整病例交集拟合风险映射，测试不参与拟合；MAOMAO 原始输出保留，AUROC/AP 不变。MAOMAO 校准来源为模型训练患者预测；最终测试独立，未声称额外独立校准队列。各结局原始/校准概率均值、Brier（95% CI）及完整原因见 [校准与 DCA 核查](MAOMAO_CLINICAL_CALIBRATION_DCA_AUDIT.md)。', '']
            audit_doc=(ROOT/'docs/MAOMAO_CLINICAL_CALIBRATION_DCA_AUDIT.md').read_text()
            audit_table=audit_doc.split('## MAOMAO：相同封存测试病例上的原始与训练校准结果',1)[1].split('## 数据与图的使用限制',1)[0].strip()
            lines += ['', '### 临床结局 MAOMAO 原始/训练校准概率指标（相同封存测试病例）', '', audit_table, '', '完整 62 个模型/状态的概率指标、患者聚类 95% CI 和训练基准发生率阈值处 DCA 见 manuscript_figures/source_data/clinical_probability_metrics.csv。输血 6h/24h 只有 3/4 个阳性，结果不稳定；所有不佳结果均保留。', '']
            for name in manuscript['verified_supplements']:
                lines.append(f'- [{name} PDF](manuscript_figures/figures/{name}.pdf) / [PNG](manuscript_figures/figures/{name}.png)')
        row_audit=load(ROOT/'outputs/maomao_manuscript_figures_20260929/external_ablation_row_verification.json')
        if row_audit and row_audit.get('complete'):
            from scripts.diagnostics.build_manuscript_external_ablation_report import build as build_external_ablations
            build_external_ablations()
            external_doc=(ROOT/'docs/MAOMAO_MANUSCRIPT_EXTERNAL_ABLATIONS.md').read_text().split('\n',1)[1].strip()
            external_doc=re.sub(r'(?m)^(#{2,5}) ',r'#\1 ',external_doc)
            lines += ['', '### Figure 3 全量外部消融与校准前后完整指标', '', external_doc, '']
        common_doc=ROOT/'docs/MAOMAO_CLINICAL_COMMON_COHORT_CURVES.md'
        if 4 in manuscript['verified_figures'] and common_doc.exists():
            common=common_doc.read_text().split('\n',1)[1].strip()
            common=re.sub(r'(?m)^(#{2,5}) ',r'#\1 ',common)
            lines += ['', '### Figure 4 共同病例分析与临床评分比较', '', common, '']
    temporary = REPORT.with_suffix(REPORT.suffix + ".tmp")
    temporary.write_text("\n".join(lines))
    temporary.replace(REPORT)
    print(json.dumps({"report": str(REPORT), "external_complete": external_done,
                      "external_sites": {site: len(data[2]) for site, data in site_data.items()}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
