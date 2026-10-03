"""Report the current experiments without mixing historical metrics."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.diagnostics.scale_ablation_scope import *
from scripts.diagnostics.build_uniform_results_report import header, metric_row

REPORT = ROOT / "docs/MAOMAO_SCALE_ABLATIONS_CURRENT.md"
LABELS = {"model_small": "小模型（256/6/8/1024）", "model_large": "大模型（512/12/16/2048）",
          "reference": "当前 MAOMAO（384/10/12/1536；复用）", "context_64": "64 事件上下文段",
          "context_128": "128 事件上下文段", **{f"vocab_{n}": f"{n} 类输出" for n in (50, 100, 150)},
          **{f"reference_vocab_{n}": f"当前 MAOMAO → 相同 {n} 类" for n in (50, 100, 150)}}


def expected_rows(name, manifest):
    train_rows = manifest["train_rows_full"]
    validation_rows = manifest["validation_rows_full"]
    if "vocab_" in name:
        spec = manifest["output_specifications"][f"vocab_{name.rsplit('_', 1)[1]}"]
        validation_rows = spec["eligible_rows"]["validation"]
        if not name.startswith("reference_"):
            train_rows = spec["eligible_rows"]["train"]
    return train_rows, validation_rows


def recorded_status(name, state):
    if (OUT / "metrics" / f"{name}.json").exists():
        return "评估指标已输出；最终来源核验另列"
    stage = state.get("stage", "")
    worker_name = "reference" if name.startswith("reference") else name
    if stage in {f"{worker_name}_training", f"{worker_name}_evaluation"}:
        if state.get("status") == "failed":
            return "该阶段失败，待处理"
        return "训练中（队列记录）" if stage.endswith("_training") else "评估中（队列记录）"
    if worker_name == "reference":
        return "复用已训练 checkpoint；等待本轮评估"
    if state.get("models", {}).get(name, {}).get("training") == "completed":
        return "训练已完成；等待评估指标"
    return "等待训练"


def main():
    manifest = read(OUT / "manifest.json")
    state_path = OUT / "queue_status.json"
    state = read(state_path) if state_path.exists() else {}
    result = read(OUT / "verification.json") if (OUT / "verification.json").exists() else {"complete": False}
    lines = ["# 当前 MAOMAO 输出类别、模型规模与上下文实验", "",
             "使用与最终五模型、七项模块消融相同的 richctx_static7 数据版本、七项静态输入和双时间尺度头。"
             "INSPIRE 患者级 90:10、seed 42；训练 89,897 名患者，验证 9,989 名患者。"
             "每个新模型从头训练 7 个完整 epoch，208,303 个训练窗口和 23,194 个验证窗口；"
             "固定 block_size=256、stride=128，batch_size=64、梯度累积=1，优化器、学习率、辅助损失等复用当前 MAOMAO 配置，未设行数或 step 上限。", "",
             "当前 MAOMAO 是已完成的独立参照，复用 checkpoint；不新增 Full MAOMAO 模块消融。"
             "原始训练目标行 14,128,539，完整验证目标行 1,563,972。", "",
             "输出词表按全部 90% 训练目标行中的类别支持数选取前 50/100/150 类，验证集不参与选择。"
             "保留原输入、原始下一事件时点和原始多标签集合，再投影到选定类别。"
             "没有选定阳性类别的行不作为该词表的事件/观测等待时间监督，但输入及辅助任务保留；"
             "真正的右删失保留。训练遍历全部窗口，评估使用该词表全部有效目标行，无随机抽样。"
             "对应 MAOMAO 参照在完全相同的目标行、类别和 softmax 归一化范围上评分；"
             "参照实际仍训练全部 210 类、14,128,539 个事件目标行，投影仅发生于评估；"
             "不会把参照写成在缩小词表上重新训练。"
             "不同词表之间的原始分数不能直接用于证明模型优劣。", "",
             "上下文实验在原有 256 事件窗口内隔离连续的 64/128 事件段，所有 encoder 层及 masked 辅助任务均无法跨段访问。"
             "每个预测目标的可见事件范围限于所在段起点至当前事件，继续执行 MAOMAO 原有因果及同时间兄弟事件屏蔽规则。"
             "64/128 是段长上限（含当前事件），实际可见事件数随目标位置及屏蔽规则变化；"
             "256 参照使用原窗口前缀。原来的窗口起点长期事件族摘要、静态变量、每 token 的阶段/观测强度保留。"
             "这是局部上下文段长度实验，不能解释为移除全部历史；目标坐标、标签和训练样本数不变。", "",
             "AUROC/AUPRC、MRR、Hit@1、Recall@5/10、Brier/ECE 的点估计使用全部有效目标行。"
             "各指标 95% CI 与当前 MAOMAO 一致：200 次固定种子 bootstrap，30,000 行均匀样本并按样本量缩放宽度，属于近似 CI。"
             "时间误差仅按 <2 h、2–<24 h、≥24 h 分别报告，使用尺度内全部目标行的 200 次 bootstrap；不汇报混合尺度均值。", ""]
    for title, names in (("模型规模", ("model_small", "reference", "model_large")),
                          ("输出类别数：与同任务 MAOMAO 配对比较", ("vocab_50", "reference_vocab_50", "vocab_100", "reference_vocab_100", "vocab_150", "reference_vocab_150")),
                          ("局部上下文段长度", ("context_64", "context_128", "reference"))):
        lines += [f"## {title}", ""] + header("版本|训练有效事件目标行|验证有效事件目标行")
        for name in names:
            path = OUT / "metrics" / f"{name}.json"
            if path.exists():
                data = read(path)
                lines.append(metric_row([LABELS[name], f"{data['train_event_rows_eligible']:,}", f"{data['evaluation_rows']:,}"], data))
            else:
                train_rows, validation_rows = expected_rows(name, manifest)
                lines.append(f"|{LABELS[name]}|{train_rows:,}|{validation_rows:,}|" + "|".join(["尚未完成"] * len(METRICS)) + "|")
        lines.append("")
    lines += ["## 实际拟合记录", "", "|版本|输出类别数|参数量|最佳 checkpoint epoch|训练预算|",
              "|---|---:|---:|---:|---|"]
    for name in ("reference", *NAMES):
        path = OUT / "metrics" / f"{name}.json"
        if path.exists():
            data = read(path)
            lines.append(f"|{LABELS[name]}|{data['output_classes']}|{data['parameter_count']:,}|{data['checkpoint_epoch']}|7 个完整 epoch|")
    lines.append("")
    if result["complete"]:
        reference = read(OUT / "metrics/reference.json")
        lines += ["## 结果解读", "",
                  "以下差值均为点估计差，不作为显著性结论；近似单模型 CI 不等于配对差值的 CI。"
                  "AUPRC/AUROC 越高更好，Brier/ECE 和分尺度 MAE 越低更好。", ""]
        small = read(OUT / "metrics/model_small.json")
        large = read(OUT / "metrics/model_large.json")
        lines += [f"小模型参数量为当前 MAOMAO 的 {small['parameter_count']/reference['parameter_count']:.2f} 倍，"
                  f"大模型为 {large['parameter_count']/reference['parameter_count']:.2f} 倍；"
                  "结合下列区分能力、校准误差和分尺度时间误差查看参数量增加带来的实际变化。", ""]
        for name in ("model_small", "model_large", "context_64", "context_128"):
            data = read(OUT / "metrics" / f"{name}.json")
            lines += [f"- {LABELS[name]} 相对当前 MAOMAO：micro-AUPRC 差值 "
                      f"{data['micro_auprc']-reference['micro_auprc']:+.4f}，micro-AUROC 差值 "
                      f"{data['micro_auroc']-reference['micro_auroc']:+.4f}，Brier 差值 "
                      f"{data['brier']-reference['brier']:+.4f}，ECE 差值 "
                      f"{data['ece']-reference['ece']:+.4f}。"]
        for size in (50, 100, 150):
            data = read(OUT / "metrics" / f"vocab_{size}.json")
            paired = read(OUT / "metrics" / f"reference_vocab_{size}.json")
            lines += [f"- {size} 类专用输出模型相对同类别、同目标行 MAOMAO：micro-AUPRC 差值 "
                      f"{data['micro_auprc']-paired['micro_auprc']:+.4f}，micro-AUROC 差值 "
                      f"{data['micro_auroc']-paired['micro_auroc']:+.4f}，Brier 差值 "
                      f"{data['brier']-paired['brier']:+.4f}，ECE 差值 "
                      f"{data['ece']-paired['ece']:+.4f}。"]
        lines.append("")
    lines += ["## 分尺度时间误差", "", "|版本|时间尺度|有效目标行|MAE（小时；95% CI）|", "|---|---|---:|---:|"]
    for name in ("reference", *NAMES, "reference_vocab_50", "reference_vocab_100", "reference_vocab_150"):
        path = OUT / "metrics" / f"{name}_time_scales.json"
        if not path.exists():
            continue
        for scale_key,group in read(path)["time_mae_by_scale"].items():
            ci = group["mae_hours_95ci"]
            value = f"{group['mae_hours']:.4f} ({ci[0]:.4f}–{ci[1]:.4f})" if ci else "不可估计"
            scale_label={"fine_0_to_2h":"<2 h","long_2h_to_tail_start":"2–<24 h","tail_from_tail_start":"≥24 h"}[scale_key]
            lines.append(f"|{LABELS[name]}|{scale_label}|{group['event_target_rows']:,}|{value}|")
    lines += ["", "## 状态与复现", "",
              "全量训练、全量评估和来源核验全部完成。" if result["complete"] else "新一轮尚在进行，旧实验保留至新结果全部核验完成。",
              "未完成行显示的是冻结数据定义下的目标行数，不表示训练或评估已完成。", "",
              "|版本|队列记录状态|", "|---|---|"]
    for name in ("reference", *NAMES, "reference_vocab_50", "reference_vocab_100", "reference_vocab_150"):
        lines.append(f"|{LABELS[name]}|{recorded_status(name, state)}|")
    lines += ["",
              "结果包中的 scale_ablations/ 包含配置、词表选择、完整指标与核验记录；不包含消融 checkpoint 和大型评分缓存。", "",
              "运行入口：scripts/diagnostics/run_scale_ablation_queue.py；核验入口：scripts/diagnostics/verify_scale_ablations.py。", ""]
    receipt = OUT / "legacy_removal.json"
    if receipt.exists() and read(receipt).get("status") == "completed" and read(receipt).get("all_directories_absent"):
        lines += ["旧版模型规模与输出词表实验及其备份汇总已删除；路径清单和替换核验证据见结果包 scale_ablations/legacy_removal.json。", ""]
        if read(receipt).get("edited_reports") and read(receipt).get("all_edited_reports_clean"):
            lines += ["历史 Markdown 快照中的旧版规模与词表数值章节也已清理，修改前后的 SHA256 记录在同一清理凭据中。", ""]
    REPORT.write_text("\n".join(lines))
    print(str(REPORT), flush=True)


if __name__ == "__main__":
    main()
