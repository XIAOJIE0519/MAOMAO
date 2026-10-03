#!/usr/bin/env python3
"""Describe the exact common-cohort ROC/PR source for manuscript Figure 4."""
import json
import sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.diagnostics.uniform_result_scope import sha256
DATA=ROOT/'outputs/maomao_manuscript_figures_20260929/source_data'
def build():
    dictionary=json.loads((DATA/'clinical_curve_dictionary.json').read_text())
    metrics=pd.read_csv(DATA/'clinical_common_metrics.csv');delta=pd.read_csv(DATA/'clinical_common_deltas.csv')
    proof=json.loads((DATA/'clinical_numerical_verification.json').read_text())
    if not proof['complete']:raise RuntimeError('Exact clinical curve verification is missing')
    def cell(row,value='value'):
        return f'{row[value]:.4f} ({row.ci_lower:.4f}–{row.ci_upper:.4f})'
    risk=json.loads((DATA/'clinical_risk_preparation.json').read_text())
    if not risk['complete'] or risk['test_patient_overlap'] or risk['fitted_on_sealed_test']:raise RuntimeError('Training-only risk provenance missing')
    lines=['# Figure 4 及 S4：同结局、同病例的临床评分 ROC / PR / 校准 / DCA','',
           '本表来自独立冻结 INSPIRE 临床评分 85:15 患者划分研究。每个结局的 MAOMAO 和全部预定评分都使用同一批完整病例；与原先评分各自按缺失情况形成的不同病例子集分开解释。原始 19,554 个测试手术记录均进入交集筛选，排除数如下。封存测试不参与训练、校准或模型筛选。','',
           'AUROC 使用经验 ROC；AP 使用所有不同分数阈值的 average precision。每一结局进行 200 次患者聚类配对 bootstrap，同一次患者重采样同时用于所有评分。所有区间为 95% percentile CI；阳性过少导致无两类的重采样被记录为不可估计，实际有效次数逐项保留。图及本表不展示 MAOMAO-minus-score；原配对差值保留在源 CSV。','',
           '主图仅六个 ROC；S4.1–S4.10 每个结局展示原始 ROC/PR、全部模型统一训练映射后的校准/DCA，以及 MAOMAO 原始未校准状态。所有风险映射均在 development 内真正参与 MAOMAO 训练的 69,920 位患者中的同一结局完整病例交集拟合；MAOMAO 用原 focal-loss sigmoid 的 logit，评分用固定风险方向的原评分。与封存测试零患者重叠，内部验证也未参与拟合。新增映射不是已发表评分概率公式；MAOMAO 的原始输出与校准结果均保留，AUROC/AP 不变。校准最多 10 个等人数分箱，保留并列值；DCA 视窗按原规则保留。MAOMAO 误差条/阴影为 200 次患者聚类 bootstrap 的点态 95% CI，条件于冻结映射；DCA 为探索性。','',
           '2026-09-30 修正了之前临床评分已映射概率而 MAOMAO 未校准的口径不一致问题。完整原因、原始/校准 Brier 和概率均值、限制见 [校准与 DCA 核查](MAOMAO_CLINICAL_CALIBRATION_DCA_AUDIT.md)。MAOMAO 校准采用模型训练病例预测，不能声称另有独立校准队列；最终封存测试独立。','',
           'NEWS2 / MEWS 为可用生理部分和，qSOFA 为两分量，GS-AKI / ARISCAT 为适配版本，RCRI 为代理版本；不能当作完整标准临床评分。SAS 按预先规定的较低分更高风险方向使用。完整字段和风险方向见 clinical_curve_dictionary.json。','',
           '输血 6h / 24h 交集仅有 3 / 4 个阳性事件，PR/AP 与差值区间不稳定。负差值或跨零区间均如实保留，不能一概声称 MAOMAO 提升。','']
    for endpoint,cohort in dictionary.items():
        lines += [f'## {endpoint}','',f'共同病例 {cohort["complete_episodes"]:,} 条，{cohort["patients"]:,} 位患者，{cohort["events"]} 个阳性；排除任一评分/标签缺失或非有限值 {cohort["excluded_missing_any_comparator_or_target"]:,} 条。','',
                  '|模型/指标|AUROC (95% CI)|AP (95% CI)|有效 bootstrap 次数（ROC/AP）|',
                  '|---|---:|---:|---:|']
        for spec in cohort['series']:
            name=spec['name'];r=metrics[(metrics.endpoint==endpoint)&(metrics.model==name)]
            au=r[r.metric=='auroc'].iloc[0];ap=r[r.metric=='average_precision'].iloc[0]
            lines.append('|'+ '|'.join([name,cell(au),cell(ap),f'{int(au.valid_bootstrap_repeats)}/{int(ap.valid_bootstrap_repeats)}'])+'|')
        lines.append('')
    lines += ['## 来源核查','',f'104 个 AUROC/AP 点估计已独立与 scikit-learn 核对，误差均 <1e−12；各 PR 阶梯曲线积分等于报告 AP。来源 CSV SHA256：{json.loads((DATA/"clinical_preparation.json").read_text())["source_sha256"]}。',
              '完整曲线、点估计/CI、配对差值/CI、病例交集计数、指标字典及核查 JSON 位于结果包 manuscript_figures/source_data/。','']
    target=ROOT/'docs/MAOMAO_CLINICAL_COMMON_COHORT_CURVES.md';target.write_text('\n'.join(lines))
    return {'report':str(target),'sha256':sha256(target),'endpoints':len(dictionary),'comparisons':42}
if __name__=='__main__':print(json.dumps(build(),ensure_ascii=False))
