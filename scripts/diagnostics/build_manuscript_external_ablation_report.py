#!/usr/bin/env python3
"""Report the fully audited external Figure 3 comparisons without retraining."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.diagnostics.verify_manuscript_external_ablation_rows import verify
from scripts.diagnostics.uniform_result_scope import sha256
from maomao.evaluation.softmax_calibration import PROTOCOL, LEGACY_PROTOCOL
OUT=ROOT/'outputs/maomao_manuscript_figures_20260929'
DEST=ROOT/'docs/MAOMAO_MANUSCRIPT_EXTERNAL_ABLATIONS.md'
SITES={'mimic':'MIMIC-IV','mover':'MOVER','eicu':'eICU','sicdb':'SICdb','surgical_pooled':'NTUH / ASAC / UQ 三源合并'}
MODELS={'no_block_causal':'移除 block-causal','no_relative_time':'移除 relative-time bias',
        'no_family_head':'移除 family hierarchical head','no_clock_phase_summary':'移除 clock/phase-summary',
        'no_measurement_intensity':'移除 measurement-intensity','no_masked_event_value':'移除 masked-event/value auxiliary loss',
        'no_event_conditioned_time':'移除 event-conditioned time','model_small':'小模型','model_large':'大模型',
        'context_64':'64 事件上下文','context_128':'128 事件上下文',
        **{f'vocab_{k}':f'{k} 类专用模型' for k in (50,100,150)},
        **{f'reference_vocab_{k}':f'当前 MAOMAO 投影至 {k} 类' for k in (50,100,150)}}
BLOCKS=(('区分能力',(('micro_auprc','micro-AUPRC'),('macro_auprc','macro-AUPRC'),('micro_auroc','micro-AUROC'),('macro_auroc','macro-AUROC'))),
        ('检索与概率误差',(('mrr','MRR'),('hit_at_1','Hit@1'),('recall_at_5','Recall@5'),('recall_at_10','Recall@10'),('brier','Brier'),('ece','ECE'))))
def read(p):return json.loads(p.read_text())
def cell(a,key):
    v=float(a[key]);lo,hi=map(float,a[key+'_95ci']);precision=4
    while lo<hi and precision<10 and f'{lo:.{precision}f}'==f'{hi:.{precision}f}':precision+=1
    return f'{v:.{precision}f} ({lo:.{precision}f}–{hi:.{precision}f})'
def build():
    proof=verify(require_complete=True)
    lines=['# Figure 3：独立外部模块、容量、输出任务与上下文比较','',
           '本节只汇总已完成逐行核查的 75 个外部评估任务，共 85 个模型/词表与队列组合、170 份校准前后结果。'
           '复用已保存的全量训练模型，不重新训练；当前 210 类 MAOMAO 参考的完整外部指标见主报告校准前后大表。','',
           '## 评估口径','',
           '- 每个数据集固定 seed42，按患者 90% 校准、10% 封存测试。NTUH/ASAC/UQ 按源记录代理 ID 划分，未提供跨记录自然人链接；合并池与三个来源重叠，不是额外独立队列。',
           '- 所有原始有效目标行均推理。50/100/150 类任务仅在投影后仍有正目标的行计算指标，专用模型与投影 MAOMAO 使用相同词表、相同有效校准行及测试行；已独立核对目标行位置哈希。',
           ('- 恢复V5原始、仅温度、温度＋类别偏置候选；在90%内部按患者/记录代理80:20拟合/选择，以正事件集合softmax NLL选择，随后用全部90%有效行重新拟合。T范围0.15–6，偏置范围−4至4，均方正则0.002。最终测试标签不用于拟合或选择。' if PROTOCOL!=LEGACY_PROTOCOL else
            '- 温度只在完整 90% 校准行拟合，范围 0.15–6；仅当该校准集 Brier 改善超过 1e−8 时采用，否则保留 T=1。封存测试标签不用于拟合、模型选择或决定是否采用温度。'),
           '- 所有点估计使用完整有效测试行；95% CI 为 200 次固定种子目标行 bootstrap，大队列使用 30,000 行均匀样本和区间宽度缩放，属于近似 CI。该区间以已拟合的校准参数为条件，未通过患者聚类重采样完整传播同一患者目标行相关性及校准方法选择、温度和类别偏置拟合的不确定性。校准集改善不能保证独立测试集改善，结果不按期望排名修改。',
           '- 下表 Hit@1 和 Recall@5/10 按事件词表计算，不是 63-family Hit，也不生成 17 临床大类 Hit 表；五模型外部 63 小类 Hit@1/5/10/All@10 仍见主报告。','']
    if PROTOCOL!=LEGACY_PROTOCOL:
        lines += ['当前恢复的是 V5 的候选方法族与目标函数；优化器采用有界解析梯度 L-BFGS，最多 80 次迭代、120 次函数评估，因此不是旧 Adam 实现的逐步复现。每次拟合的停止原因和收敛状态保留在 `calibration_fit.json`，达到上限的拟合不报告为已收敛。正事件集合 NLL、归一化目标 Brier 和 top1 ECE 衡量不同目标，不能由其中一项改善推断其余项均改善。详见 [校准核查](MAOMAO_CALIBRATION_AUDIT.md)。','']
    for site,title in SITES.items():
        pairs=[p for p in proof['pairs'] if p['site']==site]
        assert len(pairs)==17
        reports={}
        lines += [f'## {title}','','### 目标行与模型来源','',
                  '|模型|输出类数|原始校准行|有效校准行|原始测试行|有效测试行|checkpoint epoch|参数数|采用温度|校准采用状态|',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
        for model in MODELS:
            pair=next(p for p in pairs if p['model']==model)
            parent='reference_projected' if model.startswith('reference_vocab_') else model
            folder=OUT/'external_ablations'/site/parent/model
            assert sha256(folder/'metrics_before.json')==pair['before_sha256']
            assert sha256(folder/'metrics_after.json')==pair['after_sha256']
            before,after=read(folder/'metrics_before.json'),read(folder/'metrics_after.json')
            fit=read(folder/'calibration_fit.json');reports[model]=(before,after)
            values=[MODELS[model],after['output_classes'],f"{after['calibration_rows_original']:,}",f"{after['calibration_rows']:,}",
                    f"{after['test_rows_original']:,}",f"{after['test_rows']:,}",after['checkpoint_epoch'],f"{after['parameter_count']:,}",
                    f"{after['temperature']:.6f}",(fit['family'] if PROTOCOL!=LEGACY_PROTOCOL else '通过校准集 Brier guard' if fit['fitted_temperature_accepted'] else '保留 T=1')]
            lines.append('|'+'|'.join(map(str,values))+'|')
        for title,metrics in BLOCKS:
            lines += ['',f'### {title}：校准前与后（95% CI）','',
                      '|模型|状态|'+'|'.join(label for _,label in metrics)+'|',
                      '|---|---|'+'|'.join('---:' for _ in metrics)+'|']
            for model in MODELS:
                for state,a in zip(('校准前','校准后'),reports[model]):
                    lines.append('|'+'|'.join([MODELS[model],state,*[cell(a,key) for key,_ in metrics]])+'|')
        lines += ['']
    lines += ['## 可复核文件','',
              '完整每事件统计、全部点估计/95% CI、校准方法与参数、优化记录及逐行证明位于结果包 `manuscript_figures/external_ablations/`。'
              '独立源行核查位于 `manuscript_figures/source_data/external_row_audits/`；汇总证明为 `manuscript_figures/external_ablation_row_verification.json`。'
              '绘图长表与来源 SHA256 位于 `manuscript_figures/source_data/`，不包含消融 checkpoint 或中间 logits。','']
    temp=DEST.with_suffix('.md.tmp');temp.write_text('\n'.join(lines));temp.replace(DEST)
    print(DEST,flush=True)
if __name__=='__main__':build()
