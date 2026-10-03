#!/usr/bin/env python3
"""Merge requested follow-up baseline artifacts into the final results report."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/final_experiment_results_20260923'
REPORT=ROOT/'docs/MAOMAO_V5_FINAL_RESULTS.md'
DEEP=OUT/'requested_50k_models'
MARK='## 新增：50k 行轻量神经基线与全量快速机器学习'

def fmt(value, ci):
    if value is None: return '—'
    return f"{value:.4f}"+(f" ({ci[0]:.4f}–{ci[1]:.4f})" if ci else '')

def main():
    # This entry point used to append small-sample and retired model tables.
    # Preserve its CLI while regenerating the current all-row report only.
    from build_uniform_results_report import main as build_current_report
    build_current_report()
    return
    names=['lstm','cnn','rnn','gan','autoencoder','ann','maomao_full_checkpoint_shared_rows']
    rows=[]
    labels={'lstm':'LSTM','cnn':'CNN','rnn':'RNN（tanh）','gan':'GAN（条件对抗 + 监督头）',
            'autoencoder':'AutoEncoder（重建辅助头）','ann':'ANN','maomao_full_checkpoint_shared_rows':'MAOMAO（全量训练权重）'}
    metrics=['micro_auprc','macro_auprc','micro_auroc','macro_auroc','mrr','brier','ece','hit_at_1','recall_at_5','recall_at_10']
    for name in names:
        p=(OUT/'model_metrics'/'maomao_shared_20k.json') if name.startswith('maomao_') else DEEP/f'{name}.json'
        if not p.exists(): continue
        d=json.loads(p.read_text())
        if d.get('status') not in (None,'completed'): continue
        vals='|'.join(fmt(d.get(m),d.get(m+'_95ci')) for m in metrics)
        rows.append(f"|{labels[name]}|{d.get('train_rows',d.get('train_rows_full','—'))}|{d.get('evaluation_rows',d.get('validation_rows','—'))}|{vals}|")
    deep_manifest=json.loads((DEEP/'manifest.json').read_text()) if (DEEP/'manifest.json').exists() else {}
    deep='\n'.join([
      MARK,'',
      '这组补充实验是轻量筛查：从同一患者级 90:10 拆分中使用 50,000 个训练目标行和共同的 20,000 个验证目标行；每个神经基线仅训练 1 epoch，置信区间为 200 次目标行 bootstrap。LSTM、CNN、RNN 和 ANN 使用监督 next-event 多标签头；AutoEncoder 同时优化输入重建；GAN 为条件生成器加判别器，并保留监督目标头。这两个改造版分别按其实际训练目标解释，不等同于标准无监督 AE 或纯生成 GAN。此表不代表充分训练后的深度模型性能，不能和上方全量训练的 MAOMAO/GRU 主表直接比较。',
      '',
      '|模型|训练目标行|验证目标行|micro-AUPRC|macro-AUPRC|micro-AUROC|macro-AUROC|MRR|Brier|ECE|Hit@1|Recall@5|Recall@10|',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
      *rows,'',
      '本表的 RNN 是 48 维普通 tanh RNN，输入为 6 个归一化数值通道（token/type/value/presence/time/gap），不是 GRU。此前表现较好的正式 GRU 是另一模型：256 维 GRU，并为 token、事件类型、数值、时间及静态协变量学习嵌入；它在全量训练目标行上训练，按患者级完整验证集选择第 8 epoch checkpoint，并在全部 1,563,972 个验证目标行评估（micro-AUPRC 0.2620、micro-AUROC 0.9509，见上方主表）。因此，这里的 50k、单 epoch RNN 低分不能说明正式 GRU 或深度学习表现差，也不是一次同模型复现。最初误用 GRU 单元的 50k 单轮尝试（micro-AUPRC 0.0179）仅保留作审计，不纳入 RNN 行。MAOMAO 行则复用了完整训练集的 MAOMAO checkpoint，只在共同 20,000 个验证目标位置评分；其训练量远大于 50,000 行，因此仅作为 full-data reference，不能视为同训练规模的胜负比较。完整 MAOMAO 患者级 10% 验证集结果仍保留在上方主表。',
      '',
      '新增逐模型指标、95% CI、checkpoint 和 manifest 保存在 `outputs/final_experiment_results_20260923/requested_50k_models/`；MAOMAO shared-row 评分文件在 `model_metrics/maomao_shared_20k.json`。'])

    svm_path=OUT/'classical_ml_fullscale/linear_svm/metrics.json'
    status_path=OUT/'classical_ml_fullscale/linear_svm/status.json'
    if svm_path.exists():
        d=json.loads(svm_path.read_text())
        svm_status=json.loads(status_path.read_text()) if status_path.exists() else {}
        svm_elapsed=svm_status.get('elapsed_seconds')
        svm_runtime=(f"实际训练与评分耗时 {int(svm_elapsed)} 秒；" if svm_elapsed is not None else "")
        vals='|'.join(fmt(d.get(m),d.get(m+'_95ci')) for m in metrics)
        deep += '\n\n### 全量训练：线性 SVM\n\n'
        deep += f"使用多标签 hinge-loss SGD 对全部 {d.get('train_rows_full',0):,} 个有效训练目标行完整遍历 1 epoch；{svm_runtime}在患者级 10% 验证集中的共同 20,000 行上评估，200 次 bootstrap。它是已完成的快速 full-scale-fit 结果，验证集只抽取固定子集，不能与下方全量验证集主表直接混排。\n\n|模型|训练行|验证行|micro-AUPRC|macro-AUPRC|micro-AUROC|macro-AUROC|MRR|Brier|ECE|Hit@1|Recall@5|Recall@10|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n|线性 SVM（hinge SGD）|{d.get('train_rows_full')}|{d.get('validation_rows')}|{vals}|\n\n模型文件与指标位于 `classical_ml_fullscale/linear_svm/`。分数为未校准 margin，软最大值只用于保留统一指标定义。"
    elif status_path.exists():
        d=json.loads(status_path.read_text())
        deep += f"\n\n### 全量训练机器学习尝试\n\n线性 SVM 状态：`{d.get('status')}`；运行 {d.get('elapsed_seconds')} 秒；未生成可用指标。细节见 `classical_ml_fullscale/linear_svm/status.json`。"

    # Current full-scale exploratory queue. Only present a score when its
    # metrics JSON exists and declares successful completion.
    candidate_names=['random_forest','decision_tree','hist_gradient_boosting','lightgbm',
                     'catboost','xgboost','adaboost','gaussian_nb']
    queue_path=OUT/'classical_ml_fullscale/queue_status.json'
    queue_data=json.loads(queue_path.read_text()) if queue_path.exists() else {}
    queue_models={m.get('model'):m for m in queue_data.get('models',[])}
    candidate_labels={'random_forest':'Random Forest','decision_tree':'Decision Tree',
                      'hist_gradient_boosting':'HistGradientBoosting','lightgbm':'LightGBM',
                      'catboost':'CatBoost','xgboost':'XGBoost','adaboost':'AdaBoost',
                      'gaussian_nb':'Gaussian Naive Bayes'}
    candidate_rows=[]
    for name in candidate_names:
        folder=OUT/'classical_ml_fullscale'/name
        if name=='xgboost':
            trial=folder/'minimal_one_round_20260927'
            trial_metrics=trial/'metrics.json'
            trial_status=trial/'status.json'
            if trial_metrics.exists() and trial_status.exists():
                td=json.loads(trial_metrics.read_text())
                ts=json.loads(trial_status.read_text())
                if td.get('epochs_or_rounds')==1 and ts.get('status')=='completed':
                    def xgb_cell(metric): return fmt(td.get(metric),td.get(metric+'_95ci'))
                    elapsed=ts.get('elapsed_seconds','—')
                    candidate_rows.append(f"|XGBoost（极简单轮诊断）|1 轮完成；耗时 {elapsed}s；不替代 30 轮主基线|{td.get('train_rows_full','—')}|{xgb_cell('micro_auprc')}|{xgb_cell('micro_auroc')}|{xgb_cell('mrr')}|`outputs/final_experiment_results_20260923/classical_ml_fullscale/xgboost/minimal_one_round_20260927/metrics.json`|")
                    continue
        metrics_file=folder/'metrics.json'
        model_status=queue_models.get(name,{})
        status=model_status.get('status')
        if not status:
            state_file=folder/'status.json'
            status=json.loads(state_file.read_text()).get('status') if state_file.exists() else (
                'not_run_by_user_scope_change' if name!='lightgbm' else 'pending')
        if metrics_file.exists():
            d=json.loads(metrics_file.read_text())
            if d.get('status')=='completed':
                def cell(metric): return fmt(d.get(metric),d.get(metric+'_95ci'))
                candidate_rows.append(f"|{candidate_labels[name]}|已完成|{d.get('train_rows_full','—')}|{cell('micro_auprc')}|{cell('micro_auroc')}|{cell('mrr')}|`outputs/final_experiment_results_20260923/classical_ml_fullscale/{name}/metrics.json`|")
                continue
        statuses={'timeout_20min_no_result':'20 分钟超时，未产出结果',
                  'timeout_no_result':'此前 10 分钟尝试超时；本轮待重试',
                  'failed_oom_no_result':'OOM 终止，未产出结果',
                  'running':'运行中', 'pending':'排队中',
                  'not_run_by_user_scope_change':'本轮按用户范围未运行',
                  'stopped_by_user_scope_change':'本轮按用户范围停止',
                  'failed_no_result':'失败，未产出结果'}
        if status=='already_terminal':
            status=model_status.get('terminal_status',status)
        if status=='failed_no_result':
            state_file=folder/'status.json'
            try:
                attempt=int(json.loads(state_file.read_text()).get('attempt',1))
            except (ValueError,TypeError):
                attempt=1
            log_path=folder/f'run_20min_attempt_{attempt}.log'
            if not log_path.exists(): log_path=folder/'run_20min.log'
            if log_path.exists():
                log_text=log_path.read_text(errors='replace')
                import re
                match=re.search(r'Unable to allocate ([^\n]+)',log_text)
                if match:
                    statuses[status] += f"（内存不足：需分配 {match.group(1).strip()}）"
        if status=='failed_no_result' and '内存不足：' not in statuses.get(status,''):
            state_file=folder/'status.json'
            try:
                attempt=int(json.loads(state_file.read_text()).get('attempt',1))
            except (ValueError,TypeError):
                attempt=1
            log_path=folder/f'run_20min_attempt_{attempt}.log'
            if not log_path.exists(): log_path=folder/'run_20min.log'
            if log_path.exists():
                import re
                match=re.search(r'Unable to allocate ([^\n]+)',log_path.read_text(errors='replace'))
                if match:
                    statuses[status] += f"（内存不足：需分配 {match.group(1).strip()}）"
        elapsed=model_status.get('elapsed_seconds')
        if elapsed is None and status=='running' and model_status.get('started_utc'):
            try:
                elapsed=max(0,int((datetime.now(timezone.utc)-datetime.fromisoformat(
                    model_status['started_utc'])).total_seconds()))
            except (ValueError, TypeError):
                elapsed=None
        duration=f"{elapsed}s" if elapsed is not None else '—'
        status_label=statuses.get(status,status)
        previous=float(model_status.get('previous_attempt_seconds',0) or 0)
        if status=='running' and not previous:
            latest_status=folder/'status.json'
            if latest_status.exists():
                try:
                    previous=float(json.loads(latest_status.read_text()).get('previous_attempt_seconds',0) or 0)
                except (ValueError,TypeError):
                    previous=0
        if status=='running' and previous:
            cumulative=previous+(float(elapsed or 0))
            status_label += f"（跨尝试累计约 {int(cumulative)}s / 1200s）"
        history_path=folder/'attempt_history.json'
        if history_path.exists():
            try:
                history=json.loads(history_path.read_text())
                history_rows=history.get('attempts',[]) if isinstance(history,dict) else history
                oom_count=sum(1 for attempt in history_rows if attempt.get('status')=='failed_oom_no_result')
                if oom_count:
                    status_label += f"（此前 {oom_count} 次尝试被 OOM 终止）"
            except (ValueError,TypeError):
                pass
        if status=='failed_oom_no_result':
            status_data=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
            detail=status_data.get('detail','')
            if detail: status_label += f"（{detail}）"
        candidate_rows.append(f"|{candidate_labels[name]}|{status_label}|14,128,539|—|—|—|状态：`{duration}`；`outputs/final_experiment_results_20260923/classical_ml_fullscale/{name}/status.json`|")
    if candidate_rows:
        deep += '\n\n### 全量传统机器学习候选与专项尝试\n\n'
        deep += '候选模型按完整 14,128,539 行训练目标拟合，在固定患者级验证行上测评。Random Forest、Decision Tree、HistGradientBoosting 和 LightGBM 的候选队列受 20 分钟限制；XGBoost 极简单轮是后续单独诊断，耗时单独如实列出。只将写出有效 `metrics.json` 的运行标为完成；超时项不填造指标。\n\n'
        deep += '|算法|状态|训练行|micro-AUPRC (95% CI)|micro-AUROC (95% CI)|MRR (95% CI)|结果或状态文件|\n|---|---|---:|---:|---:|---:|---|\n'
        deep += '\n'.join(candidate_rows)
        deep += '\n\n候选明细、完整指标及运行日志位于 `outputs/final_experiment_results_20260923/classical_ml_fullscale/`。'

    common_path=OUT/'baseline_metrics/common_full_validation/common_full_validation_metrics.json'
    if common_path.exists():
        common=json.loads(common_path.read_text())
        model_labels={'univariate':'单变量','logistic_regression':'逻辑回归',
                      'linear_svm_saved':'linearSVM（保存模型）',
                      'linear_svm_refined':'linearSVM（额外训练一轮）',
                      'ann_saved_50k':'ANN（保存的 5 万行模型）',
                      'ann_fullscale_refined':'ANN（全量训练优化）','maomao':'MAOMAO'}
        common_rows=[]
        for key,label in model_labels.items():
            d=common.get('models',{}).get(key)
            if not d: continue
            cells='|'.join(fmt(d.get(m),d.get(m+'_95ci')) for m in
                           ('micro_auprc','macro_auprc','micro_auroc','macro_auroc',
                            'mrr','brier','ece','hit_at_1','recall_at_5','recall_at_10'))
            train_n=int(d.get('train_rows',common.get('train_target_rows',0)) or 0)
            valid_n=int(d.get('validation_rows',common.get('validation_target_rows',0)) or 0)
            common_rows.append(f"|{label}|{train_n:,}|{valid_n:,}|{cells}|")
        deep += '\n\n## 五模型同一全量 90:10 验证集比较\n\n'
        deep += f"所有模型在相同患者隔离 10% 验证集的全部 {common.get('validation_target_rows',0):,} 个有效目标行上评分；训练部分为 {common.get('train_target_rows',0):,} 行。全量点估计，指标均报告 95% CI。SVM 与 ANN 各保留原保存模型和仅基于训练集额外训练一轮的版本；没有用最终验证集调参。\n\n"
        deep += '|模型|训练目标行|验证目标行|micro-AUPRC|macro-AUPRC|micro-AUROC|macro-AUROC|MRR|Brier|ECE|Hit@1|Recall@5|Recall@10|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
        deep += '\n'.join(common_rows)
        deep += '\n\n完整指标与版本化模型位于 `outputs/final_experiment_results_20260923/baseline_metrics/common_full_validation/`。'
    else:
        deep += '\n\n## 五模型同一全量 90:10 验证集比较\n\n单变量、逻辑回归、linearSVM、ANN 和 MAOMAO 的共同全验证集对照正在整理；完成后会用全部患者隔离 10% 验证目标行报告 AUROC 和指标 95% CI。'

    scale_path=OUT/'model_metrics/maomao_dual_timescale_mae_internal.json'
    if scale_path.exists():
        scales=json.loads(scale_path.read_text()).get('time_mae_by_scale',{})
        deep += '\n\n## MAOMAO Dual-timescale head：分尺度 time MAE\n\n'
        deep += '全量内部 patient-level 10% 验证集，按观测到的下一事件间隔分层；预测量为 dual-timescale hazard head 解码的 event-conditioned expected wait。区间为每个尺度内 200 次 event-row bootstrap 95% CI。总体混合尺度 MAE 不列入报告。\n\n'
        deep += '|时间尺度|有效事件目标行|MAE（小时，95% CI）|\n|---|---:|---:|\n'
        for key in ('fine_0_to_2h','long_2h_to_tail_start','tail_from_tail_start'):
            d=scales.get(key,{})
            value=d.get('mae_hours');ci=d.get('mae_hours_95ci')
            cell='—' if value is None else fmt(value,ci)
            deep += f"|{d.get('label',key)}|{d.get('event_target_rows','—')}|{cell}|\n"
        deep += '\n分尺度 MAE JSON：`outputs/final_experiment_results_20260923/model_metrics/maomao_dual_timescale_mae_internal.json`。'
    else:
        deep += '\n\n## MAOMAO Dual-timescale head：分尺度 time MAE\n\n分尺度全量验证评估正在运行；完成后只追加 fine、long、tail 三档 MAE 及区间，不报告混合尺度总体 MAE。'
    
    report=REPORT.read_text()
    report=report.split(MARK)[0].rstrip()+"\n\n"+deep.rstrip()+"\n"
    REPORT.write_text(report)
    print(json.dumps({'report':str(REPORT),'deep_models_included':[r.split('|')[1] for r in rows],
                      'svm_metrics':svm_path.exists()},ensure_ascii=False))
if __name__=='__main__':main()
