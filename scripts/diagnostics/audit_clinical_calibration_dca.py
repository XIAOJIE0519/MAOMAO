#!/usr/bin/env python3
"""Independent sealed-test audit, raw MAOMAO controls and patient-cluster CIs.

All fitted maps and the illustrative decision threshold are frozen from
training data. Test labels are used for evaluation and uncertainty only.
"""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.diagnostics.uniform_result_scope import sha256
DATA=ROOT/'outputs/maomao_manuscript_figures_20260929/source_data'

def main():
    dictionary=json.loads((DATA/'clinical_curve_dictionary.json').read_text())
    proof=json.loads((DATA/'clinical_risk_preparation.json').read_text())
    frame=pd.read_csv(ROOT/'outputs/maomao_plot_sources/clinical_score_predictions_no_identifiers.csv')
    risks=np.load(DATA/'clinical_risk_predictions.npz');points=pd.read_csv(DATA/'clinical_common_metrics.csv')
    calibration=[];dca=[];raw_cal=[];raw_dca=[];summary=[]
    threshold_grid=np.unique(np.r_[np.geomspace(1e-6,.1,200),np.linspace(.1,.99,200)])
    for ei,(endpoint,cohort) in enumerate(dictionary.items()):
        columns=['label_'+endpoint]+[s['score_column'] for s in cohort['series']]
        part=frame.loc[np.isfinite(frame[columns].to_numpy(float)).all(1)]
        y=part['label_'+endpoint].to_numpy(float);patients,cluster=np.unique(part.anonymous_patient_cluster,return_inverse=True)
        rng=np.random.default_rng(42+ei)
        weight=np.array([np.bincount(rng.integers(0,len(patients),len(patients)),minlength=len(patients))[cluster] for _ in range(200)],dtype=float)
        totals=weight.sum(1)
        for spec in cohort['series']:
            fit=next(f for f in proof['score_mappings'] if f['endpoint']==endpoint and f['model']==spec['name'])
            states=[('calibrated',risks[spec['prefix']+'__risk'])]
            if spec['name']=='MAOMAO':states.append(('raw',risks[spec['prefix']+'__raw_risk']))
            for state,p in states:
                assert np.array_equal(y,risks[spec['prefix']+'__target'])
                auc=roc_auc_score(y,p);ap=average_precision_score(y,p)
                if spec['name']=='MAOMAO':
                    for metric,val in [('auroc',auc),('average_precision',ap)]:
                        expected=points[(points.endpoint==endpoint)&(points.model=='MAOMAO')&(points.metric==metric)].iloc[0].value
                        assert abs(val-expected)<1e-12,'MAOMAO ranking changed'
                train_threshold=fit['train_events']/fit['train_rows']
                positive=p>=train_threshold;nb_terms=positive*(y-(1-y)*train_threshold/(1-train_threshold))
                loss=(p-y)**2;pp=np.clip(p,1e-15,1-1e-15);ll=-(y*np.log(pp)+(1-y)*np.log1p(-pp))
                row=dict(endpoint=endpoint,model=spec['name'],state=state,n=len(y),patients=len(patients),events=int(y.sum()),observed_fraction=float(y.mean()),mean_risk=float(p.mean()),min_risk=float(p.min()),max_risk=float(p.max()),brier=float(loss.mean()),log_loss=float(ll.mean()),auroc=float(auc),average_precision=float(ap),train_rows=fit['train_rows'],train_events=fit['train_events'],training_prevalence_threshold=train_threshold,net_benefit_at_training_prevalence=float(nb_terms.mean()),fraction_selected_at_training_prevalence=float(positive.mean()),bootstrap_repeats=200)
                for name,terms in [('mean_risk',p),('brier',loss),('log_loss',ll),('net_benefit_at_training_prevalence',nb_terms)]:
                    ci=np.quantile((weight@terms)/totals,[.025,.975]);row[name+'_ci_lower']=float(ci[0]);row[name+'_ci_upper']=float(ci[1])
                summary.append(row)
                cuts=np.unique(np.quantile(p,np.linspace(0,1,11)))[1:-1];bins=np.searchsorted(cuts,p,side='right')
                for b in np.unique(bins):
                    hit=bins==b;nn=int(hit.sum());bin_total=weight[:,hit].sum(1);valid=bin_total>0
                    fractions=(weight[:,hit]@y[hit])[valid]/bin_total[valid];ci=np.quantile(fractions,[.025,.975])
                    item=dict(endpoint=endpoint,model=spec['name'],state=state,bin=int(b),n=nn,events=int(y[hit].sum()),mean_risk=float(p[hit].mean()),observed_fraction=float(y[hit].mean()),ci_lower=float(ci[0]),ci_upper=float(ci[1]),valid_bootstrap_repeats=int(valid.sum()))
                    (raw_cal if state=='raw' else calibration).append(item)
                # Weighted cumulative counts reproduce all >= threshold rules;
                # ties remain together, including exact-threshold observations.
                order=np.argsort(p,kind='stable');start=np.searchsorted(p[order],threshold_grid,side='left')
                tp=np.r_[0,np.cumsum(y[order])];fp=np.r_[0,np.cumsum(1-y[order])]
                true=tp[-1]-tp[start];false=fp[-1]-fp[start]
                weighted_tp=np.c_[np.zeros(200),np.cumsum(weight[:,order]*y[order],axis=1)]
                weighted_fp=np.c_[np.zeros(200),np.cumsum(weight[:,order]*(1-y[order]),axis=1)]
                boot=((weighted_tp[:,-1,None]-weighted_tp[:,start])-(weighted_fp[:,-1,None]-weighted_fp[:,start])*threshold_grid/(1-threshold_grid))/totals[:,None]
                ci=np.quantile(boot,[.025,.975],axis=0)
                for j,t in enumerate(threshold_grid):
                    item=dict(endpoint=endpoint,model=spec['name'],state=state,threshold=float(t),net_benefit=float((true[j]-false[j]*t/(1-t))/len(y)),tp=int(true[j]),fp=int(false[j]),n=len(y),ci_lower=float(ci[0,j]),ci_upper=float(ci[1,j]),bootstrap_repeats=200)
                    (raw_dca if state=='raw' else dca).append(item)
            print('Independent probability/decision audit',endpoint,flush=True)
    for name,rows in [('clinical_calibration_curves',calibration),('clinical_dca_curves',dca),('clinical_raw_maomao_calibration_curves',raw_cal),('clinical_raw_maomao_dca_curves',raw_dca),('clinical_probability_metrics',summary)]:
        pd.DataFrame(rows).to_csv(DATA/(name+'.csv'),index=False)
    audit=dict(complete=True,protocol=proof['protocol'],test_patient_overlap=0,test_set_fitted=False,states=['MAOMAO raw','MAOMAO training-calibrated','training-mapped clinical scores'],training_comparator_cases_identical=True,maomao_auroc_ap_unchanged=True,series=len(summary),patient_cluster_bootstrap_repeats=200,ci_interpretation='Conditional on the frozen fitted maps; does not include map-estimation uncertainty',
               root_causes=['MAOMAO weighted focal-loss sigmoid was treated as calibrated absolute risk','Clinical scores received training-only logistic probability mapping while MAOMAO received none'],
               arithmetic='TP/FP recomputed on identical complete cases; NB=(TP-FP*threshold/(1-threshold))/n; >= includes threshold ties',
               threshold_view='Original 0 to min(0.50,max(0.005,3*test prevalence)) exploratory viewport retained; never tuned to maximize MAOMAO net benefit; full 399 thresholds saved',
               model_loss_sha256=sha256(ROOT/'maomao/models/event_maomao.py'),raw_source_preserved=True,
               clinical_test_protocol='Separate frozen INSPIRE 85:15 landmark study; no substitution of primary 90:10 next-event probabilities',
               limitation='MAOMAO calibration inputs are in-sample model-training predictions; final sealed test is independent, but no dedicated calibration-only cohort or nested uncertainty is claimed',
               references=['https://arxiv.org/abs/2011.09172','https://www.mskcc.org/sites/default/files/node/4509/documents/dca-tutorial-2015-2-26.pdf'],
               source_hashes={n:sha256(DATA/n) for n in ['clinical_risk_preparation.json','clinical_risk_predictions.npz','clinical_probability_metrics.csv','clinical_calibration_curves.csv','clinical_dca_curves.csv','clinical_raw_maomao_calibration_curves.csv','clinical_raw_maomao_dca_curves.csv']})
    (DATA/'clinical_calibration_dca_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    metrics=pd.DataFrame(summary);lines=['# 临床校准曲线与 DCA 问题核查（2026-09-30）','','## 确认的问题与修正','',
        '之前图的概率处理口径不一致：临床评分用训练患者拟合 logistic 风险映射，MAOMAO 却直接使用未校准的 trajectory sigmoid。MAOMAO 轨迹损失是加权 focal loss（正类 0.75、负类 0.25、γ=2）。该输出可用于排序，不能未经核查就解释为准确的发生概率。此前数值核查只确认画图与计算公式一致，没有发现这项方法问题。',
        '', '例如 1h 低血压：同一批 19,322 条封存测试记录实际 152 例（0.7867%），MAOMAO 原始输出均值 13.1404%，最低输出 2.8460%。在低于这一最低输出的阈值区间，全体病例都被选择；DCA 因而与 Treat all 重合。这不是 ROC/AP 计算矛盾，而是概率尺度存在偏差。',
        '', '本次对所有模型统一使用原 MAOMAO 训练患者中的同一结局完整病例交集拟合单调 logistic 风险映射；MAOMAO 输入为原始 sigmoid 的 logit，评分使用预定风险方向。模型权重不变，内部验证和封存测试不参与拟合。原始 MAOMAO 输出完整保留；图分别展示训练校准后的公平比较和 MAOMAO 原始状态。',
        '', '全部 10 个 MAOMAO 结局的 AUROC/AP 与原值一致（误差 <1e−12），不按最终测试表现选择校准方法、参数或阈值。校准分箱观察率、DCA 和下表 Brier 均使用 200 次患者聚类 bootstrap；区间条件于冻结映射，未涵盖映射拟合的不确定性。','',
        '加权 focal loss 不能保证输出等于真实后验概率，见[原始理论研究](https://arxiv.org/abs/2011.09172)；DCA 计算与参考策略按 [MSK 教程](https://www.mskcc.org/sites/default/files/node/4509/documents/dca-tutorial-2015-2-26.pdf) 核查。','',
        '## MAOMAO：相同封存测试病例上的原始与训练校准结果','',
        '|结局|测试记录/阳性|实际发生率|原始输出均值|训练校准后均值|原始 Brier (95% CI)|校准后 Brier (95% CI)|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for endpoint in dictionary:
        sub=metrics[(metrics.endpoint==endpoint)&(metrics.model=='MAOMAO')];raw=sub[sub.state=='raw'].iloc[0];cal=sub[sub.state=='calibrated'].iloc[0]
        def brier(r):return f'{r.brier:.5f} ({r.brier_ci_lower:.5f}–{r.brier_ci_upper:.5f})'
        lines.append(f'|{endpoint}|{int(raw.n):,}/{int(raw.events)}|{raw.observed_fraction:.3%}|{raw.mean_risk:.3%}|{cal.mean_risk:.3%}|{brier(raw)}|{brier(cal)}|')
    lines+=['','## 数据与图的使用限制','',
        '- 这是独立冻结临床评分 85:15 研究；它与主任务 90:10 的 next-event softmax 校准不同。没有把两种任务的概率混用。',
        '- 每个结局的映射拟合人数、记录数、阳性数和参数均写入 clinical_risk_preparation.json；全部有效训练病例用于拟合，没有 5 万条上限。MAOMAO 训练患者预测由冻结 checkpoint 重新提取，仍为 FP32。',
        '- MAOMAO 的校准来源是模型训练病例的预测，最终测试独立；不能据此声称拥有额外独立校准队列。评分映射不是已发表的标准概率公式。',
        '- DCA 阈值视窗仍按原规则设置，未为了使 MAOMAO 更好而更改；全 399 个阈值、TP/FP、净获益和点态 95% CI 均保留。DCA 为探索性，不能替代明确干预、临床阈值和前瞻性验证。',
        '- 输血 6h/24h 只有 3/4 个阳性，校准和 DCA 的不确定性非常大。原始或校准后不佳的结果均如实保留。',
        '- clinical_probability_metrics.csv 含全部评分与 MAOMAO 原始/校准状态的概率指标、95% CI、训练基准发生率处的净获益；原始和校准曲线表、患者匿名预测数组、映射参数、公式核查 JSON 均纳入结果包。私有患者标识缓存不纳入 ZIP。','']
    (ROOT/'docs/MAOMAO_CLINICAL_CALIBRATION_DCA_AUDIT.md').write_text('\n'.join(lines))
    print(metrics[metrics.model=='MAOMAO'][['endpoint','state','mean_risk','brier','net_benefit_at_training_prevalence']].to_string(index=False),flush=True)
if __name__=='__main__':main()
