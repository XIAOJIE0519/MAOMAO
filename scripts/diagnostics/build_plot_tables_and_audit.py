#!/usr/bin/env python3
"""Tidy full results, training curves and source-backed calibration audit."""
import sys
import json
import csv
from collections import Counter
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve,precision_recall_curve
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.diagnostics.uniform_result_scope import current_xgboost_dir,sha256
from scripts.diagnostics.build_fullscale_external_rows import SOURCES
from scripts.diagnostics.scale_ablation_scope import NAMES
from maomao.evaluation.softmax_calibration import PROTOCOL, LEGACY_PROTOCOL, revision_directory
from maomao.evaluation.rank_statistics import RANK_VERSION

BASE=ROOT/"outputs/final_experiment_results_20260923"
OUT=ROOT/"outputs/maomao_plot_sources"
EXT=ROOT/"outputs/external_validation_final_maomao_uniform"
REV=revision_directory()
OLD=ROOT/"outputs/calibration_audit_original_20260929"
METRICS=("micro_auprc","macro_auprc","micro_auroc","macro_auroc","mrr","brier","ece","hit_at_1","recall_at_5","recall_at_10")
MODELS=("univariate","logistic_regression","xgboost","ann","maomao")
MODULES=("no_block_causal","no_relative_time","no_family_head","no_clock_phase_summary","no_measurement_intensity","no_masked_event_value","no_event_conditioned_time")


def read(path):return json.loads(path.read_text())


def main():
    OUT.mkdir(exist_ok=True)
    tables=[];event=[];frequency=[];time=[];fits=[]
    meta=read(ROOT/"data/perioperative_event_sequences_v5_richctx_static7/event_sequence_meta.json")
    family={name:meta["outcome_to_family"][i] for i,name in enumerate(meta["outcome_vocabulary"])}
    records=[]
    internal_targets=np.load(BASE/"classical_full_scale/validation_y.npy",mmap_mode="r")
    internal={"univariate":BASE/"baseline_metrics/univariate_fullscale_internal.json",
              "logistic_regression":BASE/"baseline_metrics/logistic_fullscale_internal.json",
              "xgboost":current_xgboost_dir()/"metrics.json",
              "ann":BASE/"baseline_metrics/common_full_validation/ann_fullscale_refined_metrics.json",
              "maomao":BASE/"model_metrics/maomao_internal.json"}
    for model,path in internal.items():records.append(("internal","inspire",model,"raw",path))
    for site in SOURCES:
        for model in MODELS:records.append(("external",site,model,"after",EXT/site/model/"metrics.json"))
        records.append(("external",site,"maomao","before",EXT/site/"maomao/metrics_uncalibrated.json"))
    for model in MODULES:records.append(("module_ablations","inspire",model,"raw",BASE/f"module_metrics/{model}.json"))
    scale=ROOT/"outputs/scale_ablations_richctx_20260928"
    for path in sorted((scale/"metrics").glob("*.json")):
        if not path.stem.endswith("_time_scales"):records.append(("scale_ablations","inspire",path.stem,"raw",path))
    for group,site,model,state,path in records:
        m=read(path);info=dict(group=group,site=site,model=model,calibration_state=state,
                  rows=m.get("event_targets"),train_rows=m.get("train_rows_internal",m.get("train_rows",m.get("train_rows_full",m.get("train_event_rows_eligible",14128539)))),
                  calibration_rows=m.get("calibration_rows"),temperature=m.get("temperature",1.),
                  source_metric_file=str(path.relative_to(ROOT)),source_sha256=sha256(path),
                  ci_method=m.get("ci_method"),ci_rows=m.get("ci_rows"),protocol=m.get("calibration_protocol","frozen internal uncalibrated"))
        relative=f"{group}/{site}/{model}/{state}" if group=="external" else f"{group}/{model}"
        provenance=OUT/relative/"provenance.json"
        if provenance.exists():
            p=read(provenance)
            p.update(group=group,site=site,model=model,calibration_state=state,reliability_bins=20,reported_ece_bins=15,
                     source_metric_file=str(path.relative_to(ROOT)),source_metric_sha256=sha256(path),
                     reported_brier=m["brier"],rank_metric_definition=m.get("rank_metric_definition"),
                     curve_point_metric_definition="threshold-grouped average precision and midrank AUROC; histogram curve areas are secondary approximations")
            if m.get("checkpoint_epoch") is not None:p["checkpoint_epoch"]=m["checkpoint_epoch"]
            if group=="external":
                row_index=np.arange(m["test_rows"],dtype=np.int64)
                alignment_scope=f"original sealed {site} test target coordinate order"
            else:
                indices=m.get("output_indices",list(range(210)))
                row_index=np.flatnonzero(np.asarray(internal_targets[:,indices],dtype=np.uint8).any(1))
                alignment_scope="original INSPIRE common validation target coordinate order"
            if len(row_index)!=p["rows"]:raise RuntimeError("Figure alignment row mismatch")
            np.savez_compressed(provenance.parent/"target_row_alignment.npz",target_row_index=row_index)
            p.update(target_row_alignment_scope=alignment_scope,alignment_indices_are_clinical_identifiers=False)
            provenance.write_text(json.dumps(p,ensure_ascii=False,indent=2)+"\n")
        for metric in METRICS:
            ci=m.get(metric+"_95ci") or [None,None]
            tables.append(dict(info,metric=metric,value=m.get(metric),ci_lower=ci[0],ci_upper=ci[1]))
        for name,x in m.get("per_event",{}).items():event.append(dict(info,event=name,family_id=family.get(name),**x))
        for name,x in m.get("frequency_strata",{}).items():frequency.append(dict(info,stratum=name,**{k:v for k,v in x.items() if k!="support_range"},support_lower=(x.get("support_range") or [None,None])[0],support_upper=(x.get("support_range") or [None,None])[1]))
    for path in [BASE/"model_metrics/maomao_dual_timescale_mae_internal.json",*sorted((scale/"metrics").glob("*_time_scales.json"))]:
        m=read(path)
        for name,x in m["time_mae_by_scale"].items():
            ci=x.get("mae_hours_95ci") or [None,None]
            time.append(dict(model=m.get("model",path.stem),scale=name,label=x["label"],rows=x["event_target_rows"],mae_hours=x["mae_hours"],ci_lower=ci[0],ci_upper=ci[1],source_file=str(path.relative_to(ROOT))))
    for site in SOURCES:
        for model in MODELS:
            p=EXT/site/model/"calibration_fit.json"
            if not p.exists():continue
            f=read(p)
            fits.append(dict(site=site,model=model,calibration_rows=f["calibration_rows"],fitted_temperature=f["fitted_temperature"],temperature=f["temperature"],accepted=f["fitted_temperature_accepted"],family=f.get('family','temperature_only'),bias_max_absolute=float(np.max(np.abs(f.get('bias',[0.])))),raw_calibration_brier=f["raw_calibration"]["brier"],fitted_calibration_brier=f["fitted_calibration"]["brier"],protocol=f["protocol"]))
    for filename,data in (("metrics_long.csv",tables),("per_event_metrics.csv",event),("frequency_strata.csv",frequency),("time_scales.csv",time),("calibration_fits.csv",fits)):
        pd.DataFrame(data).to_csv(OUT/filename,index=False)
    histories=[]
    for name,path in [("maomao",BASE/"full_maomao_reference"),*[(n,BASE/"module_ablation_runs"/n) for n in MODULES],*[(n,scale/"runs"/n) for n in NAMES]]:
        for line in (path/"validation_history.jsonl").read_text().splitlines():
            r=json.loads(line)
            histories.append(dict(model=name,**{k:v for k,v in r.items() if isinstance(v,(int,float,str,bool)) and k!="time_mae_hours"}))
    pd.DataFrame(histories).to_csv(OUT/"validation_learning_curves.csv",index=False)
    # Supplement clinical figures with deidentified score/outcome columns;
    # keep the distinct 85:15 protocol explicit.
    cp=ROOT/"outputs/classic_score_comparison/independent_15pct_test/expanded_results/expanded_test_episode_predictions.csv"
    clinical_matrix=cp.parent.parent/"clinical_score_comparison_full_matrix.csv"
    pd.read_csv(clinical_matrix).to_csv(OUT/"clinical_score_comparison_full_matrix.csv",index=False)
    clinical=pd.read_csv(cp)
    columns=[c for c in clinical if not any(term in c.lower() for term in ("id","idx","date","time","name"))]
    # Explicitly remove grouping IDs even where they do not contain 'id'.
    columns=[c for c in columns if c not in ("subject","patient","episode","admission","record")]
    clinical_export=clinical[columns].copy()
    unique=clinical.subject_id.astype(str).drop_duplicates().tolist()
    anonymous=np.random.default_rng(42).permutation(len(unique))
    cluster={name:int(index) for name,index in zip(unique,anonymous)}
    clinical_export["anonymous_patient_cluster"]=clinical.subject_id.astype(str).map(cluster)
    clinical_export.to_csv(OUT/"clinical_score_predictions_no_identifiers.csv",index=False)
    (OUT/"clinical_prediction_dictionary.json").write_text(json.dumps(dict(source_sha256=sha256(cp),rows=len(clinical),columns=clinical_export.columns.tolist(),excluded_columns=[c for c in clinical if c not in columns],anonymous_cluster_definition="seed42 random integers replace real patient IDs; no reverse mapping exported",protocol="independent INSPIRE patient-disjoint 85:15 sealed test; separate frozen MAOMAO"),indent=2)+"\n")
    clinical_summary=read(cp.parent/"expanded_comparison_summary.json")
    curve_dir=OUT/"clinical_curves";curve_dir.mkdir(exist_ok=True)
    pair_index=[]
    for endpoint,comparisons in clinical_summary["comparisons_by_endpoint"].items():
        for i,comparison in enumerate(comparisons):
            column=comparison["score_column"];label="label_"+endpoint;maomao="maomao_"+endpoint
            part=clinical.loc[clinical[label].notna()&clinical[maomao].notna()&clinical[column].notna()]
            if len(part)!=comparison["n_episodes"] or part.subject_id.nunique()!=comparison["n_patients"]:
                raise RuntimeError("Clinical pair rows disagree")
            true=part[label].to_numpy(dtype=np.int8)
            arrays={}
            for name,col,direction in (("comparator",column,-1 if comparison["comparator"]=="SAS" else 1),("maomao",maomao,1)):
                pred=part[col].to_numpy(dtype=float)*direction
                fpr,tpr,threshold=roc_curve(true,pred,drop_intermediate=False)
                precision,recall,pr_threshold=precision_recall_curve(true,pred)
                arrays.update({name+"_fpr":fpr,name+"_tpr":tpr,name+"_roc_thresholds":threshold,
                               name+"_precision":precision,name+"_recall":recall,name+"_pr_thresholds":pr_threshold})
            filename=f"{endpoint}_{i}.npz"
            np.savez_compressed(curve_dir/filename,**arrays)
            pair_index.append(dict(endpoint=endpoint,comparator=comparison["comparator"],score_column=column,
                              comparator_risk_direction=-1 if comparison["comparator"]=="SAS" else 1,
                              rows=len(part),patients=comparison["n_patients"],events=int(true.sum()),file=filename))
    (curve_dir/"pair_index.json").write_text(json.dumps(pair_index,ensure_ascii=False,indent=2)+"\n")
    if PROTOCOL!=LEGACY_PROTOCOL:
        write_v5_audit()
        return
    audit=["# MAOMAO 外部校准核查与修正", "", "## 已确认的原因", "",
           "旧流水线用sigmoid多标签BCE拟合温度，却用softmax下一事件概率计算最终Brier、ECE、AUROC/AUPRC。概率链接和拟合目标不一致；sigmoid目标还受每行logit共同平移影响，而softmax不会。因此原温度可能不合理地放大概率集中度。", "",
           "这不是重新训练或把模型排名改高。MAOMAO事件训练损失是−log(sum of positive-set softmax probabilities)，标量温度只改变事件概率集中度，不补充外部数据库缺失输入，也不校准dual-timescale时间头。正温度保持每行类排序不变，Hit@1/MRR/Recall应不变；softmax跨行归一化变化可以使按类别/微平均AUROC、AUPRC略变，不能声称所有AUROC必然恒定。", "",
           "## 修正方法（测试前固定）", "",
           f"协议版本：`{PROTOCOL}`。冻结模型权重、来源词表、原90:10患者/记录代理划分、全部有效目标行与封存10%测试坐标。", "",
           "1. 全部90%校准行使用float32模型logits；逐行减去最大logit，确保平移不变。",
           "2. 用softmax cross-entropy对归一化正事件集合q=y/sum(y)拟合一个正温度，固定范围0.15–6。此处q是既有Brier目标定义，不声称多正事件有真实互斥概率。",
           "3. 仅比较90%校准集Brier：拟合温度相对T=1改善>1e−8才采用，否则保留T=1。没有在封存测试集选温度、模型、区间或阈值。",
           "这是对先前已报告测试集的实现修正复算，不是新增未见过的独立确认性验证；测试结果用于诊断和汇报，没有反馈到温度拟合或选择规则。",
           "4. 拟合参数与规则写入calibration_fit.json后才评估原10%测试。五个冻结模型都按同一修正规则重新校准，保持比较口径一致。",
           "5. MAOMAO每来源保留校准前/后全测试指标、原权重SHA、相同目标验证与配对引用SHA；最终MD直接在外部五模型大表中展示两个状态。", "",
           "旧结果保留在项目独立审计归档calibration_audit_original_20260929，最终结果包采用修正后的当前状态，避免旧错误温度混入当前表。", "",
           "## 校准前、旧校准与修正校准的实际测试结果", "",
           "Brier与ECE越低越好；此表是核查证据，不能用来选择测试集表现更好的温度。", "",
           "|来源|测试行|旧T|新采用T|原始Brier|旧校准Brier|修正Brier|原始ECE|旧校准ECE|修正ECE|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    comparison=[]
    for site in SOURCES:
        old=read(OLD/site/"maomao/metrics.json");raw=read(REV/site/"maomao/metrics_uncalibrated.json")
        newpath=REV/site/"maomao/metrics.json"
        new=read(newpath) if newpath.exists() else None
        fmt=lambda x:f"{x:.6f}" if isinstance(x,(int,float)) else "计算中"
        audit.append("|"+"|".join([site,f'{old["test_rows"]:,}',fmt(old["temperature"]),fmt(new["temperature"] if new else None),fmt(raw["brier"]),fmt(old["brier"]),fmt(new["brier"] if new else None),fmt(raw["ece"]),fmt(old["ece"]),fmt(new["ece"] if new else None)])+"|")
        if new:
            for metric in METRICS:comparison.append(dict(site=site,metric=metric,raw=raw.get(metric),old_calibrated=old.get(metric),corrected_calibrated=new.get(metric),test_rows=new["test_rows"],raw_definition=RANK_VERSION,old_calibrated_definition=old.get("rank_metric_definition","legacy AP/AUROC without threshold tie groups"),corrected_calibrated_definition=RANK_VERSION))
    pd.DataFrame(comparison).to_csv(OUT/"calibration_audit_comparison.csv",index=False)
    audit += ["", "审计CSV的raw与corrected_calibrated使用本轮标准并列值定义；old_calibrated保留旧定义用于追溯。Brier/ECE定义没有改变，可直接检查温度修正效果；旧/新AP或AUROC差异还包含统计实现修正，不能全部归因于温度。", ""]
    audit += ["", "### 置信度偏低的具体核对", ""]
    for site in SOURCES:
        pre=OUT/f"external/{site}/maomao/before/reliability.csv";post=OUT/f"external/{site}/maomao/after/reliability.csv"
        if pre.exists() and post.exists() and (REV/site/"maomao/metrics.json").exists():
            b=read(REV/site/"maomao/metrics_uncalibrated.json");a=read(REV/site/"maomao/metrics.json")
            if a["ece"]>b["ece"]:
                r=pd.read_csv(pre);s=pd.read_csv(post)
                audit += [f'{site}：测试ECE由{b["ece"]:.6f}升至{a["ece"]:.6f}；top1正确率保持{r.sum_any_positive_correct.sum()/r.rows.sum():.6f}不变，平均置信度从{r.sum_confidence.sum()/r.rows.sum():.6f}降至{s.sum_confidence.sum()/s.rows.sum():.6f}。温度软化可能从过度自信跨过到置信度不足；CE/Brier选择准则没有保证该any-positive ECE改善，保留实际结果。', ""]
    audit += ["", "## 解释限制与附带核查", "",
              "- Brier以归一化正事件集合q评分；当前ECE以max-softmax confidence对any-positive top1正确性评分（15个等宽桶）。多正事件时二者的目标口径不同，不能保证一次温度校准同时改善所有指标；校准集改善也不保证有限测试集改善。", "",
              "例如同时有两个正事件时，q=(0.5,0.5)。即使预测正好等于q、Brier为0，top1仍必定属于正集合；此时最大置信度0.5与any-positive正确性1相差0.5。这是评分目标的结构性差别，不应把这个ECE直接当作归一化事件分布的完全校准误差。当前结果保留该既有ECE定义并标明口径，没有根据测试集另换温度以强迫它降低。", "",
              "- 小型麻醉测试集仅2–16条记录代理、23–118个重叠窗口目标行，数据很少。不能把目标行数当作独立患者数。当前95%CI沿用200次目标行bootstrap，大队列用固定100,000行样本及样本量缩放宽度，是近似CI；不是患者聚类bootstrap或温度参数不确定性的完整传播。", "",
              "- 独立查到排序指标错误：旧AUROC没有平均并列秩，旧AP没有按相同概率阈值合并。MIMIC神经模型2000行micro差较小，但大量重复概率的单变量模型更明显；旧CPU/GPU诊断1000行的macro AP区间端点可差0.00128。这不能描述成仅浮点末位问题。", "",
              "- 已统一重新统计当前内部五模型、7模块消融、11规模/输出/上下文状态及48外部状态的全部点指标和200次CI。AUROC采用与平均并列秩等价的正负配对计数，AP采用阈值分组，计数/累积为float64；JSON标记threshold_grouped_ap_midrank_auroc_float64_v2。温度拟合参数直接复用90%拟合证据，没有因测试指标变化重新选参数。", "",
              "- CI保留CPU随机行生成，GPU使用CPU计算出的相同概率与类排序；全行点估计在CPU计算。外部CI固定seed42，内部本轮经典模型固定seed42、MAOMAO/消融/规模实验固定seed4200。改变的是标准指标定义，不是改模型预测或挑选排名。原统计文件另存项目审计归档，不混入当前结果包。", "",
              "- 在实际MIMIC的前1000条校准行（仅90%校准部分）核对标准AP/AUROC：与scikit-learn最大绝对差1.12e−16，批量/单列实现差5.56e−17；8次固定种子CPU/GPU算术核查的排序指标CI端点完全一致，全部指标最大差5.97e−8。最终交付CI仍为200次。证据见plot_sources/ci_device_audit_corrected.json。报告中AUPRC数值采用average precision（AP），不是梯形PR面积。", "",
              "- 图源复算中的大模型曾因batch32/仅目标窗口布局与原batch64/全部验证窗口不同，在bf16下出现ECE差4.593967e−5。一致性检查拦下后按原布局复算，ECE精确恢复原0.02839457057416439；固定权重、原始目标坐标和训练均未改变。证据见plot_sources/inference_layout_audit.json。其余已完成状态也核对非排序指标；图源与本轮对应指标统一使用同一类并列顺序。", "",
              "- [数据清洗报告](MAOMAO_DATA_CLEANING_FLOW.md)列出来源代理、五分钟聚合时刻及基础normalizer拟合范围的已存在限制；本次没有重建病例或重训模型。", "",
              "## 文件", "",
              "最终结果包的external/<site>/<model>/calibration_fit.json提供全部校准行数量、原始/候选Brier、拟合目标与每步优化记录；plot_sources/提供全行ROC/PR直方图、reliability、排名/混淆汇总、时间向量、长表和清洗病例流。详见[画图说明](../outputs/maomao_plot_sources/README.md)与[最终报告](MAOMAO_V5_FINAL_RESULTS.md)。", ""]
    (ROOT/"docs/MAOMAO_CALIBRATION_AUDIT.md").write_text("\n".join(audit))


def write_v5_audit():
    from maomao.evaluation.event_bias_calibration import validate_fit
    fit_paths = sorted(EXT.glob('*/*/calibration_fit.json'))
    variant_paths = sorted((ROOT/'outputs/maomao_manuscript_figures_20260929/external_ablations').glob('*/*/*/calibration_fit.json'))
    assert len(fit_paths)==40 and len(variant_paths)==85, 'Optimizer audit requires every current full-row calibration fit'
    optimizer_rows=[]; selected_refits=[]; families=Counter()
    for path in fit_paths+variant_paths:
        fit=read(path);validate_fit(fit,fit['calibration_rows'],len(fit['bias']))
        families[fit['family']]+=1
        for step in fit['optimization_history']:
            optimizer_rows.append(dict(site=fit['site'],model=fit['model'],
                calibration_rows=fit['calibration_rows'],source_fit=str(path.relative_to(ROOT)),
                fit_sha256=sha256(path),selected_family=fit['family'],
                scope=step['scope'],family=step['family'],iterations=step['iterations'],
                full_row_passes=step['full_row_passes'],converged=step['converged'],
                stop_reason=step['message'],objective=step['objective']))
            if step['scope']=='all' and step['family']==fit['family']:
                selected_refits.append(optimizer_rows[-1])
    pd.DataFrame(optimizer_rows).to_csv(OUT/'calibration_optimizer_stops.csv',index=False)
    optimizer_audit=dict(complete=True,protocol=PROTOCOL,calibration_fits=125,
        main_fits=40,variant_fits=85,selected_families=dict(families),
        selected_full_refits=len(selected_refits),
        selected_full_refits_converged=sum(x['converged'] for x in selected_refits),
        selected_full_refits_not_converged=sum(not x['converged'] for x in selected_refits),
        selected_full_refit_stop_reasons=dict(Counter(x['stop_reason'] for x in selected_refits)),
        definition='Saved optimizer stopping flags; no inference, fitting or sealed-test selection performed by this audit',
        fits=[dict(path=str(p.relative_to(ROOT)),sha256=sha256(p)) for p in fit_paths+variant_paths])
    (OUT/'calibration_optimizer_audit.json').write_text(json.dumps(optimizer_audit,ensure_ascii=False,indent=2)+'\n')
    readme_path=OUT/'README.md'
    if readme_path.exists():
        readme=readme_path.read_text()
        if '|calibration_optimizer_stops.csv|' not in readme:
            entry=('|calibration_optimizer_stops.csv|全部125份校准的逐阶段迭代、全行评估次数、目标函数与停止原因|不根据封存测试延长拟合|\n'
                   '|calibration_optimizer_audit.json|125份拟合SHA、方法分布、选定全90%重拟合收敛计数|仅核查实际优化记录|\n')
            anchor='|calibration_audit_comparison.csv|'
            at=readme.index(anchor)
            end=readme.index('\n',at)+1
            readme=readme[:end]+entry+readme[end:]
        readme_path.write_text(readme)
        readme=readme_path.read_text().replace('MIMIC五模型micro ROC/PR','MIMIC五模型与MAOMAO校准前micro ROC/PR')
        if '示例脚本的字体与独立重绘' not in readme:
            readme += ('\n## 示例脚本的字体与独立重绘\n\n'
                       'plot_examples.py与相邻audit_panel_alignment.py一起交付，不依赖原机器的Codex技能路径。'
                       '安装合法Arial，或使用--font-dir指定TTF目录；脚本拒绝字体替换，字体文件不打包。'
                       '外部三个示例为210mm宽、原生矢量PDF及300dpi PNG。'
                       'MIMIC ROC/PR在同一轴比较五模型及MAOMAO原始/校准状态；图例显示报告的精确AP/AUROC，曲线由全行直方图近似重建。'
                       'PR只省略recall=0的约定precision=1哨兵，以所有正recall点确定纵轴；源桶计数完整保留。'
                       'MOVER可靠性使用最大softmax概率与任一真实事件top1命中，20个展示箱；报告ECE仍为15箱，不能当作独立临床事件风险曲线。\n')
        readme_path.write_text(readme)
    lines=['# MAOMAO 外部校准：恢复 V5 温度＋类别偏置','',
           f'当前协议：`{PROTOCOL}`。当前五模型与全部外部模块/规模/词表/上下文变体均采用本协议。','',
           '## 为什么恢复','',
           '早期V5采用softmax温度＋类别偏置，并在校准数据内部选择原始、仅温度、温度＋偏置三种方法。此前当前流程只拟合标量温度，不能改变每行事件排序；因此Hit/MRR/Recall保持不变。此次恢复类别分布适配能力，不回用历史小样本预测、不改变训练权重或原封存测试行。','',
           '## 完整拟合与选择','',
           '1. 固定原患者/记录代理90:10划分、种子42、模型SHA及全部有效目标坐标。只有外部90%标签传入校准函数。',
           '2. 将90%中有有效目标的患者/记录代理再按固定种子42分为80%拟合、20%方法选择；同一患者/记录代理的所有重叠窗口在同一侧。',
           '3. 三候选为原始输出、仅温度、温度＋类别偏置，输出为softmax((logits+bias)/T)。保留V5正事件集合NLL，即−log(正事件集合的总softmax概率)，偏置均方正则系数0.002；T范围0.15–6，偏置范围−4至4，拟合侧没有出现的类偏置为0。',
           '4. 按90%内部独立选择患者的正事件集合NLL选方法；改善不超过1e−6时优先原始。选定方法在全部90%有效校准行重新拟合。全量重拟合NLL变差则回退原始。不得根据10%测试结果改选方法、拟合参数或阈值。',
           '5. 保留相同模型族/正则/概率链接，优化器改为有界解析梯度L-BFGS，预设每次拟合最多80次迭代、120次函数评估，避免原Adam大量全队列遍历；这是当前框架下方法恢复，不是复现旧checkpoint、旧70:30划分或旧抽样实验。部分拟合达到迭代上限，不能宣称全部数值收敛；完整停止原因和优化记录逐模型保留。解析梯度和行logit共同平移不变性已用实际MIMIC校准行独立核对。',
           '6. 方法与参数先写入calibration_fit.json，再评分原封存10%完整有效目标行，重新生成AUROC/AP、Hit@1/5/10/All@10、63小类Hit、MRR/Recall、Brier/ECE及95%CI；类别偏置可能改变排序，不能再沿用“前后命中率必相同”的假设。','',
           '## 各来源 MAOMAO 实际结果','',
           '|来源|完整90%校准行|完整10%测试行|采用方法|T|校准前Hit@1|校准后Hit@1|校准前Brier|校准后Brier|校准前ECE|校准后ECE|',
           '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    rows=[]
    for site in SOURCES:
        fit=read(EXT/site/'maomao/calibration_fit.json');before=read(EXT/site/'maomao/metrics_uncalibrated.json');after=read(EXT/site/'maomao/metrics.json')
        validate_fit(fit,before['calibration_rows'],210)
        lines.append('|'+ '|'.join([site,f"{fit['calibration_rows']:,}",f"{after['test_rows']:,}",fit['family'],f"{fit['temperature']:.6f}",
           *[f"{m[k]:.6f}" for k in ('hit_at_1','brier','ece') for m in (before,after)]])+'|')
        for key in (*METRICS,'hit_at_5','hit_at_10','all_true_events_hit_at_10',
                    'same_family_hit_at_1','same_family_hit_at_5','same_family_hit_at_10','all_true_families_hit_at_10'):
            rows.append(dict(site=site,metric=key,raw=before.get(key),corrected_calibrated=after.get(key),
                test_rows=after['test_rows'],calibration_rows=fit['calibration_rows'],family=fit['family'],protocol=PROTOCOL))
    pd.DataFrame(rows).to_csv(OUT/'calibration_audit_comparison.csv',index=False)
    lines += ['', '## 数值优化停止状态','',
        f"125份实际校准拟合中，方法分布为{dict(families)}。选定非原始方法的全90%重拟合共{len(selected_refits)}份，"
        f"优化器报告收敛{optimizer_audit['selected_full_refits_converged']}份、未收敛{optimizer_audit['selected_full_refits_not_converged']}份。"
        '未收敛的参数是预设计算预算内的估计，不能称为充分收敛或最优校准；不根据测试表现选择性延长拟合。',
        '', '|全90%重拟合停止原因|份数|', '|---|---:|']
    lines += [f'|{reason}|{count}|' for reason,count in optimizer_audit['selected_full_refit_stop_reasons'].items()]
    lines += ['', '逐阶段次数、目标函数与停止原因见plot_sources/calibration_optimizer_stops.csv；全部125份拟合文件及SHA见calibration_optimizer_audit.json。']
    lines += ['', '## 解释与限制','',
        '- 外部90%带标签数据用于适配；校准后属于经过外部校准的验证。校准前才体现冻结模型直接迁移。两种结果在相同10%行上并列，不能把校准后成绩称为零样本迁移。',
        '- 恢复的是事件输出校准；不重新训练编码器，不校准dual-timescale时间头。时间报告仍按三个真实等待时间尺度分别给出MAE。独立85:15临床风险映射/DCA研究保持其已验证方法，两种概率不混用。',
        '- 正事件集合NLL衡量总正事件概率；Brier以q=y/sum(y)评分；ECE比较最大softmax概率与top1是否命中任一正事件。这三种目标不同，校准不能保证所有测试指标同时改善，也不能预设Top1达到50%–60%。历史数字需先确认采用相同指标、词表、有效目标行与测试划分，才能与当前结果比较。多正事件softmax不是各事件独立发生概率。',
        '- 全行点估计与200次95%CI保留既有口径：主外部大队列均匀100,000行、外部变体30,000行bootstrap并缩放宽度，属于目标行近似CI，未传播校准参数不确定性或完整患者聚类相关性。',
        '- 小型麻醉队列只有少量源记录代理、23–118个重叠测试目标行，不能将目标行当作独立患者。校准内部分组只统计具有有效目标的患者/记录代理，其人数可能低于90%原病例人数。',
        '- 当前10%测试集此前已报告，本次方法恢复后的复算不是新增未见过的确认性测试；测试结果仅用于核查与汇报，未反馈拟合或选择。',
        '- 前一轮标量结果保留在项目独立审计归档calibration_scalar_archive_20260930；最终交付采用当前完整结果，不夹带旧小样本输出、消融checkpoint或私人分组标识。','',
        '## 可复核文件','',
        'external/<site>/<model>/calibration_fit.json包含所有类别偏置、温度、候选选择分数、校准内部分组行数及全量优化记录。calibration_revision/包含预先固定协议、数值核对与125次校准任务（40个主模型＋85个外部变体）的推广核查；8组主MAOMAO与85组变体共93组校准前后配对。selection_partition_verification.json从原始病例表、序列长度、校准窗口坐标及词表阳性掩码独立重建全部125份拟合的80:20患者分组和行数，不读取测试标签，也不导出私人分组标识。plot_sources/提供完整测试行曲线桶计数、排名/混淆和匿名检索向量；manuscript_figures/source_data/及external_ablations/提供外部变体全行指标与证明。详见[最终报告](MAOMAO_V5_FINAL_RESULTS.md)。','']
    (ROOT/'docs/MAOMAO_CALIBRATION_AUDIT.md').write_text('\n'.join(lines))


if __name__=="__main__":main()
