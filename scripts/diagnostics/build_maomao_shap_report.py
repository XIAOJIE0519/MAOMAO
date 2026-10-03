#!/usr/bin/env python3
"""Describe the frozen, actual MAOMAO SHAP study and Delphi layout adaptation."""
import sys,json,shutil,importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.diagnostics.maomao_display_family_groups import GROUPS,display_mapping
from scripts.diagnostics.uniform_result_scope import sha256
OUT=ROOT/'outputs/maomao_plot_sources/shap'
def read(p):return json.loads(p.read_text())
def main():
    protocol=read(OUT/'protocol.json');proof=read(OUT/'numerical_verification.json');agg=read(OUT/'aggregation.json')
    if not proof['complete'] or not agg['complete']:raise RuntimeError('Full-cohort source verification missing')
    meta=read(ROOT/'data/perioperative_event_sequences_v5_richctx_static7/event_sequence_meta.json')
    (OUT/'event_sequence_meta.json').write_text(json.dumps({k:meta[k] for k in ('outcome_vocabulary','outcome_family_vocabulary','outcome_to_family','token_vocabulary','audit_counts')},indent=2)+'\n')
    mapping,labels=display_mapping(meta['outcome_family_vocabulary'])
    (OUT/'clinical_group_definitions.json').write_text(json.dumps(dict(model_families=meta['outcome_family_vocabulary'],family_to_clinical_group=mapping,clinical_group_names=labels,model_unchanged=True,
        grouping_scope='Descriptive grouping for SHAP and embedding figures, not a new trained output head. External report Hit tables use only the original 63 families.'),indent=2)+'\n')
    source=ROOT/'outputs/maomao_retrieval_revision_20260929/reference/source_provenance.json'
    shutil.copy2(source,OUT/'delphi_reference_provenance.json')
    shutil.copy2(ROOT/'outputs/maomao_family_shap_20260929/fp32_correction.json',OUT/'precision_correction.json')
    (OUT/'requirements-replot.txt').write_text('numpy==2.2.6\npandas==3.0.5\nmatplotlib==3.10.6\n')
    versions={name:importlib.metadata.version(name) for name in ('numpy','pandas','matplotlib','torch','shap','scipy','numba','llvmlite','umap-learn','scikit-learn','pillow')}
    (OUT/'runtime_versions.json').write_text(json.dumps(versions,indent=2)+'\n')
    for src,dest in [('plot_maomao_delphi_shap.py','replot_maomao_shap.py'),('maomao_display_family_groups.py','maomao_display_family_groups.py')]:shutil.copy2(ROOT/'scripts/diagnostics'/src,OUT/dest)
    shutil.copy2('/home/yunkunshi/.codex/skills/nature-figure/scripts/audit_panel_alignment.py',OUT/'audit_panel_alignment.py')
    importance=pd.read_csv(OUT/'clinical_group_shap_importance_95ci.csv').sort_values('mean_abs_log_probability_shap',ascending=False)
    budget=read(OUT/'budget_sensitivity.json');delta=[x['mean_absolute_shap_difference'] for x in budget['results']]
    with np.load(OUT/'figure4c_display_matrix.npz') as a:predictors=len(a['predictor_event_ids']);outputs=len(a['predicted_event_ids'])
    md=['# MAOMAO family SHAP 与 Delphi Figure 4a/4c 版式复现','',
        '## 原文方法与 MAOMAO 的对应关系','',
        '原文 Figure 4a 是学习到的 token embedding 的 UMAP，可用疾病章节着色并放大局部；Figure 4c 才是 SHAP 聚合热图。后者按既往预测事件距离当前查询的时间分组（<5 年和 >10 年），不是两个未来预测时间窗。原文解释的是疾病发生率的对数；MAOMAO 解释的是下一事件 softmax 概率的对数，因此本图倍数为概率贡献倍数。见 [Nature 原文](https://www.nature.com/articles/s41586-025-09529-3) 与 [官方绘图 notebook](https://github.com/gerstung-lab/Delphi/blob/fb72166be6b29d8db819227a59487e51c1df1454/shap_analysis.ipynb)。','',
        '原文 Methods 描述基于 PartitionExplainer 的临床 token masking；当前固定版本公开 wrapper 实际删除被 mask 的临床 token、保留 no-event token，并可翻转 sex。两者的具体 masker 实现需要区分。MAOMAO 采用已有辅助训练的 `<MASK>`，其 mask game 如下定义。参考代码固定到 commit `fb72166be6b29d8db819227a59487e51c1df1454`，来源哈希见图源 provenance。[官方 SHAP wrapper](https://github.com/gerstung-lab/Delphi/blob/fb72166be6b29d8db819227a59487e51c1df1454/utils.py)。','',
        '## 实际样本与冻结模型','',
        f'- 使用当前 MAOMAO 全量训练后的冻结 checkpoint；SHA256 `{protocol["model_sha256"]}`。不为 SHAP 重新训练。',
        f'- INSPIRE 内部 90:10 患者划分：训练 {proof["train_patients"]:,} 人，验证 {protocol["heldout_patients"]:,} 人；全部 {protocol["eligible_patients"]:,} 位验证患者均有有效查询，排除 {protocol["excluded_no_valid_query"]} 人。',
        '- 每位患者按 seed42 从其有效固定窗口中均匀选择一个窗口，解释该窗口最后一个有效查询。覆盖所有验证患者，但不是解释全部 1,563,972 个验证目标行。窗口选择不使用结局类型或预测表现。',
        '- 仅保留查询及此前的 causal prefix。直接患者 ID、住院 ID、窗口索引和私有源查询计划不放入图源包；文件仅使用匿名 case 编号。','',
        '## SHAP game 与可解释边界','',
        '输入特征为既往事件身份 × 时间滞后：<2h、2–<24h、≥24h。同事件在同一滞后层的多次出现共享一个特征；包括窗口前的已有事件计数。移除特征时，其临床 token 替换为 `<MASK>`，value/has_value 置零，并从 retained coalition 重算窗口前 family 计数。224 个存储类先映射到 210 个可预测事件；14 个负 remap 的输入上下文类保持固定，不能用 −1 索引死亡事件。','',
        '时间网格、gap、measurement intensity、phase、static 以及其它上下文保持固定。因此本结果是给定这些信息的事件内容贡献；summary 等上下文仍可能包含相关临床信号，不能视为删除全部该疾病信息的干预。baseline 是该患者固定上下文下的 mask baseline，不是总体健康人的风险。','',
        '`shap.PartitionExplainer` 使用按原63个 family 排序的 event/lag 分组 complete-linkage 树，max_evals=500；这是分层 coalition 的有限预算贡献近似（Owen/Partition attribution），不能声称精确穷举 Shapley 值。每次解释210个 log事件概率及63个 log(sum事件概率) family 输出，共273维。family 概率使用1e−12下限。这里的softmax是归一化下一事件概率质量，不是各临床结局在固定未来时间窗内的独立发生风险。固定batch32与bf16 forward；事件softmax、family概率矩阵求和和取log全部在autocast之外以float32计算。曾在独立核查中发现family汇总被隐式转成bf16，现已将整个解释队列重算，未复用旧φ；修正记录见precision_correction.json。经过实际273维输出逐位等价核查，运行时跳过不影响事件概率的时间/辅助输出头以加快计算，原encoder、输入变换、outcome和family head权重全部保持冻结。由于本game的时间网格固定，进一步缓存同一患者的精确relative-time/block-causal mask；与未缓存版本的30个实际coalition、273输出逐位一致，未改变mask或推理batch。标签张量不参与预测。','',
        f'独立核查所有 {proof["patients_verified"]:,} 个 case 的实际历史事件、输入分组、窗口前计数、概率归一化和全部聚合矩阵；从存储的 float32 φ 重算最大加性残差为 {proof["max_stored_float32_additivity_residual"]:.3g}（阈值1e−4）。该加性核查只验证记账一致性，不能证明近似已收敛。','',
        f'预先固定的前25个匿名 case 额外比较500与1000次预算：每case平均绝对φ差的均值 {np.mean(delta):.4f}，最大 {max(delta):.4f}；单个φ最大差 {proof["budget_sensitivity_max_single_attribution_change"]:.4f} log概率单位。因此局部贡献存在可见预算敏感性，图为描述性解释，不能当作稳定机制证据。完整对照逐项保存在 `budget_sensitivity.json/npz`。','',
        '## 63 个原 family 与 17 个临床大类','',
        '模型保留原来的63-family head。SHAP与嵌入图把输入事件按17个临床大类组织，覆盖每个原 family 恰好一次；这是描述性绘图分组。外部结果报告的family Hit仅展示原63个小类，不展示17大类Hit表。','',
        '|临床大类|包含的模型 family|','|---|---|']
    for label,families in GROUPS:md.append('|'+label+'|'+', '.join(families)+'|')
    md += ['', '对输入贡献做求和是线性的；我们没有把63个 log输出概率的SHAP简单相加后冒充17个合并输出概率的SHAP。额外17类重要性是先在同一lag内合并输入φ，再对原63输出取平均绝对值，最后求和三个lag；其含义见NPZ。','',
        '## 主图：按原 Figure 4a 与 4c 的面板结构绘制','',
        '### a：嵌入 UMAP','',
        '210个临床事件的384维输入 embedding 来自冻结 MAOMAO。对全部2,190个输入token拟合cosine UMAP（seed1413，n_neighbors30，min_dist0.05），显示210个事件。颜色为17个临床大类；另有完整63-family同坐标图。点面积按全部保留 INSPIRE 源事件频数的平方根缩放并设可读性上限，完整频数见CSV。','',
        '布局保留中央散点、四个虚线放大框、连接线、右侧family图例和频数图例。放大锚点固定为MAP低血压、低氧、AKI stage1与住院死亡，只展示真实几何邻域；死亡embedding实际孤立，没有为构造疾病簇移动坐标。MAOMAO具有显式family训练结构，输入embedding未与输出head绑定，不能套用“模型完全不知道疾病大类”的原文结论。','',
        '![MAOMAO Figure 4a](plot_sources/shap/figures/figure4a_maomao_embeddings.png)','',
        '### c：三个时间尺度的既往事件贡献热图','',
        f'从左至右分别为<2h、2–<24h、≥24h。三张热图使用完全相同的事件行列：在三个历史lag均有至少6位患者支持的 {predictors} 个预测特征，以及额外死亡输出，共 {outputs} 列。颜色为 exp(平均log事件概率SHAP)，共享0.1–10对数色阶；蓝色<1，白色≈1，红色>1，色阶外饱和但原值保留。主图按17临床大类分区，各大类占相同物理空间，类内每个事件仍保留独立原值；格子宽度不编码样本量。无支持的罕见事件不填伪值，完整210事件/63-family三层lag矩阵均保留。','',
        '三个分界与最终结果MD中的分尺度时间MAE一致：<2h、2–<24h、≥24h。SHAP图按既往输入到当前查询的滞后分层；时间MAE表按未来事件真实等待时间分层，二者共享分界但不是同一统计量。这里不解释未来分尺度风险。倍数为单个分组对概率的乘法贡献，平均φ的指数是几何平均贡献倍数，不能读作直接observed vs baseline平均风险比、hazard ratio或因果效应。','',
        '![MAOMAO Figure 4c](plot_sources/shap/figures/figure4c_maomao_shap_by_family.png)','',
        '## MAOMAO 额外贡献图','',
        '|图|含义|','|---|---|',
        '|clinical_group importance|17类输入贡献的重要性及500次患者bootstrap 95% CI；仅反映这组查询和mask game。|',
        '|death clinical-group distribution|所有观察到该大类的患者对下一事件死亡概率的带符号贡献；横向是真实φ，纵向jitter仅防遮挡。|',
        '|local clinical-group waterfalls|固定匿名case00000的AKI/death原63-family输出；显示6个主要输入大类与其它项，未挑选最好看的病例。|',
        '|63-family importance / distribution / waterfalls|同一解释的细粒度版本。|',
        '|full 63-family matrix|<2h、2–<24h、≥24h三张完整63×63原family输出贡献矩阵，各时间尺度中低于6位患者支持的预测行显示灰色，区别于贡献≈1的白色。|','',
        '![17类重要性](plot_sources/shap/figures/maomao_shap_clinical_group_importance.png)','',
        '![死亡贡献](plot_sources/shap/figures/maomao_shap_death_clinical_group_distribution.png)','',
        '![局部解释](plot_sources/shap/figures/maomao_shap_local_clinical_group_waterfalls.png)','',
        '### 实际重要性汇总','',
        '|输入临床大类|具有该历史输入的患者数|平均绝对贡献（log概率；95% CI）|','|---|---:|---:|']
    for row in importance.itertuples():md.append(f'|{row.family}|{int(row.patients_with_input):,}|{row.mean_abs_log_probability_shap:.4f} ({row.ci_lower:.4f}–{row.ci_upper:.4f})|')
    md += ['', '此表是原63-family输出下输入大类的重要性，取绝对值后无“保护/危险”的方向含义。均值对全部9,989患者取平均，未出现输入的患者贡献为0；分组大小和暴露频率都会影响总体重要性。Death作为既往输入在这些有效查询中出现0人，故输入贡献为0，不表示模型不能预测未来死亡。500次患者bootstrap没有加入SHAP预算误差，也不是外部总体的效应区间。','',
        '## 源数据、重绘和核查','',
        '`plot_sources/shap/` 包含9,989个逐case全部273输出贡献、baseline/full log概率、原始输入分组、三个lag的210×210与63×63矩阵、支持数、全部17类/63类重要性、死亡贡献向量、局部图数据、冻结embedding与UMAP坐标、完整频数、分组字典、预算敏感性对照和来源哈希。每个最终图同时导出SVG/PDF/PNG300dpi/TIFF600dpi，PDF文本与面板对齐核查随图保存。','',
        '解压后在 `plot_sources/shap/` 下安装numpy/pandas/matplotlib，运行 `python3 replot_maomao_shap.py --output replotted`，无需原始患者数据或checkpoint即可按缓存坐标与全部真实φ重绘所有图。重新拟合UMAP或重新计算SHAP仍需原项目模型和源数据。','',
        '直接患者标识未导出；匿名临床轨迹贡献仍应按研究数据管理要求处理。图源不包含消融checkpoint、原始临床表或用于选窗口的私有索引。数据字段见 `data_dictionary.json`，原文件名中的 `*_log_family_probability` 是历史命名，实际长度273（先210事件、后63family）。','']
    report=ROOT/'docs/MAOMAO_FAMILY_SHAP.md';report.write_text('\n'.join(md))
    readme='# MAOMAO SHAP 图源\n\n完整方法见结果包根 MAOMAO_FAMILY_SHAP.md。覆盖全部9,989验证患者，每人一个固定seed42查询，不是全部验证目标行。主图按17个临床大类组织，模型仍为63-family head；Figure4c与完整63-family热图均展示<2h、2–<24h、≥24h三个时间尺度。\n\n重绘：`python3 replot_maomao_shap.py --output replotted`。依赖numpy、pandas、matplotlib；使用缓存UMAP，不需要torch/shap/umap或原checkpoint。\n\n逐case数组无直接标识，全部273输出；每个feature ID=3×原210事件ID+lag0/1/2。完整字典在data_dictionary.json。baseline为固定上下文下MASK事件内容，并非健康人。\n\nmax_evals500的Partition贡献为有限预算近似；25例预算对照未证明收敛。加性检查只证明贡献总和记账一致。clinical_group_shap_importance.npz合并的是输入贡献，不声称17输出概率的SHAP。\n\n所有PNG/SVG/PDF/TIFF均来自实际模型；无伪造填色或移动embedding以创造类别簇。完整支持数、矩阵和绘图可编辑源包含在本目录。\n'
    (OUT/'README.md').write_text(readme)
    print(json.dumps(dict(report=str(report),patients=proof['patients_verified'],clinical_groups=len(labels))))
if __name__=='__main__':main()
