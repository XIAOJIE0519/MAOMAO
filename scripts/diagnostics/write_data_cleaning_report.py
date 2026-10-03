#!/usr/bin/env python3
"""Render the database cleaning account from recounted aggregate evidence."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/"outputs/maomao_plot_sources/data_cleaning/flow_evidence.json"


def main():
    data=json.loads(P.read_text());e=data["evidence"];flow=data["stages"]
    labels={"inspire":"INSPIRE","mimic":"MIMIC-IV 3.1","mover":"MOVER","eicu":"eICU 2.0","sicdb":"SICdb 1.0.8","ntuh":"NTUH ECG","asac":"Auckland ASAC / EDS","uq":"UQ Vital Signs","surgical_pooled":"NTUH + ASAC + UQ 三源合并"}
    lines=["# MAOMAO 数据库清洗流程、病例流与数据口径", "",
           "本文件依据本地发布源表、实际预处理代码及当前冻结事件数据重新核算。人数不是测量行数；一个患者可以有多次住院、ICU 停留或手术。所有排除计数采用按代码顺序执行的互斥阶段，患者排除数指该阶段后完全不再出现的患者 ID 数。", "",
           "## 起始与最终样本概览", "",
           "|数据库|起始本地源记录|起始可识别患者|最终手术/ICU/记录单元|最终患者或记录代理 ID|未进入最终分析的患者/记录|事件 token|",
           "|---|---:|---:|---:|---:|---:|---:|"]
    initial={s:next((r for r in flow if r["site"]==s),{}) for s in labels}
    for s,label in labels.items():
        x=e[s];f=initial[s]
        if s=="surgical_pooled":raw,people="176 个源记录","未提供跨记录患者身份"
        else:raw=f'{f["records"]:,}';people=f'{f["unique_person_ids"]:,}' if f.get("unique_person_ids") is not None else "未提供真实患者身份"
        removed=f'{f["unique_person_ids"]-x["final_unique_person_ids"]:,} 名患者' if f.get("unique_person_ids") is not None else f'{(176 if s=="surgical_pooled" else f["records"])-x["final_records"]:,} 个记录（人数未知）'
        lines.append(f'|{label}|{raw}|{people}|{x["final_records"]:,}|{x["final_unique_person_ids"]:,}|{removed}|{x["tokens"]:,}|')
    lines += ["", "MIMIC 起始 364,627 是发布患者总表（含无住院的患者），不是本研究手术候选数；另有 546,028 次住院、223,452 名住院患者、94,458 次 ICU 停留和 65,366 名 ICU 患者。手术候选从 ICU 内 OR Sent / OR Received 记录获得。", "",
              "MIMIC从全发布患者总表到最终手术分析少360,707名，其中大量本来没有符合本研究的住院/ICU/手术往返条件，不是360,707名都因质量差被剔除。其余大型库的净人数差也包含手术资格筛选；具体阶段见下表。", "",
              "NTUH、ASAC/EDS、UQ 的 subject_id 由记录名生成，108 / 25 / 32 是独立记录代理数，不能证明是同样数量的去重自然人。90:10 在这些记录代理 ID 上划分；大型临床库及 INSPIRE 使用实际患者 ID。三源合并是这些记录的派生汇总，不是额外独立数据库，不能与三源分别的人数再相加。", "",
              "本报告起点是当前机器可用的发布/解压源表。未提供的医院采集前筛选或发布前去标识化病例流不推断、不编造。INSPIRE 已发布 operations 表保留全部 130,960 条，本项目此阶段没有剔除患者；不是声称原始医院人群从未筛选。", "",
              "## 各数据库逐阶段清洗", ""]
    descriptions={
      "inspire":["源表 `data/train_data/operations.csv.gz`；按 subject_id + hadm_id 建住院，再保留每次手术 episode。必要 ID、入出院时间错误或重复 op_id 会使预处理报错，不会静默创造时间。当前成功数据无此类剔除。",
                 "时间单位为源分钟。观察先匹配住院：vitals 用 op_id 和时间/患者核对；其他表按患者与住院区间匹配，多住院同时匹配的观测剔除。同一 5 分钟右闭桶保留最新时刻；同刻重复数值取均值。labs 历史拼写 lacate 明确映射 lactate。",
                 "每手术窗口为 OR 进入前最多 7 天，手术结束后最多 30 天，并截断至出院/本次住院死亡。异常手术结束时间超过 OR 进入后 24 小时不用于延长窗口，不是把整个患者删除。较后住院的死亡不能延长较早住院随访。635 条后于本次出院的死亡时间仅作审计。"],
      "mimic":["发布 hosp/patients、hosp/admissions、icu/icustays 与 procedureevents。先找 OR Sent，再在同一 ICU stay 中匹配其后 24 小时内最早且未使用的 OR Received。它们是手术室往返代理，不是专门的外科手术日志。",
               "7,551 条有效 OR Sent 中 2,130 无匹配，余 5,421 对；住院/患者/ICU 表连接无进一步损失；8 对因有界 episode 无效被剔除，最终 5,413 个 episode、3,920 名患者。16,919 条 OR Received 中 11,498 未用于配对，不能将其算成被排除的患者。",
               "窗口为 max(住院入院, OR Sent−7天) 至 min(出院/死亡, OR Received+30天)。chartevents、labs、timestamped inputevents 做支持特征和单位映射；无临床 chart timestamp 的 diagnoses_icd 不进入因果历史。年龄用 anchor_age + (事件年份−anchor_year)。"],
      "mover":["源表 EPIC_EMR/EMR/patient_information.csv 65,728 行；先剔除缺 LOG_ID、MRN、OR 进入/退出时间，再限制 OR 时长 >0 且 ≤24h，按 LOG_ID 去重，最后检查住院截断后的手术窗口。",
               "5 分钟 physiology 来自发布的 flowsheets_cleaned，另外读取带时间戳 lab / medication；本机没有这些 cleaned flowsheets 的上游完整清洗账本，不能把它们宣称为未经任何处理的原始波形。",
               "窗口 max(入院, OR进入−7天) 至 min(出院, max(OR退出,麻醉结束)+30天)。Epic WEIGHT 为盎司，除 35.27396195 转 kg。BIRTH_DATE 字段实际为数值（发布表范围17–90），适配器作为年龄；本地未附该字段更早的生成说明。ICU_ADMIN_FLAG 为真的 OR 退出时刻仅是 ICU 转入代理。无时间戳的诊断、病史/计费与术后并发症字段不进入序列。"],
      "eicu":["200,859 次 ICU 记录 / 139,367 名患者；按 patientunitstayid 去重（本次没有重复）。仅保留 admissionDx 中明确 Operative 分支或 OR-within-4-hours=Yes 的 ICU，不能靠 SICU 科室名推断手术。",
              "筛出 34,243 次记录 / 32,166 名患者；再排除 unitdischargeoffset 缺失或 ≤0 的 47 次停留，其中 30 名患者因此完全退出；最终 34,196 次停留、32,136 名患者。",
              "时间为相对 ICU 入院分钟，pre-ICU 观察未提供。age >89 编为90，weight/height 用数值字段，gender 编码；periodic/charting/lab/medication 做支持特征映射，呼吸机 start/stop 单独抽取。未将 ICD9 错配成冻结 ICD10 类。"],
      "sicdb":["源 cases.csv.gz 27,350 次病例 /21,566名患者；用 d_references 映射 SurgicalAdmissionType，仅保留 Elective Surgery 或 Urgent Surgery，排除 No Surgery / Unknown。",
               "筛选后 13,041 次病例、11,094 名患者；再排除 ICU offset/时长无法形成有效窗口的27次病例，其中19名患者完全退出，最终13,014次病例、11,075名患者。",
               "ICUOffset / TimeOfStay 的秒转换为分钟；观察窗为最多7天pre-ICU、30天post-ICU且不超过实际stay。WeightOnAdmission >500 的数值除1000转kg。部分有心脏手术时刻，其他为 ICU 锚点代理。数据源为 data_float_h 小时均值，不能描述为逐分钟原始波形；不人为展开小时值为分钟值。"],
      "ntuh":["原110条手术ECG记录；case_manifest 有2条既有质量控制排除，原因分别为 extreme-amplitude artifact fraction 1.081637% / 3.379382% 超过1%。保留108条，本轮没有额外 HR 质量排除。",
              "使用已带通0.5–40Hz的500Hz ECG，经正/负极性QRS检测；peak distance至少0.30s、prominence=max(0.25,2.5×MAD标度)，保留RR0.30–2.40s。每分钟至少8个有效RR，HR=60/median(RR)，需25–200次/min；选择RR离散程度较小的极性，双极性HR差>15计入审计。整条需至少50%分钟窗口有效，再进入5分钟桶。无患者级年龄、性别、实验室、药物或可靠手术阶段详细时间，不补造这些信号。",
              "注意：旧 metadata 将下一步写成5-min median，但实际 bin_measurements 用 groupby.mean()，即分钟HR进入5分钟均值。报告以代码执行为准，注明这一历史描述差异。最终2,711条五分钟HR观察、3,343个token。"],
      "asac":["ASAC MAT 与 EDS CSV 共34条源记录；需已知operation或明确surgery-start事件。9条 operation unknown 且无明确手术开始事件被剔除；其余25条，无额外整条QC剔除。",
              "映射HR、SpO2、CO2、RR、体温、动脉/无创血压等已支持指标；有限范围检查共拒绝110,014个无效/越界数值，这个数字不是患者数。5分钟桶按均值聚合，共3,618条五分钟观察；只在有真实phase记录时加入phase。最终2,226个token。"],
      "uq":["发布32条手术麻醉记录，全部32条入选；不剔除非手术记录或整条QC记录。只使用有支持映射且时间有限/非负、数值合理的信号。",
            "剔除值而非患者：HR15、etCO2 1,088、awRR16,308、RR22,977、ARTSys12、ARTDia12、ARTMean108；其他列越界0；invalid_time_rows=0。5分钟桶均值聚合，5,207条五分钟观察、2,622个token。手术阶段详细信息缺失，不能从文件时间点伪造阶段。"],
      "surgical_pooled":["合并当前NTUH108 + ASAC/EDS25 + UQ32 =165条记录，使用来源前缀避免ID碰撞；共8,191个token。起始176条发布记录，2条ECGQC与9条手术筛选排除，共11条。",
                         "合并后重做record-proxy级90:10划分，为149/16。其测试记录与每来源单独划分的测试记录不必相同，因此不可将来源指标简单平均当作合并指标。"]}
    for site,label in labels.items():
        lines += [f"### {label}",""]
        for paragraph in descriptions[site]:lines += [paragraph,""]
        rows=[r for r in flow if r["site"]==site]
        if rows:
            lines += ["|阶段（与审计CSV一致）|记录数|患者ID/代理数|较前阶段去除记录|完全退出患者ID|","|---|---:|---:|---:|---:|"]
            for r in rows:
                v=lambda key:f'{r[key]:,}' if r.get(key) is not None else "—"
                lines.append(f'|{r["stage"]}|{v("records")}|{v("unique_person_ids")}|{v("records_removed_previous_stage")}|{v("person_ids_no_longer_present")}|')
            lines.append("")
    lines += ["## INSPIRE 观测表的分配与排除（不是患者排除）", "",
              "|源表|原始行|匹配住院行|住院区间外行|多住院歧义行|", "|---|---:|---:|---:|---:|"]
    for table in ("vitals","labs","ward_vitals","medications","diagnosis"):
        a=e["inspire"]["source_audit"][table]
        lines.append(f'|{table}|{a["rows"]:,}|{a["assigned"]:,}|{a["outside_admission"]:,}|{a["ambiguous"]:,}|')
    lines += ["", "这些计数来自住院时间匹配阶段，不能把未匹配测量行解释为独立病例剔除。该阶段各表 invalid_time/invalid_value/unknown_feature 记录为0，只说明这个已发布表与映射阶段，不能推断所有上游信号完全无异常。", "",
              "## 数值过滤、单位与五分钟桶的实际差别", "",
              "|来源/分支|桶编号与写入时刻|桶内数值|额外数值处理|", "|---|---|---|---|",
              "|INSPIRE|ceil(t/5)，写入右端点|最新时刻；同刻重复均值|发布源/住院分配及事件规则；不等同外部均值处理|",
              "|MIMIC / MOVER / eICU / SICdb|ceil(t/5)，writer写入bin×5分钟，即桶右端点|sum/count均值|支持变量、有效时间和数值转换；具体分支过滤不同|",
              "|ASAC / EDS / UQ|floor(t/5)，writer写入bin×5分钟，即桶左端点|均值|有限/非负时间及下表范围|",
              "|NTUH|分钟RR中位数得到HR，再floor(t/5)写入桶左端点|分钟HR均值|既有ECG幅度QC、QRS/RR及分钟有效率|", "",
              "MIMIC温度223761从°F转°C，FiO2≤1.5乘100。MOVER按单位/温度值转换°F→°C、FiO2比例→%；实验室glucose mmol/L×18.0182、creatinine μmol/L÷88.4、albumin/Hb g/L÷10、bilirubin μmol/L÷17.104。eICU charting温度>60按°F转换、FiO2≤1.5乘100。未见统一覆盖所有源变量的winsorization或全体患者缺失值插补，不能把不同分支写成同一套异常值过滤。", "",
              "ASAC/EDS/UQ的闭区间保留范围来自HR_FEATURES；非有限值和越界值被删除，不整条删除病例：", "",
              "|变量|保留范围|", "|---|---|", "|HR/Pulse|25–240 次/min|", "|SpO2|50–100 %|", "|ETCO2|10–100（源CO2单位未附跨源独立核验）|", "|RR/AWRR|1–80 次/min|", "|体温|25–45 °C|", "|ART/NBP SBP|30–300 mmHg|", "|ART/NBP DBP|20–200 mmHg|", "|ART/NBP MAP|20–250 mmHg|", "",
              "eICU nursing/respiratory charting分支另用：体温25–45、HR20–260、RR1–100、SpO2 40–100、MAP10–250、CVP−10–100、CI0–15、ETCO2 0–120、glucose5–1500、GCS-M1–6、GCS-E1–4、Pplat0–100、PEEP0–60、PIP0–120、FiO2 20–100、分钟通气量0–100；这些范围不应套用到其periodic/aperiodic和全部labs路径。MIMIC/MOVER常规数值分支主要剔除无法解析/非有限读数并限制episode窗口，不能声称已采用同一上述范围。", "",
              "## 统一事件表示与有效目标行", "",
              "- 冻结INSPIRE词表：2,190输入token、210预测事件类、7项静态输入。外部来源不新建输出类别；无支持映射的变量保留为未映射审计，不转换成虚构事件。",
              "- 主任务真值是下一批已记录/可抽取事件，不是210种临床病变的独立 adjudicated 结局。未记录不等于未发生；仅有ECG/少数生命体征的小库无法验证所有实验室、药物和器官结局。micro指标包含缺支持类的负目标，macro指标仅在可估计类上计算，须结合逐事件支持数解释；临床终点评分比较另有明确端点限制。",
              "- 数值读数用于状态变化、恶化/恢复、突变事件；正常稳定的每个原始读数并不都成为预测标签。药物名称按明确generic/alias映射，外部同药15分钟内去重；窗口外事件与完全相同记录去重。",
              "- 7项静态输入依次为age/100、male、BMI/40、ASA/6、emergency、weight/150、height/200。weight截到0–300kg、height到0–250cm、BMI到0–80、ASA到0–6；缺失缩放后填0，没有另加静态缺失指示列。不因缺年龄/体重等静态值整例剔除；male=0也可能代表缺性别。动态摘要使用已入桶历史，滚动观察60分钟、每30分钟checkpoint，最多48次；部分来源并无该特征。",
              "- 当前窗口长256、stride128；同一临床事件可能在重叠窗口中成为多个训练位置。有效目标行要求loss_mask有效且有非空下一事件集合；不是把源表的一行等同于一个训练目标。INSPIRE 14,128,539训练目标行、1,563,972验证目标行，对应89,897/9,989名患者。",
              "- 各外部库的完整90%校准与10%测试目标行如下；所有五模型使用相同坐标与目标，不抽样计算点估计。", "",
              "|来源|校准患者/记录代理|测试患者/记录代理|校准目标行|测试目标行|", "|---|---:|---:|---:|---:|"]
    for s in labels:
        if s=="inspire":continue
        x=e[s];sp=x["split"]
        lines.append(f'|{labels[s]}|{sp["patients_validation"]:,}|{sp["patients_test"]:,}|{x["calibration_target_rows"]:,}|{x["test_target_rows"]:,}|')
    lines += ["", "## 本次核查发现的描述与可用性限制", "",
              "1. 小型麻醉数据的subject_id是记录代理，未提供真实患者重复记录的链接，不能额外声称严格自然人无交叉。",
              "2. NTUH/ASAC/EDS/UQ适配器及其三源合并将5分钟内数值聚合到floor桶，并用桶起点作为token时刻，可能在桶内提前最多5分钟获得聚合读数。MIMIC/MOVER/eICU/SICdb实际使用ceil桶右端点，不能把这个问题扩大成所有外部库；INSPIRE也采用右闭桶，但桶内取最新值。当前冻结结果保留既有处理；小型来源严格实时解释应统一为桶结束可用时间后重建/评估。这个问题不等同温度拟合错误。",
              "3. INSPIRE基础观测normalizer在发布训练来源整体（患者划分之前）拟合median/IQR；代码中的fit_split=train指来源，不证明只用了内部90%患者。当前所有冻结模型共用这一既有输入；本次没有重训。严格的完全隔离预处理实验应在90%患者上重新拟合后重训，并单独报告。",
              "4. EventMAOMAO将七项static向量投影后广播到每个序列位置；BMI/ASA/emergency/weight/height在metadata中标为operation baseline，但没有按事件位置限制其可用时间。因此对于OR进入之前的预测，不能声称所有静态字段都已经通过可用时间隔离。动态体重/身高token虽在OR已知时刻加入，也不消除这条静态旁路。严格前瞻分析需明确实际记录时刻、按时刻屏蔽静态字段后重训/评估。",
              "5. ICU锚点、OR往返/转入代理与真实手术日志语义不同；SICdb小时均值不能伪称高频原始采样。",
              "6. 本次校准修正固定原模型/数据/测试患者，针对概率校准，不会消除上述来源或表示限制。", "",
              "## 可追溯证据", "",
              "- [逐阶段CSV](../outputs/maomao_plot_sources/data_cleaning/cohort_flow.csv)与[源证据JSON](../outputs/maomao_plot_sources/data_cleaning/flow_evidence.json)只含聚合数，无患者ID。JSON记录当前metadata/admissions文件SHA256及source_audit。",
              "- `scripts/diagnostics/audit_data_cleaning_flow.py` 从小型cohort源表重放实际筛选；`data/preprocess_mimic_validation.py`、`data/preprocess_mover_validation.py`、`data/preprocess_icu_validation.py`、`data/preprocess_surgery_external_validation.py`及`data/external_validation_common.py`是来源处理实现。",
              "- `maomao/data/preprocess_timeline.py`、`scripts/preprocess_event_sequences.py`和`maomao/data/clinical_events.py`定义INSPIRE住院分配、事件表示及阈值；当前冻结结果报告见[最终报告](MAOMAO_V5_FINAL_RESULTS.md)。", ""]
    (ROOT/"docs/MAOMAO_DATA_CLEANING_FLOW.md").write_text("\n".join(lines))


if __name__=="__main__":main()
