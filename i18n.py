"""Interface translations; machine-readable input/output keys stay stable."""
TEXT = {
'en': {
'subtitle':'Multi-horizon Anticipatory Outcome Model for Anesthesia and Operations',
'links':'[Model](https://huggingface.co/luan0519/MAOMAO) · [GitHub](https://github.com/XIAOJIE0519/MAOMAO) · [Vocabulary](https://huggingface.co/luan0519/MAOMAO/resolve/main/vocabulary.json)\n\nResearch demo',
'input':'Input JSON',
'schema':'`static`: age, male (0/1), ASA, emergency (0/1), weight (kg), height (cm).\n\n`events`: `time_min` = minutes since record start; `token` = vocabulary name; optional `value` = raw measurement.',
'example':'Load example','calibration':'Calibration','dataset':'Dataset','custom':'Custom parameters',
'cal_help':'`p = softmax((logits + bias) / temperature)`\n\nPresets are source-specific. Fit locally for a new source. Only next-event scores are calibrated; waiting time and horizon outputs remain raw.',
'run':'Predict','table':'Top 10 events',
'headers':['#','Event','Raw','Calibrated','Wait (h)','1h (raw)','6h (raw)','24h (raw)'],
'output_help':'Next-event relative probabilities sum to 1; they are not independent concurrent-event risks.',
'full':'Full output','json':'JSON','download':'Download JSON','none':'No calibration','custom_choice':'Custom',
'error':'Invalid input. Check the JSON format, vocabulary tokens, numeric ranges and calibration parameters.'},
'zh': {
'subtitle':'麻醉与手术多时域前瞻性结局预测模型',
'links':'[模型](https://huggingface.co/luan0519/MAOMAO) · [代码](https://github.com/XIAOJIE0519/MAOMAO) · [词表](https://huggingface.co/luan0519/MAOMAO/resolve/main/vocabulary.json)\n\n研究演示',
'input':'输入 JSON',
'schema':'`static`：年龄、男性（0/1）、ASA 分级、急诊（0/1）、体重（千克）、身高（厘米）。\n\n`events`：`time_min` 为记录开始后的分钟数；`token` 为词表名称；`value` 为原始测量值（可省略）。',
'example':'加载示例','calibration':'校准','dataset':'数据来源','custom':'自定义参数',
'cal_help':'`p = softmax((logits + bias) / temperature)`\n\n预设仅对应其数据来源；新来源需本地拟合。仅校准下一事件概率，等待时间与各时域输出保持原始值。',
'run':'预测','table':'概率最高的 10 项事件',
'headers':['序号','事件','原始概率','校准概率','等待（小时）','1小时（原始）','6小时（原始）','24小时（原始）'],
'output_help':'下一事件的相对概率总和为 1，不代表各事件独立发生的并发风险。',
'full':'完整输出','json':'JSON 数据','download':'下载 JSON','none':'不校准','custom_choice':'自定义',
'error':'输入无效，请检查 JSON 格式、词表名称、数值范围及校准参数。'}
}

BASE = dict(item.split('=',1) for item in '''systolic_hypotension=收缩压过低;map_hypotension=平均动脉压过低;systolic_hypertension=收缩压过高;diastolic_hypotension=舒张压过低;diastolic_hypertension=舒张压过高;bradycardia=心动过缓;tachycardia=心动过速;oxygen_desaturation=血氧饱和度下降;hypoxemia=低氧血症;bradypnea=呼吸过缓;tachypnea=呼吸过速;respiratory_distress=呼吸窘迫;low_etco2=呼气末二氧化碳过低;low_minute_ventilation=分钟通气量过低;elevated_peak_airway_pressure=气道峰压升高;elevated_plateau_pressure=气道平台压升高;high_peep_support=高呼气末正压支持;high_oxygen_requirement=高氧需求;hypothermia=低体温;elevated_temperature=体温升高;fever=发热;low_cardiac_index=心脏指数过低;elevated_cvp=中心静脉压升高;elevated_pulmonary_artery_pressure=肺动脉压升高;pulmonary_artery_pressure=肺动脉压升高;hyperglycemia=高血糖;hyperlactatemia=高乳酸血症;hyponatremia=低钠血症;hypernatremia=高钠血症;hypokalemia=低钾血症;hyperkalemia=高钾血症;hypocalcemia=低钙血症;low_ionized_calcium=离子钙过低;hypophosphatemia=低磷血症;hyperphosphatemia=高磷血症;acidemia=酸血症;alkalemia=碱血症;low_bicarbonate=碳酸氢根过低;high_bicarbonate=碳酸氢根过高;base_deficit=碱缺失;base_excess=碱过剩;low_arterial_oxygen=动脉血氧分压过低;low_arterial_oxygen_saturation=动脉血氧饱和度过低;hypocapnia=低碳酸血症;hypercapnia=高碳酸血症;azotemia=氮质血症;anemia=贫血;low_hematocrit=红细胞压积过低;thrombocytopenia=血小板减少;elevated_inr=国际标准化比值升高;prolonged_aptt=活化部分凝血活酶时间延长;low_fibrinogen=纤维蛋白原过低;hypoalbuminemia=低白蛋白血症;hyperbilirubinemia=高胆红素血症;ast_elevation=天冬氨酸氨基转移酶升高;alt_elevation=丙氨酸氨基转移酶升高;leukopenia=白细胞减少;leukocytosis=白细胞增多;crp_elevation=C反应蛋白升高;troponin_i_elevation=肌钙蛋白I升高;ckmb_elevation=肌酸激酶同工酶升高;ck_elevation=肌酸激酶升高;gcs_motor_decline=格拉斯哥运动评分下降;gcs_eye_decline=格拉斯哥睁眼评分下降;peep_support_reduced=呼气末正压支持降低;oxygen_requirement_reduced=氧需求降低;major_bleeding_signal=大出血信号;bleeding_signal=出血信号;rbc_transfusion=红细胞输注;ffp_transfusion=新鲜冰冻血浆输注;platelet_transfusion=血小板输注;cryo_transfusion=冷沉淀输注;vasopressor_start=开始升压药;vasopressor_stop=停止升压药;ventilation_start=开始机械通气;ventilation_stop=停止机械通气;crrt_start=开始连续肾脏替代治疗;crrt_stop=停止连续肾脏替代治疗;ecmo_start=开始体外膜肺氧合;iabp_start=开始主动脉内球囊反搏;bp_recovered_after_vasopressor_observed=观察到升压药使用后血压恢复;hypotension_persistent_after_vasopressor_observed=观察到升压药使用后低血压持续;or_entry=进入手术室;anesthesia_start=麻醉开始;surgery_start=手术开始;cpb_start=体外循环开始;cpb_stop=体外循环停止;surgery_end=手术结束;anesthesia_end=麻醉结束;or_exit=离开手术室;icu_transfer=转入重症监护室;icu_discharge=离开重症监护室;inhospital_death=院内死亡;aki_stage_1_signal=急性肾损伤1期信号;aki_stage_2_signal=急性肾损伤2期信号;aki_stage_3_signal=急性肾损伤3期信号;aki_recovery_signal=急性肾损伤恢复信号'''.split(';'))
NORMAL = dict(zip(
'systolic_bp map diastolic_bp heart_rate oxygenation respiratory_rate etco2 minute_ventilation airway_pressure plateau_pressure temperature cardiac_index cvp pulmonary_artery_pressure glucose lactate sodium potassium calcium ionized_calcium phosphorus ph bicarbonate base_excess arterial_oxygen arterial_oxygen_saturation paco2 bun hemoglobin hematocrit platelet inr aptt fibrinogen albumin bilirubin ast alt wbc crp troponin_i ckmb ck gcs_motor gcs_eye'.split(),
'收缩压 平均动脉压 舒张压 心率 氧合 呼吸频率 呼气末二氧化碳 分钟通气量 气道压力 气道平台压 体温 心脏指数 中心静脉压 肺动脉压 血糖 乳酸 血钠 血钾 血钙 离子钙 血磷 酸碱度 碳酸氢根 碱剩余 动脉血氧分压 动脉血氧饱和度 动脉血二氧化碳分压 尿素氮 血红蛋白 红细胞压积 血小板 国际标准化比值 活化部分凝血活酶时间 纤维蛋白原 白蛋白 胆红素 天冬氨酸氨基转移酶 丙氨酸氨基转移酶 白细胞 C反应蛋白 肌钙蛋白I 肌酸激酶同工酶 肌酸激酶 格拉斯哥运动评分 格拉斯哥睁眼评分'.split()))

def event_name(name, language):
    if language == 'en': return name.replace('_', ' ').capitalize()
    if name in BASE: return BASE[name]
    for prefix, translated in [('severely_','重度'),('severe_','重度'),('profound_','极重度'),('moderate_','中度'),('markedly_','显著'),('marked_','显著'),('very_','极'),('major_','大幅')]:
        if name.startswith(prefix): return translated + event_name(name[len(prefix):], 'zh')
    if name.endswith('_normalized'): return NORMAL[name[:-11]] + '恢复正常'
    if name.startswith('acute_'):
        feature, direction = name[6:].rsplit('_', 1)
        return '急性' + {'sbp':'收缩压','map':'平均动脉压','heart_rate':'心率','hemoglobin':'血红蛋白'}[feature] + {'drop':'下降','rise':'升高'}[direction]
    raise KeyError('Missing event translation: ' + name)

def rows(result, language):
    if not result: return []
    return [[r['rank'],event_name(r['event'],language),round(r['raw'],6),round(r['calibrated'],6),round(r['wait_hours'],3),round(r['risk_1h_raw'],6),round(r['risk_6h_raw'],6),round(r['risk_24h_raw'],6)] for r in result['next_event']]

def choices(presets, language):
    t = TEXT[language]
    names={'mimic':'MIMIC-IV','mover':'MOVER','eicu':'eICU','sicdb':'SICdb','ntuh':'NTUH','asac':'ASAC','uq':'UQ','surgical_pooled':'Pooled' if language=='en' else '合并队列'}
    return [(t['none'],'None')] + [(names[n],n) for n in presets] + [(t['custom_choice'],'Custom')]

def table_html(result, language):
    from html import escape
    t = TEXT[language]
    head = ''.join('<th>' + escape(h) + '</th>' for h in t['headers'])
    body = ''.join('<tr>' + ''.join('<td>' + escape(str(v)) + '</td>' for v in row) + '</tr>' for row in rows(result, language))
    return '<section><h3>' + escape(t['table']) + '</h3><div class="results-scroll"><table class="results-table"><thead><tr>' + head + '</tr></thead><tbody>' + body + '</tbody></table></div></section>'

TEXT['en']['service_error'] = 'Prediction is temporarily unavailable. Check your Hugging Face ZeroGPU quota and try again.'
TEXT['zh']['service_error'] = '预测暂不可用，请检查登录账户的计算额度并稍后重试。'
