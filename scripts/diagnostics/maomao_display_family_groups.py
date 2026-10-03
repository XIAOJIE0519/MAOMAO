"""Clinical display groups only; does not change the model's 63 semantic families."""
GROUPS=(
 ('Haemodynamics',('systolic_bp','map','diastolic_bp','heart_rate','cardiac_index','cvp','pulmonary_artery_pressure','vasopressor_response')),
 ('Oxygenation',('oxygen_saturation','oxygen_requirement','arterial_oxygen','arterial_oxygen_saturation')),
 ('Ventilation',('respiratory_rate','etco2','minute_ventilation','airway_pressure','plateau_pressure','peep_support','paco2','mechanical_ventilation')),
 ('Temperature',('temperature',)),
 ('Metabolism',('glucose','lactate')),
 ('Electrolytes',('sodium','potassium','calcium','ionized_calcium','phosphorus')),
 ('Acid–base',('acid_base','hco3','base_excess')),
 ('Renal',('bun','acute_kidney_injury','crrt')),
 ('Blood / bleeding',('hemoglobin','hematocrit','platelet','bleeding','transfusion')),
 ('Coagulation',('inr','aptt','fibrinogen')),
 ('Liver / albumin',('albumin','bilirubin','ast','alt')),
 ('Inflammation',('wbc','white_blood_cell','crp')),
 ('Cardiac injury',('troponin_i','myocardial_injury','ckmb','ck')),
 ('Neurologic',('gcs_motor','gcs_eye')),
 ('Organ support',('ecmo','iabp')),
 ('Care transitions',('operating_room','anesthesia_phase','surgery_phase','cardiopulmonary_bypass','icu_phase')),
 ('Death',('death',)),
)

def display_mapping(families):
    lookup={family:i for i,(_,group) in enumerate(GROUPS) for family in group}
    if len(lookup)!=sum(len(group) for _,group in GROUPS) or set(lookup)!=set(families):
        raise ValueError('Clinical display mapping must cover every existing semantic family exactly once')
    return [lookup[family] for family in families],[name for name,_ in GROUPS]
