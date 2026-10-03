#!/usr/bin/env python3
"""Recount small source cohort tables and save aggregate, identifier-free flow evidence."""
import sys
import json
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.diagnostics.build_fullscale_external_rows import SOURCES
from scripts.diagnostics.uniform_result_scope import sha256
from data.preprocess_mimic_validation import OR_SENT,OR_RECEIVED
from data.external_validation_common import to_datetime
from maomao.data.preprocess_timeline import stream

OUT=ROOT/"outputs/maomao_plot_sources/data_cleaning"
FLOW=[]
EVIDENCE={}


def stage(site,name,frame,person,previous=None):
    row=dict(site=site,stage=name,records=len(frame),unique_person_ids=int(frame[person].nunique(dropna=True)),
             missing_person_id_rows=int(frame[person].isna().sum()),independent_people_linkage=site not in ("ntuh","asac","uq","surgical_pooled"))
    if previous is not None:
        row.update(records_removed_previous_stage=len(previous)-len(frame),
                   person_ids_no_longer_present=int(len(set(previous[person].dropna())-set(frame[person].dropna()))))
    FLOW.append(row)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    # The released INSPIRE files have recorded trailing non-gzip bytes;
    # use the same audited reader as preprocessing, without truncating rows.
    ops=pd.concat(stream(ROOT/"data/train_data/operations.csv.gz"),ignore_index=True)
    stage("inspire","released_operations_input",ops,"subject_id")
    final=pd.read_csv(ROOT/"data/perioperative_event_sequences_v5_richctx_static7/admissions.csv",usecols=["subject_id","op_id"])
    stage("inspire","final_operation_episodes",final,"subject_id",ops)
    del ops,final

    root=ROOT/"data/MOVER"
    f=pd.read_csv(root/"extracted/EPIC_EMR/EMR/patient_information.csv",low_memory=False)
    stage("mover","released_patient_information",f,"MRN")
    for c in ("HOSP_ADMSN_TIME","HOSP_DISCH_TIME","IN_OR_DTTM","OUT_OR_DTTM","AN_START_DATETIME","AN_STOP_DATETIME"):
        f[c]=to_datetime(f[c])
    p=f;f=f.dropna(subset=["LOG_ID","MRN","IN_OR_DTTM","OUT_OR_DTTM"]);stage("mover","required_identifiers_and_or_times",f,"MRN",p)
    p=f;f=f[(f.OUT_OR_DTTM>f.IN_OR_DTTM)&(f.OUT_OR_DTTM<=f.IN_OR_DTTM+pd.Timedelta(hours=24))];stage("mover","positive_or_duration_at_most_24h",f,"MRN",p)
    p=f;f=f.sort_values(["LOG_ID","IN_OR_DTTM"]).drop_duplicates("LOG_ID",keep="first").copy();stage("mover","unique_operation_log_id",f,"MRN",p)
    f["episode_start"]=pd.concat([f.HOSP_ADMSN_TIME,f.IN_OR_DTTM-pd.Timedelta(days=7)],axis=1).max(axis=1)
    opend=pd.concat([f.OUT_OR_DTTM,f.AN_STOP_DATETIME],axis=1).max(axis=1)
    f["episode_end"]=pd.concat([f.HOSP_DISCH_TIME,opend+pd.Timedelta(days=30)],axis=1).min(axis=1)
    p=f;f=f[f.episode_start.notna()&f.episode_end.notna()&f.IN_OR_DTTM.ge(f.episode_start)&f.OUT_OR_DTTM.le(f.episode_end)];stage("mover","valid_bounded_episode_final",f,"MRN",p)
    if len(f)!=57764:raise RuntimeError("MOVER reconstruction mismatch")
    EVIDENCE["mover_age_field"]={"source_column":"BIRTH_DATE", "numeric_available_rows":int(pd.to_numeric(f.BIRTH_DATE,errors="coerce").notna().sum()),"source_dtype":str(f.BIRTH_DATE.dtype),"warning":"adapter interprets numeric BIRTH_DATE as age; source semantics require separate provenance confirmation"}
    del f,p

    root=ROOT/"data/eicu数据库/EICU 2.0数据"
    f=pd.read_csv(root/"patient.csv.gz",low_memory=False)
    stage("eicu","released_patient_icu_stays",f,"uniquepid")
    p=f;f=f.drop_duplicates("patientunitstayid",keep="first");stage("eicu","unique_icu_stay_id",f,"uniquepid",p)
    dx=pd.read_csv(root/"admissionDx.csv.gz",usecols=["patientunitstayid","admitdxpath"])
    t=dx.admitdxpath.fillna("").astype("string").str.lower()
    eligible=t.str.contains(r"admission diagnosis\|all diagnosis\|operative\|diagnosis|admission diagnosis\|was the patient admitted from the o\.r\. or went to the o\.r\. within 4 hours of admission\?\|yes",regex=True,na=False)
    ids=set(dx.loc[eligible,"patientunitstayid"])
    p=f;f=f[f.patientunitstayid.isin(ids)];stage("eicu","explicit_operative_admission",f,"uniquepid",p)
    EVIDENCE["eicu_operative_ids_missing_patient_table"]=int(len(ids-set(p.patientunitstayid)))
    p=f;f=f[pd.to_numeric(f.unitdischargeoffset,errors="coerce").gt(0)];stage("eicu","positive_known_icu_duration_final",f,"uniquepid",p)
    if len(f)!=34196:raise RuntimeError("eICU reconstruction mismatch")
    del f,p,dx

    root=ROOT/"data/SICdb数据库/extracted/salzburg-intensive-care-database-sicdb-a-freely-accessible-intensive-care-database-1.0.8"
    f=pd.read_csv(root/"cases.csv.gz",low_memory=False)
    refs=pd.read_csv(root/"d_references.csv.gz").set_index("ReferenceGlobalID").ReferenceValue.to_dict()
    stage("sicdb","released_cases",f,"PatientID")
    p=f;f=f[f.SurgicalAdmissionType.map(refs).isin(["Elective Surgery","Urgent Surgery"])];stage("sicdb","elective_or_urgent_surgery",f,"PatientID",p)
    offset=pd.to_numeric(f.ICUOffset,errors="coerce")/60
    end=np.minimum(pd.to_numeric(f.TimeOfStay,errors="coerce")/60,offset+43200)
    p=f;f=f[(offset-10080).clip(lower=0).notna()&end.gt(offset)];stage("sicdb","valid_icu_anchored_window_final",f,"PatientID",p)
    if len(f)!=13014:raise RuntimeError("SICdb reconstruction mismatch")
    del f,p

    root=ROOT/"data/mimic-iv-3.1"
    patients=pd.read_csv(root/"hosp/patients.csv.gz",usecols=["subject_id"])
    stage("mimic","released_patient_table",patients,"subject_id")
    admissions=pd.read_csv(root/"hosp/admissions.csv.gz",usecols=["subject_id","hadm_id","admittime","dischtime","deathtime"])
    stage("mimic","released_hospital_admissions",admissions,"subject_id")
    stays=pd.read_csv(root/"icu/icustays.csv.gz",usecols=["subject_id","hadm_id","stay_id"])
    stage("mimic","released_icu_stays",stays,"subject_id")
    events=[];raw_event_count=invalid_time=0
    for chunk in pd.read_csv(root/"icu/procedureevents.csv.gz",usecols=["subject_id","hadm_id","stay_id","starttime","itemid"],chunksize=500000):
        raw_event_count+=len(chunk)
        f=chunk[chunk.itemid.isin([OR_SENT,OR_RECEIVED])].copy()
        f["starttime"]=to_datetime(f.starttime)
        invalid_time+=int(f.starttime.isna().sum())
        events.append(f.dropna(subset=["starttime"]))
    events=pd.concat(events,ignore_index=True)
    sent=events[events.itemid.eq(OR_SENT)]
    stage("mimic","valid_or_sent_anchors",sent,"subject_id")
    pairs=[];unmatched=0
    for stay,g in events.sort_values("starttime").groupby("stay_id",sort=True):
        received=list(g.loc[g.itemid.eq(OR_RECEIVED),"starttime"]);used=set();identity=g.iloc[0]
        for start in g.loc[g.itemid.eq(OR_SENT),"starttime"]:
            candidates=[(i,end) for i,end in enumerate(received) if i not in used and start<end<=start+pd.Timedelta(hours=24)]
            if not candidates:unmatched+=1;continue
            i,end=candidates[0];used.add(i)
            pairs.append(dict(subject_id=int(identity.subject_id),hadm_id=int(identity.hadm_id),stay_id=int(stay),or_in=start,or_out=end))
    f=pd.DataFrame(pairs);stage("mimic","paired_or_return_within_24h",f,"subject_id",sent)
    for c in ("admittime","dischtime","deathtime"):admissions[c]=to_datetime(admissions[c])
    p=f;f=f.merge(admissions,on=["subject_id","hadm_id"],validate="many_to_one").merge(patients,on="subject_id",validate="many_to_one").merge(stays[["stay_id"]],on="stay_id",validate="many_to_one")
    stage("mimic","linked_admission_patient_icu",f,"subject_id",p)
    begin=pd.concat([f.admittime,f.or_in-pd.Timedelta(days=7)],axis=1).max(axis=1)
    terminal=pd.concat([f.dischtime,f.deathtime],axis=1).min(axis=1)
    end=pd.concat([terminal,f.or_out+pd.Timedelta(days=30)],axis=1).min(axis=1)
    p=f;f=f[begin.notna()&end.notna()&f.or_in.ge(begin)&f.or_out.le(end)];stage("mimic","valid_bounded_episode_final",f,"subject_id",p)
    if len(f)!=5413:raise RuntimeError("MIMIC reconstruction mismatch")
    EVIDENCE["mimic_or_pairing"]={"raw_procedureevents_rows":raw_event_count,"or_anchor_invalid_time_rows":invalid_time,"unmatched_or_sent":unmatched,"or_received_rows":int(events.itemid.eq(OR_RECEIVED).sum())}

    datasets={"inspire":"data/perioperative_event_sequences_v5_richctx_static7",**SOURCES}
    for site,path in datasets.items():
        source=ROOT/path
        meta=json.loads((source/"event_sequence_meta.json").read_text())
        final=pd.read_csv(source/"admissions.csv",usecols=["subject_id"])
        entry={"dataset":path,"metadata_sha256":sha256(source/"event_sequence_meta.json"),
               "admissions_sha256":sha256(source/"admissions.csv"),"final_records":len(final),
               "final_unique_person_ids":int(final.subject_id.nunique()),"tokens":meta["num_tokens"],
               "source_audit":meta.get("source_audit",{})}
        if site in ("ntuh","asac","uq"):
            manifest=meta["source_manifest"]
            entry.update(input_records=len(manifest),exclusion_reasons=dict(Counter(str(i.get("reason",i.get("status"))) for i in manifest if i["status"]!="included")),
                         status_counts=dict(Counter(i["status"] for i in manifest)),patient_ids_are_record_proxies=True)
            FLOW.append(dict(site=site,stage="released_record_manifest",records=len(manifest),unique_person_ids=None,independent_people_linkage=False))
            stage(site,"final_surgical_record_proxies",final,"subject_id")
            FLOW[-1]["records_removed_previous_stage"]=len(manifest)-len(final)
        if site=="surgical_pooled":
            FLOW.append(dict(site=site,stage="union_of_released_source_records",records=176,unique_person_ids=None,independent_people_linkage=False))
            stage(site,"included_source_record_proxies",final,"subject_id")
            FLOW[-1]["records_removed_previous_stage"]=176-len(final)
        if site!="inspire":
            split=json.loads((ROOT/"outputs/final_experiment_results_20260923/classical_full_scale/external"/site/"manifest.json").read_text())
            entry.update(calibration_target_rows=split["calibration_target_rows"],test_target_rows=split["test_target_rows"],split=split["split"])
        EVIDENCE[site]=entry
    (OUT/"flow_evidence.json").write_text(json.dumps({"stages":FLOW,"evidence":EVIDENCE,"no_patient_identifiers_exported":True},ensure_ascii=False,indent=2,default=str)+"\n")
    pd.DataFrame(FLOW).to_csv(OUT/"cohort_flow.csv",index=False)
    for row in FLOW:print(row)


if __name__=="__main__":main()
