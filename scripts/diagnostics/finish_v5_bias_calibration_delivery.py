#!/usr/bin/env python3
"""Observe the existing evaluator, promote, refresh, and stop for visual QA.

Does not launch or duplicate training/evaluation. Does not fabricate a human
inspection receipt or claim a final verified ZIP before visual inspection.
"""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.diagnostics.uniform_result_scope import sha256
REV=ROOT/'outputs/calibration_v5_bias_revision_20260930'
WORK=ROOT/'outputs/maomao_manuscript_figures_20260929'
PLOTS=ROOT/'outputs/maomao_plot_sources'

def read(p):return json.loads(p.read_text())
def write(p,x):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False
def run(name,*args):
    print(f'RUN {name} {" ".join(args)}',flush=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/diagnostics'/name),*args],cwd=ROOT,check=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker-pid',type=int,required=True);a=p.parse_args()
    with (REV/'delivery.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status=dict(status='waiting_for_existing_evaluator',pid=os.getpid(),worker_pid=a.worker_pid,started_utc=datetime.now(timezone.utc).isoformat())
        path=REV/'delivery_status.json';write(path,status)
        while True:
            q=read(REV/'queue_status.json')
            if q['status']=='evaluations_completed_awaiting_verified_promotion' and not alive(a.worker_pid):break
            if q['status']=='completed' and q.get('promoted'):break
            if q['status']=='failed' or not alive(a.worker_pid):raise RuntimeError('Evaluator stopped; audit its failure and resume missing work only')
            time.sleep(30)
        run('promote_v5_bias_calibration_revision.py')
        status.update(status='refreshing_current_full_row_sources_and_reports');write(path,status)
        run('complete_external_retrieval.py')
        run('verify_external_retrieval_intervals.py')
        run('build_plot_tables_and_audit.py')
        run('verify_plot_delivery.py')
        run('build_manuscript_external_ablation_report.py')
        active=read(ROOT/'outputs/external_validation_final_maomao_uniform/active_calibration_protocol.json')
        write(WORK/'source_data/external_calibration_protocol.json',dict(**active,
            definition='V5 raw / temperature / temperature+bias; patient-disjoint 80:20 inside 90%, full90 refit; full sealed10 evaluation',
            prediction='softmax((logits + class_bias) / temperature)',model_weights_frozen=True,test_labels_used_for_fit_or_selection=False))
        doc=ROOT/'docs/MAOMAO_MANUSCRIPT_FIGURES.md';text=doc.read_text()
        text=text.replace('c/d 保留原 63-family Hit@10 和 normalized-target Brier 热图的全部数值。','c/d 保留 63-family Hit@10 和 normalized-target Brier 的定义与布局，全部数值更新为当前 V5 校准结果。')
        text=text.replace('标量温度只拟合另 90%。','在另90%内部按患者/记录代理80:20选择原始/温度/温度＋类别偏置，选定方法再使用全部90%有效行重新拟合。')
        text+='\n2026-09-30 外部事件校准恢复 V5 温度＋类别偏置；所有五模型及外部消融变体使用同一校准协议。Figure 1 方法、Figure 2 全部点/CI与热图、Supplementary S3 外部配对点均更新。独立临床风险映射/DCA与三时间尺度定义保持其已验证口径；详见 MAOMAO_CALIBRATION_AUDIT.md。\n'
        doc.write_text(text)
        doc=PLOTS/'README.md';text=doc.read_text()
        text=text.replace('每来源/模型温度、是否采用、90%校准Brier','每来源/模型温度、方法族、偏置范围、是否采用、90%校准Brier')
        text=text.replace('旧错误校准与修正校准的审计数值','当前原始与V5温度＋偏置校准的审计数值')
        text=text.replace('同一封存测试；审计，不用于选温度','同一完整封存测试；审计，不用于选择校准方法或参数')
        text=text.replace('不能将排序指标的全部差异归因于温度。','当前V5类别偏置也能改变事件排序；当前统计口径在前后状态完全一致。')
        text=text.replace('SVG/PNG','PDF/PNG')
        text+='\n当前外部校准协议：'+active['protocol']+'。在外部90%内按患者/记录代理80:20选择原始、仅温度、温度＋类别偏置，随后在全部90%有效行重新拟合，原10%全行评估。所有类别偏置和选择/优化记录在相邻external/与calibration_revision/。外部三份示例图与主图/S3已同步当前结果；时间和独立临床研究的定义不变。\n'
        doc.write_text(text)
        run('build_manuscript_figures.py','--figures','1','2','3')
        subprocess.run([sys.executable,str(PLOTS/'plot_examples.py'),'--output',str(PLOTS/'example_figures'),'--external-only'],cwd=ROOT,check=True)
        review=read(WORK/'source_preflight_review.json')
        review['source_files']={name:sha256(ROOT/name) for name in review['source_files']}
        write(WORK/'source_preflight_review.json',review)
        run('verify_manuscript_figures.py','--figures','1','2','3','4','5')
        run('build_uniform_results_report.py')
        status.update(status='awaiting_actual_visual_inspection_then_final_zip_and_portable_replot',
            changed_manuscript_figures=['Figure_1','Figure_2','Figure_3','Supplementary_S3_external'],
            changed_example_figures=['external_maomao_brier','mimic_micro_roc_pr','mover_maomao_reliability'],
            finished_utc=datetime.now(timezone.utc).isoformat());write(path,status)
        print('Current source/report renders ready. Actual visual inspection and final ZIP verification are still required.',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(REV/'delivery_status.json',dict(status='failed',error=repr(error),finished_utc=datetime.now(timezone.utc).isoformat()));raise
