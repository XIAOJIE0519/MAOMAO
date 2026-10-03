#!/usr/bin/env python3
"""Original A4-width MAOMAO manuscript figures from verified result artifacts."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator

HERE=Path(__file__).resolve().parent;PORTABLE=(HERE/'source_data').is_dir()
ROOT=HERE.parent if PORTABLE else Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(HERE) if PORTABLE else '/home/yunkunshi/.codex/skills/nature-figure/scripts')
from audit_panel_alignment import require_matplotlib_panel_alignment
if PORTABLE:
 from maomao_display_family_groups import display_mapping
else:
 from scripts.diagnostics.maomao_display_family_groups import display_mapping
def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for part in iter(lambda:stream.read(8*1024**2),b''):h.update(part)
 return h.hexdigest()

OUT=HERE if PORTABLE else ROOT/'outputs/maomao_manuscript_figures_20260929'
SOURCE=ROOT/'plot_sources' if PORTABLE else ROOT/'outputs/maomao_plot_sources';FIG=OUT/'figures';DATA=OUT/'source_data'
WIDTH=210/25.4
METHODS=('univariate','logistic_regression','xgboost','ann','maomao')
LABELS={'univariate':'Univariate','logistic_regression':'Logistic regression','xgboost':'XGBoost','ann':'ANN','maomao':'MAOMAO'}
COLORS={'univariate':'#8B929A','logistic_regression':'#AD8BB8','xgboost':'#D6A464','ann':'#6D95C2','maomao':'#177B80','after':'#BC4D65'}
SITES=('inspire','mimic','mover','eicu','sicdb','surgical_pooled')
SITE_LABELS={'inspire':'INSPIRE | Internal validation','mimic':'MIMIC-IV','mover':'MOVER','eicu':'eICU','sicdb':'SICdb','surgical_pooled':'Surgical pool | NTUH + ASAC + UQ'}
def read(p):return json.loads(p.read_text())
def setup():
 for p in (Path.home()/'.local/share/fonts/maomao-arial').glob('*.TTF'):fm.fontManager.addfont(str(p))
 for weight in ('normal','bold'):
  p=Path(fm.findfont(fm.FontProperties(family='Arial',weight=weight),fallback_to_default=False))
  if fm.FontProperties(fname=str(p)).get_name()!='Arial':raise RuntimeError('Actual Arial font is required')
 mpl.rcParams.update({'font.family':'Arial','font.sans-serif':['Arial'],'font.size':8,
  'axes.titlesize':10,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
  'legend.fontsize':8,'figure.titlesize':10,'pdf.fonttype':42,'pdf.use14corefonts':False,
  'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,
  'legend.frameon':False,'savefig.facecolor':'white','axes.unicode_minus':False})
 FIG.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
def save(fig,number,axes,ids,row_groups=None,column_groups=None):
 name=f'Figure_{number}' if isinstance(number,int) else str(number);fig.canvas.draw()
 # Every text glyph in this export must use the requested real font and size.
 text_runs=[]
 from matplotlib.text import Text
 for obj in fig.findobj(Text):
  if not obj.get_visible() or not obj.get_text().strip():continue
  if obj.get_fontfamily()!=['Arial'] or min(abs(obj.get_fontsize()-8),abs(obj.get_fontsize()-10))>.001:
   raise RuntimeError(f'Noncontract text font/size: {obj.get_text()} {obj.get_fontfamily()} {obj.get_fontsize()}')
  text_runs.append({'text':obj.get_text(),'size_pt':obj.get_fontsize(),'family':obj.get_fontfamily()[0]})
 if any(a.get_rasterized() for a in fig.findobj() if hasattr(a,'get_rasterized')) or any(ax.images for ax in fig.axes):
  raise RuntimeError('Rasterized marks or bitmap axes are forbidden in strict vector PDFs')
 require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=ids,row_groups=row_groups or [],column_groups=column_groups or [],
  json_out=FIG/f'{name}.alignment.json',strict=True)
 fig.savefig(FIG/f'{name}.pdf',bbox_inches=None)
 fig.savefig(FIG/f'{name}.png',dpi=300,bbox_inches=None)
 (FIG/f'{name}.text_contract.json').write_text(json.dumps(dict(font='Arial',allowed_sizes_pt=[8,10],text_runs=text_runs,
  width_mm=210,height_mm=fig.get_figheight()*25.4,rasterized_artists=0),indent=2)+'\n')
 plt.close(fig);print(f'{name} rendered with genuine Arial 8/10 pt and vector artists',flush=True)
def panel_title(ax,letter,title):ax.set_title(f'{letter}  {title}',loc='left',fontweight='bold',pad=10)
def box(ax,x,y,w,h,title,body='',color='#EDF4E7'):
 p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.015',fc=color,ec='#738187',lw=.65)
 ax.add_patch(p)
 if body:
  ax.text(x+w/2,y+h*.77,title,ha='center',va='center',fontsize=10,fontweight='bold')
  ax.text(x+w/2,y+h*.35,body,ha='center',va='center',fontsize=8,linespacing=1.5)
 else:ax.text(x+w/2,y+h/2,title,ha='center',va='center',fontsize=8)
def arrow(ax,a,b,color='#46565D',style='->',ls='-'):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=8,lw=.75,color=color,linestyle=ls))
def figure1():
 cfg_path=ROOT/'internal/maomao/run_config.json' if PORTABLE else ROOT/'outputs/final_experiment_results_20260923/full_maomao_reference/run_config.json'
 cfg=read(cfg_path)
 if (cfg['hidden_dim'],cfg['num_layers'],cfg['num_heads'])!=(384,10,12):raise RuntimeError('Architecture source changed')
 fig=plt.figure(figsize=(WIDTH,10.6));axes=[]
 bands=[(.055,.725,.91,.23),(.055,.290,.91,.39),(.055,.055,.91,.19)]
 for pos in bands:
  ax=fig.add_axes(pos);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_axis_off();axes.append(ax)
 fig.text(.055,.984,'Figure 1 | MAOMAO learns event content and time from perioperative histories',fontsize=10,fontweight='bold',va='top')
 a,b,c=axes
 a.text(0,.99,'a  Irregular clinical histories and independent evaluation cohorts',fontsize=10,fontweight='bold',va='top')
 box(a,.015,.36,.29,.45,'INSPIRE','89,897 training patients\n14,128,539 target rows\n9,989 validation patients\n1,563,972 target rows')
 box(a,.355,.36,.29,.45,'Observed event stream','Vitals / labs / interventions\nEvent identity + value + time\nStatic patient context',color='#DEEAF7')
 box(a,.705,.36,.28,.45,'External evaluation','MIMIC-IV / MOVER / eICU\nSICdb\nNTUH + ASAC + UQ pool',color='#FCE5D6')
 arrow(a,(.309,.58),(.344,.58));arrow(a,(.649,.58),(.694,.58))
 a.plot([.05,.95],[.16,.16],color='#8A9699',lw=.6)
 for x,label,color in [( .10,'Observed t0','#8B929A'),(.33,'Concurrent t1','#177B80'),(.48,'Concurrent t1','#177B80'),(.71,'Observed t2','#6D95C2'),(.90,'Next event','#BC4D65')]:
  a.scatter([x],[.16],s=20,c=color,zorder=3);a.text(x,.07,label,fontsize=8,ha='center')
 a.text(.015,.27,'Illustrative event ordering; simultaneous outcomes remain a set',fontsize=8)
 b.text(0,.99,'b  MAOMAO architecture',fontsize=10,fontweight='bold',va='top')
 box(b,.015,.40,.23,.48,'Input representation','Token embedding\nSigned-log clinical value\nAbsolute time + elapsed gap\nStatic features + phase\nPre-window family history',color='#DEEAF7')
 box(b,.29,.47,.35,.32,'Causal encoder','10 layers / 12 heads\n384 hidden dimensions\nFeed-forward width 1,536')
 b.text(.30,.845,'[4] Clock / phase-summary tokens',fontsize=8)
 b.text(.30,.405,'[5] Measurement intensity enters the representation',fontsize=8)
 arrow(b,(.25,.65),(.28,.65));arrow(b,(.648,.65),(.70,.65))
 box(b,.71,.66,.275,.22,'[3] Event + family heads','210 event logits + 63 family logits',color='#A9C9B5')
 box(b,.71,.35,.275,.22,'[7] Event-conditioned time','24 fine bins + 44 long bins\nLog-normal long tail',color='#FCE5D6')
 box(b,.71,.075,.275,.18,'Trajectory head','Multiple future horizons',color='#E7D5EF')
 arrow(b,(.649,.58),(.70,.46));arrow(b,(.649,.53),(.70,.165))
 # Diagram, not observed data: current token can attend to itself and older
 # timestamp blocks; simultaneous siblings and future positions are hidden.
 times=np.array([0,0,1,2,2,3]);allowed=(np.arange(6)[None,:]<=np.arange(6)[:,None])&((times[:,None]!=times[None,:])|np.eye(6,dtype=bool))
 for i in range(6):
  for j in range(6):
   b.add_patch(Rectangle((.305+j*.028,.105+(5-i)*.028),.025,.025,fc='#0F73B7' if allowed[i,j] else '#DEEAF7',ec='none'))
 b.text(.015,.22,'[1] Block-causal attention',fontsize=8,va='center')
 b.text(.015,.30,'[2] Learned relative-time bias',fontsize=8,va='center')
 b.text(.29,.045,'Schematic mask: past/self allowed; future and same-time siblings blocked',fontsize=8)
 c.text(0,.99,'c  Joint training, frozen weights and patient-disjoint external calibration',fontsize=10,fontweight='bold',va='top')
 box(c,.015,.49,.97,.30,'Training objective','Next-event set loss + family supervision + observed/censored time loss + trajectory loss\n[6] Masked-event and masked-value auxiliary learning',color='#EDF4E7')
 box(c,.015,.075,.28,.23,'Frozen model weights',color='#DEEAF7')
 box(c,.36,.075,.28,.23,'90% calibration patients',color='#FCE5D6')
 box(c,.705,.075,.28,.23,'10% sealed test patients',color='#A9C9B5')
 arrow(c,(.30,.19),(.35,.19));arrow(c,(.645,.19),(.695,.19))
 c.text(.50,.415,'Within 90%: patient-disjoint 80:20 selection of raw / temperature / temperature + bias',fontsize=8,ha='center')
 c.text(.50,.345,'Refit selected event calibration on all 90% rows; evaluate the sealed 10%',fontsize=8,ha='center')
 fig.text(.055,.016,'Numbers [1–7] identify the seven prespecified module removals in Figure 3. Diagram arrows describe computation, not causal effects.',fontsize=8)
 save(fig,1,axes,['a','b','c'],column_groups=[{'id':'schematic_bands','panels':['a','b','c']}])
 write_sources(1,[cfg_path,OUT/'provenance/event_maomao.py' if PORTABLE else ROOT/'maomao/models/event_maomao.py',ROOT/'internal/maomao/metrics.json' if PORTABLE else ROOT/'outputs/final_experiment_results_20260923/model_metrics/maomao_internal.json',
                  *([DATA/'external_calibration_protocol.json'] if (DATA/'external_calibration_protocol.json').exists() else [])])
def metric_frame():return pd.read_csv(SOURCE/'metrics_long.csv')
def get_point(frame,group,site,model,metric,state=None):
 r=frame[(frame.group==group)&(frame.site==site)&(frame.model==model)&(frame.metric==metric)]
 if state:r=r[r.calibration_state==state]
 if len(r)!=1:raise RuntimeError(f'Expected one source row: {group}/{site}/{model}/{metric}/{state}, got {len(r)}')
 return r.iloc[0]
def draw_point(ax,row,y,color,marker='o',filled=True):
 if np.isfinite(row.ci_lower) and np.isfinite(row.ci_upper):ax.plot([row.ci_lower,row.ci_upper],[y,y],color=color,lw=.9)
 ax.plot(row.value,y,marker=marker,ms=4.5,mfc=color if filled else 'white',mec=color,mew=.8,ls='none')
def write_sources(number,paths,extra=None):
 result=dict(figure=number,font='Arial',font_sizes_pt=[8,10],width_mm=210,pdf_strict_vector=True,
  sources=[dict(path=str(p.relative_to(ROOT)),sha256=sha256(p)) for p in paths],**(extra or {}))
 (DATA/f'figure_{number}_provenance.json').write_text(json.dumps(result,indent=2)+'\n')
CLINICAL_SHORT={'Shock Index':'SI','Modified Shock Index':'MSI','Current MAP':'MAP','MAP burden <65':'MAP<65',
 'SAS':'SAS','ASA-PS':'ASA-PS','NEWS2 physiologic subtotal':'NEWS2*','MEWS physiologic subtotal':'MEWS*',
 'qSOFA 2-component':'qSOFA*','adapted GS-AKI':'GS-AKI*','ASA + adapted GS-AKI':'ASA+GS-AKI*',
 'ARISCAT adapted':'ARISCAT*','EBL':'EBL','Hb drop':'Hb drop','RCRI proxy':'RCRI*'}
MODULE_LABELS={'no_block_causal':'− block-causal','no_relative_time':'− relative-time bias',
 'no_family_head':'− family head','no_clock_phase_summary':'− clock/phase summary',
 'no_measurement_intensity':'− measurement intensity','no_masked_event_value':'− masked event/value loss',
 'no_event_conditioned_time':'− event-conditioned time'}
SCALE_LABELS={'model_small':'Small (256 / 6 layers)','reference':'MAOMAO (384 / 10 layers)',
 'model_large':'Large (512 / 12 layers)','context_64':'Context 64 events','context_128':'Context 128 events'}
def ablation_external_frame():
 if PORTABLE:
  proof=read(OUT/'delivery_verification.json')
  if not proof.get('complete'):raise RuntimeError('Packaged Figure 3 has not passed full delivery verification')
  return pd.read_csv(DATA/'external_ablation_metrics_long.csv'),[DATA/'external_ablation_metrics_long.csv']
 queue=read(OUT/'external_ablations/queue_status.json')
 if queue['status']!='completed' or len(queue['completed'])!=75:
  raise RuntimeError('Figure 3 awaits all 75 real full-row external evaluation jobs')
 from scripts.diagnostics.verify_manuscript_external_ablation_rows import verify as verify_rows
 verify_rows(require_complete=True)
 row_proof=OUT/'external_ablation_row_verification.json'
 audited=read(row_proof)
 rows=[];paths=[row_proof,*[DATA/f'external_row_audits/{site}.json' for site in audited['source_row_audit_files']]]
 metric_files=[]
 from scripts.diagnostics.run_manuscript_external_ablations import checkpoint,VERSION
 for p in sorted((OUT/'external_ablations').glob('*/*/*/metrics_*.json')):
  a=read(p);paths.append(p);metric_files.append(p);original='reference_projected' if a['model'].startswith('reference_vocab_') else a['model']
  manifest=ROOT/f'outputs/final_experiment_results_20260923/classical_full_scale/external/{a["site"]}/manifest.json'
  if (a['status']!='completed' or a['protocol']!=VERSION or a['patient_overlap'] or
      a['model_sha256']!=sha256(checkpoint(original)) or a['row_manifest_sha256']!=sha256(manifest) or
      not a['all_original_rows_scored'] or not a['all_eligible_test_rows_evaluated'] or
      a['test_labels_used_for_fit_or_selection'] or not a['before_after_on_identical_rows']):
   raise RuntimeError(f'External ablation provenance or complete-row evidence failed: {p}')
  for metric in ('micro_auprc','macro_auprc','micro_auroc','macro_auroc','mrr','brier','ece','hit_at_1','recall_at_5','recall_at_10'):
   ci=a[metric+'_95ci']
   if not np.isfinite([a[metric],*ci]).all():raise RuntimeError(f'Missing finite CI: {p}/{metric}')
   rows.append(dict(group='external_ablation',site=a['site'],model=a['model'],calibration_state=a['calibration_state'],
    metric=metric,value=a[metric],ci_lower=ci[0],ci_upper=ci[1],rows=a['test_rows'],original_rows=a['test_rows_original'],
    calibration_rows=a['calibration_rows'],output_classes=a['output_classes'],model_sha256=a['model_sha256'],
    source_file=str(p.relative_to(ROOT)),source_sha256=sha256(p),ci_method=a['ci_method']))
 if len(metric_files)!=170:raise RuntimeError(f'Expected 170 full before/after reports, found {len(metric_files)}')
 frame=pd.DataFrame(rows)
 for (site,model),part in frame.groupby(['site','model']):
  first=part[part.calibration_state=='before'];second=part[part.calibration_state=='after']
  if len(first)!=10 or len(second)!=10 or first.rows.iloc[0]!=second.rows.iloc[0]:raise RuntimeError('Calibration pair task mismatch')
 frame.to_csv(DATA/'external_ablation_metrics_long.csv',index=False)
 return frame,paths
ENDPOINT_TITLES={'hypotension_60m':'Hypotension | 1 h','hypotension_6h':'Hypotension | 6 h',
 'pressor_1h':'Vasopressor | 1 h','pressor_6h':'Vasopressor | 6 h','icu_24h':'ICU admission | 24 h',
 'crrt_24h':'CRRT | 24 h','ventilation_24h':'Ventilation | 24 h','rbc_6h':'RBC transfusion | 6 h',
 'rbc_24h':'RBC transfusion | 24 h','troponin_elevation_24h':'Troponin elevation | 24 h'}
sys.path.insert(0,str(HERE))
import maomao_figure_revision
maomao_figure_revision.install(sys.modules[__name__])

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--figures',nargs='+',type=int,default=[1,2]);args=parser.parse_args();setup()
 for number in args.figures:globals()[f'figure{number}']()
if __name__=='__main__':main()
