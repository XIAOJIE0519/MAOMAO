"""Two independent figures from existing common-case clinical metrics / actual SHAP.
Clinical comparator selected descriptively by maximal saved AUROC, not retrained.
History pooling is support-weighted mean log SHAP, followed by exponentiation.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap,LogNorm
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.diagnostics.maomao_display_family_groups import display_mapping
from scripts.diagnostics.maomao_figure_revision import GROUP_COLORS,SHORT_GROUP
for f in (Path.home()/'.local/share/fonts/maomao-arial').glob('*.TTF'):font_manager.fontManager.addfont(str(f))
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.titlesize':8,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
OUT=ROOT/'outputs/maomao_clinical_dumbbell_and_pooled_shap_20261001';OUT.mkdir(exist_ok=True)
CLIN=ROOT/'outputs/maomao_manuscript_figures_20260929/source_data/clinical_common_metrics.csv'
SHAP=ROOT/'outputs/maomao_plot_sources/shap'
blue='#02539F';rose='#B30337'
titles={'hypotension_60m':'Hypotension · 1 h','hypotension_6h':'Hypotension · 6 h','pressor_1h':'Vasopressor · 1 h','pressor_6h':'Vasopressor · 6 h','icu_24h':'ICU admission · 24 h','crrt_24h':'CRRT · 24 h','ventilation_24h':'Ventilation · 24 h','rbc_6h':'RBC transfusion · 6 h','rbc_24h':'RBC transfusion · 24 h','troponin_elevation_24h':'Troponin elevation · 24 h'}
short={'MAP burden <65':'MAP burden <65','ASA + adapted GS-AKI':'ASA + GS-AKI*','ARISCAT adapted':'ARISCAT*','RCRI proxy':'RCRI*'}
df=pd.read_csv(CLIN);df=df[df.metric=='auroc'];rows=[]
for endpoint in titles:
    subset=df[df.endpoint==endpoint];m=subset[subset.model=='MAOMAO'].iloc[0];c=subset[subset.model!='MAOMAO'].sort_values(['value','model'],ascending=[False,True]).iloc[0]
    assert m.n_episodes==c.n_episodes and m.n_patients==c.n_patients and m.events==c.events
    rows.append({'endpoint':endpoint,'outcome':titles[endpoint],'best_clinical_score':c.model,'clinical_auroc':c.value,'clinical_ci_lower':c.ci_lower,'clinical_ci_upper':c.ci_upper,'maomao_auroc':m.value,'maomao_ci_lower':m.ci_lower,'maomao_ci_upper':m.ci_upper,'delta_auroc':m.value-c.value,'n_episodes':int(m.n_episodes),'n_patients':int(m.n_patients),'events':int(m.events)})
pd.DataFrame(rows).to_csv(OUT/'clinical_best_score_vs_maomao.csv',index=False)
fig,ax=plt.subplots(figsize=(183/25.4,216/25.4));fig.subplots_adjust(left=.34,right=.69,bottom=.17,top=.88)
y=np.arange(10)[::-1]
for yy,r in zip(y,rows):
    ax.plot([r['clinical_auroc'],r['maomao_auroc']],[yy,yy],color='#A8B2BB',lw=1.7,zorder=1)
    ax.scatter(r['clinical_auroc'],yy,color=blue,s=24,zorder=3)
    ax.scatter(r['maomao_auroc'],yy,color=rose,s=24,zorder=3)
    ax.text(1.16,yy,f"{r['clinical_auroc']:.3f}",transform=ax.get_yaxis_transform(),ha='center',va='center')
    ax.text(1.47,yy,f"{r['maomao_auroc']:.3f}",transform=ax.get_yaxis_transform(),ha='center',va='center')
    ax.text(1.78,yy,f"{r['delta_auroc']:+.3f}",transform=ax.get_yaxis_transform(),ha='center',va='center')
ax.set_yticks(y,[r['outcome']+'\n'+short.get(r['best_clinical_score'],r['best_clinical_score']) for r in rows]);ax.tick_params(axis='y',length=0,pad=12)
ax.set(xlim=(.50,1),ylim=(-.6,9.6),xlabel='AUROC (higher is better)');ax.set_xticks([.5,.6,.7,.8,.9,1]);ax.grid(axis='x',color='#E5EAF0',lw=.5)
for x,label in [(1.16,'Best score'),(1.47,'MAOMAO'),(1.78,'Δ AUROC')]:ax.text(x,1.025,label,transform=ax.transAxes,ha='center',va='bottom')
fig.text(.055,.970,'MAOMAO versus the best available clinical score',fontweight='bold',va='top')
fig.legend(handles=[Line2D([],[],color=blue,marker='o',ls='none',label='Best clinical score'),Line2D([],[],color=rose,marker='o',ls='none',label='MAOMAO')],loc='upper left',bbox_to_anchor=(.055,.947),ncol=2,frameon=False)
fig.text(.055,.10,'Comparator = highest observed AUROC among the saved scores for each endpoint.',fontsize=8)
fig.text(.055,.080,'Endpoint-specific common cases; original patient-disjoint 85:15 clinical study.',fontsize=8)
fig.text(.055,.060,'* Adapted scores. RBC endpoints have only 3 / 4 events; estimates are imprecise.',fontsize=8)
fig.text(.055,.040,'Δ AUROC = MAOMAO − best clinical score. Full 95% CIs are supplied in the source CSV.',fontsize=8)
fig.savefig(OUT/'MAOMAO_clinical_AUROC_dumbbell.png',dpi=600);fig.savefig(OUT/'MAOMAO_clinical_AUROC_dumbbell.pdf');plt.close(fig)
# Exactly retain the predictor/output order and clinical strips of Figure 5d.
z=np.load(SHAP/'figure4c_display_matrix.npz');raw=np.load(SHAP/'event_shap_matrices_by_family.npz')
predictors=z['predictor_event_ids'];outputs=z['predicted_event_ids'];support=raw['patients_with_feature'][:,predictors]
log_shap=raw['mean_log_probability_shap'][:,predictors][:,:,outputs]
assert np.array_equal(support,z['predictor_support'])
assert np.allclose(np.exp(log_shap),z['folds'],equal_nan=True)
pooled_log=np.sum(log_shap*support[:,:,None],axis=0)/support.sum(axis=0)[:,None]
folds=np.exp(pooled_log);assert np.isfinite(folds).all()
np.savez_compressed(OUT/'Figure_5d_pooled_source.npz',pooled_mean_log_shap=pooled_log,pooled_folds=folds,predictor_event_ids=predictors,predicted_event_ids=outputs,pooled_support=support.sum(axis=0),support_per_lag=support)
meta=json.loads((SHAP/'event_sequence_meta.json').read_text());mapping,names=display_mapping(meta['outcome_family_vocabulary']);mapping=np.array(mapping);fam=np.array(meta['outcome_to_family'])
rg=mapping[fam[predictors]];cg=mapping[fam[outputs]]
def edges(g):
    e=np.r_[0,np.cumsum([1/np.sum(g==i) for i in g])];u=np.unique(g);c=[(e[np.flatnonzero(g==i)[0]]+e[np.flatnonzero(g==i)[-1]+1])/2 for i in u];return e,u,c
re,ru,rc=edges(rg);ce,cu,cc=edges(cg)
fig=plt.figure(figsize=(210/25.4,225/25.4));ax=fig.add_axes([.22,.28,.71,.64])
cmap=LinearSegmentedColormap.from_list('provided_diverging',[blue,'#BCD7ED','#FFFFFF','#F4E1DE',rose])
mesh=ax.pcolormesh(ce,re,folds,cmap=cmap,norm=LogNorm(.1,10),shading='flat',rasterized=False)
ax.set(xlim=(0,ce[-1]),ylim=(re[-1],0));ax.set_yticks(rc,[SHORT_GROUP[names[i]] for i in ru]);ax.set_xticks(cc,[SHORT_GROUP[names[i]] for i in cu],rotation=90,rotation_mode='anchor',ha='right',va='center');ax.tick_params(length=0,pad=8);ax.tick_params(axis='y',pad=19)
for e in re[np.flatnonzero(np.diff(rg))+1]:ax.axhline(e,color='#8995A1',lw=.25)
for e in ce[np.flatnonzero(np.diff(cg))+1]:ax.axvline(e,color='#8995A1',lw=.25)
for i in cu:
    ix=np.flatnonzero(cg==i);ax.add_patch(Rectangle((ce[ix[0]],-.36),ce[ix[-1]+1]-ce[ix[0]],.25,color=GROUP_COLORS[i],lw=0,clip_on=False))
for i in ru:
    ix=np.flatnonzero(rg==i);ax.add_patch(Rectangle((-.40,re[ix[0]]),.25,re[ix[-1]+1]-re[ix[0]],color=GROUP_COLORS[i],lw=0,clip_on=False))
fig.text(.06,.975,'MAOMAO SHAP | All three history windows pooled',fontsize=8,fontweight='bold',va='top')
fig.text(.06,.951,'<2 h + 2–<24 h + ≥24 h | 174 predictor events × 175 output events',fontsize=8,va='top')
cax=fig.add_axes([.25,.085,.37,.016]);cbar=fig.colorbar(mesh,cax=cax,orientation='horizontal',ticks=[.1,1,10]);cbar.solids.set_rasterized(False);cbar.solids.set_edgecolor('face');cax.set_xticklabels(['0.1','1','10']);cax.tick_params(length=2)
fig.text(.65,.090,'exp(pooled mean log-probability SHAP)',fontsize=8,va='center')
fig.text(.06,.044,'Pooled log SHAP is weighted by each event’s valid patient-window support, then exponentiated.',fontsize=8)
fig.text(.06,.024,'Same Figure 5d coordinates and color scale; repeat appearances across windows contribute separately.',fontsize=8)
fig.savefig(OUT/'MAOMAO_Figure_5d_SHAP_pooled.png',dpi=600);fig.savefig(OUT/'MAOMAO_Figure_5d_SHAP_pooled.pdf');plt.close(fig)
proof={'font':'Arial','font_size_pt':8,'clinical_rows':10,'clinical_comparator_rule':'max saved AUROC per endpoint, descriptive post-hoc selection','clinical_cases':'endpoint-specific common cases, existing 85:15 study','pool_rule':'sum(mean_log_SHAP_lag * support_lag)/sum(support_lag), then exp; no arithmetic mean of folds','history_windows':z['lag_labels'].tolist(),'same_5d_display_coordinates':True,'pool_patient_window_appearances_not_unique_patients':True,'pooled_fold_below_color_min':int((folds<.1).sum()),'pooled_fold_above_color_max':int((folds>10).sum()),'single_data_panel_per_figure_alignment':'not applicable','sources':[]}
for p in [CLIN,SHAP/'figure4c_display_matrix.npz',SHAP/'event_shap_matrices_by_family.npz',SHAP/'event_sequence_meta.json']:
    proof['sources'].append({'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(OUT/'provenance.json').write_text(json.dumps(proof,indent=2)+'\n')
print(OUT)
