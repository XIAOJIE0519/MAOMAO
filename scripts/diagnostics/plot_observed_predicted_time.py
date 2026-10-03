"""Full-row observed/predicted diagnostic styled after the user's reference.
All saved validation rows enter the density and fit. No aesthetic sampling.
Time MAE is reported only by the three observed-time strata.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap,LogNorm
for font in Path('/home/yunkunshi/.local/share/fonts/maomao-arial').glob('*.TTF'):
    font_manager.fontManager.addfont(str(font))
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'outputs/maomao_plot_sources/internal/maomao/time_predictions_no_identifiers.npz'
OUT=ROOT/'outputs/maomao_observed_predicted_time_20261001'
OUT.mkdir(parents=True,exist_ok=True)
z=np.load(SOURCE)
x=z['actual_wait_hours'].astype(float); y=z['predicted_positive_set_average_hours'].astype(float)
valid=np.isfinite(x)&np.isfinite(y)&(x>=0)&(y>=0)
x=x[valid];y=y[valid]; a=np.log1p(x); b=np.log1p(y)
limit=np.log1p(max(x.max(),y.max()))*1.015
edges=np.linspace(0,limit,221);centers=(edges[:-1]+edges[1:])/2
counts,_,_=np.histogram2d(a,b,bins=(edges,edges))
ix,iy=np.nonzero(counts)
slope=np.sum((a-a.mean())*(b-b.mean()))/np.sum((a-a.mean())**2)
intercept=b.mean()-slope*a.mean()
r2=1-np.sum((y-x)**2)/np.sum((x-x.mean())**2)
strata={}
for label,mask in [('0–<2 h',x<2),('2–<24 h',(x>=2)&(x<24)),('≥24 h',x>=24)]:
    strata[label]={'rows':int(mask.sum()),'mae_hours':float(np.mean(np.abs(y[mask]-x[mask])))}
plt.rcParams.update({'font.family':'Arial','svg.fonttype':'none','font.size':8,'axes.titlesize':8,'axes.labelsize':8,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(5.5,5.2))
fig.subplots_adjust(left=.15,right=.97,bottom=.14,top=.88)
cmap=LinearSegmentedColormap.from_list('reference_density',['#526c70','#b8d8df','#f2e9cc','#db9b88','#a84962'])
ax.scatter(centers[ix],centers[iy],c=counts[ix,iy],s=3.5,linewidths=0,cmap=cmap,norm=LogNorm(1,counts.max()),alpha=.7,rasterized=True,zorder=2)
ax.plot([0,limit],[0,limit],color='#6b7478',lw=1.15,ls='--',zorder=3)
grid=np.linspace(0,limit,400)

ticks=np.array([0,1,2,6,24,72,168,720])
ax.set_xticks(np.log1p(ticks),[str(t) for t in ticks]);ax.set_yticks(np.log1p(ticks),[str(t) for t in ticks])
ax.set(xlim=(0,limit),ylim=(0,limit),xlabel='Observed time (h)',ylabel='Predicted time (h)',title='MAOMAO')
ax.set_aspect('equal');ax.spines[['top','right']].set_visible(False)
ax.grid(False)
fig.savefig(OUT/'MAOMAO_observed_vs_predicted_time.png',dpi=300,facecolor='white')
fig.savefig(OUT/'MAOMAO_observed_vs_predicted_time.pdf',facecolor='white')
meta={'model':'MAOMAO','dataset':'INSPIRE full internal patient-disjoint 10% validation','source':str(SOURCE.relative_to(ROOT)),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'source_rows':len(valid),'plotted_rows':len(x),'excluded_rows':int((~valid).sum()),'predicted_definition':'mean predicted waiting time over observed positive event set','axis_transform':'log1p(hours); ticks display hours','density_bins':220,'density_row_count':int(counts.sum()),'fit_displayed':False,'statistics_displayed':False,'font':{'family':'Arial','size_pt':8},'r_squared_hours':float(r2),'strata':strata,'pooled_mae_reported':False,'panel_alignment':'not applicable: single data panel','exports':['png','pdf']}
(OUT/'provenance.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'r2_hours':r2,'strata':strata},ensure_ascii=False))
