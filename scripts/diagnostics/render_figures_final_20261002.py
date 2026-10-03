"""User-directed composite revision. All numeric evidence comes from saved artifacts.
Delivery is flat PNG/PDF only; provenance and QA live outside the delivery folder.
Differences are descriptive unless existing paired inference is available.
"""
import sys,json,hashlib,textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch,Rectangle,ConnectionPatch,Polygon
from matplotlib.colors import LinearSegmentedColormap,LogNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.text import Text
from matplotlib.collections import PathCollection,QuadMesh
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import build_manuscript_figures as B
import maomao_figure_revision as R
from audit_panel_alignment import require_matplotlib_panel_alignment
ROOT=B.ROOT;OUT=ROOT/'outputs/maomao_manuscript_figures_20260929/figures';QA=ROOT/'outputs/figure_revision_20261002'
OUT.mkdir(exist_ok=True);QA.mkdir(exist_ok=True);records=[]

def save(fig,name,axes,ids,row_groups=None,column_groups=None):
    name=f'Figure_{name}' if isinstance(name,int) else name
    if name=='Figure_3':adapt_three(fig,axes,ids)
    if name=='Supplementary_S3_external':
        adapt_s3(fig,axes,ids)
        for t in fig.findobj(Text):
            t.set_text(t.get_text().replace('Supplementary Figure S3 |','Supplementary Figure S1 |'))
        name='Supplementary_S1'
    # Only explicitly marked dense SHAP dot collections may become raster images.
    raster_layers=[]
    for ax in fig.axes:
        for artist in ax.get_children():
            is_shap_dots=isinstance(artist,PathCollection) and artist.get_gid()=='dense_shap_points'
            artist.set_rasterized(is_shap_dots)
            if is_shap_dots:raster_layers.append({'axes_index':fig.axes.index(ax),'kind':'SHAP dots only','points':len(artist.get_offsets())})
    fig.canvas.draw()
    texts=[]
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip():
            if t.get_fontfamily()!=['Arial'] or t.get_fontsize() not in (8,10):raise ValueError((t.get_text(),t.get_fontfamily(),t.get_fontsize()))
            texts.append({'text':t.get_text(),'size':t.get_fontsize(),'font':'Arial'})
    require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=ids,row_groups=row_groups or [],column_groups=column_groups or [],json_out=QA/f'{name}.alignment.json',strict=True)
    fig.savefig(OUT/f'{name}.pdf',dpi=600);fig.savefig(OUT/f'{name}.png',dpi=600)
    (QA/f'{name}.raster_layers.json').write_text(json.dumps({'allowed_raster_layers':raster_layers,'all_other_artists_vector':True,'pdf_raster_dpi':600},indent=2)+'\n')
    (QA/f'{name}.text.json').write_text(json.dumps({'font':'Arial','sizes':[8,10],'width_mm':fig.get_figwidth()*25.4,'height_mm':fig.get_figheight()*25.4,'texts':texts},indent=2)+'\n')
    plt.close(fig);print(name+' saved',flush=True)

def heading(fig,title,sub=None):
    fig.text(.055,.982,title,fontsize=10,fontweight='bold',va='top')
    if sub:fig.text(.055,.959,sub,fontsize=8,va='top')

def figure2():
    f=B.metric_frame();ret=pd.read_csv(B.SOURCE/'external_retrieval_metrics.csv');h=11.7
    fig=plt.figure(figsize=(B.WIDTH,h));axes=[];ids=[]
    heading(fig,'Figure 2 | Five-model comparison across six cohorts','Full validation / sealed test rows; error bars: existing 95% CIs; dashed reference: MAOMAO final state')
    handles=[Patch(facecolor=R.METHOD_COLORS[m],label=B.LABELS[m]) for m in B.METHODS[:-1]]+[Patch(facecolor=R.ROSE,alpha=.4,label='MAOMAO raw'),Patch(facecolor=R.ROSE,label='MAOMAO calibrated')]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.05,.944),ncol=3,fontsize=8,frameon=False,columnspacing=1.2)
    for si,site in enumerate(B.SITES):
        row=si//2;slot=si%2
        for mi,metric in enumerate(('micro_auroc','micro_auprc')):
            col=slot*2+mi;bottom=(8.75-row*2.30)/h
            ax=fig.add_axes([.077+col*.233,bottom,.166,1.07/h]);axes.append(ax);pid=f'{site}_{metric}';ids.append(pid)
            state='raw' if site=='inspire' else 'after';group='internal' if site=='inspire' else 'external'
            specs=[(m,state) for m in B.METHODS[:-1]]+[('maomao','raw' if site=='inspire' else 'before')]
            if site!='inspire':specs.append(('maomao','after'))
            vals=[B.get_point(f,group,site,m,metric,s) for m,s in specs];ref=vals[-1].value
            for i,((m,s),r) in enumerate(zip(specs,vals)):
                ax.bar(i,r.value,width=.64,color=R.METHOD_COLORS[m],alpha=.40 if m=='maomao' and s=='before' else .85,edgecolor=R.METHOD_COLORS[m],lw=.55,zorder=3)
                ax.errorbar(i,r.value,yerr=np.maximum(0,[[r.value-r.ci_lower],[r.ci_upper-r.value]]),fmt='none',ecolor='#333333',elinewidth=.55,capsize=1.4,zorder=4)
                records.append({'figure':'2','site':site,'metric':metric,'model':m,'state':s,'value':r.value,'ci_lower':r.ci_lower,'ci_upper':r.ci_upper,'delta_vs_maomao_final':r.value-ref,'inference':'descriptive difference; paired full-row p unavailable'})
            ax.axhline(ref,color=R.ROSE,ls='--',lw=.65,zorder=2)
            high=max(r.ci_upper for r in vals);ax.set_ylim(0,max(high*1.32,.10));ax.set_xlim(-.6,len(vals)-.4)
            ax.yaxis.set_major_locator(MaxNLocator(3));ax.set_xticks(range(len(vals)),['Uni','LR','XGB','ANN']+(['MAO'] if site=='inspire' else ['Raw','Cal']),rotation=90,ha='center')
            ax.set_title(f'{chr(97+si*2+mi)}  '+('INSPIRE' if site=='inspire' else {'mimic':'MIMIC-IV','mover':'MOVER','eicu':'eICU','sicdb':'SICdb','surgical_pooled':'Pooled'}[site]),loc='left',fontsize=8,fontweight='bold',pad=9)
            ax.set_ylabel('AUROC' if mi==0 else 'AUPRC',labelpad=2);ax.tick_params(labelsize=8,pad=2)
    cohorts=B.SITES;states=[(m,'after') for m in B.METHODS[:-1]]+[('maomao','before'),('maomao','after')]
    internal=B.read(QA/'internal_family_retrieval/internal_family_hits.json');heat_records=[]
    names=['Univariate','Logistic','XGBoost','ANN','MAOMAO raw','MAOMAO cal']
    for j,metric in enumerate(('same_family_hit_at_1','same_family_hit_at_5','same_family_hit_at_10')):
        ax=fig.add_axes([.18+j*.275,1.37/h,.235,1.35/h]);axes.append(ax);pid=f'heat_{j}';ids.append(pid)
        matrix=np.full((6,6),np.nan)
        for ci,site in enumerate(cohorts):
            for ri,(m,s) in enumerate(states):
                if site=='inspire':
                    if m=='maomao' and s=='after':continue
                    matrix[ri,ci]=internal[m][metric]
                else:
                    q=ret[(ret.site==site)&(ret.model==m)&(ret.calibration_state==s)&(ret.metric==metric)]
                    if len(q)!=1:raise ValueError('Incomplete family heatmap')
                    matrix[ri,ci]=q.iloc[0].value
                heat_records.append({'site':site,'model':m,'state':'raw' if site=='inspire' else s,'metric':metric,'value':matrix[ri,ci]})
        cmap=LinearSegmentedColormap.from_list('kept_blue',['#DEEAF7',R.BLUE]);cmap.set_bad('white');ax.pcolormesh(np.arange(7),np.arange(7),np.ma.masked_invalid(matrix[::-1]),cmap=cmap,vmin=0,vmax=1,shading='flat')
        for ri in range(6):
            for ci in range(6):
                value=matrix[ri,ci]
                if np.isfinite(value):
                    ax.text(ci+.5,5-ri+.5,f'{value:.2f}',ha='center',va='center',fontsize=8,
                            color='white' if value>.62 else '#20262C')
        ax.set_xticks(np.arange(6)+.5,['INSPIRE','MIMIC','MOVER','eICU','SICdb','Pool'],rotation=90,ha='center');ax.set_yticks(np.arange(6)+.5,names[::-1] if j==0 else ['']*6);ax.tick_params(length=0,pad=3)
        ax.set_title(f'{chr(109+j)}  63-family Hit@{[1,5,10][j]}',loc='left',fontsize=8,fontweight='bold',pad=9)
    fig.text(.055,.034,'Heatmap shading: 63-family Hit, 0 (light) to 1 (dark); INSPIRE calibrated MAOMAO is intentionally blank.',fontsize=8)
    fig.text(.055,.015,'Dashed bar references: raw MAOMAO for INSPIRE, calibrated MAOMAO externally; cell numbers are Hit values.',fontsize=8)
    pd.DataFrame(heat_records).to_csv(QA/'figure2_family_heatmap_values.csv',index=False)
    save(fig,2,axes,ids,row_groups=[{'id':f'bars{r}','panels':[f'{s}_{m}' for s in B.SITES[r*2:r*2+2] for m in ('micro_auroc','micro_auprc')]} for r in range(3)]+[{'id':'heatmaps','panels':[f'heat_{j}' for j in range(3)]}],column_groups=[{'id':f'col{c}','panels':[f'{B.SITES[r*2+c//2]}_{("micro_auroc","micro_auprc")[c%2]}' for r in range(3)]} for c in range(4)])

def adapt_three(fig,axes,ids):
    stats=pd.read_csv(B.DATA/'figure_3_box_statistics.csv') if False else None
    # Calculate directly from the same source used for existing box artists.
    pe=pd.read_csv(B.SOURCE/'per_event_metrics.csv')
    groups={'a':['maomao',*B.MODULE_LABELS],'d':['model_small','reference','model_large','context_64','context_128'],'g':[f'{a}_{k}' for k in (50,100,150) for a in ('vocab','reference_vocab')]+['maomao']}
    for ax,pid in zip(axes,ids):
        if pid.startswith(('a_','d_','g_')):
            block,metric=pid.split('_');models=groups[block];vectors=[]
            for m in models:
                group='internal' if m=='maomao' else ('module_ablations' if m in B.MODULE_LABELS else 'scale_ablations')
                v=pe[(pe.group==group)&(pe.site=='inspire')&(pe.model==m)&(pe.calibration_state=='raw')][metric].to_numpy();vectors.append(v[np.isfinite(v)])
            whisk=[R._box_stats(v) for v in vectors];lo=min(s['whislo'] for s in whisk);hi=max(s['whishi'] for s in whisk);pad=max((hi-lo)*.10,.01)
            ax.set_xlim(max(0,lo-pad),min(1,hi+pad));ax.xaxis.set_major_locator(MaxNLocator(3))
            for yy,m,v in zip(np.arange(len(models))[::-1],models,vectors):
                ref='maomao' if block=='a' or m=='maomao' else 'reference'
                if block=='g' and m.startswith(('vocab_','reference_vocab_')):ref='reference_vocab_'+m.split('_')[-1]
                rv=vectors[models.index(ref)]
                delta=float(np.median(v)-np.median(rv))
                ax.text(1.025,yy,f'{delta:+.3f}',transform=ax.get_yaxis_transform(),va='center',fontsize=8,color=R.ROSE if m==ref else '#49535C')
                records.append({'figure':'3','model':m,'metric':metric,'reference':ref,'delta_median':delta,'inference':'descriptive event-wise median difference'})
            if block in ('a','d'):
                baseline=vectors[models.index('maomao' if block=='a' else 'reference')]
                ax.axvline(np.median(baseline),color=R.ROSE,lw=.55,ls='--')
            else:
                for k in (0,2,4):ax.plot([np.median(vectors[k]),np.median(vectors[k+1])],[len(models)-1-k,len(models)-2-k],color='#8995A1',lw=.55,ls=':')
            ax.text(1.025,1.03,'Δ med.',transform=ax.transAxes,fontsize=8,ha='left',va='bottom')
        elif pid.startswith('time_'):
            pos=ax.get_position();ax.set_position([pos.x0,pos.y0,.15,pos.height])
            table=pd.read_csv(B.DATA/'figure_3_box_statistics.csv');table=table[table.metric==f'time_error_{pid.split("_")[-1]}']
            models=['model_small','reference','model_large','context_64','context_128'];baseline=float(table[table.model=='reference'].iloc[0]['med'])
            ax.axvline(baseline,color=R.ROSE,lw=.55,ls='--')
            for yy,m in zip(np.arange(5)[::-1],models):
                delta=float(table[table.model==m].iloc[0]['med'])-baseline
                ax.text(1.025,yy,f'{delta:+.2f}',transform=ax.get_yaxis_transform(),va='center',fontsize=8)
            ax.text(1.025,1.03,'Δ med.',transform=ax.transAxes,fontsize=8,va='bottom')
    for t in fig.texts:
        if t.get_text().startswith('Output-task comparisons'):t.set_text('Δ median: variant − matched MAOMAO reference; output pairs use identical labels. Time boxes use symlog hours.')

def adapt_s3(fig,axes,ids):
    source=pd.read_csv(B.DATA/'supplementary_S3_points.csv') if False else None
    for ax in axes:
        xs=np.concatenate([np.asarray(l.get_xdata(),float) for l in ax.lines if len(l.get_xdata())])
        finite=xs[np.isfinite(xs)];lo=finite.min();hi=finite.max();pad=max((hi-lo)*.09,.002)
        ax.set_xlim(max(0,lo-pad),min(1,hi+pad));ax.xaxis.set_major_locator(MaxNLocator(2))

# Existing scores, curves, calibration mappings and palette are retained.
def figure4():
    dictionary,curves,scores=R._clinical_sources();selected=[k for i,k in enumerate(dictionary) if i not in (1,3,7,8)]
    h=8.5;fig=plt.figure(figsize=(B.WIDTH,h));axes=[];ids=[]
    heading(fig,'Figure 4 | Six clinical endpoints on common complete cases','Original frozen scores; unchanged empirical AUROC values and solid model curves')
    for i,e in enumerate(selected):
        row,col=divmod(i,3);bottom=(4.72-row*3.10)/h;ax=fig.add_axes([.085+col*.315,bottom,.245,2.10/h]);axes.append(ax);ids.append(e)
        R._curve_axis(ax,e,dictionary[e],curves,scores,'roc');ax.xaxis.set_major_locator(MaxNLocator(3));ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.set_title(f'{chr(97+i)}  {B.ENDPOINT_TITLES[e]}',loc='left',fontsize=8,fontweight='bold',pad=9)
        ax.legend(loc='lower right',fontsize=8,handlelength=.7,handletextpad=.3,labelspacing=.17,borderaxespad=.1)
        c=dictionary[e];fig.text(.085+col*.315,bottom+2.10/h+.35/h,f'n={c["complete_episodes"]:,}; events={c["events"]}',fontsize=8)
    fig.text(.055,.06,'Endpoint-specific common cases; original patient-disjoint 85:15 clinical study, separate from the primary event task.',fontsize=8)
    fig.text(.055,.035,'Adapted clinical scores marked *. All ten endpoint comparisons are combined in Supplementary S2.',fontsize=8)
    save(fig,4,axes,ids,row_groups=[{'id':f'r{i}','panels':selected[i*3:i*3+3]} for i in range(2)],column_groups=[{'id':f'c{i}','panels':selected[i::3]} for i in range(3)])
    supplement4(dictionary,curves,scores)

def supplement4(dictionary,curves,scores):
    cal=pd.read_csv(B.DATA/'clinical_calibration_curves.csv');dca=pd.read_csv(B.DATA/'clinical_dca_curves.csv');h=33.4
    fig=plt.figure(figsize=(B.WIDTH,h));axes=[];ids=[]
    fig.text(.055,(h-.18)/h,'Supplementary Figure S2 | Ten clinical endpoint comparisons',fontsize=10,fontweight='bold',va='top')
    fig.text(.055,(h-.47)/h,'10 rows × 4 columns: original ROC, original PR, training-mapped calibration, training-mapped DCA',fontsize=8,va='top')
    for ri,(e,c) in enumerate(dictionary.items()):
        top=h-1.25-ri*3.18;bottom=top-2.12
        fig.text(.055,(top+.30)/h,f'{ri+1}. {B.ENDPOINT_TITLES[e]} | n={c["complete_episodes"]:,}; events={c["events"]}',fontsize=10,fontweight='bold')
        key=[Line2D([],[],color=R.CLINICAL_COLORS[si],lw=.8,label=B.CLINICAL_SHORT.get(s['name'],s['name'])) for si,s in enumerate(c['series'])]
        fig.legend(handles=key,loc='upper left',bbox_to_anchor=(.08,(top+.10)/h),ncol=len(key),fontsize=8,
                   handlelength=.65,handletextpad=.3,columnspacing=.8,borderaxespad=0,frameon=False)
        for ci,kind in enumerate(('roc','pr','calibration','dca')):
            ax=fig.add_axes([.092+ci*.228,bottom/h,.155,1.75/h]);axes.append(ax);ids.append(f'{e}_{kind}')
            if kind in ('roc','pr'):
                R._curve_axis(ax,e,c,curves,scores,kind);ax.get_legend().remove()
                handles=[]
                for si,s in enumerate(c['series']):
                    metric='auroc' if kind=='roc' else 'average_precision'
                    value=float(scores[(scores.endpoint==e)&(scores.model==s['name'])&(scores.metric==metric)].iloc[0].value)
                    handles.append(Line2D([],[],color=R.CLINICAL_COLORS[si],lw=0,label=f'{value:.3f}'))
                # Compact colored values identify the curves through the row key.
                ax.legend(handles=handles,loc='lower right' if kind=='roc' else 'upper right',ncol=1 if kind=='roc' else 2,
                          fontsize=8,handlelength=0,handletextpad=0,labelcolor='linecolor',
                          labelspacing=.15,columnspacing=.6,borderaxespad=.2,frameon=False)
            elif kind=='calibration':
                p=cal[cal.endpoint==e];mx=min(1,max(.005,float(p[['mean_risk','observed_fraction']].to_numpy().max())*1.12,float(p[p.model=='MAOMAO'].ci_upper.max())*1.12))
                for si,s in enumerate(c['series']):
                    v=p[p.model==s['name']];color=R.CLINICAL_COLORS[si]
                    if si==0:ax.errorbar(v.mean_risk,v.observed_fraction,yerr=np.maximum(0,np.array([v.observed_fraction-v.ci_lower,v.ci_upper-v.observed_fraction])),fmt='o-',ms=2,lw=.8,elinewidth=.45,capsize=1,color=color)
                    else:ax.plot(v.mean_risk,v.observed_fraction,'o-',ms=2,lw=.65,color=color)
                ax.plot([0,mx],[0,mx],color='#B8A1C3',ls=':',lw=.45);ax.set(xlim=(0,mx),ylim=(0,mx),xlabel='Predicted risk',ylabel='Observed fraction')
            else:
                xmax=min(.5,max(.005,c['prevalence']*3));p=dca[(dca.endpoint==e)&(dca.threshold<=xmax)];threshold=np.unique(p.threshold);all_nb=c['prevalence']-(1-c['prevalence'])*threshold/(1-threshold)
                for si,s in enumerate(c['series']):
                    v=p[p.model==s['name']];color=R.CLINICAL_COLORS[si]
                    if si==0:ax.fill_between(v.threshold,v.ci_lower,v.ci_upper,color=color,alpha=.10,lw=0)
                    ax.plot(v.threshold,v.net_benefit,color=color,lw=.8 if si==0 else .65)
                ax.plot(threshold,all_nb,color='#59616A',lw=.5,ls=':');ax.axhline(0,color='#8A9096',lw=.5,ls=':')
                v=p[p.model=='MAOMAO'];lo=min(float(p.net_benefit.min()),float(v.ci_lower.min()),float(all_nb.min()),0);hi=max(float(p.net_benefit.max()),float(v.ci_upper.max()),float(all_nb.max()),0);pad=max((hi-lo)*.12,.0001)
                ax.set(xlim=(0,xmax),ylim=(lo-pad,hi+pad),xlabel='Risk threshold',ylabel='Net benefit')
            ax.xaxis.set_major_locator(MaxNLocator(3));ax.yaxis.set_major_locator(MaxNLocator(3));ax.ticklabel_format(style='plain',axis='both',useOffset=False)
            ax.set_title(['a  ROC','b  Precision–recall','c  Calibration','d  DCA'][ci],loc='left',fontsize=8,fontweight='bold',pad=9)
    fig.text(.055,.019,'Colored corner values: ROC AUROC / PR average precision. Original scores and training-only risk mappings unchanged.',fontsize=8)
    fig.text(.055,.012,'MAOMAO risk bands: existing pointwise 95% patient-cluster CIs; RBC endpoints with 3/4 events remain unstable.',fontsize=8)
    save(fig,'Supplementary_S2',axes,ids,row_groups=[{'id':f'endpoint_{e}','panels':[f'{e}_{k}' for k in ('roc','pr','calibration','dca')]} for e in dictionary],column_groups=[{'id':k,'panels':[f'{e}_{k}' for e in dictionary]} for k in ('roc','pr','calibration','dca')])

def figure5():
    shap=B.SOURCE/'shap';meta=B.read(shap/'event_sequence_meta.json');mapping,names=B.display_mapping(meta['outcome_family_vocabulary']);palette=np.array(R.GROUP_COLORS)
    h=16.5;fig=plt.figure(figsize=(B.WIDTH,h));axes=[];ids=[]
    heading(fig,'Figure 5 | Learned event geometry and SHAP contributions','9,989 holdout patients; original frozen queries and Partition SHAP; clinical colors unchanged')
    frame=pd.read_csv(shap/'embedding_event_coordinates.csv');u=frame[['umap_1','umap_2']].to_numpy();group=np.array([names.index(n) for n in frame.clinical_display_group]);size=4+frame.scatter_area_pt2.to_numpy()*1.3
    main_bottom=10.70;main_height=4.50;main_top=main_bottom+main_height
    main=fig.add_axes([.065,main_bottom/h,.30,main_height/h]);axes.append(main);ids.append('embedding')
    for gi in range(17):
        take=group==gi;main.scatter(u[take,0],u[take,1],s=size[take],c=[palette[gi]],edgecolors='white',lw=.25,alpha=.92)
    main.set(xlabel='UMAP 1',ylabel='UMAP 2');main.xaxis.set_major_locator(MaxNLocator(3));main.yaxis.set_major_locator(MaxNLocator(3));B.panel_title(main,'a','Learned event embedding')
    # Freeze the original embedding bounds before adding zoom leader lines.
    main.set_xlim(main.get_xlim());main.set_ylim(main.get_ylim());main.set_autoscale_on(False)
    death_idx=frame.index[frame.event=='inhospital_death'][0]
    main.annotate('In-hospital death',xy=u[death_idx],xytext=(-6,-16),textcoords='offset points',ha='right',va='top',fontsize=8,
                  arrowprops=dict(arrowstyle='-',color='#20262C',lw=.55))
    anchors=['map_hypotension','hypoxemia','aki_stage_1_signal'];anchors.sort(key=lambda n:u[frame.index[frame.event==n][0],1],reverse=True);zoom_ids=[];zoom_records=[]
    zoom_gap=.24;zoom_height=(main_height-2*zoom_gap)/3
    all_dist=np.sqrt(((u[:,None,:]-u[None,:,:])**2).sum(-1));np.fill_diagonal(all_dist,np.inf);radius=4*np.median(all_dist.min(1))
    for k,event in enumerate(anchors):
        anchor=frame.index[frame.event==event][0];d=np.sqrt(((u-u[anchor])**2).sum(1));near=np.argsort(d)[:8];near=near[d[near]<=radius]
        low=u[near].min(0);high=u[near].max(0);span=np.maximum(high-low,.1);low-=span*.35;high+=span*.35
        bottom=main_top-zoom_height-k*(zoom_height+zoom_gap);ax=fig.add_axes([.42,bottom/h,.12,zoom_height/h]);axes.append(ax);pid=f'zoom_{k}';ids.append(pid);zoom_ids.append(pid)
        ax.scatter(u[near,0],u[near,1],s=18+size[near]*.6,c=palette[group[near]],ec='white',lw=.3);ax.scatter(*u[anchor],s=65,fc='none',ec='#20262C',lw=.7)
        ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]));ax.set_xticks([]);ax.set_yticks([]);ax.set_title(R._human(event),fontsize=8,pad=6,fontweight='bold')
        for spine in ax.spines.values():spine.set_visible(True);spine.set_color('#8995A1');spine.set_linewidth(.65)
        ordered=near[np.argsort(u[near,1])[::-1]]
        labels=['\n'.join(textwrap.wrap(R._human(frame.iloc[idx].event),26,break_long_words=False)) for idx in ordered]
        weights=np.array([label.count('\n')+1 for label in labels]);centers=np.cumsum(weights)-weights/2
        for yi,idx in enumerate(ordered):
            name=labels[yi]
            label_y=1-centers[yi]/weights.sum()
            point_axes=ax.transAxes.inverted().transform(ax.transData.transform(u[idx]))
            ax.plot([point_axes[0],1.05,1.12],[point_axes[1],label_y,label_y],transform=ax.transAxes,color=palette[group[idx]],lw=.55,clip_on=False)
            ax.text(1.18,label_y,name,transform=ax.transAxes,ha='left',va='center',fontsize=8,linespacing=1.05)
            zoom_records.append({'anchor':event,'event':frame.iloc[idx].event,'label':name,'coordinate':u[idx].tolist()})
        main.add_patch(Rectangle(low,*(high-low),fill=False,ec=palette[group[anchor]],lw=.8))
        ymid=(high[1]+low[1])/2
        main.plot([high[0],main.get_xlim()[1]],[ymid,ymid],color='#8995A1',lw=.45)
        yfrac=(ymid-main.get_ylim()[0])/(main.get_ylim()[1]-main.get_ylim()[0])
        fig.add_artist(ConnectionPatch(xyA=(1,yfrac),coordsA=main.transAxes,xyB=(0,.5),coordsB=ax.transAxes,color='#8995A1',lw=.45))
    # Put the clinical color key on the right of the UMAP and its three insets.
    for gi,name in enumerate(names):
        x=.845;y=(main_top-.25-gi*.239)/h
        fig.add_artist(Rectangle((x,y-.020/h),.008,.070/h,transform=fig.transFigure,color=palette[gi],lw=0));fig.text(x+.013,y,R.SHORT_GROUP[name],fontsize=8,va='center')
    fig.text(.845,(main_top-.25-17*.239)/h,'Clinical groups',fontsize=8)
    assert np.isclose(axes[1].get_position().y1,main.get_position().y1)
    assert np.isclose(axes[3].get_position().y0,main.get_position().y0)
    (QA/'figure5_embedding_zoom_extent.json').write_text(json.dumps({'umap_bottom_inches':main_bottom,'umap_top_inches':main_top,'zoom_height_inches':zoom_height,'zoom_gap_inches':zoom_gap,'zoom_stack_extent_inches':3*zoom_height+2*zoom_gap,'death':'directly labeled in UMAP; no inset','legend':'right'},indent=2)+'\n')
    rank=pd.read_csv(shap/'clinical_group_shap_importance_95ci.csv').sort_values('mean_abs_log_probability_shap',ascending=False);order=np.array([names.index(x) for x in rank.family])
    imp=fig.add_axes([.205,6.15/h,.275,2.90/h]);dist=fig.add_axes([.59,6.15/h,.365,2.90/h]);axes.extend([imp,dist]);ids.extend(['importance','beeswarm'])
    for yy,gi in zip(np.arange(17)[::-1],order):
        r=rank[rank.family==names[gi]].iloc[0];imp.barh(yy,r.mean_abs_log_probability_shap,height=.65,color=palette[gi],alpha=.85);imp.plot([r.ci_lower,r.ci_upper],[yy]*2,color='#20262C',lw=.65)
    imp.set_yticks(np.arange(17)[::-1],rank.family);imp.tick_params(axis='y',length=0);imp.set_ylim(-.7,16.7);imp.set_xlabel('Mean |log-probability SHAP|');imp.xaxis.set_major_locator(MaxNLocator(3));B.panel_title(imp,'b','Input importance (95% CI)')
    death=np.load(shap/'death_clinical_group_shap_by_patient.npz');contrib=death['signed_shap'];present=death['family_present']
    for yy,gi in zip(np.arange(17)[::-1],order):
        v=contrib[present[:,gi],gi];points=dist.scatter(v,yy+R._beeswarm(v),s=1.2,c=[palette[gi]],alpha=.30,lw=0);points.set_gid('dense_shap_points')
    dist.axvline(0,color='#738187',lw=.5);dist.set_yticks(np.arange(17),['']*17);dist.tick_params(axis='y',length=0);dist.set_ylim(-.7,16.7);dist.set_xlabel('SHAP to log death-family probability');dist.xaxis.set_major_locator(MaxNLocator(3));B.panel_title(dist,'c','Observed contributions')
    fig.text(.205,5.57/h,'Same input-group order; b averages all outputs, c explains the death-family output.',fontsize=8)
    fig.text(.055,5.15/h,'d  Historical contributions at three time scales',fontsize=10,fontweight='bold')
    data=np.load(shap/'figure4c_display_matrix.npz');fam=np.array(meta['outcome_to_family']);mapped=np.array(mapping);rg=mapped[fam[data['predictor_event_ids']]];cg=mapped[fam[data['predicted_event_ids']]]
    def edges(g):
        e=np.r_[0,np.cumsum([1/np.sum(g==i) for i in g])];unique=np.unique(g);centers=[(e[np.flatnonzero(g==i)[0]]+e[np.flatnonzero(g==i)[-1]+1])/2 for i in unique];return e,unique,centers
    re,ru,rc=edges(rg);ce,cu,cc=edges(cg);cmap=LinearSegmentedColormap.from_list('unchanged',[R.BLUE,'#BCD7ED','#FFFFFF','#F4E1DE',R.ROSE]);lagids=[]
    for li,label in enumerate(data['lag_labels']):
        ax=fig.add_axes([.185+li*.27,1.88/h,.23,2.47/h]);axes.append(ax);pid=f'lag_{li}';ids.append(pid);lagids.append(pid)
        mesh=ax.pcolormesh(ce,re,data['folds'][li],cmap=cmap,norm=LogNorm(.1,10),shading='flat')
        ax.set(xlim=(0,ce[-1]),ylim=(re[-1],0));ax.set_yticks(rc,[R.SHORT_GROUP[names[g]] for g in ru] if li==0 else []);ax.set_xticks(cc,[R.SHORT_GROUP[names[g]] for g in cu],rotation=90,rotation_mode='anchor',ha='right',va='center');ax.tick_params(length=0,pad=3);ax.tick_params(axis='y',pad=10)
        for e in re[np.flatnonzero(np.diff(rg))+1]:ax.axhline(e,color='#8995A1',lw=.25)
        for e in ce[np.flatnonzero(np.diff(cg))+1]:ax.axvline(e,color='#8995A1',lw=.25)
        for gi in cu:
            ix=np.flatnonzero(cg==gi);ax.add_patch(Rectangle((ce[ix[0]],-.36),ce[ix[-1]+1]-ce[ix[0]],.25,color=palette[gi],lw=0,clip_on=False))
        for gi in ru:
            ix=np.flatnonzero(rg==gi);ax.add_patch(Rectangle((-.40,re[ix[0]]),.25,re[ix[-1]+1]-re[ix[0]],color=palette[gi],lw=0,clip_on=False))
        ax.set_title(str(label),fontsize=10,fontweight='bold',pad=14)
    cb=fig.add_axes([.30,.49/h,.33,.08/h]);cbar=fig.colorbar(mesh,cax=cb,orientation='horizontal',ticks=[.1,1,10]);cbar.solids.set_rasterized(False);cbar.solids.set_edgecolor('face');cb.set_xticklabels(['0.1','1','10']);cb.tick_params(length=2)
    fig.text(.65,.49/h,'exp(mean log-probability SHAP)',fontsize=8,va='center')
    fig.text(.055,.13/h,'174 predictors × 175 outputs; unchanged heatmap values. Individual dependencies are in Supplementary S3.',fontsize=8)
    save(fig,5,axes,ids,row_groups=[{'id':'bc','panels':['importance','beeswarm']},{'id':'lags','panels':lagids}],column_groups=[{'id':'zoom_boxes','panels':zoom_ids}])
    (QA/'figure5_zoom_labels.json').write_text(json.dumps(zoom_records,indent=2)+'\n')
    supplement5(meta,mapping,names)

def supplement5(meta,mapping,names):
    source=B.DATA/'shap_dependence_observed_history.npz';d=np.load(source);events=d['event_names'].tolist();v=d['last_observed_value'];phi=d['event_shap_to_log_death']
    candidates=[]
    for i,e in enumerate(events):
        keep=np.isfinite(v[:,i])
        if keep.sum()>=200 and len(np.unique(v[keep,i]))>=6:candidates.append((float(np.mean(abs(phi[keep,i]))),e,int(keep.sum())))
    candidates.sort(reverse=True)
    fixed=[('systolic_bp_normalized','heart_rate_normalized'),('temperature_normalized','leukocytosis'),('leukocytosis','temperature_normalized'),('bradycardia','systolic_bp_normalized')]
    selected=fixed.copy()
    for importance,e,n in candidates:
        if e in [a for a,b in selected]:continue
        ei=events.index(e)
        for _,partner,_ in candidates:
            if partner==e:continue
            pi=events.index(partner);ok=np.isfinite(v[:,ei])&np.isfinite(v[:,pi])
            if ok.sum()>=200 and len(np.unique(v[ok,ei]))>=6 and len(np.unique(v[ok,pi]))>=6:
                selected.append((e,partner));break
        if len(selected)==12:break
    assert len(selected)==12
    h=15.0;fig=plt.figure(figsize=(B.WIDTH,h));axes=[];ids=[];selection=[]
    heading(fig,'Supplementary Figure S3 | Individual SHAP and observed-variable relationships','Original Figure 5e relationships + high-contribution inputs; actual observed values, no causal interaction claim')
    cmap=LinearSegmentedColormap.from_list('unchanged_pairs',[R.BLUE,'#49BCD6','#F9E2AD',R.ROSE])
    for i,(e,partner) in enumerate(selected):
        row,col=divmod(i,4);bottom=h-2.70-row*3.13
        ax=fig.add_axes([.080+col*.235,bottom/h,.173,1.56/h]);axes.append(ax);pid=f'dep_{i}';ids.append(pid)
        ei,pi=events.index(e),events.index(partner);ok=np.isfinite(v[:,ei])&np.isfinite(v[:,pi]);x=v[ok,ei];y=phi[ok,ei];color=v[ok,pi]
        sc=ax.scatter(x,y,c=color,cmap=cmap,s=2.5,alpha=.6,lw=0);sc.set_gid('dense_shap_points');ax.axhline(0,color='#B8A1C3',lw=.4)
        ax.set_xlabel('Last observed value');ax.set_ylabel('SHAP to log death' if col==0 else '');ax.xaxis.set_major_locator(MaxNLocator(3));ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.set_title(f'{chr(97+i)}  '+ '\n'.join(textwrap.wrap(R._human(e),24)),fontsize=8,fontweight='bold',loc='left',pad=9)
        cb=fig.add_axes([.080+col*.235,(bottom-.62)/h,.173,.065/h]);bar=fig.colorbar(sc,cax=cb,orientation='horizontal');bar.solids.set_rasterized(False);bar.solids.set_edgecolor('face');cb.xaxis.set_major_locator(MaxNLocator(3));cb.tick_params(length=2)
        fig.text(.080+col*.235,(bottom-.99)/h,'Color: '+R._human(partner),fontsize=8)
        selection.append({'input':e,'color_partner':partner,'pair_complete_patients':int(ok.sum()),'mean_abs_shap_pair':float(abs(y).mean()),'kept_original_5e':i<4})
    # Select four explanatory cases with distinct dominant clinical groups;
    # all other cases remain in the full cohort source, not a representativeness claim.
    s=B.SOURCE/'shap';cohort=np.load(s/'death_clinical_group_shap_by_patient.npz');values=cohort['signed_shap'];dominant=np.argmax(abs(values),axis=1);strength=abs(values).sum(1);caseids=[];seen=set()
    signed=values.sum(axis=1)
    for direction in (1,-1):
        taken=0
        for ix in np.argsort(strength)[::-1]:
            if signed[ix]*direction<=0 or int(dominant[ix]) in seen:continue
            seen.add(int(dominant[ix]));caseids.append(int(ix));taken+=1
            if taken==2:break
    assert len(caseids)==4
    death=210+meta['outcome_family_vocabulary'].index('death');water=[]
    for col,case in enumerate(caseids):
        a=np.load(s/f'patients/case_{case:05d}.npz');con=np.zeros(210);np.add.at(con,a['input_group_ids']//3,a['shap_values'][:,death]);rank=np.argsort(abs(con))[::-1];top=rank[:4];vals=list(con[top])+[float(con[rank[4:]].sum())];labels=[R._human(events[k]) for k in top]+['Other inputs']
        baseline=float(a['baseline_log_family_probability'][death]);full=float(a['full_log_family_probability'][death]);assert np.isclose(baseline+sum(vals),full,atol=1e-4)
        bottom=.98;ax=fig.add_axes([.135+col*.235,bottom/h,.102,1.94/h]);axes.append(ax);pid=f'case_{col}';ids.append(pid);cursor=baseline;points=[cursor]
        for yy,value,label in zip(np.arange(5)[::-1],vals,labels):
            end=cursor+value;lo,hi=sorted([cursor,end]);tip=min(abs(value)*.18,.05)
            verts=[(lo,yy-.23),(hi-tip,yy-.23),(hi,yy),(hi-tip,yy+.23),(lo,yy+.23)] if value>=0 else [(hi,yy-.23),(lo+tip,yy-.23),(lo,yy),(lo+tip,yy+.23),(hi,yy+.23)]
            ax.add_patch(Polygon(verts,fc=R.ROSE if value>=0 else R.BLUE,ec='none'))
            ax.text(1.04,yy,f'{value:+.2f}',transform=ax.get_yaxis_transform(),va='center',fontsize=8,color=R.ROSE if value>=0 else R.BLUE)
            if yy>0:ax.plot([end,end],[yy-.23,yy-.77],color='#8995A1',lw=.4)
            water.append({'case':case,'event':label,'value':value,'start':cursor,'end':end,'baseline':baseline,'full':full});cursor=end;points.append(cursor)
        ax.set_yticks(np.arange(5)[::-1],['\n'.join(textwrap.wrap(l,12,break_long_words=False,break_on_hyphens=False)) for l in labels]);ax.tick_params(axis='y',length=0,pad=3);ax.set_ylim(-.6,4.6);span=max(max(points)-min(points),.3);ax.set_xlim(min(points)-span*.08,max(points)+span*.15);ax.xaxis.set_major_locator(MaxNLocator(2));ax.set_xlabel('Log probability')
        ax.set_title(f'{chr(109+col)}  Case {case:05d}\np(next death) = {np.exp(full):.2g}',fontsize=8,fontweight='bold',loc='left',pad=12)
    fig.text(.055,3.34/h,'m–p  Two positive / two negative high-contribution cases; distinct dominant groups; top four event inputs + all others',fontsize=8,fontweight='bold')
    fig.text(.055,.033,'Dependencies include all original pair-complete patients; color is another measurement, not an exact SHAP interaction.',fontsize=8)
    fig.text(.055,.017,'Case selection is explanatory, not population-representative. Red / blue contributions increase / decrease log probability.',fontsize=8)
    save(fig,'Supplementary_S3',axes,ids,row_groups=[{'id':f'dep_row{r}','panels':[f'dep_{i}' for i in range(r*4,r*4+4)]} for r in range(3)]+[{'id':'cases','panels':[f'case_{i}' for i in range(4)]}],column_groups=[{'id':f'dep_col{c}','panels':[f'dep_{i}' for i in range(c,12,4)]} for c in range(4)])
    (QA/'s5_dependency_selection.json').write_text(json.dumps({'selection':selection,'additional_rule':'descending mean |SHAP| among observed inputs with >=200 patients and >=6 distinct values; partner must be co-observed in >=200 patients','original_5e_retained':True,'cases':caseids,'case_rule':'two positive and two negative total contributions; highest total absolute group contribution; distinct dominant groups'},indent=2)+'\n');pd.DataFrame(water).to_csv(QA/'s5_waterfalls.csv',index=False)

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--figures',nargs='+',type=int,default=[2,3,4,5]);a=parser.parse_args()
    B.setup();B.save=save;R.B=B
    for n in a.figures:
        if n==2:figure2()
        elif n==3:R.figure3()
        elif n==4:figure4()
        elif n==5:figure5()
    if records:
        current=pd.DataFrame(records)
        record_path=QA/'comparison_annotations.csv'
        if record_path.exists():
            previous=pd.read_csv(record_path,dtype={'figure':str})
            previous=previous[~previous.figure.astype(str).isin(current.figure.astype(str).unique())]
            current=pd.concat([previous,current],ignore_index=True)
        current.to_csv(record_path,index=False)
    (QA/'revision_scope.json').write_text(json.dumps({'user_requirements':{'S2':'10 rows × 4 columns; a–d only','S1':'adaptive x limits remove unused left region','Figure5a':'zoom boxes ordered by UMAP height; every inset event annotated','Figure5e':'removed from main; moved and expanded in S3','S3':'4 columns, multiple rows, original + high-contribution measured dependencies and individual cases','Figure4':'3 columns × 2 rows','Figure3':'adaptive box axes; matched MAOMAO median differences','Figure2':'12 bar panels, 4 × 3; equal bar widths; three family Hit heatmaps; final-state MAOMAO references and differences','Figure1':'omitted from delivery','delivery':'one flat folder, PNG/PDF only, 600 dpi, actual Arial 8/10 pt, 210 mm width'},'inference_limitations':'No paired full-row p values exist for Figure 2/3. Comparisons are labeled descriptive. No unsupported significance stars.'},indent=2)+'\n')
if __name__=='__main__':main()
