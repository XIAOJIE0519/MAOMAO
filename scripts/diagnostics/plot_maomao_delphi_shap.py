#!/usr/bin/env python3
"""Original Python rendering of Delphi Fig.4a/c structure from MAOMAO artifacts."""
import sys,json,argparse,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,ConnectionPatch,Polygon
from matplotlib.colors import LogNorm

PORTABLE=(Path(__file__).parent/'protocol.json').exists()
ROOT=Path(__file__).resolve().parents[2]
if not PORTABLE:
    sys.path.insert(0,str(ROOT))
    sys.path.insert(0,'/home/yunkunshi/.codex/skills/nature-figure/scripts')
from audit_panel_alignment import require_matplotlib_panel_alignment
if PORTABLE:
    from maomao_display_family_groups import display_mapping
else:
    from scripts.diagnostics.maomao_display_family_groups import display_mapping

def sha256(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):result.update(block)
    return result.hexdigest()

OUT=Path(__file__).parent if PORTABLE else ROOT/'outputs/maomao_plot_sources/shap'
DATA=OUT if PORTABLE else ROOT/'data/perioperative_event_sequences_v5_richctx_static7'
FIG=OUT/'figures'
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
    'font.size':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,
    'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.5,
    'legend.frameon':False,'axes.spines.top':False,'axes.spines.right':False})

def read(p):return json.loads(p.read_text())
def colors(n):
    a=[*plt.get_cmap('tab20').colors,*plt.get_cmap('tab20b').colors,*plt.get_cmap('tab20c').colors,
        '#614f88','#8a8a8a','#0e1327']
    result=np.array([mpl.colors.to_hex(x) for x in a[:n]]);result[-1]='#0e1327'
    return result

def export(fig,name,axes,ids,row_groups=None,column_groups=None):
    FIG.mkdir(parents=True,exist_ok=True)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=ids,row_groups=row_groups or [],
        column_groups=column_groups or [],json_out=FIG/f'{name}.alignment.json',strict=True)
    fig.savefig(FIG/f'{name}.svg',bbox_inches=None)
    fig.savefig(FIG/f'{name}.pdf',bbox_inches=None)
    fig.savefig(FIG/f'{name}.png',dpi=300)
    fig.savefig(FIG/f'{name}.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)

def embedding(fine=False):
    meta=read(DATA/'event_sequence_meta.json');names=meta['outcome_vocabulary'];families=meta['outcome_family_vocabulary']
    checkpoint=ROOT/'outputs/final_experiment_results_20260923/full_maomao_reference/best_model.pt'
    expected_sha=read(OUT/'protocol.json')['model_sha256'] if PORTABLE else sha256(checkpoint)
    token_ids=np.array([meta['token_vocabulary']['event:'+name] for name in names])
    emb=OUT/'embedding_umap.npz'
    if emb.exists():
        with np.load(emb) as a:
            u=a['umap_coordinates'];weights_sha=str(a['checkpoint_sha256']);embedding_dimension=a['token_embeddings'].shape[1]
        if weights_sha!=expected_sha:raise RuntimeError('UMAP source model changed')
    else:
        import torch,umap
        weight=torch.load(checkpoint,map_location='cpu',weights_only=False)['model']['token_embedding.weight'].float().numpy()
        embedding_dimension=weight.shape[1]
        reducer=umap.UMAP(random_state=1413,n_neighbors=30,min_dist=.05,metric='cosine',n_jobs=1)
        all_u=reducer.fit_transform(weight);u=all_u[token_ids]
        u=-(u-np.median(all_u,axis=0))
        np.savez_compressed(emb,umap_coordinates=u,token_embeddings=weight[token_ids],token_ids=token_ids,
            checkpoint_sha256=expected_sha,event_names=np.array(names),outcome_to_family=np.array(meta['outcome_to_family']))
    counts=np.array([meta.get('audit_counts',{}).get(name,0) for name in names],dtype=float)
    ids=np.array(meta['outcome_to_family']);mapping,group_names=display_mapping(families)
    color_ids=ids if fine else np.array(mapping)[ids]
    labels=families if fine else group_names
    palette=colors(len(labels))
    # Scale areas by sqrt(entry count); all 210 event embeddings are shown.
    size=1.2+.035*np.sqrt(counts)
    size=np.minimum(size,60)
    table=pd.DataFrame(dict(event_code=[f'E{i+1:03d}' for i in range(len(names))],event=names,
        family=[families[i] for i in ids],family_code=[f'F{i+1:02d}' for i in ids],entries_in_retained_source=counts.astype(int),
        clinical_display_group=[group_names[mapping[i]] for i in ids],
        umap_1=u[:,0],umap_2=u[:,1],scatter_area_pt2=size))
    table.to_csv(OUT/'embedding_event_coordinates.csv',index=False)
    fig=plt.figure(figsize=(7.5,6.7))
    main=fig.add_axes([.205,.22,.405,.48]);main.set_axis_off()
    main.scatter(u[:,0],u[:,1],s=size,c=palette[color_ids],alpha=.73,edgecolors='white',linewidths=.25)
    fig.text(.01,.972,'a',fontsize=12,fontweight='bold',va='top')
    fig.text(.205,.93,'MAOMAO learned event embeddings',fontsize=10)
    # Prespecified anchors; true geometric neighbours are displayed, not curated clusters.
    anchors=['map_hypotension','hypoxemia','aki_stage_1_signal','inhospital_death']
    anchors=[names.index(x) if x in names else int(k) for x,k in zip(anchors,[3,17,170,209])]
    positions=[(.025,.765,.16,.14),(.645,.765,.16,.14),(.025,.035,.16,.14),(.435,.035,.16,.14)]
    inset_axes=[]
    for k,(anchor,pos) in enumerate(zip(anchors,positions)):
        dist=np.sqrt(((u[:,None]-u[None,:])**2).sum(-1));np.fill_diagonal(dist,np.inf)
        radius=4*np.median(dist.min(1))
        distances=np.sqrt(((u-u[anchor])**2).sum(1))
        near=np.argsort(distances)[:12];near=near[distances[near]<=radius]
        low=u[near].min(0);high=u[near].max(0);span=np.maximum(high-low,.1);low-=span*.45;high+=span*.45
        ax=fig.add_axes(pos);inset_axes.append(ax)
        mark_size=12+size[near]*.5
        ax.scatter(u[near,0],u[near,1],s=mark_size,c=palette[color_ids[near]],alpha=.9,edgecolors='white',linewidths=.2)
        ax.set_xlim(low[0],high[0]);ax.set_ylim(low[1],high[1]);ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_visible(True);spine.set_linestyle((0,(3,3)));spine.set_linewidth(.6)
        fig.canvas.draw();renderer=fig.canvas.get_renderer();prior=[]
        centres=ax.transData.transform(u[near]);pixel=fig.dpi/72
        from matplotlib.transforms import Bbox
        obstacles=[Bbox.from_bounds(p[0]-np.sqrt(s)*pixel/2-2,p[1]-np.sqrt(s)*pixel/2-2,
                                   np.sqrt(s)*pixel+4,np.sqrt(s)*pixel+4) for p,s in zip(centres,mark_size)]
        for j in dict.fromkeys((anchor,near[min(5,len(near)-1)])):
            for dx,dy in [(dx,dy) for dy in (10,-14,22,-24,0,35,-35) for dx in (8,-26,20,-40,35,-55)]:
                text=ax.annotate(f'E{j+1:03d}',u[j],xytext=(dx,dy),textcoords='offset points',fontsize=6,va='bottom')
                fig.canvas.draw();bbox=text.get_window_extent(renderer)
                if ax.bbox.contains(bbox.x0,bbox.y0) and ax.bbox.contains(bbox.x1,bbox.y1) and not any(bbox.overlaps(o) for o in obstacles+prior):
                    prior.append(bbox);break
                text.remove()
            else:raise RuntimeError(f'No collision-free event label position for E{j+1:03d}')
        rect=Rectangle(low,*(high-low),fill=False,ls=(0,(3,3)),lw=.6,color='#555555');main.add_patch(rect)
        # Leader endpoints on box corners avoid passing through inset labels.
        target=tuple(low) if k%2==0 else (float(high[0]),float(low[1]))
        origin=(1,0) if k==0 else ((0,0) if k==1 else (1,1))
        line=ConnectionPatch(xyA=origin,coordsA=ax.transAxes,xyB=target,coordsB=main.transData,lw=.45,color='#666666')
        fig.add_artist(line)
        title=names[anchor].replace('_',' ')
        fig.text(pos[0],pos[1]+pos[3]+.012,f'E{anchor+1:03d}: {title}',fontsize=6.5,va='bottom')
    # Family key occupies the right-hand side, mirroring the reference layout.
    legend=fig.add_axes([.635,.19,.355,.53]);legend.set_axis_off()
    legend.text(0,1.055,'Event family' if fine else 'Clinical family group',fontsize=7.5)
    for i,name in enumerate(labels):
        col=i//32 if fine else 0;row=i%32 if fine else i;x=col*.51;y=1-row/(32 if fine else 24)
        legend.scatter([x+.018],[y],c=[palette[i]],s=13)
        short=name.replace('_',' ')
        if len(short)>22:short=short[:21]+'…'
        legend.text(x+.045,y,f'F{i+1:02d} {short}' if fine else short,fontsize=5.5 if fine else 7,va='center')
    legend.set_xlim(0,1.03);legend.set_ylim(-.04,1.1)
    fig.text(.655,.115,'Entries in retained INSPIRE source',fontsize=6.5)
    freq=fig.add_axes([.655,.045,.32,.06]);freq.set_axis_off()
    for i,count in enumerate((100,10000,100000)):
        freq.scatter(i,0,s=min(1.2+.035*np.sqrt(count),60),color='#444444',alpha=.8)
        freq.text(i+.14,0,f'{count:,}',fontsize=6,va='center')
    freq.set_xlim(-.15,3.);freq.set_ylim(-1,1)
    fig.text(.205,.901,'210 clinical event tokens; cosine UMAP',fontsize=6.5)
    export(fig,'maomao_embeddings_63families' if fine else 'figure4a_maomao_embeddings',inset_axes,['inset_1','inset_2','inset_3','inset_4'],
        row_groups=[{'id':'top_insets','panels':['inset_1','inset_2']},{'id':'bottom_insets','panels':['inset_3','inset_4']}])
    (OUT/('embedding_provenance_63families.json' if fine else 'embedding_provenance.json')).write_text(json.dumps(dict(checkpoint_sha256=expected_sha,events=210,
        fitted_embeddings=len(meta['token_vocabulary']),embedding_dimension=embedding_dimension,umap_seed=1413,n_neighbors=30,min_dist=.05,metric='cosine',
        model_semantic_families=families,display_group_labels=labels,all_events_displayed=True,frequency_scope='All retained INSPIRE source clinical event entries, not just the SHAP subset',
        point_area='1.2 + 0.035 sqrt(count), capped at 60 pt² for readability; actual counts provided in CSV',
        zoom_rule='Up to twelve nearest UMAP neighbours within four times median nearest-neighbour distance of four prespecified anchors; isolated events remain isolated. No category-driven repositioning.',
        caveat='MAOMAO input embeddings are not tied to its outcome head; UMAP proximity is not a hazard, SHAP score or causal relation.'),indent=2)+'\n')

def heatmaps(preview=False):
    agg=read(OUT/'aggregation.json')
    if not agg['complete'] and not preview:raise RuntimeError('Full SHAP cohort not finished; refusing final heatmaps')
    with np.load(OUT/'event_shap_matrices_by_family.npz') as a:
        values=a['exp_mean_shap'];support=a['patients_with_feature'];names=a['event_names'].tolist();families=a['family_names'].tolist();ids=a['outcome_to_family']
    # Use identical, adequately supported rows in all three history-lag panels.
    lag_ids=np.arange(3,dtype=int)
    lag_labels=['<2 h','2–<24 h','≥24 h']
    panel_ids=['recent','intermediate','remote']
    eligible=np.flatnonzero(np.all(support>=6,axis=0))
    eligible=eligible[np.argsort(ids[eligible],kind='stable')]
    death=names.index('inhospital_death');eligible=eligible[eligible!=death]
    if len(eligible)<2:raise RuntimeError('Insufficient joint three-lag event support')
    display_ids,display_names=display_mapping(families)
    display_ids=np.array(display_ids)
    eligible=eligible[np.argsort(display_ids[ids[eligible]],kind='stable')]
    columns=np.r_[eligible,death]
    np.savez_compressed(OUT/'figure4c_display_matrix.npz',folds=values[:,eligible][:,:,columns],
        predictor_event_ids=eligible,predicted_event_ids=columns,predictor_support=support[:,eligible],
        lag_ids=lag_ids,lag_labels=np.array(lag_labels),minimum_patient_support=np.array(6))
    (OUT/'figure4c_provenance.json').write_text(json.dumps(dict(
        time_axis='Observed predictor age before the current query, not future prediction horizon',
        lag_ids=lag_ids.tolist(),lag_intervals_hours=['[0,2)','[2,24)','[24,infinity)'],
        panels=panel_ids,patients=agg['patients_completed'],minimum_patient_support_each_lag=6,
        predictor_rows=len(eligible),predicted_columns=len(columns),
        same_event_rows_and_columns_in_every_panel=True,colour_scale_shared=[.1,10],
        source_matrix_sha256=sha256(OUT/'event_shap_matrices_by_family.npz'),
        display_matrix_sha256=sha256(OUT/'figure4c_display_matrix.npz'),
        layout='Three comparable horizontal panels; enlarged canvas preserves labels and equal space per clinical display group.',
        report_time_scale_boundaries_match=True,
        report_time_mae_axis='Actual future waiting time; shares thresholds but is a different statistic.'),indent=2)+'\n')
    pal=colors(len(display_names));fig=plt.figure(figsize=(11.5,5.8));axes=[]
    fig.text(.012,.98,'c',fontsize=12,fontweight='bold',va='top')
    positions=[(.145,.10,.24,.51),(.435,.10,.24,.51),(.725,.10,.24,.51)]
    for i,pos in enumerate(positions):
        ax=fig.add_axes(pos);axes.append(ax)
        matrix=values[i][np.ix_(eligible,columns)]
        row_fam=display_ids[ids[eligible]];col_fam=display_ids[ids[columns]]
        # Allocate equal physical space per display group; individual event
        # cells remain separate and carry their original unmodified SHAP value.
        rw=np.array([1/(row_fam==f).sum() for f in row_fam]);cw=np.array([1/(col_fam==f).sum() for f in col_fam])
        re=np.r_[0,rw.cumsum()];ce=np.r_[0,cw.cumsum()]
        im=ax.pcolormesh(ce,re,matrix,cmap='RdBu_r',norm=LogNorm(.1,10),rasterized=True,shading='flat')
        ax.set_xlim(ce[0],ce[-1]);ax.set_ylim(re[-1],re[0])
        row_centers=[(re[np.flatnonzero(row_fam==f)[0]]+re[np.flatnonzero(row_fam==f)[-1]+1])/2 for f in np.unique(row_fam)]
        col_centers=[(ce[np.flatnonzero(col_fam==f)[0]]+ce[np.flatnonzero(col_fam==f)[-1]+1])/2 for f in np.unique(col_fam)]
        ax.set_yticks(row_centers,[display_names[f] for f in np.unique(row_fam)] if i==0 else ['']*len(row_centers))
        ax.set_xticks(col_centers,[display_names[f] for f in np.unique(col_fam)],rotation=90,rotation_mode='anchor',ha='left',va='center')
        ax.xaxis.tick_top();ax.tick_params(length=0);ax.tick_params(axis='y',pad=13,labelsize=6)
        ax.tick_params(axis='x',pad=12,labelsize=6)
        for edge in re[np.flatnonzero(row_fam[1:]!=row_fam[:-1])+1]:ax.axhline(edge,color='#666666',lw=.3,alpha=.6)
        for edge in ce[np.flatnonzero(col_fam[1:]!=col_fam[:-1])+1]:ax.axvline(edge,color='#666666',lw=.3,alpha=.6)
        top=fig.add_axes([pos[0],pos[1]+pos[3]+.004,pos[2],.014]);top.set_axis_off()
        top.imshow(np.array([mpl.colors.to_rgb(pal[f]) for f in np.unique(col_fam)])[None,:,:],aspect='auto')
        left=fig.add_axes([pos[0]-.016,pos[1],.010,pos[3]]);left.set_axis_off()
        left.imshow(np.array([mpl.colors.to_rgb(pal[f]) for f in np.unique(row_fam)])[:,None,:],aspect='auto')
        fig.text(pos[0]+pos[2]/2,.875,'Predicted event token',ha='center',fontsize=7)
        fig.text(pos[0]+pos[2]/2,.055,f'Predictor observed {lag_labels[i]} earlier',ha='center',fontsize=7)
    fig.text(.017,.30,'Predictor event token',rotation=90,rotation_mode='anchor',ha='center',fontsize=7)
    cb=fig.add_axes([.025,.73,.016,.14]);fig.colorbar(im,cax=cb,ticks=[.1,1,10]);cb.tick_params(labelsize=8,length=2)
    fig.text(.068,.775,'Probability\nchange, folds\nexp(mean SHAP)',fontsize=6,va='center')
    prefix='PREVIEW · ' if preview else ''
    fig.text(.145,.94,prefix+f'{len(eligible)} common supported predictors across three time scales; {len(columns)} outputs; {agg["patients_completed"]:,} patients',fontsize=7)
    export(fig,'figure4c_maomao_shap_by_family',axes,panel_ids,row_groups=[{'id':'lag_heatmaps','panels':panel_ids}])
    # Full 63-family aggregation with names as a separate, larger companion.
    with np.load(OUT/'family_shap_matrices.npz') as a:fv=a['exp_mean_shap'];fs=a['patients_with_feature']
    fig,axes=plt.subplots(1,3,figsize=(22,10),gridspec_kw={'wspace':.10})
    for i,ax in enumerate(axes):
        m=fv[i].copy();m[fs[i]<6]=np.nan
        im=ax.imshow(m,norm=LogNorm(.1,10),cmap=plt.get_cmap('RdBu_r').with_extremes(bad='#e6e6e6'),interpolation='nearest',aspect='auto',rasterized=True)
        ax.set_xticks(range(len(families)),[f'F{j+1:02d}' for j in range(len(families))],rotation=90)
        ax.set_yticks(range(len(families)),[f'F{j+1:02d} '+f.replace('_',' ') for j,f in enumerate(families)] if i==0 else ['']*len(families))
        ax.tick_params(length=0,labelsize=6);ax.set_title(('Recent (<2 h)','Intermediate (2–<24 h)','Remote (≥24 h)')[i],fontsize=9)
        ax.set_xlabel('Predicted family')
    fig.subplots_adjust(left=.14,right=.975,bottom=.08,top=.90)
    fig.text(.14,.965,'Grey rows: fewer than 6 patients with that predictor family in the shown time scale',fontsize=8)
    cb=fig.add_axes([.70,.955,.26,.013]);fig.colorbar(im,cax=cb,orientation='horizontal',ticks=[.1,1,10]);cb.tick_params(labelsize=8,length=2)
    fig.text(.70,.983,'Probability contribution folds · exp(mean SHAP)',fontsize=8)
    export(fig,'maomao_family_shap_full_matrix',list(axes),panel_ids,row_groups=[{'id':'full_family_heatmaps','panels':panel_ids}])

def extras(coarse=False):
    with np.load(OUT/'family_shap_global_importance.npz') as a:x=a['mean_abs_shap_by_case'];names=a['family_names'].tolist()
    target_names=names.copy();prefix='clinical_group' if coarse else 'family'
    if coarse:
        with np.load(OUT/'clinical_group_shap_importance.npz') as a:x=a['mean_abs_shap_by_case'];names=a['clinical_group_names'].tolist()
    exposures=pd.read_csv(OUT/'family_shap_feature_support.csv',usecols=['anonymous_case','predictor_family'])
    if coarse:
        group_map,group_names=display_mapping(target_names)
        lookup={name:group_names[group_map[i]] for i,name in enumerate(target_names)}
        exposures['predictor_family']=exposures['predictor_family'].map(lookup)
    support=exposures.drop_duplicates(['anonymous_case','predictor_family']).groupby('predictor_family').size().reindex(names,fill_value=0).to_numpy()
    mean=x.mean(0);order=np.argsort(mean)[-18:]
    rng=np.random.default_rng(42);draw=np.stack([x[rng.integers(0,len(x),len(x))][:,order].mean(0) for _ in range(500)])
    ci=np.quantile(draw,[.025,.975],axis=0)
    pd.DataFrame(dict(family=[names[i] for i in order],patients_with_input=support[order],mean_abs_log_probability_shap=mean[order],ci_lower=ci[0],ci_upper=ci[1])).to_csv(OUT/f'{prefix}_shap_importance_95ci.csv',index=False)
    fig,ax=plt.subplots(figsize=(7.2,4.6));ax.barh(range(len(order)),mean[order],color='#236b85',height=.65)
    ax.errorbar(mean[order],range(len(order)),xerr=np.vstack([mean[order]-ci[0],ci[1]-mean[order]]),fmt='none',ecolor='#333333',lw=.6,capsize=1.5)
    ax.set_yticks(range(len(order)),[names[i].replace('_',' ')+f' (n={support[i]:,})' for i in order]);ax.set_xlabel('Mean absolute family SHAP (log next-event probability)')
    ax.set_title(f'MAOMAO input family importance · {len(x):,} held-out patients',fontsize=9)
    fig.subplots_adjust(left=.31,bottom=.13,right=.98,top=.92)
    fig.text(.31,.018,'n: patients with observed input family; zeros included in cohort mean.',fontsize=6)
    export(fig,f'maomao_shap_{prefix}_importance', [ax],['importance'])
    # Signed family contributions to the death probability, across actual patients.
    meta=read(DATA/'event_sequence_meta.json');death=meta['outcome_family_vocabulary'].index('death');ids=np.array(meta['outcome_to_family']);nevents=len(meta['outcome_vocabulary'])
    if coarse:
        mapping,_=display_mapping(target_names);ids=np.array(mapping)[ids]
    signed=np.zeros_like(x);present=np.zeros_like(x,dtype=bool)
    for j,p in enumerate(sorted((OUT/'patients').glob('*.npz'))[:len(x)]):
        with np.load(p) as a:
            for g,val in zip(a['input_group_ids'],a['shap_values']):
                f=ids[int(g)//3];signed[j,f]+=val[nevents+death];present[j,f]=True
    ranking=np.argsort(np.abs(signed).mean(0))[-12:]
    np.savez_compressed(OUT/f'death_{prefix}_shap_by_patient.npz',signed_shap=signed,family_present=present,family_names=np.array(names))
    fig,ax=plt.subplots(figsize=(7.2,4.6));rng=np.random.default_rng(42)
    for row,f in enumerate(ranking):
        keep=present[:,f];val=signed[keep,f];y=row+rng.uniform(-.27,.27,len(val))
        ax.scatter(val,y,s=2,alpha=.15,c=np.where(val>=0,'#bd334d','#236b85'),rasterized=True,linewidths=0)
    ax.axvline(0,color='#888888',lw=.6);ax.set_yticks(range(len(ranking)),[names[i].replace('_',' ') for i in ranking]);ax.set_xlabel('Family SHAP on log next-event death probability')
    ax.set_title('Signed patient-level family contributions; all observed cases shown',fontsize=9)
    fig.subplots_adjust(left=.31,bottom=.13,right=.98,top=.92)
    export(fig,f'maomao_shap_death_{prefix}_distribution',[ax],['death_distribution'])
    # Local explanations on a prespecified anonymous case, not chosen for effect size.
    with np.load(OUT/'patients/case_00000.npz') as a:
        group=a['input_group_ids'];v=a['shap_values'];base=a['baseline_log_family_probability'];full=a['full_log_family_probability']
    family_v=np.zeros((len(names),len(target_names)));has=np.zeros(len(names),dtype=bool)
    for g,val in zip(group,v):
        f=ids[int(g)//3];family_v[f]+=val[nevents:];has[f]=True
    targets=[target_names.index('acute_kidney_injury'),death]
    fig=plt.figure(figsize=(7.2,6.0));axes=[];records=[]
    for panel,target in enumerate(targets):
        pos=[.40,.58 if panel==0 else .13,.49,.29];ax=fig.add_axes(pos);axes.append(ax)
        observed=np.flatnonzero(has)
        order=observed[np.argsort(np.abs(family_v[observed,target]))[-6:]]
        residual=float(family_v[:,target].sum()-family_v[order,target].sum())
        effects=np.r_[residual,family_v[order,target]]
        labels=['Other observed families']+[names[f].replace('_',' ') for f in order]
        start=0.;starts=[];ends=[]
        for row,(effect,label) in enumerate(zip(effects,labels)):
            end=start+effect;starts.append(start);ends.append(end)
            lo,hi=sorted((start,end));tip=min(.10,(hi-lo)*.2);color='#e6a2ad' if effect>=0 else '#8abdcf'
            vertices=[(start,row-.28),(end-tip if effect>=0 else end+tip,row-.28),(end,row),(end-tip if effect>=0 else end+tip,row+.28),(start,row+.28)]
            ax.add_patch(Polygon(vertices,facecolor=color,edgecolor='#666666',linewidth=.45))
            # A fixed value column keeps text away from tiny arrow boundaries.
            yfig=pos[1]+pos[3]*(row+.55)/(len(effects)+.10)
            fig.text(.925,yfig,f'×{np.exp(effect):.2g}',ha='left',va='center',fontsize=6)
            records.append(dict(anonymous_case=0,target_family=target_names[target],predictor=label,log_probability_shap=effect,contribution_fold=float(np.exp(effect))))
            start=end
        lower=min(starts+ends);upper=max(starts+ends);span=max(upper-lower,1)
        ax.set_xlim(lower-span*.12,upper+span*.15);ax.set_ylim(-.55,len(effects)-.45)
        ax.set_yticks(range(len(labels)),labels);ax.tick_params(axis='y',length=0,labelsize=6.5)
        ticks=np.arange(np.floor(lower),np.ceil(upper)+1)
        if len(ticks)>6:ticks=np.linspace(lower,upper,5)
        ax.set_xticks(ticks,[f'{np.exp(t):.2g}' for t in ticks]);ax.set_xlabel('Relative next-event family probability (folds)',fontsize=7)
        ratio=np.exp(full[nevents+target]-base[nevents+target]);prob=np.exp(full[nevents+target])
        ax.set_title(f'{target_names[target].replace("_"," ").capitalize()} · probability {prob:.3g} · relative {ratio:.2g}×',fontsize=7.5,pad=9)
    fig.text(.03,.94,'MAOMAO local family SHAP · anonymous case 00000',fontsize=10)
    fig.text(.03,.025,'Baseline: this patient’s fixed time grid, intensity, static and other context, with clinical event contents masked.',fontsize=6)
    export(fig,f'maomao_shap_local_{prefix}_waterfalls',axes,['aki','death'],column_groups=[{'id':'local_targets','panels':['aki','death']}])
    pd.DataFrame(records).to_csv(OUT/f'local_{prefix}_waterfall_values.csv',index=False)

def main():
    global FIG
    parser=argparse.ArgumentParser();parser.add_argument('--embedding-only',action='store_true');parser.add_argument('--preview',action='store_true');parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.preview:FIG=ROOT/'outputs/maomao_family_shap_20260929/preview'
    if args.output:FIG=args.output
    OUT.mkdir(parents=True,exist_ok=True);embedding();embedding(fine=True)
    if not args.embedding_only:
        heatmaps(args.preview);extras()
        if (OUT/'clinical_group_shap_importance.npz').exists():extras(coarse=True)

if __name__=='__main__':main()
