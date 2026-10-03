"""User-directed main/supplement figure architecture from actual saved sources."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch, Polygon
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.ticker import MaxNLocator

B=None
BLUE='#02539F'; ROSE='#B30337'; SKY='#0F73B7'; PURPLE='#946FB0'; GREEN='#77BA47'; CORAL='#E34234'
METHOD_COLORS={'univariate':BLUE,'logistic_regression':PURPLE,'xgboost':GREEN,'ann':CORAL,'maomao':ROSE,'after':ROSE}
GROUP_COLORS=[BLUE,SKY,GREEN,PURPLE,'#F1AD6D',CORAL,'#EC777B','#4F7BAF','#A9C9B5','#49BCD6','#B8A1C3','#F68967','#72A8D9','#A39FE7','#A0E9B9','#F39AC8','#00468B']
CLINICAL_COLORS=[ROSE,BLUE,GREEN,PURPLE,CORAL,SKY]
SITE_MARKERS={'inspire':'o','mimic':'s','mover':'^','eicu':'D','sicdb':'v','surgical_pooled':'P'}
SHORT_GROUP={'Haemodynamics':'Haemodyn.','Electrolytes':'Electrolytes','Inflammation':'Inflammation','Ventilation':'Ventilation','Temperature':'Temperature','Care transitions':'Care transition','Blood / bleeding':'Blood/bleeding','Oxygenation':'Oxygenation','Metabolism':'Metabolism','Liver / albumin':'Liver/albumin','Acid–base':'Acid–base','Coagulation':'Coagulation','Renal':'Renal','Neurologic':'Neurologic','Cardiac injury':'Cardiac injury','Organ support':'Organ support','Death':'Death'}
def install(base):
    global B
    B=base;B.COLORS.update(METHOD_COLORS)
    for i in (2,3,4,5):setattr(base,f'figure{i}',globals()[f'figure{i}'])

def heading(fig,height,text,sub):
    fig.text(.055,1-.20/height,text,fontsize=10,fontweight='bold',va='top')
    fig.text(.055,1-.47/height,sub,fontsize=8,va='top')

def provenance(name,paths,**extra):
    (B.DATA/f'{name}.provenance.json').write_text(json.dumps(dict(name=name,
      sources=[{'path':str(p.relative_to(B.ROOT)),'sha256':B.sha256(p)} for p in paths],**extra),indent=2)+'\n')

def figure2():
    from maomao_figure2_by_database import render
    render(B)

def _box_stats(values):
    v=np.asarray(values,dtype=float);valid=np.isfinite(v);v=v[valid]
    if not len(v):raise RuntimeError('Empty box distribution')
    q1,med,q3=np.quantile(v,[.25,.5,.75]);iqr=q3-q1;inlier=v[(v>=q1-1.5*iqr)&(v<=q3+1.5*iqr)]
    return dict(q1=float(q1),med=float(med),q3=float(q3),whislo=float(inlier.min()),whishi=float(inlier.max()),fliers=[],n=len(v),nonfinite=int((~valid).sum()),minimum=float(v.min()),maximum=float(v.max()),hidden_tail_points=int(np.sum((v<inlier.min())|(v>inlier.max()))))

def _boxes(ax,items,labels,colors,values,metric,records):
    stats=[]
    for model,v in zip(items,values):
        s=_box_stats(v);records.append(dict(model=model,metric=metric,**s));stats.append({k:s[k] for k in ('q1','med','q3','whislo','whishi','fliers')})
    bp=ax.bxp(stats,positions=np.arange(len(items))[::-1],vert=False,widths=.40,showfliers=False,patch_artist=True,
       medianprops={'color':'#20262C','lw':1},whiskerprops={'color':'#49535C','lw':.65},capprops={'color':'#49535C','lw':.65})
    for p,c in zip(bp['boxes'],colors):p.set_facecolor(c);p.set_alpha(.70);p.set_edgecolor(c);p.set_linewidth(.8)
    ax.set_yticks(np.arange(len(items))[::-1],labels);ax.tick_params(axis='y',length=0);ax.set_ylim(-.7,len(items)-.3);ax.grid(axis='x',color='#DEEAF7',lw=.4)

def figure3():
    pe=pd.read_csv(B.SOURCE/'per_event_metrics.csv');height=11.7;fig=plt.figure(figsize=(B.WIDTH,height));axes=[];ids=[];records=[]
    heading(fig,height,'Figure 3 | Internal ablation and capacity distributions','Full validation rows; horizontal boxes summarize genuine event-wise discrimination and row-wise time error')
    modules=['maomao',*B.MODULE_LABELS];labels=['MAOMAO (reference)',*B.MODULE_LABELS.values()]
    size=['model_small','reference','model_large','context_64','context_128'];sl=[B.SCALE_LABELS[m] for m in size]
    outs=[];ol=[]
    for k in (50,100,150):outs += [f'vocab_{k}',f'reference_vocab_{k}'];ol += [f'{k} classes | specialized',f'{k} classes | MAOMAO']
    outs += ['maomao'];ol += ['210 classes | MAOMAO']
    blocks=[('a','Module removals',modules,labels,7.9,1.95),('d','Model size and context',size,sl,5.72,1.35),('g','Matched output tasks',outs,ol,3.25,1.75)]
    for letter,title,models,names,bottom,h in blocks:
        fig.text(.055,(bottom+h+.19)/height,f'{letter}  {title}',fontsize=10,fontweight='bold')
        for col,metric in enumerate(('auprc','auroc')):
            ax=fig.add_axes([.295+col*.355,bottom/height,.29,h/height]);axes.append(ax);ids.append(f'{letter}_{metric}')
            vectors=[]
            for model in models:
                group='internal' if model=='maomao' else ('module_ablations' if model in B.MODULE_LABELS else 'scale_ablations')
                part=pe[(pe.group==group)&(pe.site=='inspire')&(pe.model==model)&(pe.calibration_state=='raw')]
                expected=int(model.split('_')[-1]) if model.startswith(('vocab_','reference_vocab_')) else 210
                if len(part)!=expected:raise RuntimeError(f'{model}: incomplete event-wise box source')
                vectors.append(part[metric].to_numpy())
            colors=[ROSE if m in ('maomao','reference') or m.startswith('reference_vocab') else [BLUE,PURPLE,GREEN,SKY,CORAL][i%5] for i,m in enumerate(models)]
            _boxes(ax,models,names if col==0 else ['']*len(models),colors,vectors,metric,records)
            ax.set_xlim(0,1);ax.set_xticks([0,.5,1]);ax.set_xlabel('Event AP' if col==0 else 'Event AUROC')
    fig.text(.055,2.68/height,'j  Future waiting-time strata | full-row absolute-error distributions',fontsize=10,fontweight='bold')
    for si,(low,high,title) in enumerate([(0,2,'<2 h'),(2,24,'2–<24 h'),(24,np.inf,'≥24 h')]):
        ax=fig.add_axes([.295+si*.222,1.00/height,.188,1.20/height]);axes.append(ax);ids.append(f'time_{si}');vectors=[]
        for model in size:
            path=B.SOURCE/('internal/maomao' if model=='reference' else 'scale_ablations/'+model)/'time_predictions_no_identifiers.npz';a=np.load(path)
            t=a['actual_wait_hours'];v=a['absolute_error_hours'];keep=np.isfinite(t)&np.isfinite(v)&(t>=low)&(t<high);vectors.append(v[keep])
        _boxes(ax,size,sl if si==0 else ['']*len(size),[BLUE,ROSE,GREEN,PURPLE,SKY],vectors,f'time_error_{si}',records)
        maxwhis=max(_box_stats(v)['whishi'] for v in vectors);ax.set_xscale('symlog',linthresh=1);ax.set_xlim(0,maxwhis*1.14)
        ticks=[x for x in (0,1,10,100,1000) if x<=maxwhis*1.14];ax.set_xticks(ticks,[str(x) for x in ticks]);ax.set_xlabel('Absolute error (h)');ax.set_title(title,fontsize=10,fontweight='bold',pad=8)
    fig.text(.055,.046,'Boxes: median and interquartile range; whiskers: 1.5×IQR. All observations enter summaries; tail points are hidden.',fontsize=8)
    fig.text(.055,.030,'AP/AUROC boxes describe between-event heterogeneity, not training repeats or confidence intervals of micro metrics.',fontsize=8)
    fig.text(.055,.014,'Output-task comparisons use the same selected labels within each pair. Time boxes use a symlog hour axis.',fontsize=8)
    B.save(fig,3,axes,ids,row_groups=[{'id':letter,'panels':[f'{letter}_auprc',f'{letter}_auroc']} for letter,_,_,_,_,_ in blocks]+[{'id':'time','panels':[f'time_{i}' for i in range(3)]}])
    pd.DataFrame(records).to_csv(B.DATA/'figure_3_box_statistics.csv',index=False)
    B.write_sources(3,[B.SOURCE/'per_event_metrics.csv',B.SOURCE/'time_scales.csv',B.OUT/'external_ablation_row_verification.json',B.DATA/'figure_3_box_statistics.csv'],dict(main_panels=['a','d','g','j'],box_unit={'AP_AUROC':'event classes','time':'all full-validation target rows within true future-wait strata'},whiskers='1.5 IQR',confidence_intervals_shown_in_supplement=True))
    external_supplement()

def external_supplement():
    external,source_paths=B.ablation_external_frame();internal=B.metric_frame();height=19.8;fig=plt.figure(figsize=(B.WIDTH,height));axes=[];ids=[];rows=[];cursor=18.65
    heading(fig,height,'Supplementary Figure S3 | External ablations before and after calibration','Open blue circles: before calibration; filled rose diamonds: after calibration; horizontal whiskers: 95% CIs')
    modules=['maomao',*B.MODULE_LABELS];ml=['MAOMAO (reference)',*B.MODULE_LABELS.values()]
    size=['model_small','maomao','model_large','context_64','context_128'];sl=[B.SCALE_LABELS[x] for x in ['model_small','reference','model_large','context_64','context_128']]
    outs=[];ol=[]
    for k in (50,100,150):outs += [f'vocab_{k}',f'reference_vocab_{k}'];ol += [f'{k} classes | specialized',f'{k} classes | MAOMAO']
    outs+=['maomao'];ol+=['210 classes | MAOMAO']
    for letter,models,labels,metric,title,h in [('b',modules,ml,'micro_auprc','Module removals | AP',2.1),('c',modules,ml,'micro_auroc','Module removals | AUROC',2.1),('e',size,sl,'micro_auprc','Size and context | AP',1.55),('f',size,sl,'micro_auroc','Size and context | AUROC',1.55),('h',outs,ol,'micro_auprc','Matched output tasks | AP',1.95),('i',outs,ol,'micro_auroc','Matched output tasks | AUROC',1.95)]:
        bottom=cursor-h;fig.text(.055,(cursor+.30)/height,f'{letter}  {title}',fontsize=10,fontweight='bold');local=[]
        group_points=[B.get_point(internal if m=='maomao' else external,'external' if m=='maomao' else 'external_ablation',site,m,metric,state)
                      for site in B.SITES[1:] for m in models for state in ('before','after')]
        upper=max(r.ci_upper for r in group_points);lower=min(r.ci_lower for r in group_points)
        for si,site in enumerate(B.SITES[1:]):
            # Leave room between fractional endpoint ticks on adjacent panels.
            ax=fig.add_axes([.295+si*.137,bottom/height,.100,h/height]);axes.append(ax);pid=f'{letter}_{site}';ids.append(pid);local.append(pid)
            for yi,m in zip(np.arange(len(models))[::-1],models):
                for state,dy,marker,color in [('before',.15,'o',BLUE),('after',-.15,'D',ROSE)]:
                    r=B.get_point(internal if m=='maomao' else external,'external' if m=='maomao' else 'external_ablation',site,m,metric,state)
                    B.draw_point(ax,r,yi+dy,color,marker,filled=state=='after');rows.append(dict(panel=letter,site=site,model=m,state=state,metric=metric,value=r.value,ci_lower=r.ci_lower,ci_upper=r.ci_upper,rows=int(r.rows)))
            ax.set_yticks(np.arange(len(models))[::-1],labels if si==0 else ['']*len(models));ax.tick_params(axis='y',length=0);ax.set_ylim(-.6,len(models)-.4)
            ax.set_xlim((0,min(1,max(.4,np.ceil((upper+.01)/.05)*.05))) if metric=='micro_auprc' else (max(0,min(.5,np.floor((lower-.01)/.05)*.05)),1))
            ax.xaxis.set_major_locator(MaxNLocator(3));ax.grid(axis='x',lw=.4,color='#DEEAF7')
            ax.set_title({'mimic':'MIMIC-IV','mover':'MOVER','eicu':'eICU','sicdb':'SICdb','surgical_pooled':'Surgical pool'}[site],fontsize=8,pad=8);ax.set_xlabel('AP' if metric=='micro_auprc' else 'AUROC')
        cursor=bottom-1.04
    fig.text(.055,.035,'Paired markers share sealed 10% targets. V5 temperature + bias refitting uses every eligible row of the other 90%.',fontsize=8)
    fig.text(.055,.020,'These are full-row micro metrics with declared target-row CI approximations; output-task AP is comparable within pairs.',fontsize=8)
    groups=[{'id':l,'panels':[f'{l}_{s}' for s in B.SITES[1:]]} for l in ('b','c','e','f','h','i')]
    B.save(fig,'Supplementary_S3_external',axes,ids,row_groups=groups)
    pd.DataFrame(rows).to_csv(B.DATA/'supplementary_S3_points.csv',index=False)
    provenance('supplementary_S3',source_paths,panels=['b','c','e','f','h','i'],marker_definitions={'open_blue_circle':'MAOMAO variant before calibration','filled_rose_diamond':'same frozen variant after calibration'})

def _clinical_sources():
    dictionary=B.read(B.DATA/'clinical_curve_dictionary.json');curves=np.load(B.DATA/'clinical_common_curves.npz');scores=pd.read_csv(B.DATA/'clinical_common_metrics.csv')
    return dictionary,curves,scores

def _curve_axis(ax,endpoint,cohort,curves,scores,kind):
    maximum=max(float(curves[s['prefix']+'__precision'][curves[s['prefix']+'__recall']>0].max()) for s in cohort['series']) if kind=='pr' else 1
    pr_limit=min(1.03,max(.01,maximum*1.10))
    for si,spec in enumerate(cohort['series']):
        prefix=spec['prefix'];color=CLINICAL_COLORS[si];name=B.CLINICAL_SHORT.get(spec['name'],spec['name'])
        metric='auroc' if kind=='roc' else 'average_precision';value=scores[(scores.endpoint==endpoint)&(scores.model==spec['name'])&(scores.metric==metric)].iloc[0].value
        label=f'{name}  {value:.3f}';lw=1.35 if si==0 else .9
        if kind=='roc':ax.plot(curves[prefix+'__fpr'],curves[prefix+'__tpr'],color=color,lw=lw,ls='-',label=label)
        else:
            # Clip only the conventional precision=1, recall=0 sentinel to the
            # adaptive viewport; every positive-recall empirical point is intact.
            recall=curves[prefix+'__recall'];precision=curves[prefix+'__precision'].copy()
            precision[recall==0]=np.minimum(precision[recall==0],pr_limit)
            ax.step(recall,precision,where='pre',color=color,lw=lw,ls='-',label=label)
    if kind=='roc':
        ax.plot([0,1],[0,1],color='#B8A1C3',lw=.45,ls=':');ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='False positive rate',ylabel='True positive rate')
    else:
        ax.axhline(cohort['prevalence'],color='#B8A1C3',lw=.5,ls=':');ax.set(xlim=(0,1),ylim=(0,pr_limit),xlabel='Recall',ylabel='Precision')
    ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(4))
    legend_options={'loc':'lower right' if kind=='roc' else 'upper right','borderaxespad':.35}
    if endpoint=='rbc_6h' and kind=='roc':legend_options={'loc':'upper left','bbox_to_anchor':(0,-.14),'ncol':2,'columnspacing':.8,'borderaxespad':0}
    ax.legend(**legend_options,handlelength=1.15,handletextpad=.4,labelspacing=.18)

def figure4():
    dictionary,curves,scores=_clinical_sources();selected=[k for i,k in enumerate(dictionary) if i not in (1,3,7,8)]
    height=11.1;fig=plt.figure(figsize=(B.WIDTH,height));axes=[];ids=[]
    heading(fig,height,'Figure 4 | Six clinical endpoints on common complete cases','Frozen clinical-score study; solid lines for every model; legend values are empirical AUROC')
    for i,endpoint in enumerate(selected):
        row,col=divmod(i,2);bottom=(7.64-row*3.04)/height;ax=fig.add_axes([.105+col*.47,bottom,.365,2.21/height]);axes.append(ax);ids.append(endpoint)
        cohort=dictionary[endpoint];_curve_axis(ax,endpoint,cohort,curves,scores,'roc')
        B.panel_title(ax,chr(97+i),B.ENDPOINT_TITLES[endpoint]);fig.text(.105+col*.47,bottom+2.21/height+.34/height,f'n={cohort["complete_episodes"]:,}; events={cohort["events"]}; prevalence={cohort["prevalence"]:.2%}',fontsize=8)
    fig.text(.055,.044,'Endpoint-specific common cases; original patient-disjoint 85:15 clinical study, separate from the primary 90:10 event task.',fontsize=8)
    fig.text(.055,.026,'Adapted scores are marked *. All ten ROC/PR, calibration and decision curves are supplied as Supplementary S4.1–S4.10.',fontsize=8)
    fig.text(.055,.010,'No MAOMAO-minus-score panel is displayed. Numerical metrics and uncertainty remain in the source tables.',fontsize=8)
    B.save(fig,4,axes,ids,row_groups=[{'id':f'row_{i}','panels':selected[2*i:2*i+2]} for i in range(3)],column_groups=[{'id':f'column_{i}','panels':selected[i::2]} for i in range(2)])
    B.write_sources(4,[B.DATA/'clinical_common_curves.npz',B.DATA/'clinical_common_metrics.csv',B.DATA/'clinical_curve_dictionary.json'],dict(main_endpoints=selected,excluded_original_panels=['b','d','h','i'],main_roc_count=6,all_model_lines_solid=True,roc_legend='inside lower right',maomao_minus_score_panels=False))
    clinical_supplements(dictionary,curves,scores)

def clinical_supplements(dictionary,curves,scores):
    proof=B.read(B.DATA/'clinical_risk_preparation.json')
    if not proof['complete'] or proof['fitted_on_sealed_test'] or proof['test_patient_overlap'] or not proof['maomao_calibrated']:raise RuntimeError('Clinical risk provenance invalid')
    cal=pd.read_csv(B.DATA/'clinical_calibration_curves.csv');dca=pd.read_csv(B.DATA/'clinical_dca_curves.csv')
    raw_cal=pd.read_csv(B.DATA/'clinical_raw_maomao_calibration_curves.csv');raw_dca=pd.read_csv(B.DATA/'clinical_raw_maomao_dca_curves.csv');views=[]
    for ei,(endpoint,cohort) in enumerate(dictionary.items()):
        height=12.8;fig=plt.figure(figsize=(B.WIDTH,height));axes=[];ids=[]
        heading(fig,height,f'Supplementary Figure S4.{ei+1} | {B.ENDPOINT_TITLES[endpoint]}',f'Common sealed cases: n={cohort["complete_episodes"]:,}; events={cohort["events"]}; prevalence={cohort["prevalence"]:.3%}')
        panels=[('ROC (original scores)','roc'),('Precision–recall (original scores)','pr'),
                ('Calibration: training-mapped risks','calibration'),('DCA: training-mapped risks','dca'),
                ('MAOMAO before risk calibration','raw_calibration'),('MAOMAO DCA before risk calibration','raw_dca')]
        for k,(title,kind) in enumerate(panels):
            row,col=divmod(k,2);bottom=(9.45-row*3.35)/height
            ax=fig.add_axes([.105+col*.47,bottom,.365,2.30/height]);axes.append(ax);ids.append(kind)
            is_raw=kind.startswith('raw_');specs=[cohort['series'][0]] if is_raw else cohort['series']
            if kind in ('roc','pr'):_curve_axis(ax,endpoint,cohort,curves,scores,kind)
            elif 'calibration' in kind:
                part=(raw_cal if is_raw else cal);part=part[part.endpoint==endpoint]
                upper=max(float(part[['mean_risk','observed_fraction']].to_numpy().max()),float(part[part.model=='MAOMAO'].ci_upper.max()))
                mx=min(1,max(.005,upper*1.12))
                for si,spec in enumerate(specs):
                    data=part[part.model==spec['name']];color=BLUE if is_raw else CLINICAL_COLORS[si]
                    label='MAOMAO raw' if is_raw else ('MAOMAO calibrated' if si==0 else B.CLINICAL_SHORT.get(spec['name'],spec['name']))
                    if si==0:
                        ax.errorbar(data.mean_risk,data.observed_fraction,yerr=np.maximum(0,np.array([data.observed_fraction-data.ci_lower,data.ci_upper-data.observed_fraction])),fmt='o-',ms=2.8,lw=1.1,elinewidth=.55,capsize=1.6,color=color,label=label)
                    else:ax.plot(data.mean_risk,data.observed_fraction,'o-',ms=2.8,lw=.85,color=color,label=label)
                ax.plot([0,mx],[0,mx],color='#B8A1C3',lw=.5,ls=':');ax.set(xlim=(0,mx),ylim=(0,mx),xlabel='Mean predicted risk' if not is_raw else 'Mean uncalibrated sigmoid',ylabel='Observed event fraction')
                handles,labels=ax.get_legend_handles_labels();order=sorted(range(len(labels)),key=lambda i:not labels[i].startswith('MAOMAO'))
                legend_options={'loc':'center right','bbox_to_anchor':(1,.40)} if cohort['events']<10 and not is_raw else {'loc':'upper left'}
                ax.legend([handles[i] for i in order],[labels[i] for i in order],**legend_options,handlelength=1,labelspacing=.18,handletextpad=.4)
                ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(4))
            else:
                xmax=min(.50,max(.005,cohort['prevalence']*3));part=(raw_dca if is_raw else dca)
                part=part[(part.endpoint==endpoint)&(part.threshold<=xmax)]
                threshold=np.unique(part.threshold);prevalence=cohort['prevalence'];all_nb=prevalence-(1-prevalence)*threshold/(1-threshold)
                for si,spec in enumerate(specs):
                    data=part[part.model==spec['name']];color=BLUE if is_raw else CLINICAL_COLORS[si]
                    label='MAOMAO raw' if is_raw else ('MAOMAO calibrated' if si==0 else B.CLINICAL_SHORT.get(spec['name'],spec['name']))
                    if si==0:ax.fill_between(data.threshold,data.ci_lower,data.ci_upper,color=color,alpha=.10,lw=0)
                    ax.plot(data.threshold,data.net_benefit,color=color,lw=1.25 if si==0 else .85,label=label)
                ax.plot(threshold,all_nb,color='#59616A',lw=.6,ls=':',label='Treat all');ax.axhline(0,color='#8A9096',lw=.6,ls=':',label='Treat none')
                h=part[part.model=='MAOMAO'];lo=min(float(part.net_benefit.min()),float(h.ci_lower.min()),float(all_nb.min()),0)
                hi=max(float(part.net_benefit.max()),float(h.ci_upper.max()),float(all_nb.max()),0);pad=max((hi-lo)*.12,.0001)
                ax.set(xlim=(0,xmax),ylim=(lo-pad,hi+pad),xlabel='Risk threshold',ylabel='Net benefit')
                ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(4))
                legend_options={'loc':'upper center','bbox_to_anchor':(.5,-.13),'ncol':4,'columnspacing':.8} if cohort['events']<10 and not is_raw else {'loc':'lower left'}
                ax.legend(**legend_options,handlelength=1,labelspacing=.15,handletextpad=.4)
            B.panel_title(ax,chr(97+k),title);views.append(dict(endpoint=endpoint,kind=kind,xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim())))
        fig.text(.055,.142,'a/b: original frozen scores. c/d: all risk mappings fitted on identical training complete cases, including MAOMAO.',fontsize=8)
        fig.text(.055,.123,'69,920 eligible training patients; full available landmarks. Internal validation and sealed test never enter fits.',fontsize=8)
        fig.text(.055,.104,'e/f retain the uncalibrated focal-loss sigmoid. Raw and calibrated MAOMAO AUROC/AP are unchanged.',fontsize=8)
        fig.text(.055,.085,'MAOMAO bars/bands: pointwise 95% patient-cluster bootstrap CIs (200 draws), conditional on frozen risk mappings.',fontsize=8)
        fig.text(.055,.066,'Equal-count bins retain ties. The same DCA threshold viewport is used before/after; all thresholds are supplied.',fontsize=8)
        fig.text(.055,.047,'Calibration axes adapt to each state; compare coordinates, not panel slopes. Adapted-risk DCA is exploratory.',fontsize=8)
        if cohort['events']<10:fig.text(.055,.028,f'Only {cohort["events"]} events: PR, calibration and decision curves are highly unstable.',fontsize=8,color=ROSE)
        B.save(fig,f'Supplementary_S4_{ei+1:02d}_{endpoint}',axes,ids,
               row_groups=[{'id':'top','panels':['roc','pr']},{'id':'middle','panels':['calibration','dca']},{'id':'bottom','panels':['raw_calibration','raw_dca']}],
               column_groups=[{'id':'left','panels':['roc','calibration','raw_calibration']},{'id':'right','panels':['pr','dca','raw_dca']}])
    pd.DataFrame(views).to_csv(B.DATA/'clinical_adaptive_axis_ranges.csv',index=False)
    provenance('supplementary_S4',[B.DATA/n for n in ['clinical_risk_preparation.json','clinical_risk_predictions.npz','clinical_calibration_curves.csv','clinical_dca_curves.csv','clinical_raw_maomao_calibration_curves.csv','clinical_raw_maomao_dca_curves.csv','clinical_calibration_dca_audit.json','clinical_probability_metrics.csv','clinical_adaptive_axis_ranges.csv']],
               endpoints=10,curve_types=['ROC','PR','training-calibrated comparison','raw MAOMAO calibration','raw MAOMAO DCA'],roc_legend='inside lower right',pr_legend='inside upper right',all_model_lines_solid=True,maomao_raw_retained=True,common_training_mapping=True)

def _human(event):
    special={'map_hypotension':'MAP hypotension','hypoxemia':'Hypoxemia','aki_stage_1_signal':'AKI stage 1','inhospital_death':'In-hospital death','systolic_bp_normalized':'SBP recovery','temperature_normalized':'Temperature recovery','heart_rate_normalized':'Heart-rate recovery','leukocytosis':'Leukocytosis','bradycardia':'Bradycardia'}
    return special.get(event,event.replace('_',' ').replace('normalized','recovery').capitalize())

def _beeswarm(values):
    values=np.asarray(values)
    if not len(values):return np.zeros(0)
    bins=np.searchsorted(np.linspace(values.min(),values.max()+1e-9,90),values,side='right');jitter=np.zeros(len(values))
    for b in np.unique(bins):
        ix=np.flatnonzero(bins==b);order=np.argsort(values[ix],kind='stable');n=len(ix);spread=min(.31,n/700)
        jitter[ix[order]]=np.linspace(-spread,spread,n)
    return jitter

def figure5():
    shap=B.SOURCE/'shap';proof=B.read(shap/'delivery_verification.json')
    if not proof['complete']:raise RuntimeError('Incomplete actual SHAP sources')
    height=17.5;fig=plt.figure(figsize=(B.WIDTH,height));axes=[];ids=[]
    heading(fig,height,'Figure 5 | Learned event geometry and patient-level contributions','9,989 holdout patients; original frozen queries and Partition SHAP; consistent clinical-group color key')
    meta=B.read(shap/'event_sequence_meta.json');mapping,names=B.display_mapping(meta['outcome_family_vocabulary']);palette=np.array(GROUP_COLORS)
    frame=pd.read_csv(shap/'embedding_event_coordinates.csv');u=frame[['umap_1','umap_2']].to_numpy();group=np.array([names.index(n) for n in frame.clinical_display_group]);size=4+frame.scatter_area_pt2.to_numpy()*1.3
    main=fig.add_axes([.065,12.65/height,.455,3.25/height]);axes.append(main);ids.append('embedding')
    for gi in range(17):
        take=group==gi;main.scatter(u[take,0],u[take,1],s=size[take],c=[palette[gi]],edgecolors='white',lw=.25,alpha=.92)
    main.set(xlabel='UMAP 1',ylabel='UMAP 2');main.xaxis.set_major_locator(MaxNLocator(4));main.yaxis.set_major_locator(MaxNLocator(4));B.panel_title(main,'a','Learned event embedding')
    anchors=['map_hypotension','hypoxemia','aki_stage_1_signal','inhospital_death'];zooms=[]
    for k,name in enumerate(anchors):
        anchor=frame.index[frame.event==name][0];dist=np.sqrt(((u-u[anchor])**2).sum(1));near=np.argsort(dist)[:8]
        all_dist=np.sqrt(((u[:,None,:]-u[None,:,:])**2).sum(-1));np.fill_diagonal(all_dist,np.inf);radius=4*np.median(all_dist.min(1));near=near[dist[near]<=radius]
        low=u[near].min(0);high=u[near].max(0);span=np.maximum(high-low,.1);low-=span*.35;high+=span*.35
        bottom=15.52-k*1.15;ax=fig.add_axes([.565,bottom/height,.15,.87/height]);axes.append(ax);pid=f'zoom_{k}';ids.append(pid);zooms.append(pid)
        ax.scatter(u[near,0],u[near,1],s=18+size[near]*.6,c=palette[group[near]],edgecolors='white',lw=.3)
        ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]));ax.set_xticks([]);ax.set_yticks([]);ax.set_title(_human(name),fontsize=8,pad=5,fontweight='bold')
        for spine in ax.spines.values():spine.set_visible(True);spine.set_color('#8995A1');spine.set_linewidth(.65)
        # Named anchors are unambiguous; two nearest neighbours are labeled in the source dictionary.
        # The title names the clinically relevant anchor; a ring identifies its
        # exact token without a duplicate label crossing nearby points.
        ax.scatter(*u[anchor],s=65,facecolor='none',edgecolor='#20262C',lw=.7)
        main.add_patch(Rectangle(low,*(high-low),fill=False,ec=palette[group[anchor]],lw=.9))
        fig.add_artist(ConnectionPatch(xyA=(high[0],(high[1]+low[1])/2),coordsA=main.transData,xyB=(0,.5),coordsB=ax.transAxes,color='#8995A1',lw=.45))
    # The single right-side key covers all plots, including restored heatmap strips.
    fig.text(.758,16.23/height,'Clinical input group',fontsize=8,fontweight='bold')
    for gi,name in enumerate(names):
        y=(15.98-gi*.195)/height;fig.add_artist(Rectangle((.758,y-.027/height),.011,.09/height,transform=fig.transFigure,color=palette[gi],lw=0));fig.text(.777,y,name,fontsize=8,va='center')
    fig.text(.758,12.45/height,'All 210 event tokens;',fontsize=8);fig.text(.758,12.28/height,'area reflects source count.',fontsize=8)
    rank=pd.read_csv(shap/'clinical_group_shap_importance_95ci.csv').sort_values('mean_abs_log_probability_shap',ascending=False);order=np.array([names.index(x) for x in rank.family])
    imp=fig.add_axes([.205,8.72/height,.275,2.90/height]);dist=fig.add_axes([.59,8.72/height,.365,2.90/height]);axes.extend([imp,dist]);ids.extend(['importance','beeswarm'])
    for yy,gi in zip(np.arange(17)[::-1],order):
        r=rank[rank.family==names[gi]].iloc[0];imp.barh(yy,r.mean_abs_log_probability_shap,height=.65,color=palette[gi],alpha=.85)
        imp.plot([r.ci_lower,r.ci_upper],[yy]*2,color='#20262C',lw=.65)
    imp.set_yticks(np.arange(17)[::-1],rank.family);imp.tick_params(axis='y',length=0);imp.set_ylim(-.7,16.7);imp.set_xlabel('Mean |log-probability SHAP|');imp.xaxis.set_major_locator(MaxNLocator(3));B.panel_title(imp,'b','Input importance (95% CI)')
    death=np.load(shap/'death_clinical_group_shap_by_patient.npz');contrib=death['signed_shap'];present=death['family_present']
    for yy,gi in zip(np.arange(17)[::-1],order):
        values=contrib[present[:,gi],gi];dist.scatter(values,yy+_beeswarm(values),s=1.2,c=[palette[gi]],alpha=.30,lw=0)
    dist.axvline(0,color='#738187',lw=.5);dist.set_yticks(np.arange(17),['']*17);dist.tick_params(axis='y',length=0);dist.set_ylim(-.7,16.7);dist.set_xlabel('SHAP to log death-family probability');dist.xaxis.set_major_locator(MaxNLocator(4));B.panel_title(dist,'c','Observed contributions')
    fig.text(.205,8.18/height,'Shared input-group order; b averages all outputs, c shows the death-family output.',fontsize=8)
    fig.text(.055,7.97/height,'d  Historical contributions at three time scales',fontsize=10,fontweight='bold')
    data=np.load(shap/'figure4c_display_matrix.npz');fam=np.array(meta['outcome_to_family']);mapped=np.array(mapping)
    row_group=mapped[fam[data['predictor_event_ids']]];col_group=mapped[fam[data['predicted_event_ids']]]
    def edges(group):
        widths=np.array([1/np.sum(group==g) for g in group]);e=np.r_[0,np.cumsum(widths)];unique=np.unique(group);centers=[(e[np.flatnonzero(group==g)[0]]+e[np.flatnonzero(group==g)[-1]+1])/2 for g in unique];return e,unique,centers
    re,rg,rc=edges(row_group);ce,cg,cc=edges(col_group)
    cmap=LinearSegmentedColormap.from_list('provided_diverging',[BLUE,'#BCD7ED','#FFFFFF','#F4E1DE',ROSE]);lag_ids=[]
    for li,label in enumerate(data['lag_labels']):
        ax=fig.add_axes([.185+li*.27,4.92/height,.23,2.47/height]);axes.append(ax);pid=f'lag_{li}';ids.append(pid);lag_ids.append(pid)
        mesh=ax.pcolormesh(ce,re,data['folds'][li],cmap=cmap,norm=LogNorm(.1,10),shading='flat',rasterized=False)
        ax.set(xlim=(0,ce[-1]),ylim=(re[-1],0));ax.set_yticks(rc,[SHORT_GROUP[names[g]] for g in rg] if li==0 else []);ax.set_xticks(cc,[SHORT_GROUP[names[g]] for g in cg],rotation=90,rotation_mode='anchor',ha='right',va='center');ax.tick_params(length=0,pad=3);ax.tick_params(axis='y',pad=10)
        for e in re[np.flatnonzero(np.diff(row_group))+1]:ax.axhline(e,color='#8995A1',lw=.25)
        for e in ce[np.flatnonzero(np.diff(col_group))+1]:ax.axvline(e,color='#8995A1',lw=.25)
        for gi in cg:
            ix=np.flatnonzero(col_group==gi);ax.add_patch(Rectangle((ce[ix[0]],-.36),ce[ix[-1]+1]-ce[ix[0]],.25,color=palette[gi],lw=0,clip_on=False))
        for gi in rg:
            ix=np.flatnonzero(row_group==gi);ax.add_patch(Rectangle((-.40,re[ix[0]]),.25,re[ix[-1]+1]-re[ix[0]],color=palette[gi],lw=0,clip_on=False))
        ax.set_title(str(label),fontsize=10,fontweight='bold',pad=14)
    cb=fig.add_axes([.30,3.66/height,.33,.08/height]);colorbar=fig.colorbar(mesh,cax=cb,orientation='horizontal',ticks=[.1,1,10]);colorbar.solids.set_rasterized(False);colorbar.solids.set_edgecolor('face');cb.set_xticklabels(['0.1','1','10']);cb.tick_params(length=2)
    fig.text(.65,3.66/height,'exp(mean log-probability SHAP)',fontsize=8,va='center')
    fig.text(.185,3.97/height,'174 predictors × 175 outputs; same coordinates, color scale and clinical-group strips.',fontsize=8)
    fig.text(.055,3.18/height,'e  Four event-level SHAP dependencies with paired clinical measurements',fontsize=10,fontweight='bold')
    dep=np.load(B.DATA/'shap_dependence_observed_history.npz');events=dep['event_names'].tolist();v=dep['last_observed_value'];phi=dep['event_shap_to_log_death']
    pairs=[('systolic_bp_normalized','heart_rate_normalized','SBP recovery','Latest SBP (mmHg)','HR (beats/min)'),('temperature_normalized','leukocytosis','Temperature recovery','Latest temperature (°C)','WBC (source units)'),('leukocytosis','temperature_normalized','Leukocytosis','Latest WBC (source units)','Temperature (°C)'),('bradycardia','systolic_bp_normalized','Bradycardia','Latest HR (beats/min)','SBP (mmHg)')]
    scatter_cmap=LinearSegmentedColormap.from_list('paired_measurement',[BLUE,'#49BCD6','#F9E2AD',ROSE]);dep_ids=[];selection=[]
    for i,(event,partner,title,xlabel,clabel) in enumerate(pairs):
        ei,pi=events.index(event),events.index(partner);keep=np.isfinite(v[:,ei])&np.isfinite(v[:,pi]);x=v[keep,ei];y=phi[keep,ei];color=v[keep,pi]
        ax=fig.add_axes([.087+i*.228,1.25/height,.183,1.45/height]);axes.append(ax);pid=f'dependence_{i}';ids.append(pid);dep_ids.append(pid)
        sc=ax.scatter(x,y,c=color,cmap=scatter_cmap,s=2.5,alpha=.6,lw=0);ax.axhline(0,color='#B8A1C3',lw=.4);ax.set_xlabel(xlabel);ax.set_ylabel('SHAP to log death' if i==0 else '');ax.set_title(title,fontsize=8,fontweight='bold',pad=7);ax.xaxis.set_major_locator(MaxNLocator(3));ax.yaxis.set_major_locator(MaxNLocator(3))
        colorax=fig.add_axes([.087+i*.228,.63/height,.183,.065/height]);cbar=fig.colorbar(sc,cax=colorax,orientation='horizontal');cbar.solids.set_rasterized(False);cbar.solids.set_edgecolor('face');colorax.xaxis.set_major_locator(MaxNLocator(3));colorax.tick_params(length=2)
        fig.text(.087+i*.228,.27/height,clabel,fontsize=8);selection.append(dict(event=event,paired_measurement=partner,patients=int(keep.sum()),all_pair_complete_patients_shown=True))
    fig.text(.055,.0045,'Conditional dependencies; paired measurements are not exact SHAP interactions or causal effects.',fontsize=8)
    B.save(fig,5,axes,ids,row_groups=[{'id':'bc','panels':['importance','beeswarm']},{'id':'lags','panels':lag_ids},{'id':'dependencies','panels':dep_ids}],column_groups=[{'id':'zooms','panels':zooms}])
    (B.DATA/'figure_5_dependency_selection.json').write_text(json.dumps(dict(selection=selection,rule='Four high-contribution event inputs spanning distinct physiological variables; every pair-complete original query shown, no outcome-based case selection',x_values='Actual last observed source measurement before the original query',y_values='Existing event-group SHAP summed across the three lags'),indent=2)+'\n')
    B.write_sources(5,[shap/'embedding_event_coordinates.csv',shap/'clinical_group_shap_importance_95ci.csv',shap/'figure4c_display_matrix.npz',shap/'death_clinical_group_shap_by_patient.npz',B.DATA/'shap_dependence_observed_history.npz',B.DATA/'shap_dependence_preparation.json',B.DATA/'figure_5_dependency_selection.json'],dict(layout_rows=['a: UMAP with four named zooms and right legend','b/c: importance and beeswarm','d: three lag maps with group-color strips','e: four measured dependence plots'],all_patients=9989,lag_labels=['<2 h','2–<24 h','≥24 h'],individual_waterfall_main=False))
    waterfall_supplement(meta,mapping,names)

def waterfall_supplement(meta,mapping,names):
    height=12.6;fig=plt.figure(figsize=(B.WIDTH,height));axes=[];ids=[];records=[];death=210+meta['outcome_family_vocabulary'].index('death');family=np.array(meta['outcome_to_family']);coarse=np.array(mapping);shap=B.SOURCE/'shap'
    heading(fig,height,'Supplementary Figure S5 | Nine fixed patient-level death-family explanations','Original cases 00000–00008; arrow waterfalls; values sum from the conditional mask baseline to the actual prediction')
    for i in range(9):
        a=np.load(shap/f'patients/case_{i:05d}.npz');con=np.zeros(17)
        np.add.at(con,coarse[family[a['input_group_ids']//3]],a['shap_values'][:,death]);rank=np.argsort(abs(con))[::-1];top=rank[:4]
        vals=list(con[top])+[float(con[rank[4:]].sum())];labels=[SHORT_GROUP[names[x]] for x in top]+['Other inputs']
        base=float(a['baseline_log_family_probability'][death]);full=float(a['full_log_family_probability'][death])
        if not np.isclose(base+sum(vals),full,atol=1e-4):raise RuntimeError('Waterfall not additive')
        row,col=divmod(i,3);bottom=(8.64-row*3.48)/height;ax=fig.add_axes([.145+col*.30,bottom,.150,2.30/height]);axes.append(ax);pid=f'case_{i}';ids.append(pid)
        cumulative=base;points=[base]
        for yy,(value,label) in zip(np.arange(5)[::-1],zip(vals[::-1],labels[::-1])):
            end=cumulative+value;left,right=sorted([cumulative,end]);span=max(abs(value),1e-9);tip=min(span*.18,.05)
            vertices=[(left,yy-.26),(right-tip,yy-.26),(right,yy),(right-tip,yy+.26),(left,yy+.26)] if value>=0 else [(right,yy-.26),(left+tip,yy-.26),(left,yy),(left+tip,yy+.26),(right,yy+.26)]
            ax.add_patch(Polygon(vertices,closed=True,facecolor=ROSE if value>=0 else BLUE,edgecolor='none'))
            # Values use a separate aligned column; tiny arrows cannot carry
            # legible 8 pt text. The original exact contributions remain in CSV.
            ax.text(1.03,(yy+.6)/5.2,f'{value:+.2f}',transform=ax.transAxes,fontsize=8,ha='left',va='center',color=ROSE if value>=0 else BLUE)
            if yy>0:ax.plot([end,end],[yy-.26,yy-.74],color='#8995A1',lw=.45)
            records.append(dict(case=i,predictor=label,value=value,start=cumulative,end=end,baseline=base,full=full));cumulative=end;points.append(end)
        ax.set_yticks(np.arange(5)[::-1],labels[::-1]);ax.tick_params(axis='y',length=0);ax.set_ylim(-.6,4.6);span=max(max(points)-min(points),.3);ax.set_xlim(min(points)-span*.08,max(points)+span*.15);ax.xaxis.set_major_locator(MaxNLocator(2));ax.set_xlabel('Log probability')
        ax.set_title(f'Case {i:05d}\np = {np.exp(full):.2g}',loc='left',fontsize=8,fontweight='bold',pad=8)
        fig.text(.145+col*.30,bottom-.64/height,f'Base {base:.2f}; full {full:.2f}',fontsize=8)
    fig.text(.055,.037,'Red arrows increase and blue arrows decrease the explained log probability. Four strongest input groups plus all others.',fontsize=8)
    fig.text(.055,.020,'Baseline retains time, intensity, phase, static and other context. These selected queries do not represent patient prognosis.',fontsize=8)
    B.save(fig,'Supplementary_S5_waterfalls',axes,ids,row_groups=[{'id':f'row_{i}','panels':[f'case_{j}' for j in range(i*3,i*3+3)]} for i in range(3)],column_groups=[{'id':f'column_{i}','panels':[f'case_{j}' for j in range(i,9,3)]} for i in range(3)])
    pd.DataFrame(records).to_csv(B.DATA/'supplementary_S5_waterfall_values.csv',index=False)
    provenance('supplementary_S5',[shap/f'patients/case_{i:05d}.npz' for i in range(9)],layout='3x3',cases=list(range(9)),target='log original death-family next-event probability',additivity_checked=True)
