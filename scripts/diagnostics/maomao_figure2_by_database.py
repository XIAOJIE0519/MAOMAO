"""Database-specific Figure 2: two five-model grouped bar charts, two tables.
Style-only adaptation of user reference; unchanged model palette and saved CIs.
No evaluation is rerun. Internal recall is never relabelled as external Hit.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SITES=('inspire','mimic','mover','eicu','sicdb','ntuh','asac','uq','surgical_pooled')
COLORS={'univariate':'#02539F','logistic_regression':'#946FB0','xgboost':'#77BA47','ann':'#E34234','maomao':'#B30337'}
NAMES={'inspire':'INSPIRE','mimic':'MIMIC-IV','mover':'MOVER','eicu':'eICU','sicdb':'SICdb','ntuh':'NTUH','asac':'ASAC','uq':'UQ','surgical_pooled':'NTUH + ASAC + UQ'}

def render(B):
    original_figure_dir=B.FIG
    B.FIG=original_figure_dir/"Figure_2_by_database"
    B.FIG.mkdir(parents=True,exist_ok=True)
    frame=B.metric_frame(); retrieval=pd.read_csv(B.SOURCE/'external_retrieval_metrics.csv')
    collected=[];outputs=[]
    for site in SITES:
        internal=site=='inspire';group='internal' if internal else 'external'
        states=[(m,'raw' if internal else 'after') for m in B.METHODS[:-1]]+([('maomao','raw')] if internal else [('maomao','before'),('maomao','after')])
        ticklabels=['Uni','LR','XGB','ANN']+(['MAOMAO'] if internal else ['Raw','Cal'])
        rowlabels=['Univariate','LR','XGBoost','ANN']+(['MAOMAO'] if internal else ['MAOMAO raw','MAOMAO cal'])
        def point(model,state,metric,kind='standard'):
            if kind=='standard':
                r=B.get_point(frame,group,site,model,metric,state)
            else:
                q=retrieval[(retrieval.site==site)&(retrieval.model==model)&(retrieval.calibration_state==state)&(retrieval.metric==metric)]
                if len(q)!=1:raise RuntimeError(f'Missing retrieval {site}/{model}/{state}/{metric}')
                r=q.iloc[0]
            collected.append({'site':site,'model':model,'state':state,'metric':metric,'value':float(r.value),'ci_lower':float(r.ci_lower),'ci_upper':float(r.ci_upper),'rows':int(r['rows'] if 'rows' in r.index else r.test_rows),'source_kind':kind})
            return r
        fig=plt.figure(figsize=(B.WIDTH,170/25.4));axes=[];ids=[]
        fig.text(.06,.98,f'Figure 2 | {NAMES[site]}',fontsize=8,fontweight='bold',va='top')
        n=int(B.get_point(frame,group,site,'maomao','micro_auprc','raw' if internal else 'after')['rows'])
        fig.text(.06,.055 if internal else .095,f'Full {"validation" if internal else "sealed test"} set: {n:,} rows; bars show existing 95% CIs.',fontsize=8)
        handles=[Patch(facecolor=COLORS[m],label=('MAOMAO calibrated' if m=='maomao' and not internal else B.LABELS[m])) for m in B.METHODS]
        if not internal:handles.append(Patch(facecolor=COLORS['maomao'],edgecolor=COLORS['maomao'],alpha=.4,label='MAOMAO raw'))
        fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.055,.93),ncol=3,fontsize=8,columnspacing=1.3)
        panels=[('a','micro_auroc','AUROC',[.09,.48,.37,.35]),
                ('b','micro_auprc','Average precision (AP)',[.60,.48,.37,.35])]
        for letter,metric,title,pos in panels:
            ax=fig.add_axes(pos);axes.append(ax);ids.append(letter)
            values=[point(m,s,metric) for m,s in states]
            positions=np.array([0,1,2,3,4] if internal else [0,1,2,3,3.81,4.19])
            for i,((m,state),r) in enumerate(zip(states,values)):
                width=.62 if internal or i<4 else .32
                ax.bar(positions[i],r.value,width=width,color=COLORS[m],edgecolor=COLORS[m],linewidth=.7,alpha=.85 if state!='before' else .40,hatch='///' if state=='before' else None,zorder=3)
                if np.isfinite(r.ci_lower) and np.isfinite(r.ci_upper):
                    ax.errorbar(positions[i],r.value,yerr=np.maximum(0,[[r.value-r.ci_lower],[r.ci_upper-r.value]]),fmt='none',ecolor='#333333',elinewidth=.65,capsize=2,capthick=.65,zorder=4)
            ceiling=min(1,max(.10,max(float(r.ci_upper) for r in values))*1.15)
            ax.set_ylim(0,ceiling);ax.set_xlim(-.6,4.6)
            ax.set_xticks(np.arange(5),['Uni','LR','XGB','ANN','MAOMAO'])
            ax.tick_params(axis='both',labelsize=8,pad=2)
            ax.set_title(f'{letter}  {title}',loc='left',fontsize=8,fontweight='bold',pad=9)
            ax.set_ylabel('Micro AUROC' if metric=='micro_auroc' else 'Micro AP',fontsize=8,labelpad=3)
            ax.grid(axis='y',color='#eeeeee',linewidth=.4,zorder=0)
        tables=[]
        if internal:
            specs=[('hit_at_1','Hit@1'),('recall_at_5','R@5'),('recall_at_10','R@10'),('mrr','MRR')];kind='standard';title='c  Event retrieval (higher is better)'
        else:
            specs=[('same_family_hit_at_1','Hit@1'),('same_family_hit_at_5','Hit@5'),('same_family_hit_at_10','Hit@10'),('all_true_families_hit_at_10','All@10')];kind='retrieval';title='c  63-family retrieval (higher is better)'
        table_specs=[('c',[.06,.13,.445,.20],title,specs,kind),('d',[.555,.13,.415,.20],'d  Probability error (lower is better)',[('brier','Brier'),('ece','ECE')],'standard')]
        for letter,pos,title,specs,kind in table_specs:
            ax=fig.add_axes(pos);ax.axis('off');axes.append(ax);ids.append(letter)
            ax.set_title(title,loc='left',fontsize=8,fontweight='bold',pad=10)
            content=[]
            for label,(model,state) in zip(rowlabels,states):
                cells=[label]
                for metric,_ in specs:
                    r=point(model,state,metric,kind);cells.append(f'{r.value:.3f}')
                content.append(cells)
            widths=([.36]+[.16]*4) if letter=='c' else [.48,.26,.26]
            table=ax.table(cellText=content,colLabels=['Model',*[x[1] for x in specs]],cellLoc='center',colWidths=widths,bbox=[0,0,1,1])
            table.auto_set_font_size(False);table.set_fontsize(8)
            for (ri,ci),cell in table.get_celld().items():
                cell.set_edgecolor('#D0D7DE');cell.set_linewidth(.45)
                if ri==0:cell.set_facecolor('#DEEAF7');cell.get_text().set_fontweight('bold')
                elif ci==0:cell.get_text().set_color(COLORS[states[ri-1][0]]);cell.get_text().set_ha('left')
                else:cell.set_facecolor('#F7FAFC' if ri%2 else 'white')
        note='Internal MAOMAO is uncalibrated. Recall@K is retained as saved; it differs from Hit@K.' if internal else 'Raw / Cal: MAOMAO before / after calibration on identical sealed 10% rows; calibration uses the other 90%.'
        fig.text(.06,.070 if not internal else .075,note,fontsize=8)
        if not internal:fig.text(.06,.047,'All@10: all true 63-family labels retrieved in the first 10 ranks. Brier uses normalized target probabilities.',fontsize=8)
        fig.text(.06,.025,'CI: saved 200-repeat target-row bootstrap; large-cohort intervals use the declared subsample-width approximation.',fontsize=8)
        name=f'Figure_2_{site}'
        B.save(fig,name,axes,ids,row_groups=[{'id':'model_comparison_bars','panels':['a','b']}])
        outputs.append(name)
    pd.DataFrame(collected).drop_duplicates().to_csv(B.DATA/'figure_2_by_database_points.csv',index=False)
    proof={'databases':SITES,'outputs':outputs,'bar_panels_per_database':2,'model_groups_per_bar_panel':5,'tables_per_database':2,'font':'Arial','font_size_pt':8,'model_colors':COLORS,'raw_after_same_rows':True,'data_reused_without_retraining':True,'sources':[{'path':str(p.relative_to(B.ROOT)),'sha256':B.sha256(p)} for p in [B.SOURCE/'metrics_long.csv',B.SOURCE/'external_retrieval_metrics.csv']]}
    (B.DATA/'figure_2_by_database_provenance.json').write_text(json.dumps(proof,indent=2)+'\n')

    B.FIG=original_figure_dir
