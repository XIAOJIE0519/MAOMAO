"""Render a source-backed, six-cohort scientific benchmark table."""
from pathlib import Path
import csv, json, math, hashlib, sys, textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/maomao_benchmark_table_20261001'
SKILL_SCRIPTS = Path.home() / '.codex/skills/nature-figure/scripts'
sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment

MODELS = [('maomao', 'MAOMAO (Calibration)'), ('ann', 'ANN'),
          ('xgboost', 'XGBoost'), ('logistic_regression', 'LogisticRegression'),
          ('univariate', 'Univariate')]
COHORTS = [('inspire','INSPIRE'), ('mimic','MIMIC-IV'), ('mover','MOVER'),
           ('eicu','eICU'), ('sicdb','SICdb'), ('surgical_pooled','Pooled')]
METRICS = [('micro_auprc','AUPRC'), ('micro_auroc','AUROC'),
           ('all_true_events_hit_at_10','All-hit@10'), ('recall_at_10','Recall@10')]

def read(path): return json.loads(path.read_text())

def sig3(value):
    return f'{value:.3f}'

def gather():
    base = ROOT/'outputs/final_experiment_results_20260923'
    internal = {
        'maomao': base/'model_metrics/maomao_internal.json',
        'ann': base/'baseline_metrics/common_full_validation/ann_fullscale_refined_metrics.json',
        'xgboost': base/'classical_ml_fullscale/xgboost/conservative_three_rounds_label_serial_20260928/metrics.json',
        'logistic_regression': base/'baseline_metrics/logistic_fullscale_internal.json',
        'univariate': base/'baseline_metrics/univariate_fullscale_internal.json',
    }
    retrieval_path = OUT/'internal_all_hit_10.json'
    retrieval = read(retrieval_path)
    provenance = [retrieval_path]
    records = []
    for site, label in COHORTS:
        reports = {}
        for model, display in MODELS:
            path = internal[model] if site == 'inspire' else ROOT/f'outputs/external_validation_final_maomao_uniform/{site}/{model}/metrics.json'
            provenance.append(path)
            reports[model] = read(path)
            if site == 'inspire':
                assert retrieval[model]['rows'] == reports[model]['event_targets'] == 1563972
                reports[model]['all_true_events_hit_at_10'] = retrieval[model]['all_true_events_hit_at_10']
        if site != 'inspire':
            path = ROOT/f'outputs/external_validation_final_maomao_uniform/{site}/maomao/metrics_uncalibrated.json'
            provenance.append(path)
            raw = read(path)
            assert raw['event_targets'] == reports['maomao']['event_targets']
        for metric, metric_label in METRICS:
            row = dict(cohort=label, metric=metric_label, metric_key=metric,
                       evaluation_rows=reports['maomao']['event_targets'], values={})
            for model, display in MODELS:
                value = float(reports[model][metric])
                assert math.isfinite(value) and 0 <= value <= 1
                row['values'][model] = dict(raw=float(raw[metric]) if model=='maomao' and site!='inspire' else value,
                                          calibrated=value if model=='maomao' and site!='inspire' else None,
                                          state='calibrated' if site!='inspire' else 'raw')
            candidates = [v['raw'] for v in row['values'].values()]
            candidates += [v['calibrated'] for v in row['values'].values() if v['calibrated'] is not None]
            row['best'] = max(candidates)
            for cell in row['values'].values():
                cell['raw_best'] = math.isclose(cell['raw'], row['best'], rel_tol=0, abs_tol=1e-12)
                cell['calibrated_best'] = cell['calibrated'] is not None and math.isclose(cell['calibrated'], row['best'], rel_tol=0, abs_tol=1e-12)
            records.append(row)
    assert len(records) == 24
    with (OUT/'benchmark_source_data.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=['cohort','metric','model','evaluation_rows','raw_or_baseline','maomao_calibrated','raw_best','calibrated_best'])
        writer.writeheader()
        for row in records:
            for model, display in MODELS:
                cell=row['values'][model]
                writer.writerow(dict(cohort=row['cohort'], metric=row['metric'], model=display,
                                     evaluation_rows=row['evaluation_rows'],raw_or_baseline=cell['raw'],
                                     maomao_calibrated=cell['calibrated'],raw_best=cell['raw_best'],calibrated_best=cell['calibrated_best']))
    (OUT/'benchmark_values.json').write_text(json.dumps(records, indent=2)+'\n')
    hashes=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(set(provenance))]
    (OUT/'provenance.json').write_text(json.dumps(dict(sources=hashes, decimal_places=3,
        best_rule='Maximum full-precision value among all displayed estimates in each row; exact ties all bold.',
        baseline_state='External baseline columns use their current calibrated outputs.',
        uncertainty='Point estimates only, requested table has no confidence intervals.',
        row_count=24, model_columns=5, figure_size_mm=[100,220.8], layout_note='MAOMAO original (calibrated) on one line.', pooled='NTUH + ASAC/EDS + UQ; overlaps component sources.'),indent=2)+'\n')
    lines=['# MAOMAO benchmark comparison', '', '| Cohort | Metric | '+' | '.join(v for _,v in MODELS)+' |',
           '|---|---|'+'---:|'*5]
    for row in records:
        values=[]
        for model,_ in MODELS:
            cell=row['values'][model]
            text=sig3(cell['raw'])
            if cell['raw_best']: text='**'+text+'**'
            if cell['calibrated'] is not None:
                after=sig3(cell['calibrated'])
                if cell['calibrated_best']:after='**'+after+'**'
                text+=' ('+after+')'
            values.append(text)
        lines.append('| '+row['cohort']+' | '+row['metric']+' | '+' | '.join(values)+' |')
    lines += ['', 'MAOMAO: original (calibrated); INSPIRE has no calibrated value.',
              'AUPRC and AUROC are micro metrics. All-hit@10 requires every true event to appear among the top ten predictions.',
              'Values use three decimal places. Bold uses unrounded values. External baseline columns show calibrated outputs. Pooled comprises NTUH, ASAC/EDS, and UQ.']
    (OUT/'benchmark_table.md').write_text('\n'.join(lines)+'\n')
    return records

def render(records):
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':8,'pdf.fonttype':42,'svg.fonttype':'none',
                         'axes.linewidth':0.7,'savefig.facecolor':'white'})
    width,height = 100, 170
    fig,ax=plt.subplots(figsize=(width/25.4,height/25.4))
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.set(xlim=(0,width),ylim=(height,0));ax.axis('off')
    boundaries=[1.875,12.875,25.75,47.125,57.625,69,85.25,98.125]
    row_height=4.8; top=26; bottom=top+24*row_height
    blue='#3C85BB'; pale='#E8F3FC'; strong='#BBDFFF'; ink='#14212B'
    ax.text(1.5,7.2,'Perioperative model comparison',fontsize=7,weight='bold',color=ink,va='center')
    ax.text(1.5,13.6,'Six cohorts · Full-cohort point estimates',fontsize=5,color='#59626A',va='center')
    ax.add_patch(Rectangle((25.75,17.6),21.375,bottom-17.6,facecolor=pale,edgecolor='none',zorder=0))
    headers=['Cohort','Metric','MAOMAO\n(Calibration)','ANN','XGBoost','Logistic\nRegression','Univariate']
    for i,text in enumerate(headers):
        ax.text((boundaries[i]+boundaries[i+1])/2,21.8,text,ha='center',va='center',
                fontsize=5.4,color=ink,linespacing=1.2)

    def cell_text(spans,x0,x1,y):
        renderer=fig.canvas.get_renderer()
        sizes=[]
        for text,bold in spans:
            prop=FontProperties(family='Arial',size=6.2,weight='bold' if bold else 'normal')
            w,_,_=renderer.get_text_width_height_descent(text,prop,False)
            sizes.append(w / fig.dpi * 25.4)
        x=(x0+x1-sum(sizes))/2
        assert sum(sizes) < x1-x0-0.4, 'Cell text exceeds width'
        for (text,bold),w in zip(spans,sizes):
            ax.text(x,y,text,ha='left',va='center',fontsize=6.2,
                    weight='bold' if bold else 'normal',color=ink)
            x+=w

    for i,row in enumerate(records):
        y0=top+i*row_height; y=y0+row_height/2
        if i%4==0:
            ax.text(2.25,y0+2*row_height,row['cohort'],va='center',fontsize=5.8,weight='bold',color=ink)
        ax.text(13.75,y,row['metric'],va='center',fontsize=5.8,color=ink)
        for col,(model,_) in enumerate(MODELS,2):
            x0,x1=boundaries[col],boundaries[col+1];cell=row['values'][model]
            if cell['raw_best'] or cell['calibrated_best']:
                ax.add_patch(Rectangle((x0,y0),x1-x0,row_height,facecolor=strong if col==2 else '#E9EDF0',edgecolor='none',zorder=0.5))
            spans=[(sig3(cell['raw']),cell['raw_best'])]
            if cell['calibrated'] is not None:
                spans += [(' (',False),(sig3(cell['calibrated']),cell['calibrated_best']),(')',False)]
            cell_text(spans,x0,x1,y)
        if (i+1)%4==0:
            ax.plot([1.875,98.125],[y0+row_height]*2,color='#8B9298',lw=0.75,zorder=1)
        else:
            ax.plot([12.875,98.125],[y0+row_height]*2,color='#B8BDC1',lw=0.40,zorder=1)
    ax.plot([1.875,98.125],[top]*2,color='#8B9298',lw=0.75)
    ax.add_patch(Rectangle((25.75,17.6),21.375,bottom-17.6,facecolor='none',edgecolor=blue,lw=1.0,zorder=2))
    notes=[
        'MAOMAO: original (calibrated); INSPIRE has no calibrated estimate.',
        'Bold marks the highest unrounded value among all displayed estimates in each row.',
        'AUPRC and AUROC are micro metrics. All-hit@10 requires coverage of every true event.',
        'Recall@10 measures true-event coverage. Values have three decimal places; no CIs shown.',
        'External baseline columns use calibrated outputs. Pooled = NTUH + ASAC/EDS + UQ.'
    ]
    note_y=bottom+6.4
    for note in notes:
        for line in textwrap.wrap(note, width=84):
            ax.text(1.5,note_y,line,fontsize=5.0,color='#50585F',va='center')
            note_y+=2.6
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(OUT/'benchmark_table.alignment.json'),
                                       tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(OUT/'benchmark_table.pdf')
    fig.savefig(OUT/'benchmark_table.svg')
    fig.savefig(OUT/'benchmark_table.png',dpi=300)
    fig.savefig(OUT/'benchmark_table.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    render(gather())
