import spaces
import os, json, tempfile
from pathlib import Path
import torch
import gradio as gr
from inference import Predictor
from i18n import TEXT, table_html, choices

HERE = Path(__file__).parent
predictor = Predictor(os.environ.get('MAOMAO_MODEL_DIR'), device='cuda' if os.environ.get('SPACE_ID') or torch.cuda.is_available() else 'cpu')
EXAMPLE = (HERE / 'example.json').read_text()

@spaces.GPU(duration=5)
def gpu_predict(text, calibration, custom):
    return predictor.predict(text, calibration, custom)

def predict(text, calibration, custom, language='en'):
    try:
        predictor.encode(text, device='cpu')
        result = gpu_predict(text, calibration, custom)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        raise gr.Error(TEXT[language]['error'], title='Error' if language == 'en' else '错误')
    except Exception as e:
        raise gr.Error(TEXT[language]['service_error'], title='Error' if language == 'en' else '错误') from e
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as fd:
        json.dump(result, fd, indent=2)
    return table_html(result, language), result, fd.name, result

with gr.Blocks(title='MAOMAO', theme=gr.themes.Soft(primary_hue='blue'), css='footer{display:none!important}.results-scroll{overflow-x:auto}.results-table{width:100%;border-collapse:collapse;font-size:14px}.results-table th,.results-table td{padding:10px 12px;text-align:left;border-bottom:1px solid #dce3ee;white-space:nowrap}.results-table th{background:#eef4fb}.results-table tbody tr:nth-child(even){background:#f7f9fc}') as demo:
    cached = gr.State(None)
    t = TEXT['en']
    with gr.Row():
        title = gr.Markdown('# MAOMAO\n' + t['subtitle'])
        language = gr.Dropdown([('🌐 English', 'en'), ('🌐 中文', 'zh')], value='en', show_label=False, container=False, scale=0, min_width=145, filterable=False, elem_id='language-switch')
    links = gr.Markdown(t['links'])
    with gr.Row():
        with gr.Column():
            text = gr.Code(value=EXAMPLE, language='json', label=t['input'], lines=13)
            schema = gr.Markdown(t['schema'])
            example = gr.Button(t['example'])
        with gr.Column():
            with gr.Accordion(t['calibration'], open=True) as calibration_panel:
                calibration = gr.Dropdown(choices(list(predictor.calibrations), 'en'), value='None', label=t['dataset'], filterable=False)
                custom = gr.Code(value='{"temperature":1.0}', language='json', label=t['custom'], lines=3)
                calibration_help = gr.Markdown(t['cal_help'])
            run = gr.Button(t['run'], variant='primary')
    table = gr.HTML(value=table_html(None, 'en'))
    output_help = gr.Markdown(t['output_help'])
    with gr.Accordion(t['full'], open=False) as full:
        output = gr.JSON(label=t['json'])
        download = gr.File(label=t['download'])

    def change_language(lang, selected, result):
        t = TEXT[lang]
        return (gr.update(value='# MAOMAO\n' + t['subtitle']), gr.update(value=t['links']),
                gr.update(label=t['input']), gr.update(value=t['schema']), gr.update(value=t['example']),
                gr.update(label=t['calibration']), gr.update(choices=choices(list(predictor.calibrations), lang), value=selected, label=t['dataset']),
                gr.update(label=t['custom']), gr.update(value=t['cal_help']), gr.update(value=t['run']),
                gr.update(value=table_html(result, lang)),
                gr.update(value=t['output_help']), gr.update(label=t['full']), gr.update(label=t['json']), gr.update(label=t['download']))

    language.change(change_language, [language, calibration, cached],
                    [title, links, text, schema, example, calibration_panel, calibration, custom, calibration_help, run, table, output_help, full, output, download],
                    queue=False, api_name=False)
    example.click(lambda: EXAMPLE, outputs=text, queue=False, api_name=False)
    run.click(predict, [text, calibration, custom, language], [table, output, download, cached], api_name='predict', show_progress='hidden')

if __name__ == '__main__':
    demo.queue(max_size=32).launch(show_api=False)
