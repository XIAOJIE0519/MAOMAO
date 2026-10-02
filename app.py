import spaces
import os,json,tempfile
from pathlib import Path
import torch
import gradio as gr
from inference import Predictor

HERE=Path(__file__).parent
predictor=Predictor(os.environ.get('MAOMAO_MODEL_DIR'),device='cuda' if os.environ.get('SPACE_ID') or torch.cuda.is_available() else 'cpu')
EXAMPLE=(HERE/'example.json').read_text()
HEADERS=['#','事件 / Event','原始 / Raw','校正 / Calibrated','等待(h) / Wait','1h (raw)','6h (raw)','24h (raw)']

@spaces.GPU(duration=30)
def predict(text,calibration,custom):
    try:
        result=predictor.predict(text,calibration,custom)
    except (ValueError,KeyError,TypeError,json.JSONDecodeError) as e:
        raise gr.Error(str(e))
    table=[[r['rank'],r['event'],round(r['raw'],6),round(r['calibrated'],6),round(r['wait_hours'],3),round(r['risk_1h_raw'],6),round(r['risk_6h_raw'],6),round(r['risk_24h_raw'],6)] for r in result['next_event']]
    fd=tempfile.NamedTemporaryFile(mode='w',suffix='.json',delete=False)
    json.dump(result,fd,indent=2);fd.close()
    return table,result,fd.name

with gr.Blocks(title='MAOMAO',theme=gr.themes.Soft(primary_hue='blue')) as demo:
    gr.Markdown('# MAOMAO\nMulti-horizon Anticipatory Outcome Model for Anesthesia and Operations')
    gr.Markdown('[模型 / Model](https://huggingface.co/luan0519/MAOMAO) · [GitHub](https://github.com/XIAOJIE0519/MAOMAO) · [词表 / Vocabulary](https://huggingface.co/luan0519/MAOMAO/resolve/main/vocabulary.json)\n\n研究演示 / Research demo')
    with gr.Row():
        with gr.Column():
            text=gr.Code(value=EXAMPLE,language='json',label='输入 / Input JSON',lines=13)
            gr.Markdown('`static`: 年龄、性别(0/1)、ASA、急诊(0/1)、体重(kg)、身高(cm)。\n`events`: `time_min` 从记录开始的分钟；`token` 词表名称；`value` 原始数值(可省略)。\n\n`static`: age, male(0/1), ASA, emergency(0/1), weight(kg), height(cm).\n`events`: minutes since record start, vocabulary token, optional raw value.')
            example=gr.Button('加载示例 / Load example')
            with gr.Accordion('校准 / Calibration',open=True):
                calibration=gr.Dropdown(['None']+list(predictor.calibrations)+['Custom'],value='None',label='数据来源 / Dataset')
                custom=gr.Code(value='{"temperature":1.0}',language='json',label='自定义参数 / Custom parameters',lines=3)
                gr.Markdown('`p = softmax((logits + bias) / temperature)`\n\n预设仅对应其来源；新来源需本地拟合。只校准下一事件分布；时间与 1/6/24h 输出保持原始。\nPresets are source-specific. Fit locally for a new source. Calibration applies only to next-event scores; time and horizon outputs remain raw.')
            run=gr.Button('预测 / Predict',variant='primary')
        with gr.Column():
            table=gr.Dataframe(headers=HEADERS,label='输出 / Top 10 events',interactive=False)
            gr.Markdown('Raw / Calibrated: 下一事件相对概率（总和为1），不是独立并发风险。\nNext-event relative probabilities (sum to 1), not independent concurrent-event risks.')
            with gr.Accordion('完整输出 / Full output',open=False):
                output=gr.JSON(label='JSON')
                download=gr.File(label='下载 / Download JSON')
    example.click(lambda:EXAMPLE,outputs=text)
    run.click(predict,[text,calibration,custom],[table,output,download],api_name='predict')
if __name__=='__main__':demo.queue(max_size=32).launch()
