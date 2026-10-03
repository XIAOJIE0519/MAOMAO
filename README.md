# MAOMAO

**Multi-horizon Anticipatory Outcome Model for Anesthesia and Operations**

[English](https://github.com/XIAOJIE0519/MAOMAO/blob/main/README.md) · [简体中文](https://github.com/XIAOJIE0519/MAOMAO/blob/main/README_CN.md)

[Demo](https://huggingface.co/spaces/luan0519/MAOMAO) · [GitHub](https://github.com/XIAOJIE0519/MAOMAO) · [Model weights](https://huggingface.co/luan0519/MAOMAO)

![MAOMAO overview](overview.jpg)

MAOMAO uses irregular perioperative event histories to predict the next clinical event, its expected waiting time, and event occurrence across future horizons. This repository provides the released model architecture, pretrained weights access, calibration utilities, and an interactive inference demo.

## Input → Output

| Input | Format |
|---|---|
| Patient features | `static`: `age_at_operation` (years), `male` (0/1), `asa` (0–6), `emergency` (0/1), `weight_kg`, `height_cm`; missing values default to 0. BMI is derived from height and weight when available, or supplied as `bmi`. |
| Observed events | `events`: `time_min` (minutes since record start), `token` ([vocabulary](vocabulary.json)), optional `value` in original measurement units. |

See [example.json](example.json) for a synthetic example. Submit only history observed by the query time, without patient identifiers. Raw waveforms must first be converted into supported event tokens.

| Output | Meaning |
|---|---|
| `raw` / `calibrated` | Relative next-event probabilities, normalized across 210 events. |
| `wait_hours` | Event-specific expected waiting time in hours. |
| `risk_1h_raw` / `risk_6h_raw` / `risk_24h_raw` | Uncalibrated sigmoid scores from the 1/6/24-hour event-occurrence heads. |
| `event_logits` | 210 raw event scores in vocabulary order. |

The demo displays probabilities as percentages with three significant digits and waiting times in both hours and minutes. Downloaded JSON preserves numeric probabilities and `wait_hours` for analysis.

## Run

### Hardware and GPU memory

**We trained MAOMAO on an NVIDIA GeForce RTX 4090 with 24 GB of GPU memory, using BF16 mixed precision.** For training or larger GPU workloads, we recommend **more than 12 GB of GPU memory**; 24 GB provides more headroom for the reference configuration.

This is a recommendation, not a fixed minimum or a guarantee that every configuration fits. Peak memory depends on micro-batch size, sequence length, precision, and gradient/optimizer states. If training runs out of memory, reduce the micro-batch size and use gradient accumulation; use BF16 where supported. Longer attention sequences can substantially increase memory use.

Single-record inference needs less memory than training. The command-line script defaults to **CPU**, so the example does not require a GPU. The local web app uses CUDA when available; the hosted demo uses Hugging Face ZeroGPU and is subject to its compute quota.

### Installation and inference

```bash
pip install -r requirements.txt
python inference.py example.json --calibration None
python app.py
```

Assets download automatically from [the model repository](https://huggingface.co/luan0519/MAOMAO). A local asset directory must contain `model.safetensors`, `config.json`, `vocabulary.json`, and `calibrations.json`:

```bash
python inference.py example.json --model-dir /path/to/model-assets
MAOMAO_MODEL_DIR=/path/to/model-assets python app.py
```

For GPU inference through Python, explicitly select CUDA:

```python
from pathlib import Path
from inference import Predictor

predictor = Predictor(device="cuda")
result = predictor.predict(Path("example.json").read_text(), calibration="None")
```

## Calibration

Only the next-event distribution is calibrated:

$$
p^{\mathrm{cal}}_e=\mathrm{softmax}\left(\frac{\mathbf z+\mathbf b}{T}\right)_e,\qquad T>0.
$$

Select a dataset preset, or select **Custom** and supply `temperature` and an optional 210-element `bias`. Presets apply to their source distributions and do not establish validity at a new hospital. `None` retains raw probabilities.

For a new source, split patients into 90% calibration and 10% held-out test. Within calibration, use 80% for fitting and 20% for selection, then refit on all calibration patients. Freeze model weights and do not use test labels.

```bash
python calibrate.py --logits logits.npy --labels labels.npy --groups patient_groups.npy --output custom_calibration.json
```

`logits.npy` and `labels.npy` have shape `[N,210]`; labels are binary multi-hot. `patient_groups.npy` contains `[N]` integer patient-group IDs. Supply calibration rows only. Event calibration does not change waiting times or horizon outputs.

## Model

**384 hidden dimensions · 10 Transformer layers · 12 attention heads · 210 event classes · 63 event families · 1/6/24-hour heads.** Development used INSPIRE: 99,886 patients and 130,960 surgical records. Internal micro-AUROC was 0.939; Recall@10 was 0.695. External evaluation uses frozen weights, with raw and source-calibrated results reported separately.

Inference encodes the most recent 256 tokens together with earlier event-family counts, perioperative phase, and monitoring features. Concurrent-event masking follows the training model.

## Technical and mathematical principles

MAOMAO combines event tokens, continuous time, measurement values, patient features, and compact history memory. Relative-time attention handles irregular intervals, while a causal mask blocks future positions and distinct events sharing a timestamp:

$$
A_{ij}=\frac{\mathbf q_i^\top\mathbf k_j}{\sqrt{d_h}}+b(t_i-t_j)+M_{ij}.
$$

Here $b$ is a learned time bias and $M$ is the attention mask. Event and clinical-family scores are combined for next-event prediction. When several events occur at the next timestamp, the model learns from their set $Y_i$ rather than selecting one arbitrary target:

$$
\mathcal L_{\mathrm{event},i}=-\log\sum_{e\in Y_i}p_{i,e}.
$$

The design jointly predicts **what happens next, when it may happen, and what may occur within 1/6/24 hours**. Event-specific waiting times use fine five-minute bins over 0–2 hours, coarser thirty-minute bins over 2–24 hours, and a log-normal tail. Multi-horizon and masked-reconstruction objectives support shared learning. See [model.py](model.py) for implementation.

## Research use and release contents

Recorded events include care decisions; outputs do not establish treatment effects. Waiting-time error and external domain shift remain limitations. The overview summarizes the study; retrospective endpoint results do not constitute prospective clinical validation.

Weights are distributed as SafeTensors, without patient records, optimizer states, or identifiers. See [release_manifest.json](release_manifest.json).

## Contributors

Shanjie Luan and Yunkun Shi
