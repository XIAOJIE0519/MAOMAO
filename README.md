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

### Continuous-time representation and history memory

For token $x_i$ at time $t_i$, the representation combines token kind $k_i$, measurement $v_i$, observation indicator $m_i$, static features $s$, phase $\phi_i$, monitoring features $o_i$, and earlier event-family counts $c$:

$$
\begin{aligned}
\mathbf u_i={}&E_x(x_i)+E_k(k_i)+P_v[\psi(v_i),m_i]\\
&+\mathrm{CTE}(t_i,\Delta t_i)+P_s(s)+E_\phi(\phi_i)\\
&+P_o(o_i)+P_c(\log(1+c)).
\end{aligned}
$$

Here $\psi(v)=\mathrm{clip}(\mathrm{sign}(v)\log(1+|v|),-12,12)$, $\Delta t_i=t_i-t_{i-1}$, and $P$ denotes learned projections. Continuous-time encoding projects sine/cosine features of absolute time and time gaps. This preserves irregular timestamps without a dense five-minute input grid. Family-count memory summarizes history before the current window; monitoring features include recent observation density and time since a family's last observation.

### Relative-time attention and concurrent-event masking

Each attention head uses:

$$
A_{ij}=\frac{\mathbf q_i^\top\mathbf k_j}{\sqrt{d_h}}+b(t_i-t_j)+M_{ij}.
$$

The learned bias $b$ uses signed logarithmic time differences and a same-time indicator. $M_{ij}=-\infty$ for future positions and distinct tokens sharing a timestamp, and zero otherwise; self-attention remains allowed. Concurrent events cannot reveal one another through arbitrary within-timestamp ordering, while earlier observations remain available.

### Hierarchical next-event prediction and set supervision

The clinical-family head contributes to each exact event/severity score:

$$
z_{i,e}=(W_e\mathbf h_i+b_e)_e+(W_f\mathbf h_i+b_f)_{f(e)},\qquad
p_{i,e}=\frac{e^{z_{i,e}}}{\sum_{r=1}^{210}e^{z_{i,r}}}.
$$

With $f(e)$ mapping events to families and $Y_i$ the observed event set at the next timestamp, training uses:

$$
\mathcal L_{\mathrm{event},i}=-\log\sum_{e\in Y_i}p_{i,e}.
$$

This connects clinical categories to specific events without assigning simultaneous outcomes one arbitrary target. Softmax outputs describe relative next-event probabilities, not independent concurrent-event risks.

### Dual-timescale event-specific waiting time

The released time head uses **24 five-minute bins over 0–2 hours**, **44 thirty-minute bins over 2–24 hours**, and a log-normal residual tail beyond 24 hours. For event $e$ and bin $k$:

$$
q_{e,k}=\sigma(a_{e,k}),\qquad S_{e,k}=\prod_{j=1}^{k}(1-q_{e,j}),\qquad P_{e,k}=S_{e,k-1}q_{e,k}.
$$

With $S_{e,0}=1$, bin midpoint $m_k$, final bin edge $\tau_K=24$ hours, and residual $R_e\sim\mathrm{LogNormal}(\mu_e,\sigma_e^2)$, decoding uses:

$$
\widehat{\mathbb E}[\Delta t_e]=\sum_{k=1}^{K}P_{e,k}m_k
+S_{e,K}\left(\tau_K+e^{\mu_e+\sigma_e^2/2}\right).
$$

Tail parameters and decoded waits are bounded for numerical stability. Fine bins capture near-term changes; coarser bins and the tail cover longer intervals. Right-censored episodes contribute survival terms to the training likelihood instead of fabricated event times.

### Multi-horizon and auxiliary learning

Separate sigmoid heads predict event occurrence over $H\in\lbrace 1,6,24\rbrace$ hours:

$$
r_{e,H}=\sigma\left((W_H\mathbf h_i+b_H)_e\right).
$$

The reference training objective combines event identity, family, waiting time, future trajectories, masked-token reconstruction, and masked-value reconstruction:

$$
\begin{aligned}
\mathcal L={}&\mathcal L_{\mathrm{event}}+0.25\mathcal L_{\mathrm{family}}+\mathcal L_{\mathrm{time}}\\
&+0.5\mathcal L_{\mathrm{trajectory}}+0.2\mathcal L_{\mathrm{masked\text{-}token}}\\
&+0.1\mathcal L_{\mathrm{masked\text{-}value}}.
\end{aligned}
$$

Auxiliary targets are used only where available; missing measurements are not treated as clinical zeros. Horizon scores are separate outputs, not sums of next-event softmax probabilities.

**Design contributions.** MAOMAO jointly integrates irregular continuous-time inputs, leakage-aware concurrent-event masking, hierarchical event/family prediction, compact history memory, and event-specific timing at multiple resolutions. These designs address **what may happen next, when it may happen, and which events may occur within 1/6/24 hours**. See [model.py](model.py), [inference.py](inference.py), and [config.json](config.json).

## Research use and release contents

Recorded events include care decisions; outputs do not establish treatment effects. Waiting-time error and external domain shift remain limitations. The overview summarizes the study; retrospective endpoint results do not constitute prospective clinical validation.

Weights are distributed as SafeTensors, without patient records, optimizer states, or identifiers. See [release_manifest.json](release_manifest.json).
