# MAOMAO

**Multi-horizon Anticipatory Outcome Model for Anesthesia and Operations：麻醉与手术多时域前瞻性结局预测模型**

[English](https://github.com/XIAOJIE0519/MAOMAO/blob/main/README.md) · [简体中文](https://github.com/XIAOJIE0519/MAOMAO/blob/main/README_CN.md)

[在线演示](https://huggingface.co/spaces/luan0519/MAOMAO) · [GitHub 代码](https://github.com/XIAOJIE0519/MAOMAO) · [模型权重](https://huggingface.co/luan0519/MAOMAO)

![MAOMAO 模型总览](overview.jpg)

MAOMAO 根据不规则采样的围手术期事件历史，预测下一临床事件、其预期等待时间，以及多个未来时域内的事件发生情况。本仓库提供公开模型结构、预训练权重访问、校准工具和交互式推理演示。

## 输入 → 输出

| 输入 | 格式 |
|---|---|
| 患者基础信息 | `static`：`age_at_operation`（岁）、`male`（0/1）、`asa`（0–6）、`emergency`（0/1）、`weight_kg`、`height_cm`；缺失值默认填 0。身高和体重可用时自动计算 BMI，也可通过 `bmi` 提供。 |
| 已观察事件 | `events`：`time_min`（记录开始后的分钟数）、`token`（见[词表](vocabulary.json)），以及可选的 `value`（使用原始测量单位）。 |

[example.json](example.json) 提供合成示例。仅输入查询时刻之前已观察到的历史，不含患者身份信息。原始波形需要先转换为模型支持的事件 token。

| 输出 | 含义 |
|---|---|
| `raw` / `calibrated` | 在 210 类事件之间归一化的下一事件相对概率。 |
| `wait_hours` | 特定事件的预期等待时间，单位为小时。 |
| `risk_1h_raw` / `risk_6h_raw` / `risk_24h_raw` | 1/6/24 小时事件发生预测头输出的未校准 sigmoid 分数。 |
| `event_logits` | 按词表顺序排列的 210 项原始事件分数。 |

演示界面将概率显示为三位有效数字的百分比，等待时间同时显示小时和分钟。下载的 JSON 保留数值概率和 `wait_hours` 字段，便于后续分析。

## 运行

### 硬件与显存

**我们使用 NVIDIA GeForce RTX 4090（24GB 显存）训练 MAOMAO，采用 BF16 混合精度。** 对于训练或较大规模 GPU 任务，建议使用**显存大于 12GB** 的 GPU；24GB 显存能够为参考配置提供更充足的运行余量。

这是推荐参考值，并非固定最低门槛，也不保证任何配置都能运行。峰值显存取决于单次前向/反向传播的批量大小、序列长度、数值精度，以及梯度和优化器状态。训练遇到显存不足时，可减小单次批量并使用梯度累积；硬件支持时使用 BF16。增加注意力序列长度可能显著提高显存消耗。

单条记录推理的显存需求低于训练。命令行脚本默认使用 **CPU**，因此下方示例不要求 GPU。本地网页应用在 CUDA 可用时使用 GPU；在线演示使用 Hugging Face ZeroGPU，受平台计算额度限制。

### 安装与推理

```bash
pip install -r requirements.txt
python inference.py example.json --calibration None
python app.py
```

模型资源自动从[模型仓库](https://huggingface.co/luan0519/MAOMAO)下载。本地资源目录需包含 `model.safetensors`、`config.json`、`vocabulary.json` 和 `calibrations.json`：

```bash
python inference.py example.json --model-dir /path/to/model-assets
MAOMAO_MODEL_DIR=/path/to/model-assets python app.py
```

通过 Python 使用 GPU 推理时，可显式选择 CUDA：

```python
from pathlib import Path
from inference import Predictor

predictor = Predictor(device="cuda")
result = predictor.predict(Path("example.json").read_text(), calibration="None")
```

## 校准

校准仅调整下一事件的概率分布：

$$
p^{\mathrm{cal}}_e=\mathrm{softmax}\left(\frac{\mathbf z+\mathbf b}{T}\right)_e,\qquad T>0.
$$

可选择数据来源预设，或选择 **Custom**，提供 `temperature` 和可选的 210 项 `bias`。预设仅适用于对应来源分布，不代表模型已在新医院得到验证。`None` 保留原始概率。

对于新来源，按患者划分 90% 校准集和 10% 独立测试集；校准部分内部使用 80% 拟合、20% 选择，再用全部校准患者重新拟合。保持模型权重冻结，不使用测试集标签。

```bash
python calibrate.py --logits logits.npy --labels labels.npy --groups patient_groups.npy --output custom_calibration.json
```

`logits.npy` 和 `labels.npy` 的形状为 `[N,210]`，标签为二进制 multi-hot；`patient_groups.npy` 为 `[N]` 个整数患者分组编号。仅提供校准集数据。事件概率校准不改变等待时间和各时域输出。

## 模型

**384 维隐藏表示 · 10 层 Transformer · 12 个注意力头 · 210 类事件 · 63 个事件家族 · 1/6/24 小时预测头。** 开发队列为 INSPIRE，包含 99,886 名患者和 130,960 条手术记录。内部 micro-AUROC 为 0.939，Recall@10 为 0.695。外部评估保持模型权重冻结，分别报告原始结果与来源校准后的结果。

推理编码最近 256 个 token，并结合更早的事件家族计数、围手术期阶段和监测特征。推理沿用训练时的并发事件屏蔽规则。

## 技术与数学原理

MAOMAO 融合事件 token、连续时间、测量值、患者特征与紧凑历史记忆。相对时间注意力处理不规则时间间隔，因果屏蔽阻断未来位置，以及时间戳相同但彼此不同的事件：

$$
A_{ij}=\frac{\mathbf q_i^\top\mathbf k_j}{\sqrt{d_h}}+b(t_i-t_j)+M_{ij}.
$$

其中 $b$ 为可学习时间偏置，$M$ 为注意力屏蔽。下一事件预测融合具体事件与临床家族评分；当下一个时间戳包含多个事件时，模型学习事件集合 $Y_i$，避免任意选择单一目标：

$$
\mathcal L_{\mathrm{event},i}=-\log\sum_{e\in Y_i}p_{i,e}.
$$

设计联合预测**接下来发生什么、可能多久后发生，以及未来 1/6/24 小时内可能出现哪些事件**。事件特异性等待时间采用 0–2 小时的五分钟细区间、2–24 小时的三十分钟粗区间和对数正态尾部；多时域预测与掩码重建目标支持共享学习。实现见 [model.py](model.py)。

## 研究用途与发布内容

记录的事件包含临床决策，输出不代表治疗因果效应。等待时间误差和外部数据分布差异仍是局限。总览图汇总研究结果；回顾性终点结果不等同于前瞻性临床验证。

权重以 SafeTensors 格式发布，不含患者记录、优化器状态或身份信息。见 [release_manifest.json](release_manifest.json)。

## 贡献者

Shanjie Luan and Yunkun Shi
