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

### 连续时间表示与历史记忆

对于时间 $t_i$ 的 token $x_i$，表示由 token 类型 $k_i$、测量值 $v_i$、是否观测到数值的指示量 $m_i$、静态特征 $s$、阶段 $\phi_i$、监测特征 $o_i$ 和更早的事件家族计数 $c$ 共同构成：

$$
\begin{aligned}
\mathbf u_i={}&E_x(x_i)+E_k(k_i)+P_v[\psi(v_i),m_i]\\
&+\mathrm{CTE}(t_i,\Delta t_i)+P_s(s)+E_\phi(\phi_i)\\
&+P_o(o_i)+P_c(\log(1+c)).
\end{aligned}
$$

其中 $\psi(v)=\mathrm{clip}(\mathrm{sign}(v)\log(1+|v|),-12,12)$，$\Delta t_i=t_i-t_{i-1}$，$P$ 表示可学习投影。连续时间编码对绝对时间与相邻时间间隔的正弦/余弦特征进行投影，保留不规则时间戳，无需构造密集的五分钟输入网格。家族计数概括当前窗口之前的历史；监测特征包括近期观测密度，以及距某一家族上次观测的时间。

### 相对时间注意力与并发事件屏蔽

每个注意力头使用：

$$
A_{ij}=\frac{\mathbf q_i^\top\mathbf k_j}{\sqrt{d_h}}+b(t_i-t_j)+M_{ij}.
$$

可学习偏置 $b$ 使用带符号的对数时间差和同时间指示量。对于未来位置，以及时间戳相同但彼此不同的 token，$M_{ij}=-\infty$；其余位置为零，token 仍可关注自身。因此，同时间事件不会因任意排列顺序而相互泄露信息，同时保留对更早观测的访问。

### 下一事件层级预测与集合监督

临床家族预测头参与每个具体事件或严重程度类别的评分：

$$
z_{i,e}=(W_e\mathbf h_i+b_e)_e+(W_f\mathbf h_i+b_f)_{f(e)},\qquad
p_{i,e}=\frac{e^{z_{i,e}}}{\sum_{r=1}^{210}e^{z_{i,r}}}.
$$

$f(e)$ 将事件映射到家族；$Y_i$ 是下一个时间戳的已观察事件集合，训练使用：

$$
\mathcal L_{\mathrm{event},i}=-\log\sum_{e\in Y_i}p_{i,e}.
$$

这样既连接临床大类与具体事件，也避免将同时出现的多个结局强行设为一个任意目标。softmax 表示下一事件的相对概率，并非多个并发事件独立发生的风险。

### 双时间尺度的事件特异性等待时间

公开时间预测头采用 **0–2 小时的 24 个五分钟区间**、**2–24 小时的 44 个三十分钟区间**，并在 24 小时后接入对数正态残余时间尾部。对于事件 $e$ 和区间 $k$：

$$
q_{e,k}=\sigma(a_{e,k}),\qquad S_{e,k}=\prod_{j=1}^{k}(1-q_{e,j}),\qquad P_{e,k}=S_{e,k-1}q_{e,k}.
$$

设 $S_{e,0}=1$，$m_k$ 为区间中点，最后一个区间边界 $\tau_K=24$ 小时，残余等待时间 $R_e\sim\mathrm{LogNormal}(\mu_e,\sigma_e^2)$，解码使用：

$$
\widehat{\mathbb E}[\Delta t_e]=\sum_{k=1}^{K}P_{e,k}m_k
+S_{e,K}\left(\tau_K+e^{\mu_e+\sigma_e^2/2}\right).
$$

实现中对尾部参数与解码等待时间设定数值边界，以保持稳定性。细时间区间刻画近期变化，较粗区间和尾部覆盖更长的等待时间。右删失 episode 使用生存项参与训练似然，不人为补造事件发生时间。

### 多时域预测与辅助学习

独立的 sigmoid 预测头估计 $H\in\lbrace 1,6,24\rbrace$ 小时内的事件发生情况：

$$
r_{e,H}=\sigma\left((W_H\mathbf h_i+b_H)_e\right).
$$

参考训练目标联合事件类别、事件家族、等待时间、未来轨迹、掩码 token 重建和掩码数值重建：

$$
\begin{aligned}
\mathcal L={}&\mathcal L_{\mathrm{event}}+0.25\mathcal L_{\mathrm{family}}+\mathcal L_{\mathrm{time}}\\
&+0.5\mathcal L_{\mathrm{trajectory}}+0.2\mathcal L_{\mathrm{masked\text{-}token}}\\
&+0.1\mathcal L_{\mathrm{masked\text{-}value}}.
\end{aligned}
$$

仅在相应目标可用时使用辅助监督，不把缺失测量当作临床零值。时域分数是独立输出，不由下一事件 softmax 概率累加得到。

**创新设计要点。** MAOMAO 联合整合不规则连续时间输入、防止并发事件泄露的注意力屏蔽、事件与家族的层级预测、紧凑历史记忆，以及多分辨率事件时间预测。设计回答三个关联问题：**接下来可能发生什么、可能多久后发生，以及未来 1/6/24 小时内可能出现哪些事件**。实现与配置见 [model.py](model.py)、[inference.py](inference.py) 和 [config.json](config.json)。

## 研究用途与发布内容

记录的事件包含临床决策，输出不代表治疗因果效应。等待时间误差和外部数据分布差异仍是局限。总览图汇总研究结果；回顾性终点结果不等同于前瞻性临床验证。

权重以 SafeTensors 格式发布，不含患者记录、优化器状态或身份信息。见 [release_manifest.json](release_manifest.json)。
