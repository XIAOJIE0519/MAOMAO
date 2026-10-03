"""Frozen scope for the current size, vocabulary and context experiments."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from maomao.models.event_maomao import EventMAOMAO
from scripts.diagnostics.uniform_result_scope import sha256

DATA = ROOT / "data/perioperative_event_sequences_v5_richctx_static7"
BASE = ROOT / "outputs/final_experiment_results_20260923"
ROWS = BASE / "classical_full_scale"
REFERENCE = BASE / "full_maomao_reference"
OUT = ROOT / "outputs/scale_ablations_richctx_20260928"
NAMES = ("model_small", "model_large", "vocab_50", "vocab_100", "vocab_150", "context_64", "context_128")
CHANGES = {"model_small": {"hidden_dim": 256, "num_layers": 6, "num_heads": 8, "ffn_dim": 1024},
           "model_large": {"hidden_dim": 512, "num_layers": 12, "num_heads": 16, "ffn_dim": 2048},
           **{f"vocab_{n}": {"outcome_projection_file": str(OUT / f"specifications/vocab_{n}.json")} for n in (50, 100, 150)},
           **{f"context_{n}": {"context_segment_length": n} for n in (64, 128)}}
METRICS = ("micro_auprc", "macro_auprc", "micro_auroc", "macro_auroc", "mrr", "brier", "ece", "hit_at_1", "recall_at_5", "recall_at_10")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def configuration(name):
    cfg = read(REFERENCE / "run_config.json")
    cfg.update(outcome_projection_file=None, context_segment_length=0)
    cfg.update(CHANGES[name])
    cfg["output_dir"] = str(OUT / "runs" / name)
    return cfg


def training_command(name, resume=False):
    command = [sys.executable, str(ROOT / "scripts/train.py")]
    for key, value in configuration(name).items():
        if value is None:
            continue
        if isinstance(value, bool):
            if key in {"require_cuda", "compile", "enhanced_time_encoding", "decoupled_time_head", "reset_early_stopping"}:
                if value:
                    command.append("--" + key)
            else:
                command.append(("--" if value else "--no-") + key)
        else:
            command.extend(["--" + key, str(value)])
    if resume:
        command.extend(["--resume", "auto"])
    return command


def model_for(dataset, payload, device):
    cfg = payload["args"]
    model = EventMAOMAO(
        dataset.num_tokens, dataset.num_outcomes, dataset.num_static,
        cfg["hidden_dim"], cfg["num_layers"], cfg["num_heads"], cfg["ffn_dim"],
        cfg["dropout"], cfg["initial_event_interval_hours"], cfg["decoupled_time_head"],
        cfg["enhanced_time_encoding"], len(dataset.trajectory_horizons_hours), cfg["lognormal_time_head"],
        outcome_family_ids=torch.tensor(dataset.meta["outcome_to_family"]),
        use_family_head=cfg["use_family_head"], same_time_block_causal=cfg["same_time_block_causal"],
        relative_time_attention=cfg["relative_time_attention"], event_conditioned_time_head=cfg["event_conditioned_time_head"],
        phase_memory=cfg["phase_memory"], clock_phase_context=cfg["clock_phase_context"],
        observation_intensity=cfg["observation_intensity"], value_reconstruction=cfg["masked_value_loss_weight"] > 0,
        dual_timescale_time_head=cfg["dual_timescale_time_head"], fine_time_bins=cfg["fine_time_bins"], long_time_bins=cfg["long_time_bins"],
        context_segment_length=cfg.get("context_segment_length", 0), model_family_count=dataset.meta.get("model_family_count"),
    ).to(device)
    model.clock_phase_token_ids = torch.tensor([i for token, i in dataset.meta["token_vocabulary"].items()
                                               if token == "<CLOCK>" or token.startswith("phase_summary:")], device=device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model
