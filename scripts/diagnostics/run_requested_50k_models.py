#!/usr/bin/env python3
"""One-epoch 50k-row deep-baseline sweep on the saved shared row sample.

The requested GAN and autoencoder are supervised adaptations (conditional
adversarial regularization and reconstruction auxiliary loss respectively),
not claimed as canonical generative or unsupervised models.
"""
from __future__ import annotations

import json, time, sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from maomao.evaluation.event_metrics import event_metric_report_with_subsample_ci

OUT = ROOT / "outputs/final_experiment_results_20260923/requested_50k_models"
ROWS = ROOT / "outputs/baseline_comparison_richctx_fair/baseline_rows_internal.npz"
DATA = ROOT / "data/perioperative_event_sequences_v5_richctx_static7"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SequenceNet(nn.Module):
    def __init__(self, kind: str, static_n: int, targets: int):
        super().__init__(); self.kind = kind
        if kind in ("lstm", "rnn"):
            cell = nn.LSTM if kind == "lstm" else nn.RNN
            self.encoder = cell(6, 48, batch_first=True)
            width = 48
        elif kind == "cnn":
            self.encoder = nn.Sequential(nn.Conv1d(6, 32, 5, padding=2), nn.GELU(),
                                          nn.Conv1d(32, 48, 5, padding=2), nn.GELU(),
                                          nn.AdaptiveAvgPool1d(1), nn.Flatten())
            width = 48
        else:
            width = 512 if kind == "ann" else 192
            self.encoder = nn.Sequential(nn.Linear(1549, width), nn.GELU(),
                                          nn.Dropout(.15), nn.Linear(width, width // 2), nn.GELU())
            width //= 2
        self.head = nn.Linear(width + static_n if kind not in ("ann", "autoencoder") else width, targets)
        if kind == "autoencoder":
            self.decoder = nn.Linear(width, 1549)
        if kind == "gan":
            self.discriminator = nn.Sequential(nn.Linear(width + static_n + targets, 96), nn.LeakyReLU(.2), nn.Linear(96, 1))

    def encode(self, x, seq, static):
        if self.kind in ("lstm", "rnn"):
            h = self.encoder(seq)[0][:, -1]
        elif self.kind == "cnn":
            h = self.encoder(seq.transpose(1, 2))
        else:
            h = self.encoder(x)
        return h

    def forward(self, x, seq, static):
        h = self.encode(x, seq, static)
        if self.kind not in ("ann", "autoencoder"):
            h = torch.cat((h, static), dim=1)
        return self.head(h), h


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False)); tmp.replace(path)


def prepare(x, y):
    seq = x[:, 1:1537].reshape(-1, 256, 6).copy()
    static = x[:, 1537:].copy()
    return (torch.from_numpy(x.copy()), torch.from_numpy(seq), torch.from_numpy(static),
            torch.from_numpy(y.astype(np.float32)))


def run_model(name, xtr, ytr, xva, yva, outcomes):
    start = time.monotonic()
    xt, st, ct, yt = prepare(xtr, ytr); xv, sv, cv, yv = prepare(xva, yva)
    tr = DataLoader(TensorDataset(xt, st, ct, yt), batch_size=512, shuffle=True,
                    generator=torch.Generator().manual_seed(42), num_workers=0)
    model = SequenceNet(name, ct.shape[1], yt.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    dopt = torch.optim.AdamW(model.discriminator.parameters(), lr=8e-4) if name == "gan" else None
    model.train(); losses=[]
    for x,s,c,y in tr:
        x,s,c,y = (v.to(DEVICE, non_blocking=True) for v in (x,s,c,y))
        logits,h = model(x,s,c)
        supervised = F.binary_cross_entropy_with_logits(logits, y)
        loss = supervised
        if name == "autoencoder":
            loss = loss + .1 * F.mse_loss(model.decoder(h), x)
        elif name == "gan":
            # Conditional discriminator: true targets versus the generator's
            # predicted target vector; generator remains supervised to labels.
            with torch.no_grad(): fake = torch.sigmoid(logits)
            real_score = model.discriminator(torch.cat((h, y), 1))
            fake_score = model.discriminator(torch.cat((h, fake), 1))
            dloss = F.binary_cross_entropy_with_logits(real_score, torch.ones_like(real_score)) + \
                    F.binary_cross_entropy_with_logits(fake_score, torch.zeros_like(fake_score))
            dopt.zero_grad(set_to_none=True); dloss.backward(retain_graph=True); dopt.step()
            loss = loss + .05 * F.binary_cross_entropy_with_logits(
                model.discriminator(torch.cat((h, torch.sigmoid(logits)), 1)),
                torch.ones((len(y),1), device=DEVICE))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        losses.append(float(supervised.detach()))
    model.eval(); logits=[]
    with torch.inference_mode():
        for i in range(0, len(xva), 512):
            x,s,c = (v[i:i+512].to(DEVICE) for v in (xv,sv,cv))
            z,_=model(x,s,c); logits.append(z.float().cpu())
    scores=torch.cat(logits)
    report=event_metric_report_with_subsample_ci(scores, yv.bool(), outcomes,
                 bootstrap_repeats=200, bootstrap_seed=4200, max_ci_rows=20_000)
    report.update({"model": name, "status":"completed", "train_rows":len(xtr),
        "validation_rows":len(xva), "epochs":1, "mean_train_bce":float(np.mean(losses)),
        "training_seconds":round(time.monotonic()-start,2), "device":str(DEVICE),
        "sample_archive":str(ROWS.relative_to(ROOT)), "split_seed":42,
        "patient_disjoint":True,
        "model_note":("conditional GAN with supervised generator/classifier head; one epoch" if name=="gan" else
                      "encoder with next-event head and auxiliary input reconstruction; one epoch" if name=="autoencoder" else
                      "one-epoch supervised next-event baseline")})
    write_json(OUT/f"{name}.json",report)
    torch.save({"model":model.state_dict(),"model_name":name,"epoch":1}, OUT/f"{name}.pt")
    return {k:report.get(k) for k in ("model","status","micro_auprc","micro_auroc","macro_auroc","mrr","brier","ece","hit_at_1","recall_at_5","recall_at_10","training_seconds")}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    z=np.load(ROWS)
    required=("train_window_indices","train_positions","validation_window_indices","validation_positions")
    if not all(k in z for k in required): raise RuntimeError("shared patient-disjoint row coordinates are missing")
    meta=json.loads((DATA/"event_sequence_meta.json").read_text())
    manifest={"status":"running","train_rows":len(z['y_train']),"validation_rows":len(z['y_validation']),
      "patient_split":"90:10, patient-disjoint, seed=42","train_target_rows":50000,
      "validation_target_rows":20000,"epochs":1,"bootstrap_repeats":200,
      "outcome_count":int(len(meta['outcome_vocabulary'])),"models":[]}
    write_json(OUT/"manifest.json",manifest)
    for name in ("lstm","cnn","rnn","gan","autoencoder","ann"):
        if (OUT/f"{name}.json").exists():
            existing=json.loads((OUT/f"{name}.json").read_text())
            manifest["models"].append({k:existing.get(k) for k in ("model","status","micro_auprc","micro_auroc","macro_auroc","training_seconds")}); continue
        print(f"START {name}",flush=True)
        try:
            result=run_model(name,z['x_train'],z['y_train'],z['x_validation'],z['y_validation'],meta['outcome_vocabulary'])
            manifest["models"].append(result); print(json.dumps(result),flush=True)
        except Exception as e:
            failure={"model":name,"status":"failed","error":repr(e)}
            manifest["models"].append(failure); write_json(OUT/f"{name}_failure.json",failure)
            print(json.dumps(failure),flush=True)
        write_json(OUT/"manifest.json",manifest)
    manifest["status"]="completed" if all(m.get("status") == "completed" for m in manifest["models"]) else "partial"
    write_json(OUT/"manifest.json",manifest)
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=="__main__": main()
