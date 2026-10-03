#!/usr/bin/env python3
"""Actual saved-model inference and disposable optimizer steps for storage QA."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from maomao.data.event_sequence import EventSequenceDataset, collate_event_sequences
from maomao.models.event_maomao import event_time_loss, masked_event_loss, masked_value_loss
from scripts.train import make_masked_batch
from scripts.diagnostics.evaluate_external_validation import build_model, move_batch
from scripts.diagnostics.run_requested_50k_models import SequenceNet
from scripts.diagnostics.uniform_result_scope import current_xgboost_model

OUT = ROOT/'outputs/storage_rebuild_20261001'
BASE = ROOT/'outputs/final_experiment_results_20260923'


def weights_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=('before', 'after'), required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    device = torch.device('cuda')
    torch.manual_seed(42)
    np.random.seed(42)
    rows = BASE/'classical_full_scale'
    x = np.array(np.load(rows/'train_X.npy', mmap_mode='r')[128:192], copy=True)
    y = np.array(np.load(rows/'train_y.npy', mmap_mode='r')[128:192], dtype=np.float32, copy=True)
    values = dict(feature_batch=x, target_batch=y)
    records = {}
    payload = torch.load(BASE/'baseline_metrics/logistic_fullscale.pt', map_location='cpu', weights_only=False)
    linear = torch.nn.Linear(1549, 210).to(device)
    linear.weight.data.copy_(payload['weight'].to(device))
    linear.bias.data.copy_(payload['bias'].to(device))
    scaler = np.load(BASE/'baseline_metrics/logistic_scaler.npz')
    xb = torch.tensor(x, device=device)
    logistic_x = (xb-torch.tensor(scaler['mean'], device=device, dtype=torch.float32))/torch.tensor(scaler['scale'], device=device, dtype=torch.float32).clamp_min(1e-8)
    labels = torch.tensor(y, device=device)
    ann = SequenceNet('ann', 0, 210).to(device)
    ann.load_state_dict(torch.load(BASE/'baseline_metrics/common_full_validation/ann_fullscale_refined.pt', map_location='cpu', weights_only=False)['model'])
    for name, model, batch in (('logistic', linear, logistic_x), ('ann', ann, xb)):
        model.eval()
        with torch.no_grad():
            score = model(batch) if name == 'logistic' else model(batch, None, None)[0]
        values[name+'_inference'] = score.detach().cpu().numpy()
        model.train()
        torch.manual_seed(42)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        score = model(batch) if name == 'logistic' else model(batch, None, None)[0]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(score, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        records[name] = dict(disposable_training_loss=float(loss.detach()), updated_parameter_sha256=weights_digest(model))
    import xgboost as xgb
    booster = xgb.Booster(params={'nthread': 4})
    booster.load_model(str(current_xgboost_model()))
    values['xgboost_inference'] = booster.predict(xgb.DMatrix(x, nthread=4), output_margin=True)
    dataset = EventSequenceDataset(ROOT/'data/perioperative_event_sequences_v5_richctx_static7', 256, 128, dynamic_windows=False)
    windows = np.load(rows/'train_window_indices.npy', mmap_mode='r')
    ids = [int(windows[128]), int(windows[1000])]
    batch = move_batch(collate_event_sequences([dataset[i] for i in ids]), device)
    checkpoint = torch.load(BASE/'full_maomao_reference/best_model.pt', map_location='cpu', weights_only=False)
    cfg = checkpoint['args']
    maomao = build_model(dataset, checkpoint, device)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        prediction = maomao(batch)
    for name in ('logits', 'family_logits', 'fine_hazard_logits', 'long_hazard_logits',
                 'tail_mu', 'tail_log_sigma', 'time_mu', 'time_log_sigma',
                 'family_time_mu', 'family_time_log_sigma', 'trajectory_logits'):
        values['maomao_'+name] = getattr(prediction, name).cpu().numpy()
    maomao.train()
    torch.manual_seed(42)
    optimizer = torch.optim.AdamW(maomao.parameters(), lr=cfg['lr'])
    with torch.autocast('cuda', dtype=torch.bfloat16):
        o = maomao(batch)
        loss = event_time_loss(o.logits, batch['target_set'], batch['target_dt_hours'], batch['loss_mask'], batch['time_mask'], cfg['time_loss_weight'], cfg['max_wait_hours'], -1, o.log_total_rate, o.time_mu, o.time_log_sigma, o.trajectory_logits, batch['trajectory_target'], batch['trajectory_mask'], cfg['trajectory_loss_weight'], None, o.family_logits, maomao.outcome_family_ids, cfg['family_loss_weight'], o.family_time_mu, o.family_time_log_sigma, o.fine_hazard_logits, o.long_hazard_logits, o.tail_mu, o.tail_log_sigma)['loss']
    assert torch.isfinite(loss)
    optimizer.zero_grad()
    loss.backward()
    masked_batch, masked_labels, value_labels, value_mask = make_masked_batch(
        batch, dataset.meta['token_vocabulary']['<MASK>'], dataset.num_tokens,
        cfg['masked_event_probability'])
    with torch.autocast('cuda', dtype=torch.bfloat16):
        auxiliary = maomao(masked_batch, causal=False, return_token_logits=True, masked_only=True)
        masked_loss = masked_event_loss(auxiliary.token_logits, masked_labels)
        value_loss = masked_value_loss(auxiliary.value_prediction, value_labels, value_mask)
        auxiliary_loss = cfg['masked_event_loss_weight']*masked_loss+cfg['masked_value_loss_weight']*value_loss
    assert torch.isfinite(auxiliary_loss)
    auxiliary_loss.backward()
    torch.nn.utils.clip_grad_norm_(maomao.parameters(), cfg['grad_clip'])
    optimizer.step()
    records['maomao'] = dict(disposable_training_loss=float(loss.detach()),
                          masked_event_loss=float(masked_loss.detach()),
                          masked_value_loss=float(value_loss.detach()),
                          updated_parameter_sha256=weights_digest(maomao), actual_train_windows=ids)
    # These are QA-only copies. No saved model, metrics, or article source is updated.
    np.savez(OUT/f'operational_{args.phase}.npz', **values)
    (OUT/f'operational_{args.phase}.json').write_text(json.dumps(records, indent=2)+'\n')
    if args.phase == 'after':
        with np.load(OUT/'operational_before.npz') as before:
            assert set(before.files) == set(values)
            comparisons = {k:bool(before[k].shape == value.shape and before[k].dtype == value.dtype
                                   and np.array_equal(np.ascontiguousarray(before[k]).view(np.uint8),
                                                      np.ascontiguousarray(value).view(np.uint8)))
                           for k, value in values.items()}
            assert all(comparisons.values()), comparisons
        prior = json.loads((OUT/'operational_before.json').read_text())
        assert prior == records, 'Disposable optimizer step changed'
        (OUT/'operational_verification.json').write_text(json.dumps(dict(complete=True, inference_and_input_arrays_bitwise_equal=comparisons, disposable_optimizer_steps_equal=True, scope='Saved logistic,ANN,XGBoost,MAOMAO inference; disposable logistic/ANN/MAOMAO optimizer steps on fixed real training inputs; not a full retraining'), indent=2)+'\n')
    print(f'{args.phase}: saved-model inference and disposable optimizer steps completed', flush=True)


if __name__ == '__main__':
    main()
