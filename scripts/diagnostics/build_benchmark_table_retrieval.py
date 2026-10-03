"""Compute missing INSPIRE all-event Hit@10 from frozen full-cohort predictions."""
import sys, json, gc
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.diagnostics import score_uniform_external_five as scoring
from scripts.diagnostics.scale_ablation_scope import model_for
from maomao.data.event_sequence import EventSequenceDataset, collate_event_sequences

OUT = ROOT / 'outputs/maomao_benchmark_table_20261001'
BASE = scoring.INTERNAL
ROWS = BASE / 'classical_full_scale'

def count_batch(logits, target):
    z = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    y = torch.from_numpy(np.asarray(target, dtype=np.uint8))
    top = torch.argsort(z, descending=True, dim=1)[:, :10]
    hits = torch.gather(y, 1, top).sum(1)
    cardinality = y.sum(1)
    assert torch.all(cardinality > 0)
    return np.array([(hits == cardinality).sum().item(),
                     (hits / cardinality).double().sum().item(), len(y)], dtype=np.float64)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    result_path = OUT / 'internal_all_hit_10.json'
    results = json.loads(result_path.read_text()) if result_path.exists() else {}
    y = np.load(ROWS / 'validation_y.npy', mmap_mode='r')
    assert len(y) == 1563972

    def save(name, totals):
        assert int(totals[2]) == len(y)
        results[name] = dict(all_true_events_hit_at_10=float(totals[0]/totals[2]),
                            recall_at_10_recomputed=float(totals[1]/totals[2]),
                            rows=int(totals[2]), full_cohort=True,
                            tie_protocol='torch_cpu_argsort_on_logits_v1',
                            training_changed=False)
        result_path.write_text(json.dumps(results, indent=2)+'\n')
        print(name, results[name], flush=True)

    if 'maomao' not in results:
        ds = EventSequenceDataset(scoring.TRAINING, 256, 128, dynamic_windows=False)
        checkpoint = torch.load(BASE/'full_maomao_reference/best_model.pt', map_location='cpu', weights_only=False)
        model = model_for(ds, checkpoint, scoring.DEVICE)
        del checkpoint
        windows = np.load(ROWS/'validation_window_indices.npy', mmap_mode='r')
        positions = np.load(ROWS/'validation_positions.npy', mmap_mode='r')
        unique = np.load(BASE/'full_maomao_reference/patient_validation_split.npz')['validation_window_indices']
        loader = DataLoader(Subset(ds, unique), batch_size=64, shuffle=False, num_workers=0,
                            collate_fn=collate_event_sequences)
        totals = np.zeros(3, dtype=np.float64)
        with torch.inference_mode():
            for batch_no, batch in enumerate(loader):
                row_parts, local_parts = [], []
                for b, window in enumerate(unique[batch_no*64:(batch_no+1)*64]):
                    lo = np.searchsorted(windows, window, side='left')
                    hi = np.searchsorted(windows, window, side='right')
                    row_parts.append(np.arange(lo, hi))
                    local_parts.append(np.full(hi-lo, b, dtype=np.int64))
                rows = np.concatenate(row_parts); local = np.concatenate(local_parts)
                pos = np.asarray(positions[rows], dtype=np.int64)
                target = batch['target_set'][local, pos].numpy().astype(np.uint8)
                assert np.array_equal(target, y[rows])
                batch = {k:v.to(scoring.DEVICE) if torch.is_tensor(v) else v for k,v in batch.items()}
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    encoded = model(batch, masked_only=True).hidden_state
                    event_logits = model.outcome_head(encoded).float()
                    if model.family_head is not None:
                        event_logits = event_logits + model.family_head(encoded).float()[..., model.outcome_family_ids]
                    if batch_no == 0:
                        full = model(batch).logits
                        assert torch.equal(full, event_logits), 'Event-only forward differs from full forward'
                logits = event_logits[local, pos].cpu().numpy()
                totals += count_batch(logits, target)
                if (batch_no+1) % 25 == 0 or batch_no+1 == len(loader):
                    print(f'MAOMAO {batch_no+1}/{len(loader)} batches, {int(totals[2])} rows', flush=True)
        save('maomao', totals)
        del model, ds, loader, batch, encoded, event_logits
        gc.collect(); torch.cuda.empty_cache()

    for name in ('ann', 'logistic_regression', 'univariate', 'xgboost'):
        if name in results: continue
        saved = {'ann':BASE/'baseline_metrics/common_full_validation/ann_fullscale_refined_validation_logits.npy',
                 'logistic_regression':BASE/'baseline_metrics/logistic_validation_logits.npy'}.get(name)
        temp = None
        if saved is None:
            temp = OUT / f'.temporary_{name}_logits.npy'
            z = np.lib.format.open_memmap(temp, mode='w+', dtype=np.float32, shape=y.shape)
            scoring.predict_flat(name, ROWS, 'validation', z)
            z.flush()
        else:
            z = np.load(saved, mmap_mode='r')
        assert z.shape == y.shape
        totals = np.zeros(3, dtype=np.float64)
        for start in range(0, len(y), 8192):
            totals += count_batch(np.array(z[start:start+8192]), np.array(y[start:start+8192]))
        save(name, totals)
        del z; gc.collect()
        if temp is not None: temp.unlink()

if __name__ == '__main__': main()
