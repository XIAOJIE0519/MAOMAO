"""Compute INSPIRE 63-family Hit@1/5/10 from frozen full-cohort predictions."""
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

OUT = ROOT / 'outputs/figure_revision_20261002/internal_family_retrieval'
BASE = scoring.INTERNAL
ROWS = BASE / 'classical_full_scale'

FAMILY_IDS = torch.tensor(json.loads((scoring.TRAINING/'event_sequence_meta.json').read_text())['outcome_to_family'],dtype=torch.long)
MEMBERS = torch.nn.functional.one_hot(FAMILY_IDS,63).float()

def count_batch(logits, target):
    z=torch.from_numpy(np.asarray(logits,dtype=np.float32))
    y=torch.from_numpy(np.asarray(target,dtype=np.uint8))
    top=torch.argsort(z,descending=True,dim=1)[:,:10]
    hits=torch.gather(y,1,top).sum(1);cardinality=y.sum(1)
    assert torch.all(cardinality>0)
    true_f=(y.float()@MEMBERS).bool()
    family=[]
    for k in (1,5,10):
        pred=torch.zeros_like(true_f).scatter_(1,FAMILY_IDS[top[:,:k]],True)
        family.append((pred&true_f).any(1).sum().item())
    return np.array([(hits==cardinality).sum().item(),(hits/cardinality).double().sum().item(),len(y),*family,torch.gather(y,1,top[:,:1]).any(1).sum().item()],dtype=np.float64)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    result_path = OUT / 'internal_family_hits.json'
    results = json.loads(result_path.read_text()) if result_path.exists() else {}
    y = np.load(ROWS / 'validation_y.npy', mmap_mode='r')
    assert len(y) == 1563972

    def save(name, totals):
        assert int(totals[2]) == len(y)
        original=json.loads((ROOT/f'outputs/maomao_v5_final_results/internal/{name}/metrics.json').read_text())
        audit=dict(model=name,hit_at_1=float(totals[6]/totals[2]),original_hit_at_1=original['hit_at_1'],difference=float(totals[6]/totals[2])-original['hit_at_1'],recall10=float(totals[1]/totals[2]),original_recall10=original['recall_at_10'])
        print('Alignment audit',audit,flush=True)
        (OUT/f'{name}_reproduction_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        assert abs(audit['difference'])<=3/len(y) and abs(audit['recall10']-audit['original_recall10'])<2e-6, 'Frozen full-cohort reproduction exceeds numerical tolerance'
        results[name] = dict(all_true_events_hit_at_10=float(totals[0]/totals[2]),
                            recall_at_10_recomputed=float(totals[1]/totals[2]),
                            same_family_hit_at_1=float(totals[3]/totals[2]),
                            same_family_hit_at_5=float(totals[4]/totals[2]),
                            same_family_hit_at_10=float(totals[5]/totals[2]),
                            event_hit_at_1_recomputed=float(totals[6]/totals[2]),
                            rows=int(totals[2]),full_cohort=True,
                            family_definition='Map the top-k event predictions and all true events through the original 63 semantic families; any family match.',
                            tie_protocol='torch_cpu_argsort_on_logits_v1',
                            event_hit_at_1_verified_within_three_row_numerical_tolerance=True,
                            event_hit_at_1_difference=audit['difference'],training_changed=False)
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
        totals = np.zeros(7, dtype=np.float64)
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
                    event_logits = model(batch).logits
                logits = event_logits[local, pos].cpu().numpy()
                totals += count_batch(logits, target)
                if (batch_no+1) % 25 == 0 or batch_no+1 == len(loader):
                    print(f'MAOMAO {batch_no+1}/{len(loader)} batches, {int(totals[2])} rows', flush=True)
        save('maomao', totals)
        del model, ds, loader, batch, event_logits
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
        totals = np.zeros(7, dtype=np.float64)
        for start in range(0, len(y), 8192):
            totals += count_batch(np.array(z[start:start+8192]), np.array(y[start:start+8192]))
        save(name, totals)
        del z; gc.collect()
        if temp is not None: temp.unlink()

if __name__ == '__main__': main()
