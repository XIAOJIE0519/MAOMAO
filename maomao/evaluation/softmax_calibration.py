"""Full-calibration-row scalar temperature; sealed test labels are never read.

Prespecified fit objective: softmax cross entropy to the normalized target set.
Prespecified selection: use fitted temperature only if full calibration Brier
improves over T=1. Both decisions are made exclusively on the external 90%.
"""
from pathlib import Path
import json
import numpy as np
import torch

LEGACY_PROTOCOL = "softmax_normalized_target_ce_calibration_brier_guard_v1"
_ROOT = Path(__file__).resolve().parents[2]
_ACTIVE = _ROOT / "outputs/external_validation_final_maomao_uniform/active_calibration_protocol.json"
PROTOCOL = json.loads(_ACTIVE.read_text())["protocol"] if _ACTIVE.exists() else LEGACY_PROTOCOL


def revision_directory():
    """Resolve only a completely promoted calibration revision."""
    return _ROOT / (json.loads(_ACTIVE.read_text())["revision_directory"] if _ACTIVE.exists()
                    else "outputs/calibration_softmax_revision_20260929")


def fit_temperature(logits_path, labels_path, device):
    if PROTOCOL != LEGACY_PROTOCOL:
        raise RuntimeError("Scalar fitter is historical; use run_v5_bias_calibration_revision.py for the active V5 method")
    z = np.load(logits_path, mmap_mode="r")
    y = np.load(labels_path, mmap_mode="r")
    if z.shape != y.shape or len(z) == 0:
        raise ValueError("Invalid calibration arrays")
    def statistics(beta):
        ce = gradient = hessian = brier = 0.
        for start in range(0, len(y), 65536):
            x = torch.as_tensor(np.array(z[start:start+65536], dtype=np.float32), device=device)
            target = torch.as_tensor(np.array(y[start:start+65536], dtype=np.float32), device=device)
            count = target.sum(-1, keepdim=True)
            if bool((count == 0).any()) or not bool(torch.isfinite(x).all()):
                raise ValueError("Nonfinite score or empty target")
            q = target/count
            # Exact row-shift invariance, including derivatives.
            x = x-x.max(-1, keepdim=True).values
            lp = torch.log_softmax(beta*x, -1)
            p = lp.exp()
            mu = (p*x).sum(-1)
            ce += float((-(q*lp).sum(-1)).sum(dtype=torch.float64))
            gradient += float((mu-(q*x).sum(-1)).sum(dtype=torch.float64))
            hessian += float(((p*x.square()).sum(-1)-mu.square()).clamp_min(0).sum(dtype=torch.float64))
            brier += float((p-q).square().sum(dtype=torch.float64))
        return {"cross_entropy":ce/len(y), "gradient":gradient/len(y),
                "hessian":hessian/len(y), "brier":brier/len(y)}
    raw = statistics(1.)
    lower, upper = 1/6., 1/.15
    beta = 1.
    history = []
    for iteration in range(16):
        s = statistics(beta)
        history.append(dict(iteration=iteration+1, inverse_temperature=beta, **s))
        print(f"softmax calibration {iteration+1}: beta={beta:.7g} CE={s['cross_entropy']:.7g} Brier={s['brier']:.7g}", flush=True)
        if abs(s["gradient"]) < 1e-7:
            break
        if s["gradient"] > 0:
            upper = beta
        else:
            lower = beta
        candidate = beta-s["gradient"]/max(s["hessian"],1e-12)
        if candidate <= lower or candidate >= upper:
            candidate = (lower+upper)/2
        if abs(candidate-beta) < 1e-6:
            beta = candidate
            break
        beta = candidate
    fitted = statistics(beta)
    accepted = fitted["brier"] < raw["brier"]-1e-8
    return {"protocol":PROTOCOL, "calibration_rows":int(len(y)),
            "objective":"softmax cross entropy to y/sum(y)",
            "selection":"full 90% calibration Brier must improve by >1e-8; otherwise T=1",
            "temperature_bounds":[.15,6.], "fitted_temperature":1/beta,
            "temperature":1/beta if accepted else 1., "fitted_temperature_accepted":accepted,
            "raw_calibration":raw, "fitted_calibration":fitted, "optimization_history":history,
            "test_labels_used_for_fit_or_selection":False, "row_logit_shift_invariant":True}
