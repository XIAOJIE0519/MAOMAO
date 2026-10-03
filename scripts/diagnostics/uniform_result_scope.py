"""Current external/report scope and promoted XGBoost model provenance."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/final_experiment_results_20260923"
XGB_ROOT = BASE / "classical_ml_fullscale/xgboost"
XGB_CANDIDATE = XGB_ROOT / "conservative_three_rounds_label_serial_20260928"


def current_xgboost_dir():
    pointer = XGB_ROOT / "current_model.json"
    if pointer.exists():
        return ROOT / json.loads(pointer.read_text())["directory"]
    return XGB_ROOT / "minimal_one_round_20260927"


def current_xgboost_model():
    directory = current_xgboost_dir()
    status = json.loads((directory / "status.json").read_text())
    return directory / status["model_file"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def xgboost_revision():
    return sha256(current_xgboost_model())
