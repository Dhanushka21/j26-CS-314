"""Fit the chosen baselines on all profiling data and persist them for serving."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).parent))
from baseline import fit_all, load_windows  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC, MODELS = ROOT / "data" / "processed", ROOT / "models"
MODELS.mkdir(exist_ok=True)

# Operating point chosen from out/tradeoff_isolation_forest.csv:
# 100% spoofing detection at 11.5% false restriction. See README.
ALGORITHM = "isolation_forest"
THRESHOLD = 0.15

if __name__ == "__main__":
    df = load_windows(PROC / "profiling_train.csv", PROC / "profiling_test.csv")
    baselines = fit_all(df, algorithm=ALGORITHM)
    joblib.dump(baselines, MODELS / "device_baselines.joblib")
    (MODELS / "config.json").write_text(json.dumps({
        "algorithm": ALGORITHM,
        "threshold": THRESHOLD,
        "window_seconds": 30.0,
        "stride_seconds": 15.0,
        "devices": sorted(baselines),
        "score_convention": "1.0 = behaves like claimed device, 0.0 = does not",
    }, indent=2))
    print(f"saved {len(baselines)} baselines -> {MODELS/'device_baselines.joblib'}")
    for d, b in sorted(baselines.items()):
        print(f"  {d:<24} trained on {b.n_train} windows")
