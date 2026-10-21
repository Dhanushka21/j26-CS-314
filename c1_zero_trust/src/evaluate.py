"""
C1 Function 1 - identity verification experiment.

For each device D with a learned baseline:
    positives = D's own held-out windows        (should score high)
    negatives = every other device's windows    (should score low)

That is exactly the cloned-identity case: a session claims to be D while the
traffic was produced by something else. AUC over this separation is the
identity-verification performance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from baseline import fit_all, load_windows, temporal_split  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)


def run(algorithm: str) -> pd.DataFrame:
    df = load_windows(PROC / "profiling_train.csv", PROC / "profiling_test.csv")
    train_df, test_df = temporal_split(df, frac=0.7)

    print(f"\n=== {algorithm} ===")
    print(f"train windows {len(train_df)}  test windows {len(test_df)}  "
          f"devices {df['device'].nunique()}")
    baselines = fit_all(train_df, algorithm=algorithm)

    rows = []
    for device, bl in sorted(baselines.items()):
        own = test_df[test_df["device"] == device]
        other = test_df[test_df["device"] != device]
        if len(own) < 3 or len(other) < 3:
            continue
        s_own = bl.score(own)
        s_other = bl.score(other)
        y = np.r_[np.ones(len(s_own)), np.zeros(len(s_other))]
        s = np.r_[s_own, s_other]
        rows.append({
            "device": device,
            "n_train": bl.n_train,
            "n_own_test": len(own),
            "n_impostor": len(other),
            "mean_own": float(s_own.mean()),
            "mean_impostor": float(s_other.mean()),
            "auc": float(roc_auc_score(y, s)),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    results = {}
    for algo in ("isolation_forest", "one_class_svm"):
        res = run(algo)
        results[algo] = res
        print()
        print(res.to_string(index=False,
                            formatters={"mean_own": "{:.3f}".format,
                                        "mean_impostor": "{:.3f}".format,
                                        "auc": "{:.3f}".format}))
        print(f"  mean AUC = {res['auc'].mean():.3f}")
        res.to_csv(OUT / f"identity_auc_{algo}.csv", index=False)

    print("\n=== comparison ===")
    merged = results["isolation_forest"][["device", "auc"]].merge(
        results["one_class_svm"][["device", "auc"]],
        on="device", suffixes=("_iforest", "_ocsvm"))
    print(merged.to_string(index=False,
                           formatters={"auc_iforest": "{:.3f}".format,
                                       "auc_ocsvm": "{:.3f}".format}))
