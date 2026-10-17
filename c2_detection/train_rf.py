"""STEP 7: Random Forest baseline (single-flow model).

Run from repo root:   python c2_detection/train_rf.py
Input : data/processed/dataset.npz  (from Step 6)
Output: models/rf.joblib, results/rf_metrics.json, rf_per_class.json, rf_confusion.csv
"""
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

import config as C
from src.metrics import evaluate, single_event_latency_ms


def last_flow(x: np.ndarray) -> np.ndarray:
    """(n, 10, 45) -> (n, 45): RF / MLP only see the newest flow of each window."""
    return np.ascontiguousarray(x[:, -1, :])


def main():
    d = np.load(C.PROCESSED_DIR / "dataset.npz")
    x_tr, y_tr = last_flow(d["X_train"]), d["y_train"]
    x_val, y_val = last_flow(d["X_val"]), d["y_val"]
    x_te, y_te = last_flow(d["X_test"]), d["y_test"]
    print(f"train={x_tr.shape}  val={x_val.shape}  test={x_te.shape}")

    model = RandomForestClassifier(
        n_estimators=100,        # number of trees
        min_samples_leaf=2,      # a leaf needs >= 2 samples (less overfitting, smaller file)
        n_jobs=-1,               # use all CPU cores for training
        random_state=C.SEED,
    )
    t0 = time.time()
    model.fit(x_tr, y_tr)
    print(f"trained in {time.time() - t0:.0f}s")

    val_f1 = f1_score(y_val, model.predict(x_val), average="weighted")
    print(f"validation weighted-F1 = {val_f1:.4f}")

    proba = model.predict_proba(x_te)
    model.n_jobs = 1                                     # the API classifies one flow at a time
    latency = single_event_latency_ms(lambda s: model.predict_proba(s), x_te[:1])
    evaluate("rf", y_te, proba, latency)

    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, C.MODELS_DIR / "rf.joblib", compress=3)
    print(f"\nsaved models/rf.joblib")


if __name__ == "__main__":
    main()