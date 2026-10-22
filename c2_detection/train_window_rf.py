"""STEP 12: Random Forest on window statistics for ONE window length.

  python c2_detection/train_window_rf.py --window 100

Run it once per window length (10, 30, 50, 100, 150) - one process per run keeps RAM low.
Then  python c2_detection/window_curve.py  draws the curve and picks the best window.

For the given window length w:
  - TRAIN / VAL come from the Train/ files with a TEMPORAL split per file:
      windows ending in the first 90% of the file -> train
      windows ending in the last 10%              -> validation   (gap of w rows, so no overlap)
  - TEST  = the official Test/ files (never seen)
  - SMOTE on train only, RF with 150 trees
Saves models/rf_win<w>.joblib, results/rf_win<w>_metrics.json / _per_class.json / _confusion.csv
      results/rf_win<w>_window.json  (validation + test scores, used by window_curve.py)
"""
import argparse
import json
import time

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

import config as C  # noqa: E402
from src.data import list_csvs, read_features  # noqa: E402
from src.metrics import evaluate, single_event_latency_ms  # noqa: E402
from src.window_features import file_window_stats  # noqa: E402

TRAIN_CAP, TEST_CAP, VAL_CAP = 10_000, 5_000, 2_000


def sample(a: np.ndarray, cap: int, rng) -> np.ndarray:
    return a if len(a) <= cap else a[np.sort(rng.choice(len(a), cap, replace=False))]


def build_train_val(w: int, rng):
    xtr, ytr, xva, yva = [], [], [], []
    for i, (cls, files) in enumerate(list_csvs(C.TRAIN_DIR).items()):
        tr_parts, va_parts = [], []
        for f in files:
            st = file_window_stats(read_features(f), w)       # row j = window ending at flow j+w-1
            n = len(st)
            cut = int(n * 0.9)
            tr_parts.append(st[:max(cut - w, 0):C.TRAIN_STRIDE].copy())   # gap of w -> no shared flows
            va_parts.append(st[cut::C.TEST_STRIDE].copy())   # .copy() frees the full array
        tr, va = np.concatenate(tr_parts), np.concatenate(va_parts)
        tr, va = sample(tr, TRAIN_CAP, rng), sample(va, VAL_CAP, rng)
        xtr.append(tr); ytr.append(np.full(len(tr), i))
        xva.append(va); yva.append(np.full(len(va), i))
    return np.concatenate(xtr), np.concatenate(ytr), np.concatenate(xva), np.concatenate(yva)


def build_test(w: int, rng):
    xs, ys = [], []
    for i, (cls, files) in enumerate(list_csvs(C.TEST_DIR).items()):
        parts = [file_window_stats(read_features(f), w)[::C.TEST_STRIDE].copy() for f in files]
        a = sample(np.concatenate(parts), TEST_CAP, rng)
        xs.append(a); ys.append(np.full(len(a), i))
    return np.concatenate(xs), np.concatenate(ys)


def run_one(w: int) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(C.SEED)
    xtr, ytr, xva, yva = build_train_val(w, rng)
    xte, yte = build_test(w, rng)

    cnt = np.bincount(ytr, minlength=len(C.CLASSES))
    small = {c: TRAIN_CAP for c, k in enumerate(cnt) if 0 < k < TRAIN_CAP}
    k = int(min(5, min(cnt[c] for c in small) - 1)) if small else 5
    if small:
        xtr, ytr = SMOTE(sampling_strategy=small, k_neighbors=k, random_state=C.SEED).fit_resample(xtr, ytr)

    model = RandomForestClassifier(n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=C.SEED)
    model.fit(xtr, ytr)
    val_f1 = f1_score(yva, model.predict(xva), average="weighted")

    proba = model.predict_proba(xte)
    model.n_jobs = 1
    lat = single_event_latency_ms(lambda s: model.predict_proba(s), xte[:1])
    res = evaluate(f"rf_win{w}", yte, proba, lat)
    joblib.dump(model, C.MODELS_DIR / f"rf_win{w}.joblib", compress=3)

    row = {"window": w, "val_weighted_f1": float(val_f1), "test_weighted_f1": res["weighted_f1"],
           "test_macro_f1": res["macro_f1"], "test_accuracy": res["accuracy"],
           "min_test_windows": int(np.bincount(yte).min()), "seconds": round(time.time() - t0)}
    print(f"== w={w}: val wF1={val_f1:.4f}  test wF1={res['weighted_f1']:.4f}  "
          f"smallest test class={row['min_test_windows']} windows  ({row['seconds']}s)\n")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, required=True)
    w = ap.parse_args().window
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    row = run_one(w)
    (C.RESULTS_DIR / f"rf_win{w}_window.json").write_text(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()