"""STEP 14: blocked k-fold cross validation on the TRAIN files (proposal section 3.5).

  python c2_detection/cross_validate.py --model rf
  python c2_detection/cross_validate.py --model mlp
  python c2_detection/cross_validate.py --model lstm          (slow: set $env:C2_EPOCHS="10")
  python c2_detection/cross_validate.py --model rf_win --window 150
  python c2_detection/cross_validate.py --summary             (table of all finished models)

Why "blocked" and not random k-fold?
  Neighbouring windows share flows. A random split would put almost the same window in train and
  validation and give a fake, too-high score. So every train file is cut into K time blocks:
    fold k validation = windows that END inside block k
    fold k train      = windows that do not touch block k (plus a gap of w flows before it)
  Every model gets exactly the same folds, scaling/SMOTE are refit inside each fold (no leakage).
Output: results/cv_<model>.json   (per-fold + mean / std of weighted-F1 and macro-F1)
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import config as C
from src.data import list_csvs, read_features
from src.nets import LSTMClassifier, MLP, predict_proba, train_model
from src.preprocess import fit_scaler, smote_windows, transform
from src.window_features import file_window_stats

K = 5
TRAIN_CAP, VAL_CAP = 10_000, 2_000


def sample(a, cap, rng):
    return a if len(a) <= cap else a[np.sort(rng.choice(len(a), cap, replace=False))]


def fold_data(kind: str, w: int, k: int, rng):
    """Build train / val arrays for fold k. Files are read again for every fold to keep RAM low."""
    xtr, ytr, xva, yva = [], [], [], []
    for ci, (cls, files) in enumerate(list_csvs(C.TRAIN_DIR).items()):
        tr_parts, va_parts = [], []
        for f in files:
            raw = read_features(f)
            n = len(raw)
            if n < w:
                continue
            starts = np.arange(0, n - w + 1)
            ends = starts + w - 1
            lo, hi = n * k // K, n * (k + 1) // K
            va_idx = np.where((ends >= lo) & (ends < hi))[0][::C.TEST_STRIDE]
            tr_idx = np.where((ends <= lo - w) | (starts >= hi))[0][::C.TRAIN_STRIDE]
            if kind == "rf_win":
                st = file_window_stats(raw, w)              # row j = window starting at flow j
                tr_parts.append(st[tr_idx].copy())
                va_parts.append(st[va_idx].copy())
                del st
            else:                                            # raw windows (n, w, 45)
                empty = np.empty((0, w, raw.shape[1]), np.float32)
                tr_parts.append(np.stack([raw[s:s + w] for s in tr_idx]) if len(tr_idx) else empty)
                va_parts.append(np.stack([raw[s:s + w] for s in va_idx]) if len(va_idx) else empty)
        tr = sample(np.concatenate(tr_parts), TRAIN_CAP, rng)
        va = sample(np.concatenate(va_parts), VAL_CAP, rng)
        xtr.append(tr); ytr.append(np.full(len(tr), ci))
        xva.append(va); yva.append(np.full(len(va), ci))
    return np.concatenate(xtr), np.concatenate(ytr), np.concatenate(xva), np.concatenate(yva)


def run_fold(kind: str, w: int, k: int) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(C.SEED + k)
    xtr, ytr, xva, yva = fold_data(kind, w, k, rng)

    if kind == "rf_win":
        cnt = np.bincount(ytr, minlength=len(C.CLASSES))
        small = {c: TRAIN_CAP for c, n in enumerate(cnt) if 0 < n < TRAIN_CAP}
        if small:
            kn = int(min(5, min(cnt[c] for c in small) - 1))
            xtr, ytr = SMOTE(sampling_strategy=small, k_neighbors=kn, random_state=C.SEED).fit_resample(xtr, ytr)
        model = RandomForestClassifier(n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=C.SEED)
        pred = model.fit(xtr, ytr).predict(xva)
    else:
        scaler = fit_scaler(xtr)                                   # fit inside the fold only
        xtr, xva = transform(xtr, scaler), transform(xva, scaler)
        xtr, ytr = smote_windows(xtr, ytr, C.SMOTE_TARGET_PER_CLASS, C.SEED)
        idx = C.MODEL_FEATURE_IDX
        if kind in ("rf", "mlp"):                                  # newest flow only
            xtr, xva = np.ascontiguousarray(xtr[:, -1, idx]), np.ascontiguousarray(xva[:, -1, idx])
        else:
            xtr, xva = np.ascontiguousarray(xtr[:, :, idx]), np.ascontiguousarray(xva[:, :, idx])
        if kind == "rf":
            model = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=C.SEED)
            pred = model.fit(xtr, ytr).predict(xva)
        else:
            torch.manual_seed(C.SEED)
            net = MLP(len(idx), len(C.CLASSES)) if kind == "mlp" else LSTMClassifier(len(idx), len(C.CLASSES))
            # early stopping uses 10% of the fold's TRAIN data, never the fold's validation block
            x_a, x_es, y_a, y_es = train_test_split(xtr, ytr, test_size=0.1, stratify=ytr, random_state=C.SEED)
            net, _ = train_model(net, x_a, y_a, x_es, y_es, C.EPOCHS, C.BATCH_SIZE, C.LR, C.PATIENCE)
            pred = predict_proba(net, xva).argmax(1)

    res = {"fold": k + 1, "weighted_f1": float(f1_score(yva, pred, average="weighted")),
           "macro_f1": float(f1_score(yva, pred, average="macro")), "n_val": int(len(yva)),
           "seconds": round(time.time() - t0)}
    print(f"  fold {k + 1}/{K}: weighted-F1={res['weighted_f1']:.4f}  macro-F1={res['macro_f1']:.4f}  "
          f"({res['seconds']}s)", flush=True)
    return res


def summary():
    rows = []
    for p in sorted(C.RESULTS_DIR.glob("cv_*.json")):
        d = json.loads(p.read_text())
        rows.append({"model": d["model"], "folds": len(d["folds"]),
                     "weighted_f1": f"{d['weighted_f1_mean']:.4f} ± {d['weighted_f1_std']:.4f}",
                     "macro_f1": f"{d['macro_f1_mean']:.4f} ± {d['macro_f1_std']:.4f}"})
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS_DIR / "cv_summary.csv", index=False)
    print(df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["rf", "mlp", "lstm", "rf_win"])
    ap.add_argument("--window", type=int, default=150, help="only for rf_win")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        return summary()
    if not a.model:
        raise SystemExit("give --model or --summary")

    w = a.window if a.model == "rf_win" else C.WINDOW_SIZE
    name = f"rf_win{w}" if a.model == "rf_win" else a.model
    print(f"== {K}-fold blocked CV: {name}  (window={w}, dropped={C.DROP_FEATURES})")
    folds = [run_fold(a.model, w, k) for k in range(K)]
    wf = np.array([f["weighted_f1"] for f in folds])
    mf = np.array([f["macro_f1"] for f in folds])
    out = {"model": name, "k": K, "folds": folds,
           "weighted_f1_mean": float(wf.mean()), "weighted_f1_std": float(wf.std()),
           "macro_f1_mean": float(mf.mean()), "macro_f1_std": float(mf.std())}
    (C.RESULTS_DIR / f"cv_{name}.json").write_text(json.dumps(out, indent=2))
    print(f"\n{name}: weighted-F1 = {wf.mean():.4f} ± {wf.std():.4f}   macro-F1 = {mf.mean():.4f} ± {mf.std():.4f}")


if __name__ == "__main__":
    main()