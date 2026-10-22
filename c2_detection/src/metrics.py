"""Shared evaluation - every model (RF, MLP, LSTM) is scored with exactly this code."""
import json
import time

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

from config import CLASSES, RESULTS_DIR


def evaluate(name: str, y_true: np.ndarray, proba: np.ndarray, latency_ms: float) -> dict:
    """proba: (n, 19) class probabilities. Saves results/<name>_*.json/csv."""
    y_pred = proba.argmax(axis=1)
    labels = list(range(len(CLASSES)))
    res = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "latency_ms_per_event": latency_ms,
    }
    report = classification_report(y_true, y_pred, labels=labels, target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{name}_metrics.json").write_text(json.dumps(res, indent=2))
    (RESULTS_DIR / f"{name}_per_class.json").write_text(json.dumps(report, indent=2))
    np.savetxt(RESULTS_DIR / f"{name}_confusion.csv", cm, fmt="%d", delimiter=",")

    print(f"\n[{name.upper()}]  accuracy={res['accuracy']:.4f}  weighted-F1={res['weighted_f1']:.4f}  "
          f"macro-F1={res['macro_f1']:.4f}  latency={latency_ms:.2f} ms/event")
    print(f"\n  {'class':<26} {'prec':>6} {'recall':>6} {'f1':>6} {'n':>6}")
    for c in CLASSES:
        r = report[c]
        print(f"  {c:<26} {r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1-score']:>6.3f} {int(r['support']):>6}")
    return res


def single_event_latency_ms(predict_one, sample, n: int = 200) -> float:
    """Average time to classify ONE event - this is what the API does per request."""
    for _ in range(10):
        predict_one(sample)                       # warm-up
    t0 = time.perf_counter()
    for _ in range(n):
        predict_one(sample)
    return (time.perf_counter() - t0) / n * 1000