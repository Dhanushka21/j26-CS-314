"""STEP 9 / 13: compare RF vs MLP vs LSTM vs RF-on-window-stats, pick the best, draw the charts.

Run from repo root:   python c2_detection/compare.py
Creates:
  results/comparison.csv           one row per model (all metrics)
  results/comparison.png           accuracy / weighted-F1 / macro-F1 bars + 73.3% baseline + 90% target
  results/per_class_f1.png         F1 of every class for the 3 models
  results/<best>_confusion.png     normalised confusion matrix of the best model
  models/best_model.json           which model the API (Step 10) will serve
"""
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config as C  # noqa: E402

MODELS = ["rf", "mlp", "lstm"]                      # main comparison (no IAT)
LABELS = {"rf": "Random Forest", "mlp": "MLP", "lstm": "LSTM"}

# Step 13: add the window-statistics RF whose window was chosen on VALIDATION (window_curve.py)
_choice = C.RESULTS_DIR / "window_choice.json"
WIN = json.loads(_choice.read_text())["best_window"] if _choice.exists() else None
if WIN:
    MODELS.append(f"rf_win{WIN}")
    LABELS[f"rf_win{WIN}"] = f"RF window-stats (w={WIN})"


def load_metrics() -> pd.DataFrame:
    rows = []
    for m in MODELS + [f"{m}_withIAT" for m in MODELS]:
        p = C.RESULTS_DIR / f"{m}_metrics.json"
        if p.exists():
            rows.append(json.loads(p.read_text()))
    if not rows:
        raise SystemExit("No results found - run train_rf.py / train_nn.py first.")
    return pd.DataFrame(rows).set_index("model")


def plot_overall(df: pd.DataFrame):
    main = df.loc[[m for m in MODELS if m in df.index]]
    main = main.rename(index=LABELS)
    ax = main[["accuracy", "weighted_f1", "macro_f1"]].plot.bar(figsize=(10, 5), rot=0, width=0.75)
    ax.axhline(C.BASELINE_MULTICLASS_ACC, ls="--", c="red", label="Yapa et al. 2025 (73.3%)")
    ax.axhline(C.TARGET_WEIGHTED_F1, ls=":", c="green", label="Target (90%)")
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.2f", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (test set)")
    ax.set_title("C2: 19-class detection on CICIoMT2024 (IAT removed, test set)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(C.RESULTS_DIR / "comparison.png", dpi=150)
    plt.close()


def plot_per_class():
    data = {}
    for m in MODELS:
        p = C.RESULTS_DIR / f"{m}_per_class.json"
        if p.exists():
            rep = json.loads(p.read_text())
            data[LABELS[m]] = [rep[c]["f1-score"] for c in C.CLASSES]
    df = pd.DataFrame(data, index=C.CLASSES)
    ax = df.plot.barh(figsize=(9, 9), width=0.8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1-score (test set)")
    ax.set_title("Per-class F1: RF vs MLP vs LSTM")
    plt.tight_layout()
    plt.savefig(C.RESULTS_DIR / "per_class_f1.png", dpi=150)
    plt.close()


def plot_confusion(best: str):
    cm = np.loadtxt(C.RESULTS_DIR / f"{best}_confusion.csv", delimiter=",")
    cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)          # each row sums to 1 (= recall)
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(C.CLASSES)), C.CLASSES, rotation=90, fontsize=7)
    ax.set_yticks(range(len(C.CLASSES)), C.CLASSES, fontsize=7)
    for i in range(len(C.CLASSES)):
        for j in range(len(C.CLASSES)):
            if cm[i, j] >= 0.05:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Normalised confusion matrix - {LABELS.get(best, best)}")
    fig.colorbar(im, fraction=0.046)
    plt.tight_layout()
    plt.savefig(C.RESULTS_DIR / f"{best}_confusion.png", dpi=150)
    plt.close()


def main():
    df = load_metrics()
    df.to_csv(C.RESULTS_DIR / "comparison.csv")
    show = df[["accuracy", "weighted_f1", "macro_f1", "weighted_recall", "latency_ms_per_event"]]
    print(show.round(4).to_string())

    candidates = [m for m in MODELS if m in df.index]
    best = df.loc[candidates, "weighted_f1"].idxmax()
    b = df.loc[best]
    print(f"\nBEST = {best.upper()}   weighted-F1={b['weighted_f1']:.4f}   macro-F1={b['macro_f1']:.4f}   "
          f"latency={b['latency_ms_per_event']:.2f} ms")
    print(f"  vs Yapa et al. baseline 73.3% : {(b['accuracy'] - C.BASELINE_MULTICLASS_ACC) * 100:+.1f} points accuracy")
    print(f"  vs target 90% weighted-F1    : {(b['weighted_f1'] - C.TARGET_WEIGHTED_F1) * 100:+.1f} points")
    print(f"  latency < {C.MAX_LATENCY_MS} ms             : {b['latency_ms_per_event'] < C.MAX_LATENCY_MS}")

    is_win = best.startswith("rf_win")
    (C.MODELS_DIR / "best_model.json").write_text(json.dumps({
        "model": "rf_win" if is_win else best,
        "window": WIN if is_win else C.WINDOW_SIZE,
        "weighted_f1": float(b["weighted_f1"]),
        "macro_f1": float(b["macro_f1"]),
        "dropped_features": C.DROP_FEATURES,
    }, indent=2))

    plot_overall(df)
    plot_per_class()
    plot_confusion(best)
    print("\nsaved: results/comparison.csv, comparison.png, per_class_f1.png, "
          f"{best}_confusion.png, models/best_model.json")


if __name__ == "__main__":
    main()