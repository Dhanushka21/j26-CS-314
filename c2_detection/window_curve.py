"""STEP 12b: collect results/rf_win<w>_window.json -> curve + best window (chosen on VALIDATION).

  python c2_detection/window_curve.py
Creates results/window_curve.csv, results/window_curve.png, results/window_choice.json
"""
import json

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config as C  # noqa: E402


def main():
    rows = [json.loads(p.read_text()) for p in C.RESULTS_DIR.glob("rf_win*_window.json")]
    if not rows:
        raise SystemExit("run train_window_rf.py --window <w> first")
    df = pd.DataFrame(rows).sort_values("window")
    df.to_csv(C.RESULTS_DIR / "window_curve.csv", index=False)
    print(df.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["window"], df["val_weighted_f1"], "o-", label="validation weighted-F1")
    ax.plot(df["window"], df["test_weighted_f1"], "s-", label="test weighted-F1")
    ax.plot(df["window"], df["test_macro_f1"], "^--", label="test macro-F1")
    ax.axhline(C.TARGET_WEIGHTED_F1, ls=":", c="green", label="target 90%")
    ax.axhline(C.BASELINE_MULTICLASS_ACC, ls="--", c="red", label="Yapa et al. 73.3%")
    ax.set_xlabel("window length (flows)")
    ax.set_ylabel("score")
    ax.set_ylim(0.6, 1.0)
    ax.set_title("Detection quality vs window length (RF on window statistics)")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(C.RESULTS_DIR / "window_curve.png", dpi=150)
    plt.close()

    best = df.loc[df["val_weighted_f1"].idxmax()]
    choice = {"best_window": int(best["window"]), "chosen_on": "validation weighted-F1",
              "val_weighted_f1": float(best["val_weighted_f1"]),
              "test_weighted_f1": float(best["test_weighted_f1"])}
    (C.RESULTS_DIR / "window_choice.json").write_text(json.dumps(choice, indent=2))
    print(f"\nBest window on VALIDATION: w={choice['best_window']}  "
          f"(val wF1={choice['val_weighted_f1']:.4f}, test wF1={choice['test_weighted_f1']:.4f})")


if __name__ == "__main__":
    main()