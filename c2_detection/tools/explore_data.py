"""STEP 3: understand the dataset before building anything.

Run from repo root:   python c2_detection/tools/explore_data.py
Creates:
  results/class_distribution.csv   rows per class (train / test)
  results/class_distribution.png   bar chart (log scale) - for the progress presentation
  results/data_quality.csv         column / NaN / inf / constant-column checks
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")                # save charts to files, no pop-up window
import matplotlib.pyplot as plt     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # c2_detection/ on path
import config as C                  # noqa: E402
from src.data import list_csvs      # noqa: E402


def count_rows(path: Path) -> int:
    """Count data rows without loading the whole file into memory."""
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1          # minus the header line


def class_distribution() -> pd.DataFrame:
    rows = []
    for split, folder in (("train", C.TRAIN_DIR), ("test", C.TEST_DIR)):
        print(f"\n== {split}: {folder}")
        for cls, files in list_csvs(folder).items():
            n = sum(count_rows(f) for f in files)
            rows.append({"class": cls, "split": split, "files": len(files), "rows": n})
            print(f"  {cls:<26} files={len(files):<2} rows={n:>10,}")
    df = pd.DataFrame(rows).pivot(index="class", columns="split", values="rows")
    df = df.loc[C.CLASSES, ["train", "test"]]
    df["train_%"] = (df["train"] / df["train"].sum() * 100).round(3)
    return df.sort_values("train", ascending=False)


def data_quality(sample_rows: int = 20_000) -> pd.DataFrame:
    """Read the first `sample_rows` of every train file and check the columns."""
    issues, parts = [], []
    for cls, files in list_csvs(C.TRAIN_DIR).items():
        for f in files:
            df = pd.read_csv(f, nrows=sample_rows)
            cols = [c.strip() for c in df.columns]
            if cols != C.FEATURES:
                issues.append(f"{f.name}: columns differ from features.json")
            parts.append(df.set_axis(cols, axis=1))
    if issues:
        print("\n".join(issues))
    else:
        print("\nAll train files have exactly the 45 features in features.json order.")

    x = pd.concat(parts, ignore_index=True)[C.FEATURES]
    q = pd.DataFrame({
        "nan": x.isna().sum(),
        "inf": np.isinf(x.to_numpy()).sum(axis=0),
        "min": x.min(),
        "max": x.max(),
        "mean": x.mean(),
        "n_unique": x.nunique(),
    })
    q["constant"] = q["n_unique"] <= 1
    return q


def plot_distribution(df: pd.DataFrame, out: Path):
    ax = df[["train", "test"]].plot.barh(figsize=(9, 7), logx=True)
    ax.invert_yaxis()
    ax.set_xlabel("Rows (log scale)")
    ax.set_title("CICIoMT2024 (Wi-Fi & MQTT) - rows per class")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def main():
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dist = class_distribution()
    dist.to_csv(C.RESULTS_DIR / "class_distribution.csv")
    plot_distribution(dist, C.RESULTS_DIR / "class_distribution.png")

    biggest, smallest = dist["train"].idxmax(), dist["train"].idxmin()
    ratio = dist["train"].max() / dist["train"].min()
    print(f"\nTOTAL train={dist['train'].sum():,}  test={dist['test'].sum():,}")
    print(f"Imbalance: {biggest} / {smallest} = {ratio:,.0f}x")

    q = data_quality()
    q.to_csv(C.RESULTS_DIR / "data_quality.csv")
    print(f"NaN total={int(q['nan'].sum())}   inf total={int(q['inf'].sum())}")
    print(f"Constant columns (same value everywhere): {list(q.index[q['constant']])}")
    print(f"\nSaved to {C.RESULTS_DIR}")


if __name__ == "__main__":
    main()