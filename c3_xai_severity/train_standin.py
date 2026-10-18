"""Train a stand-in detector so you can build Component 3 without waiting for Component 2.

  python train_standin.py                       # synthetic data (works right now)
  python train_standin.py --csv path/to/file.csv --label-col <name>   # real CICIoMT2024 CSV

For real data: every column except the label must be numeric. Drop or encode anything else first.
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src import config, taxonomy

CLASSES = [config.BENIGN_LABEL] + list(taxonomy.ATTACK_SEVERITY)


def synthetic(n_per_class=120, n_features=46, seed=0):
    rng = np.random.default_rng(seed)
    cols = [f"f{i}" for i in range(n_features)]
    frames = []
    for k, cls in enumerate(CLASSES):
        centre = rng.normal(0, 3, n_features)
        data = rng.normal(centre, 1.0, (n_per_class, n_features))
        df = pd.DataFrame(data, columns=cols)
        df["label"] = cls
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--max-rows", type=int, default=200000)
    ap.add_argument("--out", default=config.MODEL_PATH)
    a = ap.parse_args()

    if a.csv:
        df = pd.read_csv(a.csv)
        if len(df) > a.max_rows:
            df = df.sample(a.max_rows, random_state=0)
        df = df.rename(columns={a.label_col: "label"})
    else:
        print("No --csv given: using synthetic data.")
        df = synthetic()

    X, y = df.drop(columns=["label"]), df["label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = RandomForestClassifier(n_estimators=60, max_depth=14, n_jobs=-1, random_state=42)
    model.fit(Xtr, ytr)
    print(classification_report(yte, model.predict(Xte), zero_division=0))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    joblib.dump(model, a.out)
    print("saved", a.out)

    unknown = sorted(set(model.classes_) - set(CLASSES))
    if unknown:
        print("\nWARNING: these labels are not in src/taxonomy.py, fix the spelling or add them:")
        for u in unknown:
            print("  ", u)


if __name__ == "__main__":
    main()
