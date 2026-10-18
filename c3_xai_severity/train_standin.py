"""Train a stand-in detector so you can build Component 3 without waiting for Component 2.

  python train_standin.py                                          # synthetic data (works right now)
  python train_standin.py --csv path/to/file.csv --label-col <name>   # real CICIoMT2024 CSV

Features and classes come from the group's contracts (45 features, 19 classes), so the model fits
Component 2's payloads. For real data the CSV needs all 45 feature columns (names as in contracts/features.json).
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src import config
from src.contract_loader import CLASSES, FEATURES


def synthetic(n_per_class=120, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for cls in CLASSES:
        centre = rng.normal(0, 3, len(FEATURES))
        df = pd.DataFrame(rng.normal(centre, 1.0, (n_per_class, len(FEATURES))), columns=FEATURES)
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
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            raise SystemExit(f"CSV is missing contract feature columns: {missing}")
        df = df[FEATURES + ["label"]].replace([np.inf, -np.inf], np.nan).dropna()
    else:
        print("No --csv given: using synthetic data.")
        df = synthetic()

    X, y = df[FEATURES], df["label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = RandomForestClassifier(n_estimators=60, max_depth=14, n_jobs=-1, random_state=42)
    model.fit(Xtr, ytr)
    print(classification_report(yte, model.predict(Xte), zero_division=0))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    joblib.dump(model, a.out)
    print("saved", a.out)

    unknown = sorted(set(model.classes_) - set(CLASSES))
    if unknown:
        print("\nWARNING: labels not in contracts/classes.json (fix spelling or ask the group):")
        for u in unknown:
            print("  ", u)


if __name__ == "__main__":
    main()
