"""Step 4: train the ML severity classifier and compare it with the rule-based baseline.

  python train_severity.py                 # uses models/detector.joblib and synthetic data
  python train_severity.py --csv file.csv --label-col <name>

Honest warning: if the labels come only from attack type and device type, the model just
re-learns those rules and scores ~100%. The comparison is only interesting when the label
policy includes something continuous (e.g. set C3_ESCALATE_CONF=0.97 so High becomes Critical
on high-confidence detections). Agree that policy with your supervisor first.
"""
import argparse
import os
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src import config, taxonomy
from src.contract_loader import DEVICES, FEATURES
from src.explain import Explainer
from src.severity import ml_features
from train_standin import load_big_csv, synthetic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--sample", type=int, default=1500, help="rows used (SHAP is slow, 1500 is enough)")
    ap.add_argument("--out", default=config.SEVERITY_MODEL_PATH)
    a = ap.parse_args()

    detector = joblib.load(config.MODEL_PATH)
    explainer = Explainer(detector)

    if a.csv:
        df = load_big_csv(a.csv, a.label_col, per_class=300)      # chunked read + label-name mapping
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    else:
        df = synthetic(n_per_class=100)
    # attacks only, and only classes the detector model actually knows
    df = df[df["label"].isin(taxonomy.ATTACK_SEVERITY) & df["label"].isin(explainer.classes)]
    df = df.sample(min(a.sample, len(df)), random_state=0)

    devices = list(DEVICES)
    rows, labels = [], []
    for _, r in df.iterrows():
        x = [float(r[c]) for c in FEATURES]
        cls = r["label"]
        proba = detector.predict_proba(pd.DataFrame([x], columns=FEATURES))[0]
        order = np.argsort(-proba)
        conf = float(proba[explainer.classes.index(cls)])
        margin = float(proba[order[0]] - proba[order[1]])
        _, mag = explainer.analyse(x, cls)
        dev = random.choice(devices)
        lc = taxonomy.device_is_life_critical(dev)
        rows.append(ml_features(cls, conf, mag, margin, lc).iloc[0])
        base = taxonomy.rule_severity(cls, conf, config.ESCALATE_CONF)
        labels.append(taxonomy.apply_override(base, lc)[0])

    X, y = pd.DataFrame(rows), pd.Series(labels)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    tree = DecisionTreeClassifier(max_depth=6, random_state=42).fit(Xtr, ytr)
    print("Decision Tree on held-out data:")
    print(classification_report(yte, tree.predict(Xte), zero_division=0))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    joblib.dump(tree, a.out)
    print("saved", a.out, "| run the service with C3_SEVERITY_MODE=ml to use it")


if __name__ == "__main__":
    main()
