"""Train a stand-in detector so you can build Component 3 without waiting for Component 2.

  python train_standin.py                                          # synthetic data (works right now)
  python train_standin.py --csv path/to/file.csv --label-col <name>   # one CSV that has a label column
  python train_standin.py --folder path/to/CSV_folder                # CICIoMT2024 folder (label = file name)

Features and classes come from the group's contracts (45 features, 19 classes), so the model fits
Component 2's payloads. For real data the CSV needs all 45 feature columns (names as in contracts/features.json).
"""
import argparse
import glob
import os
import re

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


# The Kaggle version of CICIoMT2024 (train_iomt.csv / test_iomt.csv) uses short names; map them to contracts/classes.json.
LABEL_MAP = {
    "benign": "Benign", "ARP Spoofing": "ARP_Spoofing",
    "DDoS Connect Flood": "MQTT-DDoS-Connect_Flood", "DDoS Publish Flood": "MQTT-DDoS-Publish_Flood",
    "DoS Connect Flood": "MQTT-DoS-Connect_Flood", "DoS Publish Flood": "MQTT-DoS-Publish_Flood",
    "Malformed Data": "MQTT-Malformed_Data",
    "OS Scan": "Recon-OS_Scan", "Ping Sweep": "Recon-Ping_Sweep", "Port Scan": "Recon-Port_Scan",
    "Recon VulScan": "Recon-VulScan",
    **{f"DDoS {p}": f"TCP_IP-DDoS-{p}" for p in ("ICMP", "SYN", "TCP", "UDP")},
    **{f"DoS {p}": f"TCP_IP-DoS-{p}" for p in ("ICMP", "SYN", "TCP", "UDP")},
}


def label_from_filename(path):
    """'TCP_IP-DDoS-SYN3_train.pcap.csv' -> 'TCP_IP-DDoS-SYN' (how contracts/classes.json names are made)."""
    name = os.path.basename(path)
    name = re.sub(r"\.pcap\.csv$|\.csv$", "", name, flags=re.I)
    name = re.sub(r"_(train|test)$", "", name, flags=re.I)
    return re.sub(r"\d+$", "", name)


def load_folder(folder, per_file):
    """Read every CSV in a CICIoMT2024 folder; the attack name comes from the file name."""
    files = sorted(glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True))
    if not files:
        raise SystemExit(f"No .csv files found in {folder}")
    parts = []
    for f in files:
        d = pd.read_csv(f)
        d.columns = [c.strip() for c in d.columns]
        if len(d) > per_file:
            d = d.sample(per_file, random_state=0)
        d["label"] = label_from_filename(f)
        parts.append(d)
        print(f"  {os.path.basename(f):<55} -> {d['label'].iloc[0]:<28} {len(d)} rows")
    return pd.concat(parts, ignore_index=True)


def load_big_csv(path, label_col, per_class, chunk=200_000):
    """Read a huge CSV in chunks and keep at most `per_class` random rows of each label (saves RAM)."""
    kept, have = [], {}
    for part in pd.read_csv(path, chunksize=chunk, low_memory=False):
        part.columns = [c.strip() for c in part.columns]
        if label_col not in part.columns:
            raise SystemExit(f"No column called '{label_col}'. Columns are: {list(part.columns)}")
        part = part.rename(columns={label_col: "label"}).sample(frac=1, random_state=0)
        part["label"] = part["label"].astype(str).str.strip().map(lambda x: LABEL_MAP.get(x, x))
        for lab, g in part.groupby("label"):
            room = per_class - have.get(lab, 0)
            if room > 0:
                kept.append(g.head(room))
                have[lab] = have.get(lab, 0) + min(room, len(g))
    df = pd.concat(kept, ignore_index=True)
    print("rows kept per label:")
    print(df["label"].value_counts().to_string())
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--test", help="optional separate test CSV (e.g. test_iomt.csv) for an honest accuracy check")
    ap.add_argument("--folder", help="folder of CICIoMT2024 CSVs; label is taken from each file name")
    ap.add_argument("--per-class", type=int, default=20000, help="--csv mode: max rows kept per label")
    ap.add_argument("--per-file", type=int, default=20000, help="max rows kept per CSV file (keeps classes balanced)")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--max-rows", type=int, default=500000)
    ap.add_argument("--out", default=config.MODEL_PATH)
    a = ap.parse_args()

    if a.csv or a.folder:
        if a.folder:
            df = load_folder(a.folder, a.per_file)
        else:
            df = load_big_csv(a.csv, a.label_col, a.per_class)
        if len(df) > a.max_rows:
            df = df.sample(a.max_rows, random_state=0)
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            raise SystemExit(f"CSV is missing contract feature columns: {missing}")
        df = df[FEATURES + ["label"]].replace([np.inf, -np.inf], np.nan).dropna()
    else:
        print("No --csv given: using synthetic data.")
        df = synthetic()

    X, y = df[FEATURES], df["label"]
    if a.test:
        tdf = load_big_csv(a.test, a.label_col, 5000)[FEATURES + ["label"]].replace([np.inf, -np.inf], np.nan).dropna()
        Xtr, ytr, Xte, yte = X, y, tdf[FEATURES], tdf["label"]
        print("\nEvaluating on the separate test file.")
    else:
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