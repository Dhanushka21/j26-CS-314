"""Dataset helpers for C2.

CICIoMT2024 CSVs have NO label column and NO timestamp/device column.
- Label : from the file name   'TCP_IP-DDoS-SYN3_train.pcap.csv' -> 'TCP_IP-DDoS-SYN'
- Order : rows inside one file follow the original capture order, so consecutive
          rows of ONE file form a time-ordered sequence. Windows never cross files.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import CLASSES, FEATURES

# matches the ending '_train.pcap.csv' or '_test.pcap.csv'
_SUFFIX = re.compile(r"_(train|test)\.pcap\.csv$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Step 3: labels + file lists
# ---------------------------------------------------------------------------
def label_from_filename(name: str) -> str:
    """'TCP_IP-DDoS-SYN3_train.pcap.csv' -> 'TCP_IP-DDoS-SYN'"""
    base = _SUFFIX.sub("", name)       # remove '_train.pcap.csv'
    base = re.sub(r"\d+$", "", base)   # remove the part number at the end (1, 2, 3...)
    return base


def list_csvs(split_dir: Path) -> dict[str, list[Path]]:
    """Group the CSV files of one folder (Train or Test) by class.
    Returns {class_name: [file paths]} in classes.json order."""
    by_class = {c: [] for c in CLASSES}
    for path in sorted(split_dir.glob("*.csv")):
        label = label_from_filename(path.name)
        if label in by_class:
            by_class[label].append(path)
        else:
            print(f"[warn] unknown class '{label}' from file {path.name} - skipped")

    missing = [c for c, files in by_class.items() if not files]
    if missing:
        print(f"[warn] no files for: {missing}")
    return by_class


# ---------------------------------------------------------------------------
# Step 4: read features + build sliding windows
# ---------------------------------------------------------------------------
def read_features(csv_path: Path) -> np.ndarray:
    """Read one CSV and return its 45 features as float32, in features.json order."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    x = df[FEATURES].to_numpy(dtype=np.float32)
    # safety: the dataset has no NaN/inf, but live traffic in the API might
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def window_starts(n_rows: int, window: int, stride: int) -> np.ndarray:
    """Start index of every window inside one file.
    n_rows=25, window=10, stride=5  ->  [0, 5, 10, 15]"""
    if n_rows < window:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, n_rows - window + 1, stride, dtype=np.int64)


def build_class_windows(files: list[Path], window: int, stride: int, cap: int,
                        rng: np.random.Generator) -> np.ndarray:
    """All windows of ONE class -> randomly keep at most `cap` of them.
    Returns array of shape (n_windows, window, 45).
    A window never crosses from one file into another."""
    per_file = []                                   # [(rows of file, start indices)]
    for f in files:
        x = read_features(f)
        starts = window_starts(len(x), window, stride)
        if len(starts):
            per_file.append((x, starts))

    total = sum(len(s) for _, s in per_file)
    if total == 0:
        return np.empty((0, window, len(FEATURES)), dtype=np.float32)

    keep = min(cap, total)
    chosen = np.sort(rng.choice(total, size=keep, replace=False))   # global window ids

    out, offset = [], 0
    for x, starts in per_file:
        mine = chosen[(chosen >= offset) & (chosen < offset + len(starts))] - offset
        for s in starts[mine]:
            out.append(x[s:s + window])
        offset += len(starts)
    return np.stack(out).astype(np.float32)


def build_split(split_dir: Path, window: int, stride: int, cap: int, seed: int):
    """Windows for every class of one folder.
    X: (n, window, 45) raw values (not scaled yet),  y: (n,) class index 0..18"""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for idx, (cls, files) in enumerate(list_csvs(split_dir).items()):
        w = build_class_windows(files, window, stride, cap, rng)
        print(f"  {idx:>2} {cls:<26} files={len(files):<2} windows={len(w):>6}")
        xs.append(w)
        ys.append(np.full(len(w), idx, dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys)