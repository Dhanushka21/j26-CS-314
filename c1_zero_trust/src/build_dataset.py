"""Run feature extraction across the profiling captures and write a CSV."""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from features import FEATURES, device_from_filename, extract  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

META = ["device", "activity", "t_start", "n_packets", "src_mac", "source_file"]

# captures that are not a single labelled device
SKIP = {"Idle", "Benign_test", "ARP_Spoofing_test", "ARP_Spoofing_train"}


def collect(folder: Path, skip_unlabelled: bool = True) -> list[Path]:
    files = sorted(folder.glob("*.pcap"))
    if skip_unlabelled:
        files = [f for f in files if device_from_filename(f) not in SKIP]
    return files


def build(files: list[Path], out_csv: Path, window_s: float, stride_s: float,
          label: str | None = None) -> int:
    total = 0
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=META + (["label"] if label else []) + FEATURES)
        writer.writeheader()
        for f in files:
            t0 = time.time()
            rows = extract(f, window_s=window_s, stride_s=stride_s)
            for r in rows:
                r["source_file"] = f.name
                if label:
                    r["label"] = label
                writer.writerow(r)
            total += len(rows)
            print("  %-46s %4d windows  (%.1fs)" % (f.name, len(rows), time.time() - t0),
                  flush=True)
    return total


if __name__ == "__main__":
    window_s, stride_s = 30.0, 15.0

    print("== train profiling ==", flush=True)
    n = build(collect(RAW / "train"), OUT / "profiling_train.csv", window_s, stride_s)
    print("train windows:", n, flush=True)

    print("== test profiling ==", flush=True)
    n = build(collect(RAW / "test"), OUT / "profiling_test.csv", window_s, stride_s)
    print("test windows:", n, flush=True)
