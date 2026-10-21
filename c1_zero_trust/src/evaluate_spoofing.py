"""
C1 Function 1 - identity-compromise evaluation against ARP_Spoofing.

The certificate-only baseline accepts any session with a valid credential, so by
construction it detects 0% of credential-valid impersonation. The behavioural
layer is asked the separate question: does this traffic still look like the
device it claims to be?

Every spoofing window is scored against every device baseline. A spoofing
session claims to be some device; since the claim is not recorded in the
capture, scoring against all baselines and reporting the *best* score the
attacker could achieve is the conservative measure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from baseline import fit_all, load_windows, temporal_split  # noqa: E402
from build_dataset import build  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "out"

THRESHOLD = 0.5  # provisional; the trade-off curve sweeps this


def ensure_spoofing_csv() -> Path:
    out = PROC / "arp_spoofing.csv"
    if not out.exists():
        src = RAW / "test" / "ARP_Spoofing_test.pcap"
        print(f"extracting {src.name} ...", flush=True)
        build([src], out, window_s=30.0, stride_s=15.0, label="ARP_Spoofing")
    return out


if __name__ == "__main__":
    spoof = pd.read_csv(ensure_spoofing_csv())
    print(f"spoofing windows: {len(spoof)}")

    df = load_windows(PROC / "profiling_train.csv", PROC / "profiling_test.csv")
    train_df, test_df = temporal_split(df, frac=0.7)

    for algo in ("isolation_forest", "one_class_svm"):
        baselines = fit_all(train_df, algorithm=algo)
        best = np.zeros(len(spoof))
        per_device = {}
        for device, bl in sorted(baselines.items()):
            s = bl.score(spoof)
            per_device[device] = float(s.mean())
            best = np.maximum(best, s)

        benign_scores = np.concatenate(
            [baselines[d].score(test_df[test_df["device"] == d])
             for d in baselines if len(test_df[test_df["device"] == d]) > 0])

        detected = float((best < THRESHOLD).mean())
        false_restrict = float((benign_scores < THRESHOLD).mean())

        print(f"\n=== {algo} ===")
        print("  mean spoof score per baseline:")
        for d, v in per_device.items():
            print(f"    {d:<24} {v:.3f}")
        print(f"  best-case attacker score (mean) : {best.mean():.3f}")
        print(f"  legitimate held-out score (mean): {benign_scores.mean():.3f}")
        print(f"  detection rate  @{THRESHOLD}      : {detected:.1%}")
        print(f"  false-restrict  @{THRESHOLD}      : {false_restrict:.1%}")
        print(f"  certificate-only baseline        : 0.0%  (valid credential accepted)")

        # threshold sweep -> the trade-off curve
        rows = []
        for t in np.arange(0.05, 1.0, 0.05):
            rows.append({
                "threshold": round(float(t), 2),
                "detection_rate": float((best < t).mean()),
                "false_restriction_rate": float((benign_scores < t).mean()),
            })
        pd.DataFrame(rows).to_csv(OUT / f"tradeoff_{algo}.csv", index=False)
    print(f"\ntrade-off curves written to {OUT}")
