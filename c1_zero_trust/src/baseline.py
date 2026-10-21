"""
C1 Function 1 - per-device behavioural baselines.

Trains one unsupervised anomaly model per device class on that device's benign
traffic only, then scores arbitrary windows against it.

Score convention (matches contracts/samples/c1_to_c2.json):
    1.0 = behaves like the claimed device
    0.0 = does not

sklearn's decision_function is higher-is-normal but unbounded, so it is squashed
with a logistic centred on the model's own training distribution.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

_CONTRACTS = Path(__file__).resolve().parents[1] / "contracts"
FEATURES: list[str] = json.loads((_CONTRACTS / "features.json").read_text())["features"]

# Behaviourally meaningful subset used by the model. The full 45 are still
# carried in the payload to C2; these are the ones that characterise how a
# device behaves rather than what single packet it sent.
MODEL_FEATURES = [
    "Duration", "Rate", "Srate", "Drate",
    "Header_Length", "Protocol Type",
    "ack_count", "syn_count", "fin_count", "rst_count",
    "TCP", "UDP", "ARP", "ICMP", "IPv", "LLC",
    "HTTP", "HTTPS", "DNS", "DHCP",
    "Tot sum", "Min", "Max", "AVG", "Std", "Tot size",
    "IAT", "Number", "Magnitue", "Radius", "Covariance", "Variance", "Weight",
]


@dataclass
class DeviceBaseline:
    """One device's learned normal behaviour."""

    device: str
    algorithm: str = "isolation_forest"
    scaler: StandardScaler = field(default=None, repr=False)
    model: object = field(default=None, repr=False)
    _centre: float = 0.0
    _scale: float = 1.0
    n_train: int = 0

    def fit(self, X: pd.DataFrame) -> "DeviceBaseline":
        self.n_train = len(X)
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X[MODEL_FEATURES].to_numpy(dtype=float))

        if self.algorithm == "isolation_forest":
            self.model = IsolationForest(
                n_estimators=200, contamination=0.02, random_state=42, n_jobs=-1
            ).fit(Xs)
        elif self.algorithm == "one_class_svm":
            self.model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.02).fit(Xs)
        else:
            raise ValueError(f"unknown algorithm {self.algorithm}")

        # calibrate the squash on the training distribution
        raw = self.model.decision_function(Xs)
        self._centre = float(np.median(raw))
        spread = float(np.percentile(raw, 84) - np.percentile(raw, 16)) / 2.0
        self._scale = spread if spread > 1e-9 else 1.0
        return self

    def raw_scores(self, X: pd.DataFrame) -> np.ndarray:
        Xs = self.scaler.transform(X[MODEL_FEATURES].to_numpy(dtype=float))
        return self.model.decision_function(Xs)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """behaviour score in [0, 1]; 1 = normal for this device."""
        raw = self.raw_scores(X)
        z = (raw - self._centre) / self._scale
        return 1.0 / (1.0 + np.exp(-z))


def load_windows(*csv_paths: str | Path) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    # SenseUBaby and SenseU are the same physical device under two capture names
    df["device"] = df["device"].replace({"SenseUBaby": "SenseU"})
    return df


def temporal_split(df: pd.DataFrame, frac: float = 0.7
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each capture chronologically: first `frac` trains, remainder tests.

    Chronological rather than random, so the held-out windows are genuinely
    later traffic from the same device, not interleaved duplicates.
    """
    train_parts, test_parts = [], []
    for _, grp in df.groupby(["device", "source_file"], sort=False):
        grp = grp.sort_values("t_start")
        cut = max(int(len(grp) * frac), 1)
        train_parts.append(grp.iloc[:cut])
        test_parts.append(grp.iloc[cut:])
    return (pd.concat(train_parts, ignore_index=True),
            pd.concat(test_parts, ignore_index=True))


def fit_all(train_df: pd.DataFrame, algorithm: str = "isolation_forest",
            min_windows: int = 12) -> dict[str, DeviceBaseline]:
    baselines: dict[str, DeviceBaseline] = {}
    for device, grp in train_df.groupby("device"):
        if len(grp) < min_windows:
            print(f"  skip {device}: only {len(grp)} training windows")
            continue
        baselines[device] = DeviceBaseline(device=device, algorithm=algorithm).fit(grp)
    return baselines
