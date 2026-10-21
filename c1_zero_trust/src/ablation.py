"""
Objective 6, part 2 - safety-constraint ablation on real data.

Runs the identical pipeline twice over the same inputs, once with the clinical
safety constraint enabled and once disabled, and counts the life-critical
sessions the unconstrained configuration would sever.

That count is the measurement. Reporting "100% of life-critical sessions
sustained" with the constraint on is true by construction; the number that
would have been severed without it is the evidence the constraint does
something.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from baseline import load_windows, temporal_split  # noqa: E402
from decision import REGISTER, DecisionEngine  # noqa: E402
from pipeline import DEVICE_MAP, load_baselines, run, to_contract_device  # noqa: E402

PROC, OUT = _ROOT / "data" / "processed", _ROOT / "out"

LIFE_CRITICAL_CAPTURES = [
    cap for cap, m in DEVICE_MAP.items()
    if REGISTER.get(m["device_id"], {}).get("life_critical")
]


def main() -> None:
    baselines = load_baselines()
    spoof = pd.read_csv(PROC / "arp_spoofing.csv")
    df = load_windows(PROC / "profiling_train.csv", PROC / "profiling_test.csv")
    _, benign_test = temporal_split(df, frac=0.7)

    print("life-critical devices:",
          [d for d in REGISTER if REGISTER[d]["life_critical"]])
    print("captures mapped to them:", LIFE_CRITICAL_CAPTURES)

    rows = []
    for constraint in (True, False):
        engine = DecisionEngine(safety_constraint=constraint)
        severed = 0
        contained = 0
        total_lc = 0

        # impersonation: spoofing traffic claiming to be each life-critical device
        for claimed in LIFE_CRITICAL_CAPTURES:
            res = run(spoof, baselines, engine, claimed_device=claimed)
            total_lc += len(res)
            severed += sum(r["decision"] == "REVOKE" for r in res)
            contained += sum(r["decision"] == "CONTAIN" for r in res)

        # legitimate traffic from those same devices
        for claimed in LIFE_CRITICAL_CAPTURES:
            own = benign_test[benign_test["device"] == claimed]
            if own.empty:
                continue
            res = run(own, baselines, engine)
            total_lc += len(res)
            severed += sum(r["decision"] == "REVOKE" for r in res)
            contained += sum(r["decision"] == "CONTAIN" for r in res)

        rows.append({
            "safety_constraint": constraint,
            "life_critical_sessions": total_lc,
            "severed_REVOKE": severed,
            "contained_CONTAIN": contained,
            "availability_preserved": f"{(total_lc - severed) / total_lc:.1%}" if total_lc else "n/a",
        })

    res = pd.DataFrame(rows)
    print()
    print(res.to_string(index=False))
    res.to_csv(OUT / "ablation_safety_constraint.csv", index=False)

    on = rows[0]
    off = rows[1]
    print()
    print(f"  with the constraint    : {on['severed_REVOKE']} life-critical sessions severed, "
          f"{on['contained_CONTAIN']} contained")
    print(f"  without the constraint : {off['severed_REVOKE']} life-critical sessions severed")
    print(f"  => the constraint prevents {off['severed_REVOKE'] - on['severed_REVOKE']} "
          f"disconnections of life-critical devices across this evaluation")


if __name__ == "__main__":
    main()
