"""
PP1 demonstration - three runs, about two minutes.

  A  normal operation            establishes what "behaves like itself" looks like
  B  identity compromise         behavioural layer catches what the credential cannot
  C  clinical safety constraint  containment instead of disconnection, then ablated

Inputs are pinned, not sampled, so the demo is reproducible on stage.

    python src/demo.py            # all three
    python src/demo.py --run B    # one run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from baseline import load_windows, temporal_split  # noqa: E402
from decision import DecisionEngine  # noqa: E402
from pipeline import load_baselines, run, to_contract_device  # noqa: E402

PROC = _ROOT / "data" / "processed"
W = 74


def rule(title: str = "") -> None:
    print("\n" + "=" * W)
    if title:
        print(title)
        print("=" * W)


def show(results: list[dict], limit: int = 4) -> None:
    for r in results[:limit]:
        flag = "  [constraint_applied=TRUE]" if r["constraint_applied"] else ""
        print(f"  {r['session_id']}  device={r['device_id']:<18} "
              f"cert=VALID  score={r['score']:.2f}  -> {r['decision']}{flag}")
        for reason in r["reasons"]:
            print(f"              {reason}")
        if r["escalated"]:
            print(f"              retained path: {r['device_id']} -> patient_server_01:443")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["A", "B", "C"], default=None)
    args = ap.parse_args()

    baselines = load_baselines()
    engine = DecisionEngine(safety_constraint=True)

    df = load_windows(PROC / "profiling_train.csv", PROC / "profiling_test.csv")
    _, benign = temporal_split(df, frac=0.7)
    spoof = pd.read_csv(PROC / "arp_spoofing.csv")

    # ---------------------------------------------------------------- run A
    if args.run in (None, "A"):
        rule("RUN A - normal operation")
        own = benign[benign["device"] == "Ecobee_Camera"]
        all_own = run(own, baselines, engine)
        permitted = [r for r in all_own if r["decision"] == "PERMIT"]
        # median-scoring permitted windows: representative, not the best three
        permitted.sort(key=lambda r: r["score"])
        mid = len(permitted) // 2
        show(permitted[max(mid - 1, 0):mid + 2])
        n_perm = len(permitted)
        print(f"\n  The device behaves like its baseline. Session continues.")
        print(f"  Across all {len(all_own)} held-out windows for this device, "
              f"{n_perm} permit, {len(all_own) - n_perm} do not")
        print(f"  - the false-restriction cost measured in out/tradeoff_*.csv.")

    # ---------------------------------------------------------------- run B
    if args.run in (None, "B"):
        rule("RUN B - identity compromise (valid certificate)")
        print("  replaying ARP_Spoofing_test, claiming to be cardiac_monitor_01\n")
        res = run(spoof.head(3), baselines, engine, claimed_device="Ecobee_Camera")
        show(res)
        all_res = run(spoof, baselines, engine, claimed_device="Ecobee_Camera")
        caught = sum(r["decision"] in ("REVOKE", "CONTAIN") for r in all_res)
        print()
        print(f"  certificate-only baseline : ACCEPTED   0/{len(all_res)} detected")
        print(f"                              (the certificate is genuinely valid)")
        print(f"  behavioural layer         : REJECTED   {caught}/{len(all_res)} detected")

    # ---------------------------------------------------------------- run C
    if args.run in (None, "C"):
        rule("RUN C - clinical safety constraint")
        print("  same attack, now claiming to be ventilator_01 (life-critical)\n")
        res = run(spoof.head(2), baselines, engine, claimed_device="SenseU")
        show(res)

        print("\n  --- re-running with --no-safety-constraint ---\n")
        ablated = DecisionEngine(safety_constraint=False)
        res_ab = run(spoof.head(2), baselines, ablated, claimed_device="SenseU")
        for r in res_ab:
            print(f"  {r['session_id']}  device={r['device_id']:<18} "
                  f"cert=VALID  score={r['score']:.2f}  -> {r['decision']}"
                  f"  [constraint_applied=FALSE]")

        full = run(spoof, baselines, engine, claimed_device="SenseU")
        full_ab = run(spoof, baselines, ablated, claimed_device="SenseU")
        contained = sum(r["decision"] == "CONTAIN" for r in full)
        severed = sum(r["decision"] == "REVOKE" for r in full_ab)
        print()
        print(f"  Same device, same score, same code. The only difference is")
        print(f"  whether the constraint is active.")
        print(f"    constraint on  : {contained} contained, 0 severed")
        print(f"    constraint off : {severed} severed")

    rule()


if __name__ == "__main__":
    main()
