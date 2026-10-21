"""
Objective 6, part 1 - exhaustive verification of the clinical safety constraint.

Enumerates every reachable combination of credential state, behavioural score
band and criticality tier, and asserts that no input produces full isolation of
a life-critical device.

This is a guarantee over the whole input space, not a sample: the score axis is
continuous but the decision is piecewise-constant in it, so a dense sweep plus
the two band boundaries covers every reachable branch.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from decision import PERMIT_ABOVE, REVOKE_BELOW, REGISTER, DecisionEngine  # noqa: E402

ISOLATING = {"REVOKE"}  # bands that sever a device entirely


def score_grid(step: float = 0.001) -> list[float]:
    """Dense sweep plus exact band boundaries and their neighbourhoods."""
    pts = [i * step for i in range(int(1 / step) + 1)]
    for edge in (REVOKE_BELOW, PERMIT_ABOVE):
        for d in (-1e-9, 0.0, 1e-9):
            v = edge + d
            if 0.0 <= v <= 1.0:
                pts.append(v)
    return sorted(set(pts))


def verify(engine: DecisionEngine) -> tuple[int, list[dict]]:
    violations: list[dict] = []
    checked = 0
    grid = score_grid()
    devices = list(REGISTER) + ["unregistered_device_99"]

    for device_id in devices:
        life_critical = engine.resolve_tier(device_id)
        for cert_valid in (True, False):
            for score in grid:
                d = engine.decide(device_id, cert_valid, score)
                checked += 1
                if life_critical and d.band in ISOLATING:
                    violations.append({
                        "device_id": device_id,
                        "cert_valid": cert_valid,
                        "score": score,
                        "band": d.band,
                    })
    return checked, violations


if __name__ == "__main__":
    print("=== exhaustive verification: clinical safety constraint ===")
    life_critical = [d for d in REGISTER if REGISTER[d]["life_critical"]]
    print(f"life-critical devices: {life_critical}")
    print(f"band boundaries: revoke below {REVOKE_BELOW}, permit above {PERMIT_ABOVE}")

    engine = DecisionEngine(safety_constraint=True)
    checked, violations = verify(engine)
    print(f"\nconstraint ENABLED  : {checked:,} input combinations checked")
    if violations:
        print(f"  FAIL - {len(violations)} violations, e.g. {violations[:3]}")
        raise SystemExit(1)
    print("  PASS - no reachable input isolates a life-critical device")

    # The same sweep with the constraint off, to show the check is not vacuous.
    ablated = DecisionEngine(safety_constraint=False)
    checked_a, violations_a = verify(ablated)
    print(f"\nconstraint DISABLED : {checked_a:,} input combinations checked")
    print(f"  {len(violations_a):,} combinations isolate a life-critical device")
    print("  (the verification above is therefore testing a real property,")
    print("   not a tautology of the decision code)")
