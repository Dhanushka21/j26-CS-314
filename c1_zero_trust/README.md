# C1 — Behavioural Zero-Trust Device Authentication

Component C1 of J-26-CS-314. Owner: R. N. Dhanushka (IT23284234).

Two implemented functions:

| | Function | What it does |
|---|---|---|
| **F1** | Behavioural verification pipeline | `PCAP → 30s windows → 45 features → per-device baseline → score ∈ [0,1]` |
| **F2** | Graduated decision engine | `(cert, score, criticality) → PERMIT / RESTRICT / REVOKE / CONTAIN` |

A certificate proves possession of a key. F1 asks the separate question —
**does this traffic still behave like the device it claims to be?** F2 turns
that answer into an access decision that can never disconnect a life-critical
device.

Score convention matches `contracts/samples/c1_to_c2.json`: `1.0` = behaves like
the claimed device, `0.0` = does not.

## Quick start

```bash
python src/demo.py              # the three PP1 runs
python src/verify_constraint.py # exhaustive safety verification
python src/ablation.py          # what the constraint prevents
```

## Results

Trained on CICIoMT2024 WiFi profiling captures (8 devices, 814 windows).
Held-out split is **chronological** — first 70% of each capture trains, the
later 30% is tested — so held-out windows are genuinely later traffic.

### Identity verification (own device vs. impostor)

| Device | AUC (IForest) | AUC (OC-SVM) |
|---|---|---|
| Blink_Camera | 0.954 | 0.946 |
| Blink_Mini_Camera | 0.834 | 0.987 |
| Ecobee_Camera | 0.905 | 0.933 |
| M1T_Camera | 0.942 | 0.988 |
| Multifunctional_Pager | 0.896 | 0.938 |
| Owltron_Camera | 0.894 | 0.822 |
| SenseU | 0.890 | 0.932 |
| Singcall | 0.971 | 0.972 |
| **mean** | **0.911** | **0.940** |

### Identity compromise (ARP_Spoofing_test, 16 windows)

Every spoofing window is scored against every baseline; the reported figure is
the **best** score an attacker could achieve, which is the conservative measure.

| | Isolation Forest | One-Class SVM |
|---|---|---|
| best-case attacker score | 0.080 | 0.000 |
| legitimate held-out score | 0.478 | 0.501 |
| detection @ chosen threshold | 100% | 100% |
| false restriction | 11.5% @ t=0.15 | 12.3% @ t=0.05 |
| **certificate-only baseline** | **0.0%** | **0.0%** |

The certificate-only baseline detects 0% by construction — the credential is
valid, so credential validation cannot distinguish these cases. That gap is the
research claim.

### The trade-off this produces

Full sweep in `out/tradeoff_*.csv`. Isolation Forest:

| threshold | detection | false restriction |
|---|---|---|
| 0.05 | 6.2% | 4.6% |
| 0.10 | 81.2% | 8.0% |
| **0.15** | **100%** | **11.5%** |
| 0.30 | 100% | 21.5% |
| 0.50 | 100% | 46.0% |

**11.5% false restriction is too high to deploy on its own.** That is a real
finding, not a failure: it is precisely why the clinical safety constraint in
Function 2 exists. An imperfect classifier is survivable only when a
false positive on a life-critical device results in containment rather than
disconnection.

## Function 2 — graduated decision engine

Bands come from the measured curve above, not from a guess:

| Condition | Decision |
|---|---|
| certificate invalid | REVOKE |
| score ≥ 0.50 | PERMIT |
| 0.15 ≤ score < 0.50 | RESTRICT — minimum paths, elevated monitoring |
| score < 0.15 | REVOKE |
| score < 0.15 **and life-critical** | **CONTAIN** — retain one clinical path, strip the rest, escalate |

### Exhaustive verification (Objective 6)

Every reachable combination of credential state × score × criticality tier:

```
constraint ENABLED  : 16,080 combinations checked
  PASS - no reachable input isolates a life-critical device

constraint DISABLED : 16,080 combinations checked
  3,468 combinations isolate a life-critical device
```

The second run proves the first is testing a real property rather than
restating the code.

### Ablation on real data (Objective 6)

Identical pipeline, identical inputs, constraint toggled:

| | life-critical sessions | severed | contained | availability |
|---|---|---|---|---|
| constraint on | 115 | **0** | 58 | **100%** |
| constraint off | 115 | **58** | 0 | 49.6% |

**The constraint prevents 58 disconnections of life-critical devices** across
this evaluation. That number is the measurement — "100% sustained" with the
constraint on would be true by construction and prove nothing.

## Layout

```
contracts/      shared team contracts (do not edit alone)
config/
  device_map.json       capture device → simulated clinical device, with justification
src/
  features.py           PCAP → 45 contract features, sliding window      [F1]
  build_dataset.py      batch extraction → data/processed/*.csv          [F1]
  baseline.py           DeviceBaseline: fit / score, temporal split      [F1]
  evaluate.py           identity verification experiment (AUC)           [F1]
  evaluate_spoofing.py  identity-compromise + trade-off sweep            [F1]
  train_and_save.py     fit on all data, persist to models/              [F1]
  decision.py           bands, criticality, safety constraint            [F2]
  verify_constraint.py  exhaustive verification                          [F2]
  ablation.py           what the constraint prevents                     [F2]
  pipeline.py           F1 → F2, emits C1ToC2 / C1ToC3 payloads
  demo.py               the three PP1 runs
models/         device_baselines.joblib, config.json
out/            experiment results, demo transcript, verification log
```

## Contract note

`C1ToC2.access_decision` is currently `PERMIT | RESTRICT | REVOKE`. `CONTAIN` is
emitted internally but downgraded to `RESTRICT` on the wire until the team adds
it. Two additive fields are needed:

```python
access_decision: Literal["PERMIT", "RESTRICT", "REVOKE", "CONTAIN"]
constraint_applied: bool = False
```

Without `constraint_applied` the ablation above is not countable from the
payload stream, and without `CONTAIN` the safety constraint is invisible to
C2/C3/C4.

## Reproduce

```bash
pip install scapy scikit-learn pandas joblib
python src/build_dataset.py        # PCAP → windows  (~6 min)
python src/evaluate.py             # identity AUC
python src/evaluate_spoofing.py    # spoofing detection + trade-off
python src/train_and_save.py       # persist baselines
```

## Decisions and their reasons

| Decision | Value | Why |
|---|---|---|
| Window | 30 s, 15 s stride | IoMT telemetry is periodic on the order of seconds; 30 s spans several cycles |
| Split | chronological 70/30 per capture | held-out windows are later traffic, not interleaved duplicates |
| Model features | 33 of the 45 | behavioural subset; all 45 still pass through to C2 unchanged |
| Algorithm | Isolation Forest | lower false restriction at full detection than OC-SVM |
| Threshold | 0.15 | 100% detection at the lowest false restriction available |

## Known limitations

1. **Devices are not clinical.** The WiFi profiling captures are cameras, a
   nurse-call pager and a baby monitor. Mapping to `ventilator_01` and similar
   is a modelling choice recorded in the topology config, not a property of the
   data. Nine genuine medical devices exist in the Bluetooth captures but are
   HCI-layer, so the 45 IP/TCP contract features do not apply to them.
2. **ARP spoofing is a proxy** for credential cloning. It is network-address
   impersonation, which is the closest labelled analogue available.
3. **Five derived features are best-effort.** `Magnitue`, `Radius`,
   `Covariance`, `Variance` and `Weight` follow the published CICIoT2023
   definitions; the exact formulas are not documented in `features.json`.
   Confirm with C2's owner before integration.
4. **16 spoofing windows** is a small evaluation set. `ARP_Spoofing_train.pcap`
   (80 MB) is available and should be added.
