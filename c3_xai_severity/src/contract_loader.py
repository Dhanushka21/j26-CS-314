"""Loads the group's shared contracts (repo-root `contracts/` folder) so we never copy them.

Contracts are the single source of truth: classes.json, features.json, devices.json and models.py.
Lookup order: env C3_CONTRACTS_DIR, then ../contracts (the repo layout), then ./contracts.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent


def _find() -> Path:
    candidates = []
    if os.environ.get("C3_CONTRACTS_DIR"):
        candidates.append(Path(os.environ["C3_CONTRACTS_DIR"]))
    candidates += [_PROJECT.parent / "contracts", _PROJECT / "contracts"]
    for c in candidates:
        if (c / "models.py").exists():
            return c
    raise FileNotFoundError(
        "Could not find the shared 'contracts' folder. Keep c3_xai_severity inside the group repo "
        "(next to contracts/) or set C3_CONTRACTS_DIR. Looked in: " + ", ".join(str(c) for c in candidates)
    )


CONTRACTS_DIR = _find()

_spec = importlib.util.spec_from_file_location("iomt_contracts_models", CONTRACTS_DIR / "models.py")
models = importlib.util.module_from_spec(_spec)
sys.modules["iomt_contracts_models"] = models      # pydantic needs the module registered
_spec.loader.exec_module(models)

CLASSES = list(models.CLASSES)
FEATURES = list(models.FEATURES)
DEVICES = dict(models.DEVICES)
N_FEATURES = models.N_FEATURES

C2ToC3 = models.C2ToC3
C1ToC3 = models.C1ToC3
C3ToC4 = models.C3ToC4
ShapFeature = models.ShapFeature
Severity = models.Severity


def sample(name: str) -> dict:
    """Read one of the group's example payloads, e.g. sample('c2_to_c3')."""
    return json.loads((CONTRACTS_DIR / "samples" / f"{name}.json").read_text())
