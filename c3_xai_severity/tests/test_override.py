import pytest

from src import config, taxonomy
from src.severity import SeverityEngine


def test_every_life_critical_type_forces_critical_for_every_attack():
    engine = SeverityEngine(mode="rule")
    life_critical_ids = [d for d, t in config.DEVICE_REGISTRY.items() if t in config.LIFE_CRITICAL_TYPES]
    assert life_critical_ids
    for dev in life_critical_ids:
        for attack in taxonomy.ATTACK_SEVERITY:
            sev, flag = engine.decide(attack, 0.5, 0.0, dev)
            assert sev == "Critical" and flag is True, (dev, attack)


def test_override_wins_even_if_a_model_says_low():
    assert taxonomy.apply_override("Low", True) == ("Critical", True)


def test_non_critical_device_keeps_model_severity_and_flag_false():
    sev, flag = SeverityEngine(mode="rule").decide("Recon-Port_Scan", 0.9, 0.0, "Wearable-07")
    assert (sev, flag) == ("Low", False)


def test_unknown_device_default_is_not_critical():
    assert taxonomy.device_is_life_critical("Nobody-99") is False


def test_flag_true_only_when_severity_critical():
    engine = SeverityEngine(mode="rule")
    for dev in config.DEVICE_REGISTRY:
        for attack in taxonomy.ATTACK_SEVERITY:
            sev, flag = engine.decide(attack, 0.9, 0.0, dev)
            if flag:
                assert sev == "Critical"
