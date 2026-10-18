from src import taxonomy
from src.contract_loader import DEVICES
from src.severity import SeverityEngine

LIFE_CRITICAL = [d for d, v in DEVICES.items() if v["life_critical"]]
NORMAL = [d for d, v in DEVICES.items() if not v["life_critical"]]


def test_contracts_have_both_kinds_of_device():
    assert LIFE_CRITICAL and NORMAL


def test_every_life_critical_device_forces_critical_for_every_attack():
    engine = SeverityEngine(mode="rule")
    for dev in LIFE_CRITICAL:
        for attack in taxonomy.ATTACK_SEVERITY:
            sev, flag = engine.decide(attack, 0.5, 0.0, dev)
            assert sev == "Critical" and flag is True, (dev, attack)


def test_override_wins_even_if_a_model_says_low():
    assert taxonomy.apply_override("Low", True) == ("Critical", True)


def test_non_critical_device_keeps_severity_and_flag_false():
    assert SeverityEngine(mode="rule").decide("Recon-Port_Scan", 0.9, 0.0, "bp_monitor_01") == ("Low", False)


def test_unknown_device_default_is_not_critical():
    assert taxonomy.device_is_life_critical("Nobody-99") is False


def test_flag_true_only_when_severity_critical():
    engine = SeverityEngine(mode="rule")
    for dev in DEVICES:
        for attack in taxonomy.ATTACK_SEVERITY:
            sev, flag = engine.decide(attack, 0.9, 0.0, dev)
            if flag:
                assert sev == "Critical"


def test_identity_event_override():
    engine = SeverityEngine(mode="rule")
    assert engine.decide_identity("ventilator_01") == ("Critical", True)
    sev, flag = engine.decide_identity("bp_monitor_01")
    assert flag is False
