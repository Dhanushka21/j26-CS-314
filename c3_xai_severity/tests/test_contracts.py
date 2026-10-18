"""Keep Component 3 in step with the group's contracts folder."""
from src import taxonomy
from src.contract_loader import C3ToC4, CLASSES, FEATURES, N_FEATURES, sample
from src.schemas import Alert


def test_45_features_and_19_classes():
    assert N_FEATURES == 45 == len(FEATURES)
    assert len(CLASSES) == 19


def test_taxonomy_covers_exactly_the_18_attack_classes():
    assert set(taxonomy.ATTACK_SEVERITY) == set(CLASSES) - {"Benign"}


def test_alert_equals_c3toc4_without_recommended_action():
    # The group agreed to drop recommended_action (Component 4 decides). If the contract drops it too, this still passes.
    assert set(Alert.model_fields) == set(C3ToC4.model_fields) - {"recommended_action"}


def test_group_sample_alert_matches_our_fields():
    s = sample("c3_to_c4")
    assert set(s) - {"recommended_action"} == set(Alert.model_fields)
    Alert(**{k: v for k, v in s.items() if k != "recommended_action"})
