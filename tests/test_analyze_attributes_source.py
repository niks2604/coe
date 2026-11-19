"""Source-side attribute behaviour on pbc_val, for the bottleneck checkpoints.

Reads analysis/*.npz and nothing else. Shares the synthetic dump factory with
test_analyze_temperature so the two analyses are exercised against identical files.
"""
import json

import numpy as np
import pytest

from scripts.analyze_attributes_source import AnalysisError, analyze_attributes
from test_analyze_temperature import ATTRIBUTES, CLASSES, analysis, write_npz  # noqa: F401


# --- source-side attributes ------------------------------------------------------------

def test_the_bottleneck_checkpoints_are_analysed(analysis):
    report = analyze_attributes(analysis())
    assert {r["run_id"] for r in report["checkpoints"]} == {"bottleneck"}
    assert len(report["checkpoints"]) == 3


def test_the_baseline_is_skipped_because_it_has_no_attribute_heads(analysis):
    report = analyze_attributes(analysis())
    assert "baseline" in json.dumps(report["excluded"])


def test_the_control_is_excluded_as_uninterpretable(analysis):
    """lambda is 0, so nothing supervised those heads; scoring them against WBCAtt
    labels would report the accuracy of a layer that never saw one."""
    report = analyze_attributes(analysis())
    assert "control" in json.dumps(report["excluded"])
    assert "uninterpretable" in json.dumps(report["excluded"]).lower()


def test_per_attribute_accuracy_is_reported_for_all_eleven(analysis):
    row = analyze_attributes(analysis())["checkpoints"][0]
    assert len(row["per_attribute"]) == 11
    for name, entry in row["per_attribute"].items():
        assert 0.0 <= entry["accuracy"] <= 1.0
        assert 0.0 <= entry["balanced_accuracy"] <= 1.0
        assert 0.0 <= entry["majority_rate"] <= 1.0


def test_accuracy_is_above_chance_on_the_synthetic_fixture(analysis):
    """The fixture puts mass on the true value, so a correct implementation must beat
    chance; an implementation that mis-split the heads would not."""
    row = analyze_attributes(analysis())["checkpoints"][0]
    assert all(e["accuracy"] > 0.4 for e in row["per_attribute"].values())


def test_the_per_class_mean_attribute_distribution_sums_to_one_per_head(analysis):
    row = analyze_attributes(analysis())["checkpoints"][0]
    for cls, heads in row["per_class_mean_attribute"].items():
        for name, distribution in heads.items():
            assert sum(distribution.values()) == pytest.approx(1.0, abs=1e-4)


def test_every_class_present_has_a_row(analysis):
    row = analyze_attributes(analysis())["checkpoints"][0]
    assert set(row["per_class_mean_attribute"]) <= set(CLASSES)
    assert row["per_class_mean_attribute"]


def test_the_distribution_is_keyed_by_value_name_not_index(analysis):
    row = analyze_attributes(analysis())["checkpoints"][0]
    heads = next(iter(row["per_class_mean_attribute"].values()))
    assert set(heads["attr_0"]) == set(ATTRIBUTES["attr_0"])


def test_the_attribute_analysis_refuses_a_non_pbc_val_file(analysis):
    out = analysis()
    write_npz(out / "bottleneck_seed0_raabin_train.npz", domain="raabin_train")
    with pytest.raises(AnalysisError, match="pbc_val|domain"):
        analyze_attributes(out)


def test_no_target_domain_appears_in_any_attribute_row(analysis):
    report = analyze_attributes(analysis())
    blob = json.dumps([report["checkpoints"], report["excluded"]])
    for name in ("raabin", "lisc", "pbc_test"):
        assert name not in blob


def test_a_file_without_attribute_arrays_is_not_scored(analysis, tmp_path):
    out = analysis(models=("bottleneck",), seeds=(0,))
    (out / "bottleneck_seed0_pbc_val.npz").unlink()
    write_npz(out / "bottleneck_seed0_pbc_val.npz", model="bottleneck",
              has_attributes=False)
    report = analyze_attributes(out)
    assert not report["checkpoints"]
