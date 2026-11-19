"""Attribute drift: how far the predicted attribute distributions move, source to target.

Bottleneck checkpoints only. The control's heads are unsupervised at lambda 0, so they
are not attribute predictions and a distance between them would measure nothing about
morphology. The baseline has no heads at all.

Jensen-Shannon distance, per class and per attribute, between the mean predicted
distribution on pbc_test and on each Raabin domain. Every attribute and every seed is
reported, ranked, with nothing filtered out: a filter applied after the numbers are
seen is a choice made by the numbers.

Post-hoc under the CLAUDE.md sec 3 amendment of 2026-11-19. It explains the sec 6
verdict and cannot revise it.

Synthetic dumps only.
"""
import json

import numpy as np
import pytest

from scripts.analyze_attribute_drift import (
    AnalysisError,
    drift_report,
    jensen_shannon_distance,
)

CLASSES = ["neutrophil", "lymphocyte", "monocyte", "eosinophil", "basophil"]
ATTRIBUTES = {f"attr_{i}": [f"v{j}" for j in range(2 + (i % 2))] for i in range(11)}
TARGETS = ("raabin_train", "raabin_testA", "raabin_testB")


def write_dump(path, domain, run_id="bottleneck", seed=0, n=120, shift=0.0,
               classes_present=None, weight=1.0, has_attributes=True, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    classes_present = list(range(len(CLASSES))) if classes_present is None \
        else classes_present
    y = rng.choice(classes_present, size=n).astype(np.int16)

    offsets = [0]
    for values in ATTRIBUTES.values():
        offsets.append(offsets[-1] + len(values))

    record = {"image_ids": np.array([f"{domain}/{i}.jpg" for i in range(n)]),
              "y_true": y,
              "class_logits": rng.normal(0, 1, size=(n, 5)).astype(np.float32)}
    if has_attributes:
        probs = np.zeros((n, offsets[-1]), dtype=np.float32)
        for head, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
            block = rng.random((n, b - a)).astype(np.float32) + 1.0
            block[:, 0] += shift          # a controllable, known drift
            probs[:, a:b] = block / block.sum(axis=1, keepdims=True)
        record["attribute_probs"] = probs
        record["attribute_group_offsets"] = np.array(offsets, dtype=np.int32)

    meta = {"domain": domain, "run_id": run_id, "seed": seed, "model_type": run_id,
            "classes": CLASSES, "attribute_names": list(ATTRIBUTES),
            "attribute_values": {k: list(v) for k, v in ATTRIBUTES.items()},
            "has_attributes": has_attributes, "attribute_loss_weight": weight,
            "attributes_interpretable": bool(has_attributes and weight),
            "class_logits_mode": "soft", "n": n,
            "checkpoint_sha256": f"{run_id}{seed}".ljust(64, "0"),
            "git": {"commit": "abc1234", "dirty": False}}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **record)
    return path


@pytest.fixture
def dumps(tmp_path):
    def build(models=("bottleneck",), seeds=(0, 1, 2), shift=1.0, **kwargs):
        out = tmp_path / "analysis"
        for model in models:
            weight = 0.0 if model == "control" else 1.0
            for seed in seeds:
                write_dump(out / f"{model}_seed{seed}_pbc_test.npz", "pbc_test",
                           run_id=model, seed=seed, shift=0.0, weight=weight,
                           has_attributes=model != "baseline", rng_seed=seed, **kwargs)
                for domain in TARGETS:
                    present = [0, 1] if domain == "raabin_testB" else None
                    write_dump(out / f"{model}_seed{seed}_{domain}.npz", domain,
                               run_id=model, seed=seed, shift=shift, weight=weight,
                               classes_present=present,
                               has_attributes=model != "baseline",
                               rng_seed=seed + 100, **kwargs)
        return out
    return build


# --- the distance itself ------------------------------------------------------------

def test_identical_distributions_are_zero_apart():
    p = np.array([0.2, 0.3, 0.5])
    assert jensen_shannon_distance(p, p) == pytest.approx(0.0, abs=1e-12)


def test_disjoint_distributions_are_one_apart():
    """The distance - not the divergence - is bounded at 1 for disjoint support."""
    assert jensen_shannon_distance(np.array([1.0, 0.0]),
                                   np.array([0.0, 1.0])) == pytest.approx(1.0, abs=1e-9)


def test_the_distance_is_symmetric():
    p, q = np.array([0.7, 0.2, 0.1]), np.array([0.1, 0.1, 0.8])
    assert jensen_shannon_distance(p, q) == pytest.approx(
        jensen_shannon_distance(q, p))


def test_the_distance_is_bounded_in_zero_one():
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = rng.random(4); p /= p.sum()
        q = rng.random(4); q /= q.sum()
        assert 0.0 <= jensen_shannon_distance(p, q) <= 1.0 + 1e-12


def test_a_bigger_shift_gives_a_bigger_distance():
    near = jensen_shannon_distance(np.array([0.5, 0.5]), np.array([0.6, 0.4]))
    far = jensen_shannon_distance(np.array([0.5, 0.5]), np.array([0.95, 0.05]))
    assert far > near


# --- who is included -------------------------------------------------------------------

def test_only_the_bottleneck_is_analysed(dumps):
    report = drift_report(dumps(models=("baseline", "bottleneck", "control")))
    assert {r["run_id"] for r in report["rows"]} == {"bottleneck"}


def test_the_control_is_excluded_as_uninterpretable(dumps):
    report = drift_report(dumps(models=("bottleneck", "control")))
    blob = json.dumps(report["excluded"]).lower()
    assert "control" in blob and "uninterpretable" in blob


def test_the_baseline_is_excluded_for_having_no_heads(dumps):
    report = drift_report(dumps(models=("baseline", "bottleneck")))
    assert "baseline" in json.dumps(report["excluded"])


def test_a_missing_pbc_test_dump_is_fatal(dumps):
    """There is no drift without the source side to measure it from."""
    out = dumps()
    (out / "bottleneck_seed0_pbc_test.npz").unlink()
    with pytest.raises(AnalysisError, match="pbc_test|source"):
        drift_report(out)


# --- coverage: nothing is filtered ---------------------------------------------------------

def test_every_attribute_and_seed_and_domain_appears(dumps):
    report = drift_report(dumps())
    assert {r["attribute"] for r in report["rows"]} == set(ATTRIBUTES)
    assert {r["seed"] for r in report["rows"]} == {0, 1, 2}
    assert {r["domain"] for r in report["rows"]} == set(TARGETS)


def test_the_rows_are_ranked_by_distance(dumps):
    rows = drift_report(dumps())["rows"]
    distances = [r["jensen_shannon"] for r in rows]
    assert distances == sorted(distances, reverse=True)


def test_nothing_is_filtered_out(dumps):
    """11 attributes x 3 seeds x 3 domains x classes present in both sides."""
    report = drift_report(dumps())
    expected = sum(
        len(ATTRIBUTES) * (2 if domain == "raabin_testB" else len(CLASSES))
        for domain in TARGETS) * 3
    assert len(report["rows"]) == expected
    assert report["filtered"] == 0


def test_a_class_absent_from_a_target_is_reported_not_dropped(dumps):
    """raabin_testB holds two of the five. The other three are not measurable there,
    which is a null and a listed reason, not a silent omission."""
    report = drift_report(dumps())
    absent = [r for r in report["unmeasurable"] if r["domain"] == "raabin_testB"]
    assert {r["cls"] for r in absent} == {"monocyte", "eosinophil", "basophil"}
    assert all(r["jensen_shannon"] is None for r in absent)


def test_no_row_carries_a_zero_for_an_absent_class(dumps):
    report = drift_report(dumps())
    for row in report["rows"]:
        assert row["jensen_shannon"] is not None


# --- the numbers mean what they say ------------------------------------------------------------

def test_a_shifted_target_drifts_more_than_an_unshifted_one(tmp_path):
    out = tmp_path / "analysis"
    write_dump(out / "bottleneck_seed0_pbc_test.npz", "pbc_test", shift=0.0, rng_seed=1)
    write_dump(out / "bottleneck_seed0_raabin_train.npz", "raabin_train", shift=0.0,
               rng_seed=2)
    write_dump(out / "bottleneck_seed0_raabin_testA.npz", "raabin_testA", shift=6.0,
               rng_seed=3)
    rows = drift_report(out)["rows"]
    quiet = np.mean([r["jensen_shannon"] for r in rows
                     if r["domain"] == "raabin_train"])
    loud = np.mean([r["jensen_shannon"] for r in rows
                    if r["domain"] == "raabin_testA"])
    assert loud > quiet


def test_each_row_carries_both_mean_distributions(dumps):
    row = drift_report(dumps())["rows"][0]
    assert set(row["source_mean"]) == set(row["target_mean"])
    assert sum(row["source_mean"].values()) == pytest.approx(1.0, abs=1e-4)
    assert sum(row["target_mean"].values()) == pytest.approx(1.0, abs=1e-4)


def test_the_distributions_are_keyed_by_value_name(dumps):
    row = next(r for r in drift_report(dumps())["rows"] if r["attribute"] == "attr_0")
    assert set(row["source_mean"]) == set(ATTRIBUTES["attr_0"])


def test_the_report_says_it_is_post_hoc(dumps):
    assert "post-hoc" in json.dumps(drift_report(dumps())["scope"]).lower()


def test_provenance_travels_with_each_row(dumps):
    row = drift_report(dumps())["rows"][0]
    assert row["checkpoint_sha256"] and row["source_domain"] == "pbc_test"
