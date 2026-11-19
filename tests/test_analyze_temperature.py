"""Temperature scaling fitted on pbc_val, and the synthetic dumps both analyses use.

It reads analysis/*.npz and nothing else - no checkpoint, no image, no ledger, no
eval record. The temperature is fitted on pbc_val and is never applied to a target
domain; there is no target domain in these files to apply it to, and a test asserts
the analyses refuse a file that claims one.

Synthetic .npz only.
"""
import json

import numpy as np
import pytest

from scripts.analyze_temperature import (
    AnalysisError,
    ece_of,
    fit_temperature,
    temperature_report,
)

CLASSES = ["neutrophil", "lymphocyte", "monocyte", "eosinophil", "basophil"]
ATTRIBUTES = {f"attr_{i}": [f"v{j}" for j in range(2 + (i % 2))] for i in range(11)}


def write_npz(path, model="bottleneck", n=200, weight=1.0, scale=3.0, seed=0,
              domain="pbc_val", has_attributes=None, **meta_overrides):
    rng = np.random.default_rng(seed)
    has_attributes = model != "baseline" if has_attributes is None else has_attributes
    y_true = rng.integers(0, len(CLASSES), size=n).astype(np.int16)

    # Logits that are right most of the time and over-confident, so a temperature
    # above 1 genuinely helps and the fit has something to find.
    logits = rng.normal(0.0, 1.0, size=(n, len(CLASSES))).astype(np.float32)
    logits[np.arange(n), y_true] += scale

    record = {"image_ids": np.array([f"IMG_{i}.jpg" for i in range(n)]),
              "y_true": y_true, "class_logits": logits}

    offsets = [0]
    for values in ATTRIBUTES.values():
        offsets.append(offsets[-1] + len(values))
    if has_attributes:
        attribute_true = np.stack(
            [rng.integers(0, len(v), size=n) for v in ATTRIBUTES.values()],
            axis=1).astype(np.int16)
        probs = np.zeros((n, offsets[-1]), dtype=np.float32)
        for head, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
            block = rng.random((n, b - a)).astype(np.float32)
            # Push mass onto the true value so accuracy is above chance.
            block[np.arange(n), attribute_true[:, head]] += 2.0
            probs[:, a:b] = block / block.sum(axis=1, keepdims=True)
        record["attribute_true"] = attribute_true
        record["attribute_probs"] = probs
        record["attribute_group_offsets"] = np.array(offsets, dtype=np.int32)

    meta = {"domain": domain, "run_id": model, "seed": seed, "model_type": model,
            "checkpoint_sha256": f"{model}{seed}".ljust(64, "0"),
            "git": {"commit": "abc1234", "dirty": False},
            "classes": CLASSES, "attribute_names": list(ATTRIBUTES),
            "attribute_values": {k: list(v) for k, v in ATTRIBUTES.items()},
            "has_attributes": has_attributes, "attribute_loss_weight": weight,
            "attributes_interpretable": bool(has_attributes and weight),
            "attributes_note": "" if weight else "UNINTERPRETABLE: unsupervised heads",
            "class_logits_mode": "soft" if has_attributes else "none",
            "selection_role": "headline", "n": n}
    meta.update(meta_overrides)
    record["meta"] = np.array(json.dumps(meta))

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix("") if path.suffix == ".npz" else path,
                        **record)
    return path


@pytest.fixture
def analysis(tmp_path):
    def build(models=("baseline", "bottleneck", "control"), seeds=(0, 1, 2), **kwargs):
        out = tmp_path / "analysis"
        for seed in seeds:
            for model in models:
                write_npz(out / f"{model}_seed{seed}_pbc_val.npz", model=model,
                          seed=seed, weight=0.0 if model == "control" else 1.0,
                          **kwargs)
        return out

    return build


# --- temperature ----------------------------------------------------------------------

def test_a_temperature_is_fitted_per_checkpoint(analysis):
    report = temperature_report(analysis())
    assert len(report["checkpoints"]) == 9
    assert all(r["temperature"] > 0 for r in report["checkpoints"])


def test_the_fit_lowers_ece_on_overconfident_logits(analysis):
    for row in temperature_report(analysis())["checkpoints"]:
        assert row["ece_after"] <= row["ece_before"] + 1e-6, row


def test_ece_before_and_after_are_both_reported(analysis):
    row = temperature_report(analysis())["checkpoints"][0]
    assert "ece_before" in row and "ece_after" in row


def test_a_temperature_of_one_leaves_ece_unchanged(analysis):
    out = analysis()
    path = sorted(out.glob("*.npz"))[0]
    with np.load(path, allow_pickle=False) as data:
        logits, y = data["class_logits"], data["y_true"]
    assert ece_of(logits, y) == pytest.approx(ece_of(logits / 1.0, y))


def test_the_fit_uses_pbc_val_and_refuses_anything_else(analysis, tmp_path):
    """The temperature is fitted on pbc_val only. A file claiming another domain is a
    file that should not exist, so it is refused rather than quietly included."""
    out = analysis()
    write_npz(out / "bottleneck_seed0_raabin_testA.npz", domain="raabin_testA")
    with pytest.raises(AnalysisError, match="pbc_val|domain"):
        temperature_report(out)


def test_no_target_domain_appears_in_any_reported_row(analysis):
    """On the rows, not the prose: the scope sentence names pbc_test and the targets
    precisely to say the temperature is not carried to them."""
    report = temperature_report(analysis())
    blob = json.dumps(report["checkpoints"])
    for name in ("raabin", "lisc", "pbc_test"):
        assert name not in blob


def test_the_report_says_the_temperature_is_not_applied_elsewhere(analysis):
    assert "not applied" in temperature_report(analysis())["scope"].lower()


def test_provenance_travels_with_each_row(analysis):
    row = temperature_report(analysis())["checkpoints"][0]
    assert row["checkpoint_sha256"] and row["git"]["commit"] == "abc1234"


def test_an_empty_directory_is_refused(tmp_path):
    (tmp_path / "analysis").mkdir()
    with pytest.raises(AnalysisError, match="no .*npz|empty|found"):
        temperature_report(tmp_path / "analysis")


def test_fit_temperature_scales_with_the_logits():
    """Scaling the logits by k scales the fitted temperature by k.

    The property the fitter must have. Asserting that logits*k give exactly k would
    assume the unscaled logits are already NLL-optimal at T=1, which these are not -
    they are under-confident by construction, so their own optimum is below 1.
    """
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.integers(0, 5, size=n)
    logits = rng.normal(0, 1, size=(n, 5))
    logits[np.arange(n), y] += 3.0
    base = fit_temperature(logits, y)
    assert fit_temperature(logits * 2.5, y) == pytest.approx(2.5 * base, rel=0.05)


def test_fit_temperature_finds_a_lower_nll_than_leaving_it_alone():
    from scripts.analyze_temperature import _nll

    rng = np.random.default_rng(1)
    n = 3000
    y = rng.integers(0, 5, size=n)
    logits = rng.normal(0, 1, size=(n, 5))
    logits[np.arange(n), y] += 2.0
    t = fit_temperature(logits, y)
    assert _nll(logits, y, t) <= _nll(logits, y, 1.0) + 1e-9
