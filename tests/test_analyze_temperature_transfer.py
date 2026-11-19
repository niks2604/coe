"""Temperature fitted on pbc_val, applied to each Raabin domain.

The fit is still pbc_val only - the amendment permits retaining target predictions, not
fitting anything on them - and a test asserts the fit never sees a target logit. All
three model types, ECE before and after on every domain.

Post-hoc under the CLAUDE.md sec 3 amendment of 2026-11-19.

Synthetic dumps only.
"""
import json

import numpy as np
import pytest

from scripts.analyze_temperature_transfer import (
    AnalysisError,
    transfer_report,
)
from test_analyze_attribute_drift import CLASSES, TARGETS, write_dump


@pytest.fixture
def dumps(tmp_path):
    def build(models=("baseline", "bottleneck", "control"), seeds=(0, 1, 2),
              domains=("pbc_val", "pbc_test", *TARGETS)):
        out = tmp_path / "analysis"
        for model in models:
            for seed in seeds:
                for i, domain in enumerate(domains):
                    write_dump(out / f"{model}_seed{seed}_{domain}.npz", domain,
                               run_id=model, seed=seed,
                               weight=0.0 if model == "control" else 1.0,
                               has_attributes=model != "baseline",
                               rng_seed=seed * 10 + i)
        return out
    return build


def test_all_three_model_types_are_reported(dumps):
    report = transfer_report(dumps())
    assert {r["run_id"] for r in report["rows"]} == {"baseline", "bottleneck",
                                                     "control"}


def test_one_temperature_per_checkpoint_shared_across_domains(dumps):
    report = transfer_report(dumps())
    for (run_id, seed), rows in _grouped(report).items():
        assert len({r["temperature"] for r in rows}) == 1, (run_id, seed)


def _grouped(report):
    out = {}
    for row in report["rows"]:
        out.setdefault((row["run_id"], row["seed"]), []).append(row)
    return out


def test_ece_before_and_after_is_reported_for_every_target_domain(dumps):
    report = transfer_report(dumps())
    for domain in TARGETS:
        rows = [r for r in report["rows"] if r["domain"] == domain]
        assert len(rows) == 9
        assert all("ece_before" in r and "ece_after" in r for r in rows)


def test_the_fit_uses_pbc_val_only(dumps):
    """The amendment permits retaining target predictions, not fitting on them."""
    report = transfer_report(dumps())
    assert report["fitted_on"] == "pbc_val"
    assert all(r["fitted_on"] == "pbc_val" for r in report["rows"])


def test_a_missing_pbc_val_dump_is_fatal(dumps):
    out = dumps()
    (out / "bottleneck_seed0_pbc_val.npz").unlink()
    with pytest.raises(AnalysisError, match="pbc_val"):
        transfer_report(out)


def test_the_fit_never_sees_a_target_logit(dumps, monkeypatch):
    """Structural. The fitter is wrapped and every call is checked against the pbc_val
    logits, so a fit on anything else fails the test rather than the review."""
    import scripts.analyze_temperature_transfer as module

    out = dumps(models=("bottleneck",), seeds=(0,))
    with np.load(out / "bottleneck_seed0_pbc_val.npz", allow_pickle=False) as data:
        permitted = data["class_logits"].tobytes()

    real = module.fit_temperature
    seen = []

    def spy(logits, y_true, **kwargs):
        seen.append(np.asarray(logits, dtype=np.float32).tobytes())
        return real(logits, y_true, **kwargs)

    monkeypatch.setattr(module, "fit_temperature", spy)
    transfer_report(out)
    assert seen, "the fitter was never called"
    for blob in seen:
        assert blob == permitted, "the temperature was fitted on something but pbc_val"


def test_the_source_domain_is_reported_beside_the_targets(dumps):
    report = transfer_report(dumps())
    assert "pbc_test" in {r["domain"] for r in report["rows"]}


def test_a_temperature_that_helps_on_pbc_val_may_not_help_on_a_target(dumps):
    """Recorded as a result rather than hidden: that is the whole question."""
    report = transfer_report(dumps())
    assert any("ece_delta" in r for r in report["rows"])


def test_the_report_says_it_is_post_hoc(dumps):
    assert "post-hoc" in json.dumps(transfer_report(dumps())["scope"]).lower()


def test_rows_carry_provenance(dumps):
    row = transfer_report(dumps())["rows"][0]
    assert row["checkpoint_sha256"]


def test_a_temperature_on_the_search_bound_is_flagged(dumps):
    """A fit that ran out of interval is clipped, not fitted, and must not read like a
    fitted value. Found by running the script: synthetic logits pinned T at 10."""
    from scripts.analyze_temperature import at_search_bound

    assert at_search_bound(10.0) and at_search_bound(0.05)
    assert not at_search_bound(1.7)
    assert all("at_search_bound" in r for r in transfer_report(dumps())["rows"])
