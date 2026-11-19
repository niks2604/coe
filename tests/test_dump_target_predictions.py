"""Per-image dumps on pbc_test and the three Raabin domains.

Permitted by the scoped CLAUDE.md sec 3 amendment of 2026-11-19, for the nine fixed
headline checkpoints and nothing else. What may be retained is enumerated there:
class logits, attribute probabilities, image ids, true class. Pixels and every
pixel-derived statistic stay forbidden, so this file asserts the written fields against
an allowlist rather than trusting the writer.

The dumps are recorded in analysis_ledger.jsonl, which is a separate file conferring no
permission to evaluate anything. target_eval_ledger.jsonl and eval_records/ are never
opened for writing, and tests assert both are untouched.

Synthetic trees only.
"""
import json

import numpy as np
import pytest

pytest.importorskip("torch")

from scripts.dump_target_predictions import (
    ALLOWED_ARRAYS,
    ALLOWED_META,
    DOMAINS,
    DumpError,
    dump_checkpoint,
    output_path,
)

# Exact field names that would describe the pixels rather than the model's reading of
# them. Matched exactly, not as substrings: "cell_size" is a WBCAtt attribute and
# "model_type" and "class_logits_mode" are provenance, and a substring scan would
# reject all three. The two allowlists are what actually enforce this.
FORBIDDEN_FIELDS = ("width", "height", "image_size", "size", "mode", "image_mode",
                    "channel_mean", "channel_std", "mean_pixel", "std_pixel",
                    "file_hash", "image_sha256", "md5", "n_bytes", "exif",
                    "histogram", "pixels", "raw")


@pytest.fixture
def checkpoint(experiment_config, tiny_raw_loadable, mapping, label_space, tmp_path):
    from conftest import DELETE
    from src.config import load_config, set_seed
    from src.train import save_checkpoint

    def write(model_type="bottleneck", role="headline", **extra):
        overrides = {"data.image_size": 32, "train.batch_size": 2, "train.epochs": 1,
                     "run_id": model_type, "model.type": model_type, **extra}
        if model_type == "baseline":
            overrides["bottleneck"] = DELETE
            overrides["eval.attribute_modes"] = DELETE
        elif model_type == "control":
            overrides["bottleneck.attribute_loss_weight"] = 0.0
        cfg = load_config(experiment_config(**overrides))
        path = tmp_path / f"{model_type}_{role}_{len(list(tmp_path.glob('*.pt')))}.pt"
        set_seed(cfg.seed)
        save_checkpoint(path, cfg, mapping, label_space,
                        history=[{"epoch": 0, "val_macro_f1": 0.5}], role=role)
        return cfg, path

    return write


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def meta_of(path):
    return json.loads(str(load(path)["meta"]))


def dump_all(path, out, ledger, **kwargs):
    return [dump_checkpoint(path, out, domain=d, ledger_path=ledger, device="cpu",
                            **kwargs) for d in DOMAINS]


# --- what may be written, and nothing else ----------------------------------------------

@pytest.mark.parametrize("domain", DOMAINS)
def test_only_allowed_arrays_are_written(checkpoint, tmp_path, domain):
    cfg, path = checkpoint()
    out = dump_checkpoint(path, tmp_path / "analysis", domain=domain,
                          ledger_path=tmp_path / "l.jsonl", device="cpu")
    assert set(load(out)) <= set(ALLOWED_ARRAYS), set(load(out)) - set(ALLOWED_ARRAYS)


@pytest.mark.parametrize("domain", DOMAINS)
def test_no_pixel_derived_field_is_written(checkpoint, tmp_path, domain):
    """The amendment permits the model's reading of the image, never the image.

    Checked on the field NAMES against an allowlist, not on the serialised blob: the
    WBCAtt vocabulary legitimately contains "cell_size", which is a hematologist's
    reading of the cell and not a statistic of the image file.
    """
    cfg, path = checkpoint()
    out = dump_checkpoint(path, tmp_path / "analysis", domain=domain,
                          ledger_path=tmp_path / "l.jsonl", device="cpu")
    meta = meta_of(out)
    assert set(meta) <= set(ALLOWED_META), set(meta) - set(ALLOWED_META)
    for key in list(meta) + list(load(out)):
        assert key.lower() not in FORBIDDEN_FIELDS, key


def test_an_unlisted_field_is_refused_at_write_time(checkpoint, tmp_path, monkeypatch):
    """The allowlist is enforced, not merely asserted by this file."""
    import scripts.dump_target_predictions as module

    monkeypatch.setattr(module, "ALLOWED_META",
                        tuple(k for k in module.ALLOWED_META if k != "domain"))
    cfg, path = checkpoint()
    with pytest.raises(DumpError, match="metadata"):
        dump_checkpoint(path, tmp_path / "analysis", domain="raabin_testA",
                        ledger_path=tmp_path / "l.jsonl", device="cpu")


def test_a_target_dump_carries_no_attribute_labels(checkpoint, tmp_path):
    """Raabin has no WBCAtt labels; a column of them could only be invented."""
    cfg, path = checkpoint()
    out = dump_checkpoint(path, tmp_path / "analysis", domain="raabin_testA",
                          ledger_path=tmp_path / "l.jsonl", device="cpu")
    assert "attribute_true" not in load(out)


def test_the_source_dump_does_carry_attribute_labels(checkpoint, tmp_path):
    cfg, path = checkpoint()
    out = dump_checkpoint(path, tmp_path / "analysis", domain="pbc_test",
                          ledger_path=tmp_path / "l.jsonl", device="cpu")
    assert load(out)["attribute_true"].shape[1] == 11


# --- the four domains, and no others -------------------------------------------------------

def test_the_four_domains_are_exactly_the_evaluated_ones(checkpoint, tmp_path):
    assert DOMAINS == ("pbc_test", "raabin_train", "raabin_testA", "raabin_testB")


@pytest.mark.parametrize("domain", ["pbc_val", "pbc_train", "lisc", "nonsense"])
def test_any_other_domain_is_refused(checkpoint, tmp_path, domain):
    """pbc_val has its own script; lisc is out of scope; pbc_train is training data.
    The amendment names four domains and this script covers those four."""
    cfg, path = checkpoint()
    with pytest.raises(DumpError, match="domain"):
        dump_checkpoint(path, tmp_path / "analysis", domain=domain,
                        ledger_path=tmp_path / "l.jsonl", device="cpu")


# --- the same gates eval.py applies -----------------------------------------------------------

def test_a_non_headline_checkpoint_is_refused(checkpoint, tmp_path):
    """The amendment is scoped to the nine headline checkpoints."""
    cfg, path = checkpoint(role="informational")
    with pytest.raises(Exception, match="headline|informational"):
        dump_checkpoint(path, tmp_path / "analysis", domain="raabin_testA",
                        ledger_path=tmp_path / "l.jsonl", device="cpu")


def test_the_buffers_are_unchanged(checkpoint, tmp_path):
    cfg, path = checkpoint()
    meta = meta_of(dump_checkpoint(path, tmp_path / "analysis", domain="raabin_train",
                                   ledger_path=tmp_path / "l.jsonl", device="cpu"))
    assert meta["buffer_sha256_before"] == meta["buffer_sha256_after"]


def test_eval_mode_and_no_grad_hold_throughout(checkpoint, tmp_path):
    cfg, path = checkpoint()
    meta = meta_of(dump_checkpoint(path, tmp_path / "analysis", domain="raabin_testB",
                                   ledger_path=tmp_path / "l.jsonl", device="cpu"))
    assert meta["model_training_mode_seen"] is False
    assert meta["grad_enabled_seen"] is False


def test_provenance_is_recorded(checkpoint, tmp_path):
    from src.ledger import checkpoint_hash

    cfg, path = checkpoint()
    meta = meta_of(dump_checkpoint(path, tmp_path / "analysis", domain="pbc_test",
                                   ledger_path=tmp_path / "l.jsonl", device="cpu"))
    assert meta["checkpoint_sha256"] == checkpoint_hash(path)
    assert "commit" in meta["git"] and "dirty" in meta["git"]


# --- overwriting ---------------------------------------------------------------------------------

def test_an_existing_dump_is_not_overwritten(checkpoint, tmp_path):
    cfg, path = checkpoint()
    args = dict(domain="raabin_testA", ledger_path=tmp_path / "l.jsonl", device="cpu")
    out = dump_checkpoint(path, tmp_path / "analysis", **args)
    before = out.read_bytes()
    with pytest.raises(DumpError, match="exists|overwrite"):
        dump_checkpoint(path, tmp_path / "analysis", **args)
    assert out.read_bytes() == before


def test_a_refused_overwrite_writes_no_second_ledger_line(checkpoint, tmp_path):
    cfg, path = checkpoint()
    ledger = tmp_path / "l.jsonl"
    args = dict(domain="raabin_testA", ledger_path=ledger, device="cpu")
    dump_checkpoint(path, tmp_path / "analysis", **args)
    with pytest.raises(DumpError):
        dump_checkpoint(path, tmp_path / "analysis", **args)
    assert len(ledger.read_text().strip().splitlines()) == 1


def test_each_domain_lands_in_its_own_file(checkpoint, tmp_path):
    cfg, path = checkpoint()
    written = dump_all(path, tmp_path / "analysis", tmp_path / "l.jsonl")
    assert len({p.name for p in written}) == len(DOMAINS)
    assert output_path(cfg, tmp_path, "raabin_testB").name.endswith("_raabin_testB.npz")


# --- the analysis ledger -----------------------------------------------------------------------

def test_one_ledger_line_per_checkpoint_and_domain(checkpoint, tmp_path):
    cfg, path = checkpoint()
    ledger = tmp_path / "l.jsonl"
    dump_all(path, tmp_path / "analysis", ledger)
    rows = [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()]
    assert len(rows) == len(DOMAINS)
    assert {r["domain"] for r in rows} == set(DOMAINS)


def test_the_ledger_line_carries_the_provenance_and_the_amendment(checkpoint, tmp_path):
    from src.ledger import checkpoint_hash

    cfg, path = checkpoint()
    ledger = tmp_path / "l.jsonl"
    dump_checkpoint(path, tmp_path / "analysis", domain="raabin_train",
                    ledger_path=ledger, device="cpu")
    row = json.loads(ledger.read_text().splitlines()[0])
    for key in ("timestamp", "checkpoint_sha256", "domain", "run_id", "seed",
                "git", "output", "n", "amendment"):
        assert key in row, key
    assert row["checkpoint_sha256"] == checkpoint_hash(path)
    assert "2026-11-19" in row["amendment"]


def test_the_ledger_is_append_only(checkpoint, tmp_path):
    cfg, path = checkpoint()
    ledger = tmp_path / "l.jsonl"
    ledger.write_text('{"pre": "existing"}\n')
    dump_checkpoint(path, tmp_path / "analysis", domain="pbc_test",
                    ledger_path=ledger, device="cpu")
    lines = ledger.read_text().splitlines()
    assert json.loads(lines[0]) == {"pre": "existing"}
    assert len(lines) == 2


# --- the primary ledger and the records are never touched -------------------------------------

def test_the_primary_ledger_and_eval_records_are_unchanged(checkpoint, tmp_path):
    """The one thing this script must not do. The amendment confers no permission to
    evaluate anything, so nothing here may add to the evaluation's audit trail."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    primary = repo / "target_eval_ledger.jsonl"
    records = sorted((repo / "eval_records").glob("*.json"))
    before = (primary.read_bytes(),
              [(p.name, p.read_bytes()) for p in records])

    cfg, path = checkpoint()
    dump_all(path, tmp_path / "analysis", tmp_path / "l.jsonl")

    assert primary.read_bytes() == before[0], "target_eval_ledger.jsonl changed"
    assert [(p.name, p.read_bytes())
            for p in sorted((repo / "eval_records").glob("*.json"))] == before[1]


def test_the_script_names_its_own_ledger_not_the_primary_one(checkpoint, tmp_path):
    import inspect

    import scripts.dump_target_predictions as module

    source = inspect.getsource(module)
    assert "target_eval_ledger" not in source.replace(
        "never touches target_eval_ledger.jsonl", "")


# --- contents ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("domain", DOMAINS)
def test_the_arrays_line_up(checkpoint, tmp_path, domain):
    cfg, path = checkpoint()
    data = load(dump_checkpoint(path, tmp_path / "analysis", domain=domain,
                                ledger_path=tmp_path / "l.jsonl", device="cpu"))
    n = len(data["image_ids"])
    assert n > 0
    assert data["y_true"].shape == (n,)
    assert data["class_logits"].shape == (n, 5)
    assert data["attribute_probs"].shape[0] == n


def test_the_image_ids_are_unique_within_a_domain(checkpoint, tmp_path):
    cfg, path = checkpoint()
    for domain in DOMAINS:
        ids = load(dump_checkpoint(path, tmp_path / "analysis", domain=domain,
                                   ledger_path=tmp_path / "l.jsonl",
                                   device="cpu"))["image_ids"]
        assert len(set(ids.tolist())) == len(ids), domain


def test_the_baseline_writes_no_attribute_probabilities(checkpoint, tmp_path):
    cfg, path = checkpoint("baseline")
    data = load(dump_checkpoint(path, tmp_path / "analysis", domain="raabin_testA",
                                ledger_path=tmp_path / "l.jsonl", device="cpu"))
    assert "attribute_probs" not in data


def test_the_control_is_labelled_uninterpretable(checkpoint, tmp_path):
    cfg, path = checkpoint("control")
    meta = meta_of(dump_checkpoint(path, tmp_path / "analysis", domain="raabin_train",
                                   ledger_path=tmp_path / "l.jsonl", device="cpu"))
    assert meta["attributes_interpretable"] is False


def test_the_domain_and_mode_are_recorded(checkpoint, tmp_path):
    cfg, path = checkpoint()
    meta = meta_of(dump_checkpoint(path, tmp_path / "analysis", domain="raabin_testB",
                                   ledger_path=tmp_path / "l.jsonl", device="cpu"))
    assert meta["domain"] == "raabin_testB"
    assert meta["class_logits_mode"] == "soft"
