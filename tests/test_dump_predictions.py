"""Per-image predictions on pbc_val, for post-hoc source-side analysis.

pbc_val is the tuning split, read constantly by design and never ledgered, so dumping
it spends nothing. Every other domain is refused outright: pbc_test is held out and
ledgered, and CLAUDE.md sec 3 forbids retaining any per-image record derived from
target pixels. The refusal is structural rather than a convention - a domain argument
that reached a loader would be the whole problem.

Synthetic trees only.
"""
import json

import numpy as np
import pytest

pytest.importorskip("torch")

from scripts.dump_predictions import (
    PBC_VAL,
    DumpError,
    dump_checkpoint,
    output_path,
)


@pytest.fixture
def checkpoint(experiment_config, tiny_raw_loadable, mapping, label_space, tmp_path):
    """write(model_type, role) -> (cfg, path), saved as src.train saves one."""
    from conftest import DELETE
    from src.config import load_config, set_seed
    from src.train import save_checkpoint

    def write(model_type="bottleneck", role="headline", **extra):
        overrides = {"data.image_size": 32, "train.batch_size": 2, "train.epochs": 1,
                     # As the nine real configs do: run_id names the model, so two
                     # model types never resolve to one output file.
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


# --- pbc_val only ---------------------------------------------------------------------

def test_the_default_domain_is_pbc_val(checkpoint, tmp_path):
    cfg, path = checkpoint()
    out = dump_checkpoint(path, tmp_path / "analysis")
    assert meta_of(out)["domain"] == PBC_VAL


@pytest.mark.parametrize("domain", ["pbc_test", "raabin_train", "raabin_testA",
                                    "raabin_testB", "lisc", "pbc_train"])
def test_every_other_domain_is_refused(checkpoint, tmp_path, domain):
    """Raised, not skipped. A skip would leave a file that silently covers less than
    its name says."""
    cfg, path = checkpoint()
    with pytest.raises(DumpError, match="pbc_val"):
        dump_checkpoint(path, tmp_path / "analysis", domain=domain)


def test_a_target_domain_is_refused_before_any_loader_is_built(checkpoint, tmp_path,
                                                                monkeypatch):
    """The refusal must come before anything can touch target pixels, so a future edit
    cannot move the check below the loader and still pass the test above."""
    import src.data.loaders as loaders

    def explode(*args, **kwargs):
        raise AssertionError("a loader was built for a refused domain")

    monkeypatch.setattr(loaders, "make_eval_loader", explode)
    monkeypatch.setattr(loaders, "make_val_loader", explode)
    cfg, path = checkpoint()
    with pytest.raises(DumpError):
        dump_checkpoint(path, tmp_path / "analysis", domain="raabin_testB")


def test_no_target_domain_appears_in_the_output(checkpoint, tmp_path):
    cfg, path = checkpoint()
    out = dump_checkpoint(path, tmp_path / "analysis")
    blob = json.dumps(meta_of(out))
    for name in ("raabin", "lisc", "pbc_test"):
        assert name not in blob


# --- the role check, as eval.py does it ------------------------------------------------

def test_an_informational_checkpoint_is_refused(checkpoint, tmp_path):
    """best.pt is the pbc_val peak the final-epoch rule discarded (CLAUDE.md sec 5)."""
    cfg, path = checkpoint(role="informational")
    with pytest.raises(Exception, match="headline|informational"):
        dump_checkpoint(path, tmp_path / "analysis")


def test_a_checkpoint_with_no_recorded_role_is_refused(checkpoint, tmp_path):
    import torch

    cfg, path = checkpoint()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    del payload["selection"]
    torch.save(payload, path)
    with pytest.raises(Exception):
        dump_checkpoint(path, tmp_path / "analysis")


def test_the_recorded_role_is_headline(checkpoint, tmp_path):
    cfg, path = checkpoint()
    assert meta_of(dump_checkpoint(path, tmp_path / "analysis"))["selection_role"] \
        == "headline"


# --- overwriting -------------------------------------------------------------------------

def test_an_existing_file_is_not_overwritten(checkpoint, tmp_path):
    cfg, path = checkpoint()
    out = dump_checkpoint(path, tmp_path / "analysis")
    with pytest.raises(DumpError, match="exists|overwrite"):
        dump_checkpoint(path, tmp_path / "analysis")
    assert out.is_file()


def test_the_existing_file_is_left_untouched_by_the_refusal(checkpoint, tmp_path):
    cfg, path = checkpoint()
    out = dump_checkpoint(path, tmp_path / "analysis")
    before = out.read_bytes()
    with pytest.raises(DumpError):
        dump_checkpoint(path, tmp_path / "analysis")
    assert out.read_bytes() == before


def test_the_output_is_named_for_the_run_and_seed(checkpoint, tmp_path):
    cfg, path = checkpoint()
    assert output_path(cfg, tmp_path).name == f"{cfg.run_id}_seed{cfg.seed}_pbc_val.npz"


# --- the model is read, never changed -----------------------------------------------------

def test_the_buffers_are_unchanged(checkpoint, tmp_path):
    """BatchNorm running statistics are source statistics baked into the model; a pass
    in train mode would move them and quietly change the checkpoint's behaviour."""
    cfg, path = checkpoint()
    meta = meta_of(dump_checkpoint(path, tmp_path / "analysis"))
    assert meta["buffer_sha256_before"] == meta["buffer_sha256_after"]


def test_the_model_stays_in_eval_mode_and_grad_stays_off(checkpoint, tmp_path):
    cfg, path = checkpoint()
    meta = meta_of(dump_checkpoint(path, tmp_path / "analysis"))
    assert meta["model_training_mode_seen"] is False
    assert meta["grad_enabled_seen"] is False


# --- provenance ----------------------------------------------------------------------------

def test_the_checkpoint_hash_commit_and_dirty_flag_are_recorded(checkpoint, tmp_path):
    from src.ledger import checkpoint_hash

    cfg, path = checkpoint()
    meta = meta_of(dump_checkpoint(path, tmp_path / "analysis"))
    assert meta["checkpoint_sha256"] == checkpoint_hash(path)
    assert "commit" in meta["git"] and "dirty" in meta["git"]


# --- contents ---------------------------------------------------------------------------------

def test_the_arrays_line_up_and_are_the_right_shapes(checkpoint, tmp_path):
    cfg, path = checkpoint()
    data = load(dump_checkpoint(path, tmp_path / "analysis"))
    n = len(data["image_ids"])
    assert n > 0
    assert data["y_true"].shape == (n,)
    assert data["class_logits"].shape[0] == n
    assert data["class_logits"].shape[1] == 5


def test_the_image_ids_are_unique_and_are_file_names(checkpoint, tmp_path):
    cfg, path = checkpoint()
    ids = load(dump_checkpoint(path, tmp_path / "analysis"))["image_ids"]
    assert len(set(ids.tolist())) == len(ids)


def test_the_bottleneck_carries_eleven_attribute_heads(checkpoint, tmp_path):
    cfg, path = checkpoint("bottleneck")
    data = load(dump_checkpoint(path, tmp_path / "analysis"))
    n = len(data["image_ids"])
    offsets = data["attribute_group_offsets"]
    assert len(offsets) == 12, "11 heads need 12 boundaries"
    assert data["attribute_probs"].shape == (n, int(offsets[-1]))
    # Each head is a distribution.
    for a, b in zip(offsets[:-1], offsets[1:]):
        block = data["attribute_probs"][:, int(a):int(b)]
        assert np.allclose(block.sum(axis=1), 1.0, atol=1e-4)


def test_the_true_attribute_codes_are_saved_too(checkpoint, tmp_path):
    """Needed by the source-side attribute analysis, which scores the predicted
    attributes against the WBCAtt labels. Source data, so nothing here is constrained."""
    cfg, path = checkpoint("bottleneck")
    data = load(dump_checkpoint(path, tmp_path / "analysis"))
    assert data["attribute_true"].shape == (len(data["image_ids"]), 11)


def test_the_baseline_has_no_attribute_arrays(checkpoint, tmp_path):
    cfg, path = checkpoint("baseline")
    data = load(dump_checkpoint(path, tmp_path / "analysis"))
    assert "attribute_probs" not in data
    assert meta_of(output_path(cfg, tmp_path / "analysis"))["has_attributes"] is False


def test_the_controls_attribute_heads_are_labelled_uninterpretable(checkpoint, tmp_path):
    """lambda is 0 for the control, so nothing ever supervised those heads. They are
    the bottleneck's architecture without its supervision, and reading them as
    attributes would be reading noise."""
    cfg, path = checkpoint("control")
    meta = meta_of(dump_checkpoint(path, tmp_path / "analysis"))
    assert meta["attributes_interpretable"] is False
    assert "unsupervised" in meta["attributes_note"].lower()
    assert meta["attribute_loss_weight"] == 0.0


def test_the_bottlenecks_attribute_heads_are_interpretable(checkpoint, tmp_path):
    cfg, path = checkpoint("bottleneck")
    meta = meta_of(dump_checkpoint(path, tmp_path / "analysis"))
    assert meta["attributes_interpretable"] is True


def test_the_class_logit_mode_is_recorded(checkpoint, tmp_path):
    """A bottleneck has two; which one these logits are is not guessable from the file."""
    cfg, path = checkpoint("bottleneck")
    assert meta_of(dump_checkpoint(path, tmp_path / "analysis"))["class_logits_mode"] \
        == "soft"


def test_the_attribute_and_class_names_are_recorded(checkpoint, tmp_path):
    cfg, path = checkpoint("bottleneck")
    meta = meta_of(dump_checkpoint(path, tmp_path / "analysis"))
    assert len(meta["attribute_names"]) == 11
    assert len(meta["classes"]) == 5


def test_two_model_types_land_in_different_files(checkpoint, tmp_path):
    out = tmp_path / "analysis"
    first = dump_checkpoint(checkpoint("bottleneck")[1], out)
    second = dump_checkpoint(checkpoint("baseline")[1], out)
    assert first != second
