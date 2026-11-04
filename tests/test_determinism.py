"""The batch stream must depend on the seed alone, not on which model was built.

The confound this file exists to prevent: torchvision's random transforms and the
DataLoader's default RandomSampler both draw from the global torch RNG. Model
construction also draws from it, and the baseline's Linear(2048, 5) head draws a
different amount than the bottleneck's Linear(2048, 31) plus Linear(31, 5). Since
build_model runs before the first epoch is iterated, the three models saw different
shuffle orders AND different augmentations at the same seed - a protocol difference
correlated with the representation, which CLAUDE.md sec 5 forbids.

Source domain only; make_train_loader cannot reach anything else.
"""
import hashlib

import pytest

pytest.importorskip("torch")

from conftest import DELETE
from src.config import load_config
from src.data.loaders import make_train_loader, seed_epoch


def _tiny(experiment_config, **overrides):
    base = {"data.image_size": 32, "train.batch_size": 2, "train.epochs": 1}
    base.update(overrides)
    return load_config(experiment_config(**base))


def _digest(loader, cfg, epoch=0):
    """sha256 over every pixel, label and attribute the loader yields for one epoch."""
    seed_epoch(cfg, epoch, loader)
    h = hashlib.sha256()
    for images, labels, attributes in loader:
        h.update(images.numpy().tobytes())
        h.update(labels.numpy().tobytes())
        if attributes is not None:
            h.update(attributes.numpy().tobytes())
    return h.hexdigest()


def _build(cfg, label_space):
    from src.models.build import build_model

    return build_model(cfg, label_space)


# --- the confound ---------------------------------------------------------------------

def test_the_batch_stream_is_identical_whichever_model_was_built(
        experiment_config, tiny_raw_loadable, mapping, label_space):
    """The headline assertion. Same seed, different model type, identical batches."""
    digests = {}
    for model_type, weight in (("baseline", None), ("bottleneck", 1.0),
                               ("control", 0.0)):
        overrides = {"model.type": model_type}
        if weight is None:
            # A baseline declares neither a bottleneck block nor attribute modes.
            overrides["bottleneck"] = DELETE
            overrides["eval.attribute_modes"] = DELETE
        else:
            overrides["bottleneck.attribute_loss_weight"] = weight
        cfg = _tiny(experiment_config, **overrides)
        loader = make_train_loader(cfg, mapping, label_space)
        _build(cfg, label_space)          # consumes the global RNG, differently per type
        digests[model_type] = _digest(loader, cfg)
    assert len(set(digests.values())) == 1, digests


def test_building_a_model_twice_does_not_move_the_stream(
        experiment_config, tiny_raw_loadable, mapping, label_space):
    """Not just model type: any extra draw from the global RNG must be invisible."""
    cfg = _tiny(experiment_config)
    loader = make_train_loader(cfg, mapping, label_space)
    once = _digest(loader, cfg)
    for _ in range(3):
        _build(cfg, label_space)
    assert _digest(loader, cfg) == once


def test_an_arbitrary_global_draw_does_not_move_the_stream(
        experiment_config, tiny_raw_loadable, mapping, label_space):
    import torch

    cfg = _tiny(experiment_config)
    loader = make_train_loader(cfg, mapping, label_space)
    before = _digest(loader, cfg)
    torch.rand(1_000)
    assert _digest(loader, cfg) == before


# --- the stream is still a function of the seed and the epoch -------------------------

def test_the_same_seed_reproduces_the_same_epoch(
        experiment_config, tiny_raw_loadable, mapping, label_space):
    cfg = _tiny(experiment_config)
    first = _digest(make_train_loader(cfg, mapping, label_space), cfg, epoch=0)
    second = _digest(make_train_loader(cfg, mapping, label_space), cfg, epoch=0)
    assert first == second


def test_a_different_seed_gives_a_different_epoch(
        experiment_config, tiny_raw_loadable, mapping, label_space):
    a = _tiny(experiment_config, seed=0)
    b = _tiny(experiment_config, seed=1)
    assert (_digest(make_train_loader(a, mapping, label_space), a)
            != _digest(make_train_loader(b, mapping, label_space), b))


def test_successive_epochs_differ(experiment_config, tiny_raw_loadable, mapping,
                                  label_space):
    """Shuffling and augmentation must actually vary across epochs; a per-epoch seed
    that ignored the epoch would silently train on one fixed ordering."""
    cfg = _tiny(experiment_config)
    loader = make_train_loader(cfg, mapping, label_space)
    assert _digest(loader, cfg, epoch=0) != _digest(loader, cfg, epoch=1)


def test_the_epoch_seed_is_a_pure_function_of_seed_and_epoch(experiment_config,
                                                             tiny_raw_loadable):
    """Out of order, repeated, interleaved - the same (seed, epoch) must give the same
    RNG state every time, or resuming and reordering change the data."""
    import torch

    cfg = _tiny(experiment_config)
    seen = {}
    for epoch in (3, 1, 3, 0, 1):
        seeds = seed_epoch(cfg, epoch)
        state = torch.get_rng_state().clone()
        if epoch in seen:
            assert seen[epoch] == (seeds, state.tolist()), epoch
        seen[epoch] = (seeds, state.tolist())
    assert len(seen) == 3


# --- the sampler has its own generator ------------------------------------------------

def test_the_train_loader_owns_a_generator(experiment_config, tiny_raw_loadable,
                                           mapping, label_space):
    """Structural: without an explicit generator the RandomSampler seeds itself from the
    global RNG, which is exactly the coupling this file removes."""
    cfg = _tiny(experiment_config)
    loader = make_train_loader(cfg, mapping, label_space)
    assert loader.generator is not None


def test_two_loaders_at_one_seed_shuffle_alike(experiment_config, tiny_raw_loadable,
                                               mapping, label_space):
    cfg = _tiny(experiment_config)
    a = make_train_loader(cfg, mapping, label_space)
    b = make_train_loader(cfg, mapping, label_space)
    assert list(iter(a.sampler)) == list(iter(b.sampler))


def test_the_val_loader_needs_no_generator(experiment_config, tiny_raw_loadable,
                                           mapping, label_space):
    """pbc_val is never shuffled and never augmented, so it has nothing to seed."""
    from src.data.loaders import make_val_loader

    cfg = _tiny(experiment_config)
    loader = make_val_loader(cfg, mapping, label_space)
    assert loader.generator is None


# --- the training loop actually uses it -----------------------------------------------

def test_train_pins_every_epoch_and_records_the_seeds(experiment_config,
                                                      tiny_raw_loadable, mapping,
                                                      label_space, tmp_path):
    """The helper being correct is worth nothing if the loop does not call it, so the
    seeds it set are recorded per epoch and checked against the derivation."""
    from src.data.loaders import _epoch_seed
    from src.train import train

    cfg = _tiny(experiment_config, **{"train.epochs": 2})
    history = train(cfg, mapping, label_space, out_dir=tmp_path)["history"]
    assert len(history) == 2
    for row in history:
        assert row["epoch_seeds"] == {
            "augmentation": _epoch_seed(cfg.seed, row["epoch"], "augmentation"),
            "order": _epoch_seed(cfg.seed, row["epoch"], "order"),
        }
