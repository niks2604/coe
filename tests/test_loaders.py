"""Loader factories and the domain guards. The point of this module is that there is no
argument to any function here that yields a training loader over target data."""
import pytest

from src.config import load_config
from src.data.domains import TargetDomainLeakage
from src.data.loaders import make_eval_loader, make_train_loader, make_val_loader

pytest.importorskip("torch")

TARGET_IDS = ["raabin_train", "raabin_testA", "raabin_testB", "lisc"]


@pytest.fixture
def ready(experiment_config, tiny_raw_loadable, mapping, label_space):
    """A loaded config plus mapping and label space, over the synthetic tree."""
    cfg = load_config(experiment_config(**{"data.image_size": 16, "train.batch_size": 2}))
    return cfg, mapping, label_space


# --- the guards --------------------------------------------------------------------------

@pytest.mark.parametrize("dataset_id", TARGET_IDS)
def test_no_train_loader_for_any_target(ready, dataset_id):
    cfg, mapping, ls = ready
    with pytest.raises(TargetDomainLeakage, match=dataset_id):
        make_train_loader(cfg, mapping, ls, dataset_id=dataset_id)


@pytest.mark.parametrize("dataset_id", TARGET_IDS)
def test_no_val_loader_for_any_target(ready, dataset_id):
    cfg, mapping, ls = ready
    with pytest.raises(TargetDomainLeakage, match=dataset_id):
        make_val_loader(cfg, mapping, ls, dataset_id=dataset_id)


def test_train_loader_refuses_pbc_test(ready):
    """Source, but held out for the source-domain report."""
    cfg, mapping, ls = ready
    with pytest.raises(ValueError, match="pbc_val|pbc_train"):
        make_train_loader(cfg, mapping, ls, dataset_id="pbc_test")


def test_val_loader_refuses_pbc_test(ready):
    cfg, mapping, ls = ready
    with pytest.raises(ValueError, match="pbc_val"):
        make_val_loader(cfg, mapping, ls, dataset_id="pbc_test")


def test_train_and_val_default_to_their_pinned_splits(ready):
    cfg, mapping, ls = ready
    assert make_train_loader(cfg, mapping, ls).dataset.dataset_id == "pbc_train"
    assert make_val_loader(cfg, mapping, ls).dataset.dataset_id == "pbc_val"


# --- shuffling and augmentation ----------------------------------------------------------

def test_train_loader_shuffles_and_augments(ready):
    cfg, mapping, ls = ready
    loader = make_train_loader(cfg, mapping, ls)
    assert loader.dataset.augmented is True
    assert not isinstance(loader.sampler, __import__("torch").utils.data.SequentialSampler)


def test_val_loader_neither_shuffles_nor_augments(ready):
    import torch

    cfg, mapping, ls = ready
    loader = make_val_loader(cfg, mapping, ls)
    assert loader.dataset.augmented is False
    assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)


@pytest.mark.parametrize("dataset_id", ["raabin_train", "raabin_testA", "raabin_testB"])
def test_eval_loader_never_shuffles_or_augments(ready, dataset_id):
    import torch

    cfg, mapping, ls = ready
    loader = make_eval_loader(cfg, dataset_id, mapping, ls)
    assert loader.dataset.augmented is False
    assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)


def test_eval_loader_output_is_repeatable(ready):
    """No augmentation anywhere in the target path: two passes are bit-identical."""
    cfg, mapping, ls = ready
    loader = make_eval_loader(cfg, "raabin_testB", mapping, ls)
    first = [b[0] for b in loader]
    second = [b[0] for b in loader]
    assert all(a.equal(b) for a, b in zip(first, second))


def test_pbc_test_goes_through_the_same_eval_path(ready):
    """Source test and target must not be measured by different code."""
    cfg, mapping, ls = ready
    loader = make_eval_loader(cfg, "pbc_test", mapping, ls)
    assert loader.dataset.augmented is False
    assert loader.dataset.domain.value == "source"


def test_eval_loader_skips_an_absent_dataset(ready):
    """lisc is out of scope and absent: the loader reports it, rather than crashing."""
    cfg, mapping, ls = ready
    assert make_eval_loader(cfg, "lisc", mapping, ls) is None


# --- determinism and subsetting -----------------------------------------------------------

def test_batch_order_is_seed_reproducible(ready):
    from src.config import set_seed

    cfg, mapping, ls = ready

    def order():
        set_seed(cfg.seed)
        return [n for b in make_train_loader(cfg, mapping, ls) for n in b[1].tolist()]

    assert order() == order()


def test_different_seed_changes_batch_order(experiment_config, tiny_raw_loadable,
                                            mapping, label_space):
    from src.config import set_seed

    def order(seed):
        cfg = load_config(experiment_config(**{"seed": seed, "data.image_size": 16,
                                               "train.batch_size": 1}))
        set_seed(seed)
        return [n for b in make_train_loader(cfg, mapping, label_space)
                for n in b[1].tolist()]

    assert order(0) != order(7)


def test_subset_n_images_is_class_stratified(experiment_config, tiny_raw_loadable,
                                             mapping, label_space):
    """The sec 7 smoke test must still contain every class, or it smoke-tests nothing."""
    cfg = load_config(experiment_config(**{"data.subset_n_images": 4,
                                           "data.image_size": 16}))
    loader = make_train_loader(cfg, mapping, label_space)
    assert len(loader.dataset) == 4
    classes = {int(loader.dataset[i][1]) for i in range(len(loader.dataset))}
    assert len(classes) == 4


def test_subset_does_not_apply_to_target_domains(experiment_config, tiny_raw_loadable,
                                                 mapping, label_space):
    """subset_n_images shrinks the source for a smoke test. Shrinking a target domain
    would change what the reported gap is measured over."""
    base = {"data.image_size": 16}
    full = load_config(experiment_config(**base))
    small = load_config(experiment_config(**{**base, "data.subset_n_images": 2}))
    n_full = len(make_eval_loader(full, "raabin_testB", mapping, label_space).dataset)
    n_small = len(make_eval_loader(small, "raabin_testB", mapping, label_space).dataset)
    assert n_small == n_full
