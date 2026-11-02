"""Loss composition and class weighting."""
import pytest

from src.config import load_config
from src.losses import class_weights_from, compute_loss

pytest.importorskip("torch")


@pytest.fixture
def cfg(experiment_config):
    def build(**overrides):
        return load_config(experiment_config(**{"data.image_size": 32, **overrides}))
    return build


def fake_output(label_space, batch=4):
    import torch

    from src.models.build import ModelOutput

    return ModelOutput(
        class_logits=torch.randn(batch, 5),
        attribute_logits=[torch.randn(batch, len(v))
                          for v in label_space.attributes.values()],
    )


def fake_targets(label_space, batch=4):
    import torch

    return (torch.zeros(batch, dtype=torch.long),
            torch.zeros(batch, 11, dtype=torch.long))


# --- composition ---------------------------------------------------------------------

def test_lambda_zero_makes_total_exactly_the_class_term(cfg, label_space):
    """The control must be the baseline's loss on the bottleneck's architecture, with
    nothing left over - bitwise, not approximately."""
    import torch

    torch.manual_seed(0)
    out, targets = fake_output(label_space), fake_targets(label_space)
    c = cfg(**{"model.type": "control", "bottleneck.attribute_loss_weight": 0.0})
    total, parts = compute_loss(out, targets, c, class_weights=None)
    assert total.item() == parts["class"].item()
    assert parts["attribute"].item() == 0.0


def test_attribute_term_is_the_sum_over_eleven_heads_not_the_mean(cfg, label_space):
    """Sum vs mean silently rescales lambda by 11x, so it is pinned."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    out, targets = fake_output(label_space), fake_targets(label_space)
    _, parts = compute_loss(out, targets, cfg(), class_weights=None)
    expected = sum(F.cross_entropy(logits, targets[1][:, i])
                   for i, logits in enumerate(out.attribute_logits))
    assert parts["attribute"].item() == pytest.approx(expected.item(), rel=1e-6)


def test_lambda_scales_only_the_attribute_term(cfg, label_space):
    import torch

    torch.manual_seed(0)
    out, targets = fake_output(label_space), fake_targets(label_space)
    _, a = compute_loss(out, targets, cfg(**{"bottleneck.attribute_loss_weight": 1.0}),
                        class_weights=None)
    _, b = compute_loss(out, targets, cfg(**{"bottleneck.attribute_loss_weight": 2.0}),
                        class_weights=None)
    assert a["class"].item() == pytest.approx(b["class"].item())
    assert b["attribute"].item() == pytest.approx(a["attribute"].item())
    assert b["total"].item() == pytest.approx(
        a["class"].item() + 2 * a["attribute"].item(), rel=1e-6)


def test_baseline_loss_has_no_attribute_term(cfg, label_space):
    import torch

    from src.models.build import ModelOutput

    torch.manual_seed(0)
    out = ModelOutput(class_logits=torch.randn(4, 5), attribute_logits=None)
    c = cfg(**{"model.type": "baseline", "bottleneck": None,
               "eval.attribute_modes": None})
    total, parts = compute_loss(out, fake_targets(label_space), c, class_weights=None)
    assert parts["attribute"].item() == 0.0
    assert total.item() == parts["class"].item()


def test_attribute_cross_entropy_is_unweighted(cfg, label_space):
    """CLAUDE.md sec 6: class weighting enters the class term only. Imbalanced
    attribute values are reported via balanced accuracy instead."""
    import torch

    torch.manual_seed(0)
    out, targets = fake_output(label_space), fake_targets(label_space)
    weights = torch.tensor([5.0, 1.0, 1.0, 1.0, 1.0])
    _, unweighted = compute_loss(out, targets, cfg(), class_weights=None)
    _, weighted = compute_loss(out, targets, cfg(), class_weights=weights)
    assert weighted["attribute"].item() == pytest.approx(unweighted["attribute"].item())
    assert weighted["class"].item() != pytest.approx(unweighted["class"].item())


# --- class weights come from pbc_train only --------------------------------------------

def test_class_weights_from_pbc_train(experiment_config, tiny_raw_loadable, mapping,
                                      label_space):
    import torch

    from src.data.loaders import make_train_loader

    cfg = load_config(experiment_config(**{"data.image_size": 16}))
    dataset = make_train_loader(cfg, mapping, label_space).dataset
    weights = class_weights_from(dataset, cfg, label_space)
    assert weights is None or weights.shape == (5,)


def test_inverse_frequency_weights_are_larger_for_rarer_classes(
        experiment_config, tiny_raw_loadable, mapping, label_space):
    from src.data.loaders import make_train_loader

    cfg = load_config(experiment_config(**{"data.image_size": 16,
                                           "train.class_weighting": "inverse_frequency"}))
    dataset = make_train_loader(cfg, mapping, label_space).dataset
    weights = class_weights_from(dataset, cfg, label_space)
    counts = [dataset.class_indices.count(i) for i in range(5)]
    present = [i for i, n in enumerate(counts) if n]
    rarest = min(present, key=lambda i: counts[i])
    commonest = max(present, key=lambda i: counts[i])
    if counts[rarest] != counts[commonest]:
        assert weights[rarest] > weights[commonest]


def test_class_weighting_none_returns_none(experiment_config, tiny_raw_loadable,
                                           mapping, label_space):
    from src.data.loaders import make_train_loader

    cfg = load_config(experiment_config(**{"data.image_size": 16,
                                           "train.class_weighting": "none"}))
    dataset = make_train_loader(cfg, mapping, label_space).dataset
    assert class_weights_from(dataset, cfg, label_space) is None


@pytest.mark.parametrize("dataset_id", ["pbc_val", "pbc_test"])
def test_class_weights_refuse_val_and_test(experiment_config, tiny_raw_loadable,
                                           mapping, label_space, dataset_id):
    """val frequencies are a tuning-set statistic; test frequencies are held out."""
    from src.data.datasets import PbcWbcAttDataset

    cfg = load_config(experiment_config(**{"data.image_size": 16,
                                           "train.class_weighting": "inverse_frequency"}))
    dataset = PbcWbcAttDataset(dataset_id, cfg.paths.pbc_images, cfg.paths.wbcatt_dir,
                               mapping, label_space, 16)
    with pytest.raises(ValueError, match="pbc_train"):
        class_weights_from(dataset, cfg, label_space)


def test_class_weights_refuse_target_data(experiment_config, tiny_raw_loadable,
                                          mapping, label_space):
    from src.data.datasets import RaabinFolderDataset
    from src.data.domains import TargetDomainLeakage

    cfg = load_config(experiment_config(**{"data.image_size": 16,
                                           "train.class_weighting": "inverse_frequency"}))
    dataset = RaabinFolderDataset("raabin_testB", cfg.paths.raabin_root, mapping,
                                  label_space, 16)
    with pytest.raises(TargetDomainLeakage):
        class_weights_from(dataset, cfg, label_space)
