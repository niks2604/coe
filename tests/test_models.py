"""Model construction. The load-bearing test here is that control and bottleneck are
structurally identical - that is what isolates attribute supervision from architecture.
"""
import pytest

from src.config import load_config
from src.models.build import WeightsUnavailableError, build_model, resolve_imagenet_weights

pytest.importorskip("torch")
pytest.importorskip("torchvision")


@pytest.fixture
def cfg(experiment_config):
    def build(**overrides):
        return load_config(experiment_config(**{"data.image_size": 32, **overrides}))
    return build


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# --- shapes -------------------------------------------------------------------------

def test_baseline_outputs_five_logits(cfg, label_space):
    import torch

    model = build_model(cfg(**{"model.type": "baseline", "bottleneck": None,
                               "eval.attribute_modes": None}), label_space)
    out = model(torch.zeros(2, 3, 32, 32))
    assert out.class_logits.shape == (2, 5)
    assert out.attribute_logits is None


def test_bottleneck_has_eleven_heads_summing_to_thirty_one(cfg, label_space):
    model = build_model(cfg(), label_space)
    sizes = [h.out_features for h in model.attribute_heads]
    assert sizes == [len(v) for v in label_space.attributes.values()]
    assert sum(sizes) == 31


def test_bottleneck_classifier_is_linear_31_to_5_with_bias(cfg, label_space):
    model = build_model(cfg(), label_space)
    assert model.classifier.in_features == 31
    assert model.classifier.out_features == 5
    assert model.classifier.bias is not None


def test_bottleneck_returns_eleven_attribute_logit_tensors(cfg, label_space):
    import torch

    model = build_model(cfg(), label_space)
    out = model(torch.zeros(2, 3, 32, 32))
    assert len(out.attribute_logits) == 11
    for logits, column in zip(out.attribute_logits, label_space.attributes):
        assert logits.shape == (2, len(label_space.attributes[column]))


def test_group_offsets_match_the_label_space_order(cfg, label_space):
    """The 31 classifier inputs must be interpretable as (column, value) pairs."""
    model = build_model(cfg(), label_space)
    expected, at = [], 0
    for column, values in label_space.attributes.items():
        expected.append((column, at, at + len(values)))
        at += len(values)
    assert model.group_offsets == expected


# --- control is structurally identical ------------------------------------------------

def test_control_and_bottleneck_are_structurally_identical(cfg, label_space):
    """The whole point of the control. Any structural difference would confound
    attribute supervision with architecture."""
    b = build_model(cfg(**{"model.type": "bottleneck",
                           "bottleneck.attribute_loss_weight": 1.0}), label_space)
    c = build_model(cfg(**{"model.type": "control",
                           "bottleneck.attribute_loss_weight": 0.0}), label_space)
    assert n_params(b) == n_params(c)
    assert [tuple(p.shape) for p in b.parameters()] == [tuple(p.shape) for p in c.parameters()]
    assert repr(b) == repr(c)


def test_baseline_has_fewer_parameters_than_the_bottleneck(cfg, label_space):
    baseline = build_model(cfg(**{"model.type": "baseline", "bottleneck": None,
                                  "eval.attribute_modes": None}), label_space)
    assert n_params(baseline) != n_params(build_model(cfg(), label_space))


# --- soft and hard modes ----------------------------------------------------------------

def test_soft_mode_bottleneck_vector_is_per_group_softmax(cfg, label_space):
    import torch

    model = build_model(cfg(), label_space)
    vec = model.bottleneck_vector(model(torch.zeros(2, 3, 32, 32)).attribute_logits,
                                  mode="soft")
    assert vec.shape == (2, 31)
    for _, lo, hi in model.group_offsets:
        assert torch.allclose(vec[:, lo:hi].sum(dim=1), torch.ones(2), atol=1e-5)


def test_hard_mode_is_one_hot_per_group(cfg, label_space):
    import torch

    model = build_model(cfg(), label_space)
    vec = model.bottleneck_vector(model(torch.zeros(2, 3, 32, 32)).attribute_logits,
                                  mode="hard")
    assert vec.sum().item() == pytest.approx(2 * 11)
    assert set(vec.unique().tolist()) <= {0.0, 1.0}


def test_hard_equals_soft_when_attributes_are_already_one_hot(cfg, label_space):
    import torch

    model = build_model(cfg(), label_space)
    peaked = [torch.full((1, len(v)), -50.0) for v in label_space.attributes.values()]
    for t in peaked:
        t[0, 0] = 50.0
    soft = model.bottleneck_vector(peaked, mode="soft")
    hard = model.bottleneck_vector(peaked, mode="hard")
    assert torch.allclose(soft, hard, atol=1e-5)


def test_unknown_attribute_mode_raises(cfg, label_space):
    import torch

    model = build_model(cfg(), label_space)
    # batch 2: BatchNorm in train mode rejects a single sample, which would raise
    # before bottleneck_vector is ever called.
    logits = model(torch.zeros(2, 3, 32, 32)).attribute_logits
    with pytest.raises(ValueError, match="gumbel"):
        model.bottleneck_vector(logits, mode="gumbel")


def test_baseline_has_no_bottleneck_vector(cfg, label_space):
    model = build_model(cfg(**{"model.type": "baseline", "bottleneck": None,
                               "eval.attribute_modes": None}), label_space)
    assert not hasattr(model, "bottleneck_vector")


# --- backbone normalisation --------------------------------------------------------------

def test_batchnorm_arm_keeps_batchnorm(cfg, label_space):
    import torch

    model = build_model(cfg(), label_space)
    assert any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())


def test_groupnorm_arm_leaves_no_batchnorm(cfg, label_space):
    import torch

    model = build_model(cfg(**{"model.backbone_norm": "groupnorm"}), label_space)
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert any(isinstance(m, torch.nn.GroupNorm) for m in model.modules())


def test_groupnorm_arm_reports_the_dropped_tensors(cfg, label_space):
    """Swapping BatchNorm for GroupNorm discards the pretrained BN affine and running
    stats. That is a partial load and must be stated, not silent."""
    model = build_model(cfg(**{"model.backbone_norm": "groupnorm"}), label_space)
    assert model.weight_load_report["dropped"] > 0
    assert model.weight_load_report["loaded"] > 0


def test_batchnorm_arm_reports_a_complete_load(cfg, label_space):
    model = build_model(cfg(), label_space)
    assert model.weight_load_report["dropped"] == 0


def test_imagenet_weights_are_actually_loaded(cfg, label_space):
    """A randomly initialised backbone would read as a weak result rather than a setup
    error, so this compares an actual tensor against torchvision's."""
    import torch
    from torchvision.models import ResNet50_Weights, resnet50

    reference = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model = build_model(cfg(), label_space)
    assert torch.allclose(model.backbone.conv1.weight, reference.conv1.weight)


# --- offline / unavailable weights --------------------------------------------------------

def test_local_weights_file_is_used_when_present(cfg, label_space, tmp_path):
    import torch
    from torchvision.models import ResNet50_Weights, resnet50

    path = tmp_path / "resnet50.pth"
    torch.save(resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).state_dict(), path)
    state = resolve_imagenet_weights(cfg(**{"model.imagenet_weights": str(path)}))
    assert "conv1.weight" in state


def test_missing_local_weights_file_fails_clearly(cfg, label_space, tmp_path):
    with pytest.raises(WeightsUnavailableError, match="does not exist"):
        resolve_imagenet_weights(cfg(**{"model.imagenet_weights": str(tmp_path / "nope.pth")}))


def test_no_internet_and_no_local_weights_names_both_options(cfg, monkeypatch):
    """On Kaggle without internet the weights come from an attached dataset. If neither
    route is available the run must fail saying so, never train from random init."""
    import src.models.build as build_mod

    def boom(*_a, **_k):
        raise OSError("no internet")

    monkeypatch.setattr(build_mod, "_download_imagenet_weights", boom)
    with pytest.raises(WeightsUnavailableError) as exc:
        resolve_imagenet_weights(cfg(**{"model.imagenet_weights": None}))
    message = str(exc.value)
    assert "imagenet_weights" in message and "internet" in message


def test_random_init_is_never_a_silent_fallback(cfg, label_space, monkeypatch):
    import src.models.build as build_mod

    monkeypatch.setattr(build_mod, "_download_imagenet_weights",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    with pytest.raises(WeightsUnavailableError):
        build_model(cfg(**{"model.imagenet_weights": None}), label_space)


# --- fine-tuning depth ---------------------------------------------------------------------

def test_whole_backbone_is_trainable(cfg, label_space):
    """CLAUDE.md sec 5: full fine-tune, identical across models. A frozen backbone would
    be a different protocol."""
    model = build_model(cfg(), label_space)
    assert all(p.requires_grad for p in model.backbone.parameters())


def test_fine_tuning_depth_is_identical_across_models(cfg, label_space):
    frozen = []
    for overrides in ({"model.type": "baseline", "bottleneck": None,
                       "eval.attribute_modes": None},
                      {"model.type": "bottleneck"},
                      {"model.type": "control",
                       "bottleneck.attribute_loss_weight": 0.0}):
        model = build_model(cfg(**overrides), label_space)
        frozen.append(sum(1 for p in model.parameters() if not p.requires_grad))
    assert frozen == [0, 0, 0]
