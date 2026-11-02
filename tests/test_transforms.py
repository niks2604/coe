"""Transforms read resize and normalisation from the config only, and augmentation is
training-only and source-only."""
import pytest

from src.config import IMAGENET_MEAN, IMAGENET_STD, load_config
from src.data.transforms import build_eval_transform, build_train_transform

pytest.importorskip("torch")


def test_eval_transform_is_deterministic(experiment_config, tiny_raw):
    from PIL import Image

    cfg = load_config(experiment_config(**{"data.image_size": 16}))
    t = build_eval_transform(cfg)
    img = Image.new("RGB", (40, 33), (10, 200, 60))
    assert t(img).equal(t(img))


def test_train_transform_is_not_deterministic(experiment_config):
    """If augmentation were inert, the bottleneck and baseline would both train on the
    unaugmented source and the comparison would be against a different protocol."""
    import torch
    from PIL import Image

    cfg = load_config(experiment_config(**{"data.image_size": 16}))
    t = build_train_transform(cfg)
    img = Image.new("RGB", (40, 33))
    img.putpixel((0, 0), (255, 0, 0))
    torch.manual_seed(0)
    a = t(img)
    outs = [t(img) for _ in range(12)]
    assert any(not a.equal(o) for o in outs)


def test_eval_transform_resizes_to_the_config_size(experiment_config):
    from PIL import Image

    cfg = load_config(experiment_config(**{"data.image_size": 24}))
    out = build_eval_transform(cfg)(Image.new("RGB", (40, 33)))
    assert out.shape == (3, 24, 24)


def test_normalisation_comes_from_the_config(experiment_config):
    """A grey image maps to the value implied by the ImageNet constants, so the
    constants in the config are the ones actually applied."""
    import torch
    from PIL import Image

    cfg = load_config(experiment_config(**{"data.image_size": 8}))
    out = build_eval_transform(cfg)(Image.new("RGB", (8, 8), (0, 0, 0)))
    expected = torch.tensor([-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)])
    assert torch.allclose(out[:, 0, 0], expected, atol=1e-5)


def test_augmentation_settings_come_from_the_config(experiment_config):
    cfg = load_config(experiment_config())
    assert cfg.train.augmentation["hflip"] == 0.5
    assert "color_jitter" in cfg.train.augmentation


def test_augmentation_can_be_disabled_for_the_smoke_test(experiment_config):
    import torch
    from PIL import Image

    cfg = load_config(experiment_config(**{
        "train.augmentation": {"hflip": 0.0, "vflip": 0.0, "rotation_degrees": 0,
                               "color_jitter": None},
    }))
    t = build_train_transform(cfg)
    img = Image.new("RGB", (16, 16), (40, 90, 160))
    torch.manual_seed(0)
    assert t(img).equal(t(img))


def test_unknown_augmentation_key_is_rejected(experiment_config):
    from src.config import ConfigError

    with pytest.raises(ConfigError, match="elastic"):
        load_config(experiment_config(**{
            "train.augmentation": {"hflip": 0.5, "vflip": 0.5, "rotation_degrees": 180,
                                   "color_jitter": None, "elastic": 0.3},
        }))


def test_jitter_strength_is_capped(experiment_config):
    """Stain colour variation IS the domain shift. Strong jitter tuned upward is
    target-informed tuning in disguise, so the schema caps it (CLAUDE.md sec 3)."""
    from src.config import ConfigError

    with pytest.raises(ConfigError, match="color_jitter"):
        load_config(experiment_config(**{
            "train.augmentation.color_jitter": {"brightness": 0.8, "contrast": 0.8,
                                                "saturation": 0.8, "hue": 0.4},
        }))
