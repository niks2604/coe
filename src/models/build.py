"""Model construction: one backbone, three heads.

The three model types differ in exactly one way each, and nothing else:

  baseline    features -> Linear(2048, 5)
  bottleneck  features -> 11 attribute heads (31 values) -> per-head softmax
                       -> concat(31) -> Linear(31, 5)
  control     the bottleneck, with the attribute loss weighted 0

bottleneck and control share one class, so they are structurally identical by
construction rather than by agreement - that identity is the whole reason the control
exists (CLAUDE.md sec 6). Only the loss differs.

The 31 classifier inputs are positionally the label space's (column, value) pairs, which
makes Linear(31, 5) readable afterwards: 155 weights saying which attribute values drive
each class. group_offsets records the mapping and goes into the checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from src.data.label_space import LabelSpace

FEATURE_DIM = 2048
N_CLASSES = 5
GROUPNORM_GROUPS = 32
ATTRIBUTE_MODES = ("soft", "hard")


class WeightsUnavailableError(Exception):
    """Pretrained weights could not be obtained.

    Always raised rather than falling back to random initialisation: a randomly
    initialised backbone produces a weak result that reads as a finding instead of a
    broken environment (CLAUDE.md sec 5).
    """


@dataclass
class ModelOutput:
    class_logits: torch.Tensor
    attribute_logits: list[torch.Tensor] | None


# --- pretrained weights ----------------------------------------------------------------

def _download_imagenet_weights() -> dict:
    """Separate function so tests can simulate an offline environment."""
    return ResNet50_Weights.IMAGENET1K_V2.get_state_dict(progress=False)


def resolve_imagenet_weights(cfg) -> dict:
    """Pretrained ResNet-50 state dict, from a local file or the download.

    Two supported routes (CLAUDE.md sec 5): the environment has internet access, or
    model.imagenet_weights points at a local file - on Kaggle, an attached dataset.
    """
    configured = cfg.model.imagenet_weights
    if configured:
        path = Path(configured).expanduser()
        if not path.exists():
            raise WeightsUnavailableError(
                f"model.imagenet_weights points at {path}, which does not exist. "
                f"Supply the weights file there, or set model.imagenet_weights: null "
                f"to download them (requires internet access)."
            )
        state = torch.load(path, map_location="cpu", weights_only=True)
        return state.get("state_dict", state) if isinstance(state, dict) else state

    try:
        return _download_imagenet_weights()
    except Exception as exc:
        raise WeightsUnavailableError(
            f"could not download pretrained ResNet-50 weights ({exc}). Either run with "
            f"internet access, or download resnet50 ImageNet1K_V2 weights and point "
            f"model.imagenet_weights at the file (on Kaggle, attach them as a dataset "
            f"and use the /kaggle/input path). Training from random initialisation is "
            f"not a fallback: it would read as a weak result rather than a setup error."
        ) from exc


# --- backbone ---------------------------------------------------------------------------

def _to_groupnorm(module: nn.Module) -> None:
    """Replace every BatchNorm2d in place with GroupNorm.

    This discards the pretrained BN affine parameters and running statistics, so the
    load becomes partial - reported, never silent. BN running statistics are source
    statistics baked into the model and a known cause of domain-shift degradation, which
    is why this arm exists at all (CLAUDE.md sec 6).
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            channels = child.num_features
            groups = GROUPNORM_GROUPS if channels % GROUPNORM_GROUPS == 0 else 1
            setattr(module, name, nn.GroupNorm(groups, channels))
        else:
            _to_groupnorm(child)


def build_backbone(cfg) -> tuple[nn.Module, dict]:
    """ResNet-50 with the classifier removed, plus a load report."""
    if cfg.model.backbone != "resnet50":
        raise ValueError(f"unsupported backbone {cfg.model.backbone!r}; only resnet50")

    state = resolve_imagenet_weights(cfg)
    backbone = resnet50(weights=None)
    if cfg.model.backbone_norm == "groupnorm":
        _to_groupnorm(backbone)

    result = backbone.load_state_dict(state, strict=False)
    report = {
        "loaded": len(state) - len(result.unexpected_keys),
        "dropped": len(result.unexpected_keys),
        "dropped_keys": sorted(result.unexpected_keys)[:20],
        "randomly_initialised": sorted(result.missing_keys)[:20],
        "n_randomly_initialised": len(result.missing_keys),
        "source": cfg.model.imagenet_weights or "download:IMAGENET1K_V2",
        "backbone_norm": cfg.model.backbone_norm,
    }
    if report["loaded"] == 0:
        raise WeightsUnavailableError(
            f"no pretrained tensor matched the backbone: {report}")

    # Replaced after loading, so fc.weight/fc.bias count as loaded rather than dropped.
    backbone.fc = nn.Identity()
    return backbone, report


# --- models --------------------------------------------------------------------------------

class BaselineModel(nn.Module):
    """image -> features -> 5 classes. No attribute layer."""

    def __init__(self, cfg):
        super().__init__()
        self.backbone, self.weight_load_report = build_backbone(cfg)
        self.head = nn.Linear(FEATURE_DIM, N_CLASSES)
        self.model_type = cfg.model.type

    def forward(self, x) -> ModelOutput:
        return ModelOutput(class_logits=self.head(self.backbone(x)),
                           attribute_logits=None)


class BottleneckModel(nn.Module):
    """image -> features -> 11 attribute heads -> 31 values -> 5 classes.

    Used for both `bottleneck` and `control`. The two differ only in the attribute loss
    weight, which lives in the config and never in this class - so no structural
    difference between them is possible.
    """

    def __init__(self, cfg, label_space: LabelSpace):
        super().__init__()
        self.backbone, self.weight_load_report = build_backbone(cfg)
        self.attribute_heads = nn.ModuleList(
            [nn.Linear(FEATURE_DIM, len(values))
             for values in label_space.attributes.values()])

        offsets, at = [], 0
        for column, values in label_space.attributes.items():
            offsets.append((column, at, at + len(values)))
            at += len(values)
        self.group_offsets = offsets
        self.n_attribute_values = at

        self.classifier = nn.Linear(at, N_CLASSES, bias=True)

    def attribute_logits(self, x) -> list[torch.Tensor]:
        features = self.backbone(x)
        return [head(features) for head in self.attribute_heads]

    def bottleneck_vector(self, attribute_logits, mode: str = "soft") -> torch.Tensor:
        """The 31-d vector fed to the classifier.

        soft: per-head softmax, which is what training sees.
        hard: per-head argmax one-hot. A deliberate train/eval mismatch - the classifier
        was trained on smooth probabilities and is handed simplex vertices - which
        measures how far the bottleneck commits to discrete decisions (CLAUDE.md sec 6).
        """
        if mode not in ATTRIBUTE_MODES:
            raise ValueError(
                f"unknown attribute mode {mode!r}; expected one of {list(ATTRIBUTE_MODES)}")
        parts = []
        for logits in attribute_logits:
            probs = F.softmax(logits, dim=1)
            if mode == "hard":
                probs = F.one_hot(probs.argmax(dim=1), probs.shape[1]).to(probs.dtype)
            parts.append(probs)
        return torch.cat(parts, dim=1)

    def classify(self, vector: torch.Tensor) -> torch.Tensor:
        return self.classifier(vector)

    def forward(self, x, mode: str = "soft") -> ModelOutput:
        logits = self.attribute_logits(x)
        return ModelOutput(class_logits=self.classify(self.bottleneck_vector(logits, mode)),
                           attribute_logits=logits)


def build_model(cfg, label_space: LabelSpace):
    """The one constructor. Fine-tunes the whole backbone for every type, identically
    (CLAUDE.md sec 5) - a frozen backbone would be a different protocol."""
    if cfg.model.type == "baseline":
        model = BaselineModel(cfg)
    elif cfg.model.type in ("bottleneck", "control"):
        model = BottleneckModel(cfg, label_space)
    else:
        raise ValueError(f"unknown model.type {cfg.model.type!r}")

    for parameter in model.parameters():
        parameter.requires_grad = True
    return model
