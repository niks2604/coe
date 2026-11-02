"""Loss composition and class weighting.

    baseline:    CE(class_logits, y, weight=w)
    bottleneck:  CE(class_logits, y, weight=w) + lambda * sum_i CE(attr_logits_i, a_i)
    control:     the same expression with lambda = 0

The attribute term is a SUM over the 11 heads, not a mean. Mean would silently rescale
lambda by 11x, so lambda = 1 would not mean what CLAUDE.md sec 5 says it means.

Class weighting enters the class term only (CLAUDE.md sec 6). Imbalanced attribute
vocabularies are handled in reporting instead, via per-attribute balanced accuracy.
"""
from __future__ import annotations

from collections import Counter

import torch
import torch.nn.functional as F

from src.data.domains import Domain, assert_source_only, domain_of
from src.data.label_space import LabelSpace

TRAIN_SPLIT = "pbc_train"


def compute_loss(output, targets, cfg, class_weights=None):
    """Returns (total, parts).

    parts["attribute"] is the UNSCALED sum, so lambda's effect is visible in the record:
    total = class + lambda * attribute. When lambda is 0 - the control - the attribute
    term is not computed at all and is reported as 0.
    """
    y, attribute_targets = targets
    class_loss = F.cross_entropy(output.class_logits, y, weight=class_weights)

    lam = 0.0 if cfg.bottleneck is None else float(cfg.bottleneck.attribute_loss_weight)
    if output.attribute_logits is None or lam == 0.0:
        attribute_loss = torch.zeros((), dtype=class_loss.dtype,
                                     device=class_loss.device)
        total = class_loss
    else:
        attribute_loss = torch.stack([
            F.cross_entropy(logits, attribute_targets[:, i])
            for i, logits in enumerate(output.attribute_logits)
        ]).sum()
        total = class_loss + lam * attribute_loss

    return total, {"class": class_loss, "attribute": attribute_loss, "total": total}


def class_weights_from(dataset, cfg, label_space: LabelSpace):
    """Class weights from pbc_train frequencies, or None.

    pbc_train only. pbc_val frequencies are a tuning-set statistic and pbc_test's are
    held out, so neither may be read here even though both are source (CLAUDE.md sec 3).
    """
    assert_source_only(dataset)          # TargetDomainLeakage for any target
    dataset_id = getattr(dataset, "dataset_id", dataset)
    if dataset_id != TRAIN_SPLIT:
        raise ValueError(
            f"class weights may only be computed from {TRAIN_SPLIT!r}, not "
            f"{dataset_id!r}. pbc_val frequencies are a tuning-set statistic and "
            f"pbc_test is held out for the source-domain report."
        )

    mode = cfg.train.class_weighting
    if mode == "none":
        return None

    counts = Counter(dataset.class_indices[j] for j in dataset.indices)
    n_classes = len(label_space.classes)
    total = sum(counts.values())
    present = [c for c in range(n_classes) if counts.get(c, 0) > 0]
    if not present:
        raise ValueError(f"{dataset_id}: no labelled samples to weight")

    if mode == "inverse_frequency":
        raw = {c: total / (len(present) * counts[c]) for c in present}
    elif mode == "effective_number":
        # Needs a beta, which is a hyperparameter and so may not be hardcoded here
        # (CLAUDE.md sec 5). Declare train.class_weighting_beta before using this.
        raise NotImplementedError(
            "class_weighting: 'effective_number' needs a beta, which is a "
            "hyperparameter and must be declared in the config first. The tuning grid "
            "uses 'none' and 'inverse_frequency' only."
        )
    else:
        raise ValueError(f"unknown class_weighting {mode!r}")

    # Normalised to mean 1 over present classes, so the loss scale stays comparable to
    # the unweighted run. A class absent from pbc_train gets 0: it contributes no term.
    mean = sum(raw.values()) / len(raw)
    weights = torch.zeros(n_classes)
    for c, w in raw.items():
        weights[c] = w / mean
    return weights
