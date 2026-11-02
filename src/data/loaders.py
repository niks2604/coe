"""Loader factories, and the domain guards that make target data unreachable from
training.

There is no argument to make_train_loader or make_val_loader that yields a loader over
target data: the dataset id defaults to its pinned split, is checked against the
registry, and target ids raise. make_eval_loader is the only path to a target domain,
and it forces shuffle=False and the deterministic transform.

Source test and target domains go through make_eval_loader alike, so the source-domain
number and the target-domain number are produced by the same code. The reported result
is the gap between them, and a difference in how they are measured would land in that
gap (CLAUDE.md sec 1).
"""
from __future__ import annotations

from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from src.data.datasets import PbcWbcAttDataset, RaabinFolderDataset
from src.data.domains import (
    DATASETS,
    TUNING_SPLIT,
    ClassMapping,
    Domain,
    assert_source_only,
    domain_of,
)
from src.data.label_space import LabelSpace
from src.data.transforms import build_eval_transform, build_train_transform

TRAIN_SPLIT = "pbc_train"


def _collate(batch):
    """Collate (image, class, attributes) where attributes may be None.

    Target data has no attribute labels, and torch's default collate raises on None.
    Keeping None - rather than substituting an ignore index - is what forces a caller to
    branch instead of quietly feeding a sentinel into an attribute loss.
    """
    images = torch.stack([b[0] for b in batch])
    classes = torch.tensor([b[1] for b in batch], dtype=torch.long)
    if any(b[2] is None for b in batch):
        if not all(b[2] is None for b in batch):
            raise ValueError(
                "batch mixes items with and without attribute labels; a dataset is "
                "either fully attribute-labelled (source) or not at all (target)"
            )
        return images, classes, None
    return images, classes, torch.stack([b[2] for b in batch])


def _stratified_subset(dataset, n: int, seed: int) -> list[int]:
    """Pick n indices keeping every class present, reproducibly.

    An unstratified head would drop basophil and lymphocyte from a 500-image smoke test
    and so smoke-test nothing (CLAUDE.md sec 7).
    """
    import random

    by_class: dict[int, list[int]] = defaultdict(list)
    for j in dataset.indices:
        by_class[dataset.class_indices[j]].append(j)

    rng = random.Random(seed)
    for indices in by_class.values():
        rng.shuffle(indices)

    picked: list[int] = []
    classes = sorted(by_class)
    while len(picked) < n and any(by_class[c] for c in classes):
        for c in classes:
            if by_class[c] and len(picked) < n:
                picked.append(by_class[c].pop())
    return sorted(picked)


def _build_source(cfg, dataset_id: str, mapping: ClassMapping,
                  label_space: LabelSpace, transform, subset: bool):
    dataset = PbcWbcAttDataset(
        dataset_id=dataset_id,
        pbc_images=cfg.paths.pbc_images,
        wbcatt_dir=cfg.paths.wbcatt_dir,
        mapping=mapping,
        label_space=label_space,
        image_size=cfg.data.image_size,
        transform=transform,
    )
    if subset and cfg.data.subset_n_images is not None:
        dataset.indices = _stratified_subset(dataset, cfg.data.subset_n_images, cfg.seed)
    return dataset


def _require(dataset_id: str, expected: str) -> None:
    if dataset_id not in DATASETS:
        raise ValueError(f"unknown dataset id {dataset_id!r}; known: {sorted(DATASETS)}")
    assert_source_only(dataset_id)          # raises TargetDomainLeakage for targets
    if dataset_id != expected:
        raise ValueError(
            f"{dataset_id!r} may not be used here; this loader is pinned to "
            f"{expected!r}. pbc_test is held out for the source-domain report."
        )


def make_train_loader(cfg, mapping: ClassMapping, label_space: LabelSpace,
                      dataset_id: str = TRAIN_SPLIT) -> DataLoader:
    """pbc_train only. Shuffled and augmented."""
    _require(dataset_id, TRAIN_SPLIT)
    dataset = _build_source(cfg, dataset_id, mapping, label_space,
                            build_train_transform(cfg), subset=True)
    dataset.augmented = True
    return DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True,
                      drop_last=False, collate_fn=_collate)


def make_val_loader(cfg, mapping: ClassMapping, label_space: LabelSpace,
                    dataset_id: str = TUNING_SPLIT) -> DataLoader:
    """pbc_val only - the one split any tuning decision may read. No augmentation."""
    _require(dataset_id, TUNING_SPLIT)
    dataset = _build_source(cfg, dataset_id, mapping, label_space,
                            build_eval_transform(cfg), subset=True)
    return DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False,
                      collate_fn=_collate)


def make_eval_loader(cfg, dataset_id: str, mapping: ClassMapping,
                     label_space: LabelSpace) -> DataLoader | None:
    """The only path to target data, and the same path used for pbc_test.

    Never shuffles, never augments, and never applies subset_n_images - shrinking a
    target domain would change what the reported gap is measured over. Returns None if
    the dataset is not on disk, so an absent domain is skipped rather than fatal.
    """
    if dataset_id not in DATASETS:
        raise ValueError(f"unknown dataset id {dataset_id!r}; known: {sorted(DATASETS)}")
    transform = build_eval_transform(cfg)

    if domain_of(dataset_id) is Domain.SOURCE:
        dataset = _build_source(cfg, dataset_id, mapping, label_space, transform,
                                subset=False)
    else:
        corpus = DATASETS[dataset_id].corpus
        if corpus != "raabin":
            # LISC was never obtained and is out of scope (CLAUDE.md sec 2); no loader
            # exists for it. Any future target gets its own dataset class here.
            return None
        root = cfg.paths.raabin_root
        if not root.is_dir():
            return None
        dataset = RaabinFolderDataset(
            dataset_id=dataset_id, root=root, mapping=mapping,
            label_space=label_space, image_size=cfg.data.image_size,
            transform=transform,
        )

    return DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False,
                      collate_fn=_collate)
