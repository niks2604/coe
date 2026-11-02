"""Dataset classes. PBC is keyed off the WBCAtt CSV; Raabin off folders.

PBC is deliberately NOT built from a folder listing. The source domain is the 10,298
images that carry WBCAtt labels (CLAUDE.md sec 2), and the archive holds 17,092 - the
6,794 extra are erythroblasts, immature granulocytes and platelets. Reading the CSV
makes those unreachable by construction rather than filtered out by a predicate someone
could later weaken.

Every dataset carries .dataset_id and .domain, taken from the registry at construction,
so no call site can mislabel one.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.domains import (
    DATASETS,
    ClassMapping,
    Domain,
    domain_of,
    filename_prefix,
    resolve_class,
    target_folder,
)
from src.data.label_space import LabelSpace

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class DatasetError(Exception):
    """The data on disk does not match what the dataset expects."""


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def _candidate_paths(row: dict, pbc_images: Path) -> list[Path]:
    """Same resolution order the audit uses; it measured 10298/10298 via the first."""
    rel = (row.get("path") or "").lstrip("/")
    out = []
    if rel:
        out.append(pbc_images.parent / rel)
        out.append(pbc_images / rel)
        parts = Path(rel).parts
        if parts and parts[0] == pbc_images.name:
            out.append(pbc_images / Path(*parts[1:]))
    name, label = row.get("img_name") or "", (row.get("label") or "").lower()
    if name and label:
        out.append(pbc_images / label / name)
    return out


class PbcWbcAttDataset(Dataset):
    """The source domain: one WBCAtt split, with its 11 attribute labels."""

    def __init__(self, dataset_id: str, pbc_images: Path, wbcatt_dir: Path,
                 mapping: ClassMapping, label_space: LabelSpace, image_size: int,
                 transform=None, strip_values: bool = True, indices=None):
        spec = DATASETS.get(dataset_id)
        if spec is None or spec.csv_split is None:
            raise ValueError(
                f"{dataset_id!r} is not a CSV-backed source dataset; "
                f"source ids are {[k for k, v in DATASETS.items() if v.csv_split]}"
            )
        self.dataset_id = dataset_id
        self.domain = domain_of(dataset_id)
        self.label_space = label_space
        self.transform = transform
        self.augmented = False
        self.image_size = image_size

        csv_path = Path(wbcatt_dir) / f"pbc_attr_v1_{spec.csv_split}.csv"
        if not csv_path.exists():
            raise DatasetError(f"{dataset_id}: missing {csv_path}")
        with csv_path.open(newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))

        pbc_images = Path(pbc_images)
        self.paths: list[Path] = []
        self.img_names: list[str] = []
        self.class_indices: list[int] = []
        self.attribute_indices: list[list[int]] = []
        observed: dict[str, set[str]] = {c: set() for c in label_space.attributes}
        unresolved: list[str] = []

        for row in rows:
            path = next((c for c in _candidate_paths(row, pbc_images) if c.exists()), None)
            if path is None:
                unresolved.append(row.get("img_name") or row.get("path") or "<no name>")
                continue
            for column in label_space.attributes:
                raw = row.get(column)
                if raw is None:
                    raise DatasetError(f"{csv_path}: missing column {column!r}")
                observed[column].add(raw.strip() if strip_values else raw)
            self.paths.append(path)
            self.img_names.append(row.get("img_name") or path.name)
            self.class_indices.append(
                label_space.class_index(resolve_class(mapping, "wbcatt", row["label"])))
            self.attribute_indices.append([
                label_space.attribute_index(
                    column,
                    (row[column].strip() if strip_values else row[column]),
                )
                for column in label_space.attributes
            ])

        if unresolved:
            # The audit measured every row resolving, so a miss means the tree changed.
            # Dropping the row would silently shrink the split instead.
            raise DatasetError(
                f"{dataset_id}: {len(unresolved)} CSV row(s) could not be resolved to a "
                f"file under {pbc_images}: {unresolved[:10]}"
            )
        label_space.validate_observed(observed)
        self.indices = list(range(len(self.paths))) if indices is None else list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        j = self.indices[i]
        image = _load_rgb(self.paths[j])
        if self.transform is not None:
            image = self.transform(image)
        return (image,
                self.class_indices[j],
                torch.tensor(self.attribute_indices[j], dtype=torch.long))

    def classes_of(self, indices=None) -> list[int]:
        return [self.class_indices[j] for j in (self.indices if indices is None else indices)]


class RaabinFolderDataset(Dataset):
    """A target domain: one Raabin split, class per folder, no attribute labels."""

    def __init__(self, dataset_id: str, root: Path, mapping: ClassMapping,
                 label_space: LabelSpace, image_size: int, transform=None):
        folder = target_folder(dataset_id)   # raises for source ids
        self.dataset_id = dataset_id
        self.domain = domain_of(dataset_id)
        self.transform = transform
        self.augmented = False
        self.image_size = image_size

        self.split_dir = Path(root) / folder
        if not self.split_dir.is_dir():
            raise DatasetError(
                f"{dataset_id!r} expects {self.split_dir}, which does not exist")

        self.paths: list[Path] = []
        self.class_indices: list[int] = []
        self.unreadable: list[str] = []
        for class_dir in sorted(p for p in self.split_dir.iterdir() if p.is_dir()):
            # Raises for a folder name the mapping does not list: no implicit drop.
            class_index = label_space.class_index(
                resolve_class(mapping, DATASETS[dataset_id].corpus, class_dir.name))
            for path in sorted(class_dir.rglob("*")):
                if (not path.is_file() or path.name.startswith(".")
                        or path.suffix.lower() not in IMAGE_EXTENSIONS):
                    continue
                try:
                    # Readability only. Nothing derived from the pixels is kept
                    # (CLAUDE.md sec 3).
                    _load_rgb(path)
                except Exception as exc:
                    self.unreadable.append(f"{path.relative_to(self.split_dir)}: {exc}")
                    continue
                self.paths.append(path)
                self.class_indices.append(class_index)

        if not self.paths:
            raise DatasetError(
                f"{dataset_id!r} resolved to {self.split_dir} but found no readable "
                f"images. An empty split would report a meaningless number."
            )
        self.indices = list(range(len(self.paths)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        j = self.indices[i]
        image = _load_rgb(self.paths[j])
        if self.transform is not None:
            image = self.transform(image)
        # No attribute labels exist for target data. None rather than an ignore index,
        # so a caller must branch instead of feeding a sentinel into an attribute loss.
        return image, self.class_indices[j], None

    def classes_of(self, indices=None) -> list[int]:
        return [self.class_indices[j] for j in (self.indices if indices is None else indices)]


@dataclass
class SubsetView:
    """A filtered view of a dataset, keeping the parent reachable for reporting."""

    dataset: Dataset
    indices: list[int]
    name: str

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        return self.dataset[self.indices[i]]

    @property
    def dataset_id(self) -> str:
        return f"{self.dataset.dataset_id}:{self.name}"

    @property
    def domain(self) -> Domain:
        return self.dataset.domain

    @property
    def augmented(self) -> bool:
        return self.dataset.augmented


def build_secondary_subset(dataset: PbcWbcAttDataset, mapping: ClassMapping,
                           name: str) -> SubsetView:
    """Build a secondary subset defined in configs/class_mapping.yaml.

    The drop rules live in the mapping, so the audit's measurement and this view cannot
    disagree about what the subset is.
    """
    spec = mapping.secondary_subsets.get(name)
    if spec is None:
        raise DatasetError(
            f"{name!r} is not defined under secondary_subsets in the class mapping; "
            f"defined: {sorted(mapping.secondary_subsets)}"
        )
    if spec.get("split") != dataset.dataset_id:
        raise DatasetError(
            f"{name!r} is defined over split {spec.get('split')!r} but was given "
            f"{dataset.dataset_id!r}"
        )

    # Keys are "<corpus>:<raw class name>"; compare on the resolved class so the key's
    # corpus does not have to match the dataset's.
    drop_by_class: dict[int, list[str]] = {}
    for key, prefixes in (spec.get("drop_filename_prefixes") or {}).items():
        corpus, raw = key.split(":", 1)
        class_index = dataset.label_space.class_index(resolve_class(mapping, corpus, raw))
        drop_by_class.setdefault(class_index, []).extend(prefixes)

    keep = [j for j in dataset.indices
            if filename_prefix(dataset.img_names[j])
            not in drop_by_class.get(dataset.class_indices[j], [])]
    return SubsetView(dataset=dataset, indices=keep, name=name)
