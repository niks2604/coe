"""Dataset registry, domain tags, and the class mapping.

This module is the single source of truth for which datasets are SOURCE and which are
TARGET (CLAUDE.md sec 3). Nothing infers a domain from a path, a folder name or a
caller-supplied argument: a dataset id is looked up here or it is an error.

The trap this guards against: Raabin's own splits are named Train/TestA/TestB, so
`raabin_train` reads like training data and is not. Every Raabin split is TARGET.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path

import yaml

FIVE_CLASSES = ("neutrophil", "lymphocyte", "monocyte", "eosinophil", "basophil")

# The one split any tuning decision may read (CLAUDE.md sec 3).
TUNING_SPLIT = "pbc_val"


class Domain(enum.Enum):
    SOURCE = "source"
    TARGET = "target"


class TargetDomainLeakage(Exception):
    """Raised when target-domain data would reach a training or tuning decision."""


class UnmappedClass(Exception):
    """Raised for a class name that is neither mapped nor explicitly excluded."""


class DatasetLayoutError(Exception):
    """Raised when the folders on disk do not match the registered datasets."""


@dataclass(frozen=True)
class DatasetSpec:
    """A dataset id and where its data actually lives.

    `folder` is the split's directory name on disk, declared rather than derived: the
    Raabin archive unpacks raabin_testB to "Test-B", so lowercasing or stripping the id
    would point at a directory that does not exist and yield an empty split.
    `csv_split` is the equivalent for source datasets, whose rows come from
    pbc_attr_v1_<csv_split>.csv rather than from a folder listing.
    """

    dataset_id: str
    domain: Domain
    corpus: str  # which mapping namespace its raw class names live in
    folder: str | None = None       # target datasets: directory name under the root
    csv_split: str | None = None    # source datasets: WBCAtt CSV suffix
    note: str = ""


DATASETS: dict[str, DatasetSpec] = {
    "pbc_train": DatasetSpec("pbc_train", Domain.SOURCE, "wbcatt", csv_split="train"),
    "pbc_val": DatasetSpec("pbc_val", Domain.SOURCE, "wbcatt", csv_split="val"),
    "pbc_test": DatasetSpec("pbc_test", Domain.SOURCE, "wbcatt", csv_split="test"),
    "raabin_train": DatasetSpec(
        "raabin_train", Domain.TARGET, "raabin", folder="Train",
        note="Raabin's own split is named Train. It is TARGET data and is never trained on.",
    ),
    "raabin_testA": DatasetSpec("raabin_testA", Domain.TARGET, "raabin", folder="TestA"),
    "raabin_testB": DatasetSpec(
        "raabin_testB", Domain.TARGET, "raabin", folder="Test-B",
        note="Different camera/microscope; holds only neutrophil and lymphocyte. The "
             "folder is hyphenated, unlike Train and TestA.",
    ),
    "lisc": DatasetSpec("lisc", Domain.TARGET, "lisc", folder="."),
}

SOURCE_IDS = tuple(k for k, v in DATASETS.items() if v.domain is Domain.SOURCE)
TARGET_IDS = tuple(k for k, v in DATASETS.items() if v.domain is Domain.TARGET)


def domain_of(dataset_id: str) -> Domain:
    """Domain for a dataset id. Unknown ids raise rather than defaulting to SOURCE."""
    try:
        return DATASETS[dataset_id].domain
    except KeyError:
        raise KeyError(
            f"unknown dataset id {dataset_id!r}; register it in src/data/domains.py "
            f"with an explicit domain. Known ids: {sorted(DATASETS)}"
        ) from None


def _dataset_id_of(obj) -> str:
    return obj if isinstance(obj, str) else getattr(obj, "dataset_id")


def assert_source_only(obj) -> None:
    """Gate for anything that trains, validates, selects or tunes.

    Accepts a dataset id or any object carrying `.dataset_id`. Call this at the top of
    every train/val loader factory, early-stopping check, checkpoint selection, and
    threshold selection.
    """
    dataset_id = _dataset_id_of(obj)
    if domain_of(dataset_id) is Domain.TARGET:
        raise TargetDomainLeakage(
            f"{dataset_id!r} is TARGET data and must not reach training, validation, "
            f"model selection or tuning (CLAUDE.md sec 3). Target domains are touched "
            f"once, at final evaluation, with the checkpoint already chosen."
        )


def target_folder(dataset_id: str) -> str:
    """Directory name on disk for a target dataset."""
    spec = DATASETS[dataset_id] if dataset_id in DATASETS else None
    if spec is None:
        raise KeyError(f"unknown dataset id {dataset_id!r}")
    if spec.domain is not Domain.TARGET or spec.folder is None:
        raise ValueError(
            f"{dataset_id!r} is not a folder-backed target dataset; source datasets are "
            f"keyed off their WBCAtt CSV (csv_split), not a folder listing."
        )
    return spec.folder


def resolve_target_folders(root: Path, corpus: str) -> dict[str, Path]:
    """Map each registered target dataset of `corpus` to its directory under `root`.

    Both directions are checked, because both failure modes are silent:
    a registered split whose folder is missing or empty would evaluate on nothing, and
    a folder nobody registered would simply never be evaluated. Either is a hard error.
    """
    root = Path(root)
    if not root.is_dir():
        raise DatasetLayoutError(f"not a directory: {root}")

    registered = {spec.folder: spec.dataset_id for spec in DATASETS.values()
                  if spec.domain is Domain.TARGET and spec.corpus == corpus
                  and spec.folder not in (None, ".")}

    # Every problem is collected before raising: reporting only the first one makes
    # fixing a layout a game of whack-a-mole, and which comes first is an accident of
    # sort order ("Test-B" sorts before "TestA").
    found: dict[str, Path] = {}
    missing: list[str] = []
    empty: list[str] = []
    for folder, dataset_id in sorted(registered.items()):
        path = root / folder
        if not path.is_dir():
            missing.append(f"{dataset_id} -> {folder}/")
        elif not any(p.is_file() for p in path.rglob("*")):
            empty.append(f"{dataset_id} -> {folder}/")
        else:
            found[dataset_id] = path

    present = sorted(p.name for p in root.iterdir() if p.is_dir())
    unregistered = [name for name in present if name not in registered]

    problems: list[str] = []
    if missing:
        problems.append(f"registered but no such folder: {missing}")
    if empty:
        problems.append(
            f"registered but the folder holds no files: {empty} - an empty split would "
            f"evaluate on nothing and report a meaningless number"
        )
    if unregistered:
        problems.append(
            f"folder(s) matching no registered dataset id: {unregistered} - register "
            f"them in src/data/domains.py with an explicit domain, or they will never "
            f"be evaluated"
        )
    if problems:
        raise DatasetLayoutError(
            f"{root} does not match the registered {corpus} datasets. "
            + "; ".join(problems)
            + f". Folders present: {present}. Registered: {sorted(registered)}"
        )
    return found


def assert_tuning_split(dataset_id: str) -> None:
    """All tuning happens on pbc_val - not on pbc_test, not on any target."""
    assert_source_only(dataset_id)
    if dataset_id != TUNING_SPLIT:
        raise ValueError(
            f"{dataset_id!r} may not drive a tuning decision; use {TUNING_SPLIT!r}. "
            f"pbc_test is held out for the source-domain report."
        )


# --- class mapping -------------------------------------------------------------------

@dataclass
class ClassMapping:
    classes: dict[str, list[str]]           # class -> ["corpus:raw name", ...]
    excluded: dict[str, str]                # "corpus:raw name" -> reason
    excluded_files: dict[str, str] = field(default_factory=dict)
    secondary_subsets: dict[str, dict] = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @property
    def by_key(self) -> dict[str, str]:
        return {key: cls for cls, keys in self.classes.items() for key in keys}


def load_class_mapping(path: str | Path) -> ClassMapping:
    path = Path(path)
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    missing = [c for c in FIVE_CLASSES if c not in doc]
    if missing:
        raise ValueError(f"{path}: missing class keys {missing}")

    classes = {c: list(doc[c] or []) for c in FIVE_CLASSES}
    mapping = ClassMapping(
        classes=classes,
        excluded=dict(doc.get("excluded") or {}),
        excluded_files=dict(doc.get("excluded_files") or {}),
        secondary_subsets=dict(doc.get("secondary_subsets") or {}),
        raw=doc,
    )

    # A key listed twice would make resolution order-dependent.
    seen: dict[str, str] = {}
    for cls, keys in classes.items():
        for key in keys:
            if key in seen:
                raise ValueError(f"{path}: {key!r} listed under both {seen[key]} and {cls}")
            seen[key] = cls
    for key in mapping.excluded:
        if key in seen:
            raise ValueError(f"{path}: {key!r} is both mapped to {seen[key]} and excluded")
    return mapping


def resolve_class(mapping: ClassMapping, corpus: str, raw_name: str) -> str:
    """Raw class name -> one of the five classes. Exact, case-sensitive.

    Raises UnmappedClass for anything not listed, including explicitly excluded names:
    there is no implicit drop (CLAUDE.md sec 4). Callers that need to survive unknown
    names - the audit - use lookup_class instead.
    """
    key = f"{corpus}:{raw_name}"
    cls = mapping.by_key.get(key)
    if cls is not None:
        return cls
    if key in mapping.excluded:
        raise UnmappedClass(
            f"{key!r} is excluded from the label space ({mapping.excluded[key]}); "
            f"filter it out before resolving classes."
        )
    raise UnmappedClass(
        f"{key!r} is not in the class mapping. Add it under one of {list(FIVE_CLASSES)} "
        f"or under `excluded` with a reason in configs/class_mapping.yaml."
    )


def lookup_class(mapping: ClassMapping, corpus: str, raw_name: str) -> str | None:
    """Non-raising variant for reporting. None means unmapped or excluded."""
    return mapping.by_key.get(f"{corpus}:{raw_name}")
