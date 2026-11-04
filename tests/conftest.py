"""Synthetic fixtures. No test touches real data; everything here is built in tmp_path.

The tree deliberately contains the defects the audit must find: a corrupt image, a
filename duplicated across two class folders, a PBC image missing from the WBCAtt CSVs,
a CSV row pointing at a file that does not exist, and an attribute value with stray
whitespace. Tests assert on those, so do not "clean up" the fixture.
"""
import csv
from pathlib import Path

import pytest
import yaml
from PIL import Image

ATTR_COLUMNS = [
    "cell_size",
    "cell_shape",
    "nucleus_shape",
    "nuclear_cytoplasmic_ratio",
    "chromatin_density",
    "cytoplasm_vacuole",
    "cytoplasm_texture",
    "cytoplasm_colour",
    "granule_type",
    "granule_colour",
    "granularity",
]
CSV_COLUMNS = ATTR_COLUMNS + ["img_name", "label", "path"]

# One plausible value per attribute, so a fixture row is a complete row.
ATTR_VALUES = {
    "cell_size": "big",
    "cell_shape": "round",
    "nucleus_shape": "segmented-bilobed",
    "nuclear_cytoplasmic_ratio": "low",
    "chromatin_density": "densely",
    "cytoplasm_vacuole": "no",
    "cytoplasm_texture": "clear",
    "cytoplasm_colour": "light blue",
    "granule_type": "round",
    "granule_colour": "pink",
    "granularity": "yes",
}


def write_image(path: Path, size=(8, 8), mode="RGB", colour=(120, 90, 160)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, colour if mode == "RGB" else 120).save(path)
    return path


def write_corrupt(path: Path, data=b"this is not a JPEG"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.fixture
def tiny_raw(tmp_path) -> Path:
    """A miniature data/raw/ tree: PBC + WBCAtt CSVs + Raabin. LISC is absent."""
    raw = tmp_path / "data" / "raw"
    pbc = raw / "PBC_dataset_normal_DIB"

    rows = []

    def add(cls, fname, *, in_csv=True, split="train", **overrides):
        write_image(pbc / cls / fname)
        if in_csv:
            row = dict(ATTR_VALUES, img_name=fname, label=cls.capitalize(),
                       path=f"PBC_dataset_normal_DIB/{cls}/{fname}")
            row.update(overrides)
            rows.append((split, row))

    add("basophil", "BA_1.jpg", split="train")
    add("basophil", "BA_2.jpg", split="val")
    add("eosinophil", "EO_1.jpg", split="train")
    add("eosinophil", "EO_2.jpg", split="test")
    add("lymphocyte", "LY_1.jpg", split="train")
    add("monocyte", "MO_1.jpg", split="train")
    # Neutrophils: band, segmented and unsubtyped, all labelled Neutrophil by WBCAtt.
    add("neutrophil", "BNE_1.jpg", split="test")
    add("neutrophil", "SNE_1.jpg", split="test")
    add("neutrophil", "NEUTROPHIL_1.jpg", split="test")
    # Stray whitespace in an attribute value: the audit must surface this.
    add("lymphocyte", "LY_2.jpg", split="test", chromatin_density=" loosely ")
    # On disk but in no CSV -> must be reported as PBC-side-only.
    add("monocyte", "MO_ONLY_ON_DISK.jpg", in_csv=False)
    # Excluded class, present in the archive, absent from WBCAtt.
    add("platelet", "PLATELET_1.jpg", in_csv=False)
    # A filename duplicated across two class folders.
    add("eosinophil", "DUPE_1.jpg", in_csv=False)
    add("basophil", "DUPE_1.jpg", in_csv=False)
    # A corrupt file with an image extension.
    write_corrupt(pbc / "basophil" / "BA_CORRUPT.jpg")
    # macOS metadata masquerading as a .jpg, as in the real archive.
    write_corrupt(pbc / "neutrophil" / ".DS_169665.jpg", b"\x00\x01Bud1")

    # A CSV row whose file does not exist -> must be reported as CSV-side-only.
    rows.append(("train", dict(ATTR_VALUES, img_name="GHOST_1.jpg", label="Basophil",
                               path="PBC_dataset_normal_DIB/basophil/GHOST_1.jpg")))

    for split in ("train", "val", "test"):
        out = raw / f"pbc_attr_v1_{split}.csv"
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            w.writeheader()
            for s, row in rows:
                if s == split:
                    w.writerow(row)

    # Raabin: three splits, class folders. TestB holds only two classes.
    raabin = raw / "raabin_double_labeled"
    for split, classes in (
        ("Train", ["Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"]),
        ("TestA", ["Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"]),
        # The real archive unpacks to Test-B, hyphenated, unlike Train and TestA.
        ("Test-B", ["Neutrophil", "Lymphocyte"]),
    ):
        for cls in classes:
            for i in range(2):
                write_image(raabin / split / cls / f"{cls}_{i}.jpg", size=(12, 10))
    write_corrupt(raabin / "Test-B" / "Neutrophil" / "RB_CORRUPT.jpg")

    # data/raw/LISC deliberately absent.
    return raw


def _paths(raw: Path) -> dict:
    return {
        "local": {
            "pbc_images": str(raw / "PBC_dataset_normal_DIB"),
            "wbcatt_dir": str(raw),
            "raabin_root": str(raw / "raabin_double_labeled"),
            "lisc_root": str(raw / "LISC"),
            "results_dir": str(raw.parent.parent / "results"),
        },
        "kaggle": {
            "pbc_images": "TODO_FILL",
            "wbcatt_dir": "TODO_FILL",
            "raabin_root": "TODO_FILL",
            "lisc_root": "TODO_FILL",
            "results_dir": "/kaggle/working/results",
        },
    }


@pytest.fixture(scope="session", autouse=True)
def repo_ledger_snapshot():
    """Guard: the committed repo-root ledger must survive the test run untouched.

    Every test writes to a temp ledger. If one ever defaulted to LEDGER_PATH it would
    silently burn a real entry and make a later genuine evaluation refuse itself, so the
    bytes are compared at session start and session end.
    """
    from src.ledger import LEDGER_PATH

    before = LEDGER_PATH.read_bytes() if LEDGER_PATH.exists() else None
    yield before
    after = LEDGER_PATH.read_bytes() if LEDGER_PATH.exists() else None
    assert after == before, (
        f"{LEDGER_PATH} was modified by the test run. Tests must always pass an "
        f"explicit temp ledger path."
    )


@pytest.fixture
def mapping(class_mapping_path):
    from src.data.domains import load_class_mapping
    return load_class_mapping(class_mapping_path)


@pytest.fixture
def label_space_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "label_space.yaml"


@pytest.fixture
def label_space(label_space_path):
    from src.data.label_space import load_label_space
    return load_label_space(label_space_path)


@pytest.fixture
def tiny_raw_loadable(tiny_raw) -> Path:
    """tiny_raw with the deliberately unresolvable CSV row removed.

    GHOST_1.jpg exists so the audit can report a CSV row with no file, but it also makes
    PbcWbcAttDataset refuse to build pbc_train - which is correct behaviour and is
    tested directly. Loader tests need a tree that does build.
    """
    for split in ("train", "val", "test"):
        path = tiny_raw / f"pbc_attr_v1_{split}.csv"
        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
        keep = [r for r in rows if not r["img_name"].startswith("GHOST_")]
        if len(keep) == len(rows):
            continue
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            w.writeheader()
            w.writerows(keep)
    return tiny_raw


@pytest.fixture
def class_mapping_path() -> Path:
    """The real mapping file; it is the single source of truth, so tests use it."""
    return Path(__file__).resolve().parents[1] / "configs" / "class_mapping.yaml"


@pytest.fixture
def audit_config(tmp_path, tiny_raw, class_mapping_path):
    cfg = {
        "env": "local",
        "seed": 0,
        "paths": _paths(tiny_raw),
        "audit": {
            "class_mapping": str(class_mapping_path),
            "verify_images": True,
            "tree_max_depth": 3,
        },
    }
    path = tmp_path / "audit.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def experiment_dict(raw: Path, mapping: Path, **overrides) -> dict:
    cfg = {
        "run_id": "bottleneck",
        "env": "local",
        "seed": 0,
        "paths": _paths(raw),
        "data": {
            "class_mapping": str(mapping),
            "label_space": str(Path(__file__).resolve().parents[1] / "configs"
                               / "label_space.yaml"),
            "source": "pbc_wbcatt",
            "image_size": 224,
            "subset_n_images": None,
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "model": {"type": "bottleneck", "backbone": "resnet50", "backbone_norm": "batchnorm"},
        "bottleneck": {"attribute_loss_weight": 1.0},
        "train": {
            "epochs": 30,
            "batch_size": 32,
            "lr": 0.0003,
            "optimizer": "adamw",
            "weight_decay": 0.0001,
            "class_weighting": "none",
            "augmentation": {
                "hflip": 0.5,
                "vflip": 0.5,
                "rotation_degrees": 180,
                "color_jitter": {"brightness": 0.1, "contrast": 0.1,
                                 "saturation": 0.1, "hue": 0.02},
            },
        },
        "selection": {"rule": "final_epoch", "report_metric": "val_macro_f1",
                      "divergence_guard": "non_finite_loss"},
        "eval": {
            "domains": ["pbc_test", "raabin_train", "raabin_testA", "raabin_testB", "lisc"],
            "attribute_modes": ["soft", "hard"],
            "secondary_subsets": ["pbc_test_no_band_neutrophils"],
            "abstention": {"threshold_split": "pbc_val", "coverage_grid": [0.5, 0.7, 0.9, 1.0]},
            "ece_bins": 15,
            "macro_f1_over": "present_classes",
        },
        "bootstrap": {
            "enabled": True,
            "n": 2000,
            "ci": 0.95,
            "stratify_by": "class",
            "resample_domains_independently": True,
        },
    }
    for dotted, value in overrides.items():
        node = cfg
        parts = dotted.split(".")
        for p in parts[:-1]:
            node = node[p]
        if value is _DELETE:
            node.pop(parts[-1], None)
        else:
            node[parts[-1]] = value
    return cfg


class _Delete:
    def __repr__(self):
        return "<DELETE>"


_DELETE = _Delete()
DELETE = _DELETE


@pytest.fixture
def experiment_config(tmp_path, tiny_raw, class_mapping_path):
    """Returns write(**overrides) -> path, with dotted-key overrides."""
    counter = {"n": 0}

    def write(**overrides):
        counter["n"] += 1
        cfg = experiment_dict(tiny_raw, class_mapping_path, **overrides)
        path = tmp_path / f"exp_{counter['n']}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        return path

    return write
