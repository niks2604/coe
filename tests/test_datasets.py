"""Dataset construction. PBC is keyed off the WBCAtt CSV; Raabin off folders."""
import csv

import pytest

from src.data.datasets import (
    DatasetError,
    PbcWbcAttDataset,
    RaabinFolderDataset,
    build_secondary_subset,
)
from src.data.domains import Domain
from src.data.label_space import LabelSpaceError

pytest.importorskip("torch")


def make_pbc(tiny_raw, mapping, label_space, split="train", **kw):
    return PbcWbcAttDataset(
        dataset_id=f"pbc_{split}",
        pbc_images=tiny_raw / "PBC_dataset_normal_DIB",
        wbcatt_dir=tiny_raw,
        mapping=mapping,
        label_space=label_space,
        image_size=16,
        **kw,
    )


# --- PBC is keyed off the CSV -----------------------------------------------------------

def test_length_equals_csv_rows(tiny_raw_loadable, mapping, label_space):
    """The fixture's train split has 4 rows once the unresolvable one is stripped."""
    ds = make_pbc(tiny_raw_loadable, mapping, label_space, split="train")
    assert len(ds) == 4


def test_unresolvable_csv_row_is_a_hard_error(tiny_raw, mapping, label_space):
    """The audit measured 10298/10298 resolving, so a failure means the tree changed.
    Dropping the row instead would silently shrink the training set."""
    with pytest.raises(DatasetError, match="GHOST_1.jpg"):
        make_pbc(tiny_raw, mapping, label_space, split="train")


def test_test_split_loads(tiny_raw, mapping, label_space):
    ds = make_pbc(tiny_raw, mapping, label_space, split="test")
    assert len(ds) == 5
    assert ds.dataset_id == "pbc_test"
    assert ds.domain is Domain.SOURCE


def test_item_is_image_class_and_eleven_attributes(tiny_raw, mapping, label_space):
    import torch
    from torchvision import transforms

    to_tensor = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor()])
    ds = make_pbc(tiny_raw, mapping, label_space, split="test", transform=to_tensor)
    image, class_idx, attrs = ds[0]
    assert isinstance(image, torch.Tensor) and image.shape == (3, 16, 16)
    assert 0 <= class_idx < 5
    assert attrs.shape == (11,)
    assert attrs.dtype == torch.long


def test_attribute_indices_are_within_each_column_vocabulary(tiny_raw, mapping, label_space):
    ds = make_pbc(tiny_raw, mapping, label_space, split="test")
    _, _, attrs = ds[0]
    for i, col in enumerate(label_space.attributes):
        assert 0 <= int(attrs[i]) < len(label_space.attributes[col]), col


def test_class_index_follows_the_label_space_not_insertion_order(tiny_raw, mapping, label_space):
    ds = make_pbc(tiny_raw, mapping, label_space, split="test")
    by_name = {ds.img_names[i]: int(ds[i][1]) for i in range(len(ds))}
    assert by_name["BNE_1.jpg"] == label_space.class_index("neutrophil")
    assert by_name["LY_2.jpg"] == label_space.class_index("lymphocyte")


def test_excluded_class_images_are_unreachable(tiny_raw, mapping, label_space):
    """platelet/erythroblast/ig are not in the CSV, so they cannot be reached at all -
    they are excluded by construction, not filtered out."""
    ds = make_pbc(tiny_raw, mapping, label_space, split="test")
    assert not any("PLATELET" in n for n in ds.img_names)


def test_attribute_value_outside_the_label_space_is_a_hard_error(
        tiny_raw, mapping, label_space):
    path = tiny_raw / "pbc_attr_v1_test.csv"
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    rows[0]["chromatin_density"] = "speckled"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(LabelSpaceError, match="speckled"):
        make_pbc(tiny_raw, mapping, label_space, split="test")


def test_whitespace_in_an_attribute_value_is_a_hard_error(tiny_raw, mapping, label_space):
    """The fixture's LY_2.jpg carries " loosely ". Silently stripping it would hide a
    data defect the audit is meant to surface."""
    with pytest.raises(LabelSpaceError):
        make_pbc(tiny_raw, mapping, label_space, split="test", strip_values=False)


# --- Raabin is keyed off folders --------------------------------------------------------

def make_raabin(tiny_raw, mapping, label_space, dataset_id="raabin_testB"):
    return RaabinFolderDataset(
        dataset_id=dataset_id,
        root=tiny_raw / "raabin_double_labeled",
        mapping=mapping,
        label_space=label_space,
        image_size=16,
    )


def test_raabin_uses_the_registry_folder_name(tiny_raw, mapping, label_space):
    """raabin_testB lives in Test-B."""
    ds = make_raabin(tiny_raw, mapping, label_space)
    assert ds.split_dir.name == "Test-B"
    assert ds.domain is Domain.TARGET


def test_no_transform_yields_a_pil_image(tiny_raw, mapping, label_space):
    """transform=None is for inspecting labels without paying for decoding; the loaders
    always supply one."""
    from PIL.Image import Image as PilImage

    ds = make_pbc(tiny_raw, mapping, label_space, split="test")
    assert isinstance(ds[0][0], PilImage)


def test_raabin_returns_no_attributes(tiny_raw, mapping, label_space):
    """There are no attribute labels for target data; None forces callers to branch
    rather than quietly feeding an ignore index into an attribute loss."""
    ds = make_raabin(tiny_raw, mapping, label_space)
    _, class_idx, attrs = ds[0]
    assert attrs is None
    assert 0 <= class_idx < 5


def test_raabin_counts_exclude_unreadable_and_dotfiles(tiny_raw, mapping, label_space):
    """Test-B holds 4 readable images plus RB_CORRUPT.jpg."""
    ds = make_raabin(tiny_raw, mapping, label_space)
    assert len(ds) == 4
    assert len(ds.unreadable) == 1


def test_raabin_unknown_class_folder_is_a_hard_error(
        tiny_raw, mapping, label_space):
    (tiny_raw / "raabin_double_labeled" / "Test-B" / "Myelocyte").mkdir()
    from conftest import write_image
    write_image(tiny_raw / "raabin_double_labeled" / "Test-B" / "Myelocyte" / "a.jpg")
    from src.data.domains import UnmappedClass
    with pytest.raises(UnmappedClass, match="Myelocyte"):
        make_raabin(tiny_raw, mapping, label_space)


def test_a_source_dataset_id_is_refused_by_the_raabin_dataset(
        tiny_raw, mapping, label_space):
    with pytest.raises(ValueError):
        make_raabin(tiny_raw, mapping, label_space, dataset_id="pbc_train")


# --- the secondary subset ----------------------------------------------------------------

def test_secondary_subset_drops_band_neutrophils(tiny_raw, mapping, label_space):
    ds = make_pbc(tiny_raw, mapping, label_space, split="test")
    sub = build_secondary_subset(ds, mapping, "pbc_test_no_band_neutrophils")
    assert len(sub) == 3
    kept = {sub.dataset.img_names[i] for i in sub.indices}
    assert kept == {"EO_2.jpg", "SNE_1.jpg", "LY_2.jpg"}


def test_secondary_subset_prefixes_come_from_the_mapping(tiny_raw, mapping, label_space):
    """Patching the mapping changes the subset: nothing is hardcoded in the loader."""
    ds = make_pbc(tiny_raw, mapping, label_space, split="test")
    mapping.secondary_subsets["pbc_test_no_band_neutrophils"][
        "drop_filename_prefixes"]["pbc:neutrophil"] = ["SNE_"]
    sub = build_secondary_subset(ds, mapping, "pbc_test_no_band_neutrophils")
    kept = {sub.dataset.img_names[i] for i in sub.indices}
    assert "SNE_1.jpg" not in kept
    assert "BNE_1.jpg" in kept


def test_secondary_subset_refuses_a_mismatched_split(tiny_raw, mapping, label_space):
    ds = make_pbc(tiny_raw, mapping, label_space, split="test")
    mapping.secondary_subsets["pbc_test_no_band_neutrophils"]["split"] = "pbc_val"
    with pytest.raises(DatasetError, match="pbc_val"):
        build_secondary_subset(ds, mapping, "pbc_test_no_band_neutrophils")


def test_unknown_secondary_subset_is_a_hard_error(tiny_raw, mapping, label_space):
    ds = make_pbc(tiny_raw, mapping, label_space, split="test")
    with pytest.raises(DatasetError, match="no_such_subset"):
        build_secondary_subset(ds, mapping, "no_such_subset")
