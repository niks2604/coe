"""The leakage guards. These are the tests that matter most in this repo."""
import pytest

from src.data.domains import (
    DATASETS,
    Domain,
    TargetDomainLeakage,
    UnmappedClass,
    assert_source_only,
    domain_of,
    load_class_mapping,
    lookup_class,
    resolve_class,
)

SOURCE_IDS = ["pbc_train", "pbc_val", "pbc_test"]
TARGET_IDS = ["raabin_train", "raabin_testA", "raabin_testB", "lisc"]


@pytest.mark.parametrize("dataset_id", SOURCE_IDS)
def test_pbc_splits_are_source(dataset_id):
    assert domain_of(dataset_id) is Domain.SOURCE


@pytest.mark.parametrize("dataset_id", TARGET_IDS)
def test_raabin_and_lisc_are_target(dataset_id):
    assert domain_of(dataset_id) is Domain.TARGET


def test_raabin_train_is_target_despite_its_name():
    """The likeliest way this project leaks: treating Raabin's Train split as training
    data because of the folder name. The registry keys off the dataset id, never off a
    path or a substring like "train"."""
    assert domain_of("raabin_train") is Domain.TARGET


def test_every_registered_dataset_has_an_explicit_domain():
    assert set(DATASETS) == set(SOURCE_IDS) | set(TARGET_IDS)
    assert all(d.domain in (Domain.SOURCE, Domain.TARGET) for d in DATASETS.values())


def test_unknown_dataset_id_is_a_hard_error():
    """A new dataset cannot arrive untagged and silently default to SOURCE."""
    with pytest.raises(KeyError):
        domain_of("raabin_testC")


@pytest.mark.parametrize("dataset_id", TARGET_IDS)
def test_assert_source_only_rejects_targets(dataset_id):
    with pytest.raises(TargetDomainLeakage):
        assert_source_only(dataset_id)


@pytest.mark.parametrize("dataset_id", SOURCE_IDS)
def test_assert_source_only_accepts_source(dataset_id):
    assert_source_only(dataset_id)


def test_assert_source_only_accepts_an_object_with_a_dataset_id():
    class FakeDataset:
        dataset_id = "raabin_testA"

    with pytest.raises(TargetDomainLeakage):
        assert_source_only(FakeDataset())


def test_assert_source_only_names_the_offending_dataset():
    with pytest.raises(TargetDomainLeakage, match="raabin_testB"):
        assert_source_only("raabin_testB")


# --- class mapping ---------------------------------------------------------------

def test_mapping_loads_and_covers_the_five_classes(class_mapping_path):
    mapping = load_class_mapping(class_mapping_path)
    assert set(mapping.classes) == {
        "neutrophil", "lymphocyte", "monocyte", "eosinophil", "basophil",
    }


def test_resolve_class_maps_pbc_and_wbcatt_and_lisc(class_mapping_path):
    mapping = load_class_mapping(class_mapping_path)
    assert resolve_class(mapping, "pbc", "neutrophil") == "neutrophil"
    assert resolve_class(mapping, "wbcatt", "Neutrophil") == "neutrophil"
    assert resolve_class(mapping, "lisc", "neut") == "neutrophil"
    assert resolve_class(mapping, "lisc", "Baso") == "basophil"


def test_resolve_class_is_case_sensitive(class_mapping_path):
    """configs/class_mapping.yaml states matching is exact and case-sensitive."""
    mapping = load_class_mapping(class_mapping_path)
    with pytest.raises(UnmappedClass):
        resolve_class(mapping, "wbcatt", "neutrophil")


def test_excluded_class_raises_rather_than_being_dropped(class_mapping_path):
    """No implicit drop: an excluded class is a distinct, explicit refusal."""
    mapping = load_class_mapping(class_mapping_path)
    assert "pbc:platelet" in mapping.excluded
    with pytest.raises(UnmappedClass):
        resolve_class(mapping, "pbc", "platelet")


def test_unmapped_class_is_a_hard_error(class_mapping_path):
    mapping = load_class_mapping(class_mapping_path)
    with pytest.raises(UnmappedClass):
        resolve_class(mapping, "pbc", "smudge_cell")


def test_raabin_classes_are_still_unmapped(class_mapping_path):
    """Raabin is mapped only after the audit is reviewed (CLAUDE.md sec 4)."""
    mapping = load_class_mapping(class_mapping_path)
    with pytest.raises(UnmappedClass):
        resolve_class(mapping, "raabin", "Neutrophil")


def test_lookup_class_reports_instead_of_raising(class_mapping_path):
    """The audit must be able to report unmapped names without crashing."""
    mapping = load_class_mapping(class_mapping_path)
    assert lookup_class(mapping, "raabin", "Neutrophil") is None
    assert lookup_class(mapping, "pbc", "basophil") == "basophil"


def test_secondary_subsets_are_source_only(class_mapping_path):
    mapping = load_class_mapping(class_mapping_path)
    assert "pbc_test_no_band_neutrophils" in mapping.secondary_subsets
    for name, spec in mapping.secondary_subsets.items():
        assert domain_of(spec["split"]) is Domain.SOURCE, name
