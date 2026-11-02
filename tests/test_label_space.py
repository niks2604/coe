"""The committed index orderings. These decide what class index 0 means, so they are
checked against the data rather than built from it."""
import pytest
import yaml

from src.data.label_space import LabelSpaceError, load_label_space

ATTR_COLUMNS = [
    "cell_size", "cell_shape", "nucleus_shape", "nuclear_cytoplasmic_ratio",
    "chromatin_density", "cytoplasm_vacuole", "cytoplasm_texture", "cytoplasm_colour",
    "granule_type", "granule_colour", "granularity",
]


def test_committed_file_loads(label_space_path):
    ls = load_label_space(label_space_path)
    assert ls.classes == ["neutrophil", "lymphocyte", "monocyte", "eosinophil", "basophil"]
    assert list(ls.attributes) == ATTR_COLUMNS


def test_class_order_matches_claude_md_section_2(label_space_path):
    """Not frequency order, not alphabetical: the differential as sec 2 writes it."""
    ls = load_label_space(label_space_path)
    assert ls.classes[0] == "neutrophil"
    assert ls.class_index("basophil") == 4


def test_thirty_one_attribute_values(label_space_path):
    """The control's 31 outputs and the audit's measured total must agree."""
    ls = load_label_space(label_space_path)
    assert ls.n_attribute_values == 31
    assert ls.n_attributes == 11


def test_attribute_index_is_positional(label_space_path):
    ls = load_label_space(label_space_path)
    assert ls.attribute_index("nucleus_shape", "segmented-bilobed") == 0
    assert ls.attribute_index("granularity", "no") == 1


def test_yes_no_values_are_strings_not_booleans(label_space_path):
    """Unquoted in YAML, "yes"/"no" parse as booleans and then match nothing in the
    CSV, silently emptying cytoplasm_vacuole and granularity."""
    ls = load_label_space(label_space_path)
    assert ls.attributes["granularity"] == ["yes", "no"]
    assert ls.attributes["cytoplasm_vacuole"] == ["no", "yes"]


def test_boolean_value_is_rejected(tmp_path, label_space_path):
    doc = yaml.safe_load(label_space_path.read_text())
    doc["attributes"]["granularity"] = [True, False]
    p = tmp_path / "ls.yaml"
    p.write_text(yaml.safe_dump(doc))
    with pytest.raises(LabelSpaceError, match="granularity"):
        load_label_space(p)


def test_unknown_class_raises(label_space_path):
    ls = load_label_space(label_space_path)
    with pytest.raises(LabelSpaceError, match="blast"):
        ls.class_index("blast")


def test_unknown_attribute_value_raises(label_space_path):
    """A value in the data but not in the file is a hard error, never a new index."""
    ls = load_label_space(label_space_path)
    with pytest.raises(LabelSpaceError, match="speckled"):
        ls.attribute_index("chromatin_density", "speckled")


def test_unknown_attribute_column_raises(label_space_path):
    ls = load_label_space(label_space_path)
    with pytest.raises(LabelSpaceError, match="nucleus_colour"):
        ls.attribute_index("nucleus_colour", "blue")


def test_duplicate_value_in_a_column_is_rejected(tmp_path, label_space_path):
    doc = yaml.safe_load(label_space_path.read_text())
    doc["attributes"]["cell_size"] = ["big", "small", "big"]
    p = tmp_path / "ls.yaml"
    p.write_text(yaml.safe_dump(doc))
    with pytest.raises(LabelSpaceError, match="cell_size"):
        load_label_space(p)


def test_class_list_must_match_the_five_classes(tmp_path, label_space_path):
    doc = yaml.safe_load(label_space_path.read_text())
    doc["classes"] = ["neutrophil", "lymphocyte", "monocyte", "eosinophil"]
    p = tmp_path / "ls.yaml"
    p.write_text(yaml.safe_dump(doc))
    with pytest.raises(LabelSpaceError, match="basophil"):
        load_label_space(p)


def test_validate_against_observed_accepts_a_subset(label_space_path):
    ls = load_label_space(label_space_path)
    ls.validate_observed({"cell_size": {"big"}, "granularity": {"yes", "no"}})


def test_validate_against_observed_rejects_an_unseen_value(label_space_path):
    ls = load_label_space(label_space_path)
    with pytest.raises(LabelSpaceError, match="speckled"):
        ls.validate_observed({"chromatin_density": {"densely", "speckled"}})
