"""The committed index orderings for classes and attribute values.

These lists decide what class index 0 and attribute index 0 mean. Built from whatever
the dataloader saw first, index 0 could differ between runs and every confusion matrix,
per-class recall and attribute metric would be incomparable across runs - so the file is
the authority and the data is checked against it, never the other way round.

A value present in the data but absent from the file is a hard error. Appending it
silently would change the meaning of an index mid-project.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.data.domains import FIVE_CLASSES

N_ATTRIBUTES = 11


class LabelSpaceError(Exception):
    """Invalid label space, or a class/attribute value that is not in it."""


@dataclass(frozen=True)
class LabelSpace:
    classes: list[str]
    attributes: dict[str, list[str]]
    path: Path

    @property
    def n_attributes(self) -> int:
        return len(self.attributes)

    @property
    def n_attribute_values(self) -> int:
        return sum(len(v) for v in self.attributes.values())

    @property
    def attribute_columns(self) -> list[str]:
        return list(self.attributes)

    def class_index(self, name: str) -> int:
        try:
            return self.classes.index(name)
        except ValueError:
            raise LabelSpaceError(
                f"class {name!r} is not in the label space {self.classes}. "
                f"Resolve raw names through the class mapping first."
            ) from None

    def attribute_index(self, column: str, value: str) -> int:
        if column not in self.attributes:
            raise LabelSpaceError(
                f"attribute column {column!r} is not in {self.path}. "
                f"Columns: {self.attribute_columns}"
            )
        try:
            return self.attributes[column].index(value)
        except ValueError:
            raise LabelSpaceError(
                f"{column}={value!r} is not in the committed vocabulary "
                f"{self.attributes[column]}. Add it to {self.path.name} deliberately - "
                f"appending it automatically would renumber the existing indices."
            ) from None

    def validate_observed(self, observed: dict[str, set[str]]) -> None:
        """Check a dataset's vocabulary against the file. Subsets are fine; a value the
        file does not list is not."""
        problems = []
        for column, values in sorted(observed.items()):
            if column not in self.attributes:
                problems.append(f"unknown column {column!r}")
                continue
            unknown = sorted(set(values) - set(self.attributes[column]))
            if unknown:
                problems.append(f"{column}: {unknown}")
        if problems:
            raise LabelSpaceError(
                f"values present in the data but not in {self.path}: "
                + "; ".join(problems)
            )


def load_label_space(path: str | Path) -> LabelSpace:
    path = Path(path)
    if not path.exists():
        raise LabelSpaceError(f"label space not found: {path}")
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise LabelSpaceError(f"{path}: expected a YAML mapping")

    unknown = sorted(set(doc) - {"classes", "attributes"})
    if unknown:
        raise LabelSpaceError(f"{path}: unknown key(s) {unknown}")

    classes = doc.get("classes")
    if not isinstance(classes, list) or sorted(classes) != sorted(FIVE_CLASSES):
        raise LabelSpaceError(
            f"{path}: classes must be exactly the five differential classes "
            f"{sorted(FIVE_CLASSES)} in the order indices should follow; got {classes!r}"
        )
    if len(set(classes)) != len(classes):
        raise LabelSpaceError(f"{path}: duplicate class in {classes!r}")

    attributes = doc.get("attributes")
    if not isinstance(attributes, dict):
        raise LabelSpaceError(f"{path}: attributes must be a mapping")
    if len(attributes) != N_ATTRIBUTES:
        raise LabelSpaceError(
            f"{path}: expected {N_ATTRIBUTES} attribute columns, got {len(attributes)}")

    for column, values in attributes.items():
        if not isinstance(values, list) or not values:
            raise LabelSpaceError(f"{path}: {column}: expected a non-empty list")
        non_strings = [v for v in values if not isinstance(v, str)]
        if non_strings:
            # Unquoted "yes"/"no"/"on"/"off" become booleans in YAML and then match
            # nothing in the CSV, silently emptying the column.
            raise LabelSpaceError(
                f"{path}: {column}: values must be strings, got {non_strings!r}. "
                f"Quote them - YAML reads yes/no as booleans."
            )
        if len(set(values)) != len(values):
            raise LabelSpaceError(f"{path}: {column}: duplicate value in {values!r}")

    return LabelSpace(classes=list(classes),
                      attributes={k: list(v) for k, v in attributes.items()},
                      path=path)
