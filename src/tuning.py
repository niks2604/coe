"""The tuning grid.

The grid is identical for every model type, by construction: load_grid takes a
model_type argument and ignores it, because equal search budget across baseline,
bottleneck and control is what makes the three comparable (CLAUDE.md sec 5). The
argument exists so a caller cannot pass one by mistake and silently get a different
grid.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import yaml

TUNABLE_FIELDS = ("lr", "weight_decay", "class_weighting")
TIE_BREAK_RULES = {
    "lr_asc": ("lr", False),
    "lr_desc": ("lr", True),
    "weight_decay_asc": ("weight_decay", False),
    "weight_decay_desc": ("weight_decay", True),
    "class_weighting_asc": ("class_weighting", False),
}
SELECTION_METRIC = "val_macro_f1"


class GridError(Exception):
    """Invalid grid, or a selection that would read the wrong metric."""


def _load(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise GridError(f"tuning grid not found: {path}")
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = sorted(set(doc) - {"axes", "tie_break"})
    if unknown:
        raise GridError(f"{path}: unknown key(s) {unknown}")
    return doc


def load_grid(path: str | Path, model_type: str | None = None) -> list[dict]:
    """The full cartesian product of the declared axes, in a stable order.

    model_type is accepted and ignored on purpose: the grid must not vary by model.
    """
    doc = _load(path)
    axes = doc.get("axes")
    if not isinstance(axes, dict) or not axes:
        raise GridError("axes: expected a non-empty mapping")

    unknown = sorted(set(axes) - set(TUNABLE_FIELDS))
    if unknown:
        raise GridError(
            f"axes: {unknown} may not be tuned. The grid covers {list(TUNABLE_FIELDS)} "
            f"only - anything that exists for just one model type, or that CLAUDE.md "
            f"sec 5 fixes a priori, is excluded by design."
        )

    names = [f for f in TUNABLE_FIELDS if f in axes]
    for name in names:
        values = axes[name]
        if not isinstance(values, list) or not values:
            raise GridError(f"axes.{name}: expected a non-empty list")
        if len(set(map(str, values))) != len(values):
            raise GridError(f"axes.{name}: duplicate value in {values!r}")

    trials = [dict(zip(names, combination))
              for combination in itertools.product(*(axes[n] for n in names))]
    if len(trials) != len({tuple(sorted(t.items(), key=str)) for t in trials}):
        raise GridError("grid contains duplicate trials")
    return trials


def tie_break_order(path: str | Path) -> list[str]:
    doc = _load(path)
    rules = doc.get("tie_break") or []
    unknown = [r for r in rules if r not in TIE_BREAK_RULES]
    if unknown:
        raise GridError(f"tie_break: unknown rule(s) {unknown}; "
                        f"known: {sorted(TIE_BREAK_RULES)}")
    return list(rules)


def select_trial(results: list[dict], grid_path: str | Path,
                 metric: str = SELECTION_METRIC) -> dict:
    """Best trial by pbc_val macro-F1, with ties broken by the declared order.

    The metric name is checked rather than trusted: selecting on anything derived from a
    target domain is the leakage this whole protocol exists to prevent, and it would
    otherwise be a one-word change at a call site (CLAUDE.md sec 3).
    """
    if metric != SELECTION_METRIC:
        raise GridError(
            f"trial selection reads {SELECTION_METRIC!r} only, not {metric!r}. "
            f"Selecting on any other split - and a target domain above all - is model "
            f"selection using data that must be touched once, at final evaluation."
        )
    if not results:
        raise GridError("no trial results to select from")
    for row in results:
        if metric not in row:
            raise GridError(f"trial result is missing {metric!r}: {row}")

    rules = tie_break_order(grid_path)

    def sort_key(row: dict):
        key: list = [-float(row[metric])]
        for rule in rules:
            field, descending = TIE_BREAK_RULES[rule]
            value = row["trial"].get(field)
            if isinstance(value, (int, float)):
                key.append(-float(value) if descending else float(value))
            else:
                key.append(str(value))
        return key

    return sorted(results, key=sort_key)[0]
