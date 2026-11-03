"""Metrics.

Every function here is pinned by hand-worked tests rather than compared against a
reference implementation: a reference would faithfully reproduce a misreading of the
protocol, and a metric that is wrong in the same direction as the hypothesis is the
worst failure this project can have.

Two rules from CLAUDE.md sec 6 are structural here rather than advisory:

* macro-F1 averages over the classes present in y_true for that eval set, and the list
  is returned alongside the number so it is always reported with it;
* predictions are an argmax over all five classes on every domain. Nothing in this
  module restricts predictions to the classes present in a target domain - doing so
  would use target label information.
"""
from __future__ import annotations

import math
import random
from collections import Counter

CLASS_ABSENT = None  # recall for a class with no samples: not measurable, not zero


# --- macro-F1 and recall ---------------------------------------------------------------

def present_classes(y_true, classes: list[str]) -> list[str]:
    """Classes appearing in y_true, in label-space order.

    From y_true, never y_pred: deriving them from predictions would let a model shrink
    its own denominator by never predicting a class it is bad at.
    """
    seen = set(y_true)
    return [name for i, name in enumerate(classes) if i in seen]


def macro_f1_present(y_true, y_pred, classes: list[str]) -> tuple[float, list[str]]:
    """(macro-F1 over present classes, the class list used)."""
    y_true, y_pred = list(y_true), list(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} true vs {len(y_pred)} pred")
    present = present_classes(y_true, classes)

    scores = []
    for name in present:
        c = classes.index(name)
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(0.0 if precision + recall == 0
                      else 2 * precision * recall / (precision + recall))
    return (sum(scores) / len(scores) if scores else 0.0), present


def per_class_recall(y_true, y_pred, classes: list[str]) -> dict[str, float | None]:
    """Recall for every class, with None - not 0 - where the class is absent.

    Zero reads as a model failure. None is the truth: not measurable here. Keyed for all
    five classes always, so basophil and eosinophil are never silently omitted
    (CLAUDE.md sec 6).
    """
    y_true, y_pred = list(y_true), list(y_pred)
    out: dict[str, float | None] = {}
    for c, name in enumerate(classes):
        total = sum(1 for t in y_true if t == c)
        if total == 0:
            out[name] = CLASS_ABSENT
            continue
        hits = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        out[name] = hits / total
    return out


def restrict_to_classes(y_true, y_pred, keep: list[str], classes: list[str]):
    """Keep only rows whose TRUE class is in `keep`. Predictions are left untouched.

    This builds the source side of every gap: pbc_test restricted to the classes present
    in the target domain (CLAUDE.md sec 6). Filtering on the prediction instead would
    let the model choose its own denominator, and dropping predictions of classes
    outside `keep` would hide exactly the errors the restriction must preserve.
    """
    unknown = [name for name in keep if name not in classes]
    if unknown:
        raise ValueError(f"unknown class(es) {unknown}; label space is {classes}")
    indices = {classes.index(name) for name in keep}
    rows = [(t, p) for t, p in zip(y_true, y_pred) if t in indices]
    kept = [name for name in classes if name in keep]
    return [t for t, _ in rows], [p for _, p in rows], kept


# --- calibration --------------------------------------------------------------------------

def expected_calibration_error(confidences, correct, n_bins: int) -> float:
    """Equal-width binned ECE on max-probability confidence."""
    confidences, correct = list(confidences), list(correct)
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    if not confidences:
        return 0.0

    bins: dict[int, list[tuple[float, bool]]] = {}
    for conf, hit in zip(confidences, correct):
        # confidence 1.0 belongs in the last bin, not a bin of its own past the end
        index = min(int(conf * n_bins), n_bins - 1)
        bins.setdefault(index, []).append((conf, bool(hit)))

    total = len(confidences)
    ece = 0.0
    for members in bins.values():
        mean_conf = sum(c for c, _ in members) / len(members)
        accuracy = sum(1 for _, h in members if h) / len(members)
        ece += (len(members) / total) * abs(mean_conf - accuracy)
    return ece


# --- abstention ----------------------------------------------------------------------------

TUNING_SPLIT = "pbc_val"


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("no values")
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    position = q * (len(sorted_values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def select_abstention_thresholds(split_id: str, confidences,
                                 coverage_grid) -> dict[float, float]:
    """Confidence thresholds for each nominal coverage level, from pbc_val only.

    The threshold is the (1 - coverage) quantile of pbc_val confidence, and is then
    applied unchanged to every other domain. Choosing it anywhere else - including
    pbc_test - is threshold selection on held-out data (CLAUDE.md sec 3).
    """
    if split_id != TUNING_SPLIT:
        raise ValueError(
            f"abstention thresholds may only be selected on {TUNING_SPLIT!r}, not "
            f"{split_id!r}. The threshold is chosen on the source val split and applied "
            f"unchanged everywhere else."
        )
    ordered = sorted(float(c) for c in confidences)
    return {float(coverage): _quantile(ordered, 1.0 - float(coverage))
            for coverage in coverage_grid}


def accuracy_vs_coverage(confidences, correct, thresholds: dict[float, float]) -> dict:
    """Accuracy among retained samples, and the coverage actually achieved.

    Achieved coverage is reported rather than assumed: on a target domain it will differ
    from nominal, and that difference is itself a result (CLAUDE.md sec 6).
    """
    confidences, correct = list(confidences), list(correct)
    out = {}
    for coverage, threshold in sorted(thresholds.items()):
        kept = [bool(h) for c, h in zip(confidences, correct) if c >= threshold]
        out[float(coverage)] = {
            "threshold": float(threshold),
            "achieved_coverage": len(kept) / len(confidences) if confidences else 0.0,
            "n_retained": len(kept),
            # None, not 1.0: abstaining on everything is not perfect accuracy.
            "accuracy": (sum(kept) / len(kept)) if kept else None,
        }
    return out


# --- bootstrap ------------------------------------------------------------------------------

def _stratified_indices(y_true, rng: random.Random) -> list[int]:
    """One resample, drawn within each class so every class survives every replicate.

    basophil is 349 of 3099 in pbc_test; unstratified replicates can drop a rare class
    entirely and make its F1 undefined (CLAUDE.md sec 6).
    """
    by_class: dict[int, list[int]] = {}
    for i, label in enumerate(y_true):
        by_class.setdefault(label, []).append(i)
    picked: list[int] = []
    for label in sorted(by_class):
        pool = by_class[label]
        picked.extend(rng.choice(pool) for _ in range(len(pool)))
    return picked


def _replicates(metric_fn, y_true, y_pred, n: int, rng: random.Random,
                stratify_by: str, collect=None) -> list[float]:
    if stratify_by != "class":
        raise ValueError(f"only stratify_by='class' is supported, got {stratify_by!r}")
    y_true, y_pred = list(y_true), list(y_pred)
    values = []
    for _ in range(n):
        indices = _stratified_indices(y_true, rng)
        value = float(metric_fn([y_true[i] for i in indices],
                                [y_pred[i] for i in indices]))
        values.append(value)
        if collect is not None:
            collect.append(value)
    return values


def _interval(values: list[float], ci: float) -> tuple[float, float]:
    ordered = sorted(values)
    tail = (1.0 - ci) / 2.0
    return _quantile(ordered, tail), _quantile(ordered, 1.0 - tail)


def bootstrap_ci(metric_fn, y_true, y_pred, n: int, ci: float, seed: int,
                 stratify_by: str = "class", _collect=None) -> dict:
    """Point estimate on the full data, plus a percentile CI over n replicates."""
    rng = random.Random(seed)
    values = _replicates(metric_fn, y_true, y_pred, n, rng, stratify_by, _collect)
    lower, upper = _interval(values, ci)
    return {
        "point": float(metric_fn(list(y_true), list(y_pred))),
        "lower": lower,
        "upper": upper,
        "n": n,
        "ci": ci,
        "seed": seed,
        "stratify_by": stratify_by,
    }


def gap_with_ci(source, target, metric_fn, n: int, ci: float, seed: int,
                stratify_by: str = "class") -> dict:
    """source metric minus target metric, with the two resampled INDEPENDENTLY.

    Source and target are disjoint sets of images, so pairing their replicates would be
    meaningless and would also understate the interval. Two separate RNGs, two separate
    draws, differenced (CLAUDE.md sec 6).
    """
    source_true, source_pred = source
    target_true, target_pred = target

    source_values = _replicates(metric_fn, source_true, source_pred, n,
                                random.Random(seed), stratify_by)
    target_values = _replicates(metric_fn, target_true, target_pred, n,
                                random.Random(seed + 1_000_003), stratify_by)

    differences = [s - t for s, t in zip(source_values, target_values)]
    lower, upper = _interval(differences, ci)
    source_point = float(metric_fn(list(source_true), list(source_pred)))
    target_point = float(metric_fn(list(target_true), list(target_pred)))
    return {
        "gap": source_point - target_point,
        "source_point": source_point,
        "target_point": target_point,
        "lower": lower,
        "upper": upper,
        "resampled_independently": True,
        "n_source_draws": len(source_values),
        "n_target_draws": len(target_values),
        "ci": ci,
        "seed": seed,
    }


# --- attributes ---------------------------------------------------------------------------------

def per_attribute_accuracy(attribute_true, attribute_pred, label_space) -> dict:
    """Accuracy, balanced accuracy and the majority-class rate per attribute column.

    Reported on pbc_test (CLAUDE.md sec 6). Also computed on pbc_val per epoch as
    MONITORING ONLY, which is a declared exception: the value is recorded under a
    monitor_ key and nothing may select on it. Never computable on a target domain,
    which has no attribute labels at all.

    Balanced accuracy as well as accuracy because the vocabularies are imbalanced -
    granule_type runs nil 2631 against coarse 1216 - so a head that has learned only the
    majority value still scores well on accuracy. majority_rate is that head's score,
    reported alongside so "above the majority rate" can be checked without recomputing
    it, and so it is visible how weak a bar that is.
    """
    if attribute_true is None or attribute_pred is None:
        raise ValueError(
            "per-attribute accuracy needs attribute labels on both sides. Target "
            "domains have none, so this is not computable there and must not be "
            "reported as zero."
        )
    rows_true = [list(r) for r in attribute_true]
    rows_pred = [list(r) for r in attribute_pred]
    if len(rows_true) != len(rows_pred):
        raise ValueError("attribute label/prediction count mismatch")

    out = {}
    for index, column in enumerate(label_space.attributes):
        true = [r[index] for r in rows_true]
        pred = [r[index] for r in rows_pred]
        hits = sum(1 for t, p in zip(true, pred) if t == p)

        per_value_recall = []
        for value in sorted(Counter(true)):
            total = sum(1 for t in true if t == value)
            correct = sum(1 for t, p in zip(true, pred) if t == value and p == value)
            per_value_recall.append(correct / total)

        counts = Counter(true)
        out[column] = {
            "accuracy": hits / len(true) if true else None,
            "balanced_accuracy": (sum(per_value_recall) / len(per_value_recall)
                                  if per_value_recall else None),
            "majority_rate": (max(counts.values()) / len(true)) if true else None,
            "n_values_present": len(per_value_recall),
        }
    return out


def mean_max_probability(confidence) -> float | None:
    """Mean of the predicted-class probabilities.

    Reported next to ECE because ECE alone is ambiguous: a large value can mean a model
    that is confidently wrong or one that is diffidently right, and the two call for
    opposite responses. None when there is nothing to average.
    """
    values = list(confidence)
    if not values:
        return None
    return sum(float(v) for v in values) / len(values)
