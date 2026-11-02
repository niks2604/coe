"""Metrics. Hand-worked cases throughout: a metric that is wrong in the same direction
as the hypothesis is the worst failure this project can have."""
import pytest

from src.metrics import (
    accuracy_vs_coverage,
    bootstrap_ci,
    expected_calibration_error,
    gap_with_ci,
    macro_f1_present,
    per_attribute_accuracy,
    per_class_recall,
    restrict_to_classes,
    select_abstention_thresholds,
)

CLASSES = ["neutrophil", "lymphocyte", "monocyte", "eosinophil", "basophil"]


# --- macro-F1 over present classes ------------------------------------------------------

def test_macro_f1_on_a_hand_worked_case():
    """3 neutrophils (2 right), 3 lymphocytes (3 right).
    neutrophil: P=2/2=1.0  R=2/3     F1=0.8
    lymphocyte: P=3/4=0.75 R=3/3=1.0 F1=6/7
    macro = (0.8 + 6/7) / 2
    """
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 1, 1, 1, 1]
    value, present = macro_f1_present(y_true, y_pred, CLASSES)
    assert present == ["neutrophil", "lymphocyte"]
    assert value == pytest.approx((0.8 + 6 / 7) / 2)


def test_present_classes_come_from_y_true_not_y_pred():
    """A spurious monocyte prediction must not add monocyte to the averaged set - that
    would let a model shrink its own denominator."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 2, 1, 1]
    _, present = macro_f1_present(y_true, y_pred, CLASSES)
    assert present == ["neutrophil", "lymphocyte"]


def test_predicting_an_absent_class_counts_as_an_error():
    """TestB holds 2 of 5 classes. Predicting monocyte there is a false negative for the
    true class and nothing else. Masking logits to the present classes would be leakage
    (CLAUDE.md sec 6)."""
    y_true = [0, 0, 1, 1]
    perfect, _ = macro_f1_present(y_true, [0, 0, 1, 1], CLASSES)
    leaked, _ = macro_f1_present(y_true, [0, 2, 1, 1], CLASSES)
    assert leaked < perfect


def test_present_class_order_follows_the_label_space():
    y_true = [4, 1, 0]
    _, present = macro_f1_present(y_true, [4, 1, 0], CLASSES)
    assert present == ["neutrophil", "lymphocyte", "basophil"]


def test_macro_f1_records_two_classes_for_a_testb_shaped_set():
    y_true = [0] * 10 + [1] * 5
    value, present = macro_f1_present(y_true, y_true, CLASSES)
    assert len(present) == 2
    assert value == pytest.approx(1.0)


# --- per-class recall --------------------------------------------------------------------

def test_absent_class_recall_is_none_not_zero():
    """Zero reads as a model failure; null is the truth - not measurable here."""
    recall = per_class_recall([0, 0, 1], [0, 0, 1], CLASSES)
    assert recall["monocyte"] is None
    assert recall["neutrophil"] == pytest.approx(1.0)


def test_recall_is_hand_checkable():
    recall = per_class_recall([0, 0, 0, 1], [0, 1, 1, 1], CLASSES)
    assert recall["neutrophil"] == pytest.approx(1 / 3)
    assert recall["lymphocyte"] == pytest.approx(1.0)


def test_rare_classes_are_always_keyed_even_when_absent():
    """CLAUDE.md sec 6 requires basophil and eosinophil to be reported separately."""
    recall = per_class_recall([0], [0], CLASSES)
    assert "basophil" in recall and "eosinophil" in recall


# --- ECE ----------------------------------------------------------------------------------

def test_ece_is_zero_for_perfect_calibration():
    confidences = [0.5] * 100
    correct = [True] * 50 + [False] * 50
    assert expected_calibration_error(confidences, correct, n_bins=10) == pytest.approx(0.0)


def test_ece_is_one_for_total_overconfidence():
    assert expected_calibration_error([1.0] * 10, [False] * 10, n_bins=10) == pytest.approx(1.0)


def test_ece_on_a_hand_worked_two_bin_case():
    """Bin [0.0,0.5): conf 0.2, acc 0.0 -> |0.2| * 2/4
       Bin [0.5,1.0]: conf 0.9, acc 1.0 -> |0.1| * 2/4
    """
    value = expected_calibration_error([0.2, 0.2, 0.9, 0.9],
                                       [False, False, True, True], n_bins=2)
    assert value == pytest.approx(0.2 * 0.5 + 0.1 * 0.5)


def test_ece_uses_the_configured_bin_count():
    a = expected_calibration_error([0.3, 0.7], [False, True], n_bins=2)
    b = expected_calibration_error([0.3, 0.7], [False, True], n_bins=15)
    assert a != b


# --- abstention ----------------------------------------------------------------------------

def test_thresholds_come_from_pbc_val_only():
    with pytest.raises(ValueError, match="pbc_val"):
        select_abstention_thresholds("raabin_testA", [0.9, 0.5], [0.9])


@pytest.mark.parametrize("split", ["pbc_test", "raabin_testB", "raabin_train"])
def test_thresholds_refuse_every_other_split(split):
    with pytest.raises(ValueError, match="pbc_val"):
        select_abstention_thresholds(split, [0.9, 0.5], [0.9])


def test_threshold_at_full_coverage_admits_everything():
    thresholds = select_abstention_thresholds("pbc_val", [0.1, 0.5, 0.9], [1.0])
    assert thresholds[1.0] <= 0.1


def test_threshold_is_the_coverage_quantile_of_val_confidence():
    confidences = [i / 100 for i in range(1, 101)]
    thresholds = select_abstention_thresholds("pbc_val", confidences, [0.9])
    assert thresholds[0.9] == pytest.approx(0.1, abs=0.02)


def test_coverage_curve_reports_achieved_coverage_not_nominal():
    """Target coverage will differ from nominal, and that difference is a result
    (CLAUDE.md sec 6)."""
    curve = accuracy_vs_coverage([0.95, 0.95, 0.2, 0.2], [True, True, False, False],
                                 {0.9: 0.5})
    assert curve[0.9]["achieved_coverage"] == pytest.approx(0.5)
    assert curve[0.9]["accuracy"] == pytest.approx(1.0)


def test_ninety_percent_coverage_is_the_headline_key():
    curve = accuracy_vs_coverage([0.9, 0.1], [True, False], {0.9: 0.5, 1.0: 0.0})
    assert 0.9 in curve


def test_zero_coverage_accuracy_is_none_not_one():
    curve = accuracy_vs_coverage([0.1, 0.2], [True, True], {0.9: 0.99})
    assert curve[0.9]["achieved_coverage"] == pytest.approx(0.0)
    assert curve[0.9]["accuracy"] is None


# --- bootstrap -----------------------------------------------------------------------------

def test_bootstrap_ci_is_seed_reproducible():
    y_true = [0, 0, 1, 1, 2, 2] * 10
    args = (lambda t, p: macro_f1_present(t, p, CLASSES)[0], y_true, y_true)
    a = bootstrap_ci(*args, n=50, ci=0.95, seed=7, stratify_by="class")
    b = bootstrap_ci(*args, n=50, ci=0.95, seed=7, stratify_by="class")
    assert a == b


def test_bootstrap_ci_differs_with_a_different_seed():
    y_true = [0, 0, 1, 1, 2, 2] * 10
    y_pred = [0, 1, 1, 1, 2, 0] * 10
    fn = lambda t, p: macro_f1_present(t, p, CLASSES)[0]
    a = bootstrap_ci(fn, y_true, y_pred, n=50, ci=0.95, seed=1, stratify_by="class")
    b = bootstrap_ci(fn, y_true, y_pred, n=50, ci=0.95, seed=2, stratify_by="class")
    assert a != b


def test_stratified_resampling_keeps_every_class_in_every_replicate():
    """basophil has 349 of 3099 in pbc_test. Unstratified replicates can drop a class
    entirely and make macro-F1 undefined for it."""
    y_true = [0] * 100 + [4] * 2
    seen = []
    bootstrap_ci(lambda t, p: float(len(set(t))), y_true, y_true, n=30, ci=0.95,
                 seed=0, stratify_by="class", _collect=seen)
    assert all(v == 2.0 for v in seen)


def test_bootstrap_returns_point_estimate_and_bounds():
    y_true = [0, 1] * 20
    out = bootstrap_ci(lambda t, p: macro_f1_present(t, p, CLASSES)[0], y_true, y_true,
                       n=40, ci=0.95, seed=0, stratify_by="class")
    assert out["lower"] <= out["point"] <= out["upper"]


# --- the gap ---------------------------------------------------------------------------------

def test_gap_is_source_minus_target():
    source = ([0, 0, 1, 1], [0, 0, 1, 1])
    target = ([0, 0, 1, 1], [0, 1, 1, 1])
    out = gap_with_ci(source, target, lambda t, p: macro_f1_present(t, p, CLASSES)[0],
                      n=40, ci=0.95, seed=0)
    assert out["gap"] > 0


def test_gap_resamples_the_two_domains_independently():
    """Paired resampling would be wrong - source and target are disjoint image sets -
    and would also understate the interval."""
    source = ([0, 1] * 20, [0, 1] * 20)
    target = ([0, 1] * 20, [0, 0] * 20)
    out = gap_with_ci(source, target, lambda t, p: macro_f1_present(t, p, CLASSES)[0],
                      n=60, ci=0.95, seed=0)
    assert out["resampled_independently"] is True
    assert out["n_source_draws"] == out["n_target_draws"] == 60


def test_gap_of_a_domain_against_itself_is_about_zero():
    same = ([0, 1] * 20, [0, 1] * 20)
    out = gap_with_ci(same, same, lambda t, p: macro_f1_present(t, p, CLASSES)[0],
                      n=60, ci=0.95, seed=0)
    assert out["gap"] == pytest.approx(0.0)


# --- the gap basis: restricting the source side ------------------------------------------------

def test_restrict_to_classes_keeps_only_the_named_classes():
    """CLAUDE.md sec 6: the source side of every gap is pbc_test restricted to the
    classes present in that target domain."""
    y_true = [0, 1, 2, 3, 4]
    y_pred = [0, 1, 2, 3, 4]
    t, p, kept = restrict_to_classes(y_true, y_pred, ["neutrophil", "lymphocyte"], CLASSES)
    assert t == [0, 1] and p == [0, 1]
    assert kept == ["neutrophil", "lymphocyte"]


def test_restriction_filters_on_true_label_not_prediction():
    """Dropping rows by prediction would let the model choose its own denominator."""
    y_true = [0, 0, 2]
    y_pred = [0, 2, 0]
    t, p, _ = restrict_to_classes(y_true, y_pred, ["neutrophil"], CLASSES)
    assert t == [0, 0] and p == [0, 2]


def test_restriction_preserves_predictions_of_absent_classes_as_errors():
    y_true = [0, 0]
    y_pred = [0, 3]
    t, p, _ = restrict_to_classes(y_true, y_pred, ["neutrophil"], CLASSES)
    value, _ = macro_f1_present(t, p, CLASSES)
    assert value < 1.0


def test_restriction_to_an_unknown_class_raises():
    with pytest.raises(ValueError, match="blast"):
        restrict_to_classes([0], [0], ["blast"], CLASSES)


# --- per-attribute accuracy -------------------------------------------------------------------

def test_per_attribute_accuracy_and_balanced_accuracy(label_space):
    """Balanced accuracy as well as accuracy: a head that has learned only the majority
    value scores well on accuracy alone (CLAUDE.md sec 6)."""
    true = [[0] * 11, [0] * 11, [1] + [0] * 10]
    pred = [[0] * 11, [0] * 11, [0] * 11]
    out = per_attribute_accuracy(true, pred, label_space)
    assert out["cell_size"]["accuracy"] == pytest.approx(2 / 3)
    assert out["cell_size"]["balanced_accuracy"] == pytest.approx(0.5)
    assert out["cell_shape"]["accuracy"] == pytest.approx(1.0)


def test_per_attribute_mean_is_over_eleven_columns(label_space):
    true = [[0] * 11]
    out = per_attribute_accuracy(true, [[0] * 11], label_space)
    assert len(out) == 11


def test_per_attribute_accuracy_refuses_missing_labels(label_space):
    """Raabin has no attribute labels, so this must raise rather than return zeros."""
    with pytest.raises(ValueError, match="attribute labels"):
        per_attribute_accuracy(None, [[0] * 11], label_space)
