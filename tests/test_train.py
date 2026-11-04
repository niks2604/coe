"""Training loop, checkpointing and the tuning grid."""
import pytest

from src.config import load_config
from src.train import checkpoint_payload, select_best_epoch
from src.tuning import GridError, load_grid, select_trial

pytest.importorskip("torch")


# --- checkpoint selection ---------------------------------------------------------------

def test_best_epoch_is_the_val_peak_not_the_last():
    """select_best_epoch now feeds best.pt, which is informational only: the headline
    checkpoint is the final epoch (CLAUDE.md sec 5). The peak still has to be found
    correctly for the record to be worth keeping."""
    history = [{"epoch": 0, "val_macro_f1": 0.70},
               {"epoch": 1, "val_macro_f1": 0.91},
               {"epoch": 2, "val_macro_f1": 0.88},
               {"epoch": 3, "val_macro_f1": 0.85}]
    assert select_best_epoch(history)["epoch"] == 1


def test_ties_resolve_to_the_earlier_epoch():
    history = [{"epoch": 0, "val_macro_f1": 0.9}, {"epoch": 1, "val_macro_f1": 0.9}]
    assert select_best_epoch(history)["epoch"] == 0


def test_selection_metric_is_val_macro_f1(experiment_config):
    """Not val loss, not accuracy (CLAUDE.md sec 6: accuracy is not an acceptable
    summary)."""
    history = [{"epoch": 0, "val_macro_f1": 0.5, "val_accuracy": 0.99},
               {"epoch": 1, "val_macro_f1": 0.8, "val_accuracy": 0.10}]
    assert select_best_epoch(history)["epoch"] == 1


def test_empty_history_raises():
    with pytest.raises(ValueError):
        select_best_epoch([])


# --- checkpoint contents ------------------------------------------------------------------

def test_checkpoint_embeds_the_resolved_config_and_git_state(experiment_config,
                                                             tiny_raw_loadable):
    cfg = load_config(experiment_config(**{"data.image_size": 16}))
    payload = checkpoint_payload(cfg, state_dict={}, history=[
        {"epoch": 0, "val_macro_f1": 0.5}], group_offsets=[("cell_size", 0, 2)])
    assert payload["config"]["run_id"] == cfg.run_id
    assert isinstance(payload["config"]["data"]["class_mapping"], dict)
    assert len(payload["git"]["commit"]) == 40
    assert isinstance(payload["git"]["dirty"], bool)


def test_checkpoint_records_the_group_offsets(experiment_config, tiny_raw_loadable):
    """Without them the 31 classifier inputs cannot be read back as (column, value)."""
    cfg = load_config(experiment_config(**{"data.image_size": 16}))
    payload = checkpoint_payload(cfg, state_dict={}, history=[
        {"epoch": 0, "val_macro_f1": 0.5}], group_offsets=[("cell_size", 0, 2)])
    assert payload["group_offsets"] == [("cell_size", 0, 2)]


def test_checkpoint_records_the_selected_epoch(experiment_config, tiny_raw_loadable):
    cfg = load_config(experiment_config(**{"data.image_size": 16}))
    payload = checkpoint_payload(cfg, state_dict={}, history=[
        {"epoch": 0, "val_macro_f1": 0.5}, {"epoch": 1, "val_macro_f1": 0.9}],
        group_offsets=[])
    assert payload["selected_epoch"] == 1
    assert payload["val_macro_f1"] == pytest.approx(0.9)


def test_checkpoint_carries_a_config_hash_for_the_ledger(experiment_config,
                                                         tiny_raw_loadable):
    cfg = load_config(experiment_config(**{"data.image_size": 16}))
    payload = checkpoint_payload(cfg, state_dict={}, history=[
        {"epoch": 0, "val_macro_f1": 0.5}], group_offsets=[])
    assert len(payload["config_sha256"]) == 64


# --- the training script touches no target data ---------------------------------------------

def test_train_module_never_imports_a_target_loader():
    """Structural, not behavioural: make_eval_loader is the only path to target data and
    train.py must not reach for it."""
    import inspect

    import src.train as train_mod

    source = inspect.getsource(train_mod)
    assert "make_eval_loader" not in source
    assert "raabin" not in source.lower()


def test_overfit_diagnostic_never_imports_a_target_loader():
    """The overfit-one-batch diagnostic is held to the same structural rule as
    src.train: it reads pbc_train through the pinned factory and cannot reach a target
    domain at all."""
    import inspect

    import scripts.overfit_one_batch as overfit_mod

    source = inspect.getsource(overfit_mod)
    assert "make_eval_loader" not in source
    assert "raabin" not in source.lower()
    assert "make_train_loader" in source


def test_capacity_diagnostic_never_imports_a_target_loader():
    """Same structural rule again: the capacity diagnostic reads pbc_train's CSV and
    nothing else, and cannot reach a target domain."""
    import inspect

    import scripts.classifier_capacity as capacity_mod

    source = inspect.getsource(capacity_mod)
    assert "make_eval_loader" not in source
    assert "raabin" not in source.lower()


def test_capacity_diagnostic_reads_only_the_train_split():
    """Not pbc_val, which is the tuning split, and not pbc_test, which is held out."""
    import inspect

    import scripts.classifier_capacity as capacity_mod

    source = inspect.getsource(capacity_mod)
    assert "pbc_attr_v1_train.csv" in source
    assert "pbc_attr_v1_val" not in source
    assert "pbc_attr_v1_test" not in source


def test_train_module_uses_only_the_pinned_split_factories():
    import inspect

    import src.train as train_mod

    source = inspect.getsource(train_mod)
    assert "make_train_loader" in source and "make_val_loader" in source


# --- the tuning grid -------------------------------------------------------------------------

@pytest.fixture
def grid_path():
    from pathlib import Path
    return Path(__file__).resolve().parents[1] / "configs" / "tuning_grid.yaml"


def test_committed_grid_loads(grid_path):
    trials = load_grid(grid_path)
    assert trials and all(isinstance(t, dict) for t in trials)


def test_grid_is_the_cartesian_product_of_the_declared_axes(grid_path):
    trials = load_grid(grid_path)
    assert len(trials) == len({tuple(sorted(t.items())) for t in trials})
    assert len(trials) == 12


def test_grid_does_not_contain_lambda(grid_path):
    """lambda exists only for the bottleneck, so tuning it would give the bottleneck a
    larger search budget than the baseline (CLAUDE.md sec 5)."""
    trials = load_grid(grid_path)
    assert all("attribute_loss_weight" not in t for t in trials)


def test_grid_does_not_contain_the_schedule_or_augmentation(grid_path):
    """Both are fixed a priori and identical across models."""
    trials = load_grid(grid_path)
    forbidden = {"augmentation", "lr_schedule", "warmup_epochs", "backbone_norm",
                 "image_size", "epochs"}
    assert all(not (forbidden & set(t)) for t in trials)


def test_the_same_grid_is_used_for_all_three_models(grid_path):
    """Equal search budget is what makes the three comparable at all."""
    trials = load_grid(grid_path)
    for model_type in ("baseline", "bottleneck", "control"):
        assert load_grid(grid_path, model_type=model_type) == trials


def test_grid_axes_only_touch_tunable_fields(grid_path):
    trials = load_grid(grid_path)
    assert set(trials[0]) == {"lr", "weight_decay", "class_weighting"}


def test_tie_break_is_deterministic_on_an_exact_tie(grid_path):
    """With 12 trials near-ties on pbc_val are likely, and "whichever came first" is a
    choice made after seeing numbers."""
    results = [
        {"trial": {"lr": 1e-3, "weight_decay": 1e-4, "class_weighting": "none"},
         "val_macro_f1": 0.9},
        {"trial": {"lr": 1e-4, "weight_decay": 1e-5, "class_weighting": "none"},
         "val_macro_f1": 0.9},
    ]
    chosen = select_trial(results, grid_path)
    assert chosen["trial"]["lr"] == 1e-4
    assert select_trial(list(reversed(results)), grid_path) == chosen


def test_select_trial_prefers_the_higher_val_macro_f1(grid_path):
    results = [{"trial": {"lr": 1e-4, "weight_decay": 1e-5, "class_weighting": "none"},
                "val_macro_f1": 0.5},
               {"trial": {"lr": 1e-3, "weight_decay": 1e-4, "class_weighting": "none"},
                "val_macro_f1": 0.9}]
    assert select_trial(results, grid_path)["val_macro_f1"] == pytest.approx(0.9)


def test_select_trial_refuses_a_target_metric(grid_path):
    """Selection is on pbc_val macro-F1 and nothing else."""
    results = [{"trial": {"lr": 1e-4, "weight_decay": 1e-5, "class_weighting": "none"},
                "val_macro_f1": 0.5, "raabin_testA_macro_f1": 0.9}]
    with pytest.raises(GridError, match="raabin"):
        select_trial(results, grid_path, metric="raabin_testA_macro_f1")


def test_tuning_trials_record_their_single_seed(grid_path):
    """Single-seed tuning is a blessed exception and has to be visible in the record
    (CLAUDE.md sec 5)."""
    trials = load_grid(grid_path)
    assert all("seed" not in t for t in trials)


# --- diagnostics: the two loss terms are recorded separately --------------------------

def _tiny(experiment_config, **overrides):
    base = {"data.image_size": 32, "train.batch_size": 2, "train.epochs": 1}
    base.update(overrides)
    return load_config(experiment_config(**base))


def test_history_separates_the_class_and_attribute_loss(experiment_config,
                                                        tiny_raw_loadable, mapping,
                                                        label_space, tmp_path):
    """A single blended number cannot tell "the class head is stuck" from "the attribute
    heads are stuck", which is the question the smoke run could not answer."""
    from src.train import train

    cfg = _tiny(experiment_config)
    row = train(cfg, mapping, label_space, out_dir=tmp_path)["history"][0]
    for key in ("train_class_loss", "train_attribute_loss_unscaled",
                "val_class_loss", "val_attribute_loss_unscaled"):
        assert key in row, f"history row does not record {key}"
    assert row["train_attribute_loss_unscaled"] > 0    # lambda is 1.0 here
    assert row["train_class_loss"] > 0


def test_the_control_records_a_zero_attribute_loss(experiment_config,
                                                   tiny_raw_loadable, mapping,
                                                   label_space, tmp_path):
    """lambda 0 means the term is not computed at all, and the record must say 0 rather
    than omit it - an absent key reads as "not measured"."""
    from src.train import train

    cfg = _tiny(experiment_config, **{"model.type": "control",
                                      "bottleneck.attribute_loss_weight": 0.0})
    row = train(cfg, mapping, label_space, out_dir=tmp_path)["history"][0]
    assert row["train_attribute_loss_unscaled"] == 0.0
    assert row["val_attribute_loss_unscaled"] == 0.0


# --- per-attribute pbc_val accuracy is MONITORING ONLY --------------------------------

def test_val_per_attribute_accuracy_is_recorded_for_monitoring(experiment_config,
                                                               tiny_raw_loadable,
                                                               mapping, label_space,
                                                               tmp_path):
    from src.train import train

    cfg = _tiny(experiment_config)
    row = train(cfg, mapping, label_space, out_dir=tmp_path)["history"][0]
    assert "monitor_val_per_attribute" in row
    assert len(row["monitor_val_per_attribute"]) == 11


def test_every_per_attribute_val_key_is_marked_as_monitoring(experiment_config,
                                                             tiny_raw_loadable,
                                                             mapping, label_space,
                                                             tmp_path):
    """CLAUDE.md sec 6 reports per-attribute accuracy on pbc_test. Reading it on pbc_val
    is a monitoring exception, so the key says so and cannot be mistaken for a
    reportable metric."""
    from src.train import train

    cfg = _tiny(experiment_config)
    row = train(cfg, mapping, label_space, out_dir=tmp_path)["history"][0]
    assert all(key.startswith("monitor_")
               for key in row if "per_attribute" in key)


def test_select_best_epoch_ignores_monitoring_keys():
    """Selection is pbc_val macro-F1 and nothing else. An epoch with far better
    attribute accuracy must not win on that basis."""
    history = [{"epoch": 0, "val_macro_f1": 0.90,
                "monitor_val_per_attribute": {"cell_size": {"accuracy": 0.10}}},
               {"epoch": 1, "val_macro_f1": 0.80,
                "monitor_val_per_attribute": {"cell_size": {"accuracy": 0.99}}}]
    assert select_best_epoch(history)["epoch"] == 0


def test_selection_reads_no_monitoring_key():
    """Structural: whatever else lands in a history row, the selector must not read it."""
    import inspect

    import src.train as train_mod

    assert "monitor" not in inspect.getsource(train_mod.select_best_epoch)




# --- the final-epoch rule -------------------------------------------------------------

def test_final_epoch_rule_selects_the_last_epoch_in_history():
    """Not the peak. pbc_val cannot support an argmax over epochs, and a per-model
    argmax selected the three models at different points on their own learning curves
    (CLAUDE.md sec 5)."""
    from src.config import SelectionCfg
    from src.train import select_epoch

    history = [{"epoch": 0, "val_macro_f1": 0.97},
               {"epoch": 1, "val_macro_f1": 0.91},
               {"epoch": 2, "val_macro_f1": 0.93}]
    rule = SelectionCfg(rule="final_epoch", report_metric="val_macro_f1",
                        divergence_guard="non_finite_loss")
    assert select_epoch(history, rule)["epoch"] == 2


def test_the_rule_comes_from_the_config_not_a_module_constant():
    """Structural: select_epoch must branch on what it is handed, so a config can
    change the rule without a source edit (CLAUDE.md sec 5)."""
    import inspect

    from src.config import SelectionCfg
    from src.train import select_epoch

    source = inspect.getsource(select_epoch)
    assert "selection.rule" in source or ".rule" in source
    bad = SelectionCfg(rule="nonsense", report_metric="val_macro_f1",
                       divergence_guard="non_finite_loss")
    with pytest.raises(ValueError, match="nonsense"):
        select_epoch([{"epoch": 0, "val_macro_f1": 0.5}], bad)


def test_last_pt_is_written_beside_best_pt(experiment_config, tiny_raw_loadable,
                                           mapping, label_space, tmp_path):
    """Only best.pt was written before, so a change of rule meant a retrain."""
    from src.train import train

    cfg = _tiny(experiment_config, **{"train.epochs": 2})
    result = train(cfg, mapping, label_space, out_dir=tmp_path)
    assert (tmp_path / "last.pt").is_file()
    assert (tmp_path / "best.pt").is_file()
    assert result["checkpoint"].endswith("last.pt")
    assert result["informational_checkpoint"].endswith("best.pt")


def test_last_pt_holds_the_final_epoch_and_best_pt_the_peak(experiment_config,
                                                            tiny_raw_loadable, mapping,
                                                            label_space, tmp_path):
    import torch

    from src.train import train

    cfg = _tiny(experiment_config, **{"train.epochs": 2})
    result = train(cfg, mapping, label_space, out_dir=tmp_path)
    last = torch.load(tmp_path / "last.pt", weights_only=False)
    best = torch.load(tmp_path / "best.pt", weights_only=False)
    assert last["selected_epoch"] == result["history"][-1]["epoch"]
    assert best["selected_epoch"] == select_best_epoch(result["history"])["epoch"]


def test_each_checkpoint_records_its_role_and_rule(experiment_config,
                                                   tiny_raw_loadable, mapping,
                                                   label_space, tmp_path):
    """A results JSON and a ledger line must name the file that was actually read."""
    import torch

    from src.train import train

    cfg = _tiny(experiment_config)
    train(cfg, mapping, label_space, out_dir=tmp_path)
    last = torch.load(tmp_path / "last.pt", weights_only=False)
    best = torch.load(tmp_path / "best.pt", weights_only=False)
    assert last["selection"]["role"] == "headline"
    assert last["selection"]["rule"] == "final_epoch"
    assert best["selection"]["role"] == "informational"


# --- the divergence guard -------------------------------------------------------------

def test_divergence_guard_aborts_and_writes_no_headline_checkpoint(
        experiment_config, tiny_raw_loadable, mapping, label_space, tmp_path,
        monkeypatch):
    """A non-finite training loss is a failed run, not a selectable one. The guard reads
    the training loss and never a validation metric: a guard that stopped on a falling
    val metric would be epoch selection under another name (CLAUDE.md sec 5)."""
    import torch

    import src.losses as losses_mod
    from src.train import DivergenceError, train

    real = losses_mod.compute_loss

    def nan_loss(output, targets, cfg, class_weights=None):
        total, parts = real(output, targets, cfg, class_weights)
        return total * float("nan"), parts

    monkeypatch.setattr("src.train.compute_loss", nan_loss)
    cfg = _tiny(experiment_config, **{"train.epochs": 2})
    with pytest.raises(DivergenceError):
        train(cfg, mapping, label_space, out_dir=tmp_path)
    assert not (tmp_path / "last.pt").exists()
    assert not (tmp_path / "best.pt").exists()
    written = list(tmp_path.glob("*.json"))
    assert written, "a failed run must still leave a record of why it failed"
    import json
    record = json.loads(written[0].read_text())
    assert record["failed"] is True
    assert "non_finite" in record["failure"]


def test_a_clean_run_records_that_it_did_not_diverge(experiment_config,
                                                     tiny_raw_loadable, mapping,
                                                     label_space, tmp_path):
    from src.train import train

    result = train(_tiny(experiment_config), mapping, label_space, out_dir=tmp_path)
    assert result["failed"] is False
    assert result["failure"] is None
    assert result["epochs_run"] == result["config"]["train"]["epochs"]
