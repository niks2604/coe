"""Training loop, checkpointing and the tuning grid."""
import pytest

from src.config import load_config
from src.train import checkpoint_payload, select_best_epoch
from src.tuning import GridError, load_grid, select_trial

pytest.importorskip("torch")


# --- checkpoint selection ---------------------------------------------------------------

def test_best_epoch_is_the_val_peak_not_the_last():
    """A val curve that peaks in the middle and then overfits. Taking the last epoch is
    the easiest way to report a worse gap than the run actually achieved."""
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
