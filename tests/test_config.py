"""Config loading, validation, and the leakage checks that live in validation."""
import pytest
import yaml

from conftest import DELETE
from src.config import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ConfigError,
    TargetDomainLeakage,
    load_audit_config,
    load_config,
    run_metadata,
    set_seed,
)


def test_canonical_config_loads(experiment_config):
    cfg = load_config(experiment_config())
    assert cfg.model.type == "bottleneck"
    assert cfg.seed == 0


# --- leakage ---------------------------------------------------------------------

def test_early_stopping_on_a_target_split_is_rejected(experiment_config):
    path = experiment_config(**{"train.early_stopping.split": "raabin_testA"})
    with pytest.raises(TargetDomainLeakage, match="raabin_testA"):
        load_config(path)


def test_early_stopping_on_raabin_train_is_rejected(experiment_config):
    path = experiment_config(**{"train.early_stopping.split": "raabin_train"})
    with pytest.raises(TargetDomainLeakage, match="raabin_train"):
        load_config(path)


def test_abstention_threshold_on_a_target_split_is_rejected(experiment_config):
    path = experiment_config(**{"eval.abstention.threshold_split": "lisc"})
    with pytest.raises(TargetDomainLeakage, match="lisc"):
        load_config(path)


def test_a_target_id_anywhere_under_train_is_rejected(experiment_config):
    """Catches leakage through fields that do not exist yet."""
    path = experiment_config(**{"train.class_weighting": "lisc"})
    with pytest.raises(TargetDomainLeakage, match="lisc"):
        load_config(path)


def test_source_split_is_accepted_for_early_stopping(experiment_config):
    cfg = load_config(experiment_config(**{"train.early_stopping.split": "pbc_val"}))
    assert cfg.train.early_stopping["split"] == "pbc_val"


def test_pbc_test_is_rejected_for_tuning(experiment_config):
    """Source, but still not a tuning split: all tuning happens on pbc_val."""
    path = experiment_config(**{"train.early_stopping.split": "pbc_test"})
    with pytest.raises(ConfigError, match="pbc_val"):
        load_config(path)


def test_unknown_eval_domain_is_rejected(experiment_config):
    path = experiment_config(**{"eval.domains": ["pbc_test", "raabin_testC"]})
    with pytest.raises(ConfigError, match="raabin_testC"):
        load_config(path)


# --- normalisation constants are fixed a priori ----------------------------------

def test_normalization_must_be_the_imagenet_constants(experiment_config):
    """Backbone constants are fixed ImageNet values, never computed from a dataset.
    Anything else is the signature of stats computed from data."""
    path = experiment_config(**{"data.normalization.mean": [0.7, 0.6, 0.65]})
    with pytest.raises(ConfigError, match="ImageNet"):
        load_config(path)


def test_normalization_is_required(experiment_config):
    with pytest.raises(ConfigError, match="normalization"):
        load_config(experiment_config(**{"data.normalization": DELETE}))


def test_imagenet_constants_are_what_we_think_they_are():
    assert IMAGENET_MEAN == [0.485, 0.456, 0.406]
    assert IMAGENET_STD == [0.229, 0.224, 0.225]


# --- model type / bottleneck coupling --------------------------------------------

def test_baseline_must_not_carry_a_bottleneck_block(experiment_config):
    path = experiment_config(**{"model.type": "baseline"})
    with pytest.raises(ConfigError, match="bottleneck"):
        load_config(path)


def test_baseline_without_bottleneck_block_loads(experiment_config):
    cfg = load_config(experiment_config(**{
        "model.type": "baseline",
        "bottleneck": DELETE,
        "eval.attribute_modes": DELETE,
    }))
    assert cfg.bottleneck is None


def test_bottleneck_block_is_required_for_bottleneck(experiment_config):
    path = experiment_config(**{"bottleneck": DELETE})
    with pytest.raises(ConfigError, match="bottleneck"):
        load_config(path)


def test_bottleneck_needs_a_nonzero_attribute_loss_weight(experiment_config):
    """Weight 0 is the control, not the bottleneck; the two must not be confusable."""
    path = experiment_config(**{"bottleneck.attribute_loss_weight": 0.0})
    with pytest.raises(ConfigError, match="control"):
        load_config(path)


def test_control_requires_attribute_loss_weight_zero(experiment_config):
    path = experiment_config(**{"model.type": "control",
                                "bottleneck.attribute_loss_weight": 1.0})
    with pytest.raises(ConfigError, match="0"):
        load_config(path)


def test_control_loads_with_weight_zero(experiment_config):
    cfg = load_config(experiment_config(**{"model.type": "control",
                                           "bottleneck.attribute_loss_weight": 0.0}))
    assert cfg.model.type == "control"
    assert cfg.bottleneck.attribute_loss_weight == 0.0


def test_attribute_modes_forbidden_for_baseline(experiment_config):
    path = experiment_config(**{"model.type": "baseline", "bottleneck": DELETE})
    with pytest.raises(ConfigError, match="attribute_modes"):
        load_config(path)


def test_backbone_norm_defaults_to_batchnorm(experiment_config):
    cfg = load_config(experiment_config(**{"model.backbone_norm": DELETE}))
    assert cfg.model.backbone_norm == "batchnorm"


def test_unknown_backbone_norm_is_rejected(experiment_config):
    with pytest.raises(ConfigError, match="backbone_norm"):
        load_config(experiment_config(**{"model.backbone_norm": "layernorm"}))


# --- metric protocol decisions are frozen in the schema --------------------------

def test_macro_f1_over_present_classes_is_the_only_allowed_value(experiment_config):
    """TestB has two of five classes. Fixing this in the schema stops a 5-class
    figure being chosen after target results are seen."""
    with pytest.raises(ConfigError, match="present_classes"):
        load_config(experiment_config(**{"eval.macro_f1_over": "all_classes"}))


def test_bootstrap_must_resample_domains_independently(experiment_config):
    with pytest.raises(ConfigError, match="resample_domains_independently"):
        load_config(experiment_config(
            **{"bootstrap.resample_domains_independently": False}))


# --- unknown keys and env resolution ---------------------------------------------

def test_effective_number_class_weighting_is_rejected(experiment_config):
    """Dropped from the enum: it needs a beta, which is a hyperparameter and may not be
    hardcoded, and declaring one would widen a grid that must stay identical across the
    three models (CLAUDE.md sec 5)."""
    with pytest.raises(ConfigError, match="effective_number"):
        load_config(experiment_config(**{"train.class_weighting": "effective_number"}))


def test_only_none_and_inverse_frequency_are_accepted(experiment_config):
    from src.config import CLASS_WEIGHTINGS

    assert CLASS_WEIGHTINGS == ("none", "inverse_frequency")
    for mode in CLASS_WEIGHTINGS:
        load_config(experiment_config(**{"train.class_weighting": mode}))


def test_unknown_top_level_key_is_rejected(experiment_config, tmp_path):
    path = experiment_config()
    cfg = yaml.safe_load(path.read_text())
    cfg["learning_rate"] = 0.1
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ConfigError, match="learning_rate"):
        load_config(path)


def test_unknown_nested_key_is_rejected(experiment_config):
    """train.momentum: AdamW has no momentum field, so this can never become real.
    The previous example, train.lr_schedule, is now a declared field."""
    path = experiment_config(**{"train.momentum": 0.9})
    with pytest.raises(ConfigError, match="momentum"):
        load_config(path)


def test_env_selects_the_matching_paths(experiment_config):
    cfg = load_config(experiment_config())
    assert cfg.env == "local"
    assert "TODO_FILL" not in str(cfg.paths.pbc_images)


def test_kaggle_todo_fill_paths_fail_loudly(experiment_config):
    with pytest.raises(ConfigError, match="TODO_FILL"):
        load_config(experiment_config(**{"env": "kaggle"}))


def test_both_env_sections_must_be_present(experiment_config):
    with pytest.raises(ConfigError, match="kaggle"):
        load_config(experiment_config(**{"paths.kaggle": DELETE}))


def test_unknown_env_is_rejected(experiment_config):
    with pytest.raises(ConfigError, match="colab"):
        load_config(experiment_config(**{"env": "colab"}))


def test_relative_paths_resolve_against_the_repo_root(experiment_config):
    path = experiment_config(**{"paths.local.results_dir": "results"})
    cfg = load_config(path)
    assert cfg.paths.results_dir.is_absolute()
    assert cfg.paths.results_dir.name == "results"


# --- the resolved config is self-contained ---------------------------------------

def test_resolved_config_embeds_the_class_mapping(experiment_config):
    """CLAUDE.md sec 5: the full resolved config is embedded, not referenced."""
    cfg = load_config(experiment_config())
    resolved = cfg.resolved
    assert isinstance(resolved["data"]["class_mapping"], dict)
    assert "neutrophil" in resolved["data"]["class_mapping"]


def test_resolved_config_records_the_selected_env_paths_only(experiment_config):
    resolved = load_config(experiment_config()).resolved
    assert resolved["env"] == "local"
    assert set(resolved["paths"]) == {
        "pbc_images", "wbcatt_dir", "raabin_root", "lisc_root", "results_dir",
    }


def test_resolved_config_is_yaml_serialisable(experiment_config):
    """It goes into results/<run_id>.json, so it must contain no live objects."""
    import json

    resolved = load_config(experiment_config()).resolved
    json.loads(json.dumps(resolved))


# --- seeding and run metadata ----------------------------------------------------

def test_set_seed_is_reproducible():
    import random

    torch = pytest.importorskip("torch")
    import numpy as np

    def draw():
        set_seed(1234)
        return random.random(), float(np.random.rand()), float(torch.rand(1))

    assert draw() == draw()


def test_different_seeds_differ():
    import random

    set_seed(1)
    a = random.random()
    set_seed(2)
    assert random.random() != a


def test_set_seed_sets_deterministic_cudnn():
    torch = pytest.importorskip("torch")
    set_seed(0)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_run_metadata_has_git_commit_and_dirty_flag():
    meta = run_metadata()
    assert len(meta["git"]["commit"]) == 40
    assert isinstance(meta["git"]["dirty"], bool)
    assert "platform" in meta


# --- audit config -----------------------------------------------------------------

def test_audit_config_loads(audit_config):
    cfg = load_audit_config(audit_config)
    assert cfg.env == "local"
    assert cfg.audit.verify_images is True


def test_audit_config_rejects_unknown_keys(audit_config):
    cfg = yaml.safe_load(audit_config.read_text())
    cfg["audit"]["compute_mean_std"] = True
    audit_config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ConfigError, match="compute_mean_std"):
        load_audit_config(audit_config)


# --- the committed experiment YAMLs state the protocol fields explicitly ----------

EXPERIMENT_YAMLS = ("baseline.yaml", "bottleneck.yaml", "control.yaml",
                    "smoke.yaml", "smoke_baseline.yaml", "smoke_control.yaml")


def _configs_dir():
    from pathlib import Path
    return Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("name", EXPERIMENT_YAMLS)
def test_experiment_yaml_states_the_schedule_explicitly(name):
    """lr_schedule and warmup_epochs are optional in the schema so short test configs
    stay valid, but every committed experiment must state them: the protocol's warmup
    length belongs in the config, not in a Python default (CLAUDE.md sec 5)."""
    doc = yaml.safe_load((_configs_dir() / name).read_text())
    assert "lr_schedule" in doc["train"], f"{name} does not state train.lr_schedule"
    assert "warmup_epochs" in doc["train"], f"{name} does not state train.warmup_epochs"


@pytest.mark.parametrize("name", EXPERIMENT_YAMLS)
def test_experiment_yaml_states_imagenet_weights_explicitly(name):
    doc = yaml.safe_load((_configs_dir() / name).read_text())
    assert "imagenet_weights" in doc["model"]


@pytest.mark.parametrize("name", EXPERIMENT_YAMLS)
def test_experiment_yaml_loads(name):
    load_config(_configs_dir() / name)


def test_the_three_headline_configs_share_every_protocol_field():
    """Differences between the three models must come from the representation, not the
    protocol (CLAUDE.md sec 5)."""
    docs = {n: yaml.safe_load((_configs_dir() / n).read_text())
            for n in ("baseline.yaml", "bottleneck.yaml", "control.yaml")}
    for field in ("optimizer", "lr_schedule", "warmup_epochs", "epochs", "batch_size",
                  "lr", "weight_decay", "class_weighting", "augmentation",
                  "early_stopping"):
        values = {n: doc["train"][field] for n, doc in docs.items()}
        assert len(set(map(str, values.values()))) == 1, f"train.{field} differs: {values}"
    for field in ("image_size", "normalization", "source", "subset_n_images"):
        values = {n: doc["data"][field] for n, doc in docs.items()}
        assert len(set(map(str, values.values()))) == 1, f"data.{field} differs: {values}"
    for n, doc in docs.items():
        assert doc["model"]["backbone"] == "resnet50"
        assert doc["model"]["backbone_norm"] == "batchnorm", f"{n} is not the headline arm"
