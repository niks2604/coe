"""Config loading, validation, and the leakage checks that live in validation."""
import pytest
import yaml

from conftest import DELETE
from src.config import (
    IMAGENET_MEAN,
    MPS_RUN_ID_PREFIXES,
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



def test_abstention_threshold_on_a_target_split_is_rejected(experiment_config):
    path = experiment_config(**{"eval.abstention.threshold_split": "lisc"})
    with pytest.raises(TargetDomainLeakage, match="lisc"):
        load_config(path)


def test_a_target_id_anywhere_under_train_is_rejected(experiment_config):
    """Catches leakage through fields that do not exist yet."""
    path = experiment_config(**{"train.class_weighting": "lisc"})
    with pytest.raises(TargetDomainLeakage, match="lisc"):
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
                    "smoke.yaml", "smoke_baseline.yaml", "smoke_control.yaml",
                    "wiring_baseline.yaml", "wiring_bottleneck.yaml",
                    "wiring_control.yaml")

WIRING_YAMLS = ("wiring_baseline.yaml", "wiring_bottleneck.yaml", "wiring_control.yaml")


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
                  "lr", "weight_decay", "class_weighting", "augmentation"):
        values = {n: doc["train"][field] for n, doc in docs.items()}
        assert len(set(map(str, values.values()))) == 1, f"train.{field} differs: {values}"
    for field in ("image_size", "normalization", "source", "subset_n_images"):
        values = {n: doc["data"][field] for n, doc in docs.items()}
        assert len(set(map(str, values.values()))) == 1, f"data.{field} differs: {values}"
    for n, doc in docs.items():
        assert doc["model"]["backbone"] == "resnet50"
        assert doc["model"]["backbone_norm"] == "batchnorm", f"{n} is not the headline arm"


# --- device ---------------------------------------------------------------------------

def test_device_defaults_to_cpu_when_absent(experiment_config):
    """Optional in the schema so short test configs stay valid, exactly as lr_schedule
    and warmup_epochs are. Every committed experiment YAML must still state it."""
    assert load_config(experiment_config()).device == "cpu"


def test_unknown_device_is_rejected(experiment_config):
    # Matches on the offending value, not on "device": the tmp_path in the message
    # contains the test name and would satisfy a looser match on its own.
    with pytest.raises(ConfigError, match="tpu"):
        load_config(experiment_config(**{"device": "tpu"}))


def test_resolved_config_records_the_device(experiment_config):
    cfg = load_config(experiment_config(**{"device": "cpu"}))
    assert cfg.resolved["device"] == "cpu"


def test_mps_is_accepted_for_a_smoke_config(experiment_config):
    """MPS cannot give bit-identical reruns, so it is for smoke and debug work only."""
    cfg = load_config(experiment_config(**{"device": "mps", "run_id": "smoke_baseline"}))
    assert cfg.device == "mps"


@pytest.mark.parametrize("run_id", ["bottleneck", "baseline", "control",
                                    "wiring_bottleneck", "tuning_trial_3"])
def test_mps_is_rejected_for_any_non_debug_config(experiment_config, run_id):
    """Deterministic cuDNN is mandated by CLAUDE.md sec 5 and MPS cannot honour it, so a
    wiring, tuning or final run may not use it."""
    with pytest.raises(ConfigError, match="debug"):
        load_config(experiment_config(**{"device": "mps", "run_id": run_id}))


def test_cpu_always_resolves(experiment_config):
    from src.config import resolve_device

    assert resolve_device(load_config(experiment_config())).type == "cpu"


def test_resolve_device_raises_for_an_unavailable_device(experiment_config):
    """Never a silent fall back to CPU: a run that reports a GPU it did not use is
    worse than a run that fails."""
    import torch

    from src.config import resolve_device

    if not torch.cuda.is_available():
        unavailable, run_id = "cuda", "bottleneck"
    elif not torch.backends.mps.is_available():
        unavailable, run_id = "mps", "smoke"
    else:
        pytest.skip("every device is available on this machine")
    cfg = load_config(experiment_config(**{"device": unavailable, "run_id": run_id}))
    with pytest.raises(ConfigError, match=unavailable):
        resolve_device(cfg)


def test_run_metadata_records_the_resolved_device():
    """The old record said "cpu" unless CUDA was present, so an MPS run would have been
    written down as a CPU run."""
    assert run_metadata(device="mps")["device"] == "mps"


@pytest.mark.parametrize("name", EXPERIMENT_YAMLS)
def test_experiment_yaml_states_the_device_explicitly(name):
    """Optional in the schema, mandatory in a committed experiment: which device a run
    used belongs in the config and in the results record, not in a Python default."""
    doc = yaml.safe_load((_configs_dir() / name).read_text())
    assert "device" in doc, f"{name} does not state device"


@pytest.mark.parametrize("name", WIRING_YAMLS)
def test_wiring_yaml_never_uses_mps(name):
    doc = yaml.safe_load((_configs_dir() / name).read_text())
    assert doc["device"] != "mps"


@pytest.mark.parametrize("name", WIRING_YAMLS)
def test_wiring_yaml_has_its_own_run_id(name):
    """A wiring run selects nothing, so its results must not land on top of a headline
    run's files."""
    doc = yaml.safe_load((_configs_dir() / name).read_text())
    assert doc["run_id"].startswith("wiring_")


@pytest.mark.parametrize("name", WIRING_YAMLS)
def test_wiring_yaml_matches_its_origin_in_every_protocol_field(name):
    """The wiring check must run the declared settings, not a variant of them:
    only run_id and device may differ from the headline config it mirrors."""
    origin = name.replace("wiring_", "")
    wiring = yaml.safe_load((_configs_dir() / name).read_text())
    headline = yaml.safe_load((_configs_dir() / origin).read_text())
    differing = {k for k in set(wiring) | set(headline)
                 if wiring.get(k) != headline.get(k)}
    # Subset, not equality: both now declare device: cuda, so device need not differ.
    # What matters is that no protocol field does.
    assert differing <= {"run_id", "device"}, differing


# --- where a config is declared to run -----------------------------------------------

def _experiment_docs():
    """Every committed config that has a run_id, discovered by globbing.

    Globbed rather than listed so a config added later is covered without anyone
    remembering to edit this test.
    """
    docs = {}
    for path in sorted(_configs_dir().glob("*.yaml")):
        doc = yaml.safe_load(path.read_text())
        if isinstance(doc, dict) and "run_id" in doc:
            docs[path.name] = doc
    return docs


def test_every_non_debug_config_runs_on_the_kaggle_gpu():
    """A wiring, tuning or final run belongs on the GPU and is declared that way in the
    config, not chosen at the command line. Only smoke and debug configs stay local."""
    offenders = {name: {"env": doc.get("env"), "device": doc.get("device")}
                 for name, doc in _experiment_docs().items()
                 if not doc["run_id"].startswith(MPS_RUN_ID_PREFIXES)
                 and (doc.get("env") != "kaggle" or doc.get("device") != "cuda")}
    assert not offenders, offenders


def test_smoke_and_debug_configs_stay_local():
    """The counterpart: a smoke config that quietly pointed at Kaggle would spend GPU
    quota on a 500-image sanity check."""
    offenders = {name: {"env": doc.get("env"), "device": doc.get("device")}
                 for name, doc in _experiment_docs().items()
                 if doc["run_id"].startswith(MPS_RUN_ID_PREFIXES)
                 and doc.get("env") != "local"}
    assert not offenders, offenders


def test_kaggle_audit_config_loads():
    """The audit can run on Kaggle from a committed config, with no CLI flag deciding
    which tree it read (CLAUDE.md sec 5)."""
    cfg = load_audit_config(_configs_dir() / "audit_kaggle.yaml")
    assert cfg.env == "kaggle"
    assert str(cfg.paths.results_dir) == "/kaggle/working/results"
    assert "kaggle/input" in str(cfg.paths.pbc_images)


def test_the_two_audit_configs_share_their_settings_and_local_paths():
    a = yaml.safe_load((_configs_dir() / "audit.yaml").read_text())
    b = yaml.safe_load((_configs_dir() / "audit_kaggle.yaml").read_text())
    assert (a["env"], b["env"]) == ("local", "kaggle")
    assert a["audit"] == b["audit"]
    assert a["paths"]["local"] == b["paths"]["local"]


def test_the_kaggle_audit_reads_the_same_tree_as_training():
    """The invariant that makes an audit worth anything: if it reads a different tree
    from the one the models train on, it has reconciled nothing."""
    audit = yaml.safe_load((_configs_dir() / "audit_kaggle.yaml").read_text())
    for name in ("baseline.yaml", "bottleneck.yaml", "control.yaml",
                 "wiring_baseline.yaml", "wiring_bottleneck.yaml",
                 "wiring_control.yaml"):
        doc = yaml.safe_load((_configs_dir() / name).read_text())
        assert doc["paths"]["kaggle"] == audit["paths"]["kaggle"], name


# --- checkpoint selection ------------------------------------------------------------

def test_selection_defaults_to_the_final_epoch_rule(experiment_config):
    """Optional in the schema, exactly as device is, so short test configs stay valid.
    The default is the protocol rule, never the old first-max one."""
    cfg = load_config(experiment_config(**{"selection": DELETE}))
    assert cfg.selection.rule == "final_epoch"
    assert cfg.selection.report_metric == "val_macro_f1"
    assert cfg.selection.divergence_guard == "non_finite_loss"


def test_an_unknown_selection_rule_is_refused(experiment_config):
    path = experiment_config(**{"selection.rule": "best_val_macro_f1"})
    with pytest.raises(ConfigError, match="selection.rule"):
        load_config(path)


def test_an_unknown_divergence_guard_is_refused(experiment_config):
    path = experiment_config(**{"selection.divergence_guard": "val_macro_f1_drop"})
    with pytest.raises(ConfigError, match="divergence_guard"):
        load_config(path)


def test_the_report_metric_may_not_be_a_target_quantity(experiment_config):
    """A guard that read a target metric would be checkpoint selection on target
    information (CLAUDE.md sec 3)."""
    path = experiment_config(**{"selection.report_metric": "raabin_testA_macro_f1"})
    with pytest.raises(ConfigError, match="report_metric"):
        load_config(path)


def test_a_target_id_anywhere_under_selection_is_rejected(experiment_config):
    """Catches leakage through selection fields that do not exist yet."""
    path = experiment_config(**{"selection.rule": "raabin_testA"})
    with pytest.raises(TargetDomainLeakage, match="raabin_testA"):
        load_config(path)


def test_the_resolved_config_records_the_selection_block(experiment_config):
    """A checkpoint embeds the resolved config, so the rule that produced it must be
    readable from the checkpoint alone."""
    cfg = load_config(experiment_config())
    assert cfg.resolved["selection"]["rule"] == "final_epoch"


@pytest.mark.parametrize("name", sorted(EXPERIMENT_YAMLS))
def test_every_experiment_yaml_declares_a_selection_block(name):
    doc = yaml.safe_load((_configs_dir() / name).read_text())
    assert "selection" in doc, f"{name} does not declare how its checkpoint is chosen"
    assert doc["selection"]["rule"] == "final_epoch", name


@pytest.mark.parametrize("name", sorted(EXPERIMENT_YAMLS))
def test_no_experiment_yaml_still_declares_early_stopping(name):
    """Patience-based early stopping is removed. A config that still declared it would
    record a setting with no effect (CLAUDE.md sec 5)."""
    doc = yaml.safe_load((_configs_dir() / name).read_text())
    assert "early_stopping" not in doc["train"], name


def test_the_three_headline_configs_share_their_selection_block():
    docs = {n: yaml.safe_load((_configs_dir() / n).read_text())
            for n in ("baseline.yaml", "bottleneck.yaml", "control.yaml")}
    values = {n: doc["selection"] for n, doc in docs.items()}
    assert len(set(map(str, values.values()))) == 1, values
