"""Typed config loading, validation, seeding, and run metadata.

Every experiment is fully specified by one YAML file (CLAUDE.md sec 5), so this module
does the work that would otherwise be scattered as defaults through the codebase:

* unknown keys are hard errors, so a typo cannot silently fall back to a default;
* `env` is the only switch between local and Kaggle, and only the selected env's paths
  are resolved;
* the leakage rules in CLAUDE.md sec 3 are enforced here, before anything loads data;
* three protocol decisions are pinned into the schema so they cannot be revisited once
  target-domain numbers are visible: fixed ImageNet normalisation constants,
  macro-F1 over present classes only, and independent resampling per domain.

torch is imported lazily inside set_seed so the audit can run without it.
"""
from __future__ import annotations

import os
import platform
import random
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.data.domains import (
    DATASETS,
    TUNING_SPLIT,
    ClassMapping,
    Domain,
    TargetDomainLeakage,
    domain_of,
    load_class_mapping,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Fixed a priori. Never computed from any dataset - deriving them from data would make
# the preprocessing depend on whichever images were used to compute them.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

ENVS = ("local", "kaggle")
PATH_KEYS = ("pbc_images", "wbcatt_dir", "raabin_root", "lisc_root", "results_dir")
TODO_SENTINEL = "TODO_FILL"

MODEL_TYPES = ("baseline", "bottleneck", "control")
BACKBONE_NORMS = ("batchnorm", "groupnorm")
ATTRIBUTE_MODES = ("soft", "hard")
CLASS_WEIGHTINGS = ("none", "inverse_frequency", "effective_number")


class ConfigError(Exception):
    """Invalid config. Always names the offending key."""


# --- schema helpers -------------------------------------------------------------------

def _check_keys(node: Any, allowed: set[str], required: set[str], where: str) -> dict:
    if not isinstance(node, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(node).__name__}")
    unknown = sorted(set(node) - allowed)
    if unknown:
        raise ConfigError(
            f"{where}: unknown key(s) {unknown}. Allowed: {sorted(allowed)}. "
            f"Unknown keys are rejected so a typo cannot silently use a default."
        )
    missing = sorted(required - set(node))
    if missing:
        raise ConfigError(f"{where}: missing required key(s) {missing}")
    return node


def _as_float_triple(value: Any, where: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ConfigError(f"{where}: expected three floats, got {value!r}")
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError):
        raise ConfigError(f"{where}: expected three floats, got {value!r}") from None


def _iter_strings(node: Any):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            yield from _iter_strings(v)
    elif isinstance(node, (list, tuple)):
        for v in node:
            yield from _iter_strings(v)


# --- dataclasses -----------------------------------------------------------------------

@dataclass(frozen=True)
class Paths:
    pbc_images: Path
    wbcatt_dir: Path
    raabin_root: Path
    lisc_root: Path
    results_dir: Path


@dataclass(frozen=True)
class AuditSettings:
    class_mapping: ClassMapping
    class_mapping_path: Path
    verify_images: bool
    tree_max_depth: int


@dataclass(frozen=True)
class DataCfg:
    class_mapping: ClassMapping
    class_mapping_path: Path
    source: str
    image_size: int
    subset_n_images: int | None
    normalization: dict[str, list[float]]


@dataclass(frozen=True)
class ModelCfg:
    type: str
    backbone: str
    backbone_norm: str = "batchnorm"


@dataclass(frozen=True)
class BottleneckCfg:
    attribute_loss_weight: float


@dataclass(frozen=True)
class TrainCfg:
    epochs: int
    batch_size: int
    lr: float
    optimizer: str
    weight_decay: float
    class_weighting: str
    early_stopping: dict


@dataclass(frozen=True)
class EvalCfg:
    domains: list[str]
    secondary_subsets: list[str]
    abstention: dict
    ece_bins: int
    macro_f1_over: str
    attribute_modes: list[str] | None = None


@dataclass(frozen=True)
class BootstrapCfg:
    enabled: bool
    n: int
    ci: float
    stratify_by: str
    resample_domains_independently: bool


@dataclass(frozen=True)
class AuditConfig:
    env: str
    seed: int
    paths: Paths
    audit: AuditSettings
    resolved: dict


@dataclass(frozen=True)
class Config:
    run_id: str
    env: str
    seed: int
    paths: Paths
    data: DataCfg
    model: ModelCfg
    train: TrainCfg
    eval: EvalCfg
    bootstrap: BootstrapCfg
    resolved: dict
    bottleneck: BottleneckCfg | None = None


# --- shared sections --------------------------------------------------------------------

def _load_yaml(path: str | Path) -> tuple[Path, dict]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ConfigError(f"{path}: expected a YAML mapping at the top level")
    return path, doc


def _resolve_path(value: Any, where: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{where}: expected a path string, got {value!r}")
    if TODO_SENTINEL in value:
        raise ConfigError(
            f"{where}: still set to {TODO_SENTINEL}. Fill in the real path before "
            f"running with this env."
        )
    p = Path(value).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def _build_paths(doc: dict, env: str) -> tuple[Paths, dict]:
    if env not in ENVS:
        raise ConfigError(f"env: {env!r} is not one of {list(ENVS)}")
    section = _check_keys(doc.get("paths"), set(ENVS), set(ENVS), "paths")
    for name in ENVS:
        _check_keys(section[name], set(PATH_KEYS), set(PATH_KEYS), f"paths.{name}")
    resolved = {k: _resolve_path(section[env][k], f"paths.{env}.{k}") for k in PATH_KEYS}

    return Paths(**resolved), {k: str(v) for k, v in resolved.items()}


def _seed_of(doc: dict) -> int:
    seed = doc.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigError(f"seed: expected an int, got {seed!r}")
    return seed


# --- audit config -------------------------------------------------------------------------

AUDIT_TOP = {"env", "seed", "paths", "audit"}
AUDIT_KEYS = {"class_mapping", "verify_images", "tree_max_depth"}


def load_audit_config(path: str | Path) -> AuditConfig:
    path, doc = _load_yaml(path)
    _check_keys(doc, AUDIT_TOP, AUDIT_TOP, str(path))
    env = doc["env"]
    seed = _seed_of(doc)
    paths, paths_resolved = _build_paths(doc, env)

    node = _check_keys(doc["audit"], AUDIT_KEYS, AUDIT_KEYS, "audit")
    mapping_path = _resolve_path(node["class_mapping"], "audit.class_mapping")
    if not mapping_path.exists():
        raise ConfigError(f"audit.class_mapping: not found: {mapping_path}")
    if not isinstance(node["verify_images"], bool):
        raise ConfigError(f"audit.verify_images: expected a bool, got {node['verify_images']!r}")
    if not isinstance(node["tree_max_depth"], int) or node["tree_max_depth"] < 1:
        raise ConfigError("audit.tree_max_depth: expected a positive int")

    mapping = load_class_mapping(mapping_path)
    settings = AuditSettings(
        class_mapping=mapping,
        class_mapping_path=mapping_path,
        verify_images=node["verify_images"],
        tree_max_depth=node["tree_max_depth"],
    )
    resolved = {
        "env": env,
        "seed": seed,
        "paths": paths_resolved,
        "audit": {
            "class_mapping": mapping.raw,
            "class_mapping_path": str(mapping_path),
            "verify_images": settings.verify_images,
            "tree_max_depth": settings.tree_max_depth,
        },
    }
    return AuditConfig(env=env, seed=seed, paths=paths, audit=settings, resolved=resolved)


# --- experiment config ----------------------------------------------------------------------

EXP_TOP = {"run_id", "env", "seed", "paths", "data", "model", "bottleneck",
           "train", "eval", "bootstrap"}
EXP_REQUIRED = EXP_TOP - {"bottleneck"}

DATA_KEYS = {"class_mapping", "source", "image_size", "subset_n_images", "normalization"}
MODEL_KEYS = {"type", "backbone", "backbone_norm"}
TRAIN_KEYS = {"epochs", "batch_size", "lr", "optimizer", "weight_decay",
              "class_weighting", "early_stopping"}
EARLY_STOP_KEYS = {"metric", "patience", "split"}
EVAL_KEYS = {"domains", "attribute_modes", "secondary_subsets", "abstention",
             "ece_bins", "macro_f1_over"}
EVAL_REQUIRED = EVAL_KEYS - {"attribute_modes"}
ABSTENTION_KEYS = {"threshold_split", "coverage_grid"}
BOOTSTRAP_KEYS = {"enabled", "n", "ci", "stratify_by", "resample_domains_independently"}


def _check_no_target_under_train(train_doc: dict) -> None:
    """Catches leakage through fields that do not exist yet."""
    for value in _iter_strings(train_doc):
        if value in DATASETS and domain_of(value) is Domain.TARGET:
            raise TargetDomainLeakage(
                f"train: references target dataset {value!r}. Target domains must not "
                f"reach training, validation, early stopping or model selection "
                f"(CLAUDE.md sec 3)."
            )


def _build_data(doc: dict) -> tuple[DataCfg, dict]:
    node = _check_keys(doc["data"], DATA_KEYS, DATA_KEYS, "data")
    mapping_path = _resolve_path(node["class_mapping"], "data.class_mapping")
    if not mapping_path.exists():
        raise ConfigError(f"data.class_mapping: not found: {mapping_path}")
    mapping = load_class_mapping(mapping_path)

    if node["source"] != "pbc_wbcatt":
        raise ConfigError(
            f"data.source: {node['source']!r} is not supported; the source domain is "
            f"'pbc_wbcatt' - the 10,298 WBCAtt-labelled PBC images (CLAUDE.md sec 2)."
        )
    if not isinstance(node["image_size"], int) or node["image_size"] < 8:
        raise ConfigError("data.image_size: expected an int >= 8")
    subset = node["subset_n_images"]
    if subset is not None and (not isinstance(subset, int) or subset < 1):
        raise ConfigError("data.subset_n_images: expected null or a positive int")

    norm = _check_keys(node["normalization"], {"mean", "std"}, {"mean", "std"},
                       "data.normalization")
    mean = _as_float_triple(norm["mean"], "data.normalization.mean")
    std = _as_float_triple(norm["std"], "data.normalization.std")
    if mean != IMAGENET_MEAN or std != IMAGENET_STD:
        raise ConfigError(
            f"data.normalization: must be the fixed ImageNet constants "
            f"mean={IMAGENET_MEAN}, std={IMAGENET_STD}. Values computed from any "
            f"dataset are not permitted (CLAUDE.md sec 3)."
        )

    cfg = DataCfg(
        class_mapping=mapping,
        class_mapping_path=mapping_path,
        source=node["source"],
        image_size=node["image_size"],
        subset_n_images=subset,
        normalization={"mean": mean, "std": std},
    )
    resolved = {
        "class_mapping": mapping.raw,
        "class_mapping_path": str(mapping_path),
        "source": cfg.source,
        "image_size": cfg.image_size,
        "subset_n_images": cfg.subset_n_images,
        "normalization": {"mean": mean, "std": std},
    }
    return cfg, resolved


def _build_model(doc: dict) -> ModelCfg:
    node = _check_keys(doc["model"], MODEL_KEYS, {"type", "backbone"}, "model")
    if node["type"] not in MODEL_TYPES:
        raise ConfigError(f"model.type: {node['type']!r} is not one of {list(MODEL_TYPES)}")
    norm = node.get("backbone_norm", "batchnorm")
    if norm not in BACKBONE_NORMS:
        raise ConfigError(
            f"model.backbone_norm: {norm!r} is not one of {list(BACKBONE_NORMS)}")
    if not isinstance(node["backbone"], str) or not node["backbone"]:
        raise ConfigError("model.backbone: expected a non-empty string")
    return ModelCfg(type=node["type"], backbone=node["backbone"], backbone_norm=norm)


def _build_bottleneck(doc: dict, model_type: str) -> BottleneckCfg | None:
    present = "bottleneck" in doc and doc["bottleneck"] is not None
    needs_block = model_type in ("bottleneck", "control")

    if needs_block and not present:
        raise ConfigError(
            f"bottleneck: required when model.type is {model_type!r} "
            f"(attribute_loss_weight must be stated explicitly)."
        )
    if not needs_block and present:
        raise ConfigError(
            f"bottleneck: must be absent when model.type is {model_type!r}; "
            f"the baseline has no attribute heads."
        )
    if not present:
        return None

    node = _check_keys(doc["bottleneck"], {"attribute_loss_weight"},
                       {"attribute_loss_weight"}, "bottleneck")
    try:
        weight = float(node["attribute_loss_weight"])
    except (TypeError, ValueError):
        raise ConfigError("bottleneck.attribute_loss_weight: expected a float") from None
    if weight < 0:
        raise ConfigError("bottleneck.attribute_loss_weight: must be >= 0")

    if model_type == "control" and weight != 0.0:
        raise ConfigError(
            f"bottleneck.attribute_loss_weight: must be exactly 0 when model.type is "
            f"'control' (same architecture, class loss only); got {weight}."
        )
    if model_type == "bottleneck" and weight == 0.0:
        raise ConfigError(
            "bottleneck.attribute_loss_weight: 0 is the 'control' configuration, not "
            "the bottleneck. Set model.type: control instead, so the two cannot be "
            "confused in the results."
        )
    return BottleneckCfg(attribute_loss_weight=weight)


def _build_train(doc: dict) -> TrainCfg:
    node = _check_keys(doc["train"], TRAIN_KEYS, TRAIN_KEYS, "train")
    _check_no_target_under_train(node)

    for key, kind in (("epochs", int), ("batch_size", int)):
        if not isinstance(node[key], kind) or node[key] < 1:
            raise ConfigError(f"train.{key}: expected a positive int")
    for key in ("lr", "weight_decay"):
        try:
            float(node[key])
        except (TypeError, ValueError):
            raise ConfigError(f"train.{key}: expected a float") from None
    if node["class_weighting"] not in CLASS_WEIGHTINGS:
        raise ConfigError(
            f"train.class_weighting: {node['class_weighting']!r} is not one of "
            f"{list(CLASS_WEIGHTINGS)}"
        )

    es = _check_keys(node["early_stopping"], EARLY_STOP_KEYS, EARLY_STOP_KEYS,
                     "train.early_stopping")
    split = es["split"]
    if split not in DATASETS:
        raise ConfigError(
            f"train.early_stopping.split: unknown dataset id {split!r}; "
            f"known: {sorted(DATASETS)}"
        )
    if domain_of(split) is Domain.TARGET:
        raise TargetDomainLeakage(
            f"train.early_stopping.split: {split!r} is TARGET data. Early stopping on "
            f"a target domain is model selection using target information "
            f"(CLAUDE.md sec 3)."
        )
    if split != TUNING_SPLIT:
        raise ConfigError(
            f"train.early_stopping.split: {split!r} may not drive model selection; use "
            f"{TUNING_SPLIT!r}. pbc_test is held out for the source-domain report."
        )
    if not isinstance(es["patience"], int) or es["patience"] < 1:
        raise ConfigError("train.early_stopping.patience: expected a positive int")

    return TrainCfg(
        epochs=node["epochs"],
        batch_size=node["batch_size"],
        lr=float(node["lr"]),
        optimizer=node["optimizer"],
        weight_decay=float(node["weight_decay"]),
        class_weighting=node["class_weighting"],
        early_stopping=dict(es),
    )


def _build_eval(doc: dict, model_type: str, mapping: ClassMapping) -> EvalCfg:
    node = _check_keys(doc["eval"], EVAL_KEYS, EVAL_REQUIRED, "eval")

    domains = node["domains"]
    if not isinstance(domains, list) or not domains:
        raise ConfigError("eval.domains: expected a non-empty list")
    for d in domains:
        if d not in DATASETS:
            raise ConfigError(
                f"eval.domains: unknown dataset id {d!r}; known: {sorted(DATASETS)}")

    modes = node.get("attribute_modes")
    if model_type == "baseline":
        if modes is not None:
            raise ConfigError(
                "eval.attribute_modes: must be absent when model.type is 'baseline'; "
                "the baseline has no attribute layer to read soft or hard."
            )
    else:
        if not isinstance(modes, list) or not modes:
            raise ConfigError(
                f"eval.attribute_modes: required when model.type is {model_type!r}; "
                f"expected a non-empty list from {list(ATTRIBUTE_MODES)}"
            )
        bad = [m for m in modes if m not in ATTRIBUTE_MODES]
        if bad:
            raise ConfigError(
                f"eval.attribute_modes: {bad} not in {list(ATTRIBUTE_MODES)}. 'hard' is "
                f"eval-time argmax of the soft-trained model, not a separate run."
            )

    subsets = node["secondary_subsets"]
    if not isinstance(subsets, list):
        raise ConfigError("eval.secondary_subsets: expected a list")
    for name in subsets:
        spec = mapping.secondary_subsets.get(name)
        if spec is None:
            raise ConfigError(
                f"eval.secondary_subsets: {name!r} is not defined under "
                f"secondary_subsets in the class mapping"
            )
        if domain_of(spec["split"]) is Domain.TARGET:
            raise TargetDomainLeakage(
                f"eval.secondary_subsets: {name!r} is defined over target split "
                f"{spec['split']!r}; secondary subsets are source-only."
            )

    abst = _check_keys(node["abstention"], ABSTENTION_KEYS, ABSTENTION_KEYS,
                       "eval.abstention")
    split = abst["threshold_split"]
    if split not in DATASETS:
        raise ConfigError(
            f"eval.abstention.threshold_split: unknown dataset id {split!r}")
    if domain_of(split) is Domain.TARGET:
        raise TargetDomainLeakage(
            f"eval.abstention.threshold_split: {split!r} is TARGET data. The abstention "
            f"threshold is selected on the source val split only (CLAUDE.md sec 3)."
        )
    if split != TUNING_SPLIT:
        raise ConfigError(
            f"eval.abstention.threshold_split: {split!r} may not select the threshold; "
            f"use {TUNING_SPLIT!r}."
        )

    if not isinstance(node["ece_bins"], int) or node["ece_bins"] < 2:
        raise ConfigError("eval.ece_bins: expected an int >= 2")
    if node["macro_f1_over"] != "present_classes":
        raise ConfigError(
            f"eval.macro_f1_over: must be 'present_classes'; got "
            f"{node['macro_f1_over']!r}. raabin_testB holds only two of the five "
            f"classes, so macro-F1 is computed over the classes present in each eval "
            f"set and reported with that class list. A 5-class figure there would not "
            f"be comparable, and the choice is fixed here so it cannot be revisited "
            f"once target numbers are visible."
        )

    return EvalCfg(
        domains=list(domains),
        secondary_subsets=list(subsets),
        abstention=dict(abst),
        ece_bins=node["ece_bins"],
        macro_f1_over=node["macro_f1_over"],
        attribute_modes=list(modes) if modes else None,
    )


def _build_bootstrap(doc: dict) -> BootstrapCfg:
    node = _check_keys(doc["bootstrap"], BOOTSTRAP_KEYS, BOOTSTRAP_KEYS, "bootstrap")
    if not isinstance(node["enabled"], bool):
        raise ConfigError("bootstrap.enabled: expected a bool")
    if not isinstance(node["n"], int) or node["n"] < 1:
        raise ConfigError("bootstrap.n: expected a positive int")
    ci = node["ci"]
    if not isinstance(ci, (int, float)) or not 0 < float(ci) < 1:
        raise ConfigError("bootstrap.ci: expected a float in (0, 1)")
    if node["stratify_by"] != "class":
        raise ConfigError(
            f"bootstrap.stratify_by: must be 'class'; got {node['stratify_by']!r}. "
            f"basophil and eosinophil are rare, so unstratified resampling can drop a "
            f"class entirely from a replicate."
        )
    if node["resample_domains_independently"] is not True:
        raise ConfigError(
            "bootstrap.resample_domains_independently: must be true. The reported "
            "quantity is the source->target gap, and source and target are disjoint "
            "sets of images, so their replicates are drawn independently."
        )
    return BootstrapCfg(
        enabled=node["enabled"],
        n=node["n"],
        ci=float(ci),
        stratify_by=node["stratify_by"],
        resample_domains_independently=True,
    )


def load_config(path: str | Path) -> Config:
    path, doc = _load_yaml(path)
    _check_keys(doc, EXP_TOP, EXP_REQUIRED, str(path))

    if not isinstance(doc["run_id"], str) or not doc["run_id"]:
        raise ConfigError("run_id: expected a non-empty string")
    env = doc["env"]
    seed = _seed_of(doc)
    paths, paths_resolved = _build_paths(doc, env)

    data, data_resolved = _build_data(doc)
    model = _build_model(doc)
    bottleneck = _build_bottleneck(doc, model.type)
    train = _build_train(doc)
    eval_cfg = _build_eval(doc, model.type, data.class_mapping)
    bootstrap = _build_bootstrap(doc)

    resolved = {
        "run_id": doc["run_id"],
        "env": env,
        "seed": seed,
        "paths": paths_resolved,
        "data": data_resolved,
        "model": {"type": model.type, "backbone": model.backbone,
                  "backbone_norm": model.backbone_norm},
        "bottleneck": (None if bottleneck is None
                       else {"attribute_loss_weight": bottleneck.attribute_loss_weight}),
        "train": {
            "epochs": train.epochs, "batch_size": train.batch_size, "lr": train.lr,
            "optimizer": train.optimizer, "weight_decay": train.weight_decay,
            "class_weighting": train.class_weighting,
            "early_stopping": train.early_stopping,
        },
        "eval": {
            "domains": eval_cfg.domains,
            "attribute_modes": eval_cfg.attribute_modes,
            "secondary_subsets": eval_cfg.secondary_subsets,
            "abstention": eval_cfg.abstention,
            "ece_bins": eval_cfg.ece_bins,
            "macro_f1_over": eval_cfg.macro_f1_over,
        },
        "bootstrap": {
            "enabled": bootstrap.enabled, "n": bootstrap.n, "ci": bootstrap.ci,
            "stratify_by": bootstrap.stratify_by,
            "resample_domains_independently": bootstrap.resample_domains_independently,
        },
    }

    return Config(
        run_id=doc["run_id"], env=env, seed=seed, paths=paths, data=data, model=model,
        train=train, eval=eval_cfg, bootstrap=bootstrap, bottleneck=bottleneck,
        resolved=resolved,
    )


# --- seeding and metadata ----------------------------------------------------------------

def set_seed(seed: int) -> dict:
    """Seed python, numpy and torch, and force deterministic kernels.

    Returns what was set, for embedding in the run record. torch is imported lazily so
    the audit does not depend on it.
    """
    record: dict[str, Any] = {"seed": seed}
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
    except ImportError:
        record["numpy"] = None
    else:
        np.random.seed(seed)
        record["numpy"] = np.__version__

    try:
        import torch
    except ImportError:
        record["torch"] = None
        return record

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Required before use_deterministic_algorithms when CUDA/cuBLAS is in play.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True)
        record["deterministic_algorithms"] = True
    except Exception as exc:  # some ops have no deterministic implementation
        record["deterministic_algorithms"] = f"unavailable: {exc}"
    record["torch"] = torch.__version__
    record["cudnn_deterministic"] = True
    record["cudnn_benchmark"] = False
    return record


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()


def run_metadata() -> dict:
    """Git state and hardware, for results/<run_id>.json (CLAUDE.md sec 5)."""
    try:
        commit = _git("rev-parse", "HEAD")
        dirty = bool(_git("status", "--porcelain"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = "0" * 40, True

    meta: dict[str, Any] = {
        "git": {"commit": commit, "dirty": dirty},
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        import torch
    except ImportError:
        meta["torch"] = None
    else:
        meta["torch"] = torch.__version__
        meta["cuda"] = torch.version.cuda
        meta["device"] = (torch.cuda.get_device_name(0)
                          if torch.cuda.is_available() else "cpu")
    return meta
