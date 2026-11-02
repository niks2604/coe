"""Evaluation. python -m src.eval --checkpoint <path>

The checkpoint is the only input. Its embedded resolved config is what drives
preprocessing, so a checkpoint cannot be evaluated under settings it was not trained
with; there is deliberately no way to pass a YAML here (CLAUDE.md sec 5).

Three things are enforced rather than intended:

* the model is in eval mode under no_grad for every batch, and the normalisation buffers
  are hashed before and after. Training mode on out-of-domain data would update the
  running statistics, which is test-time adaptation, which is leakage (sec 3);
* held-out domains go through the ledger, so a repeat is refused;
* predictions are an argmax over all five classes everywhere. The gap's source side is
  restricted instead, by true label (sec 6).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

from src.config import Config, load_config, run_metadata
from src.data.datasets import build_secondary_subset
from src.data.domains import Domain, domain_of, load_class_mapping
from src.data.label_space import load_label_space
from src.data.loaders import make_eval_loader
from src.ledger import (
    LEDGER_PATH,
    LedgerError,
    append_entry,
    assert_not_evaluated,
    checkpoint_hash,
    find_entries,
    is_ledgered,
)
from src.metrics import (
    accuracy_vs_coverage,
    bootstrap_ci,
    expected_calibration_error,
    gap_with_ci,
    macro_f1_present,
    per_attribute_accuracy,
    per_class_recall,
    present_classes,
    restrict_to_classes,
    select_abstention_thresholds,
)
from src.models.build import build_model

DRY_RUN_DOMAIN = "pbc_val"
SOURCE_TEST = "pbc_test"


class EvalError(Exception):
    """The checkpoint cannot be evaluated as given."""


# --- checkpoint ---------------------------------------------------------------------

def _rebuild_config(resolved: dict) -> Config:
    """Rebuild a Config from a checkpoint's embedded resolved copy.

    The embedded form is flat (one env already selected, mapping inlined), so it is
    rewritten into the two-env shape load_config expects and re-validated - every
    leakage check in src/config.py therefore runs again at evaluation time.
    """
    import tempfile

    import yaml

    doc = json.loads(json.dumps(resolved))
    paths = doc["paths"]
    doc["paths"] = {"local": dict(paths), "kaggle": dict(paths)}
    doc["env"] = "local"
    data = doc["data"]
    data["class_mapping"] = data.pop("class_mapping_path")
    data["label_space"] = data.pop("label_space_path")
    data.pop("class_mapping", None) if False else None
    doc["data"] = {k: v for k, v in data.items()
                   if k in {"class_mapping", "label_space", "source", "image_size",
                            "subset_n_images", "normalization"}}
    if doc.get("bottleneck") is None:
        doc.pop("bottleneck", None)
    if doc["eval"].get("attribute_modes") is None:
        doc["eval"].pop("attribute_modes", None)

    path = Path(tempfile.mkdtemp()) / "embedded.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return load_config(path)


def load_checkpoint(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise EvalError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "config" not in payload:
        raise EvalError(
            f"{path} has no embedded config. Evaluation reads the resolved config from "
            f"the checkpoint so it cannot be run under different preprocessing; a "
            f"checkpoint saved without one cannot be evaluated."
        )
    if "state_dict" not in payload:
        raise EvalError(f"{path} has no state_dict")

    cfg = _rebuild_config(payload["config"])
    mapping = load_class_mapping(cfg.data.class_mapping_path)
    label_space = cfg.data.label_space

    model = build_model(cfg, label_space)
    try:
        model.load_state_dict(payload["state_dict"])
    except RuntimeError as exc:
        raise EvalError(
            f"{path}: the saved weights do not match the model its embedded config "
            f"describes ({exc})"
        ) from None
    model.eval()
    return cfg, mapping, label_space, model, payload


def _buffer_hash(model) -> str:
    """Hash of every buffer, so a running-statistic update cannot go unnoticed."""
    digest = hashlib.sha256()
    for name, buffer in sorted(model.named_buffers()):
        digest.update(name.encode("utf-8"))
        digest.update(buffer.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


# --- one pass -------------------------------------------------------------------------

@torch.no_grad()
def _collect(model, loader, modes: list[str], observed: dict) -> dict:
    """Run a loader once and gather predictions for each attribute mode."""
    out = {mode: {"y_true": [], "y_pred": [], "confidence": []} for mode in modes}
    attribute_true: list[list[int]] = []
    attribute_pred: list[list[int]] = []

    for images, labels, attributes in loader:
        observed["model_training_mode_seen"] |= bool(model.training)
        observed["grad_enabled_seen"] |= bool(torch.is_grad_enabled())

        if hasattr(model, "bottleneck_vector"):
            attribute_logits = model.attribute_logits(images)
            for mode in modes:
                vector = model.bottleneck_vector(attribute_logits, mode=mode)
                logits = model.classify(vector)
                probabilities = torch.softmax(logits, dim=1)
                out[mode]["y_true"].extend(int(v) for v in labels)
                out[mode]["y_pred"].extend(int(v) for v in logits.argmax(dim=1))
                out[mode]["confidence"].extend(float(v) for v in probabilities.max(dim=1).values)
            if attributes is not None:
                attribute_true.extend([int(v) for v in row] for row in attributes)
                attribute_pred.extend(
                    [int(group.argmax(dim=0)) for group in row]
                    for row in zip(*[logits for logits in attribute_logits]))
        else:
            logits = model(images).class_logits
            probabilities = torch.softmax(logits, dim=1)
            for mode in modes:
                out[mode]["y_true"].extend(int(v) for v in labels)
                out[mode]["y_pred"].extend(int(v) for v in logits.argmax(dim=1))
                out[mode]["confidence"].extend(float(v) for v in probabilities.max(dim=1).values)

    return {"modes": out,
            "attribute_true": attribute_true or None,
            "attribute_pred": attribute_pred or None}


def _score(cfg, predictions: dict, classes: list[str], thresholds) -> dict:
    y_true, y_pred = predictions["y_true"], predictions["y_pred"]
    confidence = predictions["confidence"]
    correct = [t == p for t, p in zip(y_true, y_pred)]

    macro_f1, present = macro_f1_present(y_true, y_pred, classes)
    scored = {
        "n": len(y_true),
        "macro_f1": macro_f1,
        "macro_f1_classes": present,
        "per_class_recall": per_class_recall(y_true, y_pred, classes),
        "ece": expected_calibration_error(confidence, correct, cfg.eval.ece_bins),
        "accuracy": sum(correct) / len(correct) if correct else None,
    }
    if cfg.bootstrap.enabled:
        scored["macro_f1_ci"] = bootstrap_ci(
            lambda t, p: macro_f1_present(t, p, classes)[0], y_true, y_pred,
            n=cfg.bootstrap.n, ci=cfg.bootstrap.ci, seed=cfg.seed,
            stratify_by=cfg.bootstrap.stratify_by)
    if thresholds is not None:
        scored["coverage"] = accuracy_vs_coverage(confidence, correct, thresholds)
    return scored


# --- the script --------------------------------------------------------------------------

def evaluate_checkpoint(checkpoint: str | Path, ledger_path: str | Path = LEDGER_PATH,
                        dry_run: bool = False, override_reason: str | None = None) -> dict:
    started = time.time()
    checkpoint = Path(checkpoint)
    cfg, mapping, label_space, model, payload = load_checkpoint(checkpoint)
    classes = label_space.classes
    modes = cfg.eval.attribute_modes or ["soft"]
    observed = {"model_training_mode_seen": False, "grad_enabled_seen": False}
    before = _buffer_hash(model)

    # pbc_val only, no ledger: the tuning split is read constantly by design.
    domains = [DRY_RUN_DOMAIN] if dry_run else list(cfg.eval.domains)

    sha = checkpoint_hash(checkpoint)
    if not dry_run:
        # An override is permission for the domains that have been looked at before,
        # not a blanket flag. Applying it to a domain with no prior entry would make a
        # configured-but-absent domain reject an otherwise legitimate override.
        priors = {domain: bool(find_entries(ledger_path, sha, domain))
                  for domain in domains if is_ledgered(domain)}
        if override_reason is not None and not any(priors.values()):
            raise LedgerError(
                f"--override-reason was given but none of the ledgered domains "
                f"{sorted(priors)} has a prior evaluation of this checkpoint to "
                f"override. Remove the flag."
            )
        for domain in domains:
            assert_not_evaluated(ledger_path, sha, domain,
                                 override_reason if priors.get(domain) else None)

    # Thresholds come from pbc_val and are then applied unchanged everywhere.
    val_loader = make_eval_loader(cfg, DRY_RUN_DOMAIN, mapping, label_space)
    val = _collect(model, val_loader, modes, observed)
    thresholds = {
        mode: select_abstention_thresholds(
            DRY_RUN_DOMAIN, val["modes"][mode]["confidence"],
            cfg.eval.abstention["coverage_grid"])
        for mode in modes
    }

    results: dict[str, dict] = {}
    raw: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    per_attribute: dict[str, dict] = {}
    evaluated: list[str] = []

    for domain in domains:
        if domain == DRY_RUN_DOMAIN:
            collected = val
        else:
            loader = make_eval_loader(cfg, domain, mapping, label_space)
            if loader is None:
                skipped[domain] = "not present on disk"
                continue
            collected = _collect(model, loader, modes, observed)

        results[domain] = {mode: _score(cfg, collected["modes"][mode], classes,
                                        thresholds[mode]) for mode in modes}
        raw[domain] = collected["modes"]
        evaluated.append(domain)

        # Attribute labels exist for source splits only.
        if collected["attribute_true"] is not None:
            per_attribute[domain] = per_attribute_accuracy(
                collected["attribute_true"], collected["attribute_pred"], label_space)

    # Secondary subsets, source only, defined in the class mapping.
    secondary: dict[str, dict] = {}
    if not dry_run and SOURCE_TEST in evaluated:
        source_loader = make_eval_loader(cfg, SOURCE_TEST, mapping, label_space)
        for name in cfg.eval.secondary_subsets:
            view = build_secondary_subset(source_loader.dataset, mapping, name)
            keep = set(view.indices)
            secondary[name] = {
                "n": len(view),
                "modes": {
                    mode: _score(cfg, {
                        k: [v for i, v in enumerate(raw[SOURCE_TEST][mode][k])
                            if source_loader.dataset.indices[i] in keep]
                        for k in ("y_true", "y_pred", "confidence")
                    }, classes, thresholds[mode])
                    for mode in modes
                },
            }

    # Gaps. The source side is restricted to the classes present in each target domain,
    # by true label, with the restriction named (CLAUDE.md sec 6).
    gaps: dict[str, dict] = {}
    if SOURCE_TEST in evaluated:
        for domain in evaluated:
            if domain_of(domain) is not Domain.TARGET:
                continue
            gaps[domain] = {"source_domain": SOURCE_TEST, "modes": {}}
            for mode in modes:
                target = raw[domain][mode]
                keep = present_classes(target["y_true"], classes)
                source_true, source_pred, kept = restrict_to_classes(
                    raw[SOURCE_TEST][mode]["y_true"], raw[SOURCE_TEST][mode]["y_pred"],
                    keep, classes)
                gaps[domain]["source_classes"] = kept
                gaps[domain]["modes"][mode] = gap_with_ci(
                    (source_true, source_pred), (target["y_true"], target["y_pred"]),
                    lambda t, p: macro_f1_present(t, p, classes)[0],
                    n=cfg.bootstrap.n, ci=cfg.bootstrap.ci, seed=cfg.seed,
                    stratify_by=cfg.bootstrap.stratify_by)

    after = _buffer_hash(model)
    meta = run_metadata()
    out = {
        "run_id": cfg.run_id,
        "seed": cfg.seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha,
        "config_sha256": payload.get("config_sha256"),
        "config": cfg.resolved,
        "git": meta["git"],
        "hardware": {k: v for k, v in meta.items() if k != "git"},
        "dry_run": dry_run,
        "override_reason": override_reason,
        "attribute_modes": modes,
        "domains": evaluated,
        "skipped_domains": skipped,
        "results": results,
        "secondary_subsets": secondary,
        "per_attribute": per_attribute,
        "gaps": gaps,
        "abstention_thresholds": {m: t for m, t in thresholds.items()},
        "buffer_sha256_before": before,
        "buffer_sha256_after": after,
        "wall_clock_seconds": time.time() - started,
        **observed,
    }

    if not dry_run:
        for domain in evaluated:
            if not is_ledgered(domain):
                continue
            entry = {"checkpoint_sha256": sha, "domain": domain, "run_id": cfg.run_id,
                     "seed": cfg.seed, "config_sha256": payload.get("config_sha256"),
                     "selected_epoch": payload.get("selected_epoch")}
            if override_reason is not None:
                entry["override_reason"] = override_reason
            append_entry(ledger_path, entry)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved checkpoint. The checkpoint carries its own config.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ledger", default=str(LEDGER_PATH))
    parser.add_argument("--dry-run", action="store_true",
                        help="evaluate pbc_val only; writes no ledger entry")
    parser.add_argument("--override-reason", default=None,
                        help="re-evaluate a ledgered domain; recorded in the ledger")
    args = parser.parse_args(argv)

    if args.override_reason is not None and not args.override_reason.strip():
        parser.error("--override-reason must be a non-empty reason")

    out = evaluate_checkpoint(args.checkpoint, ledger_path=args.ledger,
                              dry_run=args.dry_run,
                              override_reason=args.override_reason)

    label = "DRY RUN (pbc_val only)" if out["dry_run"] else "EVALUATION"
    print(f"{label}  {out['run_id']} seed {out['seed']}  {out['checkpoint_sha256'][:12]}")
    for domain in out["domains"]:
        for mode, scored in out["results"][domain].items():
            print(f"  {domain:<14} {mode:<5} macro-F1 {scored['macro_f1']:.4f} "
                  f"over {scored['macro_f1_classes']}  ECE {scored['ece']:.4f}  "
                  f"n={scored['n']}")
    for domain, reason in out["skipped_domains"].items():
        print(f"  {domain:<14} skipped: {reason}")
    print(f"  buffers unchanged: {out['buffer_sha256_before'] == out['buffer_sha256_after']}"
          f"   wall clock {out['wall_clock_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
