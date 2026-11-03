"""Training. python -m src.train --config configs/<x>.yaml and nothing else.

This module can only reach pbc_train and pbc_val. It imports the two pinned loader
factories and never the evaluation one, so no out-of-domain data is constructible from
here at all - which is checked structurally by a test, not just by reading.

Checkpoint selection is the best pbc_val macro-F1, never the last epoch. Macro-F1 and
not accuracy: CLAUDE.md sec 6 rules accuracy out as a summary, and with basophil at
about 12% of pbc_train a majority-leaning model can look fine on accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

from src.config import load_config, resolve_device, run_metadata, set_seed
from src.data.domains import load_class_mapping
from src.data.label_space import load_label_space
from src.data.loaders import make_train_loader, make_val_loader
from src.losses import class_weights_from, compute_loss
from src.metrics import macro_f1_present, per_attribute_accuracy

SELECTION_METRIC = "val_macro_f1"


# --- selection and checkpoints ---------------------------------------------------------

def select_best_epoch(history: list[dict]) -> dict:
    """Highest val macro-F1; ties go to the earlier epoch.

    Taking the last epoch instead is the easiest way to report a worse result than the
    run actually achieved, since val macro-F1 usually peaks and then declines.
    """
    if not history:
        raise ValueError("cannot select a best epoch from an empty history")
    return min(history, key=lambda row: (-float(row[SELECTION_METRIC]), row["epoch"]))


def config_sha256(resolved: dict) -> str:
    """Stable hash of the resolved config, for the evaluation ledger."""
    canonical = json.dumps(resolved, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def checkpoint_payload(cfg, state_dict, history: list[dict], group_offsets) -> dict:
    """Everything needed to evaluate this checkpoint without any other file.

    The resolved config is embedded so evaluation reads it from here rather than from a
    YAML passed alongside, which is what stops a checkpoint being evaluated under
    preprocessing it was not trained with (CLAUDE.md sec 5). group_offsets is embedded
    so the 31 classifier inputs stay interpretable as (column, value) pairs.
    """
    best = select_best_epoch(history)
    meta = run_metadata(device=cfg.device)
    return {
        "state_dict": state_dict,
        "config": cfg.resolved,
        "config_sha256": config_sha256(cfg.resolved),
        "git": meta["git"],
        "environment": {k: v for k, v in meta.items() if k != "git"},
        "group_offsets": group_offsets,
        "history": history,
        "selected_epoch": best["epoch"],
        SELECTION_METRIC: float(best[SELECTION_METRIC]),
    }


def save_checkpoint(path: Path, cfg, mapping, label_space, history: list[dict],
                    state_dict=None, group_offsets=None) -> dict:
    """Write a checkpoint. Builds the model to obtain real parameter names if no state
    dict is supplied."""
    from src.models.build import build_model

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if state_dict is None or group_offsets is None:
        model = build_model(cfg, label_space)
        state_dict = model.state_dict() if state_dict is None else state_dict
        group_offsets = getattr(model, "group_offsets", []) if group_offsets is None \
            else group_offsets
    payload = checkpoint_payload(cfg, state_dict, history, group_offsets)
    torch.save(payload, path)
    return payload


# --- schedule ----------------------------------------------------------------------------

def build_optimizer_and_schedule(model, cfg):
    """AdamW with a short linear warmup into a cosine decay.

    Fixed a priori and identical for all three model types, so a difference between them
    cannot come from the optimisation schedule (CLAUDE.md sec 5).
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    warmup = cfg.train.warmup_epochs
    total = cfg.train.epochs

    def factor(epoch: int) -> float:
        if warmup and epoch < warmup:
            return (epoch + 1) / (warmup + 1)
        progress = (epoch - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# --- evaluation on the source val split ----------------------------------------------------

@torch.no_grad()
def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _validate(model, loader, cfg, class_weights, classes, label_space, device) -> dict:
    """pbc_val only. Called with the model in eval mode so batch statistics are frozen."""
    was_training = model.training
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    losses: list[float] = []
    class_losses: list[float] = []
    attribute_losses: list[float] = []
    attribute_true: list[list[int]] = []
    attribute_pred: list[list[int]] = []

    for images, labels, attributes in loader:
        images, labels = images.to(device), labels.to(device)
        if attributes is not None:
            attributes = attributes.to(device)
        output = model(images)
        total, parts = compute_loss(output, (labels, attributes), cfg, class_weights)
        losses.append(float(total))
        class_losses.append(float(parts["class"]))
        attribute_losses.append(float(parts["attribute"]))
        y_true.extend(int(v) for v in labels)
        y_pred.extend(int(v) for v in output.class_logits.argmax(dim=1))
        if output.attribute_logits is not None and attributes is not None:
            attribute_true.extend([int(v) for v in row] for row in attributes)
            attribute_pred.extend(
                [int(group.argmax(dim=0)) for group in row]
                for row in zip(*[logits for logits in output.attribute_logits]))
    if was_training:
        model.train()

    macro_f1, present = macro_f1_present(y_true, y_pred, classes)
    row = {
        SELECTION_METRIC: macro_f1,
        "val_present_classes": present,
        "val_loss": _mean(losses),
        # The two terms separately: one blended number cannot distinguish "the class
        # head is stuck" from "the attribute heads are stuck".
        "val_class_loss": _mean(class_losses),
        "val_attribute_loss_unscaled": _mean(attribute_losses),
        "n_val": len(y_true),
    }
    if attribute_true:
        # MONITORING ONLY, a declared exception to the pbc_test-only rule in CLAUDE.md
        # sec 6. The monitor_ prefix marks it unselectable, and select_best_epoch reads
        # SELECTION_METRIC and nothing else - there is a test for both.
        row["monitor_val_per_attribute"] = per_attribute_accuracy(
            attribute_true, attribute_pred, label_space)
    return row


# --- the loop -------------------------------------------------------------------------------

def train(cfg, mapping, label_space, out_dir: Path) -> dict:
    """Train on pbc_train, select on pbc_val, write the checkpoint and results JSON."""
    from src.models.build import build_model

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    seeding = set_seed(cfg.seed)
    device = resolve_device(cfg)

    train_loader = make_train_loader(cfg, mapping, label_space)
    val_loader = make_val_loader(cfg, mapping, label_space)
    class_weights = class_weights_from(train_loader.dataset, cfg, label_space)
    if class_weights is not None:
        class_weights = class_weights.to(device)

    model = build_model(cfg, label_space).to(device)
    optimizer, schedule = build_optimizer_and_schedule(model, cfg)
    classes = label_space.classes

    history: list[dict] = []
    seconds_per_epoch: list[float] = []
    best_state: dict | None = None
    best_metric = -math.inf
    patience = cfg.train.early_stopping["patience"]
    since_improved = 0
    early_stopped = False

    for epoch in range(cfg.train.epochs):
        epoch_started = time.time()
        model.train()
        batch_losses: list[float] = []
        batch_class_losses: list[float] = []
        batch_attribute_losses: list[float] = []
        for images, labels, attributes in train_loader:
            images, labels = images.to(device), labels.to(device)
            if attributes is not None:
                attributes = attributes.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            total, parts = compute_loss(output, (labels, attributes), cfg, class_weights)
            total.backward()
            optimizer.step()
            batch_losses.append(float(total))
            batch_class_losses.append(float(parts["class"]))
            batch_attribute_losses.append(float(parts["attribute"]))
        schedule.step()

        row = {"epoch": epoch,
               "train_loss": _mean(batch_losses),
               "train_class_loss": _mean(batch_class_losses),
               "train_attribute_loss_unscaled": _mean(batch_attribute_losses),
               "lr": optimizer.param_groups[0]["lr"]}
        row.update(_validate(model, val_loader, cfg, class_weights, classes,
                             label_space, device))
        seconds_per_epoch.append(time.time() - epoch_started)
        row["seconds"] = seconds_per_epoch[-1]
        history.append(row)

        if row[SELECTION_METRIC] > best_metric:
            best_metric = row[SELECTION_METRIC]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= patience:
                early_stopped = True
                break

    payload = checkpoint_payload(cfg, best_state or model.state_dict(), history,
                                 getattr(model, "group_offsets", []))
    torch.save(payload, out_dir / "best.pt")

    meta = run_metadata(device=cfg.device)
    result = {
        "run_id": cfg.run_id,
        "seed": cfg.seed,
        "config": cfg.resolved,
        "config_sha256": payload["config_sha256"],
        "git": meta["git"],
        "hardware": {k: v for k, v in meta.items() if k != "git"},
        "seeding": seeding,
        "weight_load_report": model.weight_load_report,
        "class_weights": None if class_weights is None else class_weights.tolist(),
        "history": history,
        "selected_epoch": payload["selected_epoch"],
        SELECTION_METRIC: payload[SELECTION_METRIC],
        "n_train": len(train_loader.dataset),
        "n_val": len(val_loader.dataset),
        "epochs_run": len(history),
        # A run that stopped at 7 of 30 must not read as having used the declared
        # budget, so this is recorded rather than inferred from epochs_run.
        "early_stopped": early_stopped,
        "stopped_after_epoch": (history[-1]["epoch"] if early_stopped else None),
        "seconds_per_epoch": seconds_per_epoch,
        "wall_clock_seconds": time.time() - started,
        "checkpoint": str(out_dir / "best.pt"),
    }
    (out_dir / f"{cfg.run_id}_seed{cfg.seed}.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train one model on the source domain.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    mapping = load_class_mapping(cfg.data.class_mapping_path)
    label_space = load_label_space(cfg.resolved["data"]["label_space_path"])
    out_dir = cfg.paths.results_dir / f"{cfg.run_id}_seed{cfg.seed}"

    result = train(cfg, mapping, label_space, out_dir)
    print(f"{cfg.run_id} seed {cfg.seed}: "
          f"best {SELECTION_METRIC} {result[SELECTION_METRIC]:.4f} at epoch "
          f"{result['selected_epoch']} of {result['epochs_run']}")
    for i, seconds in enumerate(result["seconds_per_epoch"]):
        print(f"  epoch {i}: {seconds:.1f}s")
    print(f"  wall clock {result['wall_clock_seconds']:.1f}s -> {result['checkpoint']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
