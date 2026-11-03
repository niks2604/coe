"""Overfit one fixed batch. A diagnostic, not an experiment.

Purpose: separate "the bottleneck is a bad representation" from "the bottleneck does not
train". A model with enough capacity must be able to drive the loss on 32 fixed images to
near zero. If it cannot, the smoke-run result says nothing about representations.

This is deliberately NOT a configurable experiment. It changes no protocol value: the
model, loss, lr, weight decay and batch size all come from the committed config, and
src.losses.compute_loss is used unchanged, because that function is the thing under
suspicion. The only things this script adds are a step count and the decision to hold one
batch fixed, both recorded in the output.

Reads pbc_train only, through the same pinned factory src.train uses. There is no code
path from here to a target domain, which a test in tests/test_train.py checks
structurally.

Run: python -m scripts.overfit_one_batch --config configs/bottleneck.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from src.config import load_config, run_metadata, set_seed
from src.data.domains import load_class_mapping
from src.data.label_space import load_label_space
from src.data.loaders import make_train_loader
from src.losses import class_weights_from, compute_loss
from src.metrics import per_attribute_accuracy
from src.models.build import build_model

DEFAULT_MAX_STEPS = 200
DOWNGRADED_STEPS = 100
# Above this, 200 steps x 3 models does not finish in a reasonable sitting on CPU, and
# the brief authorised 100 instead. Measured over the first few steps, never guessed.
SLOW_SECONDS_PER_STEP = 10.0
PROBE_STEPS = 5


def first_batch(cfg, mapping, label_space):
    """One batch from pbc_train, then frozen.

    The train loader augments, so the batch is augmented once and reused unchanged for
    every step after. That is what makes this an overfitting test rather than a very
    short training run.
    """
    loader = make_train_loader(cfg, mapping, label_space)
    images, labels, attributes = next(iter(loader))
    images = images.detach().clone()
    labels = labels.detach().clone()
    if attributes is not None:
        attributes = attributes.detach().clone()
    return loader, images, labels, attributes


def batch_accuracy(class_logits, labels) -> float:
    return float((class_logits.argmax(dim=1) == labels).float().mean())


def attribute_report(output, attributes, label_space):
    """Per-attribute accuracy on the fixed batch, or None for a model without heads."""
    if output.attribute_logits is None or attributes is None:
        return None
    predicted = torch.stack([logits.argmax(dim=1) for logits in output.attribute_logits],
                            dim=1)
    return per_attribute_accuracy(
        [[int(v) for v in row] for row in attributes],
        [[int(v) for v in row] for row in predicted],
        label_space,
    )


def overfit(cfg, mapping, label_space, max_steps: int) -> dict:
    seeding = set_seed(cfg.seed)
    loader, images, labels, attributes = first_batch(cfg, mapping, label_space)
    class_weights = class_weights_from(loader.dataset, cfg, label_space)

    model = build_model(cfg, label_space)
    # AdamW at the config's own lr and weight decay. No lr schedule: cosine is defined
    # over cfg.train.epochs and stepping it here would decay the lr for a reason that has
    # nothing to do with this diagnostic.
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)

    model.train()
    history: list[dict] = []
    started = time.time()
    downgraded = None
    step = 0
    while step < max_steps:
        step_started = time.time()
        optimizer.zero_grad(set_to_none=True)
        output = model(images)
        total, parts = compute_loss(output, (labels, attributes), cfg, class_weights)
        total.backward()
        optimizer.step()

        history.append({
            "step": step,
            "total": float(parts["total"]),
            "class": float(parts["class"]),
            "attribute_unscaled": float(parts["attribute"]),
            "accuracy": batch_accuracy(output.class_logits, labels),
            "seconds": time.time() - step_started,
        })
        step += 1

        if step == PROBE_STEPS and max_steps > DOWNGRADED_STEPS:
            per_step = sum(r["seconds"] for r in history) / len(history)
            if per_step > SLOW_SECONDS_PER_STEP:
                downgraded = (f"{per_step:.1f}s/step over the first {PROBE_STEPS} steps "
                              f"exceeds {SLOW_SECONDS_PER_STEP}s, so the step budget "
                              f"dropped from {max_steps} to {DOWNGRADED_STEPS}")
                max_steps = DOWNGRADED_STEPS

    # Final pass in train mode: this is a memorisation check, and eval mode would fold in
    # BatchNorm running statistics estimated from one repeated batch, which is a separate
    # question. Both are reported.
    with torch.no_grad():
        model.train()
        train_mode = model(images)
        model.eval()
        eval_mode = model(images)

    return {
        "run_id": cfg.run_id,
        "model_type": cfg.model.type,
        "attribute_loss_weight": (None if cfg.bottleneck is None
                                  else cfg.bottleneck.attribute_loss_weight),
        "lr": cfg.train.lr,
        "weight_decay": cfg.train.weight_decay,
        "batch_size": int(images.shape[0]),
        "class_weighting": cfg.train.class_weighting,
        "class_weights": None if class_weights is None else class_weights.tolist(),
        "steps_run": len(history),
        "step_budget_downgraded": downgraded,
        "device": "cpu",
        "seeding": seeding,
        "git": run_metadata()["git"],
        "label_counts": {label_space.classes[i]: int((labels == i).sum())
                         for i in range(len(label_space.classes))},
        "first_step": history[0],
        "last_step": history[-1],
        "history": history,
        "final": {
            "train_mode_accuracy": batch_accuracy(train_mode.class_logits, labels),
            "eval_mode_accuracy": batch_accuracy(eval_mode.class_logits, labels),
            "per_attribute": attribute_report(train_mode, attributes, label_space),
        },
        "wall_clock_seconds": time.time() - started,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    mapping = load_class_mapping(cfg.data.class_mapping_path)
    label_space = load_label_space(cfg.resolved["data"]["label_space_path"])

    report = overfit(cfg, mapping, label_space, args.max_steps)

    out_dir = cfg.paths.results_dir / "overfit_one_batch"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg.run_id}_seed{cfg.seed}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str) + "\n",
                        encoding="utf-8")

    first, last = report["first_step"], report["last_step"]
    print(f"{report['run_id']}  type={report['model_type']}  "
          f"lambda={report['attribute_loss_weight']}  steps={report['steps_run']}")
    if report["step_budget_downgraded"]:
        print(f"  NOTE {report['step_budget_downgraded']}")
    print(f"  step 0     total {first['total']:.4f}  class {first['class']:.4f}  "
          f"attr(unscaled) {first['attribute_unscaled']:.4f}  acc {first['accuracy']:.3f}")
    print(f"  step {last['step']:<5} total {last['total']:.4f}  class {last['class']:.4f}  "
          f"attr(unscaled) {last['attribute_unscaled']:.4f}  acc {last['accuracy']:.3f}")
    print(f"  final batch accuracy  train-mode {report['final']['train_mode_accuracy']:.3f}"
          f"  eval-mode {report['final']['eval_mode_accuracy']:.3f}")
    per_attribute = report["final"]["per_attribute"]
    if per_attribute is None:
        print("  per-attribute: n/a (no attribute heads)")
    else:
        for column, row in per_attribute.items():
            print(f"    {column:<30} acc {row['accuracy']:.3f}  "
                  f"balanced {row['balanced_accuracy']:.3f}  "
                  f"values_present {row['n_values_present']}")
    print(f"  wall clock {report['wall_clock_seconds']:.1f}s -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
