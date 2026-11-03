"""Can Linear(31,5) learn the class from the attribute code at all, in the real budget?

The overfit-one-batch diagnostic showed the bottleneck's attribute heads fit perfectly
while its class pathway stalled. That leaves two candidates: the saturated softmax
starves the class gradient, or the final Linear(31,5) is simply too slow to fit under
the declared lr, weight decay and schedule. This isolates the second one.

No images, no backbone, no attribute heads, and nothing read but pbc_train. The input is
the GROUND-TRUTH WBCAtt attribute code, one-hot encoded to 31 dimensions - that is the
best case the bottleneck could ever hand its classifier, since the heads cannot do better
than the labels they are trained on.

Every optimisation setting comes from the committed config via the same
build_optimizer_and_schedule that src.train uses, so the step budget and learning-rate
curve are the declared ones and not a reconstruction of them.

Run: python -m scripts.classifier_capacity --config configs/bottleneck.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from src.config import load_config, run_metadata, set_seed
from src.data.domains import load_class_mapping, resolve_class
from src.data.label_space import load_label_space
from src.train import build_optimizer_and_schedule

REPORT_AT_STEPS = (200, 1000, 2000)


def one_hot_pbc_train(cfg, mapping, label_space):
    """pbc_train's ground-truth attribute codes as a 31-d one-hot matrix, and labels.

    pbc_train only: pbc_val is the tuning split and pbc_test is held out, and neither is
    needed to answer a question about the classifier's capacity.
    """
    path = Path(cfg.resolved["paths"]["wbcatt_dir"]) / "pbc_attr_v1_train.csv"
    columns = label_space.attribute_columns
    offsets, at = {}, 0
    for column in columns:
        offsets[column] = at
        at += len(label_space.attributes[column])

    rows = list(csv.DictReader(path.open()))
    x = torch.zeros(len(rows), at)
    y = torch.zeros(len(rows), dtype=torch.long)
    for i, row in enumerate(rows):
        for column in columns:
            index = label_space.attribute_index(column, row[column])
            x[i, offsets[column] + index] = 1.0
        y[i] = label_space.class_index(resolve_class(mapping, "wbcatt", row["label"]))
    return x, y, path.name


@torch.no_grad()
def full_train_metrics(model, x, y) -> tuple[float, float]:
    logits = model(x)
    loss = float(F.cross_entropy(logits, y))      # unweighted: class_weighting is none
    accuracy = float((logits.argmax(dim=1) == y).float().mean())
    return loss, accuracy


def run(cfg, mapping, label_space) -> dict:
    seeding = set_seed(cfg.seed)
    x, y, source = one_hot_pbc_train(cfg, mapping, label_space)
    n = x.shape[0]

    model = nn.Linear(x.shape[1], len(label_space.classes), bias=True)
    optimizer, schedule = build_optimizer_and_schedule(model, cfg)

    batch_size = cfg.train.batch_size
    steps_per_epoch = math.ceil(n / batch_size)   # drop_last=False, as in make_train_loader
    total_steps = steps_per_epoch * cfg.train.epochs

    generator = torch.Generator().manual_seed(cfg.seed)
    checkpoints: dict[int, dict] = {}
    step = 0
    for epoch in range(cfg.train.epochs):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[batch]), y[batch])
            loss.backward()
            optimizer.step()
            step += 1

            if step in REPORT_AT_STEPS:
                full_loss, accuracy = full_train_metrics(model, x, y)
                checkpoints[step] = {
                    "step": step, "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                    "train_loss": full_loss, "train_accuracy": accuracy}
        schedule.step()

    full_loss, accuracy = full_train_metrics(model, x, y)
    checkpoints[step] = {
        "step": step, "epoch": cfg.train.epochs - 1,
        "lr": optimizer.param_groups[0]["lr"],
        "train_loss": full_loss, "train_accuracy": accuracy}

    return {
        "run_id": cfg.run_id,
        "source_csv": source,
        "n_train": n,
        "input_dim": x.shape[1],
        "n_classes": len(label_space.classes),
        "parameters": sum(p.numel() for p in model.parameters()),
        "declared": {
            "optimizer": cfg.train.optimizer, "lr": cfg.train.lr,
            "weight_decay": cfg.train.weight_decay, "batch_size": batch_size,
            "lr_schedule": cfg.train.lr_schedule,
            "warmup_epochs": cfg.train.warmup_epochs, "epochs": cfg.train.epochs,
            "class_weighting": cfg.train.class_weighting},
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "device": "cpu",
        "seeding": seeding,
        "git": run_metadata()["git"],
        "checkpoints": [checkpoints[k] for k in sorted(checkpoints)],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    mapping = load_class_mapping(cfg.data.class_mapping_path)
    label_space = load_label_space(cfg.resolved["data"]["label_space_path"])
    report = run(cfg, mapping, label_space)

    out_dir = cfg.paths.results_dir / "classifier_capacity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg.run_id}_seed{cfg.seed}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")

    print(f"Linear({report['input_dim']},{report['n_classes']}) on ground-truth "
          f"attribute codes, {report['n_train']} pbc_train rows, "
          f"{report['parameters']} parameters")
    print(f"  declared: {report['declared']}")
    print(f"  steps/epoch {report['steps_per_epoch']}  total steps {report['total_steps']}")
    for row in report["checkpoints"]:
        print(f"  step {row['step']:>5} (epoch {row['epoch']:>2})  lr {row['lr']:.3e}  "
              f"loss {row['train_loss']:.4f}  train acc {row['train_accuracy']:.4f}")
    print(f"  -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
