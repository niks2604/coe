"""Per-image dumps on pbc_test and the three Raabin domains, for post-hoc analysis.

Permitted by the scoped CLAUDE.md sec 3 amendment of 2026-11-19, which covers the nine
fixed headline checkpoints of the completed Raabin evaluation and nothing else. What
may be retained is enumerated there and enforced here by ALLOWED_ARRAYS: class logits,
attribute probabilities, image ids, true class. Pixels and every pixel-derived
statistic - size, mode, channel statistics, file hashes - stay forbidden, and a test
asserts the written fields against the allowlist rather than trusting this file.

Post-hoc. The sec 6 verdict was fixed on the records in eval_records/ before the
amendment existed. Nothing produced here may train, select, tune or design any arm.

Each dump appends one line to analysis_ledger.jsonl. That file is an audit trail for
the dumps and confers no permission to evaluate anything; the evaluation's own ledger
is a different file and is never opened here.

The device defaults to the one the checkpoint records, because these dumps are checked
against the ledgered evaluation and a different device moves the logits in the last
bits. An unavailable device is a hard error, never a silent fall back.

    python -m scripts.dump_target_predictions --checkpoints DIR --out analysis/
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import resolve_device, run_metadata                      # noqa: E402
from src.data.loaders import eval_loader_unavailable, make_eval_loader   # noqa: E402
from src.eval import _buffer_hash, _selection_role, load_checkpoint      # noqa: E402
from src.ledger import checkpoint_hash                                   # noqa: E402

DOMAINS = ("pbc_test", "raabin_train", "raabin_testA", "raabin_testB")

ANALYSIS_LEDGER = REPO / "analysis_ledger.jsonl"

AMENDMENT = ("CLAUDE.md sec 3 scoped amendment, 2026-11-19: per-image class logits, "
             "attribute probabilities, image ids and true class labels may be retained "
             "for the nine headline checkpoints. Pixels and pixel-derived statistics "
             "remain forbidden. Post-hoc; selects nothing.")

# Exhaustive. Anything not named here must not reach a file, and a test enforces it.
ALLOWED_ARRAYS = (
    "image_ids",                   # identifiers, not pixels
    "y_true",                      # the label, which the amendment names
    "class_logits",                # the model's reading of the image
    "attribute_probs",             # likewise, per head
    "attribute_group_offsets",     # the 11-head layout, from the label space
    "attribute_true",              # WBCAtt labels; source domains only
    "meta",
)

# Exhaustive, like ALLOWED_ARRAYS. Enforced at write time so a field cannot be added
# without being named here first. Note that the attribute vocabulary legitimately
# contains names like "cell_size": a WBCAtt attribute is a hematologist's reading of
# the cell, not a statistic of the image file, which is what the amendment forbids.
ALLOWED_META = (
    "domain", "run_id", "seed", "model_type", "checkpoint", "checkpoint_sha256",
    "config_sha256", "selection_role", "selection_rule", "git", "hardware",
    "device_used", "device_in_config", "n", "classes", "attribute_names",
    "attribute_values", "has_attributes", "attribute_loss_weight",
    "attributes_interpretable", "attributes_note", "class_logits_mode",
    "buffer_sha256_before", "buffer_sha256_after", "amendment",
    "model_training_mode_seen", "grad_enabled_seen",
)

CONTROL_NOTE = (
    "UNINTERPRETABLE: these attribute heads are unsupervised. The control trains with "
    "attribute_loss_weight 0, so nothing ever supervised them. They are not attribute "
    "predictions and must not be read as morphology.")


class DumpError(Exception):
    """A domain outside the amendment, or an output that already exists."""


def output_path(cfg, out_dir: str | Path, domain: str) -> Path:
    return Path(out_dir) / f"{cfg.run_id}_seed{cfg.seed}_{domain}.npz"


def _require_covered(domain: str) -> None:
    if domain not in DOMAINS:
        raise DumpError(
            f"domain {domain!r} is not covered. The amendment names {list(DOMAINS)} "
            f"and nothing else; pbc_val has its own script "
            f"(scripts/dump_predictions.py) and lisc is out of scope.")


def _image_ids(dataset) -> list[str]:
    """Identifiers, never pixels. The source dataset carries WBCAtt's img_name; the
    target one is folder-backed, so the class folder and file name identify a row."""
    if hasattr(dataset, "img_names"):
        return [dataset.img_names[j] for j in dataset.indices]
    return [f"{dataset.paths[j].parent.name}/{dataset.paths[j].name}"
            for j in dataset.indices]


@torch.no_grad()
def dump_checkpoint(checkpoint: str | Path, out_dir: str | Path, domain: str,
                    ledger_path: str | Path = ANALYSIS_LEDGER,
                    device: str | None = None) -> Path:
    _require_covered(domain)

    checkpoint = Path(checkpoint)
    cfg, mapping, label_space, model, payload = load_checkpoint(checkpoint)
    # The same gate src.eval applies, via the same helper. The amendment is scoped to
    # the headline checkpoints, so no override is accepted here either.
    selection_role, _ = _selection_role(checkpoint, payload, None)

    out = output_path(cfg, out_dir, domain)
    if out.exists():
        raise DumpError(
            f"{out} already exists and will not be overwritten. Move it aside if you "
            f"mean to replace it. Nothing was written and no ledger line was added.")

    unavailable = eval_loader_unavailable(cfg, domain)
    if unavailable is not None:
        raise DumpError(f"{domain} cannot be built: {unavailable}")

    resolved = resolve_device(cfg) if device is None else torch.device(device)
    model.to(resolved)
    model.eval()
    before = _buffer_hash(model)

    loader = make_eval_loader(cfg, domain, mapping, label_space)
    image_ids = _image_ids(loader.dataset)

    observed = {"model_training_mode_seen": False, "grad_enabled_seen": False}
    has_attributes = hasattr(model, "bottleneck_vector")
    class_logits, attribute_probs, attribute_true, y_true = [], [], [], []

    for images, labels, attributes in loader:
        images = images.to(resolved, non_blocking=True)
        observed["model_training_mode_seen"] |= bool(model.training)
        observed["grad_enabled_seen"] |= bool(torch.is_grad_enabled())
        if has_attributes:
            heads = model.attribute_logits(images)
            logits = model.classify(model.bottleneck_vector(heads, mode="soft"))
            attribute_probs.append(
                torch.cat([torch.softmax(h, dim=1) for h in heads], dim=1)
                .cpu().numpy().astype(np.float32))
        else:
            logits = model(images).class_logits
        class_logits.append(logits.cpu().numpy().astype(np.float32))
        y_true.extend(int(v) for v in labels)
        if attributes is not None:                 # source domains only
            attribute_true.append(attributes.cpu().numpy().astype(np.int16))

    after = _buffer_hash(model)
    if before != after:
        raise DumpError(
            f"{checkpoint}: buffers changed during the pass ({before[:12]} -> "
            f"{after[:12]}). Nothing was written and no ledger line was added.")

    offsets = [0]
    for values in label_space.attributes.values():
        offsets.append(offsets[-1] + len(values))
    weight = cfg.bottleneck.attribute_loss_weight if cfg.bottleneck else None
    interpretable = bool(has_attributes and weight)
    meta = run_metadata(device=str(resolved))

    record = {
        "image_ids": np.array(image_ids),
        "y_true": np.array(y_true, dtype=np.int16),
        "class_logits": np.concatenate(class_logits, axis=0),
    }
    if attribute_true:
        record["attribute_true"] = np.concatenate(attribute_true, axis=0)
    if has_attributes:
        record["attribute_probs"] = np.concatenate(attribute_probs, axis=0)
        record["attribute_group_offsets"] = np.array(offsets, dtype=np.int32)

    meta_block = {
        "domain": domain,
        "run_id": cfg.run_id,
        "seed": cfg.seed,
        "model_type": cfg.model.type,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash(checkpoint),
        "config_sha256": payload.get("config_sha256"),
        "selection_role": selection_role,
        "selection_rule": (payload.get("selection") or {}).get("rule"),
        "git": meta["git"],
        "hardware": {k: v for k, v in meta.items() if k != "git"},
        "device_used": str(resolved),
        "device_in_config": cfg.device,
        "n": len(image_ids),
        "classes": list(label_space.classes),
        "attribute_names": list(label_space.attributes),
        "attribute_values": {k: list(v) for k, v in label_space.attributes.items()},
        "has_attributes": has_attributes,
        "attribute_loss_weight": weight,
        "attributes_interpretable": interpretable,
        "attributes_note": "" if interpretable else CONTROL_NOTE,
        "class_logits_mode": "soft" if has_attributes else "none",
        "buffer_sha256_before": before,
        "buffer_sha256_after": after,
        "amendment": AMENDMENT,
        **observed,
    }
    record["meta"] = np.array(json.dumps(meta_block, default=str))

    unexpected = set(record) - set(ALLOWED_ARRAYS)
    if unexpected:
        raise DumpError(
            f"refusing to write arrays {sorted(unexpected)}: only "
            f"{list(ALLOWED_ARRAYS)} are permitted by the amendment. Nothing was "
            f"written.")
    unexpected = set(meta_block) - set(ALLOWED_META)
    if unexpected:
        raise DumpError(
            f"refusing to write metadata {sorted(unexpected)}: only "
            f"{list(ALLOWED_META)} are permitted by the amendment, which forbids "
            f"retaining anything describing the pixels. Nothing was written.")

    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.with_name(out.name + ".partial.npz")
    np.savez_compressed(staging, **record)
    if out.exists():
        staging.unlink()
        raise DumpError(f"{out} appeared while writing; nothing was overwritten")
    staging.replace(out)

    _append_ledger(ledger_path, {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint_sha256": meta_block["checkpoint_sha256"],
        "domain": domain,
        "run_id": cfg.run_id,
        "seed": cfg.seed,
        "config_sha256": meta_block["config_sha256"],
        "git": meta["git"],
        "selection_role": selection_role,
        "device_used": str(resolved),
        "n": len(image_ids),
        "output": out.name,
        "amendment": AMENDMENT,
    })
    return out


def _append_ledger(path: str | Path, row: dict) -> None:
    """Append-only, and written after the dump exists, so a line never claims a file
    that is not there."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoints", required=True,
                        help="a checkpoint, or a directory searched for */last.pt")
    parser.add_argument("--out", default=str(REPO / "analysis"))
    parser.add_argument("--ledger", default=str(ANALYSIS_LEDGER))
    parser.add_argument("--device", default=None,
                        help="defaults to the device the checkpoint records, so the "
                             "dump reproduces the ledgered evaluation")
    parser.add_argument("--domains", nargs="+", default=list(DOMAINS))
    args = parser.parse_args(argv)

    source = Path(args.checkpoints)
    paths = sorted(source.glob("*/last.pt")) if source.is_dir() else [source]
    if not paths:
        print(f"no checkpoints under {source}", file=sys.stderr)
        return 1

    failed = 0
    for path in paths:
        for domain in args.domains:
            try:
                written = dump_checkpoint(path, args.out, domain,
                                          ledger_path=args.ledger, device=args.device)
            except Exception as exc:               # noqa: BLE001 - reported, not hidden
                print(f"  {path.parent.name}/{domain}: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
                failed += 1
                continue
            print(f"  {written.name:<42} -> {written.parent}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
