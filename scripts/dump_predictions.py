"""Per-image predictions on pbc_val, for post-hoc source-side analysis.

SOURCE DATA ONLY, AND STRUCTURALLY SO. pbc_val is the one split read constantly by
design: it is the tuning split, it is not ledgered, and dumping it spends nothing.
Every other domain is refused before anything is loaded.

  pbc_test   is held out for the source-domain report and is ledgered, so a look
             outside src.eval leaves no trace of itself (CLAUDE.md sec 5).
  raabin_*   CLAUDE.md sec 3: nothing derived from target pixels may be retained,
             reported or written to disk - no per-image record of any kind. A file of
             per-image target logits is exactly that.

The refusal is the first statement in dump_checkpoint, above any loader, and a test
monkeypatches the loader factories to prove nothing downstream can reach them.

    python -m scripts.dump_predictions --checkpoints DIR --out analysis/

Reads checkpoints and writes analysis/*.npz. Touches neither eval_records/ nor
target_eval_ledger.jsonl, and changes nothing in src/eval.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import resolve_device, run_metadata          # noqa: E402
from src.data.loaders import make_val_loader                  # noqa: E402
from src.eval import _buffer_hash, _selection_role, load_checkpoint  # noqa: E402
from src.ledger import checkpoint_hash                        # noqa: E402

PBC_VAL = "pbc_val"

CONTROL_NOTE = (
    "UNINTERPRETABLE: these attribute heads are unsupervised. "
    "The control trains with attribute_loss_weight 0, so nothing ever supervised these "
    "heads. They are the bottleneck's architecture without its supervision: a 31-unit "
    "layer the classifier learned to read however it liked. The values are not "
    "attribute predictions and must not be read as morphology."
)


class DumpError(Exception):
    """A domain that must not be dumped, or an output that already exists."""


def output_path(cfg, out_dir: str | Path) -> Path:
    return Path(out_dir) / f"{cfg.run_id}_seed{cfg.seed}_{PBC_VAL}.npz"


def _require_pbc_val(domain: str) -> None:
    """The gate. Raised, not skipped: a skip would leave a file covering less than its
    name says, and for a target domain it would leave a file that must not exist."""
    if domain != PBC_VAL:
        raise DumpError(
            f"{domain!r} may not be dumped; this script reads {PBC_VAL!r} and nothing "
            f"else. pbc_test is held out and ledgered, and CLAUDE.md sec 3 forbids "
            f"retaining any per-image record derived from target pixels. If a "
            f"target-side dump is wanted, that is a protocol change: amend sec 3, log "
            f"it in sec 8, and decide how the look is ledgered."
        )


@torch.no_grad()
def dump_checkpoint(checkpoint: str | Path, out_dir: str | Path,
                    domain: str = PBC_VAL, device: str | None = None) -> Path:
    # First statement, above every load: nothing below may be reached for any other
    # domain, whatever a later edit does.
    _require_pbc_val(domain)

    checkpoint = Path(checkpoint)
    cfg, mapping, label_space, model, payload = load_checkpoint(checkpoint)
    # The same gate src.eval applies, and the same helper, so the two cannot drift:
    # best.pt is the pbc_val peak the final-epoch rule discarded and is never the
    # checkpoint a result is read from. No override is accepted here.
    selection_role, _ = _selection_role(checkpoint, payload, None)

    out = output_path(cfg, out_dir)
    if out.exists():
        raise DumpError(
            f"{out} already exists and will not be overwritten. Move it aside if you "
            f"mean to replace it; a silent overwrite would lose the only copy of a "
            f"previous pass.")

    # A forward pass for analysis, not an experiment, so it may run on a device the
    # checkpoint was not trained on - these are read-only predictions and nothing
    # selects on them. Both devices are recorded so any difference is visible rather
    # than assumed away. No silent fallback: an unavailable device is still an error.
    resolved = resolve_device(cfg) if device is None else torch.device(device)
    model.to(resolved)
    model.eval()
    before = _buffer_hash(model)

    # Structurally pinned to pbc_val: make_val_loader refuses any other id, including
    # every target one, so this call cannot reach target data even if the gate above
    # were removed.
    loader = make_val_loader(cfg, mapping, label_space)
    dataset = loader.dataset
    image_ids = [dataset.img_names[j] for j in dataset.indices]

    observed = {"model_training_mode_seen": False, "grad_enabled_seen": False}
    has_attributes = hasattr(model, "bottleneck_vector")
    class_logits: list[np.ndarray] = []
    attribute_probs: list[np.ndarray] = []
    y_true: list[int] = []
    attribute_true: list[np.ndarray] = []

    for images, labels, attributes in loader:
        images = images.to(resolved, non_blocking=True)
        observed["model_training_mode_seen"] |= bool(model.training)
        observed["grad_enabled_seen"] |= bool(torch.is_grad_enabled())
        if has_attributes:
            heads = model.attribute_logits(images)
            probabilities = [torch.softmax(h, dim=1) for h in heads]
            vector = model.bottleneck_vector(heads, mode="soft")
            logits = model.classify(vector)
            attribute_probs.append(
                torch.cat(probabilities, dim=1).cpu().numpy().astype(np.float32))
        else:
            logits = model(images).class_logits
        class_logits.append(logits.cpu().numpy().astype(np.float32))
        y_true.extend(int(v) for v in labels)
        if attributes is not None:
            attribute_true.append(attributes.cpu().numpy().astype(np.int16))

    after = _buffer_hash(model)
    meta = run_metadata(device=str(resolved))

    offsets = [0]
    for values in label_space.attributes.values():
        offsets.append(offsets[-1] + len(values))

    weight = None
    if cfg.bottleneck is not None:
        weight = cfg.bottleneck.attribute_loss_weight
    interpretable = bool(has_attributes and weight)

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

    record["meta"] = np.array(json.dumps({
        "domain": PBC_VAL,
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
        **observed,
    }, default=str))

    if before != after:
        raise DumpError(
            f"{checkpoint}: the model's buffers changed during the pass "
            f"({before[:12]} -> {after[:12]}). A forward pass must not move BatchNorm "
            f"running statistics; nothing was written.")

    out.parent.mkdir(parents=True, exist_ok=True)
    # The suffix must stay .npz: numpy appends one to any other name, and the staged
    # file would then be written somewhere this never looks.
    staging = out.with_name(out.name + ".partial.npz")
    np.savez_compressed(staging, **record)
    if out.exists():                       # lost a race; keep the earlier file
        staging.unlink()
        raise DumpError(f"{out} appeared while writing; nothing was overwritten")
    staging.replace(out)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Dump per-image pbc_val predictions for source-side analysis.")
    parser.add_argument("--checkpoints", required=True,
                        help="a checkpoint, or a directory searched for */last.pt")
    parser.add_argument("--out", default=str(REPO / "analysis"))
    parser.add_argument("--device", default="cpu",
                        help="device for the forward pass; recorded in the output "
                             "beside the checkpoint's own device")
    args = parser.parse_args(argv)

    source = Path(args.checkpoints)
    paths = sorted(source.glob("*/last.pt")) if source.is_dir() else [source]
    if not paths:
        print(f"no checkpoints under {source}", file=sys.stderr)
        return 1

    failed = 0
    for path in paths:
        try:
            written = dump_checkpoint(path, args.out, device=args.device)
        except Exception as exc:                      # noqa: BLE001 - reported, not hidden
            print(f"  {path}: {type(exc).__name__}: {exc}", file=sys.stderr)
            failed += 1
            continue
        with np.load(written, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
        print(f"  {meta['run_id']}_seed{meta['seed']:<2} {meta['model_type']:<11} "
              f"n={meta['n']:<6} buffers unchanged: "
              f"{meta['buffer_sha256_before'] == meta['buffer_sha256_after']}  "
              f"-> {written}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
