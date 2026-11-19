"""Source-side attribute behaviour on pbc_val, for the bottleneck checkpoints.

Post-hoc and exploratory. It reads analysis/*.npz and nothing else, and every file it
reads is pbc_val - the split read constantly by design. A file claiming any other
domain is refused rather than quietly included: CLAUDE.md sec 3 forbids retaining any
per-image record derived from target pixels, so a target dump in this directory is a
file that should not exist.

Reports, per bottleneck checkpoint:

  per-attribute accuracy against the WBCAtt labels, with balanced accuracy and the
  majority-class rate beside it, because the attribute vocabularies are imbalanced and
  a head that always answers the majority value ties the plain accuracy bar exactly
  (CLAUDE.md sec 6);

  the per-class mean predicted attribute distribution - for each class, the average
  predicted distribution over each head's values, keyed by value name.

The baseline has no attribute heads. The control's heads are excluded as
uninterpretable: it trains at lambda 0, so nothing ever supervised them, and scoring
them against WBCAtt labels would report the accuracy of a layer that never saw one.
Both exclusions are listed in the report rather than left to be noticed.

    python -m scripts.analyze_attributes_source --analysis analysis/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.analyze_temperature import PBC_VAL, AnalysisError, load_dumps  # noqa: E402

__all__ = ["AnalysisError", "analyze_attributes", "format_report"]


def _per_attribute(probs: np.ndarray, true_codes: np.ndarray, offsets: np.ndarray,
                   names: list[str], values: dict[str, list[str]]) -> dict:
    out = {}
    for head, name in enumerate(names):
        a, b = int(offsets[head]), int(offsets[head + 1])
        predicted = probs[:, a:b].argmax(axis=1)
        actual = true_codes[:, head]
        correct = predicted == actual

        present = np.unique(actual)
        recalls = [float(correct[actual == value].mean()) for value in present]
        counts = np.bincount(actual, minlength=b - a)
        out[name] = {
            "accuracy": float(correct.mean()),
            # Balanced as well as plain, because a head that has learned only the
            # majority value scores well on one and not the other.
            "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
            "majority_rate": float(counts.max() / counts.sum()),
            "n_values_present": int(len(present)),
            "value_names": list(values[name]),
        }
    return out


def _per_class_mean(probs: np.ndarray, y_true: np.ndarray, offsets: np.ndarray,
                    names: list[str], values: dict[str, list[str]],
                    classes: list[str]) -> dict:
    out = {}
    for index, cls in enumerate(classes):
        rows = y_true == index
        if not rows.any():
            continue                      # absent from pbc_val: no distribution to take
        mean = probs[rows].mean(axis=0)
        out[cls] = {
            name: {value: float(mean[int(offsets[head]) + j])
                   for j, value in enumerate(values[name])}
            for head, name in enumerate(names)
        }
    return out


def analyze_attributes(analysis_dir: str | Path) -> dict:
    checkpoints, excluded = [], []
    for path, arrays, meta in load_dumps(analysis_dir):
        if not meta.get("has_attributes") or "attribute_probs" not in arrays:
            excluded.append({"file": path.name, "run_id": meta["run_id"],
                             "seed": meta["seed"],
                             "reason": "no attribute heads; the baseline predicts "
                                       "classes directly"})
            continue
        if not meta.get("attributes_interpretable"):
            excluded.append({"file": path.name, "run_id": meta["run_id"],
                             "seed": meta["seed"],
                             "reason": "attribute heads are UNINTERPRETABLE: "
                                       f"attribute_loss_weight is "
                                       f"{meta.get('attribute_loss_weight')}, so "
                                       f"nothing supervised them"})
            continue
        if "attribute_true" not in arrays:
            excluded.append({"file": path.name, "run_id": meta["run_id"],
                             "seed": meta["seed"],
                             "reason": "no WBCAtt labels in the dump to score against"})
            continue

        probs = arrays["attribute_probs"]
        offsets = arrays["attribute_group_offsets"]
        names = meta["attribute_names"]
        values = meta["attribute_values"]
        checkpoints.append({
            "file": path.name,
            "run_id": meta["run_id"],
            "seed": meta["seed"],
            "model_type": meta["model_type"],
            "n": int(len(arrays["y_true"])),
            "attribute_loss_weight": meta.get("attribute_loss_weight"),
            "per_attribute": _per_attribute(probs, arrays["attribute_true"], offsets,
                                            names, values),
            "per_class_mean_attribute": _per_class_mean(
                probs, arrays["y_true"], offsets, names, values, meta["classes"]),
            "checkpoint_sha256": meta.get("checkpoint_sha256"),
            "git": meta.get("git"),
        })

    return {"domain": PBC_VAL,
            "scope": "Source side only. Nothing here is computed on, or carried to, "
                     "any target domain.",
            "checkpoints": sorted(checkpoints, key=lambda r: (r["run_id"], r["seed"])),
            "excluded": excluded}


def format_report(report: dict) -> str:
    lines = ["SOURCE-SIDE ATTRIBUTE BEHAVIOUR ON pbc_val", "=" * 86, report["scope"],
             "", "Post-hoc and exploratory: nothing here feeds the pre-declared "
             "comparison (CLAUDE.md sec 6).", ""]
    for row in report["excluded"]:
        lines.append(f"  excluded  {row['run_id']}_seed{row['seed']}: {row['reason']}")
    for row in report["checkpoints"]:
        lines.append("")
        lines.append(f"### {row['run_id']} seed {row['seed']}   n={row['n']}   "
                     f"lambda={row['attribute_loss_weight']}")
        lines.append(f"  {'attribute':<22}{'accuracy':>10}{'balanced':>10}"
                     f"{'majority':>10}")
        lines.append("  " + "-" * 52)
        for name, entry in row["per_attribute"].items():
            balanced = ("     —" if entry["balanced_accuracy"] is None
                        else f"{entry['balanced_accuracy']:>10.4f}")
            lines.append(f"  {name:<22}{entry['accuracy']:>10.4f}{balanced}"
                         f"{entry['majority_rate']:>10.4f}")
        lines.append("  per-class mean predicted attribute distribution:")
        for cls, heads in row["per_class_mean_attribute"].items():
            lines.append(f"    {cls}")
            for name, distribution in heads.items():
                top = sorted(distribution.items(), key=lambda kv: -kv[1])[:3]
                rendered = "  ".join(f"{v}={p:.3f}" for v, p in top)
                lines.append(f"      {name:<20} {rendered}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--analysis", default=str(REPO / "analysis"))
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    try:
        report = analyze_attributes(args.analysis)
    except AnalysisError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(format_report(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
