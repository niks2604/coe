"""Every dump must reproduce the evaluation it claims to come from.

A dump is a second forward pass over the same images with the same checkpoint, so
macro-F1 and ECE recomputed from it must match the matching eval_records entry. If they
do not, the dump is reading something other than what was ledgered - a different
checkpoint, different preprocessing, a different device - and no analysis may be read
off it. A mismatch is fatal, never a warning: a drift number computed on a dump that
does not reproduce the evaluation is a number about nothing.

Both metrics go through src.metrics, the same functions the evaluation used, so this
compares predictions rather than two implementations of a metric.

Reads analysis/*.npz and eval_records/*.json. Writes nothing, and opens neither
target_eval_ledger.jsonl nor any record for writing.

    python -m scripts.check_dump_consistency --analysis analysis/ --records eval_records/
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

from src.metrics import expected_calibration_error, macro_f1_present   # noqa: E402

# pbc_val is never evaluated and never ledgered, so there is nothing to compare its
# dumps with. Skipped and counted, not reported as a missing record.
UNCOMPARABLE = ("pbc_val",)

# The evaluation stored confidence as float32 and this recomputes it from float32
# logits, so the two agree to a few ulps when the dump ran on the device the
# evaluation ran on. A looser tolerance would hide exactly the drift this looks for.
DEFAULT_TOLERANCE = 1e-5
ECE_BINS = 15
MODE = "soft"


class ConsistencyError(Exception):
    """A dump that does not reproduce its evaluation. Always fatal."""


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def recompute(logits: np.ndarray, y_true, classes: list[str],
              bins: int = ECE_BINS) -> dict:
    """macro-F1 over the classes present, and ECE, through src.metrics."""
    probabilities = _softmax(np.asarray(logits, dtype=np.float32))
    y_pred = probabilities.argmax(axis=1).tolist()
    y_true = [int(v) for v in y_true]
    macro_f1, present = macro_f1_present(y_true, y_pred, classes)
    confidence = probabilities.max(axis=1).tolist()
    correct = [t == p for t, p in zip(y_true, y_pred)]
    return {"macro_f1": macro_f1, "macro_f1_classes": present,
            "ece": expected_calibration_error(confidence, correct, bins),
            "n": len(y_true)}


def _load_records(records_dir: str | Path) -> dict[tuple[str, int], dict]:
    out = {}
    for path in sorted(Path(records_dir).glob("*_eval.json")):
        if path.name.endswith("_eval_dryrun.json"):
            continue
        record = json.loads(path.read_text())
        out[(record["run_id"], record["seed"])] = record
    return out


def check_dumps(analysis_dir: str | Path, records_dir: str | Path,
                tolerance: float = DEFAULT_TOLERANCE) -> dict:
    dumps = sorted(Path(analysis_dir).glob("*.npz"))
    if not dumps:
        raise ConsistencyError(f"no .npz files found under {analysis_dir}")
    records = _load_records(records_dir)

    rows, skipped, mismatches = [], [], []
    for path in dumps:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            logits, y_true = data["class_logits"], data["y_true"]

        domain = meta["domain"]
        if domain in UNCOMPARABLE:
            skipped.append(path.name)
            continue

        key = (meta["run_id"], meta["seed"])
        record = records.get(key)
        if record is None:
            raise ConsistencyError(
                f"{path.name}: no evaluation record for {key} under {records_dir}. "
                f"A dump with nothing to check it against cannot be trusted.")
        if record.get("checkpoint_sha256") != meta.get("checkpoint_sha256"):
            raise ConsistencyError(
                f"{path.name}: checkpoint sha256 {str(meta.get('checkpoint_sha256'))[:12]} "
                f"but the record for {key} names "
                f"{str(record.get('checkpoint_sha256'))[:12]}. The dump is of a "
                f"different checkpoint than the one that was evaluated.")
        scored = (record.get("results") or {}).get(domain)
        if scored is None:
            raise ConsistencyError(
                f"{path.name}: the record for {key} has no result for domain "
                f"{domain!r}; it covers {sorted(record.get('results') or {})}.")
        recorded = scored[MODE]

        got = recompute(logits, y_true, meta["classes"])
        row = {
            "file": path.name, "run_id": meta["run_id"], "seed": meta["seed"],
            "domain": domain,
            "n_dump": got["n"], "n_record": recorded["n"],
            "macro_f1_dump": got["macro_f1"], "macro_f1_record": recorded["macro_f1"],
            "macro_f1_delta": got["macro_f1"] - recorded["macro_f1"],
            "ece_dump": got["ece"], "ece_record": recorded["ece"],
            "ece_delta": got["ece"] - recorded["ece"],
            "classes_dump": got["macro_f1_classes"],
            "classes_record": recorded.get("macro_f1_classes"),
        }
        rows.append(row)

        if got["n"] != recorded["n"]:
            mismatches.append(
                f"{path.name}: {got['n']} rows in the dump, n {recorded['n']} in the "
                f"record. The dump covers a different set of images.")
        if got["macro_f1_classes"] != recorded.get("macro_f1_classes"):
            mismatches.append(
                f"{path.name}: macro-F1 over {got['macro_f1_classes']} in the dump, "
                f"{recorded.get('macro_f1_classes')} in the record.")
        if abs(row["macro_f1_delta"]) > tolerance:
            mismatches.append(
                f"{path.name} {domain}: macro_f1 {got['macro_f1']:.6f} recomputed, "
                f"{recorded['macro_f1']:.6f} recorded, delta "
                f"{row['macro_f1_delta']:+.3e} over tolerance {tolerance:g}")
        if abs(row["ece_delta"]) > tolerance:
            mismatches.append(
                f"{path.name} {domain}: ece {got['ece']:.6f} recomputed, "
                f"{recorded['ece']:.6f} recorded, delta {row['ece_delta']:+.3e} "
                f"over tolerance {tolerance:g}")

    if mismatches:
        raise ConsistencyError(
            f"{len(mismatches)} mismatch(es) between the dumps and the ledgered "
            f"evaluation. No analysis may be read off these dumps until this is "
            f"explained:\n  " + "\n  ".join(mismatches) +
            "\n\nIf the dump ran on a different device than the evaluation did, the "
            "logits differ in the last bits; re-dump on the evaluation's device "
            "rather than loosening the tolerance.")

    return {"checked": len(rows), "skipped": skipped, "mismatches": mismatches,
            "tolerance": tolerance, "rows": rows}


def format_report(report: dict) -> str:
    lines = ["DUMP / EVALUATION CONSISTENCY", "=" * 92,
             f"macro-F1 and ECE recomputed through src.metrics, mode {MODE!r}, "
             f"tolerance {report['tolerance']:g}", "",
             f"{'file':<44}{'macro-F1 d':>12}{'ECE d':>12}{'n':>8}", "-" * 92]
    for row in report["rows"]:
        lines.append(f"{row['file']:<44}{row['macro_f1_delta']:>12.2e}"
                     f"{row['ece_delta']:>12.2e}{row['n_dump']:>8}")
    lines.append("")
    lines.append(f"{report['checked']} dump(s) match the ledgered evaluation.")
    for name in report["skipped"]:
        lines.append(f"  skipped (never evaluated, nothing to compare): {name}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--analysis", default=str(REPO / "analysis"))
    parser.add_argument("--records", default=str(REPO / "eval_records"))
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args(argv)
    try:
        report = check_dumps(args.analysis, args.records, args.tolerance)
    except ConsistencyError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
