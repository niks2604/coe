"""Temperature scaling fitted on pbc_val, and nowhere else.

Post-hoc and exploratory. It reads analysis/*.npz and nothing else - no checkpoint, no
image, no ledger, no evaluation record - and it fits one temperature per checkpoint on
pbc_val, the split that is read constantly by design.

The temperature is NOT applied to any target domain. Doing so would need target
per-image predictions, which CLAUDE.md sec 3 forbids retaining, and a calibration
fitted on the source and carried to the target is a target-domain number produced
outside the one ledgered evaluation. A file claiming any domain but pbc_val is refused
rather than quietly included.

Nothing here feeds the result. The comparison was pre-declared in CLAUDE.md sec 6 and
decided on the records in eval_records/; this can explain that result and cannot revise
it.

    python -m scripts.analyze_temperature --analysis analysis/
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

from src.metrics import expected_calibration_error          # noqa: E402

PBC_VAL = "pbc_val"
ECE_BINS = 15
SCOPE = ("Fitted on pbc_val only. NOT applied to pbc_test or to any target domain: "
         "that would be a target-domain number produced outside the one ledgered "
         "evaluation (CLAUDE.md sec 3, sec 5).")


class AnalysisError(Exception):
    """A file that must not be read, or nothing to read."""


def load_dumps(analysis_dir: str | Path) -> list[tuple[Path, dict, dict]]:
    """Every *.npz in the directory, with its metadata, refusing any non-pbc_val file.

    Refusing rather than filtering: a dump of another domain in this directory is a
    file that should not exist, and silently stepping over it would leave it there.
    """
    analysis_dir = Path(analysis_dir)
    paths = sorted(analysis_dir.glob("*.npz"))
    if not paths:
        raise AnalysisError(f"no .npz files found under {analysis_dir}")

    out = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        meta = json.loads(str(arrays.pop("meta")))
        if meta.get("domain") != PBC_VAL:
            raise AnalysisError(
                f"{path} records domain {meta.get('domain')!r}. This analysis reads "
                f"{PBC_VAL!r} only; a dump of any other domain must not be here.")
        out.append((path, arrays, meta))
    return out


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def ece_of(logits: np.ndarray, y_true: np.ndarray, bins: int = ECE_BINS) -> float:
    probabilities = _softmax(logits)
    confidence = probabilities.max(axis=1).tolist()
    correct = (probabilities.argmax(axis=1) == y_true).tolist()
    return expected_calibration_error(confidence, correct, bins)


def _nll(logits: np.ndarray, y_true: np.ndarray, temperature: float) -> float:
    scaled = logits / temperature
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    log_partition = np.log(np.exp(shifted).sum(axis=1))
    return float(np.mean(log_partition - shifted[np.arange(len(y_true)), y_true]))


def at_search_bound(temperature: float, lo: float = 0.05, hi: float = 10.0,
                    atol: float = 1e-3) -> bool:
    """True when the fit landed on an end of the search interval.

    Then the interval is the answer, not the data: the NLL was still falling when the
    search ran out of room, so the reported temperature is clipped rather than fitted.
    Flagged rather than silently widened, because a temperature of exactly the bound
    reads like a fitted value and is not one.
    """
    return abs(temperature - lo) < atol or abs(temperature - hi) < atol


def fit_temperature(logits: np.ndarray, y_true: np.ndarray,
                    lo: float = 0.05, hi: float = 10.0,
                    iterations: int = 100) -> float:
    """The temperature minimising pbc_val NLL, by golden-section search.

    A one-parameter convex-in-practice problem on a bounded interval; a search needs no
    optimiser, no learning rate and no seed, so the fit is deterministic and has nothing
    to tune. Returns the temperature, which is > 1 for an over-confident model.
    """
    logits = np.asarray(logits, dtype=np.float64)
    y_true = np.asarray(y_true)
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = _nll(logits, y_true, c), _nll(logits, y_true, d)
    for _ in range(iterations):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = _nll(logits, y_true, c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = _nll(logits, y_true, d)
        if abs(b - a) < 1e-6:
            break
    return float((a + b) / 2.0)


def temperature_report(analysis_dir: str | Path) -> dict:
    rows = []
    for path, arrays, meta in load_dumps(analysis_dir):
        logits, y_true = arrays["class_logits"], arrays["y_true"]
        temperature = fit_temperature(logits, y_true)
        rows.append({
            "file": path.name,
            "run_id": meta["run_id"],
            "seed": meta["seed"],
            "model_type": meta["model_type"],
            "class_logits_mode": meta.get("class_logits_mode"),
            "n": int(len(y_true)),
            "temperature": temperature,
            "at_search_bound": at_search_bound(temperature),
            "ece_before": ece_of(logits, y_true),
            "ece_after": ece_of(logits / temperature, y_true),
            "nll_before": _nll(logits, y_true, 1.0),
            "nll_after": _nll(logits, y_true, temperature),
            "checkpoint_sha256": meta.get("checkpoint_sha256"),
            "git": meta.get("git"),
        })
    return {"domain": PBC_VAL, "scope": SCOPE, "ece_bins": ECE_BINS,
            "checkpoints": sorted(rows, key=lambda r: (r["run_id"], r["seed"]))}


def format_report(report: dict) -> str:
    lines = ["TEMPERATURE SCALING ON pbc_val", "=" * 78, report["scope"], "",
             f"ECE over {report['ece_bins']} bins. Post-hoc and exploratory: nothing "
             f"here feeds the", "pre-declared comparison (CLAUDE.md sec 6).", "",
             f"{'run':<12}{'seed':<6}{'T':>8}{'ECE before':>13}{'ECE after':>12}"
             f"{'NLL before':>13}{'NLL after':>12}", "-" * 78]
    for row in report["checkpoints"]:
        lines.append(f"{row['run_id']:<12}{row['seed']:<6}{row['temperature']:>8.3f}"
                     f"{row['ece_before']:>13.4f}{row['ece_after']:>12.4f}"
                     f"{row['nll_before']:>13.4f}{row['nll_after']:>12.4f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--analysis", default=str(REPO / "analysis"))
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    try:
        report = temperature_report(args.analysis)
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
