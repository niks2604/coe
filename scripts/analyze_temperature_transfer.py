"""A temperature fitted on pbc_val, carried unchanged to each Raabin domain.

A sibling of analyze_temperature.py rather than an extension of it: that script
promises to read pbc_val and refuse everything else, and that promise is worth keeping
intact. This one reuses its fitter and its ECE, and reads the target dumps the sec 3
amendment of 2026-11-19 permits.

The temperature is still fitted on pbc_val ALONE. The amendment permits retaining
target predictions; it does not permit fitting anything on them, and a calibration
fitted on a target domain would be target-informed model behaviour whatever it was
called. A test wraps the fitter and asserts every call it makes is on the pbc_val
logits.

All three model types. ECE before and after on pbc_test and on each Raabin domain,
because a temperature that helps on the source and not on the target is the result
here, not a failure to be tidied away.

Post-hoc. It can explain the sec 6 verdict and cannot revise it.

    python -m scripts.analyze_temperature_transfer --analysis analysis/
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

from scripts.analyze_temperature import (                 # noqa: E402
    ECE_BINS,
    AnalysisError,
    at_search_bound,
    ece_of,
    fit_temperature,
)

FIT_DOMAIN = "pbc_val"
APPLY_DOMAINS = ("pbc_test", "raabin_train", "raabin_testA", "raabin_testB")
SCOPE = ("Post-hoc and exploratory, under the CLAUDE.md sec 3 amendment of 2026-11-19. "
         "The temperature is fitted on pbc_val alone and applied unchanged elsewhere; "
         "nothing is fitted on a target domain. The sec 6 verdict was fixed on "
         "eval_records/ before this existed and is not revised by it.")

__all__ = ["AnalysisError", "fit_temperature", "transfer_report"]


def _load(analysis_dir: Path) -> dict:
    out = {}
    for path in sorted(analysis_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            out[(meta["run_id"], meta["seed"], meta["domain"])] = {
                "class_logits": data["class_logits"], "y_true": data["y_true"],
                "meta": meta}
    return out


def transfer_report(analysis_dir: str | Path) -> dict:
    analysis_dir = Path(analysis_dir)
    dumps = _load(analysis_dir)
    if not dumps:
        raise AnalysisError(f"no .npz files found under {analysis_dir}")

    checkpoints = sorted({(run, seed) for run, seed, _ in dumps})
    rows = []
    for run_id, seed in checkpoints:
        fit = dumps.get((run_id, seed, FIT_DOMAIN))
        if fit is None:
            raise AnalysisError(
                f"{run_id} seed {seed}: no {FIT_DOMAIN} dump under {analysis_dir}. "
                f"The temperature is fitted there and nowhere else, so without it "
                f"there is nothing to carry.")
        temperature = fit_temperature(fit["class_logits"], fit["y_true"])

        for domain in (FIT_DOMAIN, *APPLY_DOMAINS):
            dump = dumps.get((run_id, seed, domain))
            if dump is None:
                continue
            logits, y_true = dump["class_logits"], dump["y_true"]
            before = ece_of(logits, y_true)
            after = ece_of(logits / temperature, y_true)
            rows.append({
                "run_id": run_id, "seed": seed,
                "model_type": dump["meta"]["model_type"],
                "domain": domain,
                "fitted_on": FIT_DOMAIN,
                "temperature": temperature,
                "at_search_bound": at_search_bound(temperature),
                "n": int(len(y_true)),
                "ece_before": before,
                "ece_after": after,
                # Negative means the source-fitted temperature helped here.
                "ece_delta": after - before,
                "checkpoint_sha256": dump["meta"].get("checkpoint_sha256"),
            })

    return {"scope": SCOPE, "fitted_on": FIT_DOMAIN, "ece_bins": ECE_BINS,
            "rows": rows}


def format_report(report: dict) -> str:
    lines = ["TEMPERATURE TRANSFER: fitted on pbc_val, applied unchanged", "=" * 92,
             report["scope"], "",
             f"ECE over {report['ece_bins']} bins. A negative delta means the "
             f"source-fitted temperature helped on that domain.", "",
             f"{'run':<12}{'seed':>5}  {'domain':<14}{'T':>8}{'ECE before':>12}"
             f"{'ECE after':>11}{'delta':>10}{'n':>8}", "-" * 92]
    for row in sorted(report["rows"],
                      key=lambda r: (r["run_id"], r["seed"],
                                     list((FIT_DOMAIN, *APPLY_DOMAINS)).index(
                                         r["domain"]))):
        lines.append(f"{row['run_id']:<12}{row['seed']:>5}  {row['domain']:<14}"
                     f"{row['temperature']:>8.3f}{row['ece_before']:>12.4f}"
                     f"{row['ece_after']:>11.4f}{row['ece_delta']:>+10.4f}"
                     f"{row['n']:>8}"
                     f"{'   T AT SEARCH BOUND' if row['at_search_bound'] else ''}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--analysis", default=str(REPO / "analysis"))
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    try:
        report = transfer_report(args.analysis)
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
