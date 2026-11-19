"""How far the predicted attribute distributions move from pbc_test to each Raabin domain.

Bottleneck checkpoints only. The control's heads are unsupervised at lambda 0, so they
are not attribute predictions and a distance between them would measure nothing about
morphology; the baseline has no heads at all. Both exclusions are listed in the report.

Per class and per attribute, the Jensen-Shannon DISTANCE - the square root of the
divergence - between the mean predicted distribution on pbc_test and on the target
domain. The distance rather than the divergence because it is a metric bounded in
[0, 1], so numbers from different attributes with different vocabulary sizes can be
put in one ranked list without the bound moving under them.

Every attribute, every seed, every domain, ranked, with nothing filtered out: a
threshold applied after the numbers are seen is a choice made by the numbers. A class
absent from a target domain is reported as unmeasurable with a reason, never as a zero.

Post-hoc under the CLAUDE.md sec 3 amendment of 2026-11-19. It can explain the sec 6
verdict and cannot revise it. It selects nothing.

    python -m scripts.analyze_attribute_drift --analysis analysis/
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

SOURCE = "pbc_test"
TARGETS = ("raabin_train", "raabin_testA", "raabin_testB")
SCOPE = ("Post-hoc and exploratory, under the CLAUDE.md sec 3 amendment of 2026-11-19. "
         "The sec 6 verdict was fixed on eval_records/ before this existed; this can "
         "explain it and cannot revise it. Nothing here selects anything.")


class AnalysisError(Exception):
    """A dump set this cannot be computed from."""


def jensen_shannon_distance(p: np.ndarray, q: np.ndarray) -> float:
    """sqrt of the Jensen-Shannon divergence in bits: a metric, bounded in [0, 1].

    Bounded, so attributes with two values and attributes with six go in one ranked
    list without the scale changing between rows.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    divergence = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return float(np.sqrt(max(divergence, 0.0)))


def _load(analysis_dir: Path) -> dict:
    out = {}
    for path in sorted(analysis_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        meta = json.loads(str(arrays.pop("meta")))
        out[(meta["run_id"], meta["seed"], meta["domain"])] = (path, arrays, meta)
    return out


def _mean_distributions(arrays: dict, meta: dict) -> dict:
    """{class_index: {attribute: np.ndarray}}, the mean predicted distribution."""
    probs = arrays["attribute_probs"]
    offsets = arrays["attribute_group_offsets"]
    y_true = arrays["y_true"]
    names = meta["attribute_names"]
    out = {}
    for index in sorted(set(int(v) for v in y_true)):
        rows = y_true == index
        mean = probs[rows].mean(axis=0)
        out[index] = {name: mean[int(offsets[h]):int(offsets[h + 1])]
                      for h, name in enumerate(names)}
    return out


def drift_report(analysis_dir: str | Path) -> dict:
    analysis_dir = Path(analysis_dir)
    dumps = _load(analysis_dir)
    if not dumps:
        raise AnalysisError(f"no .npz files found under {analysis_dir}")

    rows, unmeasurable, excluded = [], [], []
    checkpoints = sorted({(run, seed) for run, seed, _ in dumps})

    for run_id, seed in checkpoints:
        source = dumps.get((run_id, seed, SOURCE))
        if source is None:
            # Every target dump is measured against this one; without it there is no
            # drift to compute, only a number against nothing.
            if any(k[0] == run_id and k[1] == seed for k in dumps):
                raise AnalysisError(
                    f"{run_id} seed {seed}: no {SOURCE} dump under {analysis_dir}. "
                    f"The source side is what the drift is measured from.")
            continue
        _, source_arrays, source_meta = source

        if not source_meta.get("has_attributes"):
            excluded.append({"run_id": run_id, "seed": seed,
                             "reason": "no attribute heads; the baseline predicts "
                                       "classes directly"})
            continue
        if not source_meta.get("attributes_interpretable"):
            excluded.append({"run_id": run_id, "seed": seed,
                             "reason": f"attribute heads are UNINTERPRETABLE: "
                                       f"attribute_loss_weight is "
                                       f"{source_meta.get('attribute_loss_weight')}, "
                                       f"so nothing supervised them"})
            continue

        classes = source_meta["classes"]
        values = source_meta["attribute_values"]
        source_means = _mean_distributions(source_arrays, source_meta)

        for domain in TARGETS:
            target = dumps.get((run_id, seed, domain))
            if target is None:
                excluded.append({"run_id": run_id, "seed": seed, "domain": domain,
                                 "reason": "no dump for this domain"})
                continue
            _, target_arrays, target_meta = target
            target_means = _mean_distributions(target_arrays, target_meta)

            for index, cls in enumerate(classes):
                if index not in target_means:
                    for name in source_meta["attribute_names"]:
                        unmeasurable.append({
                            "run_id": run_id, "seed": seed, "domain": domain,
                            "cls": cls, "attribute": name, "jensen_shannon": None,
                            "reason": f"{cls} is absent from {domain}; the distance "
                                      f"is not measurable there"})
                    continue
                if index not in source_means:
                    for name in source_meta["attribute_names"]:
                        unmeasurable.append({
                            "run_id": run_id, "seed": seed, "domain": domain,
                            "cls": cls, "attribute": name, "jensen_shannon": None,
                            "reason": f"{cls} is absent from {SOURCE}"})
                    continue
                for name in source_meta["attribute_names"]:
                    p, q = source_means[index][name], target_means[index][name]
                    rows.append({
                        "run_id": run_id, "seed": seed, "domain": domain,
                        "source_domain": SOURCE, "cls": cls, "attribute": name,
                        "jensen_shannon": jensen_shannon_distance(p, q),
                        "n_source": int((source_arrays["y_true"] == index).sum()),
                        "n_target": int((target_arrays["y_true"] == index).sum()),
                        "source_mean": {v: float(p[j])
                                        for j, v in enumerate(values[name])},
                        "target_mean": {v: float(q[j])
                                        for j, v in enumerate(values[name])},
                        "checkpoint_sha256": source_meta.get("checkpoint_sha256"),
                    })

    rows.sort(key=lambda r: -r["jensen_shannon"])
    return {"scope": SCOPE, "source_domain": SOURCE, "metric": "jensen_shannon_distance",
            "rows": rows, "unmeasurable": unmeasurable, "excluded": excluded,
            "filtered": 0}


def format_report(report: dict) -> str:
    lines = ["ATTRIBUTE DRIFT, pbc_test -> each Raabin domain", "=" * 96, report["scope"],
             "", "Jensen-Shannon distance (metric, bounded 0-1). Every attribute, seed "
             "and domain; nothing filtered.", ""]
    for row in report["excluded"]:
        where = f" {row['domain']}" if "domain" in row else ""
        lines.append(f"  excluded  {row['run_id']}_seed{row['seed']}{where}: "
                     f"{row['reason']}")
    lines.append("")
    lines.append(f"{'JS':>8}  {'domain':<15}{'class':<13}{'attribute':<26}"
                 f"{'seed':>5}{'n_src':>8}{'n_tgt':>8}")
    lines.append("-" * 96)
    for row in report["rows"]:
        lines.append(f"{row['jensen_shannon']:>8.4f}  {row['domain']:<15}"
                     f"{row['cls']:<13}{row['attribute']:<26}{row['seed']:>5}"
                     f"{row['n_source']:>8}{row['n_target']:>8}")
    if report["unmeasurable"]:
        lines.append("")
        lines.append("not measurable (null, not zero):")
        seen = set()
        for row in report["unmeasurable"]:
            key = (row["domain"], row["cls"])
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"  {row['domain']:<15}{row['cls']:<13}{row['reason']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--analysis", default=str(REPO / "analysis"))
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    try:
        report = drift_report(args.analysis)
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
