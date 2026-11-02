"""Read-only data audit. Run: python -m src.data.audit --config configs/audit.yaml

What this does NOT do, by construction:

* it never writes to, renames or deletes anything under a data root - files are opened
  read-binary only;
* it computes no normalisation statistic of any kind. No mean, no std, no stain
  reference, no per-channel aggregate, for any dataset. Backbone constants are fixed
  ImageNet values set in the config (src/config.py);
* it reports no image size or mode distribution for a TARGET dataset. Targets get
  counts, class folder names and unreadable-file lists only, so that no pixel-level
  property of Raabin or LISC can inform a preprocessing or architecture choice
  (CLAUDE.md sec 3).

Unmapped classes are reported, not raised: Raabin is deliberately unmapped until this
report has been reviewed, so the audit has to be able to describe it.

No torch import: the audit runs before torch is needed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image

from src.config import AuditConfig, load_audit_config, run_metadata, set_seed
from src.data.domains import DATASETS, ClassMapping, lookup_class

AUDIT_VERSION = 1
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

ATTR_COLUMNS = [
    "cell_size", "cell_shape", "nucleus_shape", "nuclear_cytoplasmic_ratio",
    "chromatin_density", "cytoplasm_vacuole", "cytoplasm_texture", "cytoplasm_colour",
    "granule_type", "granule_colour", "granularity",
]
WBCATT_SPLITS = ("train", "val", "test")
# Reporting buckets for the neutrophil filename prefixes PBC uses. Anything else falls
# into "other", so a prefix we have not seen cannot vanish from the counts.
NEUTROPHIL_PREFIXES = ["SNE_", "BNE_", "NEUTROPHIL_"]
MAX_LISTED = 50  # long lists are truncated in the report, with the full count kept


def _truncate(items: list[str]) -> list[str]:
    if len(items) <= MAX_LISTED:
        return items
    return items[:MAX_LISTED] + [f"... and {len(items) - MAX_LISTED} more"]


def _image_files(root: Path) -> list[Path]:
    """Image files, excluding dotfiles.

    PBC ships data/raw/PBC_dataset_normal_DIB/neutrophil/.DS_169665.jpg - macOS
    metadata with a .jpg name. It is not an image, so counting it as a corrupt one
    buries the genuinely corrupt files in the same list. Dotfiles are reported
    separately by _dotfiles().
    """
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and not p.name.startswith(".")
                  and p.suffix.lower() in IMAGE_EXTENSIONS)


def _dotfiles(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.name.startswith("."))


def _verify(paths: Iterable[Path], root: Path, record_stats: bool) -> dict:
    """Open each file and convert to RGB. Sizes/modes recorded only when allowed."""
    unreadable: list[str] = []
    sizes: Counter = Counter()
    modes: Counter = Counter()
    for p in paths:
        try:
            with Image.open(p) as im:
                mode = im.mode
                size = im.size
                im.convert("RGB")
        except Exception as exc:
            unreadable.append(f"{p.relative_to(root)}: {type(exc).__name__}: {exc}")
            continue
        if record_stats:
            sizes[f"{size[0]}x{size[1]}"] += 1
            modes[mode] += 1
    out: dict = {"unreadable": _truncate(unreadable), "n_unreadable": len(unreadable)}
    if record_stats:
        out["image_sizes"] = dict(sizes.most_common())
        out["image_modes"] = dict(modes.most_common())
    return out


def _tree(root: Path, max_depth: int) -> dict:
    """Directory shape with per-directory file counts. Filenames are not listed."""
    out: dict[str, dict] = {}
    for d in sorted(p for p in root.rglob("*") if p.is_dir()):
        rel = d.relative_to(root)
        if len(rel.parts) > max_depth:
            continue
        files = [f for f in d.iterdir() if f.is_file()]
        out[str(rel)] = {
            "n_files": len(files),
            "n_subdirs": sum(1 for f in d.iterdir() if f.is_dir()),
            "extensions": dict(Counter(f.suffix.lower() for f in files).most_common()),
        }
    return out


def _duplicate_filenames(paths: list[Path], root: Path) -> dict[str, list[str]]:
    by_name: dict[str, list[str]] = defaultdict(list)
    for p in paths:
        by_name[p.name].append(str(p.relative_to(root)))
    return {name: locs for name, locs in sorted(by_name.items()) if len(locs) > 1}


def _scan_class_folders(root: Path, mapping: ClassMapping, corpus: str,
                        record_stats: bool) -> dict:
    """A split directory whose immediate children are class folders."""
    class_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    loose = [p for p in root.iterdir()
             if p.is_file() and not p.name.startswith(".")
             and p.suffix.lower() in IMAGE_EXTENSIONS]

    classes: dict[str, dict] = {}
    all_images: list[Path] = []
    for d in class_dirs:
        imgs = _image_files(d)
        all_images.extend(imgs)
        entry: dict = {"n_images": len(imgs), "mapped_to": lookup_class(mapping, corpus, d.name)}
        classes[d.name] = entry

    result: dict = {
        "n_images": len(all_images) + len(loose),
        "n_class_folders": len(class_dirs),
        "classes": classes,
        "images_outside_class_folders": len(loose),
        "n_ignored_dotfiles": len(_dotfiles(root)),
        "extensions": dict(Counter(p.suffix.lower() for p in all_images).most_common()),
    }
    verified = _verify(all_images + loose, root, record_stats)
    result["unreadable"] = verified["unreadable"]
    if record_stats:
        result["image_sizes"] = verified["image_sizes"]
        result["image_modes"] = verified["image_modes"]
    return result


# --- PBC ---------------------------------------------------------------------------------

def _audit_pbc(root: Path, mapping: ClassMapping, verify: bool, max_depth: int) -> dict:
    if not root.exists():
        return {"status": "absent", "path": str(root)}

    class_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    images = _image_files(root)

    classes: dict[str, dict] = {}
    excluded: list[str] = []
    unmapped: list[str] = []
    for d in class_dirs:
        imgs = _image_files(d)
        mapped = lookup_class(mapping, "pbc", d.name)
        if mapped is None:
            key = f"pbc:{d.name}"
            (excluded if key in mapping.excluded else unmapped).append(d.name)
        classes[d.name] = {
            "n_images": len(imgs),
            "mapped_to": mapped,
            # Prefix histogram: this is how band vs segmented neutrophils are told
            # apart, and the pbc_test_no_band_neutrophils subset depends on it.
            "filename_prefixes": dict(
                Counter(p.name.split("_")[0] + "_" for p in imgs).most_common()),
        }

    out: dict = {
        "status": "present",
        "path": str(root),
        "n_images": len(images),
        "classes": classes,
        "excluded_classes": excluded,
        "unmapped_classes": unmapped,
        "extensions": dict(Counter(p.suffix.lower() for p in images).most_common()),
        "non_image_files": _truncate(
            [str(p.relative_to(root)) for p in sorted(root.rglob("*"))
             if p.is_file() and not p.name.startswith(".")
             and p.suffix.lower() not in IMAGE_EXTENSIONS]),
        "ignored_dotfiles": _truncate(
            [str(p.relative_to(root)) for p in _dotfiles(root)]),
        "n_ignored_dotfiles": len(_dotfiles(root)),
        "duplicate_filenames": _duplicate_filenames(images, root),
        "tree": _tree(root, max_depth),
    }
    if verify:
        out.update(_verify(images, root, record_stats=True))
    else:
        out.update({"unreadable": [], "n_unreadable": None,
                    "image_sizes": {}, "image_modes": {}})
    return out


# --- WBCAtt -------------------------------------------------------------------------------

def _read_csvs(wbcatt_dir: Path) -> tuple[dict[str, list[dict]], list[str], dict[str, list[str]]]:
    """Headers are returned per split: the three CSVs are separate files and can
    disagree, so a single shared `header` would hide a column missing from one of them.
    """
    import csv

    per_split: dict[str, list[dict]] = {}
    missing: list[str] = []
    headers: dict[str, list[str]] = {}
    for split in WBCATT_SPLITS:
        path = wbcatt_dir / f"pbc_attr_v1_{split}.csv"
        if not path.exists():
            missing.append(path.name)
            continue
        with path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            headers[split] = list(reader.fieldnames or [])
            per_split[split] = list(reader)
    return per_split, missing, headers


def _audit_wbcatt(wbcatt_dir: Path, mapping: ClassMapping) -> dict:
    per_split, missing, headers = _read_csvs(wbcatt_dir)
    if not per_split:
        return {"status": "absent", "path": str(wbcatt_dir), "missing_csvs": missing}

    expected_columns = set(ATTR_COLUMNS) | {"img_name", "label", "path"}
    # Missing from *any* split is missing: a column present in test but absent from
    # train is still unusable.
    missing_per_split = {s: sorted(expected_columns - set(h))
                         for s, h in headers.items()}
    missing_columns = sorted({c for cols in missing_per_split.values() for c in cols})
    unexpected_columns = sorted({c for h in headers.values() for c in h}
                                - expected_columns)

    attributes: dict[str, dict] = {}
    for col in ATTR_COLUMNS:
        values: Counter = Counter()
        n_null = 0
        whitespace: list[str] = []
        for rows in per_split.values():
            for row in rows:
                raw = row.get(col)
                if raw is None or str(raw).strip() == "":
                    n_null += 1
                    continue
                values[raw] += 1
                if raw != raw.strip() and raw not in whitespace:
                    whitespace.append(raw)
        case_clashes = {
            lowered: sorted(v)
            for lowered, v in (
                (k, [x for x in values if x.lower() == k]) for k in
                {val.lower() for val in values}
            )
            if len(v) > 1
        }
        attributes[col] = {
            "n_values": len(values),
            "values": dict(values.most_common()),
            "n_null": n_null,
            "anomalies": {"whitespace": whitespace, "case_clashes": case_clashes},
            "present_in_splits": sorted(s for s, h in headers.items() if col in h),
        }

    labels: Counter = Counter()
    counts_per_split_label: dict[str, dict[str, int]] = {}
    for split, rows in per_split.items():
        per = Counter(r.get("label", "") for r in rows)
        counts_per_split_label[split] = dict(sorted(per.items()))
        labels.update(per)

    unmapped_labels = sorted({lab for lab in labels
                              if lookup_class(mapping, "wbcatt", lab) is None})
    counts_per_class: Counter = Counter()
    for lab, n in labels.items():
        cls = lookup_class(mapping, "wbcatt", lab)
        counts_per_class[cls or f"UNMAPPED:{lab}"] += n

    names_by_split = {s: {r.get("img_name", "") for r in rows}
                      for s, rows in per_split.items()}
    overlaps: dict[str, list[str]] = {}
    splits = sorted(names_by_split)
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            shared = sorted(names_by_split[a] & names_by_split[b])
            if shared:
                overlaps[f"{a}|{b}"] = _truncate(shared)

    return {
        "status": "present",
        "path": str(wbcatt_dir),
        "missing_csvs": missing,
        "columns_per_split": headers,
        "unexpected_columns": unexpected_columns,
        "missing_columns": missing_columns,
        "missing_columns_per_split": {s: c for s, c in missing_per_split.items() if c},
        "n_rows_per_split": {s: len(r) for s, r in sorted(per_split.items())},
        "n_rows_total": sum(len(r) for r in per_split.values()),
        "counts_per_class": dict(sorted(counts_per_class.items())),
        "counts_per_split_and_label": counts_per_split_label,
        "unmapped_labels": unmapped_labels,
        "attributes": attributes,
        "n_attribute_values_total": sum(a["n_values"] for a in attributes.values()),
        "neutrophil_prefixes_per_split": _neutrophil_prefixes_per_split(
            per_split, mapping, NEUTROPHIL_PREFIXES),
        "secondary_subsets": _measure_secondary_subsets(per_split, mapping),
        "split_overlaps": overlaps,
        "duplicate_img_names_within_split": {
            s: {n: c for n, c in Counter(r.get("img_name", "") for r in rows).items() if c > 1}
            for s, rows in sorted(per_split.items())
        },
    }


def _filename_prefix(img_name: str) -> str:
    """"SNE_123.jpg" -> "SNE_". Names with no underscore have no prefix."""
    head, sep, _ = img_name.partition("_")
    return head + sep if sep else ""


def _neutrophil_prefixes_per_split(per_split: dict[str, list[dict]],
                                   mapping: ClassMapping,
                                   prefixes: list[str]) -> dict[str, dict[str, int]]:
    """Band vs segmented is visible only in the filename, and the 2026-09-15 decision
    to keep bands rests on how they fall across the WBCAtt splits. Source data only.
    """
    out: dict[str, dict[str, int]] = {}
    for split, rows in sorted(per_split.items()):
        counts = {p: 0 for p in prefixes}
        counts["other"] = 0
        for row in rows:
            if lookup_class(mapping, "wbcatt", row.get("label", "")) != "neutrophil":
                continue
            prefix = _filename_prefix(row.get("img_name", ""))
            counts[prefix if prefix in counts else "other"] += 1
        out[split] = counts
    return out


def _measure_secondary_subsets(per_split: dict[str, list[dict]],
                               mapping: ClassMapping) -> dict[str, dict]:
    """Measure each secondary subset defined in the mapping, rather than restating the
    size recorded there. Drop prefixes come from the mapping, never from this module.
    """
    out: dict[str, dict] = {}
    for name, spec in mapping.secondary_subsets.items():
        split_id = spec.get("split", "")
        csv_split = DATASETS[split_id].csv_split if split_id in DATASETS else None
        if csv_split is None or csv_split not in per_split:
            out[name] = {"status": "skipped", "split": split_id,
                         "reason": f"no rows for split {split_id!r}"}
            continue

        drop = {key: list(vals) for key, vals
                in (spec.get("drop_filename_prefixes") or {}).items()}
        kept: Counter = Counter()
        dropped_prefixes: Counter = Counter()
        n_dropped = 0
        for row in per_split[csv_split]:
            cls = lookup_class(mapping, "wbcatt", row.get("label", ""))
            prefix = _filename_prefix(row.get("img_name", ""))
            # drop keys are "<corpus>:<raw class name>"; compare on the resolved class
            # so the key's corpus does not have to match the CSV's.
            hit = next(
                (p for key, prefs in drop.items()
                 if lookup_class(mapping, *key.split(":", 1)) == cls
                 for p in prefs if prefix == p),
                None,
            )
            if hit is not None:
                dropped_prefixes[hit] += 1
                n_dropped += 1
            else:
                kept[cls or "UNMAPPED"] += 1

        out[name] = {
            "status": "measured",
            "split": split_id,
            "n_images_in_split": len(per_split[csv_split]),
            "n_images": sum(kept.values()),
            "n_dropped": n_dropped,
            "dropped_prefixes": dict(sorted(dropped_prefixes.items())),
            "counts_per_class": dict(sorted(kept.items())),
        }
    return out


def _candidate_paths(row: dict, pbc_images: Path) -> list[tuple[str, Path]]:
    """WBCAtt's `path` column format is not guaranteed, so try the plausible readings
    and report which one resolved."""
    rel = (row.get("path") or "").lstrip("/")
    name = row.get("img_name") or ""
    label = (row.get("label") or "").lower()
    out = []
    if rel:
        out.append(("path_under_images_parent", pbc_images.parent / rel))
        out.append(("path_under_images", pbc_images / rel))
        # path may already start with the images folder name
        parts = Path(rel).parts
        if parts and parts[0] == pbc_images.name:
            out.append(("path_minus_images_prefix", pbc_images / Path(*parts[1:])))
    if name and label:
        out.append(("label_folder_plus_img_name", pbc_images / label / name))
    return out


def _audit_join(pbc_images: Path, wbcatt_dir: Path, mapping: ClassMapping) -> dict:
    if not pbc_images.exists():
        return {"status": "skipped", "reason": "PBC images absent"}
    per_split, _, _ = _read_csvs(wbcatt_dir)
    if not per_split:
        return {"status": "skipped", "reason": "WBCAtt CSVs absent"}

    on_disk: dict[str, list[Path]] = defaultdict(list)
    for p in _image_files(pbc_images):
        on_disk[p.name].append(p)
    n_on_disk = sum(len(v) for v in on_disk.values())
    strategies: Counter = Counter()
    in_csv_only: list[str] = []
    matched_names: set[str] = set()
    counts: dict[str, Counter] = {s: Counter() for s in per_split}

    for split, rows in per_split.items():
        for row in rows:
            resolved = None
            for strategy, candidate in _candidate_paths(row, pbc_images):
                if candidate.exists():
                    resolved, strategies[strategy] = candidate, strategies[strategy] + 1
                    break
            name = row.get("img_name", "")
            if resolved is None:
                in_csv_only.append(f"{split}:{row.get('path') or name}")
            else:
                matched_names.add(resolved.name)
            cls = lookup_class(mapping, "wbcatt", row.get("label", ""))
            counts[split][cls or f"UNMAPPED:{row.get('label', '')}"] += 1

    # Every copy of an unmatched name is listed: a filename duplicated across class
    # folders is two distinct images, and only one of them can match a CSV row.
    on_disk_only = [str(p.relative_to(pbc_images))
                    for name in sorted(set(on_disk) - matched_names)
                    for p in on_disk[name]]
    return {
        "status": "done",
        "n_on_disk": n_on_disk,
        "n_distinct_filenames_on_disk": len(on_disk),
        "n_csv_rows": sum(len(r) for r in per_split.values()),
        "n_matched": len(matched_names),
        "path_resolution_strategies": dict(strategies.most_common()),
        "in_csv_only": _truncate(sorted(in_csv_only)),
        "n_in_csv_only": len(in_csv_only),
        "on_disk_only": _truncate(on_disk_only),
        "n_on_disk_only": len(on_disk_only),
        "counts_per_split_and_class": {s: dict(sorted(c.items()))
                                       for s, c in sorted(counts.items())},
    }


# --- targets --------------------------------------------------------------------------------

def _audit_raabin(root: Path, mapping: ClassMapping) -> dict:
    """TARGET. Counts, class folder names and unreadable files only - no size or mode
    distributions, so no target pixel property can inform a design choice."""
    if not root.exists():
        return {"status": "absent", "path": str(root), "domain": "target"}

    splits: dict[str, dict] = {}
    unmapped: set[str] = set()
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        scan = _scan_class_folders(d, mapping, "raabin", record_stats=False)
        for cls, entry in scan["classes"].items():
            if entry["mapped_to"] is None:
                unmapped.add(cls)
        splits[d.name] = scan

    return {
        "status": "present",
        "path": str(root),
        "domain": "target",
        "splits": splits,
        "unmapped_classes": sorted(unmapped),
        "n_images_total": sum(s["n_images"] for s in splits.values()),
    }


def _audit_lisc(root: Path, mapping: ClassMapping) -> dict:
    """TARGET. LISC labels classes by filename prefix, not folder (see the mapping)."""
    if not root.exists():
        return {"status": "absent", "path": str(root), "domain": "target"}

    splits: dict[str, dict] = {}
    unmapped: set[str] = set()
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs = _image_files(d)
        prefixes = Counter(p.name.split("_")[0] for p in imgs)
        classes = {}
        for prefix, n in sorted(prefixes.items()):
            mapped = lookup_class(mapping, "lisc", prefix)
            if mapped is None:
                unmapped.add(prefix)
            classes[prefix] = {"n_images": n, "mapped_to": mapped}
        splits[d.name] = {
            "n_images": len(imgs),
            "classes": classes,
            "n_class_folders": 0,
            "images_outside_class_folders": len(imgs),
            "n_ignored_dotfiles": len(_dotfiles(d)),
            "extensions": dict(Counter(p.suffix.lower() for p in imgs).most_common()),
            "unreadable": _verify(imgs, d, record_stats=False)["unreadable"],
        }

    return {
        "status": "present",
        "path": str(root),
        "domain": "target",
        "class_source": "filename prefix before the first '_'",
        "splits": splits,
        "unmapped_classes": sorted(unmapped),
        "n_images_total": sum(s["n_images"] for s in splits.values()),
    }


# --- report ----------------------------------------------------------------------------------

def build_report(cfg: AuditConfig) -> dict:
    mapping = cfg.audit.class_mapping
    meta = run_metadata()
    return {
        "audit_version": AUDIT_VERSION,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git": meta["git"],
        "environment": {k: v for k, v in meta.items() if k != "git"},
        "config": cfg.resolved,
        "seeding": set_seed(cfg.seed),
        "source": {
            "pbc": _audit_pbc(cfg.paths.pbc_images, mapping, cfg.audit.verify_images,
                              cfg.audit.tree_max_depth),
            "wbcatt": _audit_wbcatt(cfg.paths.wbcatt_dir, mapping),
            "join": _audit_join(cfg.paths.pbc_images, cfg.paths.wbcatt_dir, mapping),
        },
        "targets": {
            "raabin": _audit_raabin(cfg.paths.raabin_root, mapping),
            "lisc": _audit_lisc(cfg.paths.lisc_root, mapping),
        },
    }


def format_report(rep: dict) -> str:
    L: list[str] = []
    a = L.append
    a("=" * 78)
    a(f"DATA AUDIT  (read-only)   {rep['timestamp']}")
    a(f"git {rep['git']['commit'][:10]}{' DIRTY' if rep['git']['dirty'] else ''}"
      f"   env={rep['config']['env']}")
    a("=" * 78)

    pbc = rep["source"]["pbc"]
    a("")
    a("SOURCE - PBC images")
    if pbc["status"] != "present":
        a(f"  ABSENT: {pbc['path']}")
    else:
        a(f"  {pbc['path']}")
        a(f"  {pbc['n_images']} images, {len(pbc['classes'])} class folders, "
          f"{pbc['n_unreadable']} unreadable, "
          f"ignored: {pbc['n_ignored_dotfiles']} dotfile(s)")
        for f in pbc["ignored_dotfiles"]:
            a(f"    ignored: {f}")
        for cls, e in pbc["classes"].items():
            tag = e["mapped_to"] or ("EXCLUDED" if cls in pbc["excluded_classes"]
                                     else "UNMAPPED")
            a(f"    {cls:<16} {e['n_images']:>6}  -> {tag}")
            if len(e["filename_prefixes"]) > 1:
                pref = ", ".join(f"{k}={v}" for k, v in e["filename_prefixes"].items())
                a(f"      prefixes: {pref}")
        if pbc["duplicate_filenames"]:
            a(f"  duplicate filenames: {len(pbc['duplicate_filenames'])}")
            for name, locs in list(pbc["duplicate_filenames"].items())[:10]:
                a(f"    {name}: {', '.join(locs)}")
        if pbc["unreadable"]:
            a("  unreadable:")
            for u in pbc["unreadable"]:
                a(f"    {u}")
        a(f"  sizes: {pbc['image_sizes']}")
        a(f"  modes: {pbc['image_modes']}")

    w = rep["source"]["wbcatt"]
    a("")
    a("SOURCE - WBCAtt labels")
    if w["status"] != "present":
        a(f"  ABSENT: {w['path']}  missing {w.get('missing_csvs')}")
    else:
        a(f"  rows per split: {w['n_rows_per_split']}  total {w['n_rows_total']}")
        a(f"  per class: {w['counts_per_class']}")
        if w["unexpected_columns"] or w["missing_columns"]:
            a(f"  COLUMNS unexpected={w['unexpected_columns']} "
              f"missing={w['missing_columns']}")
            if w["missing_columns_per_split"]:
                a(f"    missing by split: {w['missing_columns_per_split']}")
        if w["unmapped_labels"]:
            a(f"  UNMAPPED labels: {w['unmapped_labels']}")
        if w["split_overlaps"]:
            a(f"  SPLIT OVERLAP: {w['split_overlaps']}")
        a("  neutrophil filename prefixes per split:")
        for split, counts in w["neutrophil_prefixes_per_split"].items():
            a(f"    {split:<6} " + "  ".join(f"{k}={v}" for k, v in counts.items()))
        for name, sub in w["secondary_subsets"].items():
            if sub["status"] != "measured":
                a(f"  {name}: skipped ({sub['reason']})")
                continue
            a(f"  {name} ({sub['split']}): {sub['n_images']} of "
              f"{sub['n_images_in_split']} images, {sub['n_dropped']} dropped "
              f"{sub['dropped_prefixes']}")
            a(f"    per class: {sub['counts_per_class']}")
        a(f"  attribute values total: {w['n_attribute_values_total']} across 11 columns")
        for col, e in w["attributes"].items():
            a(f"    {col:<28} {e['n_values']:>3} values, {e['n_null']} null")
            a(f"      {dict(list(e['values'].items())[:12])}")
            if e["anomalies"]["whitespace"]:
                a(f"      WHITESPACE: {e['anomalies']['whitespace']}")
            if e["anomalies"]["case_clashes"]:
                a(f"      CASE CLASH: {e['anomalies']['case_clashes']}")

    j = rep["source"]["join"]
    a("")
    a("SOURCE - PBC <-> WBCAtt join")
    if j["status"] != "done":
        a(f"  skipped: {j.get('reason')}")
    else:
        a(f"  {j['n_matched']} of {j['n_csv_rows']} CSV rows resolved to a file; "
          f"{j['n_on_disk']} images on disk")
        a(f"  resolution strategies: {j['path_resolution_strategies']}")
        a(f"  in CSV but no file ({j['n_in_csv_only']}): {j['in_csv_only']}")
        a(f"  on disk but in no CSV ({j['n_on_disk_only']}): {j['on_disk_only']}")
        for split, counts in j["counts_per_split_and_class"].items():
            a(f"    {split:<6} {counts}")

    a("")
    a("-" * 78)
    a("TARGET - RECORD ONLY. Counts and class names only; no size/mode distributions.")
    a("Nothing below may inform any training, tuning or preprocessing decision.")
    a("-" * 78)
    for name in ("raabin", "lisc"):
        t = rep["targets"][name]
        a("")
        a(f"TARGET - {name}")
        if t["status"] != "present":
            a(f"  ABSENT: {t['path']}  (not an error)")
            continue
        a(f"  {t['path']}   {t['n_images_total']} images")
        for split, s in t["splits"].items():
            a(f"    {split:<8} {s['n_images']:>6} images, "
              f"{len(s['classes'])} classes, {len(s['unreadable'])} unreadable, "
              f"ignored: {s['n_ignored_dotfiles']} dotfile(s)")
            for cls, e in s["classes"].items():
                a(f"      {cls:<16} {e['n_images']:>6}  -> {e['mapped_to'] or 'UNMAPPED'}")
            for u in s["unreadable"]:
                a(f"      unreadable: {u}")
        if t["unmapped_classes"]:
            a(f"  UNMAPPED classes (expected until the mapping is extended): "
              f"{t['unmapped_classes']}")
    a("")
    return "\n".join(L)


def run_audit(config_path: str | Path) -> tuple[dict, Path]:
    cfg = load_audit_config(config_path)
    report = build_report(cfg)
    out_dir = cfg.paths.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = report["timestamp"].replace(":", "").replace("-", "")
    out_path = out_dir / f"audit_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n",
                        encoding="utf-8")
    return report, out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True,
                    help="path to an audit config, e.g. configs/audit.yaml")
    args = ap.parse_args(argv)

    report, out_path = run_audit(args.config)
    print(format_report(report))
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
