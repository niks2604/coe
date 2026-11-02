"""The audit is read-only, writes no normalisation statistics, and prints no
size/mode distributions for target datasets."""
import hashlib
import json
from pathlib import Path

import pytest

from src.data.audit import main, run_audit

FORBIDDEN_KEY_SUBSTRINGS = ("mean", "std", "stain", "norm", "percentile", "histogram_rgb")


def tree_fingerprint(root: Path) -> dict:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def all_keys(node):
    if isinstance(node, dict):
        for k, v in node.items():
            yield k
            yield from all_keys(v)
    elif isinstance(node, list):
        for v in node:
            yield from all_keys(v)


@pytest.fixture
def report(audit_config):
    rep, path = run_audit(audit_config)
    return rep, path


# --- read-only ---------------------------------------------------------------------

def test_audit_does_not_modify_the_data_tree(audit_config, tiny_raw):
    before = tree_fingerprint(tiny_raw)
    run_audit(audit_config)
    assert tree_fingerprint(tiny_raw) == before


def test_audit_computes_no_normalisation_statistics(report):
    rep, _ = report
    findings = {"source": rep["source"], "targets": rep["targets"]}
    offenders = [k for k in all_keys(findings)
                 if any(s in k.lower() for s in FORBIDDEN_KEY_SUBSTRINGS)]
    assert offenders == []


def test_targets_get_no_size_or_mode_distributions(report):
    """Residual-risk mitigation: target datasets report counts, corrupt files and
    class folder names only, so no target pixel statistic can inform a design choice."""
    rep, _ = report
    raabin = rep["targets"]["raabin"]
    assert "image_sizes" not in json.dumps(raabin)
    assert "image_modes" not in json.dumps(raabin)
    for split in raabin["splits"].values():
        assert set(split) <= {"n_images", "classes", "unreadable", "extensions",
                              "n_class_folders", "images_outside_class_folders",
                              "n_ignored_dotfiles"}


def test_source_does_get_size_and_mode_distributions(report):
    rep, _ = report
    assert rep["source"]["pbc"]["image_sizes"]
    assert rep["source"]["pbc"]["image_modes"]


# --- exit status and missing datasets -----------------------------------------------

def test_audit_exits_zero_on_the_fixtures(audit_config, capsys):
    assert main(["--config", str(audit_config)]) == 0
    assert capsys.readouterr().out.strip()


def test_missing_lisc_is_reported_absent_not_fatal(report):
    rep, _ = report
    assert rep["targets"]["lisc"]["status"] == "absent"


def test_present_dataset_is_reported_present(report):
    rep, _ = report
    assert rep["targets"]["raabin"]["status"] == "present"
    assert rep["source"]["pbc"]["status"] == "present"


# --- integrity findings --------------------------------------------------------------

def test_corrupt_images_are_listed_not_crashed(report):
    rep, _ = report
    assert "BA_CORRUPT.jpg" in " ".join(rep["source"]["pbc"]["unreadable"])


def test_corrupt_target_image_is_listed(report):
    rep, _ = report
    assert any("RB_CORRUPT.jpg" in u
               for u in rep["targets"]["raabin"]["splits"]["Test-B"]["unreadable"])


def test_duplicate_filenames_within_a_dataset_are_reported(report):
    rep, _ = report
    dupes = rep["source"]["pbc"]["duplicate_filenames"]
    assert "DUPE_1.jpg" in dupes
    assert len(dupes["DUPE_1.jpg"]) == 2


def test_class_folders_are_listed_with_counts(report):
    rep, _ = report
    classes = rep["source"]["pbc"]["classes"]
    assert classes["basophil"]["n_images"] == 4   # BA_1, BA_2, DUPE_1, BA_CORRUPT
    assert "platelet" in classes


def test_unmapped_and_excluded_classes_are_reported_not_fatal(report):
    rep, _ = report
    assert rep["source"]["pbc"]["excluded_classes"] == ["platelet"]


def test_raabin_classes_are_now_all_mapped(report):
    """They were added to the mapping after the 2026-11-02 audit was reviewed."""
    rep, _ = report
    assert rep["targets"]["raabin"]["unmapped_classes"] == []
    for split in rep["targets"]["raabin"]["splits"].values():
        for cls, entry in split["classes"].items():
            assert entry["mapped_to"] is not None, cls


# --- PBC <-> WBCAtt join -------------------------------------------------------------

def test_join_reports_images_on_disk_but_not_in_wbcatt(report):
    rep, _ = report
    assert "MO_ONLY_ON_DISK.jpg" in " ".join(rep["source"]["join"]["on_disk_only"])


def test_join_reports_csv_rows_with_no_file(report):
    rep, _ = report
    assert "GHOST_1.jpg" in " ".join(rep["source"]["join"]["in_csv_only"])


def test_join_reports_counts_per_split_and_class(report):
    rep, _ = report
    counts = rep["source"]["join"]["counts_per_split_and_class"]
    assert set(counts) == {"train", "val", "test"}
    assert counts["test"]["neutrophil"] == 3


def test_splits_must_not_overlap(report):
    rep, _ = report
    assert rep["source"]["wbcatt"]["split_overlaps"] == {}


def test_measured_totals_are_recorded_for_comparison(report):
    """The expected counts are not trusted; the audit measures and reports them."""
    rep, _ = report
    assert rep["source"]["wbcatt"]["n_rows_total"] > 0
    assert rep["source"]["wbcatt"]["counts_per_class"]


# --- WBCAtt attribute columns --------------------------------------------------------

def test_the_eleven_attribute_columns_are_checked(report):
    rep, _ = report
    attrs = rep["source"]["wbcatt"]["attributes"]
    assert len(attrs) == 11
    assert "nucleus_shape" in attrs


def test_unique_values_per_attribute_are_listed_with_counts(report):
    rep, _ = report
    vals = rep["source"]["wbcatt"]["attributes"]["cytoplasm_colour"]["values"]
    assert vals["light blue"] > 0


def test_total_attribute_value_count_is_reported(report):
    """Expected to be 31 across the 11 columns on the real data; measured here."""
    rep, _ = report
    assert rep["source"]["wbcatt"]["n_attribute_values_total"] == sum(
        len(a["values"]) for a in rep["source"]["wbcatt"]["attributes"].values()
    )


def test_whitespace_anomalies_in_attribute_values_are_flagged(report):
    rep, _ = report
    assert " loosely " in rep["source"]["wbcatt"]["attributes"]["chromatin_density"]["anomalies"]["whitespace"]


def test_nulls_per_attribute_are_reported(report):
    rep, _ = report
    assert rep["source"]["wbcatt"]["attributes"]["granularity"]["n_null"] == 0


# --- Raabin ---------------------------------------------------------------------------

def test_raabin_splits_are_reported_separately(report):
    rep, _ = report
    assert set(rep["targets"]["raabin"]["splits"]) == {"Train", "TestA", "Test-B"}


def test_raabin_testb_class_folders_are_listed(report):
    rep, _ = report
    assert set(rep["targets"]["raabin"]["splits"]["Test-B"]["classes"]) == {
        "Neutrophil", "Lymphocyte",
    }


# --- output ----------------------------------------------------------------------------

def test_report_json_is_written_and_parses(report):
    _, path = report
    assert path.exists()
    assert path.name.startswith("audit_") and path.suffix == ".json"
    json.loads(path.read_text())


def test_report_has_git_commit_and_dirty_flag(report):
    rep, _ = report
    assert len(rep["git"]["commit"]) == 40
    assert isinstance(rep["git"]["dirty"], bool)


def test_report_embeds_the_resolved_config(report):
    rep, _ = report
    assert rep["config"]["env"] == "local"
    assert isinstance(rep["config"]["audit"]["class_mapping"], dict)


def test_report_is_json_round_trippable(report):
    rep, _ = report
    assert json.loads(json.dumps(rep))["audit_version"] == rep["audit_version"]


# --- regressions -----------------------------------------------------------------------

def test_present_attribute_columns_are_not_reported_missing(report):
    """Regression: `set(A) | {x} - set(header)` parses as `set(A) | ({x} - header)`,
    so every attribute column was reported missing even when present."""
    rep, _ = report
    assert rep["source"]["wbcatt"]["missing_columns"] == []


def test_missing_columns_are_still_detected(tmp_path, audit_config, tiny_raw):
    """The other direction: a genuinely absent column must be reported."""
    import csv

    path = tiny_raw / "pbc_attr_v1_train.csv"
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    fields = [f for f in rows[0] if f != "granularity"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    rep, _ = run_audit(audit_config)
    assert rep["source"]["wbcatt"]["missing_columns"] == ["granularity"]


def test_join_counts_every_file_on_disk_not_unique_filenames(report):
    """Regression: on-disk images were keyed by filename, so DUPE_1.jpg counted once
    and n_on_disk understated the archive."""
    rep, _ = report
    assert rep["source"]["join"]["n_on_disk"] == rep["source"]["pbc"]["n_images"]


def test_join_reports_both_copies_of_a_duplicated_filename(report):
    rep, _ = report
    on_disk_only = rep["source"]["join"]["on_disk_only"]
    assert sum("DUPE_1.jpg" in p for p in on_disk_only) == 2


# --- (a) per-split neutrophil prefix counts, source only ------------------------------

def test_neutrophil_prefix_counts_per_wbcatt_split(report):
    """Band vs segmented is only visible in the filename, and the band decision in
    configs/class_mapping.yaml rests on these counts per split."""
    rep, _ = report
    per_split = rep["source"]["wbcatt"]["neutrophil_prefixes_per_split"]
    assert set(per_split) == {"train", "val", "test"}
    assert per_split["test"] == {"SNE_": 1, "BNE_": 1, "NEUTROPHIL_": 1, "other": 0}


def test_prefix_counts_cover_every_neutrophil_row(report):
    rep, _ = report
    w = rep["source"]["wbcatt"]
    counted = sum(sum(v.values()) for v in w["neutrophil_prefixes_per_split"].values())
    assert counted == w["counts_per_class"]["neutrophil"]


def test_secondary_subset_size_is_measured(report):
    """The mapping claims 2,606 of 3,099 for the real data; the audit measures it
    rather than restating it."""
    rep, _ = report
    sub = rep["source"]["wbcatt"]["secondary_subsets"]["pbc_test_no_band_neutrophils"]
    assert sub["split"] == "pbc_test"
    assert sub["n_images"] == 3          # 5 test rows, BNE_ and NEUTROPHIL_ dropped
    assert sub["n_dropped"] == 2
    assert sub["dropped_prefixes"] == {"BNE_": 1, "NEUTROPHIL_": 1}
    assert sub["counts_per_class"]["neutrophil"] == 1


def test_secondary_subset_prefixes_come_from_the_mapping(report):
    """Not hardcoded in the audit: the drop list lives in class_mapping.yaml."""
    rep, _ = report
    spec = rep["config"]["audit"]["class_mapping"]["secondary_subsets"]
    expected = spec["pbc_test_no_band_neutrophils"]["drop_filename_prefixes"]["pbc:neutrophil"]
    sub = rep["source"]["wbcatt"]["secondary_subsets"]["pbc_test_no_band_neutrophils"]
    assert sorted(sub["dropped_prefixes"]) == sorted(expected)


def test_prefix_counts_are_source_only(report):
    """No filename-derived breakdown for any target dataset."""
    rep, _ = report
    blob = json.dumps(rep["targets"])
    assert "prefix" not in blob.lower()
    assert "secondary_subsets" not in blob


# --- (b) dotfiles are ignored, not unreadable -----------------------------------------

def test_dotfiles_are_ignored_not_reported_unreadable(report):
    """.DS_169665.jpg is macOS metadata with a .jpg name. Counting it as a corrupt
    image buries the one genuinely corrupt file in the same list."""
    rep, _ = report
    pbc = rep["source"]["pbc"]
    assert ".DS_169665.jpg" not in " ".join(pbc["unreadable"])
    assert any(".DS_169665.jpg" in f for f in pbc["ignored_dotfiles"])
    assert pbc["n_ignored_dotfiles"] == 1


def test_dotfiles_are_excluded_from_image_counts(report):
    rep, _ = report
    pbc = rep["source"]["pbc"]
    assert pbc["n_images"] == 15          # 16 files, one of them a dotfile
    assert pbc["classes"]["neutrophil"]["n_images"] == 3


def test_dotfiles_are_excluded_from_the_join(report):
    rep, _ = report
    assert ".DS_169665.jpg" not in " ".join(rep["source"]["join"]["on_disk_only"])


def test_dotfile_prefix_is_not_reported_as_a_class_prefix(report):
    """The .DS_ prefix used to appear in the neutrophil prefix histogram."""
    rep, _ = report
    assert ".DS_" not in rep["source"]["pbc"]["classes"]["neutrophil"]["filename_prefixes"]


def test_targets_report_dotfile_counts_only(report):
    rep, _ = report
    for split in rep["targets"]["raabin"]["splits"].values():
        assert isinstance(split["n_ignored_dotfiles"], int)
