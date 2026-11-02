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
                              "n_class_folders", "images_outside_class_folders"}


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
    unreadable = " ".join(rep["source"]["pbc"]["unreadable"])
    assert "BA_CORRUPT.jpg" in unreadable
    assert ".DS_169665.jpg" in unreadable


def test_corrupt_target_image_is_listed(report):
    rep, _ = report
    assert any("RB_CORRUPT.jpg" in u
               for u in rep["targets"]["raabin"]["splits"]["TestB"]["unreadable"])


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
    assert rep["targets"]["raabin"]["unmapped_classes"]


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
    assert set(rep["targets"]["raabin"]["splits"]) == {"Train", "TestA", "TestB"}


def test_raabin_testb_class_folders_are_listed(report):
    rep, _ = report
    assert set(rep["targets"]["raabin"]["splits"]["TestB"]["classes"]) == {
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
