"""A dump must reproduce the evaluation it claims to come from.

The dumps are a second forward pass over the same images with the same checkpoint, so
macro-F1 and ECE recomputed from a dump must match the matching eval_records entry. If
they do not, the dump is reading something other than what was ledgered - a different
checkpoint, a different preprocessing, a different device - and no analysis may be read
off it. The mismatch is fatal, never a warning.

Both metrics go through src.metrics, the same functions the evaluation used, so the
comparison is of the predictions and not of two implementations.

Synthetic dumps and synthetic records only; eval_records/ is never written.
"""
import json

import numpy as np
import pytest

from scripts.check_dump_consistency import (
    ConsistencyError,
    check_dumps,
    recompute,
)

CLASSES = ["neutrophil", "lymphocyte", "monocyte", "eosinophil", "basophil"]


def make_dump(path, logits, y_true, domain="raabin_testA", run_id="bottleneck", seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"domain": domain, "run_id": run_id, "seed": seed, "model_type": run_id,
            "classes": CLASSES, "class_logits_mode": "soft",
            "checkpoint_sha256": f"{run_id}{seed}".ljust(64, "0"),
            "git": {"commit": "abc1234", "dirty": False}, "n": len(y_true)}
    np.savez_compressed(path, image_ids=np.array([f"I{i}" for i in range(len(y_true))]),
                        y_true=np.asarray(y_true, dtype=np.int16),
                        class_logits=np.asarray(logits, dtype=np.float32),
                        meta=np.array(json.dumps(meta)))
    return path


def make_record(path, scores, run_id="bottleneck", seed=0):
    """scores: {domain: {"macro_f1": x, "ece": y, "n": n}}"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "run_id": run_id, "seed": seed, "dry_run": False,
        "checkpoint_sha256": f"{run_id}{seed}".ljust(64, "0"),
        "attribute_modes": ["soft", "hard"],
        "domains": sorted(scores),
        "results": {d: {"soft": {"macro_f1": v["macro_f1"], "ece": v["ece"],
                                 "n": v["n"], "macro_f1_classes": CLASSES}}
                    for d, v in scores.items()},
    }, indent=2))
    return path


@pytest.fixture
def pair(tmp_path):
    """A dump and a record that agree, by construction."""
    def build(n=120, seed=0, run_id="bottleneck", domain="raabin_testA"):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 5, size=n)
        logits = rng.normal(0, 1, size=(n, 5)).astype(np.float32)
        logits[np.arange(n), y] += 2.0
        dump = make_dump(tmp_path / "analysis" / f"{run_id}_seed{seed}_{domain}.npz",
                         logits, y, domain=domain, run_id=run_id, seed=seed)
        truth = recompute(logits, y, CLASSES, bins=15)
        record = make_record(
            tmp_path / "eval_records" / f"{run_id}_seed{seed}_eval.json",
            {domain: {"macro_f1": truth["macro_f1"], "ece": truth["ece"], "n": n}},
            run_id=run_id, seed=seed)
        return tmp_path / "analysis", tmp_path / "eval_records", dump, record
    return build


# --- agreement ----------------------------------------------------------------------

def test_a_matching_pair_passes(pair):
    analysis, records, _, _ = pair()
    report = check_dumps(analysis, records)
    assert report["checked"] == 1
    assert not report["mismatches"]


def test_the_report_states_both_recomputed_and_recorded_values(pair):
    analysis, records, _, _ = pair()
    row = check_dumps(analysis, records)["rows"][0]
    for key in ("macro_f1_dump", "macro_f1_record", "ece_dump", "ece_record",
                "macro_f1_delta", "ece_delta"):
        assert key in row, key


def test_metrics_go_through_src_metrics(pair):
    """Not a local reimplementation: the comparison must be of the predictions."""
    import inspect

    import scripts.check_dump_consistency as module

    source = inspect.getsource(module)
    assert "from src.metrics import" in source


# --- mismatch is fatal ------------------------------------------------------------------

def test_a_macro_f1_mismatch_is_fatal(pair, tmp_path):
    analysis, records, _, record = pair()
    doc = json.loads(record.read_text())
    doc["results"]["raabin_testA"]["soft"]["macro_f1"] += 0.05
    record.write_text(json.dumps(doc))
    with pytest.raises(ConsistencyError, match="macro_f1|mismatch"):
        check_dumps(analysis, records)


def test_an_ece_mismatch_is_fatal(pair):
    analysis, records, _, record = pair()
    doc = json.loads(record.read_text())
    doc["results"]["raabin_testA"]["soft"]["ece"] += 0.05
    record.write_text(json.dumps(doc))
    with pytest.raises(ConsistencyError, match="ece|mismatch"):
        check_dumps(analysis, records)


def test_a_row_count_mismatch_is_fatal(pair):
    analysis, records, _, record = pair()
    doc = json.loads(record.read_text())
    doc["results"]["raabin_testA"]["soft"]["n"] += 1
    record.write_text(json.dumps(doc))
    with pytest.raises(ConsistencyError, match="n |rows|count"):
        check_dumps(analysis, records)


def test_a_checkpoint_hash_mismatch_is_fatal(pair):
    """A dump from a different checkpoint than the record names is the failure this
    check exists to catch, whatever the metrics happen to say."""
    analysis, records, _, record = pair()
    doc = json.loads(record.read_text())
    doc["checkpoint_sha256"] = "f" * 64
    record.write_text(json.dumps(doc))
    with pytest.raises(ConsistencyError, match="checkpoint"):
        check_dumps(analysis, records)


def test_a_dump_with_no_matching_record_is_fatal(pair, tmp_path):
    analysis, records, _, _ = pair()
    make_dump(analysis / "control_seed2_raabin_testA.npz",
              np.zeros((4, 5), dtype=np.float32), [0, 1, 2, 3],
              run_id="control", seed=2)
    with pytest.raises(ConsistencyError, match="no .*record|record"):
        check_dumps(analysis, records)


def test_a_domain_absent_from_the_record_is_fatal(pair):
    analysis, records, dump, record = pair()
    doc = json.loads(record.read_text())
    doc["results"] = {}
    doc["domains"] = []
    record.write_text(json.dumps(doc))
    with pytest.raises(ConsistencyError, match="domain|raabin_testA"):
        check_dumps(analysis, records)


def test_the_tolerance_is_reported_and_respected(pair):
    analysis, records, _, record = pair()
    doc = json.loads(record.read_text())
    doc["results"]["raabin_testA"]["soft"]["ece"] += 1e-7
    record.write_text(json.dumps(doc))
    report = check_dumps(analysis, records, tolerance=1e-5)
    assert report["tolerance"] == 1e-5
    with pytest.raises(ConsistencyError):
        check_dumps(analysis, records, tolerance=1e-12)


def test_the_error_names_the_file_and_both_values(pair):
    analysis, records, _, record = pair()
    doc = json.loads(record.read_text())
    doc["results"]["raabin_testA"]["soft"]["macro_f1"] = 0.123456
    record.write_text(json.dumps(doc))
    with pytest.raises(ConsistencyError) as excinfo:
        check_dumps(analysis, records)
    message = str(excinfo.value)
    assert "raabin_testA" in message and "0.1234" in message


# --- it writes nothing ----------------------------------------------------------------------

def test_eval_records_are_never_modified(pair):
    analysis, records, _, record = pair()
    before = record.read_bytes()
    check_dumps(analysis, records)
    assert record.read_bytes() == before


def test_the_real_repo_ledger_and_records_are_untouched(pair):
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    primary = (repo / "target_eval_ledger.jsonl").read_bytes()
    records = [(p.name, p.read_bytes())
               for p in sorted((repo / "eval_records").glob("*.json"))]
    analysis, record_dir, _, _ = pair()
    check_dumps(analysis, record_dir)
    assert (repo / "target_eval_ledger.jsonl").read_bytes() == primary
    assert [(p.name, p.read_bytes())
            for p in sorted((repo / "eval_records").glob("*.json"))] == records


def test_an_empty_analysis_directory_is_refused(tmp_path):
    (tmp_path / "analysis").mkdir()
    (tmp_path / "eval_records").mkdir()
    with pytest.raises(ConsistencyError, match="no .*npz|found"):
        check_dumps(tmp_path / "analysis", tmp_path / "eval_records")


def test_a_pbc_val_dump_is_skipped_not_compared(pair, tmp_path):
    """pbc_val is never evaluated or ledgered, so there is nothing to compare it with.
    Skipped and counted, not treated as a missing record."""
    analysis, records, _, _ = pair()
    make_dump(analysis / "bottleneck_seed0_pbc_val.npz",
              np.zeros((4, 5), dtype=np.float32), [0, 1, 2, 3], domain="pbc_val")
    report = check_dumps(analysis, records)
    assert report["skipped"] == ["bottleneck_seed0_pbc_val.npz"]
