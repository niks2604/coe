"""The target evaluation ledger. Repeated target evaluation is tuning on the target, so
a repeat is refused unless it is explicitly overridden and logged."""
import json

import pytest

from src.ledger import (
    LedgerError,
    append_entry,
    assert_not_evaluated,
    checkpoint_hash,
    read_ledger,
)

LEDGERED = ["pbc_test", "raabin_train", "raabin_testA", "raabin_testB"]


@pytest.fixture
def ledger(tmp_path):
    return tmp_path / "target_eval_ledger.jsonl"


@pytest.fixture
def ckpt(tmp_path):
    path = tmp_path / "best.pt"
    path.write_bytes(b"pretend weights")
    return path


def entry(h, domain, **kw):
    return dict({"checkpoint_sha256": h, "domain": domain, "run_id": "bottleneck",
                 "seed": 0, "config_sha256": "c" * 64, "git_commit": "a" * 40,
                 "git_dirty": False}, **kw)


# --- hashing ------------------------------------------------------------------------

def test_checkpoint_hash_is_stable_and_content_addressed(ckpt, tmp_path):
    same = tmp_path / "copy.pt"
    same.write_bytes(ckpt.read_bytes())
    assert checkpoint_hash(ckpt) == checkpoint_hash(same)
    assert len(checkpoint_hash(ckpt)) == 64


def test_different_weights_hash_differently(ckpt, tmp_path):
    other = tmp_path / "other.pt"
    other.write_bytes(b"different weights")
    assert checkpoint_hash(ckpt) != checkpoint_hash(other)


# --- refusing a repeat ----------------------------------------------------------------

@pytest.mark.parametrize("domain", LEDGERED)
def test_first_evaluation_is_allowed(ledger, ckpt, domain):
    assert_not_evaluated(ledger, checkpoint_hash(ckpt), domain)


@pytest.mark.parametrize("domain", LEDGERED)
def test_second_evaluation_of_the_same_pair_is_refused(ledger, ckpt, domain):
    h = checkpoint_hash(ckpt)
    append_entry(ledger, entry(h, domain))
    with pytest.raises(LedgerError, match=domain):
        assert_not_evaluated(ledger, h, domain)


def test_pbc_test_is_ledgered_like_a_target(ledger, ckpt):
    """It is held out for the source-domain report, so repeated looks are the same
    error (CLAUDE.md sec 5)."""
    h = checkpoint_hash(ckpt)
    append_entry(ledger, entry(h, "pbc_test"))
    with pytest.raises(LedgerError):
        assert_not_evaluated(ledger, h, "pbc_test")


def test_pbc_val_is_not_ledgered(ledger, ckpt):
    """pbc_val is the tuning split and is read constantly by design."""
    h = checkpoint_hash(ckpt)
    append_entry(ledger, entry(h, "pbc_test"))
    assert_not_evaluated(ledger, h, "pbc_val")


def test_a_different_domain_is_still_allowed(ledger, ckpt):
    h = checkpoint_hash(ckpt)
    append_entry(ledger, entry(h, "raabin_testA"))
    assert_not_evaluated(ledger, h, "raabin_testB")


def test_a_different_seed_is_a_different_checkpoint_and_allowed(ledger, ckpt, tmp_path):
    """Three seeds give three legitimate entries."""
    other = tmp_path / "seed1.pt"
    other.write_bytes(b"seed 1 weights")
    append_entry(ledger, entry(checkpoint_hash(ckpt), "raabin_testA"))
    assert_not_evaluated(ledger, checkpoint_hash(other), "raabin_testA")


def test_refusal_names_the_prior_entry(ledger, ckpt):
    h = checkpoint_hash(ckpt)
    append_entry(ledger, entry(h, "raabin_testB", timestamp="2026-11-02T10:00:00+00:00"))
    with pytest.raises(LedgerError, match="2026-11-02"):
        assert_not_evaluated(ledger, h, "raabin_testB")


# --- the override ----------------------------------------------------------------------

def test_override_requires_a_reason(ledger, ckpt):
    h = checkpoint_hash(ckpt)
    append_entry(ledger, entry(h, "raabin_testA"))
    for bad in (None, "", "   "):
        with pytest.raises(LedgerError, match="reason"):
            assert_not_evaluated(ledger, h, "raabin_testA", override_reason=bad)


def test_override_with_a_reason_is_allowed(ledger, ckpt):
    h = checkpoint_hash(ckpt)
    append_entry(ledger, entry(h, "raabin_testA"))
    assert_not_evaluated(ledger, h, "raabin_testA",
                         override_reason="metrics.py ECE bug fixed, see commit abc123")


def test_the_override_reason_is_written_into_the_ledger(ledger, ckpt):
    h = checkpoint_hash(ckpt)
    append_entry(ledger, entry(h, "raabin_testA"))
    append_entry(ledger, entry(h, "raabin_testA", override_reason="rerun after bugfix"))
    assert read_ledger(ledger)[-1]["override_reason"] == "rerun after bugfix"


def test_override_on_a_first_evaluation_is_rejected(ledger, ckpt):
    """An override with nothing to override hides the fact that it was passed."""
    with pytest.raises(LedgerError, match="no prior"):
        assert_not_evaluated(ledger, checkpoint_hash(ckpt), "raabin_testA",
                             override_reason="just in case")


# --- the file ---------------------------------------------------------------------------

def test_entries_are_append_only(ledger, ckpt):
    h = checkpoint_hash(ckpt)
    append_entry(ledger, entry(h, "raabin_testA"))
    first = ledger.read_text()
    append_entry(ledger, entry(h, "raabin_testB"))
    assert ledger.read_text().startswith(first)


def test_every_entry_records_git_state_and_a_timestamp(ledger, ckpt):
    append_entry(ledger, entry(checkpoint_hash(ckpt), "raabin_testA"))
    row = read_ledger(ledger)[0]
    assert len(row["git_commit"]) == 40
    assert isinstance(row["git_dirty"], bool)
    assert row["timestamp"]


def test_each_line_is_standalone_json(ledger, ckpt):
    h = checkpoint_hash(ckpt)
    append_entry(ledger, entry(h, "raabin_testA"))
    append_entry(ledger, entry(h, "raabin_testB"))
    for line in ledger.read_text().strip().splitlines():
        json.loads(line)


def test_a_corrupt_line_is_reported_not_skipped(ledger, ckpt):
    """Silently skipping an unreadable line would quietly permit a repeat."""
    append_entry(ledger, entry(checkpoint_hash(ckpt), "raabin_testA"))
    with ledger.open("a") as fh:
        fh.write("{not json\n")
    with pytest.raises(LedgerError, match="line 2"):
        read_ledger(ledger)


def test_an_entry_missing_a_required_field_is_rejected(ledger):
    bad = {"domain": "raabin_testA"}
    with pytest.raises(LedgerError, match="checkpoint_sha256"):
        append_entry(ledger, bad)


def test_an_unknown_domain_is_rejected(ledger, ckpt):
    with pytest.raises(LedgerError, match="raabin_testC"):
        append_entry(ledger, entry(checkpoint_hash(ckpt), "raabin_testC"))


def test_missing_ledger_file_reads_as_empty(tmp_path):
    assert read_ledger(tmp_path / "absent.jsonl") == []


def test_the_committed_ledger_lives_at_the_repo_root():
    """Under results/ it would reset whenever results are cleared (CLAUDE.md sec 5)."""
    from src.ledger import LEDGER_PATH

    assert LEDGER_PATH.name == "target_eval_ledger.jsonl"
    assert LEDGER_PATH.parent.name == "wbc-attributes"


def test_the_repo_root_ledger_is_untouched_by_the_test_run(repo_ledger_snapshot):
    """Checked mid-run as well as at session teardown, so a failure points at the test
    that caused it rather than only at the end of the session."""
    from src.ledger import LEDGER_PATH

    current = LEDGER_PATH.read_bytes() if LEDGER_PATH.exists() else None
    assert current == repo_ledger_snapshot


def test_the_repo_root_ledger_exists_and_is_committed():
    """It is committed so the history of held-out evaluations lives in git."""
    import subprocess

    from src.config import REPO_ROOT
    from src.ledger import LEDGER_PATH

    assert LEDGER_PATH.exists()
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", LEDGER_PATH.name],
                             cwd=REPO_ROOT, capture_output=True, text=True)
    assert tracked.returncode == 0, f"{LEDGER_PATH.name} is not tracked by git"
