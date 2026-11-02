"""The target evaluation ledger.

Repeated evaluation on held-out data is tuning on it: look enough times and keep the
best, and the protocol is void. So every (checkpoint, domain) pair is recorded once and
a repeat is refused unless it is explicitly overridden with a reason that is written
down (CLAUDE.md sec 5).

pbc_test is ledgered alongside the target domains. It is held out for the source-domain
report, so repeated looks at it are the same error. pbc_val is not: it is the tuning
split and is read constantly by design.

What this cannot do, stated plainly: entries key on the checkpoint hash, so three seeds
give three legitimate entries and retraining yields a new hash. The ledger is an audit
trail that makes repeat looks visible in git history, not a guarantee.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

from src.config import REPO_ROOT, run_metadata
from src.data.domains import DATASETS, Domain, domain_of

# Repo root, not results/: a ledger that resets when results are cleared constrains
# nothing. Committed to git (CLAUDE.md sec 5).
LEDGER_PATH = REPO_ROOT / "target_eval_ledger.jsonl"

REQUIRED_FIELDS = ("checkpoint_sha256", "domain", "run_id", "config_sha256")
TUNING_SPLIT = "pbc_val"


class LedgerError(Exception):
    """A refused repeat, a malformed entry, or an unreadable ledger."""


def ledgered_domains() -> list[str]:
    """Every target domain, plus pbc_test. Everything except the tuning split."""
    return sorted(d for d in DATASETS if d != TUNING_SPLIT
                  and (domain_of(d) is Domain.TARGET or d == "pbc_test"))


def is_ledgered(domain: str) -> bool:
    return domain in ledgered_domains()


def checkpoint_hash(path: str | Path) -> str:
    """sha256 of the checkpoint file's bytes. Content-addressed, so two seeds differ."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ledger(path: str | Path = LEDGER_PATH) -> list[dict]:
    """All entries. A missing file reads as empty; an unreadable line is an error.

    Skipping a corrupt line would quietly permit a repeat, which is the one failure this
    file exists to prevent.
    """
    path = Path(path)
    if not path.exists():
        return []
    entries = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise LedgerError(
                f"{path}: line {number} is not valid JSON ({exc}). The ledger is "
                f"append-only and is not repaired automatically: a skipped line would "
                f"silently permit a repeat evaluation."
            ) from None
    return entries


def append_entry(path: str | Path, entry: dict) -> dict:
    """Append one entry, filling in the timestamp and git state."""
    missing = [f for f in REQUIRED_FIELDS if not entry.get(f)]
    if missing:
        raise LedgerError(f"ledger entry is missing required field(s) {missing}: {entry}")
    domain = entry["domain"]
    if domain not in DATASETS:
        raise LedgerError(
            f"unknown domain {domain!r}; known: {sorted(DATASETS)}. An unregistered "
            f"domain would be recorded but never matched on a later look."
        )
    if not is_ledgered(domain):
        raise LedgerError(
            f"{domain!r} is not a ledgered domain; ledgered: {ledgered_domains()}")

    meta = run_metadata()
    row = dict(entry)
    row.setdefault("timestamp",
                   dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    row.setdefault("git_commit", meta["git"]["commit"])
    row.setdefault("git_dirty", meta["git"]["dirty"])

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def find_entries(path: str | Path, checkpoint_sha256: str, domain: str) -> list[dict]:
    return [row for row in read_ledger(path)
            if row.get("checkpoint_sha256") == checkpoint_sha256
            and row.get("domain") == domain]


def assert_not_evaluated(path: str | Path, checkpoint_sha256: str, domain: str,
                         override_reason: str | None = None) -> None:
    """Gate before evaluating a checkpoint on a ledgered domain."""
    if not is_ledgered(domain):
        return
    prior = find_entries(path, checkpoint_sha256, domain)

    if not prior:
        if override_reason is not None:
            # An override with nothing to override would hide that it was passed.
            raise LedgerError(
                f"--override-reason was given but there is no prior evaluation of this "
                f"checkpoint on {domain!r} to override. Remove the flag."
            )
        return

    if override_reason is None:
        seen = ", ".join(f"{row.get('timestamp')} ({row.get('git_commit', '')[:10]})"
                         for row in prior)
        raise LedgerError(
            f"this checkpoint has already been evaluated on {domain!r}: {seen}. "
            f"Repeated evaluation on held-out data is tuning on it. To do it anyway, "
            f"pass --override-reason \"<why>\"; the reason is written into the ledger."
        )
    if not str(override_reason).strip():
        raise LedgerError(
            "--override-reason must be a non-empty reason; it is the record of why a "
            "held-out domain was looked at twice."
        )
