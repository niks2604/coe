# The nine final runs

Three models × three seeds, on Kaggle, followed by one evaluation pass that touches
Raabin exactly once. Setup (cloning, dependencies, `torch.cuda.is_available()`,
`CUBLAS_WORKSPACE_CONFIG`, the test suite) is in [kaggle_runbook.md](kaggle_runbook.md)
and is not repeated here.

Read section 0 before booking any GPU time. Two things in it are blockers.

---

## 0. Blockers and prerequisites

### 0.1 RESOLVED — the nine configs

`seed` lives in the YAML and CLAUDE.md §5 forbids a CLI flag that changes behaviour the
config does not record, so each of the nine runs is its own config. Seed 0 keeps the
original names; seeds 1 and 2 are new files differing in `seed` alone:

| model | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| baseline | `baseline.yaml` | `baseline_seed1.yaml` | `baseline_seed2.yaml` |
| bottleneck | `bottleneck.yaml` | `bottleneck_seed1.yaml` | `bottleneck_seed2.yaml` |
| control | `control.yaml` | `control_seed1.yaml` | `control_seed2.yaml` |

`run_id` is shared across the three seeds of a model, so `src.train` writes to
`results/<model>_seed<N>/` — which for seeds 1 and 2 is also the config's own basename.
Tests assert `seed` is the only field that differs, and that no two experiment configs
can resolve to the same results directory.

### 0.2 RESOLVED — evaluation writes its record

Every evaluation now writes `<results_dir>/<run_id>_seed<seed>_eval.json` containing
everything `evaluate_checkpoint` returns: all metrics per domain and per mode, the
checkpoint hash, the resolved config, the git state, the selection role and rule, the
abstention thresholds and the buffer hashes. A dry run writes
`..._eval_dryrun.json` instead, so a cheap repeatable run cannot overwrite the record
of the one look.

The record is written **before** any ledger line, and a failed write aborts before the
ledger is touched: an entry saying a domain was looked at, with nothing saying what was
found, would spend the look and keep nothing. An existing record is not overwritten
without `--override-reason`.

### 0.3 ImageNet weights

All three configs carry `imagenet_weights: null`, which downloads. A *Save & Run All*
commit has no internet unless the notebook's internet setting is on, and that needs a
verified phone number on the account. If internet is off the run fails loudly rather than
training from random init (§5), which is correct but wastes the booking.

Either turn internet on for the notebook, or attach the ResNet-50 weights as a Kaggle
dataset and set `model.imagenet_weights` to that path in all nine configs. Decide before
booking.

### 0.4 Check the ledger is still empty

```bash
wc -l target_eval_ledger.jsonl        # expect 0
```

It is 0 as of this writing. A non-zero count before the final runs means a target or
`pbc_test` domain has already been looked at, and you need to read the file and know why
before adding to it.

### 0.5 Pre-run checklist

- [ ] 0.1 decided and the six configs committed, with a test
- [ ] 0.2 is in place (nothing to do)
- [ ] 0.3 decided; `imagenet_weights` consistent across all nine configs
- [ ] `target_eval_ledger.jsonl` is 0 lines
- [ ] Working tree clean and pushed — every result records the commit and a dirty flag,
      and a dirty final run is not reproducible
- [ ] `pytest` green on Kaggle, after the runbook's dependency step
- [ ] GPU quota: check the remaining weekly allowance against section 2's estimate

---

## 1. Why three notebooks and not one

Group by **seed**, not by model: one notebook runs baseline, bottleneck and control at a
single seed.

```
notebook A   baseline seed 0   bottleneck seed 0   control seed 0
notebook B   baseline seed 1   bottleneck seed 1   control seed 1
notebook C   baseline seed 2   bottleneck seed 2   control seed 2
```

After notebook A you hold a complete three-model comparison — single-seed, so not
reportable, but complete. Grouping by model would leave you after notebook A with three
baselines and nothing to compare them to. If a session dies, by-seed loses less.

It also keeps each notebook to three runs, which matters because Kaggle caps a GPU
session at 12 hours and all nine runs together are close enough to that cap to be a
gamble.

**The three seeds are not a reason to look at a target domain three times.** The ledger
keys on the checkpoint hash, so three seeds give three legitimate entries per domain —
that is expected, not an override. Section 4 covers this.

---

## 2. Expected length

From the wiring runs: **142 s/epoch**, 30 epochs, so **~71 min per run** and **~3 h 35
min per notebook**, ~10.7 GPU-hours for all nine. Against a 30 h/week quota that is about
a third of a week's allowance, and each notebook sits well inside the 12 h session cap.

Two caveats on that number:

- It predates A1 (`2685c9c`) and A2 (`ea01e25`), which remove 579 device
  synchronisations per epoch and let the host-to-device copy overlap. Neither changes a
  number; both may lower the time.
- The Step 0 measurement (`M1`/`M2'`/`M3`/`M4`) has not been run, so it is not yet known
  whether the 142 s is the data pipeline or the GPU step under deterministic kernels. If
  it is the GPU step, 142 s is roughly the floor.

Time the first run before assuming the rest, and compare against `seconds_per_epoch` in
its results JSON.

---

## 3. The notebook, cell by cell

One notebook per seed. Set `SEED` in cell 2 and change nothing else between the three.

### Cell 1 — setup

Clone, install dependencies **without torch/torchvision**, assert CUDA. Verbatim from
[kaggle_runbook.md](kaggle_runbook.md) §§2–4. Ends with:

```python
import torch
assert torch.cuda.is_available(), "no GPU: set Accelerator before Save & Run All"
print(torch.cuda.get_device_name(0))
```

### Cell 2 — the one thing that differs between notebooks

```python
SEED = 0          # 0, 1 or 2 — the ONLY edit between the three notebooks
SUFFIX = "" if SEED == 0 else f"_seed{SEED}"
CONFIGS = [f"configs/{m}{SUFFIX}.yaml" for m in ("baseline", "bottleneck", "control")]
print(CONFIGS)
```

Seed 0 uses the original filenames, seeds 1 and 2 the `_seed<N>` ones, so this cell is
the same text in all three notebooks and only `SEED` changes.

### Cell 3 — the test suite

```bash
!cd /kaggle/working/<repo> && python -m pytest -q
```

Non-negotiable before a 3.5-hour booking. The numeric lock in `tests/test_bit_identity.py`
will skip on Kaggle — it is gated on the environment it was captured in, and that is
expected, not a failure.

### Cell 4 — record what is about to run

```bash
!cd /kaggle/working/<repo> && git rev-parse HEAD && git status --porcelain && wc -l target_eval_ledger.jsonl
```

Empty `git status` output and a 0-line ledger. If either is wrong, stop here.

### Cells 5, 6, 7 — one cell per run

One run per cell, never a loop over all three. A cell is the unit Kaggle shows you
timing and output for, and if the third run dies you want the first two's output
unambiguous.

```bash
!cd /kaggle/working/<repo> && python -m src.train --config configs/baseline.yaml
```

Each prints its final-epoch `val_macro_f1`, the selected epoch, per-epoch seconds and the
checkpoint path. Expect `selected_epoch` to equal `epochs_run - 1` — that is the
final-epoch rule. A run that fires the divergence guard raises `DivergenceError`,
writes **no** checkpoint, and leaves a results JSON with `failed: true`; the cell fails
and the next one still runs.

### Cell 8 — manifest

```python
import json, glob, hashlib
for f in sorted(glob.glob("/kaggle/working/results/*/*_seed*.json")):
    d = json.load(open(f))
    print(f"{d['run_id']:<12} seed {d['seed']}  "
          f"{d['val_macro_f1']:.4f} @ epoch {d['selected_epoch']}  "
          f"failed={d['failed']}  {sum(d['seconds_per_epoch'])/60:.1f} min")
for f in sorted(glob.glob("/kaggle/working/results/*/last.pt")):
    print(hashlib.sha256(open(f,'rb').read()).hexdigest()[:16], f)
```

The `last.pt` hashes are what the ledger will key on. Keep this cell's output: it is how
you check later that the checkpoint you evaluated is the one this notebook produced.

### Cell 9 — nothing

Do **not** run `src.eval` in this notebook, not even `--dry-run`. Training and the
one-time target look are separate bookings on purpose (section 4).

---

## 4. Bringing the results home

Kaggle does not persist `/kaggle/working` beyond the session, and the ledger lives at
`<repo>/target_eval_ledger.jsonl` — inside the clone, so inside the ephemeral working
directory. **Nothing is safe until it is on your machine and committed.**

### 4.1 Download

After each notebook commit, from the notebook's **Output** tab, download the zip. It
contains `results/<run_id>_seed<seed>/` with `last.pt`, `best.pt` and the results JSON
for each of the three runs. Roughly 200 MB per run of checkpoints, so ~600 MB per
notebook.

### 4.2 What goes into git and what does not

```
results/<run_id>_seed<seed>/<run_id>_seed<seed>.json    keep locally; results/ is gitignored
results/<run_id>_seed<seed>/last.pt                     keep locally; never committed
results/<run_id>_seed<seed>/best.pt                     keep locally; informational
target_eval_ledger.jsonl                                COMMITTED — see 4.3
```

`results/` is gitignored and checkpoints are never committed. Keep them: `last.pt` is
what section 5 evaluates, and without it you retrain.

Back up the nine `last.pt` files somewhere other than the laptop before section 5. They
are ~900 MB total and ~10.7 GPU-hours of work, and the protocol gives you one look at
Raabin with them.

### 4.3 The ledger

Training writes no ledger line — no training run touches a ledgered domain. So after the
three training notebooks the ledger is still 0 lines and there is nothing to bring back.

It starts mattering in section 5. The rule there is: **download it and append to your
local copy after every evaluation, then commit and push immediately.** It is append-only
and committed to git precisely so repeat looks are visible in history; a ledger that is
recreated each session constrains nothing.

```bash
# after an evaluation session, from the repo root on your machine
cat ~/Downloads/target_eval_ledger.jsonl >> target_eval_ledger.jsonl
sort -u target_eval_ledger.jsonl | wc -l     # sanity: no duplicate lines
git add target_eval_ledger.jsonl
git commit -m "ledger: <what was evaluated and why>"
git push
```

Append, never overwrite. The Kaggle copy starts from whatever was in the clone, so if you
pushed earlier entries first the downloaded file already contains them — check the line
count before and after rather than assuming.

---

## 5. The one-time Raabin evaluation

Separate booking, after all nine training runs are complete and their checkpoints are
backed up.

### 5.1 What one command does

`eval.domains` in all three configs is:

```yaml
domains: [pbc_test, raabin_train, raabin_testA, raabin_testB]
```

So **one** `src.eval` invocation on **one** checkpoint evaluates the held-out source test
split and all three Raabin domains, and writes **four** ledger lines. Nine checkpoints ×
four domains = **36 ledger lines**, and that is the whole target-domain budget for the
project.

`pbc_test` is ledgered too. It is held out for the source-domain report, so a repeat look
at it is the same error as a repeat look at Raabin.

### 5.2 The command

One cell per checkpoint, nine cells. No flags:

```bash
!cd /kaggle/working/<repo> && python -m src.eval \
    --checkpoint /kaggle/working/results/baseline_seed0/last.pt
```

`last.pt`, never `best.pt`. `best.pt` is the `pbc_val` peak the final-epoch rule
discarded; `src/eval.py` refuses it outright (`3b00627`) unless you pass
`--override-reason`, and there is no good reason to pass one here.

Each cell prints macro-F1 and ECE per domain per attribute mode, the class list each
macro-F1 was computed over, whether the model's buffers changed during evaluation (they
must not), and the path of the record it wrote. The abstention threshold comes from
`pbc_val` and is applied unchanged to every domain.

**Download the nine `*_eval.json` files along with the ledger.** They are the only copy
of the numbers; the ledger records only that the look happened.

### 5.3 Ledger rules for this section

- **Three seeds are three legitimate entries per domain**, because the ledger keys on the
  checkpoint hash and three seeds are three different checkpoints. No override needed.
- **A second evaluation of the same (checkpoint, domain) is refused.** That is the point.
  If a cell fails *after* writing its ledger lines, re-running it needs
  `--override-reason "<text>"`, and that reason goes into the ledger and into git history
  where a reader will see it.
- **Retraining produces a new hash**, so a re-run of a training job is a new checkpoint
  and a legitimate new entry — which is also why you should not retrain casually after
  looking at Raabin.
- **A legitimate override reason is a mechanical failure**, not a number you did not like.
  "Session died after raabin_testA, re-running for testB" is a reason. "Checking whether
  the gap looks better" is the thing the ledger exists to make visible.
- **Download and commit the ledger before closing the session** (4.3). If the session
  dies with lines written and not downloaded, the next run will refuse the pairs it
  already recorded and you will have no record of what they said.

### 5.4 Immediately after

```bash
# on your machine
sort -u target_eval_ledger.jsonl | wc -l     # expect 36 after a complete pass
python -c "
import json,collections
rows=[json.loads(l) for l in open('target_eval_ledger.jsonl')]
print(collections.Counter(r['domain'] for r in rows))
print('overrides:', [r.get('override_reason') for r in rows if r.get('override_reason')])
print('roles:', collections.Counter(r.get('selection_role') for r in rows))
"
git add target_eval_ledger.jsonl && git commit -m "ledger: Raabin evaluation, 9 checkpoints x 4 domains" && git push
```

Expect 9 per domain, no overrides, and `selection_role` `headline` on all 36. Anything
else needs explaining in the write-up.

### 5.5 Reminders

- Stop the GPU session when the last cell finishes.
- Raabin's basophils come from a single leukemia patient. Any basophil recall on a Raabin
  domain carries that caveat and is one patient's cells, not a sample of basophils
  (CLAUDE.md §6).
- `raabin_testB` holds two of the five classes, so it reports a 2-class macro-F1 and the
  source side of its gap is `pbc_test` restricted to those two classes. Per-class recall
  is `null`, not 0, for a class absent from an eval set.
- There is no pooled Raabin number. Three domains, three results.
