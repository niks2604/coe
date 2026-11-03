# Kaggle runbook

How to run this project on a Kaggle GPU session. Written for whoever is driving the
session, step by step, with the reasons where a step is easy to get wrong.

Kaggle is used because the source domain is ~10k images and a full run is 30 epochs
x 193 steps per model. The same work on an 8-core CPU measured ~30-33 min per epoch,
or 15-16 h per model.

---

## 0. Two things must be settled before the first run

Neither is done yet. The runbook will not work until both are.

1. **`paths.kaggle` is still `TODO_FILL`** in every experiment config, and
   `src/config.py` rejects that sentinel loudly on purpose. It must be filled with the
   real `/kaggle/input/...` paths once the dataset tree is known.
2. **The configs say `env: local`.** A Kaggle run needs `env: kaggle`, which is what
   selects the `paths.kaggle` block and permits writing to `/kaggle/working`
   (CLAUDE.md sec 5 allows writes outside the repo in that one case).

Do not work around either by editing paths inline in a notebook cell. A run whose
paths are not in the committed config is not reproducible, which is the one thing the
reproducibility bar in CLAUDE.md sec 5 asks for.

---

## 1. Pre-run checklist

Tick all of these before starting a run that costs GPU hours.

- [ ] Notebook **Accelerator** is set to a GPU (T4 or P100), not None and not TPU.
- [ ] **Internet** is on, if the ImageNet weights are to be downloaded (see sec 5).
- [ ] The PBC + WBCAtt dataset is **attached** to the notebook.
- [ ] `paths.kaggle` is filled in and `env: kaggle` is set, and both are **committed**.
- [ ] `torch.cuda.is_available()` returned `True` (sec 4). Nothing runs before this.
- [ ] The test suite passed (sec 6).
- [ ] You know where the ledger file is and that you must download it afterwards (sec 9).

---

## 2. Clone into /kaggle/working

`/kaggle/working` is the only writable place that persists for the life of the session.

```bash
%cd /kaggle/working
!git clone https://github.com/niks2604/coe.git
%cd /kaggle/working/coe
!git log --oneline -3
```

The repo root is now `/kaggle/working/coe`. That matters for the ledger: `src/ledger.py`
puts `target_eval_ledger.jsonl` at the repo root, so on Kaggle it lives inside this
clone and **disappears when the session ends**.

---

## 3. Dependencies, WITHOUT torch or torchvision

Kaggle ships a torch built against its own CUDA driver. Installing the pinned
`torch==2.7.0` from `requirements.txt` replaces that build with a generic one and
`torch.cuda.is_available()` starts returning `False`. Never install torch here.

```bash
# Filter torch and torchvision out of the pinned requirements.
!grep -vE '^(torch|torchvision)[=<>~!]' requirements.txt > /tmp/reqs_kaggle.txt
!cat /tmp/reqs_kaggle.txt
```

Then install **only what is actually missing**, rather than applying every pin:

```bash
!pip install -q gdown
```

> **Why not `pip install -r /tmp/reqs_kaggle.txt`?** Those pins include
> `numpy==2.4.6` and `pandas==3.0.5`. Forcing a different numpy under Kaggle's
> preinstalled torch is the second most common way to break the CUDA build, after
> installing torch itself. Install the minimum, then let the import check below tell you
> if anything is genuinely absent.

Import check — this is the real test of whether dependencies are satisfied:

```python
for module in ("torch", "torchvision", "numpy", "pandas", "yaml", "PIL", "pytest"):
    __import__(module)
    print("ok", module)
```

> Note: `requirements.txt` does not list `torchvision`, although `src/data/transforms.py`
> and `src/models/build.py` both import it. Kaggle preinstalls it so this does not bite
> here, but a fresh clone elsewhere will fail until that is fixed.

---

## 4. Confirm CUDA before anything else

The wiring configs declare `device: cuda`, and `src/config.py` raises rather than
falling back to CPU if it is unavailable — a results file claiming a GPU it did not use
would be worse than a run that refuses to start. Check explicitly anyway, so the failure
comes from a one-line cell and not forty minutes into a run:

```python
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
assert torch.cuda.is_available(), "stop here: fix the accelerator or the torch install"
```

If this prints `False` after you installed something, you replaced the CUDA build.
Factory-reset the session and redo section 3 without installing torch.

---

## 5. ImageNet weights

The backbone needs pretrained ResNet-50 weights, and CLAUDE.md sec 5 forbids silently
falling back to random initialisation — that would look like a weak result rather than a
setup error. Two supported options:

**A. Internet on (default).** `model.imagenet_weights: null` downloads them. Nothing to do.

**B. Internet off.** Attach the weights as a Kaggle dataset and point the config at the
file:

```yaml
model:
  imagenet_weights: /kaggle/input/<your-weights-dataset>/resnet50-imagenet1k-v2.pth
```

If neither is available the run fails with a message naming both options. That is
intended; do not patch around it.

---

## 6. Run the test suite

Cheap, and it catches a broken environment before any GPU time is spent.

```bash
%cd /kaggle/working/coe
!python -m pytest -q
```

Expect **412 passed**. One test adapts to the hardware: the unavailable-device test
picks whichever of cuda/mps is missing, so on Kaggle it exercises the mps path.

---

## 7. Run the three wiring checks

Source domain only. These select nothing — they confirm each model trains at all under
the declared settings before the grid runs. See CLAUDE.md sec 5 for the decision rule.

```bash
%cd /kaggle/working/coe
!python -m src.train --config configs/wiring_baseline.yaml
!python -m src.train --config configs/wiring_bottleneck.yaml
!python -m src.train --config configs/wiring_control.yaml
```

One config per command and no other arguments — that is the reproducibility bar.

**Watch the first epoch before you walk away.** Each run prints seconds per epoch. Multiply
by 30 and check the total fits the session limit and your remaining GPU quota before
leaving it unattended.

**Read the step budget, not just the accuracy.** 30 epochs x 193 steps = 5,790 steps. A
bottleneck judged on fewer than ~1,000 steps is being judged inside the window where even
a best-case linear classifier on ground-truth attribute codes sits at 0.177 train accuracy
(CLAUDE.md sec 5, step budget). If early stopping fires, the results JSON records
`early_stopped` and `stopped_after_epoch` — report the stop epoch rather than implying the
full budget was used.

---

## 8. Where outputs land

With `env: kaggle`, `paths.kaggle.results_dir` is `/kaggle/working/results`:

```
/kaggle/working/results/<run_id>_seed<seed>/best.pt                  checkpoint
/kaggle/working/results/<run_id>_seed<seed>/<run_id>_seed<seed>.json results record
/kaggle/working/coe/target_eval_ledger.jsonl                         the ledger
```

The results JSON embeds the fully resolved config, the git commit and dirty flag, the
resolved device, per-epoch history with both loss terms separated, and wall-clock time.

Evaluation is a separate script and reads the config embedded in the checkpoint, never a
YAML passed alongside it:

```bash
!python -m src.eval --checkpoint /kaggle/working/results/wiring_bottleneck_seed0/best.pt --dry-run
```

`--dry-run` touches `pbc_val` only and writes no ledger entry. Drop it only when you
intend to spend a real look at a held-out or target domain.

---

## 9. After EVERY evaluation: download and push the ledger

**This is the step that cannot be skipped.** A Kaggle session does not persist. The
ledger lives in the clone at `/kaggle/working/coe/target_eval_ledger.jsonl`, and when the
session ends it is gone. A ledger that resets is a ledger that constrains nothing, and
repeated target evaluation without a visible record *is* tuning on the target — at which
point the protocol is void (CLAUDE.md sec 5).

So, after every evaluation, before stopping the session:

1. Download both of these from the Kaggle file browser (or the notebook Output tab):
   - `/kaggle/working/coe/target_eval_ledger.jsonl`
   - `/kaggle/working/results/` (at least the `*.json` records; the `.pt` files are large
     and optional unless you need the exact checkpoint)
2. On your own machine, append the new ledger lines to the repo's
   `target_eval_ledger.jsonl`, then commit and push it. The ledger is append-only: add
   lines, never replace the file with the Kaggle copy if your local one has entries the
   Kaggle one does not.
3. Keep `results/` out of git. It is gitignored for a reason; only the ledger is committed.

A zip for convenience:

```bash
!cd /kaggle/working && zip -qr /kaggle/working/run_artifacts.zip results coe/target_eval_ledger.jsonl && ls -lh /kaggle/working/run_artifacts.zip
```

---

## 10. Determinism: CUBLAS_WORKSPACE_CONFIG

`src/config.py` already sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` via `os.environ.setdefault`
before enabling deterministic algorithms. That runs inside `set_seed`, which can be after
cuBLAS has initialised in a long-lived notebook kernel. If you see a deterministic-algorithm
error mentioning cuBLAS, set it in the environment **before torch is imported**:

```python
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # must precede `import torch`
```

Or run the scripts with `!` as shown above, where each process starts fresh and the
in-code default applies cleanly. The seeding record in the results JSON reports whether
deterministic algorithms were actually enabled — check it rather than assuming.

---

## 11. Stop the GPU

**Stop the session when the run finishes.** Kaggle GPU quota is per-week and a notebook
left idle with an accelerator attached keeps consuming it. Use *Run > Stop session*; closing
the browser tab does not stop it.

Before stopping, confirm you have completed section 9. Once the session is gone the ledger
is unrecoverable.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `torch.cuda.is_available()` is `False` | Kaggle's CUDA torch was replaced | Factory-reset the session; redo sec 3 without installing torch |
| `ConfigError: ... TODO_FILL` | `paths.kaggle` not filled in | Fill it and commit (sec 0) |
| `ConfigError: device: cuda was requested but ... is False` | No accelerator attached | Set the accelerator, restart |
| `ConfigError: device: mps is for smoke and debug configs only` | An `mps` device on a wiring/tuning/final config | Use `cuda` on Kaggle |
| `WeightsUnavailableError` | No internet and no local weights | Attach weights and set `imagenet_weights` (sec 5) |
| `LedgerError: ... already evaluated` | This checkpoint/domain pair was looked at before | Intended. Do not add `--override-reason` unless the repeat look is deliberate and justified |
| Results written somewhere unexpected | `env` is still `local` | Set `env: kaggle` (sec 0) |
