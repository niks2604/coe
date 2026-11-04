# wbc-attributes

Are hematologist-style morphological attributes a more domain-invariant representation
for white blood cell classification than raw CNN features?

The test: force predictions through an 11-attribute bottleneck (nucleus shape, chromatin
density, cytoplasm colour, granularity and seven more), train on one lab's data only, and
measure how much accuracy is lost when evaluating zero-shot on images from different labs
and cameras.

**The result this project reports is a generalization gap, not an accuracy number.** A
model scoring 99% on the source domain and 70% on target domains is a worse result here
than one scoring 94% and 89%. A negative result — the bottleneck does not help — is a
valid outcome, and the protocol is fixed in advance so that it cannot be tuned away.

The full protocol lives in [CLAUDE.md](CLAUDE.md), which is the authoritative document.
This README summarises it.

---

## Protocol summary

**Three models, one protocol.** All three are trained and evaluated on exactly the same
images with exactly the same settings, so any difference between them comes from the
representation and not from the training recipe.

| model | architecture | attribute loss |
|---|---|---|
| `baseline` | image → 5 classes | — |
| `bottleneck` | image → 11 attributes (31 values) → 5 classes | λ = 1.0 |
| `control` | identical to the bottleneck | λ = 0.0 |

The control is the reason the comparison means anything: it has the same narrow 31-d
architecture as the bottleneck with the attribute supervision switched off, so a
bottleneck gain cannot be credited to the shape of the network.

**Domains.** Source is PBC, restricted to the 10,298 images carrying WBCAtt labels, using
WBCAtt's own train/val/test CSVs as the splits. Target is Raabin-WBC, treated as three
separate domains (`raabin_train`, `raabin_testA`, `raabin_testB`) with no pooled headline
number, because TestB is a different camera and pooling would hide the effect being
measured. Label space is the 5-part differential: neutrophil, lymphocyte, monocyte,
eosinophil, basophil.

**No target-domain leakage.** Target data is evaluation-only: never for training,
validation, early stopping, model or checkpoint selection, hyperparameter tuning,
normalization statistics, abstention thresholds, or any data-driven decision. All tuning
happens on the PBC val split. This is enforced structurally — at the config layer, the
domain registry, the loader factories, the loss, the metrics and the ledger — not by
convention.

**Target evaluation ledger.** `target_eval_ledger.jsonl` is append-only and committed to
git, one line per (checkpoint sha256, domain). A second look at the same pair is refused
without an explicit, recorded reason. `pbc_test` is ledgered too, since it is held out for
the source-domain report. Repeated target evaluation is tuning on the target; the ledger
makes repeat looks visible in git history. Anyone reading a result should read the ledger.

**Metrics.** Macro-F1 and per-class recall per domain, plus the source→target drop.
Basophil and eosinophil recall always reported separately, since they are rare and are
where domain shift concentrates. Accuracy alone is never an acceptable summary. Also ECE,
mean maximum probability, and accuracy-vs-coverage under abstention. Macro-F1 is computed
only over the classes present in an eval set, and the class list is recorded beside the
number; for a source→target gap the source side is restricted to the classes present in
that target domain, so a 5-class figure is never subtracted from a 2-class one.

**Fixed a priori**, before any target number was seen: λ = 1.0, AdamW with cosine decay
after a short linear warmup, full fine-tune of the whole backbone, the augmentation and
colour-jitter caps, and ImageNet normalization constants. No hyperparameter search is
run: lr 3e-4, weight decay 1e-4 and class_weighting none are fixed a priori and identical
for all three models, so every model gets a search budget of zero. The evaluated
checkpoint is the final epoch of the declared budget, never a pbc_val peak. Every
reported number is the mean ± std over 3 seeds.

---

## The datasets are NOT included

No images are in this repository and none ever have been. `data/` is gitignored. You must
obtain the datasets yourself from their original sources, under their own licences and
terms:

- **PBC** — [Mendeley Data](https://data.mendeley.com/datasets/snkd93bnjr/1)
- **WBCAtt** — attribute annotations layered on the PBC images; see the paper below
- **Raabin-WBC** — [raabindata.com](https://raabindata.com/)

`scripts/extract_raw.py` produces the `data/raw/` layout the configs expect once the
archives are in place. `configs/audit.yaml` plus `python -m src.data.audit` runs a
read-only audit that reconciles what is on disk against the class mapping.

---

## How to run

**Requirements.** Python with PyTorch and torchvision. `requirements.txt` holds the pins
for everything else.

> Two known gaps: `requirements.txt` does not list `torchvision`, although
> `src/data/transforms.py` and `src/models/build.py` both import it; and it pins
> `torch==2.7.0`, which must **not** be installed on Kaggle, where it would replace the
> preinstalled CUDA build. See [docs/kaggle_runbook.md](docs/kaggle_runbook.md).

**Every experiment is one config file.** No CLI flags change behaviour in ways the config
does not record.

```bash
python -m src.train --config configs/<name>.yaml      # train
python -m src.eval  --checkpoint results/<run>/last.pt  # evaluate
```

Evaluation reads the resolved config embedded in the checkpoint, never a YAML passed
alongside it, so a checkpoint cannot be evaluated under preprocessing it was not trained
with. There are exactly two blessed flags, each of which records itself: `--dry-run`
(evaluates `pbc_val` only, writes no ledger entry) and `--override-reason "<text>"`
(re-evaluates a ledgered pair, reason mandatory and written to the ledger).

**Start with the smoke test**, which runs locally on CPU:

```bash
python -m src.train --config configs/smoke.yaml
python -m src.eval  --checkpoint results/smoke_seed0/last.pt --dry-run
```

Note that a smoke run is only 64 optimiser steps and can show that the pipeline runs and
nothing whatever about whether a model trains — see the step-budget rule in CLAUDE.md §5.

**The configs:**

| config | purpose | runs on |
|---|---|---|
| `smoke.yaml`, `smoke_baseline.yaml`, `smoke_control.yaml` | 500 images, 2 epochs, pipeline sanity check | local CPU |
| `wiring_baseline.yaml`, `wiring_bottleneck.yaml`, `wiring_control.yaml` | full source domain at the declared settings; confirms each model trains before the headline runs; selects nothing | Kaggle GPU |
| `baseline.yaml`, `bottleneck.yaml`, `control.yaml` | the headline runs | Kaggle GPU |
| `tuning_grid.yaml` | a grid that was declared and is not run; kept as a record | — |
| `audit.yaml` | read-only data audit | local |
| `class_mapping.yaml`, `label_space.yaml` | class name mapping; committed index orderings | — |

The wiring and headline configs declare `env: kaggle` and `device: cuda`. **They will not
load until `paths.kaggle` is filled in** — it is still `TODO_FILL`, and the config layer
refuses that sentinel rather than guessing. See
[docs/kaggle_runbook.md](docs/kaggle_runbook.md) for the GPU workflow.

**Tests:**

```bash
python -m pytest
```

Much of the suite is leakage tests: that `src/train.py` cannot even import a target
loader, that a dry run opens no file under the target root, that the committed ledger is
byte-identical before and after a test run, and that selection cannot read a monitoring
metric.

---

## Repository layout

```
configs/      one YAML per experiment; the class mapping and label space
src/          config layer, domain registry, data, models, losses, metrics, train, eval, ledger
scripts/      raw-archive extraction and two training diagnostics
tests/        the test suite
docs/         the Kaggle runbook
data/         gitignored; no images are ever committed
results/      gitignored, except the ledger at the repo root
```

---

## Citations

This work uses three datasets. If you use this code, cite the datasets it depends on.

**PBC** — the source domain.

> Acevedo, A., Merino, A., Alférez, S., Molina, Á., Boldú, L., & Rodellar, J. (2020).
> A dataset of microscopic peripheral blood cell images for development of automatic
> recognition systems. *Data in Brief*, 30, 105474.
> https://doi.org/10.1016/j.dib.2020.105474

**WBCAtt** — the 11 morphological attribute annotations, and the train/val/test splits.

> Tsutsui, S., Pang, W., & Wen, B. (2023). WBCAtt: A White Blood Cell Dataset Annotated
> with Detailed Morphological Attributes. *Advances in Neural Information Processing
> Systems 36 (NeurIPS 2023)*, Datasets and Benchmarks Track.

> **Acknowledgement requested by WBCAtt.** WBCAtt provides attribute annotations layered
> on the PBC images rather than new images, and asks that work using its annotations also
> cite the original PBC dataset. Both citations are given here for that reason, and any
> publication from this work must carry both.

**Raabin-WBC** — the target domain.

> Kouzehkanan, Z. M., Saghari, S., Tavakoli, S., Rostami, P., Abaszadeh, M., Mirzadeh, F.,
> Satlsar, E. S., Gheidishahran, M., Gorgi, F., Mohammadi, S., & Hosseini, R. (2022).
> A large dataset of white blood cells containing cell locations and types, along with
> segmented nuclei and cytoplasm. *Scientific Reports*, 12, 1123.
> https://doi.org/10.1038/s41598-021-04426-x

---

## Stated limitations

Carried here from CLAUDE.md §6 because they belong in any write-up:

- One target dataset, from a single lab network, so the shift measured is mostly camera
  and optics rather than stain protocol. The result does not generalize to stain-protocol
  shift. LISC was never obtained and is out of scope.
- No hyperparameter search was run, so the three models are compared at a single
  operating point chosen a priori rather than one shown to suit any of them.
- Nothing in the protocol rewards domain robustness. Selection is the final epoch of a
  fixed budget and nothing is searched, so no choice was made for its ability to
  generalize — deliberate, since the alternative is reading the target.
- BatchNorm running statistics are source statistics baked into the model and a known
  cause of domain-shift degradation. A GroupNorm arm is pre-declared as a secondary
  comparison, and it confounds two changes at once; see CLAUDE.md §6.
- Raabin's basophils come from a single leukemia patient, so any basophil recall on a
  Raabin domain is one patient's cells and is reported with that caveat.

---

## Licence

**Not yet licensed.** No LICENSE file has been added, so default copyright applies and no
permission for reuse is granted. The datasets are not covered by any licence here in any
case; they carry their own terms from their original sources.
