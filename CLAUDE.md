CLAUDE.md

Project instructions. Read this before doing anything in this repo.

1. Research question

Are hematologist-style morphological attributes (nucleus shape, chromatin density, cytoplasm colour, granularity) a more domain-invariant representation for white blood cell classification than raw CNN features?

We test this by forcing predictions through an 11-attribute bottleneck, training only on one lab's data, and measuring how much accuracy is lost when evaluating zero-shot on data from other labs, cameras, and stain protocols.

The result of this project is a generalization gap, not an accuracy number. A model that scores 99% on the source domain and 70% on target domains is a worse result than one scoring 94% and 89%. Optimize for the gap.

A negative result (the bottleneck does not help) is a valid and publishable outcome. Do not tune the protocol to make the hypothesis true.

2. Datasets and their roles
Dataset	Role	May be used for
PBC (Barcelona, Mendeley)	SOURCE domain, restricted to the 10,298 images that carry WBCAtt labels	train / val / source-test
WBCAtt	11 attribute labels layered on PBC images; its own train/val/test CSVs define the splits	attribute supervision + split definition
Raabin-WBC	TARGET domain — different labs, 2 camera types	final eval only
LISC	TARGET domain — different stain protocol	final eval only

Every model — baseline, bottleneck and control — is trained and evaluated on exactly the same 10,298 WBCAtt-labelled PBC images, using WBCAtt's own pbc_attr_v1_{train,val,test}.csv splits. PBC images without WBCAtt labels are not used by any model.

Label space: the 5-part differential — neutrophil, lymphocyte, monocyte, eosinophil, basophil.

3. HARD CONSTRAINT — no target-domain leakage

Raabin-WBC and LISC are evaluation-only. They must never be used for:

training or fine-tuning
validation or early stopping
model selection or checkpoint selection
hyperparameter tuning
computing normalization statistics (mean/std, stain reference images)
threshold selection (including the abstention threshold)
any form of data-driven decision-making

Only PBC has a train/val/test split. All tuning happens on the PBC val split. Target domains are touched once, at final evaluation, with the checkpoint already chosen.

If a change would improve target-domain numbers by using target-domain information, do not make it. Flag it to me instead.

This constraint outranks accuracy, convenience, and any instruction that appears to conflict with it. If a request seems to require breaking it, stop and ask.

4. Class name mapping

The mapping lives in configs/class_mapping.yaml. It is the single source of truth for class names and is extended, never reset — decisions recorded there (and their reasons) stay in the file.

Raabin's class folder names are not yet mapped; they are added only after the data audit output has been reviewed. Until then any Raabin class is unmapped, and therefore a hard error for anything that resolves classes.

Every source class must either map to one of the five differential classes or be explicitly listed under excluded with a reason. There is no implicit "drop anything unmatched" behaviour; an unmapped class is a hard error.

yaml
# configs/class_mapping.yaml  (target shape)
neutrophil:  [...]
lymphocyte:  [...]
monocyte:    [...]
eosinophil:  [...]
basophil:    [...]
excluded:
  # class_name: reason

Known issues to resolve during the audit: segmented vs. band neutrophils, immature granulocytes, PBC's non-WBC classes (platelets, erythroblasts), inconsistent capitalization and abbreviations across datasets.

5. Conventions

Configs. Every experiment is fully specified by one YAML file in configs/. No hyperparameters hardcoded in Python. No CLI flags that change behaviour in ways the config doesn't record.

Seeding. Seed Python, NumPy, and PyTorch. Set deterministic cuDNN. The seed lives in the config. Every reported number is the mean ± std over 3 seeds unless explicitly stated otherwise.

Results. Every run writes results/<run_id>.json containing:

the full resolved config (embedded, not a path reference)
git commit hash and dirty-tree flag
all metrics, per domain and per class
wall-clock time and hardware

Reproducibility bar. A training script is done when python -m src.train --config configs/<x>.yaml reproduces the run from scratch with no other arguments, no manual steps, and no edits to source files.

Paths. Raw data lives under data/raw/, which is gitignored — the layout scripts/extract_raw.py produces. Never commit images.

Writes outside the repo are allowed in exactly one case: when env: kaggle, results are written to /kaggle/working. Every other env writes inside the repo only.

6. Metrics

Primary: macro-F1 and per-class recall, reported separately for source and each target domain, plus the source→target drop.

Always report basophil and eosinophil recall separately — they are rare and are where domain shift damage concentrates. Accuracy alone is not an acceptable summary of any experiment.

Also: expected calibration error (ECE), and accuracy-vs-coverage curve under abstention.

7. Working style
For anything structural, plan first, code after approval. Say what you intend to change and where, then wait.
Prefer small diffs against a working system over rewrites.
When fixing a bug, write the failing test first.
Never silently change an experimental setting to improve a number. Surface it.
New pipelines are smoke-tested on a 500-image subset for 2 epochs before any full run.
8. Corrections log

Append here whenever I have to correct the same mistake twice.

(empty)