CLAUDE.md

Project instructions. Read this before doing anything in this repo.

1. Research question

Are hematologist-style morphological attributes (nucleus shape, chromatin density, cytoplasm colour, granularity) a more domain-invariant representation for white blood cell classification than raw CNN features?

We test this by forcing predictions through an 11-attribute bottleneck, training only on one lab's data, and measuring how much accuracy is lost when evaluating zero-shot on data from other labs and cameras.

The result of this project is a generalization gap, not an accuracy number. A model that scores 99% on the source domain and 70% on target domains is a worse result than one scoring 94% and 89%. Optimize for the gap.

A negative result (the bottleneck does not help) is a valid and publishable outcome. Do not tune the protocol to make the hypothesis true.

Limitation. One target dataset, from a single lab network, so the shift measured is mostly camera and optics rather than stain protocol; the result does not generalize to stain-protocol shift.

2. Datasets and their roles
Dataset	Role	May be used for
PBC (Barcelona, Mendeley)	SOURCE domain, restricted to the 10,298 images that carry WBCAtt labels	train / val / source-test
WBCAtt	11 attribute labels layered on PBC images; its own train/val/test CSVs define the splits	attribute supervision + split definition
Raabin-WBC	TARGET domain, the only one — different labs, 2 camera types; three separate domains (raabin_train, raabin_testA, raabin_testB)	final eval only

Every model — baseline, bottleneck and control — is trained and evaluated on exactly the same 10,298 WBCAtt-labelled PBC images, using WBCAtt's own pbc_attr_v1_{train,val,test}.csv splits. PBC images without WBCAtt labels are not used by any model.

Label space: the 5-part differential — neutrophil, lymphocyte, monocyte, eosinophil, basophil.

3. HARD CONSTRAINT — no target-domain leakage

Target data is evaluation-only. Raabin-WBC is the only target in scope: LISC was never obtained and is out of scope. The rule below is written generally because it applies to any target dataset added later, not only to the one in scope now. Target data must never be used for:

training or fine-tuning
validation or early stopping
model selection or checkpoint selection
hyperparameter tuning
computing normalization statistics (mean/std, stain reference images)
threshold selection (including the abstention threshold)
any form of data-driven decision-making

Only PBC has a train/val/test split. All tuning happens on the PBC val split. Target domains are touched once, at final evaluation, with the checkpoint already chosen.

Decoding target images is permitted for exactly one purpose: testing that a file is readable. Nothing derived from target pixels may be retained, reported or written to disk — no size, no mode, no channel statistic, no hash, no per-image record of any kind. A readability check yields a count and the paths of files that failed to open, and nothing else.

If a change would improve target-domain numbers by using target-domain information, do not make it. Flag it to me instead.

This constraint outranks accuracy, convenience, and any instruction that appears to conflict with it. If a request seems to require breaking it, stop and ask.

4. Class name mapping

The mapping lives in configs/class_mapping.yaml. It is the single source of truth for class names and is extended, never reset — decisions recorded there (and their reasons) stay in the file.

Raabin's class folder names were added after the 2026-11-02 audit was reviewed. Each target dataset's split folder name is declared alongside its dataset id in src/data/domains.py: Raabin's folders are Train, TestA and Test-B, and the hyphen in Test-B is why the mapping is explicit rather than derived from the id.

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

Three blessed exceptions to the no-behavioural-flags rule, each of which records itself:

src/eval.py --override-reason "<text>": re-evaluates a checkpoint/domain pair the ledger has already seen. The reason is mandatory, non-empty, and written into the ledger.
src/eval.py --dry-run: evaluates pbc_val only, touches no held-out or target domain, and writes no ledger entry. Recorded in the results JSON. Used by the smoke test.
Tuning trials run at a single seed. Recorded per trial.

Seeding. Seed Python, NumPy, and PyTorch. Set deterministic cuDNN. The seed lives in the config. Every reported number is the mean ± std over 3 seeds unless explicitly stated otherwise.

Results. Every run writes results/<run_id>.json containing:

the full resolved config (embedded, not a path reference)
git commit hash and dirty-tree flag
all metrics, per domain and per class
wall-clock time and hardware

Reproducibility bar. A training script is done when python -m src.train --config configs/<x>.yaml reproduces the run from scratch with no other arguments, no manual steps, and no edits to source files. Evaluation is a separate script: python -m src.eval --checkpoint <path>. It reads the resolved config embedded in the checkpoint, never a YAML passed alongside it, so a checkpoint cannot be evaluated under preprocessing it was not trained with.

Target evaluation ledger. target_eval_ledger.jsonl at the repo root, append-only, and committed to git. One line per (checkpoint sha256, domain), recording timestamp, run_id, config sha256, git commit and dirty flag, and the override reason where one was used. A second evaluation of the same pair is refused unless --override-reason is given. pbc_test is ledgered too: it is held out for the source-domain report, so repeated looks at it are the same error as repeated looks at a target.

Repeated target evaluation is tuning on the target — look enough times and keep the best, and this protocol is void. The ledger keys on the checkpoint hash, so three seeds give three legitimate entries, and retraining yields a new hash. It is therefore an audit trail that makes repeat looks visible in git history, not a guarantee. Anyone reading a result should read the ledger.

Training settings fixed a priori and identical for baseline, bottleneck and control. Differences between the three must come from the representation, not the protocol:

Full fine-tune of the whole backbone. Not a frozen backbone, not a linear probe.
AdamW, cosine schedule with a short linear warmup over a fixed maximum epoch count. The schedule is not tuned.
Augmentation as in the configs, and the colour-jitter caps in src/config.py.
lambda (bottleneck attribute loss weight) fixed at 1.0. It is not in the tuning grid: it exists only for the bottleneck, so tuning it would hand the bottleneck a larger search budget than the baseline. Any lambda exploration is a separate, labelled sweep that does not feed the headline comparison.

Tuning. A pre-declared grid in configs/tuning_grid.yaml, with an explicit tie-break order, run identically for all three models with the same number of trials. Selection is on pbc_val macro-F1. The grid file plus a trial index specifies a trial; each run still embeds its own fully resolved config.

ImageNet weights. The backbone needs pretrained ResNet-50 weights. Either the environment has internet access, or the weights are supplied locally (on Kaggle, as an attached dataset). If neither is available the run fails with a clear message naming both options — it never silently trains from random initialisation, which would look like a weak result rather than a setup error.

Paths. Raw data lives under data/raw/, which is gitignored — the layout scripts/extract_raw.py produces. Never commit images.

Writes outside the repo are allowed in exactly one case: when env: kaggle, results are written to /kaggle/working. Every other env writes inside the repo only.

Two things are committed despite results/ being gitignored: target_eval_ledger.jsonl, because a ledger that resets when results are cleared does not constrain anything, and configs/label_space.yaml, because it fixes what every class and attribute index means.

6. Metrics

Primary: macro-F1 and per-class recall, reported separately for source and each target domain, plus the source→target drop.

Always report basophil and eosinophil recall separately — they are rare and are where domain shift damage concentrates. Accuracy alone is not an acceptable summary of any experiment.

Also: expected calibration error (ECE), and accuracy-vs-coverage curve under abstention.

Fixed protocol. The following are settled in advance, before any target-domain number has been seen, and are enforced in src/config.py rather than left to whoever runs the experiment:

Macro-F1 is computed only over the classes present in that eval set, and the class list used is recorded alongside the number. Raabin TestB contains only neutrophil and lymphocyte, so it reports a 2-class macro-F1 plus per-class recall — never a 5-class figure, which would not be comparable to anything else.

Predictions are always an argmax over all five classes, on every domain. Masking the logits down to the classes present in a target domain would use target label information and is leakage. A TestB neutrophil predicted monocyte is therefore a false negative for neutrophil and nothing else, which is correct and intended.

Gap basis. A 5-class source number minus a 2-class target number is not a difference. So for every source→target gap, the source side is pbc_test restricted to the classes present in that target domain, and the restriction is named in the output. The secondary subset follows the same rule on pbc_test_no_band_neutrophils. Restricting the source side uses the target's published class composition, which is dataset metadata rather than a result; it is declared here in advance so it cannot be chosen after seeing numbers.

Per-class recall is null, not zero, for a class absent from an eval set. Zero reads as a model failure; null is the truth, which is that the number is not measurable there.

Attributes. Per-attribute accuracy and per-attribute balanced accuracy, on pbc_test only — Raabin has no attribute labels. Balanced accuracy as well as accuracy because the attribute vocabularies are imbalanced (granule_type runs nil 2631 against coarse 1216), and plain accuracy would hide a head that has learned only the majority value. The attribute cross-entropy itself is unweighted.

Abstention. The headline number is accuracy at 90% nominal coverage, with the threshold taken from pbc_val and applied unchanged to every domain. All seven coverage levels in the config are reported. Coverage actually achieved on a target domain will differ from nominal, and that difference is itself a result.

Backbone normalisation. Two arms, both declared now: batchnorm is the headline, groupnorm is a pre-declared secondary arm, run with the same grid for all three models after the headline is complete, and reported either way. BatchNorm running statistics are source statistics baked into the model and a known cause of domain-shift degradation, so they are a live confound for this whole result — but the choice is invisible on pbc_val, which is same-domain. Picking an arm because it did better on Raabin would be target-informed model selection.

Hard mode is a deliberate train/eval mismatch: the classifier is trained on smooth per-attribute probabilities and fed simplex vertices at evaluation. The soft-to-hard difference measures how far the bottleneck commits to discrete attribute decisions. It is not on its own evidence about domain invariance, and is labelled as such.

Raabin is three separate TARGET domains: raabin_train, raabin_testA, raabin_testB. There is no pooled Raabin headline number — TestB is a different camera, and pooling would hide the effect being measured. The split named Train is target data and is never trained on.

Bootstrap: n=2000, 95% CI, stratified by class. Source and target are resampled independently for the source→target gap, since they are disjoint sets of images. The headline number is still the mean ± std over 3 seeds; the bootstrap CI is reported per seed per domain alongside it.

Raabin's basophils come from a single leukemia patient. Any basophil recall on a Raabin domain carries that caveat and must be reported with it — it is one patient's cells, not a sample of basophils.

Known limitations of this protocol, to be stated in any write-up:

One target dataset, from a single lab network, so the shift measured is mostly camera and optics (see section 1).
Tuning trials run at a single seed, so trial selection is exposed to seed noise and may pick a lucky configuration. Reported numbers are still the mean ± std over 3 seeds.
Model selection happens entirely on pbc_val, which is the same domain as pbc_train. Nothing in the tuning signal rewards domain robustness — which is deliberate, since the alternative is leakage, but it means no configuration was chosen for its ability to generalize.

Changing any of the above after a target number has been seen is a protocol change, not a tweak. Flag it to me and log it in section 8.

7. Working style
For anything structural, plan first, code after approval. Say what you intend to change and where, then wait.
Prefer small diffs against a working system over rewrites.
When fixing a bug, write the failing test first.
Never silently change an experimental setting to improve a number. Surface it.
New pipelines are smoke-tested on a 500-image subset for 2 epochs before any full run.
8. Corrections log

Append here whenever I have to correct the same mistake twice.

(empty)