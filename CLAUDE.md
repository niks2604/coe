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

Scoped amendment, 2026-11-19, by the author. One exception to the paragraph above, and no other.

For the nine fixed headline checkpoints of the completed Raabin evaluation — and for no other model, arm, seed or future run — these may be retained for post-hoc analysis on pbc_test, raabin_train, raabin_testA and raabin_testB: per-image class logits, per-image attribute probabilities, per-image image ids, and the true class label.

Everything else in the paragraph above stands. Target pixels may not be retained. No pixel-derived statistic may be retained, reported or written to disk — no size, no mode, no channel statistic, no hash of an image file, nothing computed from the pixels themselves.

Nothing produced under this exception may be used to train, select, tune or design any arm. It may not feed a checkpoint choice, a hyperparameter, a threshold, a normalization statistic, an architecture decision or a new arm, and it may not revise the section 6 verdict, which was fixed on the records in eval_records/ before this exception existed.

Dumps made under it are written to analysis/ and recorded in analysis_ledger.jsonl, which is separate from target_eval_ledger.jsonl and confers no permission to evaluate anything. The exception is dated after the Raabin results were seen on 2026-11-13; it is logged in section 8, and any write-up must state that these analyses are post-hoc.

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

Two blessed exceptions to the no-behavioural-flags rule, each of which records itself:

src/eval.py --override-reason "<text>": permits one of two things the protocol otherwise refuses — re-evaluating a checkpoint/domain pair the ledger has already seen, or evaluating a checkpoint that is not the headline one. The reason is mandatory, non-empty, and written into the ledger. With neither condition present it is itself an error, so the flag cannot be left on out of habit.
src/eval.py --dry-run: evaluates pbc_val only, touches no held-out or target domain, and writes no ledger entry. Recorded in the results JSON. Used by the smoke test.

Seeding. Seed Python, NumPy, and PyTorch. Set deterministic cuDNN. The seed lives in the config. Every reported number is the mean ± std over 3 seeds unless explicitly stated otherwise.

Results. Every run writes results/<run_id>.json containing:

the full resolved config (embedded, not a path reference)
git commit hash and dirty-tree flag
all metrics, per domain and per class
wall-clock time and hardware

Reproducibility bar. A training script is done when python -m src.train --config configs/<x>.yaml reproduces the run from scratch with no other arguments, no manual steps, and no edits to source files. Evaluation is a separate script: python -m src.eval --checkpoint <path>. It reads the resolved config embedded in the checkpoint, never a YAML passed alongside it, so a checkpoint cannot be evaluated under preprocessing it was not trained with.

Target evaluation ledger. target_eval_ledger.jsonl at the repo root, append-only, and committed to git. One line per (checkpoint sha256, domain), recording timestamp, run_id, config sha256, git commit and dirty flag, the selection role and rule of the checkpoint that was read, and the override reason where one was used. src/eval.py refuses any checkpoint whose recorded role is not headline — best.pt, or anything saved before the role was recorded — unless an override reason is given. A second evaluation of the same pair is refused unless --override-reason is given. pbc_test is ledgered too: it is held out for the source-domain report, so repeated looks at it are the same error as repeated looks at a target.

Repeated target evaluation is tuning on the target — look enough times and keep the best, and this protocol is void. The ledger keys on the checkpoint hash, so three seeds give three legitimate entries, and retraining yields a new hash. It is therefore an audit trail that makes repeat looks visible in git history, not a guarantee. Anyone reading a result should read the ledger.

Training settings fixed a priori and identical for baseline, bottleneck and control. Differences between the three must come from the representation, not the protocol:

Full fine-tune of the whole backbone. Not a frozen backbone, not a linear probe.
AdamW, cosine schedule with a short linear warmup over a fixed maximum epoch count. The schedule is not tuned.
Augmentation as in the configs, and the colour-jitter caps in src/config.py.
lambda (bottleneck attribute loss weight) fixed at 1.0. It exists only for the bottleneck, so searching it would hand the bottleneck a search budget the baseline cannot have. Nothing is searched, so this follows from the same rule as every other hyperparameter. Any lambda exploration is a separate, labelled sweep that does not feed the headline comparison.

Wiring check. Before the headline runs, each model is trained once on the full source domain at the declared configuration — lr 3e-4, weight decay 1e-4, class_weighting none — for the declared maximum epoch count. It is source-only, it selects nothing, and its run_id is prefixed wiring_ so its results can never be mistaken for a headline seed. The configs are configs/wiring_{baseline,bottleneck,control}.yaml, which differ from the headline configs in run_id and device alone, and a test enforces that.

Decision rule. Pass for the bottleneck at lambda 1.0: pbc_val macro-F1 within 0.05 of the baseline, and per-attribute pbc_val accuracy above the majority-class rate on at least 9 of 11 attributes. Pass means the headline runs proceed with lambda unchanged. On a fail: look for bugs first; if none is found, lambda becomes a supervisor-level protocol decision, logged in section 8 before any further run. A bottleneck that fails to train is a training failure, not a negative result — reporting it as evidence against the hypothesis would be wrong.

Step budget. Any claim that a model does not train must state how many optimiser steps it had. Measured, on ground-truth attribute codes with everything but Linear(31,5) removed (scripts/classifier_capacity.py): 0.177 train accuracy at step 200, 0.9639 at 1,000, and 0.9895 at the declared 5,790. The classifier needs roughly a thousand steps before it is doing anything at all. The section 7 smoke test — 500 images, batch 16, 2 epochs — is 64 steps, so it can show that a pipeline runs and nothing whatever about trainability.

Device. Every experiment config states device: cpu | cuda | mps. A requested device that is not available is a hard error, never a silent fall back to CPU, and the device actually resolved is recorded in the results JSON. Validation of the name is hardware-independent so a checkpoint trained on one device stays readable on another. MPS is for smoke and debug configs only: it has no deterministic-algorithm guarantee and its cuDNN flags are no-ops, so it cannot meet the seeding requirement above. src/config.py rejects it for any run_id not prefixed smoke or debug, which covers every wiring and headline run.

Checkpoint selection. The checkpoint is the final epoch of the declared budget. There is no selection among epochs: every run trains for train.epochs and the resulting model is the one evaluated, for the baseline, the bottleneck, the control and the groupnorm arm alike.

The reason is that pbc_val cannot support an argmax over epochs. It is 1,030 images with lymphocyte at 111 and basophil at 125, so one image moves a per-class recall by 0.009 and macro-F1 by about 0.002, while an argmax over 30 epochs is biased upward by construction — it reports the maximum of thirty noisy draws. Worse, the models saturate at different epochs, so a per-model argmax selected the baseline at epoch 0, still inside the linear warmup at lr 1e-4 after 193 steps, against the bottleneck at epoch 11. A gap measured between a model that has barely started and one that has trained twelve epochs is not a comparison of representations, and this section requires that differences between the three come from the representation and not the protocol. The bias also ran in the dangerous direction: under-training the baseline flatters the bottleneck, which is the hypothesis this project is supposed to be able to reject.

The cost is accepted knowingly. A genuinely better intermediate checkpoint is discarded when one exists, and every run pays the full epoch budget. Equal training length for all three models is worth both.

best.pt is still written alongside last.pt, and the per-epoch history is still recorded, so the discarded alternative stays visible. best.pt is informational and is never the evaluated checkpoint; each payload records which rule produced it, so a results JSON and a ledger line always name the file that was actually read.

Divergence guard. Patience-based early stopping is removed. What remains fires only on a non-finite training loss, and when it fires the run is marked failed and yields no headline checkpoint. That keeps the guard outside selection: a guard that stopped on a falling validation metric would be epoch selection under another name.

Recorded in the config. The rule is a field in every experiment YAML, not a module constant in src/train.py, because no behaviour may depend on something the config does not record.

yaml
selection:
  rule: final_epoch          # the evaluated checkpoint; not a tuned axis
  report_metric: val_macro_f1   # recorded per epoch; selects nothing
  divergence_guard: non_finite_loss

Hyperparameters. No search is run. lr 3e-4, weight decay 1e-4 and class_weighting none are fixed a priori, identically for all three models and both normalisation arms, and are recorded in each config. The equal-search-budget requirement that a grid was meant to satisfy is satisfied trivially: every model gets a budget of zero. configs/tuning_grid.yaml and src/tuning.py are retained unused, as a record of what was declared and not run.

The wiring runs happened to use these same values, because they were declared as the mid-grid point of the grid that was never run. They are not counted as headline seeds: they ran under the old selection rule, so seed 0 is rerun under the final-epoch rule like every other seed.

ImageNet weights. The backbone needs pretrained ResNet-50 weights. Either the environment has internet access, or the weights are supplied locally (on Kaggle, as an attached dataset). If neither is available the run fails with a clear message naming both options — it never silently trains from random initialisation, which would look like a weak result rather than a setup error.

Paths. Raw data lives under data/raw/, which is gitignored — the layout scripts/extract_raw.py produces. Never commit images.

Writes outside the repo are allowed in exactly one case: when env: kaggle, results are written to /kaggle/working. Every other env writes inside the repo only.

Three things are committed despite results/ being gitignored: target_eval_ledger.jsonl, because a ledger that resets when results are cleared does not constrain anything, configs/label_space.yaml, because it fixes what every class and attribute index means, and eval_records/, which holds the per-evaluation *_eval.json records, because they are the only copy of the numbers — the ledger records that a look happened, not what it said.

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

Monitoring exception. Per-attribute accuracy is also computed on pbc_val once per epoch, for monitoring only. It is what makes "the attribute heads fit perfectly and the class head is stuck" visible while a run is in progress instead of after it. It is recorded under a monitor_ prefixed key, it is never reported as a result, and nothing may select on it. Under the final-epoch rule no epoch is selected at all, so this applies to trial selection, which reads pbc_val macro-F1 and nothing else; a test asserts the selector reads no monitor_ key.

The majority-class rate is reported next to accuracy and balanced accuracy, because "above the majority rate" is the wiring check's bar and a head that always answers the majority value ties it exactly. Reporting the rate makes the weakness of that bar visible rather than leaving it to be recomputed.

Information ceiling. The 11-attribute code determines the class for 99.25% of pbc_train: 286 distinct codes, 19 of which carry more than one class, covering 470 images. So the bottleneck caps accuracy at about 0.9925 and no lower. A bottleneck result far below that is a statement about the model, not about the representation's sufficiency.

Loss terms. The class term and the summed attribute term are recorded separately per epoch, for both train and val, and the attribute term is recorded unscaled so lambda's effect stays visible. A single blended number cannot distinguish a stuck class head from stuck attribute heads.

Mean maximum probability is reported beside ECE. ECE on its own cannot tell a confidently wrong model from a diffidently right one, and the two imply opposite readings of the abstention curve.

Abstention. The headline number is accuracy at 90% nominal coverage, with the threshold taken from pbc_val and applied unchanged to every domain. All seven coverage levels in the config are reported. Coverage actually achieved on a target domain will differ from nominal, and that difference is itself a result.

Backbone normalisation. Two arms, both declared now: batchnorm is the headline, groupnorm is a pre-declared secondary arm, run at the same settings for all three models after the headline is complete, and reported either way. BatchNorm running statistics are source statistics baked into the model and a known cause of domain-shift degradation, so they are a live confound for this whole result — but the choice is invisible on pbc_val, which is same-domain. Picking an arm because it did better on Raabin would be target-informed model selection.

What the groupnorm arm actually does, measured rather than assumed: every BatchNorm2d layer in the backbone is replaced by GroupNorm(32, channels); the pretrained BatchNorm affine parameters gamma and beta are REUSED, because GroupNorm's weight and bias carry the same names and per-channel shapes and so load from the checkpoint; the 159 running-statistic buffers (running_mean, running_var and num_batches_tracked across 53 layers) are dropped; nothing is randomly initialised; and the whole backbone is then fine-tuned as in the headline arm.

Limitation of the secondary arm. Reusing gamma and beta is a side effect of that name collision, not a considered choice: they were learned to act on batch-normalised activations, and GroupNorm normalises over channel groups within a single sample, so they are not strictly transferable — better than ones and zeros, but not a clean transfer. The arm therefore confounds two changes at once, removing the source running statistics and re-purposing the affine parameters, and a difference between the arms cannot be attributed to either alone.

Hard mode is a deliberate train/eval mismatch: the classifier is trained on smooth per-attribute probabilities and fed simplex vertices at evaluation. The soft-to-hard difference measures how far the bottleneck commits to discrete attribute decisions. It is not on its own evidence about domain invariance, and is labelled as such.

Raabin is three separate TARGET domains: raabin_train, raabin_testA, raabin_testB. There is no pooled Raabin headline number — TestB is a different camera, and pooling would hide the effect being measured. The split named Train is target data and is never trained on.

Bootstrap: n=2000, 95% CI, stratified by class. Source and target are resampled independently for the source→target gap, since they are disjoint sets of images. The headline number is still the mean ± std over 3 seeds; the bootstrap CI is reported per seed per domain alongside it.

Raabin's basophils come from a single leukemia patient. Any basophil recall on a Raabin domain carries that caveat and must be reported with it — it is one patient's cells, not a sample of basophils.

Primary comparison. Declared in full before any Raabin evaluation had been run: the ledger was empty and no target-domain number existed when this was written. It fixes what the result is, so that reading the numbers cannot become choosing the comparison.

The quantity. For a model m, a Raabin domain d and a seed s:

gap(m, d, s) = macro-F1(m, pbc_test restricted to the classes present in d, s) − macro-F1(m, d, s)

Positive is degradation. Macro-F1 is over the classes present in the eval set, and the class list is recorded beside the number. The source side is pbc_test restricted to the classes present in that target domain, so a 5-class number is never differenced against a 2-class one, and the restriction is named in the output. Predictions are an argmax over all five classes on every domain, on both sides of the difference: restricting the source side restricts which images are counted, never which classes the model may predict.

The primary comparison is the paired difference of gaps, by seed:

Δ_bb(d, s) = gap(bottleneck, d, s) − gap(baseline, d, s)
Δ_bc(d, s) = gap(bottleneck, d, s) − gap(control, d, s)

Negative Δ means the bottleneck lost less going from source to target than the model it is compared against, which is the direction the hypothesis predicts. Δ_bb is the hypothesis itself: attributes against raw CNN features. Δ_bc separates the representation from the architecture, since the control carries the same 31-unit bottleneck at lambda 0 and differs only in whether attribute labels supervised it.

Reported per Raabin domain, separately: raabin_train, raabin_testA, raabin_testB. There is no pooled Raabin row. All three seeds' values are shown individually beside the mean ± std; three paired values are a sample of three, and the per-seed spread is the honest summary of them.

No significance test. Three paired seeds do not support one, and a p-value computed on them would be read as more than it is. The per-seed values, the mean ± std, and the per-seed bootstrap CIs on each gap are the whole statistical claim.

Secondary quantity: relative drop. relative_drop(m, d, s) = gap(m, d, s) / macro-F1(m, pbc_test restricted to d's classes, s) — the gap as a fraction of what the model had on the source side. It exists because an absolute gap of 0.10 means something different from a model at 0.99 than from a model at 0.70, and the three models are not guaranteed to land at the same source number. It is secondary and decides nothing: the decision rule below is on Δ, an absolute difference of absolute gaps. It carries no interval, because the bootstrap produces replicates for the gap and not for a ratio of a gap to a point estimate; it is reported per seed plus the mean ± std across seeds.

Secondary gap: like-for-like on band neutrophils. The same gap computed with pbc_test_no_band_neutrophils on the source side, restricted to the target's classes in the same way, recorded under secondary_gaps. PBC labels band and segmented neutrophils both as Neutrophil, and the two archives' neutrophil populations need not be composed alike, so part of any neutrophil gap could be composition rather than domain shift. This subset removes the band neutrophils from the source side and asks whether the gap survives. It is computed inside the evaluation rather than afterwards because the record carries no per-image predictions, so nothing downstream could reconstruct it.

It is non-decisive. It changes only one side of the difference, so it is not a clean control: it answers whether this gap is explained by band neutrophils alone, and nothing wider. It is reported beside the primary gap for every Raabin domain and model, and the conclusion is not restated on it.

Soft and hard attribute modes are separate columns, for the bottleneck and the control. Hard mode is a deliberate train/eval mismatch and measures how far the model commits to discrete attribute decisions; it is not on its own evidence about domain invariance. The baseline has no attribute modes and is reported once; comparison between models is between soft columns.

Caveats that travel with the numbers. Any basophil recall on a Raabin domain is one leukemia patient's cells, not a sample of basophils, and is reported with that caveat. Per-class recall carries no interval: the bootstrap is computed for macro-F1 and for the gap, and not per class.

Decision rule. Chosen by the author on 2026-11-13; pending supervisor confirmation. If the supervisor changes it before the first Raabin look, that commit is the declaration of record.

Applied separately to Δ_bb and Δ_bc, and separately per domain:

supported on a domain: Δ < 0 in all three seeds
not supported: Δ ≥ 0 in at least two seeds
otherwise: mixed

The three cases are exhaustive and do not overlap: three negative seeds is supported, exactly two is mixed, one or none is not supported. The count of negative seeds is printed beside the verdict so the rule can be checked by eye. A Δ of exactly zero counts as non-negative.

Headline domains: raabin_train and raabin_testA. Both are described in the Raabin paper as acquired under the same conditions as each other, and differing from the source in lab and optics. The hypothesis is judged on these two.

raabin_testB is reported as a camera-shift stress test and is not decisive. It holds two of the five classes and comes from a different camera, so its gap is not commensurable with the other two, and it is the only domain where the camera changes — which makes it the most interesting single number here and the least comparable. It is reported in full, with the same rule applied and the same verdict printed, and that verdict does not enter the conclusion.

A verdict of mixed on either headline domain, or disagreement between Δ_bb and Δ_bc, is reported as such. There is no tie-break, and no rule that promotes a secondary quantity to decide one.

None of the above may change once any Raabin number has been seen. A change is a protocol change, not a tweak: flag it and log it in section 8 with the date and the reason.

Known limitations of this protocol, to be stated in any write-up:

One target dataset, from a single lab network, so the shift measured is mostly camera and optics (see section 1).
No hyperparameter search was run, so the three models are compared at a single operating point chosen a priori rather than one shown to suit any of them. A configuration better for one representation than another would not have been found. Reported numbers are the mean ± std over 3 seeds.
Nothing anywhere in the protocol rewards domain robustness. Selection is the final epoch of a fixed budget and no hyperparameter is searched, so no choice at all was made for its ability to generalize — which is deliberate, since the alternative is reading the target.

Changing any of the above after a target number has been seen is a protocol change, not a tweak. Flag it to me and log it in section 8.

7. Working style
For anything structural, plan first, code after approval. Say what you intend to change and where, then wait.
Prefer small diffs against a working system over rewrites.
When fixing a bug, write the failing test first.
Never silently change an experimental setting to improve a number. Surface it.
New pipelines are smoke-tested on a 500-image subset for 2 epochs before any full run.
8. Corrections log

Append here whenever I have to correct the same mistake twice.

2026-11-19 — post-hoc source-side analyses added after the Raabin results were seen.

The one-time Raabin evaluation ran on 2026-11-13 (36 ledger lines, nine checkpoints x four domains, all role headline, no overrides) and the aggregated comparison is in aggregate_output.txt. Three scripts were added afterwards: scripts/dump_predictions.py, scripts/analyze_temperature.py and scripts/analyze_attributes_source.py.

They are exploratory and source-side only. Every one of them reads pbc_val and refuses any other domain, including pbc_test, which is held out and ledgered. Nothing derived from target pixels is read, retained, reported or written to disk by any of them, so section 3 is untouched by this entry.

They are post-hoc. The primary comparison was declared in section 6 before any Raabin number existed and was decided on the records in eval_records/. These analyses can explain that result and cannot revise it, and any write-up must say which came first. No pre-declared decision rule, no metric definition, no selection rule and no ledger entry was changed; sections 3 and 6 were not edited.

A target-side version of this analysis - attribute drift from source to target, or a source-fitted temperature applied to a target domain - was requested and is NOT done. It needs per-image target predictions, which section 3 forbids retaining. Were it ever wanted it would require a scoped amendment to section 3, an entry here, and a ruling on how a second pass over Raabin is ledgered, since all 36 (checkpoint, domain) pairs are already spent.