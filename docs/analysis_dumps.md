# Producing the analysis dumps on Kaggle

Post-hoc, under the CLAUDE.md §3 scoped amendment of 2026-11-19 and the §8 entry of the
same date. The §6 verdict was fixed on `eval_records/` on 2026-11-13 and nothing here
revises it.

**This session evaluates nothing.** It writes `analysis/*.npz` and appends to
`analysis_ledger.jsonl`. `target_eval_ledger.jsonl` and `eval_records/` are read, never
written, and the last cell proves it.

## What to attach

The same four datasets as the evaluation session, at the same paths — the checkpoints
embed them and the embedded config is not editable:

```
pbc-images   wbcatt   raabin-double-labeled   wbc-final-checkpoints
```

all under `/kaggle/input/datasets/anikakrishnaa/`. Accelerator on. Do not attach the
training notebook's output.

## Expected length

The evaluation itself ran 2026-11-13 04:02–05:12 UTC — about 70 minutes for nine
checkpoints over four domains. The target dumps are the same forward passes, so budget
the same, plus roughly ten minutes for the `pbc_val` dumps. Call it 90 minutes.

### Cell 1 — clone and check

```python
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import importlib.util
import subprocess, sys
from pathlib import Path

REPO = Path("/kaggle/working/coe")
!rm -rf /kaggle/working/coe
!git clone -q https://github.com/niks2604/coe.git /kaggle/working/coe
os.chdir(REPO); sys.path.insert(0, str(REPO))

print("HEAD:", subprocess.run(["git","-C",str(REPO),"rev-parse","--short","HEAD"],
                              capture_output=True, text=True).stdout.strip())
print("primary ledger lines:", len((REPO/"target_eval_ledger.jsonl").read_text().splitlines()))
print("analysis ledger lines:", len((REPO/"analysis_ledger.jsonl").read_text().splitlines()))
print("eval records:", len(list((REPO/"eval_records").glob("*_eval.json"))))

MODULES = {"torch": "torch", "torchvision": "torchvision", "numpy": "numpy",
           "yaml": "PyYAML", "PIL": "pillow", "pandas": "pandas"}
missing = [pkg for mod, pkg in MODULES.items() if not importlib.util.find_spec(mod)]
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
print("missing installed:", missing or "none")

import torch
assert torch.cuda.is_available(), "no GPU: set Accelerator before running"
print(torch.__version__, torch.cuda.get_device_name(0))
```

Expect 36 primary-ledger lines, 0 analysis-ledger lines, 9 eval records.

### Cell 2 — record what must not change

```python
import hashlib

def fingerprint():
    h = hashlib.sha256()
    h.update((REPO / "target_eval_ledger.jsonl").read_bytes())
    for p in sorted((REPO / "eval_records").glob("*.json")):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()

BEFORE = fingerprint()
print("ledger + records fingerprint:", BEFORE[:16])
```

### Cell 3 — the pbc_val dumps

`--device cuda` deliberately: the temperature is fitted on these logits and carried to
the target dumps, which run on cuda, and a device split between the fit and the
application would show up as calibration drift that is really arithmetic.

```python
D = "/kaggle/input/datasets/anikakrishnaa/wbc-final-checkpoints/wbc-final-checkpoints"
!cd /kaggle/working/coe && python -u -m scripts.dump_predictions \
    --checkpoints "$D" --out /kaggle/working/coe/analysis --device cuda
```

Nine files, `<run_id>_seed<N>_pbc_val.npz`. This script writes no ledger line: `pbc_val`
is the tuning split and spends nothing.

### Cell 4 — the target dumps

No `--device`: it defaults to the device the checkpoint records, which is what the
evaluation used, so cell 5 can compare tightly.

```python
!cd /kaggle/working/coe && python -u -m scripts.dump_target_predictions \
    --checkpoints "$D" --out /kaggle/working/coe/analysis
```

36 files, and 36 lines appended to `analysis_ledger.jsonl`.

### Cell 5 — the consistency check

**If this fails, stop.** A dump that does not reproduce the ledgered evaluation is
reading something else, and no analysis may be read off it.

```python
!cd /kaggle/working/coe && python -u -m scripts.check_dump_consistency \
    --analysis analysis/ --records eval_records/
```

### Cell 6 — the analyses

```python
!cd /kaggle/working/coe && python -u -m scripts.analyze_attribute_drift \
    --analysis analysis/ --json analysis/attribute_drift.json | head -60
```

```python
!cd /kaggle/working/coe && python -u -m scripts.analyze_temperature \
    --analysis analysis/ --json analysis/temperature_pbc_val.json
```

`analyze_temperature` reads `pbc_val` only and **refuses** a directory holding target
dumps, which this one now does. Run it against a `pbc_val`-only copy:

```python
!mkdir -p /kaggle/working/pbcval_only && cp /kaggle/working/coe/analysis/*_pbc_val.npz /kaggle/working/pbcval_only/
!cd /kaggle/working/coe && python -u -m scripts.analyze_temperature \
    --analysis /kaggle/working/pbcval_only --json analysis/temperature_pbc_val.json
```

```python
!cd /kaggle/working/coe && python -u -m scripts.analyze_temperature_transfer \
    --analysis analysis/ --json analysis/temperature_transfer.json
```

```python
!cd /kaggle/working/coe && python -u -m scripts.analyze_attributes_source \
    --analysis /kaggle/working/pbcval_only --json analysis/attributes_source.json | head -40
```

### Cell 7 — prove nothing protected changed

```python
assert fingerprint() == BEFORE, "target_eval_ledger.jsonl or eval_records/ CHANGED"
print("primary ledger and eval records unchanged")

import json, collections
rows = [json.loads(l) for l in (REPO/"analysis_ledger.jsonl").read_text().splitlines() if l.strip()]
print("analysis ledger:", dict(collections.Counter(r["domain"] for r in rows)), "total", len(rows))
assert len(rows) == 36, f"expected 36 analysis ledger lines, got {len(rows)}"
print("dumps:", len(list((REPO/"analysis").glob("*.npz"))), "npz files")
```

### Cell 8 — collect

```python
!mkdir -p /kaggle/working/outputs
!cp -r /kaggle/working/coe/analysis /kaggle/working/outputs/
!cp /kaggle/working/coe/analysis_ledger.jsonl /kaggle/working/outputs/
!du -sh /kaggle/working/outputs && ls /kaggle/working/outputs/analysis | head
print("\nDownload /kaggle/working/outputs, then STOP THE GPU SESSION.")
```

## Bringing it home

`analysis_ledger.jsonl` is append-only and does not persist on Kaggle, so carry it the
same way as the primary one — verify, then append only what is new:

```bash
DL=~/Downloads/outputs/analysis_ledger.jsonl
n=$(wc -l < analysis_ledger.jsonl)
if [ "$n" -gt 0 ]; then
  head -n "$n" "$DL" | cmp - analysis_ledger.jsonl || { echo "STOP: not an extension"; exit 1; }
fi
tail -n +$((n + 1)) "$DL" >> analysis_ledger.jsonl
git add analysis_ledger.jsonl && git commit -m "analysis: ledger for the post-hoc dumps"
```

Decide separately whether the `.npz` dumps themselves belong in git — they are large and
reproducible from the checkpoints, and `analysis/` is not currently in `.gitignore`.
