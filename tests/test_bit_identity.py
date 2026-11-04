"""A numeric lock on the training loop, for changes that must not move a number.

A1 (deferring the per-batch float() calls off the step path) and A2 (pin_memory with a
non-blocking transfer) are plumbing: they change when a value is read off the device and
how a tensor is copied to it, never what is computed. This file pins the recorded history
of a small real run so that claim is checked rather than asserted.

The golden is captured on one machine and committed. Loss values depend on the torch
build, the BLAS kernels and the device, so the comparison is gated on the environment
recorded alongside it and skips elsewhere - on Kaggle it will skip. That is the honest
limit of a stored constant: it locks this machine's numbers across commits, which is
exactly what a refactor needs, and nothing more.

Source domain only. Reads data/raw/, writes only into tmp_path.
"""
import json
import platform
from pathlib import Path

import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).resolve().parent / "golden" / "micro_run_history.json"

# Both, because A1 defers three float() calls and only the bottleneck exercises the
# attribute one with a non-zero value.
CONFIGS = ("smoke_baseline.yaml", "smoke.yaml")

# Small on purpose: the property under test is where a scalar is read and how a batch is
# copied, neither of which depends on the image size or the number of steps.
MICRO = {"data.subset_n_images": 32, "data.image_size": 64,
         "train.batch_size": 8, "train.epochs": 1,
         # warmup must be shorter than the run, and the run is one epoch.
         "train.warmup_epochs": 0}

# Keys whose values are wall-clock or environment, not arithmetic.
VOLATILE = {"seconds"}


def environment() -> dict:
    """What the golden's numbers depend on."""
    import torch

    return {"torch": torch.__version__,
            "machine": platform.machine(),
            "system": platform.system(),
            "device": "cpu"}


def micro_run_history(config_name: str, out_dir: Path) -> list[dict]:
    """One epoch of the real training loop on 32 real pbc_train images, on CPU."""
    from src.config import load_config, set_seed
    from src.data.domains import load_class_mapping
    from src.data.label_space import load_label_space
    from src.train import train

    cfg = load_config(REPO / "configs" / config_name)
    doc = json.loads(json.dumps(cfg.resolved))
    for dotted, value in MICRO.items():
        section, key = dotted.split(".")
        doc[section][key] = value

    # Rebuild through load_config so every validator runs on the overridden doc.
    import tempfile

    import yaml

    doc["paths"] = {"local": dict(doc["paths"]), "kaggle": dict(doc["paths"])}
    doc["env"] = "local"
    data = doc["data"]
    data["class_mapping"] = data.pop("class_mapping_path")
    data["label_space"] = data.pop("label_space_path")
    doc["data"] = {k: v for k, v in data.items()
                   if k in {"class_mapping", "label_space", "source", "image_size",
                            "subset_n_images", "normalization"}}
    if doc.get("bottleneck") is None:
        doc.pop("bottleneck", None)
    if doc["eval"].get("attribute_modes") is None:
        doc["eval"].pop("attribute_modes", None)
    path = Path(tempfile.mkdtemp()) / "micro.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")

    cfg = load_config(path)
    set_seed(cfg.seed)
    mapping = load_class_mapping(cfg.data.class_mapping_path)
    label_space = load_label_space(cfg.resolved["data"]["label_space_path"])
    return train(cfg, mapping, label_space, out_dir=out_dir)["history"]


def comparable(history: list[dict]) -> list[dict]:
    return [{k: v for k, v in row.items() if k not in VOLATILE} for row in history]


def _requires_source_data():
    pbc = REPO / "data" / "raw" / "PBC_dataset_normal_DIB"
    csv = REPO / "data" / "raw" / "pbc_attr_v1_train.csv"
    if not (pbc.is_dir() and csv.is_file()):
        pytest.skip("data/raw/ is not present; the numeric lock needs the real images")


def _requires_matching_environment(golden: dict):
    here, there = environment(), golden["environment"]
    differing = {k: (there.get(k), here[k]) for k in here if there.get(k) != here[k]}
    if differing:
        pytest.skip(f"golden was captured elsewhere (golden, here): {differing}")


# --- the lock -------------------------------------------------------------------------

@pytest.mark.parametrize("config_name", CONFIGS)
def test_the_micro_run_history_is_unchanged(config_name, tmp_path):
    """Every recorded loss, to the last bit. A1 and A2 must not move any of these."""
    if not GOLDEN.is_file():
        pytest.skip(f"no golden at {GOLDEN}; capture one before relying on this")
    golden = json.loads(GOLDEN.read_text())
    _requires_source_data()
    _requires_matching_environment(golden)
    assert config_name in golden["runs"], f"{config_name} is not in the golden"

    assert (comparable(micro_run_history(config_name, tmp_path))
            == golden["runs"][config_name])


@pytest.mark.parametrize("config_name", CONFIGS)
def test_the_micro_run_reproduces_itself_in_one_process(config_name, tmp_path):
    """Environment-independent, so this one runs everywhere the data does. It catches a
    change that makes the loop non-deterministic even where no stored constant applies.
    """
    _requires_source_data()
    first = comparable(micro_run_history(config_name, tmp_path / "a"))
    second = comparable(micro_run_history(config_name, tmp_path / "b"))
    assert first == second


def test_the_bottleneck_golden_exercises_the_attribute_loss():
    """A1 defers three float() calls. A golden where the attribute term is always zero
    would lock two of them and miss the third."""
    if not GOLDEN.is_file():
        pytest.skip(f"no golden at {GOLDEN}")
    rows = json.loads(GOLDEN.read_text())["runs"]["smoke.yaml"]
    assert all(row["train_attribute_loss_unscaled"] > 0 for row in rows)
    assert all(row["val_attribute_loss_unscaled"] > 0 for row in rows)


def test_the_golden_records_what_its_numbers_depend_on():
    """A stored constant with no environment beside it is a test that fails mysteriously
    on someone else's machine."""
    if not GOLDEN.is_file():
        pytest.skip(f"no golden at {GOLDEN}")
    golden = json.loads(GOLDEN.read_text())
    assert set(golden["environment"]) == set(environment())
    assert golden["overrides"] == MICRO
    assert set(golden["runs"]) == set(CONFIGS)
    assert all(rows for rows in golden["runs"].values()), "a golden with no history"


# --- A1: nothing is read off the device inside the step loop ---------------------------

def test_the_step_loop_reads_no_scalar_off_the_device():
    """Structural, in the style of test_selection_reads_no_monitoring_key. float() on a
    CUDA tensor synchronises; three per step is 579 blocking waits per pbc_train epoch.
    The one isfinite() check is deliberate - a diverged run must stop at the step it
    happened - and keeps the comparison on-device.
    """
    import inspect
    import re

    import src.train as train_mod

    source = inspect.getsource(train_mod.train)
    loop = re.search(r"\n( +)for images, labels, attributes in train_loader:\n"
                     r"((?:\1 +.*\n|\n)+)", source)
    assert loop, "could not locate the step loop in train()"
    body = loop.group(2)
    assert "float(" not in body, f"a scalar is read off the device per step:\n{body}"
    assert "isfinite" in body, "the divergence guard left the step loop"
