"""The evaluation script: checkpoint-driven, ledgered, and read-only with respect to
the model's state."""
import pytest

from src.eval import EvalError, evaluate_checkpoint, main

pytest.importorskip("torch")


@pytest.fixture
def trained(experiment_config, tiny_raw_loadable, mapping, label_space, tmp_path):
    """A real but tiny checkpoint, saved the way src.train saves one."""
    from src.config import load_config
    from src.train import save_checkpoint

    cfg = load_config(experiment_config(**{"data.image_size": 32,
                                           "train.batch_size": 2,
                                           "train.epochs": 1}))
    path = tmp_path / "best.pt"
    save_checkpoint(path, cfg, mapping, label_space,
                    history=[{"epoch": 0, "val_macro_f1": 0.5}])
    return cfg, path


# --- the checkpoint is the source of truth -------------------------------------------

def test_eval_reads_the_config_from_the_checkpoint(trained, tmp_path):
    """Not a YAML passed alongside it: a checkpoint must not be evaluated under
    preprocessing it was not trained with (CLAUDE.md sec 5)."""
    import inspect

    import src.eval as eval_mod

    cfg, path = trained
    assert "--config" not in inspect.getsource(eval_mod)


def test_checkpoint_without_an_embedded_config_is_refused(tmp_path):
    import torch

    path = tmp_path / "bare.pt"
    torch.save({"state_dict": {}}, path)
    with pytest.raises(EvalError, match="config"):
        evaluate_checkpoint(path, ledger_path=tmp_path / "l.jsonl")


def test_a_shape_mismatch_between_config_and_weights_is_refused(trained, tmp_path):
    import torch

    cfg, path = trained
    payload = torch.load(path, weights_only=False)
    payload["state_dict"] = {"classifier.weight": torch.zeros(5, 999)}
    bad = tmp_path / "bad.pt"
    torch.save(payload, bad)
    with pytest.raises(EvalError):
        evaluate_checkpoint(bad, ledger_path=tmp_path / "l.jsonl")


# --- dry run ---------------------------------------------------------------------------

def test_dry_run_evaluates_pbc_val_only(trained, tmp_path):
    cfg, path = trained
    ledger = tmp_path / "l.jsonl"
    out = evaluate_checkpoint(path, ledger_path=ledger, dry_run=True)
    assert set(out["domains"]) == {"pbc_val"}


def test_dry_run_touches_no_target_and_no_held_out_split(trained, tmp_path):
    cfg, path = trained
    out = evaluate_checkpoint(path, ledger_path=tmp_path / "l.jsonl", dry_run=True)
    assert "pbc_test" not in out["domains"]
    assert not any(d.startswith("raabin") for d in out["domains"])


def test_dry_run_writes_no_ledger_entry(trained, tmp_path):
    cfg, path = trained
    ledger = tmp_path / "l.jsonl"
    evaluate_checkpoint(path, ledger_path=ledger, dry_run=True)
    assert not ledger.exists()


def test_dry_run_is_recorded_in_the_results(trained, tmp_path):
    cfg, path = trained
    out = evaluate_checkpoint(path, ledger_path=tmp_path / "l.jsonl", dry_run=True)
    assert out["dry_run"] is True


def test_dry_run_can_be_repeated(trained, tmp_path):
    """pbc_val is the tuning split, read constantly by design."""
    cfg, path = trained
    ledger = tmp_path / "l.jsonl"
    for _ in range(3):
        evaluate_checkpoint(path, ledger_path=ledger, dry_run=True)


# --- the real run ------------------------------------------------------------------------

def test_full_run_evaluates_source_test_and_three_raabin_domains(trained, tmp_path):
    cfg, path = trained
    out = evaluate_checkpoint(path, ledger_path=tmp_path / "l.jsonl")
    assert set(out["domains"]) >= {"pbc_test", "raabin_train", "raabin_testA",
                                   "raabin_testB"}


def test_full_run_includes_the_secondary_subset(trained, tmp_path):
    cfg, path = trained
    out = evaluate_checkpoint(path, ledger_path=tmp_path / "l.jsonl")
    assert "pbc_test_no_band_neutrophils" in out["secondary_subsets"]


def test_full_run_ledgers_every_domain_it_touched(trained, tmp_path):
    from src.ledger import read_ledger

    cfg, path = trained
    ledger = tmp_path / "l.jsonl"
    out = evaluate_checkpoint(path, ledger_path=ledger)
    ledgered = {row["domain"] for row in read_ledger(ledger)}
    assert ledgered == set(out["domains"])


def test_a_second_full_run_is_refused(trained, tmp_path):
    from src.ledger import LedgerError

    cfg, path = trained
    ledger = tmp_path / "l.jsonl"
    evaluate_checkpoint(path, ledger_path=ledger)
    with pytest.raises(LedgerError):
        evaluate_checkpoint(path, ledger_path=ledger)


def test_a_second_run_with_a_logged_override_is_allowed(trained, tmp_path):
    from src.ledger import read_ledger

    cfg, path = trained
    ledger = tmp_path / "l.jsonl"
    evaluate_checkpoint(path, ledger_path=ledger)
    evaluate_checkpoint(path, ledger_path=ledger, override_reason="ECE bug fixed")
    assert any(r.get("override_reason") == "ECE bug fixed" for r in read_ledger(ledger))


def test_both_attribute_modes_are_reported_for_the_bottleneck(trained, tmp_path):
    cfg, path = trained
    out = evaluate_checkpoint(path, ledger_path=tmp_path / "l.jsonl")
    assert set(out["results"]["pbc_test"]) >= {"soft", "hard"}


def test_per_attribute_accuracy_is_reported_for_pbc_test_only(trained, tmp_path):
    cfg, path = trained
    out = evaluate_checkpoint(path, ledger_path=tmp_path / "l.jsonl")
    assert out["per_attribute"]["pbc_test"]
    assert "raabin_testB" not in out["per_attribute"]


def test_gap_uses_a_restricted_source_side_for_testb(trained, tmp_path):
    """raabin_testB has 2 of 5 classes, so its gap is against pbc_test restricted to
    those two, and the restriction is named (CLAUDE.md sec 6)."""
    cfg, path = trained
    out = evaluate_checkpoint(path, ledger_path=tmp_path / "l.jsonl")
    gap = out["gaps"]["raabin_testB"]
    assert gap["source_classes"] == ["neutrophil", "lymphocyte"]
    assert gap["source_domain"] == "pbc_test"


def test_results_json_carries_the_claude_md_section_5_fields(trained, tmp_path):
    cfg, path = trained
    out = evaluate_checkpoint(path, ledger_path=tmp_path / "l.jsonl")
    assert isinstance(out["config"]["data"]["class_mapping"], dict)
    assert len(out["git"]["commit"]) == 40
    assert "wall_clock_seconds" in out and "hardware" in out


# --- the model is never adapted on target data ----------------------------------------------

def test_model_is_in_eval_mode_throughout(trained, tmp_path):
    """model.train() on target data would update BatchNorm running statistics, which is
    test-time adaptation, which is leakage (CLAUDE.md sec 3)."""
    cfg, path = trained
    out = evaluate_checkpoint(path, ledger_path=tmp_path / "l.jsonl")
    assert out["model_training_mode_seen"] is False


def test_batchnorm_buffers_are_unchanged_by_evaluation(trained, tmp_path):
    cfg, path = trained
    out = evaluate_checkpoint(path, ledger_path=tmp_path / "l.jsonl")
    assert out["buffer_sha256_before"] == out["buffer_sha256_after"]


def test_evaluation_runs_without_grad(trained, tmp_path):
    cfg, path = trained
    out = evaluate_checkpoint(path, ledger_path=tmp_path / "l.jsonl")
    assert out["grad_enabled_seen"] is False


# --- CLI ---------------------------------------------------------------------------------------

def test_cli_dry_run_exits_zero(trained, tmp_path, capsys):
    cfg, path = trained
    code = main(["--checkpoint", str(path), "--dry-run",
                 "--ledger", str(tmp_path / "l.jsonl")])
    assert code == 0
    assert capsys.readouterr().out.strip()


def test_cli_rejects_an_empty_override_reason(trained, tmp_path):
    cfg, path = trained
    with pytest.raises(SystemExit):
        main(["--checkpoint", str(path), "--override-reason", "",
              "--ledger", str(tmp_path / "l.jsonl")])


# --- a dry run must not open a single out-of-domain file ---------------------------

def test_dry_run_opens_no_file_under_the_target_root(trained, tmp_path, monkeypatch):
    """Stronger than checking which domains were scored: every image read in
    src.data.datasets goes through _load_rgb, so recording its argument proves nothing
    outside the source split was opened at all."""
    import src.data.datasets as datasets_mod

    cfg, path = trained
    opened: list[str] = []
    original = datasets_mod._load_rgb

    def recording(p):
        opened.append(str(p))
        return original(p)

    monkeypatch.setattr(datasets_mod, "_load_rgb", recording)
    evaluate_checkpoint(path, ledger_path=tmp_path / "l.jsonl", dry_run=True)

    target_root = str(cfg.paths.raabin_root)
    assert opened, "the dry run opened no images at all"
    assert not [p for p in opened if p.startswith(target_root)]


def test_full_run_does_open_the_target_root(trained, tmp_path, monkeypatch):
    """The companion check: the recorder would notice if it did, so the test above is
    not vacuously true."""
    import src.data.datasets as datasets_mod

    cfg, path = trained
    opened: list[str] = []
    original = datasets_mod._load_rgb
    monkeypatch.setattr(datasets_mod, "_load_rgb",
                        lambda p: (opened.append(str(p)), original(p))[1])
    evaluate_checkpoint(path, ledger_path=tmp_path / "l.jsonl")
    assert [p for p in opened if p.startswith(str(cfg.paths.raabin_root))]


# --- mean max probability, beside ECE -------------------------------------------------

def test_score_reports_mean_max_probability_next_to_ece(trained):
    """ECE alone cannot distinguish a confidently wrong model from a diffident one, and
    the smoke run produced an ECE of 0.56 with no way to see which it was."""
    from src.eval import _score

    cfg, _ = trained
    classes = ["neutrophil", "lymphocyte", "monocyte", "eosinophil", "basophil"]
    predictions = {"y_true": [0, 1], "y_pred": [0, 1], "confidence": [0.9, 0.7]}
    scored = _score(cfg, predictions, classes, None)
    assert scored["mean_max_probability"] == pytest.approx(0.8)
    assert "ece" in scored
