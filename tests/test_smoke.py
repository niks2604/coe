"""The sec 7 smoke test: 500 images, 2 epochs, CPU, before any full run."""
import pytest

from src.config import load_config

pytest.importorskip("torch")


@pytest.fixture
def smoke_config_path():
    from pathlib import Path
    return Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"


def test_smoke_config_loads(smoke_config_path):
    cfg = load_config(smoke_config_path)
    assert cfg.data.subset_n_images == 500
    assert cfg.train.epochs == 2


def test_smoke_config_is_cpu_local(smoke_config_path):
    assert load_config(smoke_config_path).env == "local"


def test_smoke_config_keeps_the_real_image_size(smoke_config_path):
    """A smaller image would smoke-test a different model than the one that runs."""
    assert load_config(smoke_config_path).data.image_size == 224


def test_smoke_run_reports_seconds_per_epoch(experiment_config, tiny_raw_loadable,
                                             mapping, label_space, tmp_path):
    """The number the smoke test exists to produce: whether a full run is affordable."""
    from src.train import train

    cfg = load_config(experiment_config(**{"data.image_size": 32,
                                           "train.epochs": 2,
                                           "train.batch_size": 2}))
    result = train(cfg, mapping, label_space, out_dir=tmp_path)
    assert len(result["seconds_per_epoch"]) == 2
    assert all(s > 0 for s in result["seconds_per_epoch"])
    assert result["wall_clock_seconds"] > 0


def test_smoke_run_writes_a_results_json(experiment_config, tiny_raw_loadable,
                                         mapping, label_space, tmp_path):
    import json

    from src.train import train

    cfg = load_config(experiment_config(**{"data.image_size": 32, "train.epochs": 2,
                                           "train.batch_size": 2}))
    result = train(cfg, mapping, label_space, out_dir=tmp_path)
    written = json.loads((tmp_path / f"{cfg.run_id}_seed{cfg.seed}.json").read_text())
    assert written["config"]["run_id"] == cfg.run_id
    assert written["seconds_per_epoch"] == result["seconds_per_epoch"]


def test_smoke_run_is_seed_reproducible(experiment_config, tiny_raw_loadable,
                                        mapping, label_space, tmp_path):
    import torch

    from src.train import train

    def final_weights(seed, out):
        cfg = load_config(experiment_config(**{"seed": seed, "data.image_size": 32,
                                               "train.epochs": 1,
                                               "train.batch_size": 2}))
        train(cfg, mapping, label_space, out_dir=out)
        payload = torch.load(out / "last.pt", weights_only=False)
        return payload["state_dict"]["classifier.weight"]

    a = final_weights(0, tmp_path / "a")
    b = final_weights(0, tmp_path / "b")
    assert torch.allclose(a, b)


def test_different_seeds_give_different_weights(experiment_config, tiny_raw_loadable,
                                                mapping, label_space, tmp_path):
    import torch

    from src.train import train

    def final_weights(seed, out):
        cfg = load_config(experiment_config(**{"seed": seed, "data.image_size": 32,
                                               "train.epochs": 1,
                                               "train.batch_size": 2}))
        train(cfg, mapping, label_space, out_dir=out)
        return torch.load(out / "last.pt", weights_only=False)["state_dict"]["classifier.weight"]

    assert not torch.allclose(final_weights(0, tmp_path / "a"),
                              final_weights(1, tmp_path / "b"))
