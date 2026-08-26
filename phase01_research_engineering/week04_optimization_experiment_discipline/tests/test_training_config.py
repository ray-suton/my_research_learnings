import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import train as week4_train
from config_utils import load_config
from model import build_model


CONFIG_DIR = PROJECT_DIR / "configs"


def test_optimizer_and_scheduler_builders_follow_config():
    expected = {
        "adamw_cosine_baseline.yaml": ("AdamW", "CosineAnnealingLR"),
        "sgd_momentum.yaml": ("SGD", "CosineAnnealingLR"),
        "lr_low.yaml": ("AdamW", "CosineAnnealingLR"),
        "lr_high.yaml": ("AdamW", "CosineAnnealingLR"),
        "no_scheduler.yaml": ("AdamW", None),
        "batch_size_variant.yaml": ("AdamW", "CosineAnnealingLR"),
        "seed_repeat_a.yaml": ("AdamW", "CosineAnnealingLR"),
        "seed_repeat_b.yaml": ("AdamW", "CosineAnnealingLR"),
        "selected_final_seed7.yaml": ("AdamW", "CosineAnnealingLR"),
    }

    for name, (optimizer_name, scheduler_name) in expected.items():
        config = load_config(CONFIG_DIR / name)
        model = build_model(config["model_name"])
        optimizer = week4_train.build_optimizer(model, config)
        scheduler = week4_train.build_scheduler(optimizer, config)

        assert type(optimizer).__name__ == optimizer_name
        assert (type(scheduler).__name__ if scheduler is not None else None) == scheduler_name


def test_training_loop_runs_one_fake_batch_without_test_eval(monkeypatch):
    config = load_config(CONFIG_DIR / "adamw_cosine_baseline.yaml")
    config["epochs"] = 1
    config["batch_size"] = 2

    fake_loader = [(torch.randn(2, 3, 32, 32), torch.tensor([0, 1]))]
    monkeypatch.setattr(
        week4_train,
        "get_dataloaders",
        lambda **kwargs: (fake_loader, fake_loader, fake_loader),
    )

    metrics = week4_train.train(config, record_run=False)

    assert metrics["status"] == "completed"
    assert metrics["epochs"] == 1
    assert metrics["test_evaluated"] is False
    assert len(metrics["history"]) == 1


def test_training_loop_writes_run_record_for_fake_batch(monkeypatch, tmp_path):
    config = load_config(CONFIG_DIR / "adamw_cosine_baseline.yaml")
    config["epochs"] = 1
    config["batch_size"] = 2

    fake_loader = [(torch.randn(2, 3, 32, 32), torch.tensor([0, 1]))]
    monkeypatch.setattr(
        week4_train,
        "get_dataloaders",
        lambda **kwargs: (fake_loader, fake_loader, fake_loader),
    )
    monkeypatch.setattr(week4_train, "RUNS_ROOT", tmp_path)

    metrics = week4_train.train(config, record_run=True)
    run_dirs = list(tmp_path.iterdir())

    assert metrics["status"] == "completed"
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "config.yaml").exists()
    assert (run_dirs[0] / "metrics.json").exists()
    assert (run_dirs[0] / "notes.md").exists()
