import sys
from pathlib import Path

import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from config_utils import load_config, validate_config


CONFIG_DIR = PROJECT_DIR / "configs"
ALL_CONFIGS = [
    "adamw_cosine_baseline.yaml",
    "sgd_momentum.yaml",
    "lr_low.yaml",
    "lr_high.yaml",
    "no_scheduler.yaml",
    "batch_size_variant.yaml",
    "seed_repeat_a.yaml",
    "seed_repeat_b.yaml",
    "selected_final_seed7.yaml",
]


def test_all_planned_configs_load():
    for name in ALL_CONFIGS:
        config = load_config(CONFIG_DIR / name)
        assert config["run_name"]
        assert config["model_name"] == "mini_resnet"
        assert config["epochs"] == 50
        assert config["batch_size"] > 0


def test_missing_required_field_fails_clearly():
    config = load_config(CONFIG_DIR / "adamw_cosine_baseline.yaml")
    del config["learning_rate"]

    with pytest.raises(ValueError, match="Missing required config fields: learning_rate"):
        validate_config(config)


def test_sgd_requires_momentum():
    config = load_config(CONFIG_DIR / "sgd_momentum.yaml")
    del config["momentum"]

    with pytest.raises(ValueError, match="SGD configs must include momentum"):
        validate_config(config)


def test_scheduler_names_are_normalized():
    config = load_config(CONFIG_DIR / "no_scheduler.yaml")
    assert config["scheduler"] == "none"
