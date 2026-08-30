from pathlib import Path
from pprint import pformat

import yaml


REQUIRED_FIELDS = [
    "run_name",
    "corpus_path",
    "model_name",
    "epochs",
    "batch_size",
    "sequence_length",
    "embedding_dim",
    "hidden_size",
    "num_layers",
    "learning_rate",
    "seed",
    "num_workers",
    "generate_num_chars",
]


def validate_config(config):
    config = dict(config)
    missing = [field for field in REQUIRED_FIELDS if field not in config]
    if missing:
        raise ValueError(f"Missing required config fields: {', '.join(missing)}")

    positive_ints = [
        "epochs",
        "batch_size",
        "sequence_length",
        "embedding_dim",
        "hidden_size",
        "num_layers",
        "generate_num_chars",
    ]
    for field in positive_ints:
        if not isinstance(config[field], int) or config[field] <= 0:
            raise ValueError(f"{field} must be a positive integer")

    if not isinstance(config["num_workers"], int) or config["num_workers"] < 0:
        raise ValueError("num_workers must be a non-negative integer")
    if not isinstance(config["seed"], int):
        raise ValueError("seed must be an integer")
    if not isinstance(config["learning_rate"], (float, int)) or config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive")

    config["model_name"] = str(config["model_name"]).lower()
    if config["model_name"] != "lstm":
        raise ValueError("model_name must be lstm")

    config["val_fraction"] = float(config.get("val_fraction", 0.2))
    if not 0.0 < config["val_fraction"] < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    config["temperature"] = float(config.get("temperature", 0.8))
    if config["temperature"] <= 0:
        raise ValueError("temperature must be positive")

    prompts = config.get("prompts", ["#", "-", "S"])
    if isinstance(prompts, str):
        prompts = [prompts]
    if not isinstance(prompts, list) or not prompts or not all(isinstance(item, str) and item for item in prompts):
        raise ValueError("prompts must be a non-empty string or list of non-empty strings")
    config["prompts"] = prompts

    return config


def load_config(path):
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return validate_config(config)


def format_config(config):
    return pformat(config, sort_dicts=True)


def write_config(path, config):
    Path(path).write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
