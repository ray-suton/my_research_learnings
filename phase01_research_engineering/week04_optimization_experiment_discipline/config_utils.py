from pathlib import Path
from pprint import pformat

import yaml


REQUIRED_FIELDS = [
    "run_name",
    "model_name",
    "epochs",
    "batch_size",
    "seed",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "scheduler",
    "scheduler_min_lr",
    "data_root",
    "num_workers",
]


def validate_config(config):
    config = dict(config)
    missing = [field for field in REQUIRED_FIELDS if field not in config]
    if missing:
        raise ValueError(f"Missing required config fields: {', '.join(missing)}")

    if not isinstance(config["epochs"], int) or config["epochs"] <= 0:
        raise ValueError("epochs must be a positive integer")
    if not isinstance(config["batch_size"], int) or config["batch_size"] <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(config["seed"], int):
        raise ValueError("seed must be an integer")
    if not isinstance(config["num_workers"], int) or config["num_workers"] < 0:
        raise ValueError("num_workers must be a non-negative integer")
    for field in ("learning_rate", "weight_decay"):
        if not isinstance(config[field], (float, int)):
            raise ValueError(f"{field} must be numeric")

    config["optimizer"] = str(config["optimizer"]).lower()
    config["scheduler"] = str(config["scheduler"]).lower()

    if config["optimizer"] not in {"adamw", "sgd"}:
        raise ValueError("optimizer must be one of: adamw, sgd")
    if config["scheduler"] not in {"cosine", "none"}:
        raise ValueError("scheduler must be one of: cosine, none")
    if config["optimizer"] == "sgd" and "momentum" not in config:
        raise ValueError("SGD configs must include momentum")
    if config["scheduler"] == "cosine" and config["scheduler_min_lr"] is None:
        raise ValueError("cosine scheduler requires scheduler_min_lr")

    return config


def load_config(path):
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")

    return validate_config(config)


def format_config(config):
    return pformat(config, sort_dicts=True)


def write_config(path, config):
    Path(path).write_text(yaml.safe_dump(config, sort_keys=True))
