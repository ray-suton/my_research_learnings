import argparse
import json
from pathlib import Path

from config_utils import load_config
from train import train


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIGS = [
    "configs/adamw_cosine_baseline.yaml",
    "configs/sgd_momentum.yaml",
    "configs/lr_low.yaml",
    "configs/lr_high.yaml",
    "configs/no_scheduler.yaml",
    "configs/batch_size_variant.yaml",
    "configs/seed_repeat_a.yaml",
    "configs/seed_repeat_b.yaml",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="*",
        default=DEFAULT_CONFIGS,
        help="Config paths relative to the Week 4 directory, or absolute paths.",
    )
    return parser.parse_args()


def resolve_config(path):
    config_path = Path(path)
    if config_path.is_absolute():
        return config_path
    return PROJECT_ROOT / config_path


def main():
    args = parse_args()
    summaries = []

    for raw_path in args.configs:
        config_path = resolve_config(raw_path)
        config = load_config(config_path)
        print(f"\n=== Running {config['run_name']} from {config_path} ===")
        metrics = train(config)
        summaries.append(
            {
                "run_name": metrics["run_name"],
                "run_dir": metrics.get("run_dir"),
                "best_val_accuracy": metrics["best_val_accuracy"],
                "final_val_accuracy": metrics["final_val_accuracy"],
                "runtime_seconds": metrics["runtime_seconds"],
            }
        )

    print("\n=== Ablation summary ===")
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
