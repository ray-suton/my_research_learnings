"""Reproduce the fixed Week 7 Session 3, 4, and 5 experiments."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parent


def run_command(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("Running:", shlex.join(command), flush=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    if code:
        raise RuntimeError(f"Command failed with exit {code}; see {log_path}")


def summarize_ablation(runs, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import torch
    from model import vae

    rows = []
    for run in runs:
        values = run["final_validation"]
        rows.append({
            "run_name": run["run_name"], "latent_dim": run["config"]["latent_dim"],
            "validation_mse": values["reconstruction_mse"],
            "mean_latent_mse": values["mean_latent_mse"], "kl": values["kl"],
            "total": values["total"],
            "active_dimensions": values["active_latent_dimensions"],
            "parameters": run["parameter_count"], "runtime_seconds": run["runtime_seconds"],
            "checkpoint": run["checkpoint"],
        })
    rows.sort(key=lambda row: row["latent_dim"])
    summary = {
        "status": "completed", "latent_sizes": [8, 16, 32, 64],
        "fixed_settings": {
            "epochs": 5, "batch_size": 128, "learning_rate": 0.001,
            "hidden_dim": 128, "beta": 1.0, "seed": 42,
            "validation_seed": 123, "device": "cpu", "threads": 1,
        },
        "active_dimension_definition": "mean KL contribution > 0.01 nats on validation",
        "test_used_for_selection": False,
        "limitations": "One seed; final-epoch models; no significance or generation metric",
        "runs": rows,
    }
    (output_dir / "session4_latent_ablation_metrics.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    with (output_dir / "session4_latent_ablation.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    sizes = [row["latent_dim"] for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    axes[0].plot(sizes, [row["validation_mse"] for row in rows], "o-", label="posterior draw")
    axes[0].plot(sizes, [row["mean_latent_mse"] for row in rows], "o-", label="decode(mu)")
    axes[0].set_ylabel("Validation per-pixel MSE")
    axes[0].legend()
    axes[1].plot(sizes, [row["kl"] for row in rows], "o-")
    axes[1].set_ylabel("Validation KL per image (nats)")
    axes[2].plot(sizes, [row["runtime_seconds"] for row in rows], "o-")
    axes[2].set_ylabel("Training + validation seconds")
    for ax in axes:
        ax.set_xticks(sizes)
        ax.set_xlabel("Latent dimensions")
    fig.savefig(output_dir / "session4_latent_ablation.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(4, 8, figsize=(10, 5.5), constrained_layout=True)
    for row_index, row in enumerate(rows):
        checkpoint = torch.load(row["checkpoint"], map_location="cpu", weights_only=True)
        model = vae(784, 128, row["latent_dim"])
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        generator = torch.Generator().manual_seed(2026)
        with torch.no_grad():
            z = torch.randn(8, row["latent_dim"], generator=generator)
            images = model.decoder(z).reshape(8, 1, 28, 28)
        for column, image in enumerate(images):
            ax = axes[row_index, column]
            ax.imshow(image.squeeze(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if column == 0:
                ax.set_ylabel(f"latent {row['latent_dim']}")
    fig.suptitle("Prior samples after five epochs; one seed, qualitative comparison")
    fig.savefig(output_dir / "session4_prior_comparison.png", dpi=140)
    plt.close(fig)

    selected = min(rows, key=lambda row: (row["validation_mse"], row["latent_dim"]))
    selection = {
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "criterion": "minimum final validation sampled-reconstruction MSE; ties: smaller latent",
        "test_used": False, "selected": selected,
        "checkpoint_sha256": hashlib.sha256(Path(selected["checkpoint"]).read_bytes()).hexdigest(),
    }
    (output_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve()
    if output_dir.exists():
        parser.error("Output directory already exists; choose a new run directory")
    output_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/week07_vae_matplotlib")
    (output_dir / "command.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")

    runs = []
    for latent in (32, 8, 16, 64):
        name = f"session3_latent{latent}" if latent == 32 else f"session4_latent{latent}"
        run_dir = output_dir / name
        command = [
            sys.executable, "-u", str(PROJECT_DIR / "train.py"),
            "--run-name", name, "--latent", str(latent),
            "--epochs", "5", "--batch", "128", "--hidden", "128",
            "--lr", "0.001", "--beta", "1", "--seed", "42",
            "--validation-seed", "123", "--device", "cpu", "--threads", "1",
            "--output-dir", str(run_dir),
            "--checkpoint", str(checkpoint_dir / f"{name}.pt"),
        ]
        run_command(command, run_dir / "stdout.log")
        runs.append(json.loads((run_dir / "metrics.json").read_text()))
    selected = summarize_ablation(runs, output_dir)
    print(f"Locked selected latent size: {selected['latent_dim']}", flush=True)
    comparison_dir = output_dir / "session5_comparison"
    run_command([
        sys.executable, "-u", str(PROJECT_DIR / "compare_autoencoder.py"),
        "--vae-checkpoint", selected["checkpoint"], "--output-dir", str(comparison_dir),
        "--epochs", "5", "--batch", "128", "--seed", "42", "--lr", "0.001",
        "--device", "cpu", "--threads", "1",
    ], comparison_dir / "stdout.log")
    print(f"Completed all runs in {output_dir}", flush=True)


if __name__ == "__main__":
    main()
