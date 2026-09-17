"""Verify the extended CNN trajectory and compare both GANs at 20 epochs."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

from PIL import Image
import torch


RUN = Path(__file__).resolve().parent
PROJECT = RUN.parents[2]
CNN5 = RUN.parent / "session4_mnist_cnn_seed42_20260916"
MLP20 = RUN.parent / "session3_mnist_mlp_seed42_20epochs_20260915"
sys.path.insert(0, str(PROJECT))

from loader import get_dataloaders
from run_baseline import build_models, grid_array, plt, sample_generator, save_grid


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint(record, epoch):
    item = next(item for item in record["checkpoints"] if item["epoch"] == epoch)
    path = Path(item["path"])
    assert digest(path) == item["sha256"], path
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["epoch"] == epoch
    for key in ["generator", "discriminator"]:
        assert all(torch.isfinite(value).all() for value in state[key].values())
    return state


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal(a, b)
    else:
        assert left == right


def verify_prefix():
    old, new = read_json(CNN5 / "metrics.json"), read_json(RUN / "metrics.json")
    assert len(new["history"]) >= 5
    assert old["initial_fixed_scores"] == new["initial_fixed_scores"]
    for a, b in zip(old["history"], new["history"][:5]):
        assert {k: v for k, v in a.items() if k != "training_seconds"} == {
            k: v for k, v in b.items() if k != "training_seconds"}
    for epoch in range(6):
        name = f"fixed_noise_epoch_{epoch:02d}.png"
        assert digest(CNN5 / name) == digest(RUN / name), name
    a, b = checkpoint(old, 5), checkpoint(new, 5)
    for key in ["generator", "discriminator", "g_optimizer", "d_optimizer",
                "torch_rng_state", "loader_rng_state"]:
        assert_equal(a[key], b[key])
    print("PASS: CNN epochs 1–5 non-runtime metrics and epoch-0–5 grids match the original run")
    print("PASS: CNN epoch-5 model, optimizer, training RNG and loader RNG states match exactly")


def samples(architecture, record, epoch, noise):
    generator, _ = build_models(architecture, noise.shape[1])
    state = checkpoint(record, epoch)
    assert state["epoch"] == epoch
    generator.load_state_dict(state["generator"])
    return sample_generator(generator, noise)


def main(prefix_only=False):
    torch.set_num_threads(2)
    verify_prefix()
    if prefix_only:
        return
    config, metrics = read_json(RUN / "config.json"), read_json(RUN / "metrics.json")
    old_config = read_json(CNN5 / "config.json")
    mlp_config, mlp_metrics = read_json(MLP20 / "config.json"), read_json(MLP20 / "metrics.json")
    excluded = {"epochs", "selected_epoch", "command", "out_dir", "checkpoint_dir", "started_at"}
    assert all(old_config[k] == config[k] for k in old_config if k not in excluded)
    assert config["architecture"] == "cnn" and config["epochs"] == 20
    if "resumed_from_epoch" in metrics:
        continuation = read_json(RUN / "continuation.json")
        assert digest(RUN / "continue_run.py") == continuation["script_sha256"]
        assert continuation["restored_epoch"] == metrics["resumed_from_epoch"] == 12
        state = checkpoint(metrics, 20)
        assert state["config"] == config
        for key in ["g_optimizer", "d_optimizer"]:
            assert all(value["step"].item() == 8600 for value in state[key]["state"].values())
        print("PASS: continuation helper identity, original config, and final Adam step counts (8,600 each)")
    for key in ["epochs", "seed", "batch_size", "noise_dim", "lr", "device", "threads",
                "num_workers", "adam_betas", "noise_scale", "counts", "data_root",
                "train_indices_sha256", "validation_indices_sha256", "monitoring_noise_seed",
                "final_noise_seed", "selected_epoch", "selection_rule", "test_images_used", "versions"]:
        assert config[key] == mlp_config[key], key
    for name, expected in config["source_hashes"].items():
        assert digest(PROJECT / name) == expected, name
        assert digest(RUN / "sources" / Path(name).name) == expected, name
    print("PASS: CNN source/settings unchanged except declared budget; matched MLP experiment settings")

    for label, record in [("CNN", metrics), ("MLP", mlp_metrics)]:
        assert record["status"] == "completed" and record["selected_epoch"] == 20
        assert len(record["history"]) == len(record["checkpoints"]) == 20
        assert not record["test_images_used"]
        for epoch, row in enumerate(record["history"], 1):
            assert row["epoch"] == epoch and row["samples"] == 55000 and row["batches"] == 430
            assert all(math.isfinite(value) for value in row.values())
            assert abs(row["d_loss"] - row["d_real_loss"] - row["d_fake_loss"]) < 1e-6
        for item in record["checkpoints"]:
            assert digest(Path(item["path"])) == item["sha256"]
        print(f"PASS: {label} 20 epochs, 8,600 paired updates, finite metrics, all 20 checkpoint hashes")

    with (RUN / "discriminator_by_digit.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(metrics["discriminator_by_digit"]) == 20
    for actual, expected in zip(rows, metrics["discriminator_by_digit"]):
        for key, value in expected.items():
            assert actual[key] == value if key == "split" else abs(float(actual[key]) - value) < 1e-12
    for split, count in [("train", 55000), ("validation", 5000)]:
        assert sum(r["count"] for r in metrics["discriminator_by_digit"] if r["split"] == split) == count
    print("PASS: CNN discriminator CSV/JSON records and real-image audit counts agree")

    fixed = torch.randn(64, 64, generator=torch.Generator().manual_seed(123))
    final = torch.randn(64, 64, generator=torch.Generator().manual_seed(456))
    panels = {}
    finals = {}
    with tempfile.TemporaryDirectory(prefix="week08-cnn20-replay-") as tmp:
        replay = Path(tmp)
        for architecture, record, folder in [("mlp", mlp_metrics, MLP20), ("cnn", metrics, RUN)]:
            for epoch in [5, 10, 20]:
                images = samples(architecture, record, epoch, fixed)
                panels[architecture, epoch] = images
                name = f"fixed_noise_epoch_{epoch:02d}.png"
                save_grid(images, replay / name,
                          f"{architecture.upper()} GAN after epoch {epoch}: fixed noise seed 123")
                assert digest(replay / name) == digest(folder / name), (architecture, epoch)
            images = samples(architecture, record, 20, final)
            finals[architecture] = images
            save_grid(images, replay / "final.png", "Final epoch 20: independent noise seed 456")
            assert digest(replay / "final.png") == digest(folder / "final_noise_seed456.png")
    print("PASS: exact replay of both architectures at epochs 5/10/20 and both independent final panels")

    fig, axes = plt.subplots(2, 3, figsize=(15, 12))
    for row, architecture in enumerate(["mlp", "cnn"]):
        for column, epoch in enumerate([5, 10, 20]):
            axes[row, column].imshow(grid_array(panels[architecture, epoch]).numpy(), interpolation="nearest")
            axes[row, column].set_title(f"{architecture.upper()} | epoch {epoch}")
            axes[row, column].axis("off")
    fig.suptitle("Full fixed-noise progression | noise seed 123")
    fig.text(0.5, 0.015, "Positions track each model over time; digit identities are not paired across models.", ha="center")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.93,
                        wspace=0.06, hspace=0.18)
    fig.savefig(RUN / "epoch_05_10_20_comparison.png", dpi=150)
    plt.close(fig)

    _, validation, _ = get_dataloaders(128, 42, config["data_root"], 0)
    real = torch.stack([validation.dataset[i][0] for i in range(64)])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
    for axis, images, title in zip(axes, [real, finals["mlp"], finals["cnn"]], [
        "Real MNIST | validation examples", "MLP GAN | epoch 20", "CNN GAN | epoch 20",
    ]):
        axis.imshow(grid_array(images).numpy(), interpolation="nearest")
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle("Full panels at 8,600 paired updates | final noise seed 456")
    fig.text(0.5, 0.025, "Independent examples; equal noise positions do not imply matched digit identities.", ha="center")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(RUN / "mlp_vs_cnn_epoch20.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for label, record, color in [("MLP", mlp_metrics, "#0072B2"), ("CNN", metrics, "#D55E00")]:
        epochs = [r["epoch"] for r in record["history"]]
        for axis, key in zip(axes, ["d_loss", "g_loss"]):
            axis.plot(epochs, [r[key] for r in record["history"]], marker="o", markersize=4,
                      label=label, color=color)
    for axis, title in zip(axes, ["Discriminator: real BCE + fake BCE", "Generator: BCE against current discriminator"]):
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Sample-weighted epoch loss")
        axis.set_xticks([1, 5, 10, 15, 20])
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Twenty-epoch GAN training diagnostics | seed 42")
    fig.text(0.5, 0.025, "Each generator faces a different changing discriminator; these losses do not rank image quality.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(RUN / "mlp_vs_cnn_losses_20epochs.png", dpi=150)
    plt.close(fig)

    pngs = list(RUN.glob("*.png"))
    assert len(pngs) == 29
    for path in pngs:
        with Image.open(path) as image:
            image.verify()
    print(f"PASS: {len(pngs)} PNGs verified, including three comparison figures")
    print("CNN final losses:", metrics["history"][-1]["d_loss"], metrics["history"][-1]["g_loss"])
    print("CNN training seconds:", sum(row["training_seconds"] for row in metrics["history"]))
    print("MLP reused from 2026-09-15; no new MLP training was performed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-only", action="store_true")
    main(parser.parse_args().prefix_only)
