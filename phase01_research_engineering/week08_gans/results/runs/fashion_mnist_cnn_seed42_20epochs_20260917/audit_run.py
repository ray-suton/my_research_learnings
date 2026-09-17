"""Audit the completed Fashion-MNIST CNN GAN run and create review figures.

This script is intentionally run-local. It validates the frozen run artifacts
before replaying saved checkpoints; it does not train, download data, or write
checkpoints.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import struct
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "week08_gan_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import datasets


RUN_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(__file__).resolve().parents[3]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from loader import get_dataloaders
from run_baseline import build_models, grid_array, sample_generator, save_grid, sha256


EXPECTED_RUN_NAME = "fashion_mnist_cnn_seed42_20epochs_20260917"
EXPECTED_CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
RAW_FILES = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
]


def load_json(path):
    return json.loads(path.read_text())


def fail(message):
    raise RuntimeError(message)


def require(condition, message):
    if not condition:
        fail(message)


def finite_tensors(value, label):
    if torch.is_tensor(value):
        require(torch.isfinite(value).all().item(), f"non-finite tensor in {label}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            finite_tensors(item, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            finite_tensors(item, f"{label}[{index}]")


def png_size(path):
    with path.open("rb") as handle:
        header = handle.read(24)
    require(header.startswith(b"\x89PNG\r\n\x1a\n"), f"not a PNG: {path}")
    require(header[12:16] == b"IHDR", f"missing IHDR: {path}")
    width, height = struct.unpack(">II", header[16:24])
    require(width > 0 and height > 0, f"empty PNG dimensions: {path}")
    return width, height


def validate_png(path):
    size = png_size(path)
    require(path.stat().st_size > 1000, f"unexpectedly small PNG: {path}")
    with Image.open(path) as image:
        require(image.format == "PNG", f"PIL did not identify PNG format: {path}")
        image.verify()
    return size


def current_versions():
    return {
        name: importlib.metadata.version(name)
        for name in ["torch", "torchvision", "numpy", "matplotlib"]
    }


def validate_completed_run(run_dir, config, record):
    require(run_dir.name == EXPECTED_RUN_NAME, "unexpected run directory name")
    require(record.get("status") == "completed", "run is not completed")
    require(config["dataset"] == "fashion-mnist", "dataset is not Fashion-MNIST")
    require(config["dataset_class"] == "FashionMNIST", "dataset class mismatch")
    require(config["dataset_name"] == "Fashion-MNIST", "dataset display name mismatch")
    require(config["architecture"] == "cnn", "architecture is not CNN")
    require(config["device"] == "cpu", "audit expects the recorded CPU run")
    require(config["epochs"] == 20, "run was not configured for 20 epochs")
    require(config["selected_epoch"] == 20, "selected epoch is not 20")
    require(config["seed"] == 42, "seed is not 42")
    require(config["noise_dim"] == 64, "noise dimension mismatch")
    require(config["batch_size"] == 128, "batch size mismatch")
    require(config["threads"] == 2, "thread count mismatch")
    require(config["num_workers"] == 0, "num_workers mismatch")
    require(config["chunk_epochs"] == 5, "chunk size mismatch")
    require(config["generator_class"] == "CnnGenerator", "generator class mismatch")
    require(config["discriminator_class"] == "CNNDiscriminator", "discriminator class mismatch")
    require(config["test_images_used"] is False, "test images should not be used")
    require(record.get("test_images_used") is False, "record says test images were used")
    require(config["class_names"] == EXPECTED_CLASS_NAMES, "recorded class names changed")
    require(datasets.FashionMNIST.classes == EXPECTED_CLASS_NAMES, "torchvision class names changed")
    require(config["counts"] == {"train": 55000, "validation": 5000, "test": 10000}, "split counts changed")
    require(len(record["history"]) == 20, "history does not contain 20 epochs")
    require(len(record["checkpoints"]) == 20, "checkpoint record does not contain 20 epochs")
    require([row["epoch"] for row in record["history"]] == list(range(1, 21)), "history epoch sequence mismatch")
    require([row["epoch"] for row in record["checkpoints"]] == list(range(1, 21)), "checkpoint epoch sequence mismatch")
    require(sum(row["samples"] for row in record["history"]) == 1_100_000, "total sample count mismatch")
    require(all(row["batches"] == 430 for row in record["history"]), "batch count per epoch mismatch")
    require(sum(row["batches"] for row in record["history"]) == 8600, "total update count mismatch")
    validate_history_metrics(record["history"])

    expected_ranges = [
        (1, 5, False, "paused"),
        (6, 10, True, "paused"),
        (11, 15, True, "paused"),
        (16, 20, True, "completed"),
    ]
    invocations = record["invocations"]
    require(len(invocations) == 4, "expected four invocation ranges")
    for invocation, (first_epoch, last_epoch, resume, status) in zip(invocations, expected_ranges):
        require(invocation["first_epoch"] == first_epoch, f"bad first epoch for invocation {first_epoch}")
        require(invocation["planned_last_epoch"] == last_epoch, f"bad last epoch for invocation {first_epoch}")
        require(invocation["resume"] is resume, f"bad resume flag for invocation {first_epoch}")
        require(invocation["status"] == status, f"bad status for invocation {first_epoch}")


def validate_history_metrics(history):
    metric_names = [
        "d_real_loss", "d_fake_loss", "d_loss", "g_loss", "d_real_score",
        "d_fake_score_before", "d_fake_score_after", "fixed_initial_fake",
        "fixed_train_real", "fixed_validation_real", "training_seconds",
    ]
    for row in history:
        epoch = row["epoch"]
        for name in metric_names:
            value = row[name]
            require(isinstance(value, (int, float)), f"{name} is not numeric at epoch {epoch}")
            require(torch.isfinite(torch.tensor(float(value))).item(), f"{name} is not finite at epoch {epoch}")
        require(
            # Batch terms are added in float32 before sample-weighted aggregation.
            # Match the 1e-6 tolerance used by the training contract tests.
            abs(row["d_loss"] - (row["d_real_loss"] + row["d_fake_loss"])) <= 1e-6,
            f"d_loss is not d_real_loss + d_fake_loss at epoch {epoch}",
        )
        for name in [
            "d_real_score", "d_fake_score_before", "d_fake_score_after",
            "fixed_initial_fake", "fixed_train_real", "fixed_validation_real",
        ]:
            require(0.0 <= row[name] <= 1.0, f"{name} out of probability range at epoch {epoch}")
        require(row["training_seconds"] > 0.0, f"non-positive training time at epoch {epoch}")


def validate_source_identity(config):
    hashes = config["source_hashes"]
    require(len(hashes) == 7, "expected seven source hashes")
    for relative, expected in sorted(hashes.items()):
        current = PROJECT_DIR / relative
        snapshot = RUN_DIR / "sources" / Path(relative).name
        require(current.exists(), f"current source missing: {relative}")
        require(snapshot.exists(), f"source snapshot missing: {relative}")
        require(sha256(current) == expected, f"current source hash mismatch: {relative}")
        require(sha256(snapshot) == expected, f"snapshot source hash mismatch: {relative}")


def validate_environment(config):
    require(sys.version == config["python"], "Python version differs from recorded run")
    require(current_versions() == config["versions"], "package versions differ from recorded run")


def validate_dataset_identity(config):
    train_loader, val_loader, test_loader = get_dataloaders(
        config["batch_size"],
        config["seed"],
        Path(config["data_root"]),
        config["num_workers"],
        dataset=config["dataset"],
    )
    dataset = train_loader.dataset.dataset
    require(type(dataset).__name__ == "FashionMNIST", "loaded dataset class mismatch")
    require(dataset.classes == EXPECTED_CLASS_NAMES, "loaded class names mismatch")
    require(
        {"train": len(train_loader.dataset), "validation": len(val_loader.dataset), "test": len(test_loader.dataset)}
        == config["counts"],
        "loaded split counts mismatch",
    )
    train_indices = train_loader.dataset.indices
    val_indices = val_loader.dataset.indices
    require(not set(train_indices).intersection(val_indices), "train/validation split overlaps")
    require(
        hashlib.sha256(json.dumps(train_indices).encode()).hexdigest() == config["train_indices_sha256"],
        "train split hash mismatch",
    )
    require(
        hashlib.sha256(json.dumps(val_indices).encode()).hexdigest() == config["validation_indices_sha256"],
        "validation split hash mismatch",
    )
    raw_folder = Path(dataset.raw_folder)
    raw_hashes = {name: sha256(raw_folder / name) for name in RAW_FILES}
    require(raw_hashes == config["raw_data_sha256"], "raw data hash mismatch")
    return train_loader, val_loader, test_loader


def normalize_class_row(row):
    return {
        "split": row["split"],
        "count": int(row["count"]),
        "mean_real_probability": float(row["mean_real_probability"]),
        "mean_real_bce": float(row["mean_real_bce"]),
        "fraction_classified_real": float(row["fraction_classified_real"]),
        "class_id": int(row["class_id"]),
        "class_name": row["class_name"],
    }


def class_counts(split):
    counts = [0] * 10
    for _, label in split:
        counts[int(label)] += 1
    return counts


def validate_class_audit(record, train_loader, val_loader):
    json_rows = [normalize_class_row(row) for row in record["discriminator_by_class"]]
    csv_path = RUN_DIR / "discriminator_by_class.csv"
    require(csv_path.exists(), "class audit CSV missing")
    with csv_path.open(newline="") as handle:
        csv_rows = [normalize_class_row(row) for row in csv.DictReader(handle)]
    require(csv_rows == json_rows, "class audit CSV and metrics JSON differ")
    require(len(json_rows) == 20, "class audit should have train and validation rows for 10 classes")
    expected_counts = {"train": class_counts(train_loader.dataset), "validation": class_counts(val_loader.dataset)}
    seen = set()
    for row in json_rows:
        require(row["class_name"] == EXPECTED_CLASS_NAMES[row["class_id"]], "class name/id mismatch")
        require(row["count"] == expected_counts[row["split"]][row["class_id"]], "class count mismatch")
        require(0.0 <= row["mean_real_probability"] <= 1.0, "class probability out of range")
        require(row["mean_real_bce"] >= 0.0, "class BCE is negative")
        require(0.0 <= row["fraction_classified_real"] <= 1.0, "class acceptance fraction out of range")
        seen.add((row["split"], row["class_id"]))
    require(seen == {(split, class_id) for split in ["train", "validation"] for class_id in range(10)}, "class audit rows incomplete")


def validate_checkpoint(epoch, checkpoint_info, config, expected_batches):
    path = Path(checkpoint_info["path"])
    require(path.exists(), f"checkpoint missing for epoch {epoch}")
    require(sha256(path) == checkpoint_info["sha256"], f"checkpoint hash mismatch for epoch {epoch}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    require(state["epoch"] == epoch, f"checkpoint epoch mismatch for epoch {epoch}")
    require(state["config"] == config, f"checkpoint config mismatch for epoch {epoch}")
    finite_tensors(state["generator"], f"epoch {epoch} generator")
    finite_tensors(state["discriminator"], f"epoch {epoch} discriminator")
    finite_tensors(state["g_optimizer"], f"epoch {epoch} g_optimizer")
    finite_tensors(state["d_optimizer"], f"epoch {epoch} d_optimizer")
    expected_step = float(epoch * expected_batches)
    for optimizer_name in ["g_optimizer", "d_optimizer"]:
        steps = []
        for slot in state[optimizer_name]["state"].values():
            step = slot.get("step")
            require(step is not None, f"missing Adam step in {optimizer_name} epoch {epoch}")
            steps.append(float(step.item() if torch.is_tensor(step) else step))
        require(steps and all(step == expected_step for step in steps), f"Adam step mismatch in {optimizer_name} epoch {epoch}")
    return state


def validate_checkpoints(record, config):
    by_epoch = {}
    for item in record["checkpoints"]:
        epoch = item["epoch"]
        by_epoch[epoch] = validate_checkpoint(epoch, item, config, expected_batches=430)
    return by_epoch


def load_generator_from_state(state, config):
    generator, _ = build_models(config["architecture"], config["noise_dim"])
    generator.load_state_dict(state["generator"], strict=True)
    return generator.cpu()


def replay_and_compare_pngs(states, config):
    torch.set_num_threads(2)
    fixed_noise = torch.randn(
        64, config["noise_dim"], generator=torch.Generator().manual_seed(config["monitoring_noise_seed"])
    )
    final_noise = torch.randn(
        64, config["noise_dim"], generator=torch.Generator().manual_seed(config["final_noise_seed"])
    )
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        for epoch in [5, 10, 20]:
            generator = load_generator_from_state(states[epoch], config)
            images = sample_generator(generator, fixed_noise)
            replay = temporary / f"fixed_noise_epoch_{epoch:02d}.png"
            save_grid(images, replay, f"CNN GAN after epoch {epoch}: fixed noise seed 123")
            original = RUN_DIR / f"fixed_noise_epoch_{epoch:02d}.png"
            require(replay.read_bytes() == original.read_bytes(), f"fixed-noise replay mismatch at epoch {epoch}")

        generator = load_generator_from_state(states[20], config)
        final_images = sample_generator(generator, final_noise)
        replay = temporary / "final_noise_seed456.png"
        save_grid(final_images, replay, "Final epoch 20: independent noise seed 456")
        original = RUN_DIR / "final_noise_seed456.png"
        require(replay.read_bytes() == original.read_bytes(), "final-noise replay mismatch")


def save_progression(states, config, path):
    fixed_noise = torch.randn(
        64, config["noise_dim"], generator=torch.Generator().manual_seed(config["monitoring_noise_seed"])
    )
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    for ax, epoch in zip(axes, [5, 10, 20]):
        generator = load_generator_from_state(states[epoch], config)
        images = sample_generator(generator, fixed_noise)
        ax.imshow(grid_array(images).numpy(), interpolation="nearest")
        ax.set_title(f"Epoch {epoch} | fixed noise seed 123")
        ax.axis("off")
    fig.suptitle("Fashion-MNIST CNN GAN progression: all 64 fixed-noise samples per panel")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    validate_png(path)


def save_loss_plot(history, path):
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(epochs, [row["d_loss"] for row in history], marker="o", label="D loss")
    axes[0].plot(epochs, [row["d_real_loss"] for row in history], marker=".", label="D real")
    axes[0].plot(epochs, [row["d_fake_loss"] for row in history], marker=".", label="D fake")
    axes[0].set_title("Discriminator")
    axes[0].set_ylabel("Sample-weighted BCE")
    axes[1].plot(epochs, [row["g_loss"] for row in history], marker="o", color="tab:orange", label="G loss")
    axes[1].set_title("Generator")
    axes[1].set_ylabel("Sample-weighted BCE")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Fashion-MNIST CNN GAN recorded training losses")
    fig.text(
        0.5, 0.01,
        "Loss tracks the adversarial training game; it is not a direct image-quality or class-coverage metric.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    validate_png(path)


def validate_existing_pngs(record):
    for epoch in [0, *range(1, 21)]:
        validate_png(RUN_DIR / f"fixed_noise_epoch_{epoch:02d}.png")
    for name in [
        "real_validation.png", "training_curves.png", "final_noise_seed456.png",
        "real_vs_generated.png", "real_vs_generated_single.png",
    ]:
        validate_png(RUN_DIR / name)
    for checkpoint in record["checkpoints"]:
        validate_png(RUN_DIR / f"fixed_noise_epoch_{checkpoint['epoch']:02d}.png")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    global RUN_DIR
    RUN_DIR = args.run_dir.resolve()
    config = load_json(RUN_DIR / "config.json")
    record = load_json(RUN_DIR / "metrics.json")

    validate_completed_run(RUN_DIR, config, record)
    validate_source_identity(config)
    validate_environment(config)
    train_loader, val_loader, test_loader = validate_dataset_identity(config)
    require(len(test_loader.dataset) == 10000, "test split size changed")
    validate_class_audit(record, train_loader, val_loader)
    states = validate_checkpoints(record, config)
    validate_existing_pngs(record)
    replay_and_compare_pngs(states, config)
    save_progression(states, config, RUN_DIR / "audit_epoch_05_10_20_progression.png")
    save_loss_plot(record["history"], RUN_DIR / "audit_d_g_losses.png")
    print("audit passed: completed Fashion-MNIST CNN run, 20 epochs, 8600 recorded updates")
    print("wrote audit_epoch_05_10_20_progression.png and audit_d_g_losses.png")


if __name__ == "__main__":
    main()
