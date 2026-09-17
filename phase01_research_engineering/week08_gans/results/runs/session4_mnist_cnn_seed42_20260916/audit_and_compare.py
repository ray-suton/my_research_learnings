"""Audit this completed CNN run and plot full five-epoch comparison panels."""

import ast
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import torch
from PIL import Image


RUN = Path(__file__).resolve().parent
PROJECT = RUN.parents[2]
BASELINE = RUN.parent / "session3_mnist_mlp_seed42_20260915"
sys.path.insert(0, str(PROJECT))

from loader import get_dataloaders
from run_baseline import build_models, grid_array, plt, sample_generator, save_grid


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    config, metrics = read_json(RUN / "config.json"), read_json(RUN / "metrics.json")
    original = read_json(BASELINE / "config.json")
    original_metrics = read_json(BASELINE / "metrics.json")
    assert metrics["status"] == "completed"
    assert config["architecture"] == "cnn"
    assert config["generator_class"] == "CnnGenerator"
    assert config["discriminator_class"] == "CNNDiscriminator"
    assert config["selected_epoch"] == metrics["selected_epoch"] == 5
    assert not config["test_images_used"] and not metrics["test_images_used"]
    assert len(metrics["history"]) == len(metrics["checkpoints"]) == 5

    keys = ["epochs", "batch_size", "seed", "noise_dim", "lr", "device",
            "threads", "num_workers", "adam_betas", "counts", "data_root",
            "train_indices_sha256", "validation_indices_sha256",
            "monitoring_noise_seed", "final_noise_seed", "selected_epoch",
            "selection_rule", "test_images_used"]
    assert all(config[key] == original[key] for key in keys)
    print(f"PASS: {len(keys)} baseline settings and split identities match")

    for name, expected in config["source_hashes"].items():
        assert digest(PROJECT / name) == expected, name
        assert digest(RUN / "sources" / Path(name).name) == expected, name
    for name in ["train.py", "loader.py"]:
        assert digest(PROJECT / name) == original["source_hashes"][name]
    for name in ["Generator", "Discriminator"]:
        trees = [ast.parse(p.read_text()) for p in [
            PROJECT / "model.py", BASELINE / "sources/model.py",
        ]]
        definitions = [next(n for n in tree.body
                            if isinstance(n, ast.ClassDef) and n.name == name)
                       for tree in trees]
        assert ast.dump(definitions[0]) == ast.dump(definitions[1])
    print("PASS: all five source snapshots match; original MLP definitions and update/data code preserved")

    for epoch, (row, checkpoint) in enumerate(zip(metrics["history"], metrics["checkpoints"]), 1):
        assert row["epoch"] == checkpoint["epoch"] == epoch
        assert row["samples"] == 55000 and row["batches"] == 430
        assert abs(row["d_loss"] - row["d_real_loss"] - row["d_fake_loss"]) < 1e-6
        assert digest(Path(checkpoint["path"])) == checkpoint["sha256"]
    print("PASS: 5 epochs, 2,150 paired updates, finite recorded losses, 5 checkpoint hashes")
    for row in metrics["history"]:
        assert all(torch.isfinite(torch.tensor(value)) for value in row.values())

    with (RUN / "discriminator_by_digit.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 20
    for actual, expected in zip(rows, metrics["discriminator_by_digit"]):
        for key, value in expected.items():
            if key == "split":
                assert actual[key] == value
            else:
                assert abs(float(actual[key]) - value) < 1e-12
    for split, count in [("train", 55000), ("validation", 5000)]:
        assert sum(row["count"] for row in metrics["discriminator_by_digit"]
                   if row["split"] == split) == count
    pngs = list(RUN.glob("*.png"))
    assert len(pngs) >= 11
    for path in pngs:
        with Image.open(path) as image:
            image.verify()
    print(f"PASS: discriminator CSV/JSON counts agree; {len(pngs)} PNG files verify")

    cnn, _ = build_models("cnn", config["noise_dim"])
    mlp, _ = build_models("mlp", original["noise_dim"])
    cnn_checkpoint = torch.load(metrics["checkpoints"][-1]["path"], map_location="cpu", weights_only=True)
    mlp_checkpoint = torch.load(original_metrics["checkpoints"][-1]["path"], map_location="cpu", weights_only=True)
    assert cnn_checkpoint["epoch"] == mlp_checkpoint["epoch"] == 5
    assert cnn_checkpoint["config"] == config
    cnn.load_state_dict(cnn_checkpoint["generator"])
    mlp.load_state_dict(mlp_checkpoint["generator"])
    noise = torch.randn(64, config["noise_dim"], generator=torch.Generator().manual_seed(456))
    cnn_images, mlp_images = sample_generator(cnn, noise), sample_generator(mlp, noise)
    fixed_noise = torch.randn(64, config["noise_dim"], generator=torch.Generator().manual_seed(123))
    with tempfile.TemporaryDirectory(prefix="week08-cnn-replay-") as tmp:
        replay = Path(tmp)
        for name, images, title in [
            ("fixed_noise_epoch_05.png", sample_generator(cnn, fixed_noise),
             "CNN GAN after epoch 5: fixed noise seed 123"),
            ("final_noise_seed456.png", cnn_images,
             "Final epoch 5: independent noise seed 456"),
        ]:
            save_grid(images, replay / name, title)
            assert digest(replay / name) == digest(RUN / name), name
        save_grid(mlp_images, replay / "mlp_final.png", "Final epoch 5: independent noise seed 456")
        assert digest(replay / "mlp_final.png") == digest(BASELINE / "final_noise_seed456.png")
    print("PASS: final checkpoint exactly reproduces both CNN panels and the original MLP final panel")

    _, validation, _ = get_dataloaders(128, 42, config["data_root"], 0)
    real = torch.stack([validation.dataset[i][0] for i in range(64)])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
    for axis, images, title in zip(axes, [real, mlp_images, cnn_images], [
        "Real MNIST | fixed validation examples",
        "MLP GAN | 5 epochs | final noise seed 456",
        "CNN GAN | 5 epochs | final noise seed 456",
    ]):
        axis.imshow(grid_array(images).numpy(), interpolation="nearest")
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle("Full sample panels at the same 2,150-update budget")
    fig.text(0.5, 0.025, "Independent examples; latent positions do not imply matched digit identities.", ha="center")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(RUN / "mlp_vs_cnn_epoch05.png", dpi=150)
    plt.close(fig)
    with Image.open(RUN / "mlp_vs_cnn_epoch05.png") as image:
        image.verify()
    print("Saved: mlp_vs_cnn_epoch05.png")
    print("Final CNN losses:", {key: metrics["history"][-1][key] for key in ["d_loss", "g_loss"]})
    print("Training seconds:", sum(row["training_seconds"] for row in metrics["history"]))
    print("Total recorded seconds:", metrics["total_seconds"])


if __name__ == "__main__":
    main()
