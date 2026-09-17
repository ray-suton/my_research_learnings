"""Run an MNIST or Fashion-MNIST GAN with recorded, resumable CPU segments."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import resource
import shutil
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from loader import get_dataloaders, resolve_data_root
from model import CNNDiscriminator, CnnGenerator, Discriminator, Generator
from train import train_one_epoch


def build_models(architecture, noise_dim):
    """Construct one declared model pair without changing the MNIST contract."""
    if architecture == "mlp":
        return Generator(noise_dim, 784), Discriminator(28)
    if architecture == "cnn":
        return CnnGenerator(noise_dim, 784), CNNDiscriminator(28)
    raise ValueError(f"Unknown architecture: {architecture}")


def sample_generator(generator, noise):
    """Sample without changing training modes, parameters, or training RNG."""
    modes = {module: module.training for module in generator.modules()}
    devices = [noise.device.index] if noise.is_cuda else []
    try:
        generator.eval()
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            return generator(noise).detach().cpu()
    finally:
        for module, mode in modes.items():
            module.training = mode


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grid_array(images):
    pixels = ((images.detach().cpu() + 1) / 2).clamp(0, 1)
    return make_grid(pixels, nrow=8, padding=2, pad_value=1).permute(1, 2, 0)


def save_grid(images, path, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid_array(images).numpy(), interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_comparisons(real_images, generated, real_digit, out_dir, epoch,
                     dataset_name="MNIST", class_name=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax, images, title in zip(axes, [real_images, generated], [
        f"Real {dataset_name} | fixed validation examples",
        f"GAN samples | epoch {epoch}, final noise seed 456",
    ]):
        ax.imshow(grid_array(images).numpy(), interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Real versus generated: independent images, not reconstructions")
    fig.tight_layout()
    fig.savefig(out_dir / "real_vs_generated.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.7))
    for ax, image, title in zip(axes, [real_images[0], generated[0]], [
        f"Real {dataset_name} | " + (f"digit {int(real_digit)}" if class_name is None else class_name),
        f"GAN-generated | epoch {epoch}",
    ]):
        ax.imshow(((image[0] + 1) / 2).numpy(), cmap="gray",
                  vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("First examples from predeclared panels; no pairing implied")
    fig.tight_layout()
    fig.savefig(out_dir / "real_vs_generated_single.png", dpi=180)
    plt.close(fig)


def save_curves(history, path):
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for name in ["d_real_loss", "d_fake_loss", "d_loss", "g_loss"]:
        axes[0].plot(epochs, [row[name] for row in history], marker="o",
                     label=name)
    axes[0].set_title("Sample-weighted training losses")
    for name in ["d_real_score", "d_fake_score_before", "d_fake_score_after"]:
        axes[1].plot(epochs, [row[name] for row in history], marker="o",
                     label=name)
    axes[1].set_title("Scores within alternating training steps")
    for name in ["fixed_train_real", "fixed_validation_real", "fixed_initial_fake"]:
        axes[2].plot(epochs, [row[name] for row in history], marker="o",
                     label=name)
    axes[2].set_title("Post-epoch D scores on unchanged images")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes[1:]:
        ax.set_ylim(0, 1)
        ax.set_ylabel("Mean probability of real")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fixed_scores(discriminator, references, device):
    mode = discriminator.training
    try:
        discriminator.eval()
        with torch.no_grad():
            return {
                name: discriminator(images.to(device)).mean().item()
                for name, images in references.items()
            }
    finally:
        discriminator.train(mode)


def score_real_by_digit(discriminator, dataset, split, device):
    """Audit D on train/validation real images, without touching the test set."""
    loader = DataLoader(
        dataset, batch_size=256, shuffle=False, num_workers=0,
        generator=torch.Generator().manual_seed(987),
    )
    counts = torch.zeros(10, dtype=torch.long)
    scores = torch.zeros(10, dtype=torch.float64)
    losses = torch.zeros(10, dtype=torch.float64)
    accepted = torch.zeros(10, dtype=torch.long)
    mode = discriminator.training
    try:
        discriminator.eval()
        with torch.no_grad():
            for images, labels in loader:
                prediction = discriminator(images.to(device)).flatten().cpu()
                bce = torch.nn.functional.binary_cross_entropy(
                    prediction, torch.ones_like(prediction), reduction="none"
                )
                counts += torch.bincount(labels, minlength=10)
                scores.scatter_add_(0, labels, prediction.double())
                losses.scatter_add_(0, labels, bce.double())
                accepted.scatter_add_(0, labels, (prediction >= 0.5).long())
    finally:
        discriminator.train(mode)
    return [{
        "split": split, "digit": digit, "count": counts[digit].item(),
        "mean_real_probability": (scores[digit] / counts[digit]).item(),
        "mean_real_bce": (losses[digit] / counts[digit]).item(),
        "fraction_classified_real": (accepted[digit] / counts[digit]).item(),
    } for digit in range(10)]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--dataset", choices=["mnist", "fashion-mnist"], default="mnist")
    parser.add_argument("--architecture", choices=["mlp", "cnn"], default="mlp",
                        help="Generator/discriminator pair; default preserves the MLP baseline")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--chunk-epochs", type=int,
                        help="Pause after this many epochs per invocation; CPU only")
    parser.add_argument("--resume", action="store_true",
                        help="Continue a paused run using its saved configuration")
    raw_args = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(raw_args)
    if min(args.epochs, args.batch_size, args.noise_dim, args.threads) < 1:
        parser.error("epochs, batch-size, noise-dim, and threads must be positive")
    if args.chunk_epochs is not None and args.chunk_epochs < 1:
        parser.error("chunk-epochs must be positive")
    if args.resume:
        allowed = {"--resume", "--out-dir", "--chunk-epochs"}
        if any(item.startswith("--") and item.split("=", 1)[0] not in allowed for item in raw_args):
            parser.error("resume loads saved settings; use only --resume, --out-dir, and optionally --chunk-epochs")
    return args


def score_real_by_class(discriminator, dataset, split, device, class_names):
    rows = score_real_by_digit(discriminator, dataset, split, device)
    for row in rows:
        class_id = row.pop("digit")
        row.update(class_id=class_id, class_name=class_names[class_id])
    return rows


def data_metadata(train_loader, val_loader, test_loader):
    dataset = train_loader.dataset.dataset
    raw_names = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte",
                 "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]
    return {
        "dataset_class": type(dataset).__name__, "class_names": list(dataset.classes),
        "raw_data_sha256": {name: sha256(Path(dataset.raw_folder) / name) for name in raw_names},
        "counts": {"train": len(train_loader.dataset), "validation": len(val_loader.dataset),
                   "test": len(test_loader.dataset)},
        "train_indices_sha256": hashlib.sha256(json.dumps(train_loader.dataset.indices).encode()).hexdigest(),
        "validation_indices_sha256": hashlib.sha256(json.dumps(val_loader.dataset.indices).encode()).hexdigest(),
    }


def capture_training_state(epoch, generator, discriminator, g_optimizer, d_optimizer, loader, config):
    return {
        "epoch": epoch, "generator": generator.state_dict(), "discriminator": discriminator.state_dict(),
        "g_optimizer": g_optimizer.state_dict(), "d_optimizer": d_optimizer.state_dict(), "config": config,
        "torch_rng_state": torch.get_rng_state(), "loader_rng_state": loader.generator.get_state(),
    }


def restore_training_state(state, generator, discriminator, g_optimizer, d_optimizer, loader):
    generator.load_state_dict(state["generator"], strict=True)
    discriminator.load_state_dict(state["discriminator"], strict=True)
    g_optimizer.load_state_dict(state["g_optimizer"])
    d_optimizer.load_state_dict(state["d_optimizer"])
    torch.set_rng_state(state["torch_rng_state"])
    loader.generator.set_state(state["loader_rng_state"])


def load_paused_run(out_dir, project):
    config = json.loads((out_dir / "config.json").read_text())
    record = json.loads((out_dir / "metrics.json").read_text())
    if record["status"] != "paused" or config["device"] != "cpu":
        raise ValueError("Only cleanly paused CPU runs can be resumed")
    if Path(config["out_dir"]).resolve() != out_dir.resolve():
        raise ValueError("Run directory differs from the saved configuration")
    for name, expected in config["source_hashes"].items():
        if sha256(project / name) != expected:
            raise ValueError(f"Source changed since this run started: {name}")
    if not record["checkpoints"]:
        raise ValueError("Paused run has no completed checkpoint")
    last = record["checkpoints"][-1]
    completed = last["epoch"]
    if not 0 < completed < config["epochs"]:
        raise ValueError("Checkpoint is not before the declared final epoch")
    expected_epochs = list(range(1, completed + 1))
    if [row["epoch"] for row in record["history"]] != expected_epochs or [row["epoch"] for row in record["checkpoints"]] != expected_epochs:
        raise ValueError("Checkpoint/history sequence is inconsistent")
    checkpoint_dir = Path(config["checkpoint_dir"])
    expected_paths = {checkpoint_dir / f"epoch_{epoch:02d}.pt" for epoch in expected_epochs}
    if set(checkpoint_dir.glob("epoch_*.pt")) != expected_paths:
        raise ValueError("Unrecorded or missing checkpoint; inspect before resuming")
    if Path(last["path"]) != checkpoint_dir / f"epoch_{completed:02d}.pt" or sha256(Path(last["path"])) != last["sha256"]:
        raise ValueError("Checkpoint path or hash mismatch")
    state = torch.load(last["path"], map_location="cpu", weights_only=True)
    if state["epoch"] != completed or state["config"] != config:
        raise ValueError("Checkpoint configuration/epoch mismatch")
    return config, record, state


def main():
    cli = parse_args()
    out_dir = cli.out_dir.resolve()
    project = Path(__file__).resolve().parent
    state = None
    config = None
    record = None
    if cli.resume:
        config, record, state = load_paused_run(out_dir, project)
        settings = {key: config.get(key, value) for key, value in vars(cli).items()}
        settings.update(out_dir=out_dir, checkpoint_dir=Path(config["checkpoint_dir"]),
                        data_root=Path(config["data_root"]), resume=True,
                        chunk_epochs=cli.chunk_epochs or config.get("chunk_epochs"))
        args = argparse.Namespace(**settings)
    else:
        args = cli
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if args.chunk_epochs is not None and device.type != "cpu":
        raise ValueError("Segmented pause/resume is currently verified only for CPU runs")
    data_root = resolve_data_root(args.data_root)
    checkpoint_dir = (args.checkpoint_dir or (
        data_root / "checkpoints/week08_gans" / out_dir.name
    )).resolve()
    hashes = {}
    if not cli.resume:
        if out_dir.exists() or checkpoint_dir.exists():
            raise FileExistsError("Use fresh output and checkpoint directories")
        out_dir.mkdir(parents=True)
        checkpoint_dir.mkdir(parents=True)
        source_dir = out_dir / "sources"
        source_dir.mkdir()
        names = ["model.py", "train.py", "loader.py", "run_baseline.py"]
        names.extend(str(path.relative_to(project)) for path in sorted((project / "tests").glob("test_*.py")))
        for name in names:
            path = project / name
            hashes[name] = sha256(path)
            shutil.copy2(path, source_dir / path.name)
    train_loader, val_loader, test_loader = get_dataloaders(
        args.batch_size, args.seed, data_root, args.num_workers, dataset=args.dataset
    )
    identity = data_metadata(train_loader, val_loader, test_loader)
    dataset_name = "Fashion-MNIST" if args.dataset == "fashion-mnist" else "MNIST"
    generator, discriminator = build_models(args.architecture, args.noise_dim)
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    criterion = torch.nn.BCELoss()
    fixed_noise = torch.randn(64, args.noise_dim, generator=torch.Generator().manual_seed(123)).to(device)
    final_noise = torch.randn(64, args.noise_dim, generator=torch.Generator().manual_seed(456)).to(device)
    validation_images = torch.stack([val_loader.dataset[i][0] for i in range(64)])
    training_images = torch.stack([train_loader.dataset[i][0] for i in range(64)])
    initial_fake = sample_generator(generator, fixed_noise)
    references = {"fixed_train_real": training_images, "fixed_validation_real": validation_images,
                  "fixed_initial_fake": initial_fake}
    versions = {name: importlib.metadata.version(name) for name in ["torch", "torchvision", "numpy", "matplotlib"]}
    if cli.resume:
        assert config is not None and record is not None and state is not None
        if any(config[key] != value for key, value in identity.items()):
            raise ValueError("Dataset identity or split changed since this run started")
        if config["versions"] != versions or config["python"] != sys.version:
            raise ValueError("Execution environment changed since this run started")
        if fixed_scores(discriminator, references, device) != record["initial_fixed_scores"]:
            raise ValueError("Original monitoring references or initialization did not reproduce")
        start_epoch = state["epoch"]
    else:
        config = {
            **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            **identity,
            "dataset_name": dataset_name, "device": str(device), "data_root": str(data_root),
            "out_dir": str(out_dir), "checkpoint_dir": str(checkpoint_dir),
            "started_at": datetime.now(timezone.utc).isoformat(), "command": [sys.executable, *sys.argv],
            "versions": versions, "python": sys.version, "source_hashes": hashes,
            "generator_parameters": sum(p.numel() for p in generator.parameters()),
            "discriminator_parameters": sum(p.numel() for p in discriminator.parameters()),
            "generator_class": type(generator).__name__, "discriminator_class": type(discriminator).__name__,
            "generator_architecture": str(generator), "discriminator_architecture": str(discriminator),
            "discriminator_output": "sigmoid probability; mean BCELoss",
            "discriminator_negative_slope": 0.01, "adam_betas": [0.5, 0.999], "noise_scale": 1.0,
            "monitoring_noise_seed": 123, "final_noise_seed": 456, "selected_epoch": args.epochs,
            "selection_rule": "predeclared final epoch; no loss-based selection", "test_images_used": False,
            "timing_scope": "sum of active training/artifact/audit segments; excludes startup and gaps",
        }
        write_json(out_dir / "config.json", config)
        save_grid(validation_images, out_dir / "real_validation.png", f"Real {dataset_name}: 64 fixed validation examples")
        save_grid(initial_fake, out_dir / "fixed_noise_epoch_00.png",
                  f"Untrained {args.architecture.upper()} GAN: fixed noise seed 123")
        record = {"status": "running", "history": [], "checkpoints": [], "invocations": [],
                  "initial_fixed_scores": fixed_scores(discriminator, references, device), "test_images_used": False}
        start_epoch = 0
    assert config is not None and record is not None
    last_epoch = min(args.epochs, start_epoch + args.chunk_epochs) if args.chunk_epochs else args.epochs
    invocation = {"command": [sys.executable, *sys.argv], "started_at": datetime.now(timezone.utc).isoformat(),
                  "first_epoch": start_epoch + 1, "planned_last_epoch": last_epoch, "resume": cli.resume}
    record["invocations"].append(invocation)
    record["status"] = "running"
    write_json(out_dir / "metrics.json", record)
    if state is not None:
        restore_training_state(state, generator, discriminator, g_optimizer, d_optimizer, train_loader)
    start = time.perf_counter()
    print(f"{dataset_name} {config['counts']} | architecture={args.architecture} | device={device} | seed={args.seed}", flush=True)
    print(f"Generator {config['generator_parameters']} parameters; discriminator {config['discriminator_parameters']}", flush=True)
    print(f"Executing epochs {start_epoch + 1}–{last_epoch} of declared {args.epochs}", flush=True)
    try:
        for epoch in range(start_epoch + 1, last_epoch + 1):
            epoch_start = time.perf_counter()
            values = train_one_epoch(generator, 1.0, discriminator, train_loader, criterion,
                                     args.noise_dim, d_optimizer, g_optimizer)
            values.update(epoch=epoch, training_seconds=time.perf_counter() - epoch_start)
            values.update(fixed_scores(discriminator, references, device))
            images = sample_generator(generator, fixed_noise)
            save_grid(images, out_dir / f"fixed_noise_epoch_{epoch:02d}.png",
                      f"{args.architecture.upper()} GAN after epoch {epoch}: fixed noise seed 123")
            checkpoint = checkpoint_dir / f"epoch_{epoch:02d}.pt"
            if checkpoint.exists():
                raise FileExistsError(f"Checkpoint already exists: {checkpoint}")
            temporary = checkpoint.with_name(checkpoint.name + ".tmp")
            torch.save(capture_training_state(epoch, generator, discriminator, g_optimizer,
                                              d_optimizer, train_loader, config), temporary)
            temporary.replace(checkpoint)
            record["history"].append(values)
            record["checkpoints"].append({"epoch": epoch, "path": str(checkpoint), "sha256": sha256(checkpoint)})
            write_json(out_dir / "metrics.json", record)
            save_curves(record["history"], out_dir / "training_curves.png")
            print(f"epoch={epoch} D={values['d_loss']:.6f} G={values['g_loss']:.6f} seconds={values['training_seconds']:.2f}", flush=True)
        if last_epoch < args.epochs:
            record.update(status="paused", paused_at=datetime.now(timezone.utc).isoformat())
            print(f"PAUSED at epoch {last_epoch}/{args.epochs}: {out_dir}", flush=True)
            return
        final_images = sample_generator(generator, final_noise)
        save_grid(final_images, out_dir / "final_noise_seed456.png", f"Final epoch {args.epochs}: independent noise seed 456")
        class_name = config["class_names"][int(val_loader.dataset[0][1])] if args.dataset == "fashion-mnist" else None
        save_comparisons(validation_images, final_images, val_loader.dataset[0][1], out_dir,
                         args.epochs, dataset_name=dataset_name, class_name=class_name)
        audit = []
        audit_key = "discriminator_by_class" if args.dataset == "fashion-mnist" else "discriminator_by_digit"
        for split, dataset in [("train", train_loader.dataset), ("validation", val_loader.dataset)]:
            if args.dataset == "fashion-mnist":
                audit.extend(score_real_by_class(discriminator, dataset, split, device, config["class_names"]))
            else:
                audit.extend(score_real_by_digit(discriminator, dataset, split, device))
        with (out_dir / f"{audit_key}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
            writer.writeheader()
            writer.writerows(audit)
        record.update(status="completed", selected_epoch=args.epochs,
                      completed_at=datetime.now(timezone.utc).isoformat())
        record[audit_key] = audit
        record.pop("paused_at", None)
        print(f"COMPLETED: {out_dir}", flush=True)
    except Exception as exc:
        record.update(status="failed", error=repr(exc))
        raise
    finally:
        invocation.update(active_seconds=time.perf_counter() - start,
                          finished_at=datetime.now(timezone.utc).isoformat(), status=record["status"])
        record["total_seconds"] = sum(item["active_seconds"] for item in record["invocations"])
        record["peak_rss_mb"] = max(record.get("peak_rss_mb", 0), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        write_json(out_dir / "metrics.json", record)


if __name__ == "__main__":
    main()
