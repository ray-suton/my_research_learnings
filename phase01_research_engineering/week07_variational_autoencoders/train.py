"""Train and evaluate the small Fashion-MNIST VAE."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import sys
import time

import torch

from loader import FMNIST_ROOT, get_dataloaders
from loss_helper import vae_loss
from model import vae


PROJECT_DIR = Path(__file__).resolve().parent
METRICS = ("reconstruction_mse", "reconstruction_term", "kl", "total")


def batch_metrics(mse_loss, total_loss, kl_loss, pixels):
    values = {
        "reconstruction_mse": mse_loss.item(),
        "reconstruction_term": pixels * mse_loss.item(),
        "kl": kl_loss.item(),
        "total": total_loss.item(),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Non-finite reconstruction, KL, or total loss")
    return values


def train_one_epoch(loader, model, loss_fn, beta, optimizer, device):
    model.train()
    totals = {name: 0.0 for name in METRICS}
    total_images = 0

    for X, _ in loader:
        X = X.to(device)
        preds, mu, logvar = model(X)
        pixels = X[0].numel()
        mse_loss = loss_fn(preds, X)
        mse_loss, total_loss, kl_loss = vae_loss(
            mse_loss, mu, logvar, beta, pixels
        )
        values = batch_metrics(mse_loss, total_loss, kl_loss, pixels)
        optimizer.zero_grad()
        total_loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for p in model.parameters()):
            raise ValueError("Non-finite training gradient")
        optimizer.step()
        for name in METRICS:
            totals[name] += values[name] * X.shape[0]
        total_images += X.shape[0]

    if total_images == 0:
        raise ValueError("Training loader is empty")
    result = {name: value / total_images for name, value in totals.items()}
    result["n_examples"] = total_images
    print(
        f"train MSE: {result['reconstruction_mse']:.6f}, "
        f"reconstruction term: {result['reconstruction_term']:.6f}, "
        f"KL: {result['kl']:.6f}, total: {result['total']:.6f}"
    )
    return result


@torch.no_grad()
def evaluate(loader, model, loss_fn, beta, device, seed=123):
    device = torch.device(device)
    was_training = model.training
    totals = {name: 0.0 for name in METRICS}
    total_mean_mse = 0.0
    total_images = 0
    kl_by_dimension = None
    devices = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None
                   else torch.cuda.current_device()]

    try:
        model.eval()
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            for X, _ in loader:
                X = X.to(device)
                preds, mu, logvar = model(X)
                pixels = X[0].numel()
                mse_loss, total_loss, kl_loss = vae_loss(
                    loss_fn(preds, X), mu, logvar, beta, pixels
                )
                values = batch_metrics(mse_loss, total_loss, kl_loss, pixels)
                mean_preds = model.decoder(mu).reshape_as(X)
                mean_mse = loss_fn(mean_preds, X).item()
                if not math.isfinite(mean_mse):
                    raise ValueError("Non-finite mean-latent reconstruction MSE")
                for name in METRICS:
                    totals[name] += values[name] * X.shape[0]
                total_mean_mse += mean_mse * X.shape[0]
                dimension_sum = (
                    0.5 * (mu.square() + logvar.exp() - 1 - logvar)
                ).sum(0).cpu()
                if kl_by_dimension is None:
                    kl_by_dimension = torch.zeros_like(dimension_sum)
                kl_by_dimension += dimension_sum
                total_images += X.shape[0]
    finally:
        model.train(was_training)

    if total_images == 0 or kl_by_dimension is None:
        raise ValueError("Evaluation loader is empty")
    per_dimension = (kl_by_dimension / total_images).tolist()
    return {
        **{name: value / total_images for name, value in totals.items()},
        "mean_latent_mse": total_mean_mse / total_images,
        "n_examples": total_images,
        "kl_per_dimension": per_dimension,
        "active_latent_dimensions": sum(value > 0.01 for value in per_dimension),
    }


@torch.no_grad()
def save_figures(model, loader, history, output_dir, device, latent_size, seed):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    model.eval()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        X, _ = next(iter(loader))
        X = X[:16].to(device)
        mu, logvar = model.encode(X)
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(mu.shape, generator=generator, device=device)
        z = mu + (0.5 * logvar).exp() * noise
        sampled = model.decoder(z).reshape_as(X)
        mean_preds = model.decoder(mu).reshape_as(X)
        prior_z = torch.randn(
            16, latent_size, generator=generator, device=device
        )
        prior_images = model.decoder(prior_z).reshape(16, 1, 28, 28)

    fig, axes = plt.subplots(6, 8, figsize=(10, 8), constrained_layout=True)
    values = [X, mean_preds, sampled]
    names = ["Original", "Decode mu", "Posterior draw"]
    for block in range(2):
        for row, (images, label) in enumerate(zip(values, names)):
            for column in range(8):
                ax = axes[3 * block + row, column]
                index = 8 * block + column
                if index < len(images):
                    ax.imshow(images[index].cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                if column == 0:
                    ax.set_ylabel(label)
    fig.suptitle("Fixed validation images and VAE reconstructions")
    fig.savefig(output_dir / "reconstructions.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(4, 4, figsize=(5, 5), constrained_layout=True)
    for image, ax in zip(prior_images, axes.flat):
        ax.imshow(image.cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle("VAE samples from N(0, I)")
    fig.savefig(output_dir / "prior_samples.png", dpi=140)
    plt.close(fig)

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for ax, key, label in zip(
        axes, ["reconstruction_mse", "kl", "total"],
        ["Per-pixel reconstruction MSE", "KL per image (nats)", "Total objective"],
    ):
        for split in ("train", "validation"):
            ax.plot(epochs, [row[split][key] for row in history], marker="o", label=split)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.legend()
    fig.savefig(output_dir / "loss_curves.png", dpi=140)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="session3_latent32")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-seed", type=int, default=123)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    for name in ("epochs", "batch", "hidden", "latent", "threads"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")
    if not math.isfinite(args.beta) or args.beta < 0:
        parser.error("--beta must be finite and nonnegative")
    if (args.output_dir / "metrics.json").exists() or args.checkpoint.exists():
        parser.error("Run outputs already exist; choose new output/checkpoint paths")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch, download=False, seed=args.seed
    )
    model = vae(784, args.hidden, args.latent).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "input_dim": 784, "hidden_dim": args.hidden, "latent_dim": args.latent,
        "epochs": args.epochs, "batch_size": args.batch,
        "learning_rate": args.lr, "beta": args.beta, "seed": args.seed,
        "validation_seed": args.validation_seed, "device": str(device),
        "threads": args.threads, "data_root": str(FMNIST_ROOT),
        "train_size": len(train_loader.dataset),
        "validation_size": len(val_loader.dataset),
        "test_size": len(test_loader.dataset),
        "validation_sampling": "one posterior draw per image, isolated fixed seed",
        "loss_reduction": "sum pixels per image, mean images; KL sum latents, mean images",
        "python_version": sys.version.split()[0], "torch_version": str(torch.__version__),
    }
    command = shlex.join([sys.executable, *sys.argv])
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output_dir / "command.txt").write_text(command + "\n")
    started = time.perf_counter()
    history = []
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}", flush=True)
        train_metrics = train_one_epoch(
            train_loader, model, loss_fn, args.beta, optimizer, device
        )
        validation = evaluate(
            val_loader, model, loss_fn, args.beta, device, args.validation_seed
        )
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation})
        print(
            f"validation MSE: {validation['reconstruction_mse']:.6f}, "
            f"KL: {validation['kl']:.6f}, total: {validation['total']:.6f}",
            flush=True,
        )
    runtime = time.perf_counter() - started
    torch.save({
        "model_state": model.state_dict(), "config": config,
        "run_name": args.run_name, "epoch": args.epochs,
    }, args.checkpoint)
    save_figures(
        model, val_loader, history, args.output_dir, device, args.latent,
        args.validation_seed,
    )
    result = {
        "run_name": args.run_name, "status": "completed", "config": config,
        "command": command, "history": history, "final_validation": history[-1]["validation"],
        "test_evaluated": False, "test": None,
        "runtime_seconds": runtime,
        "runtime_scope": "training and validation only; figures/checkpoint excluded",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "checkpoint": str(args.checkpoint.resolve()),
        "artifacts": {name: str((args.output_dir / name).resolve()) for name in (
            "reconstructions.png", "prior_samples.png", "loss_curves.png",
        )},
        "source_sha256": {
            name: hashlib.sha256((PROJECT_DIR / name).read_bytes()).hexdigest()
            for name in ("model.py", "loader.py", "loss_helper.py", "train.py")
        },
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(f"Saved {args.output_dir / 'metrics.json'}; runtime {runtime:.2f}s", flush=True)


if __name__ == "__main__":
    main()
