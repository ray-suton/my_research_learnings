"""Compare the Week 6 AE with a VAE selected using validation data only."""

import argparse
import hashlib
import importlib.util
import json
import platform
import shlex
import sys
import textwrap
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import torch
import torchvision

from loader import FMNIST_ROOT, get_dataloaders
from model import vae


PROJECT_ROOT = Path(__file__).resolve().parent
AE_MODEL_PATH = PROJECT_ROOT.parent / "week06_autoencoders_representation_learning" / "model.py"
FIGURE_SEED = 2026
EVALUATION_SEED = 123


def load_autoencoder(input_dim, hidden_dim, latent_dim):
    spec = importlib.util.spec_from_file_location("week06_comparison_model", AE_MODEL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the Week 6 model from {AE_MODEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AutoEncoder(input_dim, hidden_dim, latent_dim)


def train_one_epoch(loader, model, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_examples = 0
    for X, _ in loader:
        X = X.to(device)
        preds = model(X)
        mse_loss = loss_fn(preds, X)
        if not torch.isfinite(mse_loss):
            raise ValueError("The AE training loss is not finite")
        optimizer.zero_grad()
        mse_loss.backward()
        optimizer.step()
        total_loss += mse_loss.item() * X.shape[0]
        total_examples += X.shape[0]
    if total_examples == 0:
        raise ValueError("The training loader is empty")
    return {"reconstruction_mse": total_loss / total_examples, "n_examples": total_examples}


@torch.no_grad()
def evaluate_autoencoder(loader, model, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    for X, _ in loader:
        X = X.to(device)
        mse_loss = loss_fn(model(X), X)
        total_loss += mse_loss.item() * X.shape[0]
        total_examples += X.shape[0]
    if total_examples == 0:
        raise ValueError("The evaluation loader is empty")
    return {"reconstruction_mse": total_loss / total_examples, "n_examples": total_examples}


def save_grid(rows, labels, output_path, title):
    samples = rows[0].shape[0]
    fig, axes = plt.subplots(
        len(rows), samples, figsize=(1.4 * samples + 2, 1.6 * len(rows)), squeeze=False
    )
    for row_index, row in enumerate(rows):
        for index in range(samples):
            ax = axes[row_index, index]
            ax.imshow(row[index].detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if index == 0:
                ax.set_ylabel(labels[row_index], rotation=0, ha="right", va="center", fontsize=9)
    fig.suptitle(textwrap.fill(title, width=50 + 5 * samples), fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


@torch.no_grad()
def save_comparison_figures(loader, autoencoder, vae_model, output_dir, device, latent_dim):
    autoencoder.eval()
    vae_model.eval()
    X, _ = next(iter(loader))
    X = X[:8].to(device)
    mu, logvar = vae_model.encode(X)
    generator = torch.Generator().manual_seed(FIGURE_SEED)
    eps = torch.randn(mu.shape, generator=generator).to(device)
    sampled_z = mu + torch.exp(0.5 * logvar) * eps
    mean_preds = vae_model.decoder(mu).reshape_as(X)
    sampled_preds = vae_model.decoder(sampled_z).reshape_as(X)
    reconstruction_path = output_dir / "session5_reconstruction_comparison.png"
    save_grid(
        [X, autoencoder(X), mean_preds, sampled_preds],
        ["Original", "AE", "VAE: decode(mu)", "VAE: posterior draw"],
        reconstruction_path,
        "Same held-out images; VAE mean code and one posterior draw shown separately",
    )

    z = torch.randn((X.shape[0], latent_dim), generator=generator).to(device)
    ae_probe = autoencoder.decoder(z).reshape_as(X)
    vae_samples = vae_model.decoder(z).reshape_as(X)
    prior_path = output_dir / "session5_random_latent_comparison.png"
    save_grid(
        [ae_probe, vae_samples],
        ["AE: arbitrary N(0,I) probe\n(no trained prior)", "VAE: N(0,I) prior"],
        prior_path,
        "Random latent decoding; AE latent coordinates are not trained to follow this prior",
    )
    return {"reconstructions": str(reconstruction_path), "random_latents": str(prior_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if min(args.epochs, args.batch, args.threads) <= 0 or args.lr <= 0:
        parser.error("epochs, batch, threads, and learning rate must be positive")
    if not args.vae_checkpoint.is_file():
        parser.error(f"The selected VAE checkpoint does not exist: {args.vae_checkpoint}")
    if (args.output_dir / "session5_comparison_metrics.json").exists():
        parser.error("Comparison output already exists; choose a new output directory")

    # Import the VAE evaluator after argument parsing so --help stays lightweight.
    from train import evaluate

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.vae_checkpoint, map_location=device, weights_only=True)
    config = checkpoint["config"]
    matched_settings = {
        "epochs": args.epochs,
        "batch_size": args.batch,
        "seed": args.seed,
        "learning_rate": args.lr,
    }
    for setting, value in matched_settings.items():
        if config[setting] != value:
            parser.error(
                f"Matched comparison requires {setting}={config[setting]} (received {value})"
            )
    if config["input_dim"] != 784:
        parser.error("The Week 6 AE comparison requires 784-pixel Fashion-MNIST inputs")
    if checkpoint["epoch"] != args.epochs:
        parser.error("Use the final fixed-epoch VAE checkpoint, not an earlier checkpoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch, download=False, seed=args.seed
    )
    autoencoder = load_autoencoder(
        config["input_dim"], config["hidden_dim"], config["latent_dim"]
    ).to(device)
    vae_model = vae(config["input_dim"], config["hidden_dim"], config["latent_dim"]).to(device)
    vae_model.load_state_dict(checkpoint["model_state"])
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    history = []
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(train_loader, autoencoder, optimizer, loss_fn, device)
        val_metrics = evaluate_autoencoder(val_loader, autoencoder, loss_fn, device)
        history.append({"epoch": epoch, "train": train_metrics, "validation": val_metrics})
        print(
            f"AE epoch {epoch}/{args.epochs}: train MSE={train_metrics['reconstruction_mse']:.6f}, "
            f"validation MSE={val_metrics['reconstruction_mse']:.6f}",
            flush=True,
        )

    # Settings and VAE selection are already fixed before either test evaluation.
    validation = {
        "ae": history[-1]["validation"],
        "vae": evaluate(
            val_loader, vae_model, loss_fn, config["beta"], device, seed=EVALUATION_SEED
        ),
    }
    test = {
        "ae": evaluate_autoencoder(test_loader, autoencoder, loss_fn, device),
        "vae": evaluate(
            test_loader, vae_model, loss_fn, config["beta"], device, seed=EVALUATION_SEED
        ),
    }
    figures = save_comparison_figures(
        test_loader, autoencoder, vae_model, args.output_dir, device, config["latent_dim"]
    )
    metrics = {
        "run_name": "session5_matched_autoencoder_comparison",
        "command": shlex.join([sys.executable, *sys.argv]),
        "config": {
            **matched_settings,
            "input_dim": config["input_dim"],
            "hidden_dim": config["hidden_dim"],
            "latent_dim": config["latent_dim"],
            "optimizer": "Adam",
            "ae_noise_std": 0,
            "device": str(device),
            "threads": args.threads,
            "evaluation_seed": EVALUATION_SEED,
            "figure_seed": FIGURE_SEED,
        },
        "vae_checkpoint": str(args.vae_checkpoint.resolve()),
        "vae_checkpoint_sha256": hashlib.sha256(args.vae_checkpoint.read_bytes()).hexdigest(),
        "vae_run_name": checkpoint["run_name"],
        "vae_config": config,
        "vae_epoch": checkpoint["epoch"],
        "selection_protocol": (
            "VAE checkpoint selected using validation only before this command; "
            "fixed final-epoch AE; no test tuning"
        ),
        "data": {
            "dataset": "Fashion-MNIST",
            "cache": str(FMNIST_ROOT),
            "download": False,
            "train_examples": len(train_loader.dataset),
            "validation_examples": len(val_loader.dataset),
            "test_examples": len(test_loader.dataset),
            "split": "torch random_split of official training set, seeded by config.seed",
        },
        "metric_definitions": {
            "ae.reconstruction_mse": (
                "Per-pixel squared error, weighted by example count across batches"
            ),
            "vae.reconstruction_mse": (
                "Per-pixel squared error from one posterior sample per image, "
                "with fixed evaluation RNG seed"
            ),
            "vae.mean_latent_mse": (
                "Per-pixel squared error from decode(mu); this is not the expected decoded image"
            ),
            "vae.reconstruction_term": "784 times posterior-sampled per-pixel MSE",
            "vae.kl": "KL to N(0,I), summed over latent dimensions and averaged across examples",
            "vae.total": (
                "Posterior-sampled reconstruction term plus beta times KL; "
                "not comparable with AE MSE directly"
            ),
        },
        "comparison_limits": [
            "This is a fresh matched AE run, not a replay of the historical Week 6 metrics.",
            "Historical Week 6 evaluation averaged batch means; this comparison weights by "
            "example count, including partial batches.",
            "Both models share data split, dimensions, epoch budget, batch size, learning rate, "
            "and seed; objectives and parameter counts differ.",
            "VAE sampled and mean-code reconstruction answer different questions "
            "and are reported separately.",
            "The AE has no trained prior: its random-normal latent probes are illustrative, "
            "not valid prior samples.",
            "One seed and a short training budget do not establish general superiority "
            "or quantitative generation quality.",
        ],
        "parameter_counts": {
            "ae": sum(parameter.numel() for parameter in autoencoder.parameters()),
            "vae": sum(parameter.numel() for parameter in vae_model.parameters()),
        },
        "ae_history": history,
        "validation": validation,
        "test": test,
        "figures": figures,
        "runtime_seconds": round(time.perf_counter() - start_time, 3),
        "versions": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "matplotlib": matplotlib.__version__,
        },
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__).resolve(), PROJECT_ROOT / "train.py",
                PROJECT_ROOT / "model.py", PROJECT_ROOT / "loss_helper.py",
                PROJECT_ROOT / "loader.py", AE_MODEL_PATH,
            )
        },
    }
    metrics_path = args.output_dir / "session5_comparison_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved comparison metrics: {metrics_path}", flush=True)
    print(f"AE test MSE: {test['ae']['reconstruction_mse']:.6f}", flush=True)
    print(f"VAE posterior-sampled test MSE: {test['vae']['reconstruction_mse']:.6f}", flush=True)
    print(f"VAE mean-code test MSE: {test['vae']['mean_latent_mse']:.6f}", flush=True)


if __name__ == "__main__":
    main()
