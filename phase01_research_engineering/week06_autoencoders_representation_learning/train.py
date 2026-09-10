import torch 
import torch.optim as optim
import argparse 
import json
import time
from pathlib import Path
from dataloader import get_dataloaders
from matplotlib import pyplot as plt 
from model import AutoEncoder
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SESSION3_OUTPUT = PROJECT_ROOT / "results" / "session3_reconstructions.png"
DEFAULT_SESSION4_OUTPUT = PROJECT_ROOT / "results" / "session4_denoising_reconstructions.png"
DEFAULT_SESSION3_METRICS = PROJECT_ROOT / "results" / "session3_metrics.json"
DEFAULT_SESSION4_METRICS = PROJECT_ROOT / "results" / "session4_denoising_metrics.json"

def add_noise(x, noise_std = 0.0):
    if noise_std <= 0:
        return x
    noise = torch.randn_like(x) * noise_std
    return torch.clamp(x + noise, 0.0, 1.0)

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate(model, val_loader, loss_fn, device, noise_std):
    model.eval()
    total_loss = 0 
    total_num = 0
    for batch_idx, (x, y) in enumerate(val_loader):
        x = x.to(device)
        noisy_x = add_noise(x, noise_std)
        output = model(noisy_x)
        loss = loss_fn(output, x)
        total_loss += loss.item()
        total_num += 1

    return total_loss / total_num


def train_one_epoch(model, train_loader, optimizer, loss_fn, device, noise_std):
    model.train()
    total_loss = 0 
    total_num = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        
        x = x.to(device)
        noisy_x = add_noise(x, noise_std)
        output = model(noisy_x)
        loss = loss_fn(output, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_num += 1
    return output, total_loss / total_num 


@torch.no_grad()
def try_out(model, test_loader, loss_fn, device, noise_std):
    
    model.eval()
    total_loss = 0 
    total_num = 0
    for batch_idx, (x, y) in enumerate(test_loader):
        x = x.to(device)
        noisy_x = add_noise(x, noise_std)
        output = model(noisy_x)
        loss = loss_fn(output, x)
        total_loss += loss.item()
        total_num += 1

    return output, total_loss / total_num


@torch.no_grad()
def average_image_baseline(train_loader, val_loader, device):
    total_image = None
    total_examples = 0
    for x, y in train_loader:
        x = x.to(device)
        batch_sum = x.sum(dim=0)
        total_image = batch_sum if total_image is None else total_image + batch_sum
        total_examples += x.shape[0]

    mean_image = total_image / total_examples
    total_loss = 0
    total_num = 0
    for x, y in val_loader:
        x = x.to(device)
        output = mean_image.unsqueeze(0).expand_as(x)
        loss = torch.nn.functional.mse_loss(output, x)
        total_loss += loss.item()
        total_num += 1

    return total_loss / total_num


@torch.no_grad()
def noisy_input_baseline(val_loader, device, noise_std):
    total_loss = 0
    total_num = 0
    for x, y in val_loader:
        x = x.to(device)
        noisy_x = add_noise(x, noise_std)
        loss = torch.nn.functional.mse_loss(noisy_x, x)
        total_loss += loss.item()
        total_num += 1

    return total_loss / total_num


@torch.no_grad()
def save_reconstruction_grid(model, test_loader, output_path, device, samples, noise_std):
    model.eval()
    x, _ = next(iter(test_loader))
    x = x.to(device)
    noisy_x = add_noise(x, noise_std)
    recon = model(noisy_x)
    samples = min(samples, x.shape[0])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = 3 if noise_std > 0 else 2
    fig, axes = plt.subplots(rows, samples, figsize=(1.5 * samples, 1.6 * rows), constrained_layout=True)
    if samples == 1:
        axes = axes.reshape(rows, 1)
    for index in range(samples):
        axes[0, index].imshow(x[index].detach().cpu().squeeze(), cmap="gray")
        axes[0, index].axis("off")
        if noise_std > 0:
            axes[1, index].imshow(noisy_x[index].detach().cpu().squeeze(), cmap="gray")
            axes[1, index].axis("off")
            axes[2, index].imshow(recon[index].detach().cpu().squeeze(), cmap="gray")
            axes[2, index].axis("off")
        else:
            axes[1, index].imshow(recon[index].detach().cpu().squeeze(), cmap="gray")
            axes[1, index].axis("off")

    axes[0, 0].set_title("Original")
    if noise_std > 0:
        axes[1, 0].set_title("Noisy")
        axes[2, 0].set_title("Denoised")
    else:
        axes[1, 0].set_title("Reconstruction")
    fig.savefig(output_path)
    plt.close(fig)


def write_metrics(metrics, metrics_path):
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def train(epochs, model, train_loader, val_loader, test_loader, learning_rate, output_path, samples, noise_std, metrics_path, run_config):
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    loss_fn = torch.nn.MSELoss()
    start_time = time.time()
    history = []
    for epoch in range(epochs):
        outputs, loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE, noise_std)
        val_loss = evaluate(model, val_loader, loss_fn, DEVICE, noise_std)
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss,
                "val_loss": val_loss,
            }
        )

        print(f"Epoch {epoch}, training_loss: {loss}, val_loss: {val_loss}")

    
    final_outputs, test_loss = try_out(model, test_loader, loss_fn, DEVICE, noise_std)
    baseline_val_loss = average_image_baseline(train_loader, val_loader, DEVICE)
    noisy_val_loss = noisy_input_baseline(val_loader, DEVICE, noise_std)
    save_reconstruction_grid(model, test_loader, output_path, DEVICE, samples, noise_std)
    runtime = time.time() - start_time
    metrics = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "noise_std": noise_std,
        "samples": samples,
        "device": str(DEVICE),
        "history": history,
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "test_loss": test_loss,
        "average_image_val_loss": baseline_val_loss,
        "noisy_input_val_loss": noisy_val_loss,
        "saved_reconstructions": str(output_path),
        "runtime_seconds": round(runtime, 3),
    }
    metrics.update(run_config)
    write_metrics(metrics, metrics_path)

    print(f"test_loss: {test_loss}")
    print(f"average_image_val_loss: {baseline_val_loss}")
    print(f"noisy_input_val_loss: {noisy_val_loss}")
    print(f"saved_reconstructions: {output_path}")
    print(f"saved_metrics: {metrics_path}")
    print(f"runtime_seconds: {runtime:.3f}")
    
    return {
        "model": model,
        "metrics": metrics,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int, default = 5)
    parser.add_argument("--input", type = int, default = 784)
    parser.add_argument("--hidden", type = int, default = 128)
    parser.add_argument("--latent", type = int, default = 32)
    parser.add_argument("--batch", type = int, default = 128)
    parser.add_argument("--lr", type = float, default = 1e-3)
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--noise-std", type = float, default = 0.0)
    parser.add_argument("--output", type = str, default = None)
    parser.add_argument("--metrics-output", type = str, default = None)
    parser.add_argument("--samples", type = int, default = 8)
    args = parser.parse_args()
    set_seed(args.seed)
    output_path = args.output
    if output_path is None:
        output_path = DEFAULT_SESSION4_OUTPUT if args.noise_std > 0 else DEFAULT_SESSION3_OUTPUT
    metrics_path = args.metrics_output
    if metrics_path is None:
        metrics_path = DEFAULT_SESSION4_METRICS if args.noise_std > 0 else DEFAULT_SESSION3_METRICS
    autoencoder = AutoEncoder(args.input,args.hidden,args.latent).to(DEVICE)
    train_loader, val_loader, test_loader = get_dataloaders(batch_size= args.batch, download = False, seed = args.seed)
    train(
        args.epochs,
        autoencoder,
        train_loader,
        val_loader,
        test_loader,
        args.lr,
        output_path,
        args.samples,
        args.noise_std,
        metrics_path,
        {
            "run_id": "session4_denoising_autoencoder" if args.noise_std > 0 else "session3_plain_autoencoder",
            "input_dim": args.input,
            "hidden_dim": args.hidden,
            "latent_dim": args.latent,
            "batch_size": args.batch,
            "seed": args.seed,
        },
    )
    
