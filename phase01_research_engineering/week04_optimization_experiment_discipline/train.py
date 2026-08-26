import torch
import torch.nn as nn
import torch.optim as opt
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from dataset import get_dataloaders
from model import build_model
from analyze import confusion_matrix
from config_utils import format_config, load_config, write_config


PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_ROOT = PROJECT_ROOT / "results" / "runs"


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    total_batches = 0
    correct = 0.0
    total_examples = 0
    for batch_idx, (x, y) in enumerate(loader):

        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        preds = torch.argmax(logits, dim = 1)
        correct += torch.sum((preds == y).float()).item()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1
        total_examples += y.size(0)

    return total_loss / total_batches, correct / total_examples

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    correct = 0.0
    total_examples = 0
    all_preds = []
    real = []
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        preds = torch.argmax(logits, dim = 1)
        correct += torch.sum((preds == y).float()).item()
        total_loss += loss.item()
        total_batches += 1
        total_examples += y.size(0)
        all_preds.extend(preds)
        real.extend(y)
    return total_loss / total_batches, correct / total_examples, all_preds, real

def build_optimizer(model, config):
    optimizer_name = config["optimizer"]
    if optimizer_name == "adamw":
        return opt.AdamW(
            params=model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
    if optimizer_name == "sgd":
        return opt.SGD(
            params=model.parameters(),
            lr=config["learning_rate"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
        )
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def build_scheduler(optimizer, config):
    scheduler_name = config["scheduler"]
    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return opt.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["epochs"],
            eta_min=config["scheduler_min_lr"],
        )
    raise ValueError(f"Unknown scheduler: {scheduler_name}")


def make_run_dir(config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_ROOT / f"{timestamp}_{config['run_name']}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_run_record(run_dir, config, metrics):
    write_config(run_dir / "config.yaml", config)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    notes = [
        f"# {config['run_name']}",
        "",
        "## Summary",
        "",
        f"- status: {metrics['status']}",
        f"- device: {metrics['device']}",
        f"- epochs: {metrics['epochs']}",
        f"- best validation accuracy: {metrics['best_val_accuracy']}",
        f"- final validation accuracy: {metrics['final_val_accuracy']}",
        f"- test evaluated: {metrics['test_evaluated']}",
        "",
        "## Notes",
        "",
        "- Add interpretation after comparing this run with other ablations.",
    ]
    (run_dir / "notes.md").write_text("\n".join(notes) + "\n")


def train(config, record_run=True):
    run_dir = make_run_dir(config) if record_run else None
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=config["batch_size"],
        data_root=config["data_root"],
        seed=config["seed"],
        num_workers=config["num_workers"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(config["model_name"])
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    print(f"Running config:\n{format_config(config)}")
    start_time = time.time()
    history = []
    best_val_accuracy = float("-inf")
    best_epoch = None
    for epoch in range(config["epochs"]):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_accuracy, _, _ = evaluate(model, val_loader, loss_fn, device)
        current_lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": current_lr,
            }
        )
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
        print(f'epoch {epoch}, training_loss = {train_loss}, train_accuracy = {train_accuracy}, val_loss = {val_loss}, val_accuracy = {val_accuracy}. Current LR = {current_lr}')
        if scheduler is not None:
            scheduler.step()

    metrics = {
        "status": "completed",
        "run_name": config["run_name"],
        "device": str(device),
        "epochs": config["epochs"],
        "optimizer": config["optimizer"],
        "scheduler": config["scheduler"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "batch_size": config["batch_size"],
        "seed": config["seed"],
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "final_train_loss": history[-1]["train_loss"],
        "final_train_accuracy": history[-1]["train_accuracy"],
        "final_val_loss": history[-1]["val_loss"],
        "final_val_accuracy": history[-1]["val_accuracy"],
        "final_learning_rate": optimizer.param_groups[0]["lr"],
        "runtime_seconds": round(time.time() - start_time, 3),
        "test_evaluated": False,
        "history": history,
    }

    if config.get("evaluate_test", False):
        test_loss, test_accuracy, pred, real = evaluate(model, test_loader, loss_fn, device)
        conf_mat = confusion_matrix(real, pred, 10)
        metrics.update(
            {
                "test_evaluated": True,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "confusion_matrix": conf_mat,
            }
        )
        print(f"Final test_loss = {test_loss}, test_accuracy = {test_accuracy}, \nConfusion Matrix: {conf_mat}")
    else:
        print("Skipped test evaluation because evaluate_test is false.")

    if run_dir is not None:
        metrics["run_dir"] = str(run_dir)
        write_run_record(run_dir, config, metrics)
        print(f"Wrote run record to {run_dir}")

    return metrics

    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)

if __name__ == "__main__":
    main()
