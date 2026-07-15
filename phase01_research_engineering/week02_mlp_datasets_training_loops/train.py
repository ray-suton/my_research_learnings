import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import argparse 
from dataset import get_dataloaders
from model import build_model


EPOCHS = 10
BATCH_SIZE = 64 
LR = 1e-3
HIDDEN_SIZE = 128
SEED = 42 
def parse_args(argv = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Fashion-MNIST MLP.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)  
    parser.add_argument("--learning-rate", type=float, default=LR) 
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--hidden-size", type = int, default = HIDDEN_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-batches", type = int, default = None)
    parser.add_argument("--max-val-batches", type = int, default = None)
    parser.add_argument("--max-test-batches", type = int, default = None)
    return parser.parse_args(argv)
    
def train_one_epoch(model, loader, loss_fn, optimizer, device, max_batches = None):
    model.train()
    total_loss = 0.0
    total_batches = 0

    for batch_idx, (x,y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / total_batches



def train(args = None):
    if args is None:
          args = parse_args()
    torch.manual_seed(args.seed)
    train_loader, val_loader, _ = get_dataloaders(
        batch_size = args.batch_size, 
        seed = args.seed, 
        num_workers = args.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(hidden_size = args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, 
            max_batches = args.max_train_batches,
        )
        train_losses.append(train_loss)
        val_loss, val_acc = evaluate(
            model, val_loader, loss_fn, device,
            max_batches = args.max_val_batches,
        )
        val_losses.append(val_loss)
        print(f"epoch {epoch+1}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc = {val_acc:.4f}"
        )
    return model, train_losses, val_losses

@torch.no_grad
def evaluate(model, loader, loss_fn, device, max_batches = None):
    model.eval()
    correct = total = total_loss = 0.0
    total_batches = 0
    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        preds = logits.argmax(dim=1)

        total_loss += loss.item()
        total_batches += 1
        correct += (preds == y).sum().item()
        total += y.size(0)
    
    return total_loss / total_batches, correct / total 


def full_train_test_val(args = None):
    if args is None:
        args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size = args.batch_size, 
        seed = args.seed, 
        num_workers = args.num_workers
    )
    model, train_losses, val_losses = train(args)
    loss_fn = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, loss_fn, device, max_batches = args.max_test_batches)
    
    print(f"test_loss:{test_loss:.4f} test_acc:{test_acc:.4f}")
    return train_losses, val_losses, test_loss 
    

if __name__ == "__main__":
    full_train_test_val()
