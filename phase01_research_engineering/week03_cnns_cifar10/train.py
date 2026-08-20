import torch
import torch.nn as nn
import torch.optim as opt
import argparse
from dataset import get_dataloaders
from model import build_model
from analyze import confusion_matrix

EPOCHS =500 # 1000


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

def train(epochs, model_name = "small_cnn"):

    train_loader, val_loader, test_loader = get_dataloaders()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(model_name)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = opt.AdamW(params = model.parameters(),lr = 1e-3, weight_decay= 1e-3)
    scheduler = opt.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs, eta_min = 1e-5)
    for epoch in range(epochs):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_accuracy, _, _ = evaluate(model, val_loader, loss_fn, device)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f'epoch {epoch}, training_loss = {train_loss}, train_accuracy = {train_accuracy}, val_loss = {val_loss}, val_accuracy = {val_accuracy}. Current LR = {current_lr}')
        scheduler.step()
    test_loss, test_accuracy, pred, real = evaluate(model, test_loader, loss_fn, device)

    conf_mat = confusion_matrix(real, pred, 10)
    
    print(f"Final test_loss = {test_loss}, test_accuracy = {test_accuracy}, \nConfusion Matrix: {conf_mat}")

    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small_cnn", choices=["small_cnn", "mini_resnet"])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    train(epochs = args.epochs, model_name = args.model)

if __name__ == "__main__":
    main()
