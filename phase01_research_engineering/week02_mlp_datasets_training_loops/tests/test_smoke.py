import torch
from torch import nn

from dataset import get_dataloaders
from model import build_model



def test_one_batch_forward_and_loss():
    train_loader, _, _ = get_dataloaders(batch_size=64)

    images, labels = next(iter(train_loader))

    model = build_model(hidden_size=128)
    logits = model(images)

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, labels)

    assert images.shape == (64, 1, 28, 28)
    assert labels.shape == (64,)
    assert logits.shape == (64, 10)
    assert torch.isfinite(loss)