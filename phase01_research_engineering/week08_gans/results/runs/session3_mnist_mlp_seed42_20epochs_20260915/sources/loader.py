"""Read cached MNIST using the roadmap's external data-root convention."""

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def resolve_data_root(data_root=None):
    value = data_root or os.environ.get(
        "AI_ROADMAP_DATA", str(Path.home() / "data/ai_research_roadmap")
    )
    return Path(value).expanduser().resolve()


def get_dataloaders(batch_size=128, seed=42, data_root=None, num_workers=0):
    """Return seeded 55k/5k/10k loaders, with downloads disabled."""
    root = resolve_data_root(data_root) / "raw/torchvision"
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),
    ])
    full_train = datasets.MNIST(
        root=str(root), train=True, download=False, transform=transform
    )
    test = datasets.MNIST(
        root=str(root), train=False, download=False, transform=transform
    )
    train, validation = random_split(
        full_train, [55000, 5000],
        generator=torch.Generator().manual_seed(seed),
    )
    loaders = []
    for offset, (dataset, shuffle) in enumerate([
        (train, True), (validation, False), (test, False),
    ]):
        loaders.append(DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, drop_last=False,
            generator=torch.Generator().manual_seed(seed + offset),
        ))
    return tuple(loaders)
