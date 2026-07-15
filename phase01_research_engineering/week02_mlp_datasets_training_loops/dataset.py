"""Fashion-MNIST dataset and dataloader helpers for the Week 2 MLP."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


BATCH_SIZE = 64
SEED = 42
TRAIN_SIZE = 50_000
VAL_SIZE = 10_000


def resolve_data_root() -> Path:
    """Return the external roadmap data root."""
    return Path(os.environ.get("AI_ROADMAP_DATA", "~/data/ai_research_roadmap")).expanduser()


def fashion_mnist_root(data_root: Path | None = None) -> Path:
    """Return the torchvision cache directory used by the roadmap data scripts."""
    root = resolve_data_root() if data_root is None else data_root
    return root / "raw" / "torchvision"


def build_transform() -> transforms.Compose:
    """Convert Fashion-MNIST PIL images into normalized tensors."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )


def get_datasets(
    data_root: Path | None = None,
    seed: int = SEED,
    download: bool = False,
):
    """Load Fashion-MNIST and split the official train set into train/val."""
    root = fashion_mnist_root(data_root)
    transform = build_transform()

    full_train = datasets.FashionMNIST(
        root=str(root),
        train=True,
        transform=transform,
        download=download,
    )
    test_dataset = datasets.FashionMNIST(
        root=str(root),
        train=False,
        transform=transform,
        download=download,
    )

    split_generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train,
        [TRAIN_SIZE, VAL_SIZE],
        generator=split_generator,
    )
    return train_dataset, val_dataset, test_dataset


def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    data_root: Path | None = None,
    num_workers: int = 0,
    download: bool = False,
):
    """Create train, validation, and test dataloaders."""
    train_dataset, val_dataset, test_dataset = get_datasets(
        data_root=data_root,
        seed=seed,
        download=download,
    )

    shuffle_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=shuffle_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the Fashion-MNIST dataloaders.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download Fashion-MNIST if it is missing from AI_ROADMAP_DATA.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        download=args.download,
    )

    images, labels = next(iter(train_loader))
    print(f"data_root={resolve_data_root()}")
    print(f"train_batches={len(train_loader)}")
    print(f"val_batches={len(val_loader)}")
    print(f"test_batches={len(test_loader)}")
    print(f"images_shape={tuple(images.shape)}")
    print(f"labels_shape={tuple(labels.shape)}")
    print(f"image_dtype={images.dtype}")
    print(f"label_dtype={labels.dtype}")


if __name__ == "__main__":
    main()
