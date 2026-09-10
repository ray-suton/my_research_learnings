from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision import datasets


DATA_ROOT = Path("/home/rui.gao/data/ai_research_roadmap")
FMNIST_ROOT = DATA_ROOT / 'raw' / "torchvision"

SEED = 42


BATCH_SIZE = 64
TRAIN_SIZE = 55000
VAL_SIZE = 5000
IMAGE_SHAPE = (1, 28, 28)

def build_transform():
    return transforms.Compose([
        transforms.ToTensor(),
    ])

def get_datasets(download: bool = False, seed: int = SEED):
    transform = build_transform()
    full_train = datasets.FashionMNIST(root = str(FMNIST_ROOT), train = True, transform = transform, download = download)

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [TRAIN_SIZE, VAL_SIZE], generator=generator)

    full_test = datasets.FashionMNIST(root = str(FMNIST_ROOT), train = False, transform = transform, download = download)

    return train_set, val_set, full_test

def get_dataloaders(batch_size: int = BATCH_SIZE, download: bool = False, seed: int = SEED):

    train_data, val_data, test_data = get_datasets(download = download, seed = seed)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, generator = generator)
    val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False)

    return train_loader, val_loader, test_loader
