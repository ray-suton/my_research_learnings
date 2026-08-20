import torch
from pathlib import Path 
from torchvision import transforms
from torch.utils.data import random_split, DataLoader, Subset 
from torchvision import datasets
DATA_ROOT = Path("/home/rui.gao/data/ai_research_roadmap")

CIFAR10_ROOT = DATA_ROOT / 'raw' / "torchvision"

SEED = 42


BATCH_SIZE = 64
TRAIN_SIZE = 45000
VAL_SIZE = 5000
NUM_CLASSES = 10
IMAGE_SHAPE = (3, 32, 32)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

def build_eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

def build_train_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)

    ])

def get_datasets(download: bool = False):
    eval_transform = build_eval_transform()
    train_transform = build_train_transform()
    
    split_base = datasets.CIFAR10(
        root = str(CIFAR10_ROOT),
        train = True,
        transform = eval_transform,
        download = download,
    )

    train_split, val_split = random_split(
        split_base,
        [TRAIN_SIZE, VAL_SIZE],
        generator = torch.Generator().manual_seed(SEED),
    )
    train_base = datasets.CIFAR10(
        root = str(CIFAR10_ROOT),
        train = True,
        transform = train_transform,
        download = download,
    )
    val_base = datasets.CIFAR10(
        root = str(CIFAR10_ROOT),
        train = True,
        transform = eval_transform,
        download = download,
    )

    full_test = datasets.CIFAR10(root = str(CIFAR10_ROOT), train = False, transform = eval_transform, download = download)
    train_data = Subset(train_base, train_split.indices)
    val_data = Subset(val_base, val_split.indices)

    return train_data, val_data, full_test

def get_dataloaders(batch_size: int = BATCH_SIZE, download: bool = False):

    train_data, val_data, test_data = get_datasets(download = download)

    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)

    val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = False)

    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False)

    return train_loader, val_loader, test_loader
