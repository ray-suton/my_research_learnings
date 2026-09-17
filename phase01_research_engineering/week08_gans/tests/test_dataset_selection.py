"""Dataset-selection checks for cached Week 8 GAN loaders."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset
from torchvision import datasets


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import loader
from loader import get_dataloaders, resolve_data_root


def cached_or_skip(testcase, dataset_name):
    root = resolve_data_root() / "raw/torchvision" / dataset_name / "raw"
    if not root.exists():
        testcase.skipTest(f"cached {dataset_name} data not found at {root}")


def base_dataset(split):
    return split.dataset if hasattr(split, "dataset") else split


class DatasetSelectionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_split_contract(self, train_loader, val_loader, test_loader):
        self.assertEqual(len(train_loader.dataset), 55000)
        self.assertEqual(len(val_loader.dataset), 5000)
        self.assertEqual(len(test_loader.dataset), 10000)

        train_indices = set(train_loader.dataset.indices)
        val_indices = set(val_loader.dataset.indices)
        self.assertFalse(train_indices.intersection(val_indices))
        self.assertEqual(len(train_indices.union(val_indices)), 60000)

    def assert_normalized_batch(self, train_loader):
        images, labels = next(iter(train_loader))
        self.assertEqual(images.shape, (8, 1, 28, 28))
        self.assertEqual(labels.shape, (8,))
        self.assertTrue(torch.isfinite(images).all())
        self.assertGreaterEqual(images.min().item(), -1.0)
        self.assertLessEqual(images.max().item(), 1.0)
        self.assertLess(images.min().item(), 0.0)

    def test_default_uses_cached_mnist_with_original_contract(self):
        cached_or_skip(self, "MNIST")

        train_loader, val_loader, test_loader = get_dataloaders(
            batch_size=8, seed=42
        )

        self.assertIs(type(base_dataset(train_loader.dataset)), datasets.MNIST)
        self.assertIs(type(base_dataset(val_loader.dataset)), datasets.MNIST)
        self.assertIs(type(test_loader.dataset), datasets.MNIST)
        self.assert_split_contract(train_loader, val_loader, test_loader)
        self.assert_normalized_batch(train_loader)

    def test_fashion_mnist_selects_cached_class_and_preserves_split_contract(self):
        cached_or_skip(self, "FashionMNIST")

        train_loader, val_loader, test_loader = get_dataloaders(
            batch_size=8, seed=42, dataset="fashion-mnist"
        )

        self.assertIs(type(base_dataset(train_loader.dataset)), datasets.FashionMNIST)
        self.assertIs(type(base_dataset(val_loader.dataset)), datasets.FashionMNIST)
        self.assertIs(type(test_loader.dataset), datasets.FashionMNIST)
        self.assert_split_contract(train_loader, val_loader, test_loader)
        self.assert_normalized_batch(train_loader)

    def test_split_and_loader_order_are_deterministic_without_global_rng_change(self):
        cached_or_skip(self, "FashionMNIST")
        torch.manual_seed(1234)
        before = torch.get_rng_state().clone()

        first = get_dataloaders(batch_size=8, seed=99, dataset="fashion-mnist")
        after_first = torch.get_rng_state().clone()
        second = get_dataloaders(batch_size=8, seed=99, dataset="fashion-mnist")
        after_second = torch.get_rng_state().clone()

        self.assertTrue(torch.equal(before, after_first))
        self.assertTrue(torch.equal(before, after_second))
        self.assertEqual(first[0].dataset.indices, second[0].dataset.indices)
        self.assertEqual(first[1].dataset.indices, second[1].dataset.indices)

        first_images, first_labels = next(iter(first[0]))
        second_images, second_labels = next(iter(second[0]))
        self.assertTrue(torch.equal(first_images, second_images))
        self.assertTrue(torch.equal(first_labels, second_labels))

    def test_download_is_never_enabled_for_selected_dataset(self):
        calls = []

        class RecordingDataset(TensorDataset):
            def __init__(self, root, train, download, transform):
                calls.append({
                    "root": root,
                    "train": train,
                    "download": download,
                    "transform": transform,
                })
                size = 60000 if train else 10000
                images = torch.zeros(size, 1, 28, 28)
                labels = torch.zeros(size, dtype=torch.long)
                super().__init__(images, labels)

        with tempfile.TemporaryDirectory() as data_root:
            with patch.dict(loader.DATASETS, {"fashion-mnist": RecordingDataset}):
                train_loader, val_loader, test_loader = get_dataloaders(
                    batch_size=8,
                    seed=42,
                    data_root=data_root,
                    dataset="fashion-mnist",
                )

        self.assertEqual(len(calls), 2)
        self.assertEqual([call["train"] for call in calls], [True, False])
        self.assertEqual([call["download"] for call in calls], [False, False])
        for call in calls:
            self.assertTrue(call["root"].endswith("raw/torchvision"))
            self.assertIsNotNone(call["transform"])
        self.assert_split_contract(train_loader, val_loader, test_loader)

    def test_invalid_dataset_rejected_before_any_data_access(self):
        class RaisingDataset:
            def __init__(self, *args, **kwargs):
                raise AssertionError("dataset should not be constructed")

        with patch.dict(
            loader.DATASETS,
            {"mnist": RaisingDataset, "fashion-mnist": RaisingDataset},
        ):
            with self.assertRaisesRegex(ValueError, "Unknown dataset"):
                get_dataloaders(dataset="cifar10")


if __name__ == "__main__":
    unittest.main()
