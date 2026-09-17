"""Small CPU checks; run directly with Python using cached MNIST only."""

import contextlib
import io
import math
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from loader import get_dataloaders
from model import CNNDiscriminator, CnnGenerator, Discriminator, Generator
from run_baseline import build_models, parse_args, sample_generator
import train


METRICS = (
    "d_real_loss", "d_fake_loss", "d_loss", "g_loss", "d_real_score",
    "d_fake_score_before", "d_fake_score_after",
)


def parameters(model):
    return [parameter.detach().clone() for parameter in model.parameters()]


def unchanged(before, model):
    return all(torch.equal(old, new) for old, new in zip(before, model.parameters()))


class RecordingBCE(nn.BCELoss):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, prediction, target):
        loss = super().forward(prediction, target)
        self.calls.append((prediction.detach().clone(), target.detach().clone(), loss.item()))
        return loss


class BaselineChecks(unittest.TestCase):
    architecture = "mlp"

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(17)
        self.generator, self.discriminator = build_models(self.architecture, 64)
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        self.real = torch.rand(5, 1, 28, 28) * 2 - 1

    def update(self, loss_fn=None, real=None):
        return train.train_batch(
            self.generator, self.discriminator,
            self.real if real is None else real,
            self.g_optimizer, self.d_optimizer,
            nn.BCELoss() if loss_fn is None else loss_fn,
            noise_dim=64, noise_scale=1.0,
        )

    def assert_metrics(self, metrics):
        for name in METRICS:
            self.assertIsInstance(metrics[name], float, name)
            self.assertTrue(math.isfinite(metrics[name]), name)
        self.assertAlmostEqual(
            metrics["d_loss"], metrics["d_real_loss"] + metrics["d_fake_loss"],
            delta=1e-6,
        )
        for name in METRICS[4:]:
            self.assertGreaterEqual(metrics[name], 0.0)
            self.assertLessEqual(metrics[name], 1.0)

    def assert_gradients(self, model):
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(gradients)
        self.assertTrue(all(grad is not None for grad in gradients))
        self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients))
        self.assertTrue(any(torch.count_nonzero(grad).item() for grad in gradients))

    def test_shapes_and_ranges_for_single_and_regular_batches(self):
        with torch.no_grad():
            for size in (1, 8):
                with self.subTest(batch_size=size):
                    fake = self.generator(torch.randn(size, 64))
                    self.assertEqual(fake.shape, (size, 1, 28, 28))
                    self.assertTrue(torch.isfinite(fake).all())
                    self.assertTrue(((fake >= -1) & (fake <= 1)).all())
                    for images in (fake, torch.rand_like(fake) * 2 - 1):
                        prediction = self.discriminator(images)
                        self.assertEqual(prediction.shape, (size, 1))
                        self.assertTrue(torch.isfinite(prediction).all())
                        self.assertTrue(((prediction >= 0) & (prediction <= 1)).all())

    def test_bce_numerical_contract(self):
        loss = nn.BCELoss()
        half = torch.full((8, 1), 0.5)
        real_loss = loss(half, torch.ones_like(half))
        fake_loss = loss(half, torch.zeros_like(half))
        self.assertAlmostEqual(real_loss.item(), math.log(2), places=6)
        self.assertAlmostEqual(fake_loss.item(), math.log(2), places=6)
        self.assertAlmostEqual((real_loss + fake_loss).item(), 2 * math.log(2), places=6)
        low, high = torch.tensor([[0.1]]), torch.tensor([[0.9]])
        self.assertLess(loss(high, torch.ones_like(high)), loss(low, torch.ones_like(low)))
        self.assertGreater(loss(high, torch.zeros_like(high)), loss(low, torch.zeros_like(low)))

    def test_optimizer_boundaries_targets_and_reported_losses(self):
        g_before = parameters(self.generator)
        d_before = parameters(self.discriminator)
        d_after = []
        phases = []
        forwards = []
        generated = []
        loss_fn = RecordingBCE()
        original_d_step = self.d_optimizer.step
        original_g_step = self.g_optimizer.step

        # Stale gradients must be cleared before the phase-isolation assertions.
        for model in (self.generator, self.discriminator):
            for parameter in model.parameters():
                parameter.grad = torch.ones_like(parameter)

        def d_step(*args, **kwargs):
            self.assertEqual(phases, [])
            self.assertTrue(unchanged(g_before, self.generator))
            self.assertTrue(all(p.grad is None for p in self.generator.parameters()))
            self.assert_gradients(self.discriminator)
            result = original_d_step(*args, **kwargs)
            self.assertFalse(unchanged(d_before, self.discriminator))
            self.assertTrue(unchanged(g_before, self.generator))
            d_after.extend(parameters(self.discriminator))
            phases.append("d")
            return result

        def g_step(*args, **kwargs):
            self.assertEqual(phases, ["d"])
            self.assertTrue(unchanged(d_after, self.discriminator))
            self.assertTrue(all(p.grad is None for p in self.discriminator.parameters()))
            self.assertTrue(all(not p.requires_grad for p in self.discriminator.parameters()))
            self.assert_gradients(self.generator)
            result = original_g_step(*args, **kwargs)
            self.assertFalse(unchanged(g_before, self.generator))
            self.assertTrue(unchanged(d_after, self.discriminator))
            phases.append("g")
            return result

        def record_d_forward(module, inputs, output):
            forwards.append((inputs[0], tuple(phases)))

        def record_g_forward(module, inputs, output):
            generated.append(output)

        d_hook = self.discriminator.register_forward_hook(record_d_forward)
        g_hook = self.generator.register_forward_hook(record_g_forward)
        try:
            with patch.object(self.d_optimizer, "step", side_effect=d_step), \
                    patch.object(self.g_optimizer, "step", side_effect=g_step):
                metrics = self.update(loss_fn)
        finally:
            d_hook.remove()
            g_hook.remove()

        self.assertEqual(phases, ["d", "g"])
        self.assertEqual(len(generated), 1)
        self.assertEqual(len(forwards), 3)
        self.assertFalse(forwards[1][0].requires_grad)
        self.assertTrue(forwards[2][0].requires_grad)
        self.assertEqual(forwards[1][0].data_ptr(), generated[0].data_ptr())
        self.assertIs(forwards[2][0], generated[0])
        self.assertEqual([phase for _, phase in forwards], [(), (), ("d",)])
        self.assertEqual(len(loss_fn.calls), 3)
        for (prediction, target, _), value in zip(loss_fn.calls, (1, 0, 1)):
            self.assertEqual(target.shape, (5, 1))
            self.assertEqual(target.shape, prediction.shape)
            self.assertEqual(target.dtype, prediction.dtype)
            self.assertEqual(target.device, prediction.device)
            self.assertTrue(torch.equal(target, torch.full_like(prediction, value)))
        for name, call in zip(("d_real_loss", "d_fake_loss", "g_loss"), loss_fn.calls):
            self.assertAlmostEqual(metrics[name], call[2], places=6)
        for name, call in zip(METRICS[4:], loss_fn.calls):
            self.assertAlmostEqual(metrics[name], call[0].mean().item(), places=6)
        self.assertTrue(all(p.requires_grad for p in self.discriminator.parameters()))
        self.assert_metrics(metrics)

    def test_two_consecutive_updates_on_same_synthetic_batch(self):
        for iteration in range(2):
            with self.subTest(iteration=iteration):
                before_g = parameters(self.generator)
                before_d = parameters(self.discriminator)
                self.assert_metrics(self.update())
                self.assertFalse(unchanged(before_g, self.generator))
                self.assertFalse(unchanged(before_d, self.discriminator))
                for model in (self.generator, self.discriminator):
                    self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
                self.assert_gradients(self.generator)

    def test_discriminator_gradient_flags_restore_after_generator_failure(self):
        next(self.discriminator.parameters()).requires_grad_(False)
        flags = [p.requires_grad for p in self.discriminator.parameters()]
        calls = 0

        def fail_on_generator(prediction, target):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("deliberate generator loss failure")
            return nn.functional.binary_cross_entropy(prediction, target)

        with self.assertRaisesRegex(RuntimeError, "deliberate generator loss failure"):
            self.update(fail_on_generator)
        self.assertEqual(calls, 3)
        self.assertEqual([p.requires_grad for p in self.discriminator.parameters()], flags)

    def test_epoch_weights_short_batch_and_ignores_digit_labels(self):
        labels = torch.tensor([9, 8, 7, 6, 5])
        loader = DataLoader(TensorDataset(self.real, labels), batch_size=3, shuffle=False)
        batches = []
        loss_fn = nn.BCELoss()
        self.generator.eval()
        self.discriminator.eval()

        def fake_update(generator, discriminator, real_images, g_optimizer, d_optimizer,
                        passed_loss, noise_dim=64, noise_scale=1.0):
            self.assertIs(generator, self.generator)
            self.assertIs(discriminator, self.discriminator)
            self.assertIs(g_optimizer, self.g_optimizer)
            self.assertIs(d_optimizer, self.d_optimizer)
            self.assertIs(passed_loss, loss_fn)
            self.assertEqual(noise_dim, 64)
            self.assertEqual(noise_scale, 2.5)
            batches.append(real_images.clone())
            value = 0.2 if len(batches) == 1 else 0.7
            return {name: value * (index + 1) for index, name in enumerate(METRICS)}

        with patch.object(train, "train_batch", side_effect=fake_update):
            metrics = train.train_one_epoch(
                self.generator, 2.5, self.discriminator, loader, loss_fn, 64,
                self.d_optimizer, self.g_optimizer,
            )
        self.assertEqual([len(batch) for batch in batches], [3, 2])
        self.assertTrue(torch.equal(torch.cat(batches), self.real))
        for index, name in enumerate(METRICS):
            self.assertAlmostEqual(metrics[name], (3 * 0.2 + 2 * 0.7) / 5 * (index + 1))
        self.assertEqual(metrics["samples"], 5)
        self.assertEqual(metrics["batches"], 2)
        self.assertTrue(self.generator.training)
        self.assertTrue(self.discriminator.training)

    def test_actual_epoch_includes_single_image_final_batch(self):
        loader = DataLoader(TensorDataset(self.real, torch.arange(5)), batch_size=2)
        metrics = train.train_one_epoch(
            self.generator, 1.0, self.discriminator, loader, nn.BCELoss(), 64,
            self.d_optimizer, self.g_optimizer,
        )
        self.assert_metrics(metrics)
        self.assertEqual(metrics["samples"], 5)
        self.assertEqual(metrics["batches"], 3)

    def test_empty_epoch_is_rejected(self):
        loader = DataLoader(TensorDataset(self.real[:0], torch.empty(0, dtype=torch.long)))
        with self.assertRaises(ValueError):
            train.train_one_epoch(
                self.generator, 1.0, self.discriminator, loader, nn.BCELoss(), 64,
                self.d_optimizer, self.g_optimizer,
            )

    def test_monitoring_preserves_parameters_mode_and_training_rng(self):
        noise = torch.randn(8, 64, generator=torch.Generator().manual_seed(123))
        original_noise = noise.clone()
        for training in (True, False):
            with self.subTest(training=training):
                self.generator.train(training)
                before = parameters(self.generator)
                rng = torch.get_rng_state().clone()
                first = sample_generator(self.generator, noise)
                second = sample_generator(self.generator, noise)
                self.assertEqual(first.shape, (8, 1, 28, 28))
                self.assertEqual(first.device.type, "cpu")
                self.assertFalse(first.requires_grad)
                self.assertIsNone(first.grad_fn)
                self.assertTrue(torch.equal(first, second))
                self.assertTrue(torch.equal(noise, original_noise))
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertTrue(unchanged(before, self.generator))
                self.assertEqual(self.generator.training, training)

    def test_monitoring_restores_mode_on_error(self):
        self.generator.train()
        with patch.object(self.generator, "forward", side_effect=RuntimeError("sample failure")):
            with self.assertRaisesRegex(RuntimeError, "sample failure"):
                sample_generator(self.generator, torch.zeros(1, 64))
        self.assertTrue(self.generator.training)

    def test_cached_mnist_splits_normalization_and_independent_rng(self):
        data_root = Path(os.environ.get(
            "AI_ROADMAP_DATA", "/home/rui.gao/data/ai_research_roadmap"
        )).expanduser()
        if not (data_root / "raw/torchvision/MNIST").is_dir():
            self.skipTest(f"MNIST prerequisite missing: {data_root / 'raw/torchvision/MNIST'}")
        rng = torch.get_rng_state().clone()
        with patch("torchvision.datasets.MNIST.download", side_effect=AssertionError("Download forbidden")):
            # Explicit roots override environment; the default honors the environment.
            with patch.dict(os.environ, {"AI_ROADMAP_DATA": "/missing/test-only-data-root"}):
                loaders = get_dataloaders(batch_size=8, seed=42, data_root=data_root)
            with patch.dict(os.environ, {"AI_ROADMAP_DATA": str(data_root)}):
                repeated = get_dataloaders(batch_size=8, seed=42)
        train_loader, val_loader, test_loader = loaders
        self.assertEqual([len(loader.dataset) for loader in loaders], [55000, 5000, 10000])
        train_ids = set(train_loader.dataset.indices)
        val_ids = set(val_loader.dataset.indices)
        self.assertFalse(train_ids & val_ids)
        self.assertEqual(train_ids | val_ids, set(range(60000)))
        expected_indices = torch.randperm(60000, generator=torch.Generator().manual_seed(42)).tolist()
        self.assertEqual(list(train_loader.dataset.indices), expected_indices[:55000])
        self.assertEqual(list(val_loader.dataset.indices), expected_indices[55000:])
        self.assertTrue(train_loader.dataset.dataset.train)
        self.assertFalse(test_loader.dataset.train)
        self.assertEqual(Path(train_loader.dataset.dataset.root), data_root / "raw/torchvision")
        for original, again in zip(loaders[:2], repeated[:2]):
            self.assertEqual(list(original.dataset.indices), list(again.dataset.indices))
        for loader, again in zip(loaders, repeated):
            images, labels = next(iter(loader))
            other_images, other_labels = next(iter(again))
            self.assertEqual(images.shape, (8, 1, 28, 28))
            self.assertEqual(labels.shape, (8,))
            self.assertTrue(torch.isfinite(images).all())
            self.assertTrue(((images >= -1) & (images <= 1)).all())
            self.assertTrue(torch.equal(images, other_images))
            self.assertTrue(torch.equal(labels, other_labels))
        raw = val_loader.dataset.dataset
        for index in val_loader.dataset.indices[:8]:
            image, label = raw[index]
            expected = raw.data[index].to(torch.float32).unsqueeze(0) / 255 * 2 - 1
            self.assertTrue(torch.allclose(image, expected, atol=1e-7, rtol=0))
            self.assertEqual(label, raw.targets[index].item())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))


class CnnBaselineChecks(BaselineChecks):
    architecture = "cnn"


class ArchitectureSelectionChecks(unittest.TestCase):
    def test_default_preserves_mlp_classes_and_initialization(self):
        args = parse_args(["--out-dir", "unused-test-output"])
        self.assertEqual(args.architecture, "mlp")
        torch.manual_seed(42)
        actual = build_models(args.architecture, args.noise_dim)
        torch.manual_seed(42)
        expected = (Generator(64, 784), Discriminator(28))
        for selected, original in zip(actual, expected):
            self.assertIs(type(selected), type(original))
            self.assertTrue(unchanged(parameters(original), selected))

    def test_explicit_cnn_selection_honors_noise_width(self):
        args = parse_args(["--out-dir", "unused-test-output",
                           "--architecture", "cnn", "--noise-dim", "13"])
        generator, discriminator = build_models(args.architecture, args.noise_dim)
        self.assertIsInstance(generator, CnnGenerator)
        self.assertIsInstance(discriminator, CNNDiscriminator)
        with torch.no_grad():
            images = generator(torch.randn(2, 13))
            self.assertEqual(images.shape, (2, 1, 28, 28))
            self.assertEqual(discriminator(images).shape, (2, 1))

    def test_unknown_architecture_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            parse_args(["--out-dir", "unused-test-output", "--architecture", "unknown"])
        self.assertEqual(error.exception.code, 2)
        with self.assertRaises(ValueError):
            build_models("unknown", 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
