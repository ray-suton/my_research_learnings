"""CPU segment/restart equivalence and runner provenance checks."""

import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import run_baseline as runner


def assert_nested(test, left, right):
    if isinstance(left, torch.Tensor):
        test.assertEqual(left.dtype, right.dtype)
        test.assertEqual(left.shape, right.shape)
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            assert_nested(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_nested(test, a, b)
    else:
        test.assertEqual(left, right)


class RunControlChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_dataset_cli_and_execution_controls(self):
        self.assertEqual(runner.parse_args(["--out-dir", "unused"]).dataset, "mnist")
        args = runner.parse_args(["--out-dir", "unused", "--dataset", "fashion-mnist",
                                  "--epochs", "20", "--chunk-epochs", "5"])
        self.assertEqual((args.dataset, args.epochs, args.chunk_epochs), ("fashion-mnist", 20, 5))
        for extra in [["--dataset", "unknown"], ["--chunk-epochs", "0"],
                      ["--resume", "--lr", "0.01"], ["--resume", "--architecture", "mlp"]]:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    runner.parse_args(["--out-dir", "unused", *extra])

    def test_restored_epoch_matches_uninterrupted_for_both_architectures(self):
        for architecture in ["mlp", "cnn"]:
            with self.subTest(architecture=architecture):
                torch.manual_seed(314)
                dataset = TensorDataset(torch.rand(7, 1, 28, 28) * 2 - 1, torch.arange(7))
                def setup():
                    g, d = runner.build_models(architecture, 64)
                    go, do = [torch.optim.Adam(net.parameters(), lr=0.0002, betas=(0.5, 0.999)) for net in (g, d)]
                    loader = DataLoader(dataset, batch_size=3, shuffle=True,
                                        generator=torch.Generator().manual_seed(42))
                    return g, d, go, do, loader
                g, d, go, do, loader = setup()
                criterion = torch.nn.BCELoss()
                runner.train_one_epoch(g, 1.0, d, loader, criterion, 64, do, go)
                saved = copy.deepcopy(runner.capture_training_state(1, g, d, go, do, loader, {}))
                expected_metrics = runner.train_one_epoch(g, 1.0, d, loader, criterion, 64, do, go)
                expected = copy.deepcopy(runner.capture_training_state(2, g, d, go, do, loader, {}))
                torch.manual_seed(999)
                g, d, go, do, loader = setup()
                runner.restore_training_state(saved, g, d, go, do, loader)
                actual_metrics = runner.train_one_epoch(g, 1.0, d, loader, criterion, 64, do, go)
                assert_nested(self, actual_metrics, expected_metrics)
                assert_nested(self, runner.capture_training_state(2, g, d, go, do, loader, {}), expected)

    def test_pause_resume_preserves_config_and_defers_finalization(self):
        with tempfile.TemporaryDirectory(prefix="week08-run-control-") as temporary:
            root = Path(temporary)
            raw = root / "data"
            raw.mkdir()
            for name in ["train-images-idx3-ubyte", "train-labels-idx1-ubyte",
                         "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]:
                (raw / name).write_bytes(name.encode())

            class FixtureImages(Dataset):
                classes = [f"garment-{i}" for i in range(10)]
                raw_folder = str(raw)
                def __len__(self):
                    return 192
                def __getitem__(self, index):
                    return torch.full((1, 28, 28), (index % 17) / 8 - 1), index % 10

            def loaders(batch_size, seed, data_root, num_workers, dataset):
                self.assertEqual(dataset, "fashion-mnist")
                full = FixtureImages()
                return tuple(DataLoader(Subset(full, indices), batch_size=batch_size, shuffle=shuffle,
                                        generator=torch.Generator().manual_seed(seed + offset))
                             for offset, (indices, shuffle) in enumerate([
                                 (list(range(128)), True), (list(range(128, 192)), False), (list(range(20)), False)]))

            out = root / "run"
            argv = ["run_baseline.py", "--out-dir", str(out), "--checkpoint-dir", str(root / "checkpoints"),
                    "--data-root", str(raw), "--dataset", "fashion-mnist", "--architecture", "mlp",
                    "--epochs", "2", "--chunk-epochs", "1", "--batch-size", "16", "--threads", "1", "--device", "cpu"]
            with patch.object(runner, "get_dataloaders", side_effect=loaders), \
                    patch.object(runner, "save_grid", side_effect=lambda images, path, title: path.write_text(title)), \
                    patch.object(runner, "save_curves", side_effect=lambda history, path: path.write_text("curves")), \
                    patch.object(runner, "save_comparisons") as comparisons, \
                    contextlib.redirect_stdout(io.StringIO()):
                with patch.object(sys, "argv", argv):
                    runner.main()
                config_bytes = (out / "config.json").read_bytes()
                config = json.loads(config_bytes)
                record = json.loads((out / "metrics.json").read_text())
                self.assertEqual((record["status"], len(record["history"])), ("paused", 1))
                self.assertEqual((config["epochs"], config["selected_epoch"]), (2, 2))
                self.assertFalse((out / "final_noise_seed456.png").exists())
                self.assertFalse((out / "discriminator_by_class.csv").exists())
                self.assertIn("Fashion-MNIST", (out / "real_validation.png").read_text())
                comparisons.assert_not_called()
                with patch.object(sys, "argv", ["run_baseline.py", "--resume", "--out-dir", str(out)]):
                    runner.main()
                self.assertEqual(config_bytes, (out / "config.json").read_bytes())
                record = json.loads((out / "metrics.json").read_text())
                self.assertEqual((record["status"], record["selected_epoch"]), ("completed", 2))
                self.assertEqual([row["epoch"] for row in record["history"]], [1, 2])
                self.assertEqual([row["status"] for row in record["invocations"]], ["paused", "completed"])
                self.assertAlmostEqual(record["total_seconds"], sum(row["active_seconds"] for row in record["invocations"]))
                self.assertTrue((out / "final_noise_seed456.png").exists())
                self.assertTrue((out / "discriminator_by_class.csv").exists())
                self.assertNotIn("discriminator_by_digit", record)
                self.assertEqual(len(record["discriminator_by_class"]), 20)
                self.assertTrue(all("class_id" in row and "class_name" in row and "digit" not in row
                                    for row in record["discriminator_by_class"]))
                self.assertEqual(comparisons.call_args.kwargs["dataset_name"], "Fashion-MNIST")
                self.assertEqual(comparisons.call_args.kwargs["class_name"], "garment-8")
                with self.assertRaisesRegex(ValueError, "cleanly paused"):
                    runner.load_paused_run(out, PROJECT)

    def test_resume_rejects_inconsistent_or_changed_provenance(self):
        with tempfile.TemporaryDirectory(prefix="week08-resume-guards-") as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "model.py").write_text("original")
            checkpoints = root / "checkpoints"
            checkpoints.mkdir()
            config = {"device": "cpu", "out_dir": str(root), "checkpoint_dir": str(checkpoints),
                      "epochs": 2, "source_hashes": {"model.py": runner.sha256(source / "model.py")}}
            path = checkpoints / "epoch_01.pt"
            torch.save({"epoch": 1, "config": config}, path)
            record = {"status": "paused", "history": [{"epoch": 1}],
                      "checkpoints": [{"epoch": 1, "path": str(path), "sha256": runner.sha256(path)}]}
            runner.write_json(root / "config.json", config)
            runner.write_json(root / "metrics.json", record)
            runner.load_paused_run(root, source)
            (source / "model.py").write_text("changed")
            with self.assertRaisesRegex(ValueError, "Source changed"):
                runner.load_paused_run(root, source)
            (source / "model.py").write_text("original")
            extra = checkpoints / "epoch_02.pt"
            extra.write_bytes(b"unrecorded")
            with self.assertRaisesRegex(ValueError, "Unrecorded"):
                runner.load_paused_run(root, source)
            extra.unlink()
            wrong = dict(config, epochs=3)
            runner.write_json(root / "config.json", wrong)
            with self.assertRaisesRegex(ValueError, "configuration/epoch"):
                runner.load_paused_run(root, source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
