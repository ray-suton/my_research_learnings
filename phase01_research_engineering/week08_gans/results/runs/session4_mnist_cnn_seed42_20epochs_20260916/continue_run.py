"""Continue this interrupted CPU experiment at its last complete epoch."""

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

RUN = Path(__file__).resolve().parent
PROJECT = RUN.parents[2]
sys.path.insert(0, str(PROJECT))

import run_baseline as runner
from audit_and_compare import assert_equal


def capture(epoch, generator, discriminator, go, do, loader, config):
    return {
        "epoch": epoch, "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "g_optimizer": go.state_dict(), "d_optimizer": do.state_dict(),
        "config": config, "torch_rng_state": torch.get_rng_state(),
        "loader_rng_state": loader.generator.get_state(),
    }


def restore(state, generator, discriminator, go, do, loader):
    generator.load_state_dict(state["generator"], strict=True)
    discriminator.load_state_dict(state["discriminator"], strict=True)
    go.load_state_dict(state["g_optimizer"])
    do.load_state_dict(state["d_optimizer"])
    torch.set_rng_state(state["torch_rng_state"])
    loader.generator.set_state(state["loader_rng_state"])
    assert torch.equal(torch.get_rng_state(), state["torch_rng_state"])
    assert torch.equal(loader.generator.get_state(), state["loader_rng_state"])


def optimizers(generator, discriminator, lr=0.0002):
    return tuple(torch.optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999))
                 for net in (generator, discriminator))


def check_resume_equivalence():
    torch.manual_seed(314)
    real = torch.rand(7, 1, 28, 28) * 2 - 1
    dataset = TensorDataset(real, torch.arange(7))
    def make_loader():
        return DataLoader(dataset, batch_size=3, shuffle=True,
                          generator=torch.Generator().manual_seed(42))
    g, d = runner.build_models("cnn", 64)
    go, do = optimizers(g, d)
    loader = make_loader()
    loss = torch.nn.BCELoss()
    runner.train_one_epoch(g, 1.0, d, loader, loss, 64, do, go)
    saved = copy.deepcopy(capture(1, g, d, go, do, loader, {}))
    expected_metrics = runner.train_one_epoch(g, 1.0, d, loader, loss, 64, do, go)
    expected = copy.deepcopy(capture(2, g, d, go, do, loader, {}))
    torch.manual_seed(999)
    g, d = runner.build_models("cnn", 64)
    go, do = optimizers(g, d)
    loader = make_loader()
    restore(saved, g, d, go, do, loader)
    actual_metrics = runner.train_one_epoch(g, 1.0, d, loader, loss, 64, do, go)
    assert_equal(actual_metrics, expected_metrics)
    assert_equal(capture(2, g, d, go, do, loader, {}), expected)
    print("PASS: interrupted/restored CNN epoch exactly matches uninterrupted metrics, models, Adam states and both RNG states", flush=True)


def main():
    config = json.loads((RUN / "config.json").read_text())
    record = json.loads((RUN / "metrics.json").read_text())
    assert config["architecture"] == "cnn" and config["device"] == "cpu"
    assert record["status"] in {"running", "interrupted"}
    last = record["checkpoints"][-1]
    start_epoch = last["epoch"]
    assert 0 < start_epoch < config["epochs"]
    assert [r["epoch"] for r in record["history"]] == list(range(1, start_epoch + 1))
    assert [r["epoch"] for r in record["checkpoints"]] == list(range(1, start_epoch + 1))
    checkpoint_dir = Path(config["checkpoint_dir"])
    for epoch in range(start_epoch + 1, config["epochs"] + 1):
        if (checkpoint_dir / f"epoch_{epoch:02d}.pt").exists():
            raise FileExistsError(f"Unrecorded checkpoint at epoch {epoch}; inspect before continuing")
    for name, expected in config["source_hashes"].items():
        assert runner.sha256(PROJECT / name) == expected, name
    assert runner.sha256(Path(last["path"])) == last["sha256"]
    state = torch.load(last["path"], map_location="cpu", weights_only=True)
    assert state["epoch"] == start_epoch and state["config"] == config
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config["seed"])
    train, validation, _ = runner.get_dataloaders(
        config["batch_size"], config["seed"], config["data_root"], config["num_workers"])
    for loader, key in [(train, "train_indices_sha256"), (validation, "validation_indices_sha256")]:
        assert hashlib.sha256(json.dumps(loader.dataset.indices).encode()).hexdigest() == config[key]
    g, d = runner.build_models("cnn", config["noise_dim"])
    go, do = optimizers(g, d, config["lr"])
    fixed = torch.randn(64, config["noise_dim"], generator=torch.Generator().manual_seed(123))
    final = torch.randn(64, config["noise_dim"], generator=torch.Generator().manual_seed(456))
    validation_images = torch.stack([validation.dataset[i][0] for i in range(64)])
    training_images = torch.stack([train.dataset[i][0] for i in range(64)])
    references = {"fixed_train_real": training_images,
                  "fixed_validation_real": validation_images,
                  "fixed_initial_fake": runner.sample_generator(g, fixed)}
    assert runner.fixed_scores(d, references, torch.device("cpu")) == record["initial_fixed_scores"]
    provenance = {
        "reason": "Initial process terminated with exit 143; no training process remained",
        "restored_epoch": start_epoch, "checkpoint": last,
        "script_sha256": runner.sha256(Path(__file__)),
        "command": [sys.executable, *sys.argv],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "restored": ["generator", "discriminator", "g_optimizer", "d_optimizer",
                     "torch_rng_state", "loader_rng_state"],
        "original_config_preserved": True,
        "timing_note": "Sum of epoch training_seconds covers completed epochs only; wall elapsed includes interruption/recovery and any lost partial epoch",
    }
    if (RUN / "continuation.json").exists():
        raise FileExistsError("Continuation provenance already exists")
    runner.write_json(RUN / "continuation.json", provenance)
    record.update(status="running", resumed_from_epoch=start_epoch,
                  initial_process_exit_code=143)
    runner.write_json(RUN / "metrics.json", record)
    criterion = torch.nn.BCELoss()
    restore(state, g, d, go, do, train)
    print(f"Restored epoch {start_epoch}; continuing through epoch {config['epochs']}", flush=True)
    start = time.perf_counter()
    try:
        for epoch in range(start_epoch + 1, config["epochs"] + 1):
            tick = time.perf_counter()
            values = runner.train_one_epoch(g, 1.0, d, train, criterion, config["noise_dim"], do, go)
            values.update(epoch=epoch, training_seconds=time.perf_counter() - tick)
            values.update(runner.fixed_scores(d, references, torch.device("cpu")))
            images = runner.sample_generator(g, fixed)
            runner.save_grid(images, RUN / f"fixed_noise_epoch_{epoch:02d}.png",
                             f"CNN GAN after epoch {epoch}: fixed noise seed 123")
            path = checkpoint_dir / f"epoch_{epoch:02d}.pt"
            torch.save(capture(epoch, g, d, go, do, train, config), path)
            record["history"].append(values)
            record["checkpoints"].append({"epoch": epoch, "path": str(path), "sha256": runner.sha256(path)})
            runner.write_json(RUN / "metrics.json", record)
            runner.save_curves(record["history"], RUN / "training_curves.png")
            print(f"epoch={epoch} D={values['d_loss']:.6f} G={values['g_loss']:.6f} seconds={values['training_seconds']:.2f}", flush=True)
        final_images = runner.sample_generator(g, final)
        runner.save_grid(final_images, RUN / "final_noise_seed456.png",
                         f"Final epoch {config['epochs']}: independent noise seed 456")
        runner.save_comparisons(validation_images, final_images, validation.dataset[0][1], RUN, config["epochs"])
        audit = []
        for split, dataset in [("train", train.dataset), ("validation", validation.dataset)]:
            audit.extend(runner.score_real_by_digit(d, dataset, split, torch.device("cpu")))
        with (RUN / "discriminator_by_digit.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
            writer.writeheader()
            writer.writerows(audit)
        now = datetime.now(timezone.utc)
        record.update(status="completed", selected_epoch=config["epochs"], discriminator_by_digit=audit,
                      completed_at=now.isoformat(), continuation_seconds=time.perf_counter() - start,
                      total_seconds=(now - datetime.fromisoformat(config["started_at"])).total_seconds(),
                      total_seconds_includes_recovery_delay=True,
                      continuation_peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        provenance.update(completed_at=now.isoformat(), continuation_seconds=record["continuation_seconds"])
        runner.write_json(RUN / "continuation.json", provenance)
        print(f"COMPLETED: {RUN}", flush=True)
    except Exception as exc:
        record.update(status="failed", error=repr(exc), continuation_seconds=time.perf_counter() - start)
        raise
    finally:
        runner.write_json(RUN / "metrics.json", record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if args.check_only:
        check_resume_equivalence()
    else:
        main()
