# Configuration

The runner now supports `--dataset mnist|fashion-mnist`, with MNIST remaining
the default. Resolved configs identify the dataset class, label names, hashes
of the four raw IDX files, and split-index hashes. Equal split indices alone
cannot identify two datasets of the same size.

For long CPU jobs, declare the total budget once (for example `--epochs 20`)
and use `--chunk-epochs 5`. Each invocation ends after a complete checkpoint
with status `paused`. Continue with `--resume --out-dir <same-run-directory>`;
the saved configuration is loaded without changing the target budget.
Only `--chunk-epochs` may be overridden during resume. Core source, data,
environment, checkpoint identity, and original reference scores must match.
Only cleanly paused CPU runs are supported by this resume path. Arbitrary
interrupted/inconsistent runs require inspection, and CUDA restoration is
not claimed. Final panels and the class audit run only at the declared final
epoch. Active timing is summed across invocations and excludes gaps/startup.

See [fashion_mnist_extension.md](../fashion_mnist_extension.md) for the
explicitly requested CNN-only experiment. The following describes retained
MNIST results and compatible defaults.

The completed Session 3 run saves its resolved [config.json](../results/runs/session3_mnist_mlp_seed42_20260915/config.json)
before training. It includes the exact command, data paths/split hashes,
software versions, source hashes, model parameter counts, loss convention,
noise seeds, optimizer settings, device, and checkpoint-selection rule.

The runner is [run_baseline.py](../run_baseline.py). Main defaults are five
epochs, seed 42, noise width 64, batch 128, Adam learning rate 0.0002 and betas
(0.5, 0.999), and two CPU threads. D uses the learner's sigmoid output with
mean BCE and LeakyReLU slope 0.01. Monitoring seed 123 and final seed 456 are
independent of training noise. The measured run selected epoch 5 in advance.

`--architecture mlp` is the default and preserves the original baseline.
`--architecture cnn` selects `CnnGenerator` and `CNNDiscriminator`. Both
choices use the same MNIST interface and update loop. New resolved configs
include architecture choice, class names, complete layer descriptions, and
parameter counts. The completed [Session 4 config](../results/runs/session4_mnist_cnn_seed42_20260916/config.json)
records the five-epoch CNN run; [session4.md](../session4.md) contains its
declared comparison and command.

Use a fresh --out-dir for each attempt. By default checkpoints go under
AI_ROADMAP_DATA/checkpoints/week08_gans/<run-directory-name>; --checkpoint-dir
can select another explicitly recorded external location. The runner refuses
to reuse an existing output or checkpoint directory. Do not change a completed
run's configuration retrospectively.
