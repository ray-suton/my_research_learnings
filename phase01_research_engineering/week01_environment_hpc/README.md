# Week 1 - Environment and HPC

This folder contains the Week 1 bootstrap work for the AI research roadmap. The goal was to verify the local PyTorch setup, confirm GPU access through SLURM, and run a tiny synthetic linear-regression training loop before starting larger experiments.

## Scope

Week 1 covers:

- local repository and Python environment setup
- PyTorch import and autograd sanity checks
- GPU smoke testing on the HPC through SLURM
- a minimal hand-written linear-regression training loop
- saved logs and plots for the training run

## Files

```text
phase01_research_engineering/week01_environment_hpc/
  README.md
  linear_regression_torch.py
  results/
    logs/
      linear_reg_training_log.txt
    plots/
      linear_reg_loss.png
      linear_reg_pred_vs_actual.png
```

Related Week 1 files outside this folder:

```text
scripts/smoke_tests/gpu_smoke_test.py
scripts/smoke_tests/torch_autograd_test.py
scripts/slurm/gpu_smoke_test.slurm
notes/weekly/week01_environment_hpc.md
results/gpu_smoke_test_143178.out
results/gpu_smoke_test_143178.err
```

## How To Rerun

From the repository root, activate the virtual environment first:

```bash
source .venv/bin/activate
```

Run the autograd sanity check:

```bash
python scripts/smoke_tests/torch_autograd_test.py
```

Run the tiny linear-regression experiment:

```bash
python phase01_research_engineering/week01_environment_hpc/linear_regression_torch.py
```

Run the GPU smoke test through SLURM:

```bash
sbatch scripts/slurm/gpu_smoke_test.slurm
```

## Verified Results

The SLURM GPU smoke test passed on:

- GPU: NVIDIA RTX 5000 Ada Generation
- GPU memory: about 31.5 GB
- PyTorch: 2.11.0+cu128
- CUDA available: true

The linear-regression run passed with:

- final loss: 0.03953052684664726
- learned weight: 3.0077109336853027
- learned bias: 2.409120559692383
- sample count: 100
- learning rate: 0.01
- epochs run: 2000

The synthetic data rule is:

```text
y = 3x + 2 + noise
```

The learned weight is greater to 3, and the learned bias is reasonable because the noise term is sampled from `torch.rand`, which adds positive noise.

## Output Artifacts

Training log:

```text
phase01_research_engineering/week01_environment_hpc/results/logs/linear_reg_training_log.txt
```

Loss plot:

```text
phase01_research_engineering/week01_environment_hpc/results/plots/linear_reg_loss.png
```

Prediction-vs-actual plot:

```text
phase01_research_engineering/week01_environment_hpc/results/plots/linear_reg_pred_vs_actual.png
```

## What This Week Proves

Week 1 proves that the repo can:

- run PyTorch locally
- compute gradients with autograd
- submit a GPU job through SLURM
- train a tiny model end to end
- save a reproducible log and plots

## Next Step

The next roadmap stage is Week 2: build an MLP classifier on MNIST or Fashion-MNIST with a proper dataset loader, train/eval loop, metrics, checkpointing, and plots.

