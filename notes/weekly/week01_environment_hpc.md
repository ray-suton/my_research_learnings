# Week 1 - Environment and HPC Setup

## Goals

- Verify the local PyTorch environment.
- Verify GPU access through SLURM.
- Run a tiny autograd sanity check.
- Run a tiny linear-regression training loop.
- Save logs and plots for the Week 1 experiment.

## Environment

- Environment file: `environment.yml`
- Main packages used directly this week: `torch` and `matplotlib`
- HPC scheduler: SLURM
- GPU tested: NVIDIA RTX 5000 Ada Generation

## What works

- HPC/SLURM GPU test passed
- PyTorch autograd test passed
- Linear-regression training script passed
- Training log and plots were saved under `phase01_research_engineering/week01_environment_hpc/results/`

## Result files

scripts/smoke_tests/gpu_smoke_test.py
scripts/slurm/gpu_smoke_test.slurm
scripts/smoke_tests/torch_autograd_test.py
phase01_research_engineering/week01_environment_hpc/linear_regression_torch.py


Autograd script run:
- Status: passed
- Output value: 127
- Gradient: 50

Linear regression script run:
- Status: passed as a training script
- Loss decreased during training
- Final loss: 0.03953052684664726
- Learned weight: 3.0077109336853027
- Learned bias: 2.409120559692383

Plot check:
- Loss plot created at `phase01_research_engineering/week01_environment_hpc/results/plots/linear_reg_loss.png`
- PNG validity: valid 640 x 480 image
- Visual check: readable loss curve, steep early drop, then flat convergence

Prediction-vs-actual plot:
- Created at `phase01_research_engineering/week01_environment_hpc/results/plots/linear_reg_pred_vs_actual.png`
- PNG validity: valid 700 x 500 image
- Visual check: actual points closely follow the predicted line


All intended checks pass.

## New things learned

SLURM allocates a requested compute node/partition on the HPC cluster. Direct Python runs on the current interactive node, while sbatch runs on the node assigned by SLURM.

got it. so slurm provides like a rental node and u give it a job and it does it for u on it's gpus

SLURM is like a cluster resource manager. You give it a job request:

I need 1 GPU, 2 CPUs, 8 GB RAM, for 10 minutes, on the gpu partition.

Then SLURM:

1. finds a node that matches
2. reserves those resources for you
3. runs your script there
4. saves output/error logs
5. releases the resources when done

So your terminal command is just:

sbatch job.slurm

And inside the job file you tell SLURM what to run:

source .venv/bin/activate
python train.py

The important mental model:

Interactive shell = you drive the current machine directly.
SLURM job = you submit instructions; scheduler runs them on an allocated machine.

“Rental node” is a good analogy, except it’s not paid per job by you directly; it’s allocated from the shared HPC pool according to cluster rules, partitions, QOS, and availability.

Use `sinfo`, `sbatch`, and `squeue` to check cluster/job status. Use `scontrol show partition` to inspect a partition.

Remember to run `source .venv/bin/activate` before launching local Python commands.

- `@` serves the same purpose as `torch.matmul` for matrix multiplication.

- `with torch.no_grad():` lets parameter updates happen without tracking those update operations in autograd.

- `w.grad.zero_()` and `b.grad.zero_()` reset accumulated gradients after each update.

- Gradients accumulate by default, so they must be reset between training steps.

- `torch.item()` extracts a plain Python number from a one-element tensor.

- `torch.rand` samples values from 0 to 1.

- `torch.randn` samples values centered around 0 with standard deviation 1.
