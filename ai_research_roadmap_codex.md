# AI Research Roadmap - Continual Codex-Ready Plan

## Starting point

The student has completed:

- Andrew Ng / DeepLearning.AI Machine Learning Specialization
- Andrew Ng / DeepLearning.AI Deep Learning Specialization

The student wants to become a strong incoming AI student researcher with both broad knowledge and real coding ability.

Available time:

- 1.5 hours per weekday
- Monday to Friday only
- No planned weekend work
- Around 7.5 hours per full week

Hardware:

- One school HPC GPU
- Exact GPU VRAM unknown at the beginning

Main goal:

- Learn the major AI model families and research directions
- Build small versions of important architectures
- Fine-tune and evaluate real pretrained models
- Learn how to run experiments cleanly
- Produce a portfolio of research-style repos and notes
- Build a continual research practice that can keep extending beyond the first year

Operating principle:

Every topic follows this loop:

1. Read one paper or authoritative tutorial.
2. Implement the smallest possible working version.
3. Run a controlled experiment.
4. Evaluate with a metric or comparison table.
5. Write a short lab note.

This roadmap starts on Tuesday, June 16, 2026.

Because the start date is a Tuesday, Week 1 is a 4-day bootstrap week from Tuesday to Friday. Week 2 begins on Monday, June 22, 2026, and the normal Monday-Friday rhythm continues after that.

The first 52 weeks are the Year 1 core, not the end of the project. After Week 52, continue with extension tracks, deeper reproductions, safety/evaluation work, systems work, and original research cycles.

---

# How Codex should use this file

Use this file as the project specification.

When asked to implement tasks from this roadmap:

- Create clean, minimal, readable code.
- Prefer scripts over messy notebooks unless visualization is the main task.
- Keep experiments reproducible.
- Use config files where useful.
- Keep README files updated.
- Save logs, plots, and results in each project folder.
- Avoid huge models or long training jobs unless explicitly requested.
- Make all scripts runnable from the command line.
- Add short comments only where they clarify non-obvious logic.
- Do not assume a specific HPC scheduler beyond SLURM unless the user confirms otherwise.
- Use placeholders for HPC account, partition, and time limits.

Recommended repo name:

```text
ai-research-roadmap
```

Recommended repo structure:

```text
ai-research-roadmap/
  README.md
  ROADMAP.md
  TODO.md
  requirements.txt
  environment.yml
  .gitignore
  scripts/
    slurm/
    smoke_tests/
  notes/
    weekly/
    papers/
  results/
  phase01_research_engineering/
  phase02_llms/
  phase03_efficiency_interpretability/
  phase04_vision_multimodal_generation/
  phase05_embodied_rl/
  phase06_capstone/
```

Standard project structure inside each experiment:

```text
project_name/
  README.md
  config.yaml
  dataset.py
  model.py
  train.py
  eval.py
  tests/
  requirements.txt
  results/
    runs/
      YYYYMMDD_HHMM_experiment_name/
        config.yaml
        metrics.json
        stdout.log
        notes.md
        plots/
    plots/
    logs/
    checkpoints/
```

Standard Friday lab-note template:

```text
# Week N - Title

## Goal

## Reading completed

## What I implemented

## Experiment setup

## Results

## What failed

## What I understand better now

## Next steps
```

Research operating system:

Every project, from Week 2 onward, should maintain a lightweight research record. This is more important than adding many extra model families.

Before coding:

- State the hypothesis or learning question.
- Define the baseline.
- Define the metric before seeing results.
- Define the train/validation/test split or evaluation set.
- Record what result would falsify or weaken the idea.
- Record expected runtime, memory, and hardware.

During experiments:

- Save each meaningful run under `results/runs/YYYYMMDD_HHMM_experiment_name/`.
- Save `config.yaml`, `metrics.json`, `stdout.log`, `notes.md`, and plots.
- Use fixed seeds when comparing methods.
- Run at least 3 repeated runs for important comparisons when compute allows.
- Track runtime and GPU/CPU memory when relevant.
- Keep a failure-case table, not only aggregate metrics.
- Do not trust one lucky run.

After experiments:

- Compare against the baseline.
- Run at least one ablation for nontrivial claims.
- Write limitations honestly.
- Record whether the result supports, weakens, or fails to test the hypothesis.
- Write the next reproduction idea and one possible extension.

Paper-note template:

```text
# Paper Title

## Problem
## Method
## Key equation or algorithm
## Strongest empirical result
## Weakest assumption
## What I would reproduce
## What I would extend
## How this connects to my roadmap
```

Monthly research checkpoint:

At the end of every 4 weeks, write a short checkpoint note:

- Best experiment result.
- Most important failure.
- What was overclaimed or under-tested.
- One paper-method insight.
- One systems or reproducibility issue.
- One safety/evaluation issue.
- Updated capstone or project ideas.
- Outside feedback received or requested.

Monthly mini-paper:

Every 4 to 8 weeks, turn one project into a mini-paper:

```text
# Title

## Abstract
## Introduction
## Method
## Experiments
## Results
## Limitations
## Future work
## Reproduction instructions
```

Data, safety, and environment rules:

- Record dataset source, license, preprocessing, and redistribution constraints.
- Keep large files out of Git.
- Record where large files, checkpoints, and logs live.
- Keep Hugging Face tokens, API keys, and gated-model credentials out of the repo.
- Add prompt-injection, data-leakage, contamination, jailbreak, and misuse-risk notes for LLM, RAG, and agent projects.
- Add exact CUDA/PyTorch install notes once the HPC environment is known.
- Add a pinned environment artifact later, such as `requirements.lock.txt`, `uv.lock`, or `conda-lock`.

---

# Exact Week 1 plan

## Week 1 dates

- Tuesday, June 16, 2026
- Wednesday, June 17, 2026
- Thursday, June 18, 2026
- Friday, June 19, 2026

This is a bootstrap week. The goal is not to train a serious model yet. The goal is to create a clean research workspace, verify the GPU/HPC workflow, and run a tiny PyTorch sanity experiment.

## Week 1 outcomes

By the end of Friday, the repo should contain:

```text
ai-research-roadmap/
  README.md
  ROADMAP.md
  TODO.md
  requirements.txt
  environment.yml
  .gitignore
  scripts/
    smoke_tests/
      gpu_smoke_test.py
      torch_autograd_test.py
    slurm/
      gpu_smoke_test.slurm
  notes/
    weekly/
      week01_environment_hpc.md
  phase01_research_engineering/
    week01_environment_hpc/
      README.md
      linear_regression_torch.py
      results/
        logs/
        plots/
```

Week 1 success criteria:

- Git repo exists.
- Python environment exists.
- PyTorch imports correctly.
- CUDA availability has been checked.
- GPU name and VRAM are recorded if available.
- One CPU PyTorch script runs.
- One GPU smoke test runs on the HPC, preferably through SLURM.
- One tiny autograd or linear regression script runs.
- A Week 1 lab note is written.
- All setup problems and fixes are recorded.

---

## Week 1 - Tuesday, June 16, 2026 - Repo and local environment

Time budget: 90 minutes.

### Goal

Create the repo, environment files, folder structure, and first smoke-test script.

### Tasks

1. Create the repo folder:

```bash
mkdir ai-research-roadmap
cd ai-research-roadmap
```

2. Initialize Git:

```bash
git init
```

3. Create folders:

```bash
mkdir -p scripts/smoke_tests scripts/slurm notes/weekly notes/papers results
mkdir -p phase01_research_engineering/week01_environment_hpc/results/logs
mkdir -p phase01_research_engineering/week01_environment_hpc/results/plots
mkdir -p phase02_llms phase03_efficiency_interpretability
mkdir -p phase04_vision_multimodal_generation phase05_embodied_rl phase06_capstone
```

4. Create `.gitignore`:

```text
__pycache__/
*.pyc
.ipynb_checkpoints/
.env
.venv/
env/
venv/
*.log
*.out
*.err
results/checkpoints/
*.pt
*.pth
*.ckpt
wandb/
.DS_Store
```

5. Create `requirements.txt`:

```text
torch
torchvision
torchaudio
transformers
datasets
accelerate
peft
trl
evaluate
scikit-learn
numpy
pandas
matplotlib
jupyter
pyyaml
tqdm
```

Note: remove leading spaces before package names if they appear after copying.

6. Create `environment.yml`:

```yaml
name: ai-research-roadmap
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pip
  - numpy
  - pandas
  - matplotlib
  - scikit-learn
  - pyyaml
  - tqdm
  - jupyter
  - pip:
      - torch
      - torchvision
      - torchaudio
      - transformers
      - datasets
      - accelerate
      - peft
      - trl
      - evaluate
```

7. Create `scripts/smoke_tests/gpu_smoke_test.py`:

```python
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU name:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("GPU memory GB:", round(props.total_memory / 1024**3, 2))
else:
    device = torch.device("cpu")
    print("Using CPU")

x = torch.randn(2048, 2048, device=device)
y = x @ x
print("Result mean:", y.mean().item())
print("Smoke test passed")
```

8. Run locally or on login node without GPU if allowed:

```bash
python scripts/smoke_tests/gpu_smoke_test.py
```

9. Create `README.md` with:

```text
# AI Research Roadmap

This repo tracks my continual AI research preparation roadmap.

Starting background:
- Completed Machine Learning Specialization
- Completed Deep Learning Specialization

Time commitment:
- 1.5 hours per weekday
- No planned weekend work

Hardware:
- One school HPC GPU
```

10. Commit:

```bash
git add .
git commit -m "Initialize AI research roadmap repo"
```

### Tuesday deliverable

A repo skeleton with environment files and a smoke-test script.

---

## Week 1 - Wednesday, June 17, 2026 - HPC and SLURM GPU test

Time budget: 90 minutes.

### Goal

Verify that a Python job can run on the school HPC GPU.

### Tasks

1. Log into the HPC.

2. Find or confirm the following:

```text
HPC login command:
Project/account name:
GPU partition name:
Max job time:
Available GPU type:
Python/conda/module setup:
Storage location for projects:
```

3. Create `scripts/slurm/gpu_smoke_test.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=gpu_smoke_test
#SBATCH --output=results/gpu_smoke_test_%j.out
#SBATCH --error=results/gpu_smoke_test_%j.err
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Replace these lines with your HPC-specific setup.
# Example:
# module load cuda
# module load anaconda
# source activate ai-research-roadmap

python scripts/smoke_tests/gpu_smoke_test.py
```

4. Submit the job:

```bash
sbatch scripts/slurm/gpu_smoke_test.slurm
```

5. Check job status:

```bash
squeue -u $USER
```

6. Inspect logs:

```bash
cat results/gpu_smoke_test_*.out
cat results/gpu_smoke_test_*.err
```

7. Record the GPU name and memory in `notes/weekly/week01_environment_hpc.md`.

8. If SLURM fails, record:

```text
Exact command:
Exact error:
Likely cause:
Fix attempted:
Final status:
```

### Wednesday deliverable

A successful GPU smoke-test job, or a detailed failure note that can be debugged later.

---

## Week 1 - Thursday, June 18, 2026 - PyTorch autograd and tiny training script

Time budget: 90 minutes.

### Goal

Run a tiny training loop to verify PyTorch basics before real model work begins.

### Tasks

1. Create `scripts/smoke_tests/torch_autograd_test.py`:

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
w = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
y = (x * w).sum()
y.backward()

print("y:", y.item())
print("grad:", w.grad)
print("Autograd test passed")
```

2. Run:

```bash
python scripts/smoke_tests/torch_autograd_test.py
```

3. Create `phase01_research_engineering/week01_environment_hpc/linear_regression_torch.py`:

```python
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def make_data(n: int = 512, noise_std: float = 0.2, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, generator=generator)
    y = 3.0 * x + 2.0 + noise_std * torch.randn(n, 1, generator=generator)
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="phase01_research_engineering/week01_environment_hpc/results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    x, y = make_data()
    x = x.to(device)
    y = y.to(device)

    model = torch.nn.Linear(1, 1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    losses = []
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    with torch.no_grad():
        weight = model.weight.item()
        bias = model.bias.item()

    log_text = f"device={device}\nfinal_loss={losses[-1]:.6f}\nweight={weight:.4f}\nbias={bias:.4f}\n"
    print(log_text)
    (out_dir / "logs" / "linear_regression_log.txt").write_text(log_text)

    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Linear regression training loss")
    plt.tight_layout()
    plt.savefig(out_dir / "plots" / "linear_regression_loss.png")


if __name__ == "__main__":
    main()
```

4. Run:

```bash
python phase01_research_engineering/week01_environment_hpc/linear_regression_torch.py
```

5. Verify that these files are created:

```text
phase01_research_engineering/week01_environment_hpc/results/logs/linear_regression_log.txt
phase01_research_engineering/week01_environment_hpc/results/plots/linear_regression_loss.png
```

### Thursday deliverable

A tiny PyTorch training script with a saved log and loss plot.

---

## Week 1 - Friday, June 19, 2026 - Clean up and write lab note

Time budget: 90 minutes.

### Goal

Finish the setup week with a clean record of what works and what still needs fixing.

### Tasks

1. Create `notes/weekly/week01_environment_hpc.md`:

```text
# Week 1 - Environment and HPC setup

## Dates
Tuesday, June 16, 2026 to Friday, June 19, 2026

## Goal
Set up a reproducible AI research workspace and verify PyTorch + HPC GPU access.

## Environment
Python version:
PyTorch version:
CUDA available:
GPU name:
GPU memory:
HPC scheduler:
GPU partition:
Project/account:

## What works
- [ ] Repo created
- [ ] Python environment created
- [ ] PyTorch imports
- [ ] CUDA checked
- [ ] GPU smoke test ran
- [ ] SLURM script created
- [ ] Tiny autograd test ran
- [ ] Tiny linear regression script ran
- [ ] Loss plot saved

## Problems encountered

### Problem 1
Error:
Cause:
Fix:
Status:

## Results

Linear regression final loss:
Learned weight:
Learned bias:

## What I understand better now

## Next week
Train an MLP on Fashion-MNIST or MNIST with a proper train/eval loop, checkpointing, and plots.
```

2. Update `TODO.md`:

```text
# TODO

## Week 1
- [ ] Create repo skeleton
- [ ] Set up Python environment
- [ ] Run PyTorch smoke test
- [ ] Submit GPU smoke test to HPC
- [ ] Run autograd test
- [ ] Run tiny linear regression training script
- [ ] Write Week 1 lab note

## Week 2 preview
- [ ] MLP classifier on Fashion-MNIST or MNIST
- [ ] Dataloader
- [ ] Train/eval split
- [ ] Accuracy and loss curves
- [ ] Checkpointing
```

3. Update `phase01_research_engineering/week01_environment_hpc/README.md`:

```text
# Week 1 - Environment and HPC

This folder contains the setup and smoke tests for the AI research roadmap.

Files:
- linear_regression_torch.py: tiny PyTorch training loop
- results/logs/: text logs
- results/plots/: saved plots
```

4. Run a final check:

```bash
python scripts/smoke_tests/gpu_smoke_test.py
python scripts/smoke_tests/torch_autograd_test.py
python phase01_research_engineering/week01_environment_hpc/linear_regression_torch.py
```

5. Commit:

```bash
git add .
git commit -m "Complete week 1 environment and HPC setup"
```

### Friday deliverable

A clean setup repo with smoke tests, logs, plots, and a Week 1 lab note.

---

# Year 1 core roadmap

The Year 1 core is a breadth-building foundation. It should create enough practical skill to read papers, implement small versions of major ideas, run controlled experiments, and choose deeper follow-up tracks. It is intentionally not the whole learning plan.

Phase-local roadmaps are also split into the corresponding phase folders:

- `phase01_research_engineering/phase01_research_engineering_roadmap.md`
- `phase02_llms/phase02_llms_roadmap.md`
- `phase03_efficiency_interpretability/phase03_efficiency_interpretability_roadmap.md`
- `phase04_vision_multimodal_generation/phase04_vision_multimodal_generation_roadmap.md`
- `phase05_embodied_rl/phase05_embodied_rl_roadmap.md`
- `phase06_capstone/phase06_capstone_roadmap.md`

## Phase 1 - Research engineering and traditional neural networks

### Week 1 - Environment, Git, PyTorch, HPC bootstrap

Deliverable: working repo, Python environment, GPU smoke test, SLURM template, tiny PyTorch training script, Week 1 lab note.

Daily plan: see the exact Week 1 section above.

### Week 2 - MLPs, datasets, training loops

Goal: build the first proper supervised-learning experiment.

- Monday: Review PyTorch datasets, dataloaders, losses, optimizers, train/eval mode.
- Tuesday: Implement an MLP classifier on MNIST or Fashion-MNIST.
- Wednesday: Add validation split, accuracy, loss curves, checkpointing, seed control.
- Thursday: Run three experiments: baseline, different learning rate, different hidden size.
- Friday: Plot results and write a lab note comparing the runs.

Deliverable: MLP experiment with reproducible config, logs, checkpoint, and plot.

### After Week 2 MLP - Basic ML mini-project bridge

Goal: revisit the core non-neural ML toolkit with small, fast projects before moving into CNNs and deeper neural-network work.

Keep these projects intentionally light. Use tiny generated toy datasets and implement model logic with PyTorch tensors and standard Python only; do not use scikit-learn or other ML libraries for algorithm implementations. Write one short comparison note instead of turning each model into a large standalone project.

Suggested mini-projects:

- Linear regression: fit a small regression dataset, inspect coefficients, residuals, MSE/MAE, and one regularized variant.
- Logistic regression: train a binary or multiclass classifier, compare accuracy, log loss, calibration, and the confusion matrix.
- k-nearest neighbors: compare a few `k` values and show how feature scaling changes results.
- k-means: cluster a toy or small real dataset, plot clusters, and compare inertia/silhouette across `k`.
- DBSCAN: show density-based clustering on a non-spherical toy dataset and record where parameter sensitivity appears.
- EM / Gaussian mixtures: fit a Gaussian mixture model, compare with k-means, and explain soft assignments.
- SVM: implement a linear SVM with hinge loss and SGD on a small separable classification dataset.
- Trees: compare decision tree, random forest, and gradient-boosted trees on one tabular dataset, including overfitting and feature-importance notes.
- Probabilistic models: run at least two of Naive Bayes, Gaussian discriminant analysis, LDA/QDA, Bayesian ridge regression, or a small hidden Markov model-style toy if time allows.

Deliverable: one `basic_ml_mini_projects` folder with small pure-PyTorch scripts, a metrics table, 2-4 plots, and a short comparison note explaining which model assumptions mattered most.

### Week 3 - CNNs and mini-ResNets

Goal: learn image pipelines and residual connections.

- Monday: Read about CNNs, residual connections, batch norm, augmentation.
- Tuesday: Implement a small CNN on CIFAR-10.
- Wednesday: Add augmentations, weight decay, learning-rate schedule, confusion matrix.
- Thursday: Run a mini-ResNet-style model and compare with CNN.
- Friday: Write why residual connections help optimization.

Deliverable: CNN vs mini-ResNet comparison.

### Week 4 - Optimization and experiment discipline

Goal: learn controlled ablations, experiment tracking, and basic testing.

- Monday: Study Adam, SGD momentum, LR schedules, regularization, batch size.
- Tuesday: Add config-driven training: model size, optimizer, LR, batch size, seed.
- Wednesday: Add pytest smoke tests: dataset shape, model forward pass, config loading, and a run that finishes under 30 seconds.
- Thursday: Run optimizer and learning-rate ablations, saving each run under `results/runs/`.
- Friday: Create an experiment table with metric, runtime, memory, repeated-run variance if available, and a reproducibility note.

Deliverable: reusable experiment template with config, tracking, tests, and ablation table.

### Week 5 - RNNs, LSTMs, and sequence modeling

Goal: understand recurrence before Transformers.

- Monday: Read about RNNs, vanishing gradients, LSTM gates, teacher forcing.
- Tuesday: Implement a character-level RNN or LSTM language model.
- Wednesday: Train on a tiny text corpus and generate samples.
- Thursday: Compare RNN/LSTM with a simple sequence baseline.
- Friday: Write where recurrence helps and where it struggles.

Deliverable: small sequence model and samples.

### Week 6 - Autoencoders and representation learning

Goal: learn latent representations.

- Monday: Read about autoencoders, bottlenecks, reconstruction loss.
- Tuesday: Implement a plain autoencoder on MNIST or Fashion-MNIST.
- Wednesday: Visualize latent space and reconstructions.
- Thursday: Add a denoising autoencoder variant.
- Friday: Write supervised learning vs representation learning notes.

Deliverable: autoencoder reconstructions and latent visualization.

### Week 7 - Variational autoencoders

Goal: learn latent-variable generative modeling.

- Monday: Read VAE concepts: ELBO, encoder, decoder, reparameterization.
- Tuesday: Implement a small VAE.
- Wednesday: Debug KL/reconstruction balance.
- Thursday: Run latent-size ablation.
- Friday: Write plain autoencoder vs VAE comparison.

Deliverable: VAE reconstructions and generated samples.

### Week 8 - GANs

Goal: learn adversarial generation and instability.

- Monday: Read GAN concepts: generator, discriminator, minimax game.
- Tuesday: Implement a tiny GAN on MNIST.
- Wednesday: Debug instability and mode collapse.
- Thursday: Run one DCGAN-style variant if feasible.
- Friday: Write why GANs are powerful but hard to train.

Deliverable: GAN samples and failure analysis.

---

## Phase 2 - LLM foundations, full pipeline

### Week 9 - Tokenization and text data

Goal: prepare language-modeling data correctly.

- Monday: Learn BPE, WordPiece, unigram tokenization, special tokens, padding, truncation, packing.
- Tuesday: Train a tiny tokenizer on a small corpus.
- Wednesday: Compare the custom tokenizer with a pretrained tokenizer.
- Thursday: Build tokenized chunks, attention masks, and train/val split.
- Friday: Write how tokenization affects context length and model behavior.

Deliverable: tokenizer and prepared LM dataset.

### Week 10 - Transformer from scratch I: attention block

Goal: implement the core Transformer block.

- Monday: Read self-attention, multi-head attention, positional encodings, residuals, FFN.
- Tuesday: Implement scaled dot-product attention and causal masking.
- Wednesday: Implement multi-head attention and shape tests.
- Thursday: Add residual connections, layer norm, and MLP block.
- Friday: Write a line-by-line explanation of the block.

Deliverable: tested Transformer block.

### Week 11 - Transformer from scratch II: mini-GPT

Goal: train a tiny decoder-only language model.

- Monday: Study embeddings, positional embeddings, decoder blocks, LM head, next-token loss.
- Tuesday: Implement mini-GPT using the Transformer block.
- Wednesday: Overfit on a tiny batch to verify correctness.
- Thursday: Train on a small corpus on HPC.
- Friday: Plot loss and generate samples.

Deliverable: mini-GPT trained from scratch.

### Week 12 - Sampling, decoding, and perplexity

Goal: understand generation behavior.

- Monday: Learn greedy, temperature, top-k, top-p, repetition penalty, beam search, perplexity.
- Tuesday: Implement greedy, temperature, top-k, and top-p sampling manually.
- Wednesday: Evaluate validation loss and perplexity.
- Thursday: Compare generations across decoding methods.
- Friday: Write why low loss does not always mean good generation.

Deliverable: decoding playground and comparison note.

### Week 13 - BERT-style masked language modeling

Goal: understand encoder-only Transformers.

- Monday: Read masked LM, bidirectionality, classification heads.
- Tuesday: Create a masked-token dataset.
- Wednesday: Implement a tiny encoder-only Transformer.
- Thursday: Train and inspect masked predictions.
- Friday: Write causal LM vs masked LM comparison.

Deliverable: tiny BERT-style model.

### Week 14 - T5 and encoder-decoder models

Goal: understand text-to-text modeling.

- Monday: Read encoder-decoder Transformers and task prefixes.
- Tuesday: Use a small pretrained T5 model for summarization or QA.
- Wednesday: Fine-tune on a tiny dataset.
- Thursday: Compare pretrained vs fine-tuned results.
- Friday: Write decoder-only vs encoder-only vs encoder-decoder notes.

Deliverable: small T5 fine-tuning experiment.

### Week 15 - Toy-scale LLM pretraining

Goal: understand what pretraining teaches and what it cannot teach at small scale.

- Monday: Study data scale, compute scale, optimizer stability, and evaluation.
- Tuesday: Clean a larger small corpus.
- Wednesday: Train mini-GPT longer.
- Thursday: Run context-length and model-width ablations.
- Friday: Write toy pretraining limitations.

Deliverable: toy pretraining report.

### Week 16 - Mid-training / continued pretraining

Goal: adapt a base LM to a domain corpus.

- Monday: Learn domain-adaptive and task-adaptive pretraining.
- Tuesday: Choose domain corpus: AI abstracts, class notes, lab text, code docs.
- Wednesday: Continue pretraining a small pretrained model.
- Thursday: Compare base vs continued-pretrained model on domain prompts/perplexity.
- Friday: Write when continued pretraining helps or overfits.

Deliverable: domain-adapted small LM.

### Week 17 - Supervised fine-tuning with LoRA

Goal: learn parameter-efficient instruction tuning.

- Monday: Learn LoRA and PEFT.
- Tuesday: Prepare a small instruction dataset.
- Wednesday: Run LoRA SFT on a small model.
- Thursday: Compare base vs SFT outputs on fixed prompts.
- Friday: Write pretraining vs SFT vs LoRA notes.

Deliverable: LoRA adapter and comparison table.

### Week 18 - QLoRA and memory-efficient fine-tuning

Goal: adapt models under limited GPU memory.

- Monday: Learn 4-bit quantization plus LoRA adapters.
- Tuesday: Check exact GPU VRAM and choose model size.
- Wednesday: Run QLoRA smoke test.
- Thursday: Run one real small QLoRA job.
- Friday: Record GPU memory, batch size, sequence length, training speed.

Deliverable: QLoRA run log.

### Week 19 - Post-training and preference tuning with DPO

Goal: understand post-training as the stage after pretraining and mid-training, then run a small preference-tuning experiment.

- Monday: Learn the post-training stack: instruction SFT, reward modeling conceptually, RLHF/PPO conceptually, and DPO practically.
- Tuesday: Prepare prompt/chosen/rejected preference dataset.
- Wednesday: Run DPO on a small model or adapter.
- Thursday: Compare SFT-only vs SFT+DPO outputs.
- Friday: Write pretraining vs mid-training vs post-training notes, including what preference tuning changes and what it cannot guarantee.

Deliverable: post-training mini-report plus small DPO experiment.

### Week 20 - LLM evaluation

Goal: evaluate models repeatably.

- Monday: Learn perplexity, multiple-choice eval, exact match, GSM-style math, win rate, human eval.
- Tuesday: Install and run lm-evaluation-harness on a tiny model.
- Wednesday: Create a 20-50 prompt personal evaluation set.
- Thursday: Evaluate base, SFT, and DPO/QLoRA variants, including failure categories and contamination concerns.
- Friday: Write model/method/score/failure table and start `capstone_candidates.md` with at least five possible long-term project ideas.

Deliverable: LLM evaluation report plus first capstone candidate list.

### Week 21 - Inference, serving, and KV cache

Goal: understand LLM serving and performance.

- Monday: Learn prefill, decode, KV cache, batching, latency, throughput, memory.
- Tuesday: Serve a small model with vLLM or equivalent.
- Wednesday: Benchmark tokens/sec, latency, batch size, sequence length.
- Thursday: Compare Hugging Face generation with vLLM serving.
- Friday: Write why inference engineering differs from training.

Deliverable: inference benchmark.

### Week 22 - Retrieval-augmented generation

Goal: build a RAG system over personal notes or papers.

- Monday: Learn chunking, embeddings, vector search, reranking, context injection.
- Tuesday: Build a small document corpus from notes or papers.
- Wednesday: Implement retrieval and prompt construction.
- Thursday: Evaluate with and without retrieval.
- Friday: Write when RAG helps and fails.

Deliverable: RAG assistant over notes.

### Week 23 - Test-time compute: CoT, self-consistency, Tree of Thoughts

Goal: compare reasoning methods.

- Monday: Read chain-of-thought, self-consistency, and Tree of Thoughts conceptually.
- Tuesday: Build a small reasoning test set.
- Wednesday: Compare direct answer, CoT-style prompting, and self-consistency.
- Thursday: Implement a simple Tree-of-Thoughts-style search on one toy task.
- Friday: Write when extra test-time compute helps.

Deliverable: reasoning-method comparison.

### Week 24 - Agents and tool use

Goal: build a small tool-using loop.

- Monday: Read ReAct, tool-use concepts, and the idea of an agent as a system around model inference rather than a model itself.
- Tuesday: Build a tool-calling loop with calculator, local search, or Python execution.
- Wednesday: Add action/observation logging and a minimal short-term memory state.
- Thursday: Evaluate on 20 tasks requiring tools.
- Friday: Write agent failure modes and explain why agents sit above inference, tool use, test-time compute, memory, evaluation, and system design.

Deliverable: small tool-using agent.

Agent concept note:

An agent is not usually a new model class. In this roadmap, treat an agent as a system layer built on top of one or more model inference calls. The model predicts tokens or actions; the agent system decides how to wrap those calls with prompts, tools, retrieval, memory, test-time search, action/observation loops, guardrails, logging, and evaluation. A stronger model can make an agent better, but agent quality also depends on tool design, memory design, task decomposition, control flow, latency, reliability, and failure handling.

Agent stack to learn:

```text
model weights
  -> inference engine and decoding
  -> prompting and tool schemas
  -> test-time compute / search / reflection
  -> memory and retrieval
  -> planner-controller loop
  -> tools and environment actions
  -> logging, evals, safety checks, and system design
```

---

## Phase 3 - Efficiency, architecture variants, and interpretability

### Week 25 - FlashAttention and attention efficiency

Goal: understand why attention becomes expensive.

- Monday: Read FlashAttention conceptually.
- Tuesday: Profile mini-GPT attention memory/time as context length increases.
- Wednesday: Compare regular attention with optimized attention if available.
- Thursday: Run context-length benchmark.
- Friday: Write why attention is memory-bound.

Deliverable: attention efficiency benchmark.

### Week 26 - Quantization and memory

Goal: understand precision and memory.

- Monday: Learn fp32, fp16, bf16, int8, int4, activation memory, optimizer state.
- Tuesday: Load a small model in full precision and quantized form.
- Wednesday: Compare memory usage and output quality.
- Thursday: Run inference benchmarks by precision setting.
- Friday: Write training memory vs inference memory notes.

Deliverable: quantization comparison.

### Week 27 - Mixture of Experts

Goal: understand sparse expert routing.

- Monday: Read MoE and Mixtral concepts.
- Tuesday: Implement a toy MoE feed-forward layer.
- Wednesday: Add top-1 or top-2 routing and load-balancing diagnostics.
- Thursday: Compare dense FFN vs toy MoE on a small task.
- Friday: Write capacity vs compute notes.

Deliverable: toy MoE layer and comparison.

### Week 28 - Mamba and state-space alternatives

Goal: understand alternatives to dense attention.

- Monday: Read selective state-space model concepts.
- Tuesday: Use an existing small Mamba-style implementation or model.
- Wednesday: Compare with a tiny Transformer on a sequence task.
- Thursday: Benchmark longer sequence lengths.
- Friday: Write why alternatives to attention exist.

Deliverable: Transformer vs Mamba-style comparison.

### Week 29 - Hybrid architectures

Goal: understand why models combine Transformer, SSM, and MoE ideas.

- Monday: Read hybrid architecture concepts.
- Tuesday: Sketch possible hybrid block designs.
- Wednesday: Build a tiny hybrid toy model or block simulation.
- Thursday: Run one small synthetic sequence experiment.
- Friday: Write why hybrid architectures are attractive.

Deliverable: hybrid architecture note.

### Week 30 - Interpretability I: attention and activations

Goal: inspect model internals.

- Monday: Read Transformer Circuits-style interpretability introduction.
- Tuesday: Add hooks to mini-GPT to save activations and attention maps.
- Wednesday: Visualize attention heads on short sequences.
- Thursday: Try activation patching on a toy behavior.
- Friday: Write what attention maps can and cannot show.

Deliverable: activation-inspection notebook.

### Week 31 - Interpretability II: sparse autoencoders

Goal: learn feature discovery from activations.

- Monday: Read sparse autoencoder interpretability concepts.
- Tuesday: Collect hidden activations from a small model.
- Wednesday: Train a small sparse autoencoder on those activations.
- Thursday: Inspect top-activating tokens/examples for a few features.
- Friday: Write what makes a feature interpretable or not.

Deliverable: tiny SAE experiment.

### Week 32 - Infra consolidation

Goal: make the LLM pipeline reproducible.

- Monday: Review training, inference, evaluation, memory, HPC notes.
- Tuesday: Clean the LLM pipeline repo.
- Wednesday: Add README instructions to reproduce one experiment.
- Thursday: Rerun one experiment from scratch using only the README.
- Friday: Write phase retrospective.

Deliverable: reproducible LLM mini-pipeline.

---

## Phase 4 - Vision, multimodal models, and generation

### Week 33 - CNN to Vision Transformer

Goal: compare image inductive biases.

- Monday: Read ViT concepts: image patches as tokens, Transformer encoder.
- Tuesday: Implement image patching and patch embeddings.
- Wednesday: Build a tiny ViT classifier.
- Thursday: Compare CNN, ResNet, and ViT on a small dataset.
- Friday: Write inductive bias notes.

Deliverable: CNN/ResNet/ViT comparison.

### Week 34 - Self-supervised vision

Goal: learn contrastive and self-supervised features.

- Monday: Learn SimCLR/DINO-style intuition.
- Tuesday: Implement a small contrastive image representation experiment.
- Wednesday: Train on a small unlabeled image subset.
- Thursday: Freeze features and train a linear classifier.
- Friday: Write supervised vs self-supervised feature notes.

Deliverable: contrastive representation experiment.

### Week 35 - CLIP-style image-text learning

Goal: connect image and text representations.

- Monday: Read CLIP concepts.
- Tuesday: Use pretrained CLIP for zero-shot image classification.
- Wednesday: Build a small image-text retrieval demo.
- Thursday: Evaluate retrieval on a small custom dataset.
- Friday: Write why language supervision improves flexibility.

Deliverable: CLIP retrieval demo.

### Week 36 - Segmentation and SAM-style models

Goal: understand segmentation foundation models.

- Monday: Read SAM concepts.
- Tuesday: Run a pretrained segmentation model on sample images.
- Wednesday: Compare point, box, and mask prompts if supported.
- Thursday: Build a small visual demo.
- Friday: Write classification vs detection vs segmentation notes.

Deliverable: segmentation demo.

### Week 37 - Multimodal LLMs and VLM architecture

Goal: understand visual features connected to LLMs.

- Monday: Read Flamingo/LLaVA-style concepts: vision encoder, connector, LLM, instruction tuning.
- Tuesday: Run a compact VLM for image captioning or VQA.
- Wednesday: Build a 20-image VQA evaluation set.
- Thursday: Compare VLM outputs with CLIP retrieval or image classifier outputs.
- Friday: Write VLMs vs pure vision models notes.

Deliverable: compact VLM evaluation.

### Week 38 - Small VLM adaptation

Goal: adapt a VLM or simulate multimodal adaptation.

- Monday: Learn visual instruction tuning and multimodal data format.
- Tuesday: Prepare a tiny image-question-answer dataset.
- Wednesday: Run LoRA-style VLM fine-tuning if possible; otherwise use frozen vision features plus a small head.
- Thursday: Compare before/after behavior.
- Friday: Write why multimodal fine-tuning is harder than text-only fine-tuning.

Deliverable: small VLM adaptation experiment.

### Week 39 - Diffusion models from scratch

Goal: implement a tiny DDPM.

- Monday: Read forward noising, reverse denoising, noise prediction, sampling loop.
- Tuesday: Implement forward noising.
- Wednesday: Train a tiny denoising network on small images.
- Thursday: Generate samples and run one ablation.
- Friday: Write diffusion vs VAE vs GAN comparison.

Deliverable: tiny DDPM.

### Week 40 - Diffusers and conditional generation

Goal: learn library-side diffusion workflows.

- Monday: Study Hugging Face Diffusers course/tutorials.
- Tuesday: Run a pretrained diffusion pipeline.
- Wednesday: Try conditioning: class labels, text prompt, or image-to-image.
- Thursday: Build or modify a small custom pipeline.
- Friday: Write what from-scratch DDPM taught that the library hides.

Deliverable: diffusion library demo.

---

## Phase 5 - Graphs, audio, RL, world models, and embodied AI

### Week 41 - Graph neural networks

Goal: learn message passing and graph attention.

- Monday: Read GCN/GAT concepts.
- Tuesday: Use PyTorch Geometric to load Cora/Citeseer or another node-classification dataset.
- Wednesday: Implement GCN or GAT.
- Thursday: Run hidden size, heads, dropout ablations.
- Friday: Write why graphs need different assumptions.

Deliverable: GAT node-classification project.

### Week 42 - Audio and speech sampler

Goal: experience audio modeling.

- Monday: Learn wav2vec 2.0 and Hugging Face audio tooling.
- Tuesday: Load a small speech dataset.
- Wednesday: Run inference with a pretrained ASR model.
- Thursday: Fine-tune or evaluate on a tiny subset if feasible.
- Friday: Write why audio differs from text/images.

Deliverable: small audio/speech experiment.

### Week 43 - RL basics

Goal: learn model-free RL basics.

- Monday: Review MDPs, policy, value function, Q-learning, policy gradients, actor-critic.
- Tuesday: Train a simple RL agent on CartPole or equivalent.
- Wednesday: Plot reward curves and debug instability.
- Thursday: Compare random, heuristic, and learned policies.
- Friday: Write why RL is harder to evaluate than supervised learning.

Deliverable: basic RL agent.

### Week 44 - World models

Goal: understand learned environment models.

- Monday: Read Dreamer-style world model concepts.
- Tuesday: Implement a tiny learned dynamics model or run a small existing example.
- Wednesday: Compare model-free vs model-based intuition.
- Thursday: Visualize predicted next states or latent transitions.
- Friday: Write what world model means in practice.

Deliverable: toy world-model experiment.

### Week 45 - Imitation learning and behavior cloning

Goal: learn offline policy learning from demonstrations.

- Monday: Learn behavior cloning, offline data, distribution shift, compounding errors.
- Tuesday: Create or load expert trajectories in a simple environment.
- Wednesday: Train a behavior-cloning policy.
- Thursday: Evaluate rollout success and failure cases.
- Friday: Write imitation learning vs RL notes.

Deliverable: behavior-cloning policy.

### Week 46 - VLA and robot foundation models

Goal: distinguish VLMs, VLAs, and robot policies.

- Monday: Read RT-2 and OpenVLA concepts.
- Tuesday: Study LeRobot data format or another robot dataset format.
- Wednesday: Load or inspect a robot dataset.
- Thursday: Run a small policy inference or dataset visualization notebook.
- Friday: Write VLM vs VLA vs policy-model notes.

Deliverable: embodied-AI survey note plus small data demo.

Agentic RL bridge:

The roadmap separates ordinary RL, LLM agents, and embodied policies for clarity, but the modern research frontier increasingly connects them. Treat agentic RL as the study of optimizing interactive behavior over multi-step environments, tools, or tasks. This can include classical RL policies, RLHF-style language-model post-training, RL with verifiable rewards, tool-use rewards, self-play or synthetic task environments, and robot/VLA policies.

Small add-on project if time permits:

- Define a toy agent environment with clear rewards, such as calculator use, web-free document search, code execution, or a gridworld.
- Compare a prompt-only tool-using agent against a simple reward-shaped or rule-improved agent loop.
- Track success rate, tool calls, failure modes, and reward hacking.
- Write what is being optimized: model weights, prompts, policies, tool-selection rules, search budget, memory, or system control flow.

---

## Phase 6 - Capstone research project

### Week 47 - Choose a capstone area

Goal: choose a feasible small research reproduction.

- Monday: Review all Friday notes from the year.
- Tuesday: Choose three candidate papers/projects.
- Wednesday: Evaluate feasibility with GPU/time.
- Thursday: Choose one paper and define a small reproduction target.
- Friday: Write a capstone proposal.

Deliverable: capstone proposal.

### Week 48 - Reproduce baseline

Goal: get the baseline working.

- Monday: Read the chosen paper carefully.
- Tuesday: Set up baseline code and dataset.
- Wednesday: Run a tiny smoke test.
- Thursday: Run baseline experiment on HPC.
- Friday: Write baseline results and compare with paper numbers if available.

Deliverable: baseline reproduction.

### Week 49 - Reproduce main method

Goal: implement or adapt the core method.

- Monday: Identify the smallest version of the paper's main method.
- Tuesday: Implement or adapt the method.
- Wednesday: Debug on a tiny subset.
- Thursday: Run main experiment.
- Friday: Compare baseline vs method.

Deliverable: first reproduction attempt.

### Week 50 - Ablations

Goal: test which factors matter.

- Monday: Choose three ablations: model size, data size, objective, prompt, retrieval, routing, context length, or loss term.
- Tuesday: Run ablation 1.
- Wednesday: Run ablation 2.
- Thursday: Run ablation 3.
- Friday: Create results table and plot.

Deliverable: ablation table and plot.

### Week 51 - Failure analysis and diagnostics

Goal: understand why the model fails.

- Monday: Collect failure cases.
- Tuesday: Categorize errors manually.
- Wednesday: Add one diagnostic: confusion matrix, attention visualization, retrieval inspection, activation inspection, or prompt analysis.
- Thursday: Rerun one corrected experiment.
- Friday: Write failure-analysis section.

Deliverable: failure-analysis report.

### Week 52 - Final report and portfolio packaging

Goal: publish a clean capstone artifact.

- Monday: Draft abstract, intro, related work, method, experiments, results.
- Tuesday: Finish results, plots, and tables.
- Wednesday: Write limitations and future work.
- Thursday: Clean code, README, requirements, reproduction instructions.
- Friday: Publish/archive the project and write final reflection.

Deliverable: final capstone report and polished repo.

---

# Continual learning extensions

The roadmap does not stop after Week 52. Treat Weeks 1-52 as Year 1 core. After that, continue with extension seasons. Each extension season should give a serious taste of a research area, not force a full specialization before interests are clear. Default to 4 to 8 weeks per track, then revisit which area deserves deeper work.

The order below is intentionally front-loaded toward the most current and research-heavy frontier areas: post-training, agents, long-context/memory, safety/evals, data/evaluation, and systems. Broader or more domain-specific areas come later as samplers.

Suggested Year 2 timeline:

```text
Weeks 53-60: Modern post-training, RLHF/RLAIF/RLVR, and agentic RL
Weeks 61-68: Coding agents, computer-use agents, and AI for software engineering
Weeks 69-74: Long-context models and memory systems
Weeks 75-80: AI safety, security, and robust evals
Weeks 81-86: Data-centric AI and evaluation design
Weeks 87-92: AI systems, scaling, and ML infrastructure
Weeks 93-96: Research methods, writing, and reproducibility upgrade
Weeks 97-100: AI for science sampler
Weeks 101-104: Causal ML, uncertainty, ranking systems, and extension synthesis
```

This Year 2 timeline is a default path, not a hard endpoint. It is a tasting menu for important research areas. After it, choose one or two areas for a longer specialization only if the projects made the interest real.

## Extension Track A - Modern Post-Training and Agentic RL

Goal: sample the most important current post-training and interactive-optimization ideas beyond basic SFT and DPO.

Topics:

- RLHF.
- RLAIF.
- RL with verifiable rewards / RLVR.
- Process reward models.
- Verifier models.
- Rejection sampling.
- Preference data quality.
- PPO conceptually.
- GRPO-style reasoning optimization conceptually.
- Tool-use rewards.
- Agent environments.
- Reward hacking.

Projects:

- Compare SFT, DPO, and rejection sampling on a small instruction task.
- Build a tiny verifier for math, code, or structured outputs.
- Create a toy tool-use environment with explicit rewards.
- Compare prompt-only agents vs reward-shaped agent loops.
- Write what is being optimized: model weights, prompts, policies, tool rules, search budget, memory, or control flow.

Deliverables:

- Post-training comparison report.
- One runnable verifier or toy agentic-RL environment.
- Clear notes on which parts are conceptual only versus actually implemented.

## Extension Track B - Coding Agents and Computer-Use Agents

Goal: sample AI systems that operate over repositories, tests, tools, browsers, and GUIs.

Topics:

- Code generation.
- Repository-level agents.
- Program repair.
- Test generation.
- SWE-bench-style evaluation.
- Tool-using coding loops.
- Browser agents.
- Desktop/UI agents.
- Observation/action loops.
- Action grounding.
- Tool failure recovery.
- Sandboxing and permission boundaries.

Projects:

- Build a tiny coding-agent loop over a small local repo with tests.
- Evaluate program repair on 10-20 seeded bugs.
- Build a test-generation assistant and measure whether generated tests catch real bugs.
- Create a mini SWE-bench-style benchmark from local toy issues.
- Build a browser-free computer-use simulation with observations, actions, and failure recovery.

Deliverable: coding-agent or computer-use-agent benchmark with success rate, failure taxonomy, and safety constraints.

## Extension Track C - Long-Context Models and Memory Systems

Goal: sample model and system approaches for using information beyond a short prompt. This is a high-priority frontier area because long-context behavior, retrieval, memory, and agent state increasingly determine whether LLM systems work in real research and software tasks.

Topics:

- Long-context evaluation.
- Attention vs retrieval.
- Context compression.
- Memory stores.
- Lost-in-the-middle behavior.
- Summarization memory.
- Episodic vs semantic memory.
- Retrieval freshness and forgetting.
- Long-context cost and latency.
- When to use context, RAG, cache, summary, or external state.

Projects:

- Build a long-context evaluation set over notes, papers, or code.
- Compare long prompt, RAG, and summary-memory variants.
- Measure lost-in-the-middle failures.
- Benchmark cost/latency as context length grows.
- Write when memory should be context, retrieval, summary, cache, or external state.

Deliverable: long-context and memory comparison report with accuracy, failure cases, cost/latency notes, and design recommendations.

## Extension Track D - AI Safety, Security, and Robustness

Goal: sample frontier model behavior, agent safety, and deployment risk as first-class research topics.

Topics:

- Hallucination evaluation.
- Prompt injection.
- Jailbreaks.
- Data leakage and data exfiltration.
- Eval gaming.
- Distribution shift.
- Misuse risk.
- Tool and agent sandboxing.
- RAG security.
- Model behavior under adversarial prompts.
- Safety tradeoffs between helpfulness, refusal, and task completion.

Projects:

- Build a small prompt-injection benchmark for a RAG assistant.
- Compare direct prompting vs tool-using agent behavior under adversarial instructions.
- Create a jailbreak/refusal/factuality eval table for one small model.
- Write a sandboxing and threat-model note for agents.

Deliverable: safety/security eval report with attack examples, mitigations, and remaining risks.

## Extension Track E - Data-Centric AI and Evaluation Design

Goal: learn that data and evals often matter more than model novelty.

Topics:

- Dataset quality.
- Deduplication.
- Train/validation/test leakage.
- Benchmark contamination.
- Labeling noise.
- Synthetic data.
- Data filtering.
- Dataset documentation and licensing.
- Custom eval set design.
- Adversarial evals.
- Confidence intervals and repeated runs.
- Human vs automated evaluation.
- Failure taxonomies.

Projects:

- Build a small contaminated-vs-clean evaluation set and measure score inflation.
- Create a dataset card for one roadmap dataset.
- Compare one model across clean, noisy, adversarial, and shifted test sets.
- Write a failure taxonomy for one LLM, vision, or RL project.

Deliverable: data and evaluation report with dataset card, metrics, failure cases, and reproducibility notes.

## Extension Track F - AI Systems, Scaling, and ML Infrastructure

Goal: sample how real training and inference systems fail, scale, recover, and cost money.

Topics:

- `torch.profiler`.
- GPU memory accounting.
- Mixed precision.
- Gradient accumulation.
- Checkpoint/resume.
- Dataloader bottlenecks.
- Failed-job recovery on SLURM.
- Experiment tracking.
- Exact CUDA/PyTorch install notes.
- Lockfiles or pinned environments.
- Model/data/compute scaling.
- Chinchilla-style tradeoffs.
- Inference latency, throughput, batching, and tokens per dollar.
- Why small experiments may not extrapolate.

Projects:

- Profile one training loop and remove the largest bottleneck.
- Add checkpoint/resume to one project and test an interrupted run.
- Compare mixed precision, batch size, and gradient accumulation.
- Build an HPC failure-recovery checklist.
- Benchmark inference cost across batch size, context length, and quantization.

Deliverable: systems report with profiler traces, memory table, recovery workflow, and inference-cost benchmark.

## Extension Track G - Research Methods and Scientific Writing

Goal: improve research process and writing using projects already completed.

Core skills:

- Hypothesis-driven experiments.
- Baseline choice.
- Ablation design.
- Error analysis.
- Limitation writing.
- Falsification criteria.
- Reproduction instructions.
- Monthly mini-paper writing.
- External critique from a professor, PhD student, classmate, research group, or open-source reviewer.

Deliverables:

- `RESEARCH_PROCESS.md`.
- `notes/papers/PAPER_NOTE_TEMPLATE.md`.
- One mini-paper from an existing project.
- One reviewed project writeup with outside feedback.

## Extension Track H - AI for Science Sampler

Goal: understand how foundation-model techniques transfer into scientific domains.

Topics:

- Protein or genomics models.
- Molecule generation.
- Materials discovery.
- Weather or climate models.
- Theorem proving.
- Algorithm discovery.
- Scientific dataset leakage and evaluation.

Projects:

- Run inference with a small protein, molecule, theorem, or climate model.
- Reproduce one small scientific ML baseline.
- Write what changes when the data is scientific rather than internet text/images.

Deliverable: AI-for-science survey note plus one runnable small demo.

## Extension Track I - Causal ML, Uncertainty, and Ranking Systems

Goal: broaden research maturity beyond generative-model trends.

Topics:

- Causality vs correlation.
- Uncertainty calibration.
- Bayesian intuition.
- Conformal prediction.
- OOD detection.
- Recommender systems.
- Retrieval and ranking.
- Two-tower models.
- Learning-to-rank.
- Feedback loops.
- Bandits.

Projects:

- Add calibration and uncertainty reporting to one classifier.
- Build a small retrieval + ranking system.
- Compare offline ranking metrics with simulated user feedback.
- Write why causality, uncertainty, and feedback loops matter for deployed AI.

Deliverable: uncertainty or ranking project with metrics, calibration/failure plots, and limitations.

## Open Research Cycles

After Year 2, move from syllabus mode into cycles:

1. Choose a paper, benchmark, or open problem.
2. Reproduce the smallest credible baseline.
3. Add one controlled modification.
4. Run ablations and error analysis.
5. Write a mini-paper.
6. Ask for external critique.
7. Decide whether to continue, pivot, or archive.

Keep a living `capstone_candidates.md` starting in Week 20. Each candidate should include:

- Problem.
- Why it matters.
- Required data.
- Required compute.
- Baseline.
- Evaluation metric.
- Main risk.
- Smallest useful reproduction.
- Possible extension.

---

# Model-size guidance for one GPU

Use the exact GPU VRAM from Week 1 to adjust model choices.

```text
8-12 GB VRAM:
- From-scratch mini-GPT
- DistilGPT2 / GPT-2 small
- Small T5
- Tiny LoRA experiments
- Small image and diffusion models

16-24 GB VRAM:
- 0.5B-3B models with QLoRA
- Better batch sizes
- Compact VLM experiments

32-48 GB VRAM:
- 3B-7B QLoRA
- Longer contexts
- Heavier evaluation

80 GB VRAM:
- 7B LoRA more comfortably
- Larger VLM/VLA experiments
- Longer-context experiments
```

Rule:

- Train tiny models from scratch for understanding.
- Fine-tune pretrained models for realistic experiments.
- Do not attempt serious LLM pretraining from scratch on one school GPU.

---

# Reading map by topic

Use this as the core conceptual reading list. Do not read all of these before coding. Read each when its week arrives.

## Traditional neural networks

- CNNs and ResNets
- RNNs, LSTMs, GRUs
- Autoencoders
- Variational autoencoders
- GANs

## LLM foundations

- Attention Is All You Need
- BERT
- T5
- GPT-style decoder-only modeling
- Llama-style modern dense Transformers

## LLM training stages

- Data cleaning and tokenization
- Pretraining
- Continued pretraining / mid-training
- Post-training
- Supervised fine-tuning / instruction tuning
- LoRA and QLoRA
- Reward modeling conceptually
- RLHF and PPO conceptually
- DPO practically
- RL with verifiable rewards conceptually

## LLM inference and evaluation

- Sampling and decoding
- KV cache
- Quantization
- vLLM-style serving
- lm-evaluation-harness or similar benchmark tools
- Custom evaluation sets
- Benchmark contamination
- Confidence intervals and repeated runs
- Failure taxonomies

## Data-centric AI and evaluation design

- Dataset quality
- Deduplication
- Labeling noise
- Train/validation/test leakage
- Synthetic data
- Data filtering
- Dataset cards
- Human vs automated evaluation

## Test-time compute and agents

- Chain-of-thought prompting
- Self-consistency
- Tree of Thoughts
- ReAct
- Tool use
- RAG
- Memory systems
- Agent control loops
- Agent evaluations

Agents are not models by default. They are systems built above model inference, tool use, test-time compute, memory and retrieval, environment interaction, and reliability engineering. Study them as system design problems as much as modeling problems.

## Long-context models and memory systems

- Long-context evaluation
- Attention vs retrieval
- Context compression
- Memory stores
- Lost-in-the-middle behavior
- Summarization memory
- Episodic vs semantic memory
- Retrieval freshness and forgetting

## Coding agents and computer-use agents

- Repository-level agents
- Program repair
- Test generation
- SWE-bench-style evaluation
- Browser and GUI agents
- Observation/action loops
- Action grounding
- Tool failure recovery
- Sandboxing and permissions

## Safety, security, and robustness

- Hallucination evaluation
- Prompt injection
- Jailbreaks
- Data exfiltration
- Eval gaming
- RAG security
- Agent sandboxing
- Distribution shift
- Misuse risk

## Architecture variants

- FlashAttention
- Mixture of Experts
- Mamba / state-space models
- Hybrid Transformer-SSM-MoE designs

## Interpretability

- Activation hooks
- Attention visualization
- Activation patching
- Sparse autoencoders

## Vision and multimodal

- CNNs and ViTs
- SimCLR/DINO-style self-supervised vision
- CLIP
- SAM
- Flamingo/LLaVA-style VLMs
- Small VLM fine-tuning/adaptation

## Generative models

- VAE
- GAN
- DDPM
- Conditional diffusion
- Diffusers-style pipelines

## Embodied AI and RL

- Basic RL
- Behavior cloning
- Offline imitation
- World models / Dreamer-style ideas
- Agentic RL
- RL for tool use and multi-step tasks
- RLHF / RLVR-style optimization conceptually
- VLM vs VLA
- RT-2 / OpenVLA-style systems
- LeRobot-style datasets and tooling

## AI systems, scaling, and infrastructure

- `torch.profiler`
- GPU memory accounting
- Mixed precision
- Gradient accumulation
- Checkpoint/resume
- Dataloader bottlenecks
- SLURM recovery
- Pinned environments
- Tokens per dollar
- Scaling laws and compute economics
- Model/data/compute scaling
- Chinchilla-style tradeoffs

## AI for science

- Protein and genomics models
- Molecule generation
- Materials discovery
- Weather and climate models
- Theorem proving
- Algorithm discovery

## Causal ML, uncertainty, and ranking

- Causality vs correlation
- Uncertainty calibration
- Bayesian intuition
- Conformal prediction
- OOD detection
- Two-tower recommenders
- Learning-to-rank
- Feedback loops and bandits

---

# Year 1 core portfolio target

By the end of the Year 1 core, the student should have:

1. A reproducible PyTorch/HPC experiment template.
2. MLP, CNN, ResNet, RNN/LSTM, autoencoder, VAE, and GAN projects.
3. A mini-GPT trained from scratch.
4. A masked LM or encoder-only Transformer.
5. A T5 or encoder-decoder fine-tuning project.
6. A continued-pretraining experiment.
7. LoRA, QLoRA, SFT, and DPO experiments.
8. A post-training comparison note covering SFT, preference tuning, RLHF/PPO conceptually, and DPO practically.
9. An LLM evaluation report.
10. An inference benchmark.
11. A RAG system over notes.
12. Test-time compute comparisons.
13. A small tool-using agent with memory, tool logging, and failure analysis.
14. FlashAttention/memory benchmark.
15. Toy MoE and Mamba-style comparison.
16. Interpretability notebook.
17. CNN vs ViT comparison.
18. CLIP retrieval demo.
19. SAM/segmentation demo.
20. Compact VLM evaluation.
21. Tiny DDPM and diffusion pipeline demo.
22. GAT graph-learning project.
23. Audio/speech sampler project.
24. RL agent.
25. World-model or behavior-cloning project.
26. Agentic RL bridge note or toy tool-use reward experiment.
27. VLA/robotics survey and data demo.
28. Capstone reproduction report.

# Continual extension portfolio targets

After the Year 1 core, keep adding portfolio artifacts:

1. Modern post-training or RLVR/verifier toy project.
2. Coding-agent or computer-use-agent benchmark.
3. Long-context and memory comparison report.
4. Safety/security eval suite for RAG or agents.
5. Data-centric AI and evaluation-design report.
6. AI systems, scaling, profiling, and recovery report.
7. Research process guide and paper-note template.
8. Monthly mini-paper archive.
9. AI-for-science runnable sampler.
10. Causal uncertainty or ranking-system project.
11. Second capstone or open-ended research cycle.

---

# First Codex task to run

Use this prompt with Codex after creating or opening the repo:

```text
Read ai_research_roadmap_codex.md. Implement Week 1 only.

Create the repo structure, environment files, smoke-test scripts, SLURM template, Week 1 README, and Week 1 lab-note template exactly as specified.

Do not implement Week 2 yet.

Acceptance criteria:
- scripts/smoke_tests/gpu_smoke_test.py exists and runs.
- scripts/smoke_tests/torch_autograd_test.py exists and runs.
- scripts/slurm/gpu_smoke_test.slurm exists with placeholders for HPC-specific setup.
- phase01_research_engineering/week01_environment_hpc/linear_regression_torch.py exists and can save a log and loss plot.
- notes/weekly/week01_environment_hpc.md exists and contains the checklist.
- README.md and TODO.md exist.
```
