# Phase 1 Roadmap - Research Engineering and Traditional Neural Networks

Parent roadmap: `../ai_research_roadmap_codex.md`

Scope note: this file is the Phase 1 slice of the Year 1 core roadmap. The root roadmap remains the source of truth for continual-learning extensions, research-process rules, and any later pacing changes.

Phase goal: build the local/HPC research workflow, then implement core neural-network families with controlled experiments and short lab notes.

Research-process overlay:

- From Week 2 onward, define the hypothesis, baseline, metric, split, and falsification condition before coding.
- Save meaningful runs under `results/runs/YYYYMMDD_HHMM_experiment_name/`.
- Track config, metrics, stdout, notes, plots, runtime, and memory where relevant.
- Keep failure cases and limitations, not only aggregate scores.
- Treat Week 4 as the first reusable experiment-template checkpoint.

## Weeks

### Week 1 - Environment, Git, PyTorch, HPC bootstrap

Goal: create a clean research workspace, verify local PyTorch, prepare GPU/HPC workflow, and run a tiny sanity experiment.

Deliverable: working repo, Python environment, GPU smoke test, SLURM template, tiny PyTorch training script, Week 1 lab note.

Detailed daily bootstrap plan remains in the root roadmap's exact Week 1 section.

### Week 2 - MLPs, datasets, training loops

Goal: build the first proper supervised-learning experiment.

- Monday: Review PyTorch datasets, dataloaders, losses, optimizers, train/eval mode.
- Tuesday: Implement an MLP classifier on MNIST or Fashion-MNIST.
- Wednesday: Add validation split, accuracy, loss curves, checkpointing, seed control.
- Thursday: Run three experiments: baseline, different learning rate, different hidden size.
- Friday: Plot results and write a lab note comparing the runs.

Deliverable: MLP experiment with reproducible config, logs, checkpoint, and plot.

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

## Phase Outputs

- Reproducible PyTorch/HPC experiment template.
- MLP, CNN, mini-ResNet, RNN/LSTM, autoencoder, VAE, and GAN experiments.
- Configs, logs, plots, and Friday lab notes for each project.
- First project-level testing and experiment-tracking pattern.

## Continual Extension Connections

Phase 1 creates the infrastructure that later extension tracks reuse:

- AI systems, scaling, and ML infrastructure: profiling, checkpoint/resume, SLURM recovery, pinned environments.
- Data-centric AI and evaluation design: split discipline, leakage checks, repeated runs, failure tables.
- Research methods and scientific writing: hypotheses, baseline comparisons, ablations, limitations, and reproducible reports.
