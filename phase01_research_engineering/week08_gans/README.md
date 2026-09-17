# Week 8 - Generative Adversarial Networks

## Purpose

Week 8 introduces adversarial generation after the Week 7 VAE. Learn how a
generator and discriminator provide each other's training signal, why their
updates must be handled separately, and how convincing samples can coexist
with poor diversity or unstable training.

The target is a small, inspectable MNIST GAN with recorded sample grids,
training diagnostics, and failure analysis. Start with fully connected models;
try one DCGAN-style variant only if the baseline and time budget allow it.

## Current Checkpoint

**All Week 8 GAN work is finished, including Session 5 and the Fashion-MNIST
extension, at the learner's request on 2026-09-17.**
The [conceptual review](session5.md) records both demonstrated reasoning and
guided corrections; the [final lab note](LAB_NOTE.md) consolidates the results,
VAE/GAN comparison, and limits. No additional closing quiz is required.
The separate CNN-only twenty-epoch Fashion-MNIST extension is also complete:
8,600 paired updates, 34 passing preflight tests, and verified checkpoint/sample
replay. The [extension record](fashion_mnist_extension.md) contains the results,
[sample comparison](results/runs/fashion_mnist_cnn_seed42_20epochs_20260917/real_vs_generated.png),
and [loss graph](results/runs/fashion_mnist_cnn_seed42_20epochs_20260917/audit_d_g_losses.png).
The panels show garment-like structure with remaining artifacts and limited
apparent class variety. Week 9 tokenization/text data is next.

The final discussion identified likely partial mode collapse / mode dropping
in the Fashion-MNIST panels. Conditional generation and diversity-aware
training were discussed as possible remedies; they were not implemented or
run. The learner chose to stop GAN work here. These limitations are retained
as findings, with no further GAN experiment or closing quiz planned.

## Completed MNIST Comparison

**The twenty-epoch MLP/CNN comparison is complete.** The prior MLP result was
reused after verification; the CNN completed twenty epochs, recovering from
an interruption after epoch 12 with verified optimizer/RNG restoration.
Both have 8,600 paired updates. All forty checkpoint hashes and both models'
epoch-5/10/20 and final sample replays passed. The CNN panel is cleaner and
visibly more varied, with some ambiguous digits remaining. See the
[final comparison](results/runs/session4_mnist_cnn_seed42_20epochs_20260916/mlp_vs_cnn_epoch20.png),
[loss graph](results/runs/session4_mnist_cnn_seed42_20epochs_20260916/mlp_vs_cnn_losses_20epochs.png),
and [run notes](results/runs/session4_mnist_cnn_seed42_20epochs_20260916/notes.md).
The interpretation and final lab note are now recorded in Session 5.

## Five-Epoch Session 4 Checkpoint

**Session 4's five-epoch CNN comparison completed on 2026-09-16.** The learner
implemented the model pair and explicitly requested CLI/config integration
and execution. All 25 checks passed. The run completed 2,150 paired updates
in 262.733 seconds of CPU training and retained five checkpoints with exact
final-sample replay. The [full comparison](results/runs/session4_mnist_cnn_seed42_20260916/mlp_vs_cnn_epoch05.png)
shows cleaner backgrounds and more continuous strokes than the five-epoch
MLP, while many digits remain malformed. See [session4.md](session4.md) and
[run notes](results/runs/session4_mnist_cnn_seed42_20260916/notes.md).
The subsequent discussion and lab note are recorded in Session 5.

## Retained Session 3 Checkpoint

**Session 3 is closed, including the measured baseline and twenty-epoch
follow-up.** Final documentation was completed on 2026-09-15. Earlier that day the
learner explicitly requested practical execution after the guided discussions.
The MLP GAN trained for five epochs on 55,000 MNIST images using the declared
seed-42, batch-128, sigmoid/BCE setup. All 11 checks passed. The selected
checkpoint is the predeclared final epoch, and test images were unused in
measured training, scoring, and selection.

[Real-versus-generated comparison](results/runs/session3_mnist_mlp_seed42_20260915/real_vs_generated.png) ·
[Single-image pair](results/runs/session3_mnist_mlp_seed42_20260915/real_vs_generated_single.png) ·
[Training curves](results/runs/session3_mnist_mlp_seed42_20260915/training_curves.png) ·
[Run notes and reproduction](results/runs/session3_mnist_mlp_seed42_20260915/notes.md)

Generated samples have digit-like structure but remain noisy and malformed.
This is a recorded one-seed baseline, not evidence of convergence. See
[RESULTS.md](RESULTS.md) and [session3.md](session3.md) for the measurements
and limitations. MNIST and the five checkpoints live under the external data
root. The completed Session 4 comparison is recorded above.

A [20-epoch training-budget follow-up](results/runs/session3_mnist_mlp_seed42_20epochs_20260915/notes.md) also completed with
unchanged source/settings and an exactly matching first five epochs.
[Same-noise progression](results/runs/session3_mnist_mlp_seed42_20epochs_20260915/epoch_05_10_20_comparison.png) shows a clear improvement
in individual-image quality, while many 1-like outputs remain a separate
diversity concern. Both runs are
retained. Session 3 has no remaining execution tasks.

## Flexible Session Plan

### Session 1 - Concepts, Objectives, and Shapes

- Connect VAE prior sampling to the GAN generator's noise input.
- Explain generator/discriminator roles and the minimax game.
- Distinguish real/fake targets from MNIST digit labels.
- Trace the two updates and define the image, noise, and logit shapes.
- Explain why neither reconstruction MSE nor discriminator accuracy is a
  standalone GAN-quality measure.

Deliverable: completed discussion of [session1.md](session1.md) and a clear
shape/loss contract before coding.

Outcome on 2026-09-11: conceptual discussion concluded. The remaining
shape/loss and gradient checks move to the start of Session 2 before coding;
they are not recorded as independently passed.

### Session 2 - Minimum Trainable GAN

- Review exact losses, tensor shapes, pixel ranges, and both gradient paths
  using the Session 1 carryover checklist.
- Prepare or verify MNIST under the shared external data root.
- Implement a small MLP generator and discriminator using two optimizers.
- Add checks for shapes, ranges, loss targets, gradient flow, and isolated
  parameter updates.
- Run a tiny synthetic-batch check before dataset-backed training.

Original practical deliverable: runnable baseline and passing smoke checks.
The learning discussion is concluded. Those execution checks were subsequently completed for the explicitly requested
Session 3 baseline.

### Session 3 - First Measured Run and Failure Inspection

- Resolve the learner draft's remaining runtime/data-flow blockers and verify
  two complete D/G cycles on one synthetic minibatch.
- Prepare or verify MNIST and its documented splits before dataset-backed work.
- Run the predeclared short baseline.
- Save samples from the same noise vectors before training and after each epoch.
- Track both losses, real/fake discriminator scores, runtime, and configuration.
- Inspect blank images, repeated outputs, unstable losses, and stalled updates.

Deliverable: a measured run, complete sample grids, and evidence-backed notes.

Learning discussion: concluded with guided corrections on 2026-09-15.
The practical deliverable above completed on 2026-09-15 after explicit execution authorization.

### Session 4 - One Controlled Comparison

- If feasible, compare with one small DCGAN-style architecture.
- Preserve dataset, image resolution, split, seed, noise panels, and step budget.
- Record capacity and runtime differences; this is an architecture comparison.
- If time is tight, defer the variant and document one baseline failure case.

Deliverable: a comparison or an explicit deferral with failure analysis.

The five-epoch CNN comparison completed on 2026-09-16; see
[session4.md](session4.md) for the measured result and discussion handoff.

### Session 5 - Lab Note and VAE Comparison

- Summarize what worked, what failed, and what the evidence cannot establish.
- Compare VAE and GAN training signals and generation conceptually.
- Keep Week 7 Fashion-MNIST and Week 8 MNIST results separate; the datasets and
  objectives do not support a numerical claim that one model family is better.
- Record what carries forward into Week 9 tokenization and text data.

Deliverable: final results, sample/failure comparison, and `LAB_NOTE.md`.

Concluded with guided corrections on 2026-09-17. See [session5.md](session5.md)
and [LAB_NOTE.md](LAB_NOTE.md); this is a learning checkpoint, not a claim of
independent formal mastery.

## Working Rules

- Normal lessons are learner-owned coding with instructor guidance. The
  learner explicitly requested practical baseline completion on 2026-09-15;
  that request authorized supporting repairs and execution for this run.
- The first GAN is unconditional: neither network receives digit labels.
- Keep training diagnostics separate from claims about sample quality.
- Retain complete fixed-noise grids, including failed samples.
- Use validation references during development; keep test images out of tuning.
- Keep datasets, caches, checkpoints, and bulky logs outside Git under
  `AI_ROADMAP_DATA`.
- Start with direct functions and small scripts, matching earlier weeks.

## Folder Guide

Existing documentation:

- [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md): proposed data, shapes, objectives,
  settings, evidence, and failure criteria.
- [session1.md](session1.md): concept notes, discussion evidence, corrections,
  and the Session 2 carryover checklist.
- [session2.md](session2.md): concluded learning discussion, assessment record,
  and practical checks carried into Session 3.
- [session3.md](session3.md): completed discussion, baseline, and measured diagnostic.
- [session4.md](session4.md): learner-designed CNN pair, controlled run, and comparison.
- [session5.md](session5.md): completed conceptual review and assessment limits.
- [LAB_NOTE.md](LAB_NOTE.md): final Week 8 synthesis and VAE/GAN comparison.
- [RUN_LOG.md](RUN_LOG.md): executed commands and honest status.
- [RESULTS.md](RESULTS.md): measured baseline, images, and limitations.
- [configs/README.md](configs/README.md): recorded configuration and reproduction.
- [tests/README.md](tests/README.md): recorded verification checks and commands.
- [results/README.md](results/README.md): artifact policy.

The baseline uses [model.py](model.py), [train.py](train.py),
[loader.py](loader.py), and [run_baseline.py](run_baseline.py). Its model
architecture is the learner's; the training path and supporting run machinery
were completed for explicitly requested experiments. All five core sessions
and the final Week 8 lab note are now documented.

## Reading

Start with Section 3 of the [original GAN paper](https://arxiv.org/html/1406.2661v1#S3).
Use the [official PyTorch DCGAN tutorial source](https://github.com/pytorch/tutorials/blob/main/beginner_source/dcgan_faces_tutorial.py)
for alternating updates and fixed-noise monitoring. Its face-image architecture
is a later reference, not the Week 8 28-by-28 baseline specification.
