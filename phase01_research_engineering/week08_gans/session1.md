# Session 1 - GAN Concepts, Objectives, and Shape Contract

## Status

Conceptual discussion concluded on **2026-09-11**, with guided corrections.
The learner requested closure after demonstrating the core intuition about
alternating updates, mode collapse, and fixed-noise comparisons. Session 2 is
next. Exact loss equations, tensor shapes, pixel normalization, and a practical
gradient-path check carry into its opening checkpoint; they have not been
independently verified. No GAN implementation or experiment has run.

## Goal

Explain how a model can learn to generate images without reconstructing a
particular input image, and trace which network learns during each update.
Use roughly one 90-minute session: concepts, objective/shape discussion, then
checkpoint questions. Adjust the pace to understanding.

## From the VAE to a GAN

Your Week 7 VAE could generate by drawing a normal noise vector and passing it
through the decoder. The GAN generator also maps random noise to an image.
Its training feedback comes from a learned discriminator.

```text
normal noise z -> generator G -> generated image -> discriminator logit
real MNIST image -------------------------------> discriminator logit
```

The first GAN has no encoder, posterior mean/log-variance heads, paired image
reconstruction target, or explicit VAE KL term. Real examples teach the
discriminator; the generator receives gradients through that discriminator.

## Two Roles and One Game

The discriminator learns to distinguish real dataset images from generated
images. The generator learns to make its outputs receive the discriminator's
real target. Both change during training, so each network faces a changing
objective. The original minimax objective and practical non-saturating
generator alternative are introduced in
[the GAN paper, Section 3](https://arxiv.org/html/1406.2661v1#S3).

Think of the two losses as separate instructions:

- Discriminator: score real images as real and generated images as fake.
- Generator: change generated images so the current discriminator scores them
  as real.

The discriminator's probability-like score is a learned judgement, not an
external certificate that an image is realistic.

## Labels and Noise

Digit labels such as 2 or 7 are unused in this unconditional model. The
discriminator's targets are 1 for real and 0 for generated. The generator uses
the target 1 when learning from discriminator feedback; that does not relabel
the generated image as actual training data.

`z` is sampled independently from a standard normal distribution. For a fixed
trained MLP generator with no dropout, the same `z` produces the same image.
Different inputs need not map to distinct outputs: repeated outputs are one
failure to watch for.

## Shapes and Pixel Ranges

The first experiment uses MNIST and the following contract:

```text
noise:                  (B, 64)
generator flat output:  (B, 784)
real/generated image:   (B, 1, 28, 28), values in [-1, 1]
discriminator logits:   (B, 1)
real/fake targets:      (B, 1), floating point
each loss:              scalar
```

Real images are normalized to match the generator's tanh output. A range
mismatch gives the discriminator an irrelevant shortcut.

The discriminator produces logits. `BCEWithLogitsLoss` handles sigmoid
internally; applying sigmoid first would change the intended computation.
[PyTorch documents this loss contract](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html).
The full shapes, defaults, and loss equations are in
[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md).

## Trace the Two Updates

For the discriminator update, generated images are detached so this loss does
not train the generator. For the generator update, keep the discriminator's
parameters fixed but preserve differentiation through its input. The
generator needs that feedback path. Use a fresh forward pass after stepping
the discriminator.

The [official PyTorch tutorial source](https://github.com/pytorch/tutorials/blob/main/beginner_source/dcgan_faces_tutorial.py)
shows separate optimizers, detached discriminator inputs, and fixed-noise
monitoring. Read the training-loop comments first; the later convolutional
model is not needed to understand the initial MLP.

## What Would Count as Progress?

We want samples to evolve from the untrained baseline toward recognizable,
varied digit-like shapes, with finite losses and verified parameter updates.
Inspect the same noise panel across epochs and retain failures.

A few sharp digits can hide repeated outputs. A discriminator near 50% can
also be weak or poorly trained. Neither observation alone proves the
generator matches the data distribution. GAN loss is not a replacement for
the VAE's reconstruction metric.

## Checkpoint Questions

These original prompts remain as a reference. The discussion record below
distinguishes demonstrated understanding from explanation and deferred checks;
the entire list has not been independently passed.

1. In Week 7, an input image helped train the VAE decoder through
   reconstruction. In a GAN, what tells the generator how to change its image?
2. Why does the discriminator output one score instead of ten digit scores?
3. Which network's parameters should change during each of the two updates?
4. Why detach generated images for the discriminator step but keep their
   gradient path for the generator step?
5. What shortcut could the discriminator learn if only real images lie in
   `[0, 1]` while generated images lie in `[-1, 1]`?
6. If different noise vectors all produce convincing versions of the digit 3,
   what has worked and what has failed?
7. Why are fixed-noise grids useful, and why is low generator loss insufficient
   evidence of a good generator?

## Definition of Done

Explain the two roles, the real/fake targets, the noise/image/logit shapes,
both gradient paths, and one meaningful failure case. This is a practical
checkpoint; a formal convergence proof is not required.

Closure decision on 2026-09-11: the conceptual discussion is sufficient to move
forward. The shape/range and exact loss/gradient checks from this original
target carry into Session 2 rather than being marked complete without evidence.

## Discussion Record - 2026-09-11

This records the learner's answers and the explanations supplied during the
conversation. It is a guided learning checkpoint, not an independent exam or
an implementation test.

### Understanding Demonstrated

| Topic | Evidence from the discussion | Assessment |
| --- | --- | --- |
| Generator goal | Identified that G wants D to call its generated images real. | Core objective understood. |
| Alternating updates | Stated that updating D leaves G unchanged and updating G leaves D unchanged. Predicted that better fake detection raises G's loss while G is fixed. | Parameter-update separation understood after correction of the initial update rule. |
| Detach | After explanation, stated that D can learn from a detached generated image while G cannot learn through that image. | Basic effect understood with guidance; independently tracing the G-step gradient path remains a practical check. |
| Mode collapse | Described exploiting the current discriminator as reward hacking and recognized that different noise inputs can produce the same output. | Understood that realism and random inputs do not guarantee diversity. |
| Independent evaluation | Proposed using human judgment to evaluate images when D may be weak. | Useful evaluation idea; limits and the need to inspect realism and diversity were explained. |
| Fixed-noise monitoring | Explained that the same noise should yield more realistic images as training improves G, calling this a controlled experiment. | Correctly connected fixed inputs with isolating model changes. |

### Corrections And Concepts Covered

- A GAN shares the VAE decoder's noise-to-image idea, but its generator learns
  from a discriminator rather than a paired reconstruction target. The basic
  generator is not a denoiser recovering a clean image hidden in its noise.
  D is a real/fake binary classifier. Either model can use fully connected or
  convolutional components; the first planned baseline uses MLPs.
- A correct binary decision does not generally have zero cross-entropy loss.
  D and G use different targets for generated images. The baseline alternates
  one D step and one G step; a higher G loss does not automatically schedule
  extra G updates. The initial success/failure-gating idea was corrected.
- Freezing D's parameters during the G step preserves differentiation through
  D's input. Detaching the generated image cuts its connection back to G;
  this is appropriate for the D step but blocks the intended G-step feedback.
  The distinction between separate parameters and dependent gradients needed
  explanation and should be revisited in code.
- The ideal equilibrium was explained as matching the real distribution,
  with an optimal discriminator output of 0.5 under balanced real/fake
  sampling. Neither zero loss nor observed 0.5 scores certify convergence;
  an actual neural-network training run is not guaranteed to reach that ideal.
- In the balanced mixture example, 100 real images included 10 threes, while
  100 generated images were all threes. Assuming generated threes have the
  same appearance distribution as real threes, the real fraction among threes
  is 10/110, about 9.1%. The learner estimated 10%; the denominator was
  corrected. D can penalize overrepresented outputs without an explicit
  repetition penalty, although G may then exploit another current weakness.
- Human review and complete sample grids can reveal failures that losses
  hide. They do not establish distribution matching or rule out memorization.
  Fixed monitoring noise is separate from fresh noise used for training.

The instructor checked the [original GAN paper](https://arxiv.org/html/1406.2661v1)
while explaining objectives and equilibrium. This does not record the learner
as having independently read the paper or derived its results.

### Session 2 Handoff

Start with a short practical review tied to implementation:

1. Write the exact real/fake targets and D/G losses from
   [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md), including the logits convention
   and loss reductions. These were explained but not independently derived.
2. Check the declared noise, image, and logit shapes and matched pixel ranges.
   The normalization-shortcut question was not assessed in this discussion.
3. Trace both updates, including detach in the D step and differentiation
   through frozen D parameters in the G step. Then verify the behavior with
   the planned [smoke checks](tests/README.md) when code exists.
4. Prepare or verify MNIST under the shared external data root before
   dataset-backed execution. The last inspection found the cache absent;
   no preparation occurred during this lesson.

Continue with learner-owned implementation. Prefer intuitive predictions,
failure analysis, and experiment-design questions over repeated easy recall
questions, as requested by the learner. No training result should be recorded
before a run occurs.
