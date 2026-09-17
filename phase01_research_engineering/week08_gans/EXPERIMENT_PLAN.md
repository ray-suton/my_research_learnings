# Experiment Plan

## Status and Learning Question

The first measured baseline completed on **2026-09-15**, following the learner's
explicit request to perform the practical Session 3 work. The original model
architecture and selected sigmoid/BCELoss pairing were used; the recorded D
negative slope is 0.01. [Run notes](results/runs/session3_mnist_mlp_seed42_20260915/notes.md) retain actual settings,
results, source snapshots, and limitations.

Learning question: can a small unconditional MNIST GAN learn recognizable and
varied digit-like samples, and what does alternating training reveal about
instability and mode collapse? The short run produced imperfect digit-like
structures; it does not establish distribution matching or convergence.

## Dataset and Storage

- Dataset: MNIST, as specified by the Week 8 roadmap.
- Input: grayscale images, `(1, 28, 28)`; 784 pixels when flattened.
- Convert pixels from `[0, 1]` to `[-1, 1]` to match generator `tanh` outputs.
- Train/validation: deterministic 55,000/5,000 split of official training data,
  seed 42. Only training images enter discriminator optimizer updates.
- Test: official 10,000 images, reserved for any final fixed-protocol reference
  comparison. No test-set reconstruction score is planned.
- Ignore digit labels in both training objectives.
- Resolve `AI_ROADMAP_DATA` with fallback `/home/rui.gao/data/ai_research_roadmap`;
  the loader should honor the environment variable.
- Expected cache: `<data_root>/raw/torchvision/MNIST/`.

At initialization MNIST was absent. It was prepared on 2026-09-15 with the
existing data script: 60,000 official training and 10,000 test images. The
loader makes the seeded 55,000/5,000 split and disables downloads during runs.
The preparation manifest path is recorded in the run notes.

Existing preparation commands, to run from the repository root before
Session 3's dataset-backed work, are:

```bash
python3 scripts/data/prepare_standard_datasets.py --dataset mnist --dry-run
python3 scripts/data/prepare_standard_datasets.py --dataset mnist
```

The command without --dry-run was executed for this baseline; the preview was not. The existing
`--dry-run` skips downloads but still creates data-root directories and a
preparation manifest. Actual access, dataset counts, versions, and preparation
outcome must be logged then.
Ordinary training should read the cache with downloads disabled.

## First Model and Shape Contract

Recorded baseline: MLPs with no batch normalization or dropout, keeping the
first gradient-flow checks easy to interpret.

| Quantity | Shape / choice |
| --- | --- |
| Real image batch | `(B, 1, 28, 28)` |
| Noise, standard normal | `(B, 64)` |
| Generator | `64 -> 128 -> 256 -> 784`, ReLU hidden layers, tanh output |
| Generated image batch | `(B, 1, 28, 28)` after reshape |
| Discriminator | flatten, `784 -> 256 -> 128 -> 1`, LeakyReLU hidden layers |
| Discriminator hidden negative slope | `0.01` in the measured run; initial `0.2` proposal not used |
| Discriminator output | sigmoid probabilities, `(B, 1)` |
| Real/fake targets | floating-point tensors `(B, 1)` on the same device |
| Each reduced loss | scalar |

The discriminator accepts the same image layout for real and generated data.
Handle the final short batch and `B=1` without dropping the output dimension.
Convert generated pixels back with `(image + 1) / 2` for display.
The measured run preserved the learner's default negative slope of 0.01.

## Objectives and Update Contract

Let `s(x)` be the discriminator logit and `D(x) = sigmoid(s(x))`. The original
game is `min_G max_D E[log D(x)] + E[log(1 - D(G(z)))]`. For implementation,
use the non-saturating generator objective described in
[the original paper, Section 3](https://arxiv.org/html/1406.2661v1#S3).

The learner chose D's sigmoid output with plain BCE during Session 2.
With `BCE` meaning mean-reduced `BCELoss`, minimize:

```text
L_D_real = BCE(D(real), ones)
L_D_fake = BCE(D(fake_detached), zeros)
L_D = L_D_real + L_D_fake
L_G = BCE(D(G(z)), ones)
```

`L_D` is the sum of two batch means, with no additional division by two.
Real/fake targets describe origin; they are not digit classes. D already
applies sigmoid, so pass its probabilities directly to
[PyTorch BCELoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)
with target tensors matching prediction shape, dtype, and device. Do not
apply another sigmoid. The original raw-logit/BCEWithLogitsLoss proposal is
an alternative pairing, not the current learner implementation.

One training iteration has two distinct updates:

1. Clear relevant gradients, generate fake images, detach them for the
   discriminator loss, and step only the discriminator optimizer.
2. Clear G's gradients and hold D's parameters fixed. After D's update,
   recompute D's prediction on the original connected generated images and
   calculate G's loss. Differentiate through D's input into G and step only
   G's optimizer. Restore D's parameter-gradient settings for its next update.

Do not detach fake images or wrap the discriminator forward pass in `no_grad`
during the generator update. Freezing parameters still allows input gradients.
Fresh noise is drawn for each training iteration; fixed monitoring noise is
separate. The same G output can serve both phases when D's phase used its
detached view. Alternatively, regenerate images for G's phase. A second G
forward pass is optional; the post-update D prediction is the required refresh.
The [PyTorch tutorial training loop](https://github.com/pytorch/tutorials/blob/main/beginner_source/dcgan_faces_tutorial.py)
provides a reference for detaching during the discriminator update and
re-evaluating the discriminator for generator training.

## Recorded First Run

| Setting | Initial choice |
| --- | --- |
| Epochs | 5, after a tiny smoke run |
| Batch size | 128 |
| Training/split seed | 42 |
| Generator/discriminator optimizer | separate Adam instances |
| Learning rates | `0.0002` for both |
| Adam betas | `(0.5, 0.999)` |
| Updates per minibatch | 1 discriminator, then 1 generator |
| Monitoring | 64 fixed noise vectors, separate seed 123 |
| Final sample panel | 64 noise vectors, separate seed 456, after setup is fixed |
| Device | CPU, two threads; CUDA unavailable in the execution session |
| Workers | 0 initially |

Measure one epoch's time before choosing whether to use the available HPC GPU.
Do not promise recognizable digits within five epochs. Record any change to
the budget before rerunning rather than replacing an unsuccessful run.

## Evidence and Comparisons

- Save the untrained generator's grid as the visual baseline, then the same
  64 noise inputs at each epoch. Save full panels, not selected good examples.
- Log `L_D_real`, `L_D_fake`, `L_D`, and `L_G`; aggregate epoch batch means
  weighted by batch size.
- Log mean `sigmoid(s(real))` and generated-image scores with the update phase
  stated. Scores and losses diagnose this changing two-model system; they do
  not rank generator quality independently.
- Fix validation reference images before the first run. Use those references
  for visual inspection, without feeding them into optimizer updates.
- Record recognizable shapes, variety, repeated outputs, blank/noisy images,
  and changes across epochs. These are qualitative observations, not class
  coverage or distribution-matching proof.
- Report the final epoch by the declared budget; do not select a checkpoint
  solely because its generator loss is lowest.
- Save the separate final noise panel only after architecture and budget are
  fixed. A fresh panel checks dependence on the monitoring examples but does
  not constitute a quantitative held-out-data benchmark.
- Save exact configuration, command, source revision or hashes, versions,
  device, runtime, and checkpoint paths. Checkpoints live under the external
  data root; use new run directories for each attempt.

The optional DCGAN-style variant must keep MNIST at 28-by-28, the split,
normalization, noise panels, and training-step budget fixed. Document its
architecture before running, including changed parameter counts. If a resource
constraint requires another change, label the comparison accordingly.

The Week 7 VAE used Fashion-MNIST. Use it for conceptual comparison only;
GAN adversarial losses cannot be compared numerically to VAE reconstruction
MSE or ELBO, and this unconditional GAN has no paired reconstruction target.

## Failure Criteria and Limits

- Incorrect shapes, ranges, targets, non-finite losses, or wrong parameter
  updates block interpretation until the implementation is checked.
- Repeated images from different noise inputs suggest mode collapse even if
  some samples look convincing.
- Discriminator scores near 0.5 can reflect a weak or broken discriminator;
  they do not establish successful generation.
- Oscillating losses alone do not identify a bug or prove healthy learning.
  Inspect sample evolution and the two update paths together.
- One seed and visual inspection cannot establish general superiority,
  coverage of all digit modes, or absence of memorization.
- No FID/IS pipeline, external classifier, or new dependency is required for
  this first learning exercise. Stronger generation claims would need a
  separately specified evaluation protocol.

## Deliverables

Passing planned [smoke checks](tests/README.md), one measured MLP run,
loss/score records, fixed and final sample panels, a failure-case note, an
optional architecture comparison, and a final lab note. The baseline, checks,
sample panels, and failure observations are complete in the Session 3 run.
The optional Session 4 comparison and final Week 8 lab note remain pending.
