# Session 2 - Minimum Trainable GAN

## Status

Started on **2026-09-12** in guided learning mode. Session 1's conceptual
discussion is concluded. **Session 2's concepts and research-practice
discussion concluded on 2026-09-14 with guided corrections**, and its closure
was documented on **2026-09-15** at the learner's request.

This completes the agreed learning discussion. Runnable-code verification is
carried into Session 3: the September 14 preflight passed D's synthetic
forward/BCE checks for batches 1 and 8, but G construction failed on undefined
`x`, and the training draft referenced undefined `model` and `z`. Both source
hashes still match that preflight on September 15. At that discussion closure, full D/G updates,
parameter isolation, and repeated iterations had not passed verification;
the later practical completion below supersedes that status.

At entry, Week 8 contained documentation only. A read-only filesystem check
found `AI_ROADMAP_DATA` unset and the expected MNIST directory absent at
`/home/rui.gao/data/ai_research_roadmap/raw/torchvision/MNIST`. No download or
data preparation was run. This does not block the opening conceptual checks
or later synthetic-batch tests.

## Later Practical Completion - 2026-09-15

After the discussion closure, the learner explicitly requested execution of
the full Session 3 baseline. Its supporting training repairs, loader, and
verification were completed. All 11 checks passed and the five-epoch MNIST
run completed. The old draft failures below are historical and do not describe
the current runnable baseline. See [Session 3](session3.md) and the
[run evidence](results/runs/session3_mnist_mlp_seed42_20260915/verification.txt). The original understanding assessment
and normal teacher-guided preference are unchanged.

## Goal

The original practical goal was to build a small unconditional MNIST GAN and
verify that each optimizer updates the intended network. The learner scoped
this session's closure around concepts and research thinking/practice;
the remaining execution checks must precede the measured Session 3 run.
Continue from the MLP baseline in
[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md). The learner writes the implementation;
the instructor guides reasoning and reviews it unless direct implementation
is explicitly requested.

## Practical Work Sequence And Handoff

1. Review the practical carryovers from [Session 1](session1.md): matched
   pixel ranges, tensor shapes, exact loss/target conventions, and both
   gradient paths. Start with the normalization-shortcut thought experiment.
2. Implement the generator and discriminator, using the declared noise,
   image, and probability-output shapes. Keep the first models small and inspectable.
3. Implement separate D and G updates and check their behavior on tiny
   synthetic batches using the contract in [tests/README.md](tests/README.md).
4. Before the Session 3 experiment, prepare or verify MNIST under the external
   data root, then check normalization, deterministic splits, and cached
   loading with downloads disabled before dataset-backed execution.
5. Record actual commands and outcomes in [RUN_LOG.md](RUN_LOG.md). Save
   passing check evidence when checks exist and have run. The first measured
   baseline and sample evolution belong to Session 3.

## Opening Checkpoint

The planned generator uses a tanh output, with pixel values in `[-1, 1]`.
Suppose real images are instead supplied in `[0, 1]`. Explain what shortcut
the discriminator could exploit before learning digit shapes.

Discussed on 2026-09-14: the learner identified negative pixel values and
image statistics as possible shortcuts. This demonstrates the conceptual
point; it does not verify a real-data normalization implementation.

## Technical Verification Carried Into Session 3

Scope agreed on 2026-09-14: one synthetic minibatch is enough for the core
mechanics, with two consecutive D/G cycles to check reuse of the loop. MNIST
preparation and measured experiments move to Session 3; they are deferred,
not complete.

- [x] Learner-owned model and training drafts exist and have been reviewed.
- [x] G construction and the full synthetic update execute successfully.
- [x] D's real/fake losses and G's real-target loss are finite in that update.
- [x] D's step changes D only; G's step changes G only with nonzero G gradients.
- [x] Both gradient paths and restoration of D's gradient settings are verified.
- [x] Two consecutive cycles work without stale/freed-graph errors.
- [x] The concepts/research-practice discussion concluded with guided corrections.
- [x] Reproducible successful update commands and results are retained.

## Closure Preflight - 2026-09-14

The [verification report](../../.omx/reports/task-2026-09-14-212821.md)
retains source hashes, runtime output, and the undefined-name diagnostic.
Syntax and import checks passed. D's probability shape/range and BCE
interface checks passed on synthetic CPU inputs. G construction failed;
full iteration, update-isolation, and repeatability checks were not run.
No implementation or test file was modified for these checks.

The learner selected a sigmoid output with plain BCE. The earlier raw-logit
proposal is not the interface being reviewed. The learner also clarified
that the focus is concepts and research thinking/practice. The assistant
guides and verifies; the learner writes the implementation. In this teaching
context, requests to work through implementation together do not authorize
the assistant to write it on the learner's behalf.

## Closing Assessment - 2026-09-14

| Topic | Learner response or work | Assessment |
| --- | --- | --- |
| Gradient isolation | Expected no new G gradients during D's step because the generated image was detached. | Correct expectation for the hypothetical check after old gradients are cleared. |
| Gradients versus parameter updates | Explained that G's weights can remain unchanged because G's optimizer is not used. | Correctly distinguished gradient calculation from applying an update. |
| Controlled comparison | Proposed matching training-update counts when comparing GAN architectures. | Correctly identified extra training as a confound. |
| Repeated experiments | Proposed folds and seeds to investigate a lucky result. | Seeds directly address training randomness; the different role of folds was explained. |
| Interpreting generator loss | Answered that lucky runs can explain lower average G loss. | Incomplete: the instructor also explained discriminator strength and mode collapse; the corrected claim still needs care in future experiments. |

The closing decision is sufficient core intuition for this learning stage,
with guidance. It is not a claim of independent mastery of every equation,
a working training run, or successful generation.

## Concepts And Corrections Retained

- D is a binary classifier of image origin. Reusing its weights on real and
  fake batches does not make this a Siamese objective: each score is compared
  with its own target, rather than with another image's representation.
- The learner chose sigmoid probabilities with plain BCE. D minimizes the
  sum of real-target-1 and fake-target-0 BCE terms. G minimizes BCE of D's
  generated-image prediction against target 1; it has no paired real-image
  reconstruction target. Targets match prediction shape, dtype, and device.
- Updates follow the alternating schedule, not a rule that selects a model
  to update only when D's hard classification is wrong. Continuous predictions
  can have nonzero BCE loss even when their thresholded classification is right.
- Detach leaves pixel values unchanged while cutting a gradient connection.
  D uses a detached view of the generated image. G later uses the original
  connected image so feedback can reach its parameters.
- G's image may be reused across the two steps. After D changes, recompute
  D's prediction and G's loss using the updated D. A second G forward pass
  is optional, not a requirement. The instructor corrected earlier wording
  that unnecessarily required regenerating the image.
- A hypothetical nonzero G gradient after D-only backward was an assessment
  scenario, not an observed defect in the learner's detached branch. The
  actual full update could not execute during preflight.
- Matching update counts controls the training-step budget; report runtime
  and model capacity as well. Repeated training seeds and data folds probe
  different sources of variability. Lower G loss alone cannot establish
  better images; examine realism, diversity, and complete sample panels.

## Remaining Practical Work

Before claiming a successful Session 3 experiment, resolve the current
constructor/name/data-flow blockers and the noise batch shape, then verify
one synthetic minibatch for two consecutive D/G cycles. Record actual loss
values, gradient paths, parameter changes, and restored D settings. Prepare
MNIST and its evaluation protocol afterward; no dataset run is recorded yet.

The current `model.py` and `train.py` are learner-owned. The earlier
assistant-written implementation and tests were reverted at the learner's
request; their seven passing checks do not apply to the current draft.

The assistant remains a teacher: inspect and explain, run agreed checks,
and document requested outcomes. Do not implement or fix code on the learner's
behalf from phrases such as “let's implement” or “finish the session.” Prefer
intuitive predictions and research reasoning over repeated easy questions.
The closing quiz is finished; do not restart it automatically.

See the [documentation closure report](../../.omx/reports/task-2026-09-15-103317.md)
for this update's scope and verification.
