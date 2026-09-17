# Session 3 - First Measured Run And Failure Inspection

## Status

**Session 3 is complete and closed: understanding discussion, five-epoch
baseline, and controlled twenty-epoch follow-up.** Final documentation was
completed on 2026-09-15 at the learner's request.
The discussion concluded earlier on 2026-09-15 with guided corrections.
The learner then explicitly requested that the assistant finish the actual
baseline run and supporting work. That operational request authorized the
necessary training repairs, loader, checks, execution, and artifacts for this run.

The [five-epoch MNIST run](results/runs/session3_mnist_mlp_seed42_20260915/notes.md) completed on 2026-09-15. All 11
checks passed, including the 55k/5k/10k split, normalized batch interfaces,
separate parameter updates, repeated cycles, and monitoring RNG preservation.
The actual run processed 55,000 training images per epoch; the official test
split did not enter its training, scoring, or selection. Checkpoints, source
snapshots, full sample panels, metrics, and comparisons are saved.

The [earlier constructor/training blockers](../../.omx/reports/task-2026-09-15-112340.md)
are historical and were resolved for this explicitly requested execution.
The learner's model architecture and sigmoid/BCE choice were preserved.

## Goal

Turn the GAN concepts into an inspectable experiment: prepare the data,
verify the update mechanics, run the small MLP baseline, and examine both
sample realism and diversity. Follow [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)
and record actual outcomes in [RUN_LOG.md](RUN_LOG.md).

The learner writes and revises the code. The instructor guides, reviews,
and runs agreed checks. No implementation, fixes, dataset preparation, or
training were performed at this kickoff.

## Work Sequence

1. Clarify the data/evaluation split and what the experiment can establish.
2. Resolve the learner draft's execution blockers, then verify two D/G cycles
   on one synthetic minibatch, including gradient and parameter isolation.
3. Prepare or verify MNIST and its loader before dataset-backed execution.
   Only the training split enters discriminator updates.
4. Fix the run settings and monitoring noise before the measured baseline.
5. Retain the untrained and per-epoch sample panels, both D loss terms, G loss,
   runtime/device information, exact settings, and source identity.
6. Inspect failures and write a limited conclusion supported by the artifacts.

## Opening Research-Practice Question

The planned split is 55,000 training images, 5,000 validation images, and the
official 10,000 test images. The GAN is unconditional and ignores digit labels.

Explain why test images must still stay out of discriminator training even
though their digit labels are unused.

This original prompt was not independently answered; the learner redirected
the lesson toward GAN-specific interpretation and research reasoning rather
than repeating familiar training infrastructure. The later operational run
prepared the dataset and verified the split. Retain the prompt as a reference,
not another closing quiz or an unverified data-preparation task.

## Understanding Assessment - 2026-09-15

| Topic | Learner response | Assessment and guidance |
| --- | --- | --- |
| Changing evaluator | Suggested that rising G loss despite better-looking samples means D is doing a bad job. | Incomplete: D could instead be improving faster than G and detecting subtler flaws. |
| Fixed-input monitoring | Proposed inspecting changes in D's score. | Useful measurement; the instructor clarified that the actual images must stay fixed and that both real and fake groups need checking. |
| Reporting the trajectory | Proposed reporting a curve over time rather than selecting one attractive result. | Useful for transparency; fixed-noise image panels and a predeclared checkpoint/budget were added through guidance. |
| Diversity failure | Called sharper outputs that all become 3s a failure. | Correctly recognized a failed diversity goal; attributing it specifically to D overfitting was not established. |
| Diagnostic hypotheses | Suggested resets or penalties, then described D rejecting other real images while accepting 3s. | These are candidate interventions and a possible digit-bias observation. Overfitting requires a training-versus-held-out comparison under comparable conditions. |

The discussion supports a guided understanding checkpoint. It does not
establish independent mastery of causal diagnosis, nor does it substitute for
running and evaluating the experiment.

## Corrections To Retain

- G loss depends on the current D. Better-looking images can receive lower
  realness scores as D improves, so a rising G loss alone does not diagnose
  either model as bad.
- D rejecting fixed fake images more strongly is insufficient by itself;
  inspect its scores on fixed real images too. A general shift toward calling
  everything fake is different from improved discrimination.
- Report complete trajectories and fixed-noise sample panels. A loss curve
  is not an image-quality measure and does not by itself specify which
  checkpoint should be reported. Declare that rule before inspecting results.
- G producing mostly one digit is mode collapse in its samples. D accepting
  mainly that digit suggests a narrow or biased decision rule. Neither
  observation alone proves D overfitting.
- Evidence consistent with overfitting is good performance on training
  examples and worse performance on comparable held-out examples. Compare
  like-for-like digit groups so differences in class mix do not masquerade
  as a generalization gap.
- Treat resets and diversity penalties as hypotheses for controlled follow-up
  experiments. First verify what data the models actually see and retain the
  original run rather than erasing an inconvenient result.

## Historical Diagnostic Availability At Discussion Closure

The learner asked the instructor to handle the proposed diagnostic. A
read-only availability check found no MNIST cache, no loader, and no recorded
trained-model experiment or sample panels in this project. The actual
digit-conditioned diagnostic therefore could not be performed. Scoring
random synthetic images with an untrained D would not test the hypothesis.

When the prerequisites exist, the diagnostic is to inspect real training-batch
digit coverage and compare D scores by digit on training and held-out real
images, with fixed generated-image references. Digit labels may be used for
this audit without becoming inputs to unconditional GAN training. The
all-3s examples discussed here were hypothetical, not observed project results.

## Practical Baseline And Diagnostic

The actual [results](RESULTS.md) now include real-versus-generated images,
loss/score trajectories, and an audit of D on all training and validation real
images by digit. The supporting [run notes](results/runs/session3_mnist_mlp_seed42_20260915/notes.md) separate observations
from claims: generated samples acquire digit-like structures but remain noisy.
D loss fell to 0.394094 and G loss rose to 3.368496;
these changing-evaluator losses are not quality rankings.

The real-image audit gives mean scores of 0.9098/0.9073 and acceptance rates
of 95.80%/95.70% for training/validation. It does not show D accepting only
3s or a large generalization gap on these measurements. The hypothetical
all-3s collapse was not fabricated as an observed result. No quantitative
coverage or memorization claim follows from the saved sample panels.

## Longer-Training Follow-Up - 2026-09-15

The learner explicitly requested a longer run. The [20-epoch follow-up](results/runs/session3_mnist_mlp_seed42_20epochs_20260915/notes.md)
used the same model, seed, split, and settings, with an exactly matching first
five-epoch state. It completed with final D/G losses 0.525343/2.534042.
[Fixed-noise comparisons](results/runs/session3_mnist_mlp_seed42_20epochs_20260915/epoch_05_10_20_comparison.png) show cleaner
strokes and backgrounds, with a clear visual improvement over epoch 5.
Many outputs remain 1-like, so the improvement in individual-image quality
coexists with a mode-concentration concern.
It does not establish complete digit coverage or remove the one-seed limitation.
The original five-epoch baseline remains intact.

## Final Interpretation And Closure

The learner's hypothesis that five epochs was an early checkpoint is supported
for visual quality in this training trajectory. With architecture, seed,
data split, optimizer settings, and monitoring noise fixed, additional training
produced cleaner backgrounds, sharper and more continuous strokes, and more
recognizable individual digits. The matching first-five-epoch weights,
optimizer/RNG states, metrics, and grids make this a controlled budget comparison.

This improvement should be credited directly. The remaining concentration on
1-like outputs is a separate weakness; it does not erase the improvement in
individual images. No generated-class coverage metric, independent-seed
replication, or convergence claim was established.

| Completed run | Epochs | Paired D/G iterations | Final D loss | Final G loss |
| --- | --- | --- | --- | --- |
| Original baseline | 5 | 2,150 | 0.394094 | 3.368496 |
| Same-settings follow-up | 20 | 8,600 | 0.525343 | 2.534042 |

Both final checkpoints were selected by their declared epoch budgets. All
intermediate panels and checkpoints are retained. The different losses are
training diagnostics, not a ranking of image quality.

The technical verification, measured runs, plots, real-image discriminator
audits, interpretation, and documentation satisfy Session 3's practical goals.
There are no remaining Session 3 execution tasks. Session 4's controlled
architecture comparison and the final Week 8 lab note remain future work.

## Completion Evidence

- [x] Understanding/research-practice discussion concluded with guided corrections.
- [x] Runtime/data-flow blockers repaired for the explicitly requested baseline.
- [x] Eleven checks passed, including isolated updates and consecutive cycles.
- [x] MNIST prepared externally; splits, normalization, and cached loading checked.
- [x] Fixed run configuration and separate monitoring/final noise saved.
- [x] Five-epoch baseline completed with all panels, checkpoints, and loss records.
- [x] Failure observations and measured discriminator audit documented with limits.
- [x] Real-versus-generated comparison created and visually inspected.
- [x] Twenty-epoch follow-up completed with the original baseline preserved.
- [x] Same-noise epoch-5/10/20 comparison and exact initial-state match verified.
- [x] Visual improvement, diversity limitation, and one-seed limits documented.
- [x] Session 3 closed and the Session 4 handoff recorded.

Run evidence: [config](results/runs/session3_mnist_mlp_seed42_20260915/config.json), [metrics](results/runs/session3_mnist_mlp_seed42_20260915/metrics.json),
[stdout](results/runs/session3_mnist_mlp_seed42_20260915/stdout.txt), [verification](results/runs/session3_mnist_mlp_seed42_20260915/verification.txt),
and [comparison](results/runs/session3_mnist_mlp_seed42_20260915/real_vs_generated.png).

Follow-up evidence: [run notes](results/runs/session3_mnist_mlp_seed42_20epochs_20260915/notes.md),
[epoch-5/10/20 comparison](results/runs/session3_mnist_mlp_seed42_20epochs_20260915/epoch_05_10_20_comparison.png),
[metrics](results/runs/session3_mnist_mlp_seed42_20epochs_20260915/metrics.json),
and [verification](results/runs/session3_mnist_mlp_seed42_20epochs_20260915/verification.txt).
The [final closure report](../../.omx/reports/task-2026-09-15-203942.md) records
the documentation checks and preservation of the run evidence.

Next: Session 4's controlled architecture comparison. Normal lessons remain
teacher-guided with learner-owned coding; the operational baseline request
was an explicit exception for completing this experiment.
