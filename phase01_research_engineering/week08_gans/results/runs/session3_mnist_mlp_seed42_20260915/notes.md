# Session 3 MNIST MLP-GAN Baseline

Status: completed on 2026-09-15. Run: session3_mnist_mlp_seed42_20260915.
The learner explicitly requested that the assistant complete the practical
baseline, including the supporting repairs, data preparation, execution,
figures, and records. This follows the earlier guided discussion and is a
specific execution request, not a change to the normal teaching preference.

## Experiment

- MNIST: 55,000 training images, 5,000 validation images, official 10,000 test
  images reserved. Only training images entered optimizer updates. Digit
  labels were unused in objectives and used only for the real-image audit.
- Five epochs, batch size 128, seed 42, standard-normal noise width 64.
- Learner MLP architecture preserved: G 64/128/256/784 with ReLU and tanh;
  D 784/256/128/1 with LeakyReLU slope 0.01 and sigmoid.
- Separate Adam optimizers, learning rate 0.0002, betas (0.5, 0.999).
- Mean BCE on probabilities; D sums real/fake terms. One D update followed
  by one G update, reusing G's image with detach only in the D phase.
- CPU, two PyTorch threads, deterministic algorithms enabled. CUDA was not
  available in this session. 25.523 seconds of training;
  31.290 seconds for the recorded training/artifact/audit
  phase. Peak process RSS: 758.3 MiB.
- 430 minibatches per epoch, including the final 88-image batch: 2,150 paired
  D/G iterations total. Optimizers were reused across all epochs.

## Reproduction And Provenance

See [config.json](config.json) for the exact command, versions, source hashes,
parameter counts, split-index hashes, and settings. Executed source snapshots
are retained in sources/. [stdout.txt](stdout.txt) records the actual run output;
[verification.txt](verification.txt) records checks and artifact validation.

Run from the repository root with a fresh output directory:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/week08-mpl-run OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -u phase01_research_engineering/week08_gans/run_baseline.py --device cpu --epochs 5 --batch-size 128 --seed 42 --threads 2 --out-dir phase01_research_engineering/week08_gans/results/runs/session3_mnist_mlp_seed42_repeat
```

The loader reads cached data without downloading. The measured run's data
preparation manifest is:
/home/rui.gao/data/ai_research_roadmap/manifests/prepare_standard_datasets_20260915T093506Z.json

The selected checkpoint is the predeclared final epoch, not the lowest-loss
or nicest-looking checkpoint. Its path and SHA256, and those of all earlier
epochs, are in [metrics.json](metrics.json). Checkpoints are outside Git at:
/home/rui.gao/data/ai_research_roadmap/checkpoints/week08_gans/session3_mnist_mlp_seed42_20260915

## Observations

The untrained generator produced weak random texture. Epoch 1 outputs show
blurred loop/stroke patterns. By epoch 5, several outputs have recognizable
digit-like structures, with cleaner backgrounds in places, but substantial
speckle, malformed strokes, and ambiguous shapes remain. The generated images
are visibly less clean than real MNIST. The panels contain varied structures;
no claim of all-ten-digit coverage, absence of collapse, or memorization safety
is established by this visual check.

G loss rose from 1.054339 to 3.368496 while D loss fell from 0.939656 to
0.394094. D's epoch-average probability for generated images after its update
fell from 0.369538 to 0.068469. These describe a changing discriminator and
are not independent image-quality scores. Scores on the unchanged initial
fake panel also fell (0.262415 after epoch 1 to 0.006974 after epoch 5).

All fixed-noise grids (seed 123) from epochs 0 through 5 were kept. A separate
64-sample final panel uses seed 456. [real_vs_generated.png](real_vs_generated.png)
shows the complete final panel next to fixed validation examples.
[real_vs_generated_single.png](real_vs_generated_single.png) uses index 0 of
both declared panels. The real image is digit 1; no generated digit label is
asserted. These are independent images, not reconstruction pairs.

## Real-Image Discriminator Audit

The [per-digit CSV](discriminator_by_digit.csv) evaluates all 55,000 training
and 5,000 validation real images with the final D. All ten digit groups are
represented. Mean real probabilities are 0.9098 (train) and 0.9073 (validation).
At threshold 0.5, real acceptance is 95.80% and 95.70%, respectively. These are
real-image acceptance rates, not full binary-classification accuracy.

Validation mean scores range from 0.8504 for digit 5 to 0.9818 for digit 0.
The results do not show D accepting only 3s or a large train/validation gap
on this audit. They do not rule out every form of overfitting or evaluate
coverage of generated digits. Test images were not used in the measured run's
training, scoring, or checkpoint selection.

## Limits And Next Step

This is one short, one-seed baseline. Sample quality remains limited; no
FID, classifier-based coverage, or repeated-seed confidence interval was
computed. The completed experiment demonstrates the pipeline and records an
honest imperfect result. It does not show convergence or superiority over a
VAE, especially since Week 7 used Fashion-MNIST.

Use this retained baseline for the Session 4 controlled comparison, keeping
data, update budget, and evaluation panels comparable. Do not erase this run
or silently extend its budget to replace the outcome.
