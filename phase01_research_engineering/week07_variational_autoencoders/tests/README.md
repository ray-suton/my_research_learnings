# Tests

`test_session2_smoke.py` is a plain Python script with eight check functions,
simple assertions, and a `main()` that runs them in order. It follows the direct
function style of the earlier learning scripts. It uses the existing PyTorch
and torchvision dependencies; no test-framework dependency is required.

## Run From The Repository Root

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 phase01_research_engineering/week07_variational_autoencoders/tests/test_session2_smoke.py
```

The suite reads the existing Fashion-MNIST cache through `loader.py`, currently
under `/home/rui.gao/data/ai_research_roadmap/raw/torchvision`. It passes
`download=False`: missing data fails the suite rather than downloading or
silently skipping the data check. See the root `DATA_PLAN.md` for preparation.

Use the script command above, rather than unittest discovery. Run without `-O`:
the script rejects optimized Python execution so its assertions cannot be
silently disabled. A successful run ends with `8 smoke checks passed`.

## Eight Checks

- Cached dataset sizes, train/validation separation, and normalized batch shape.
- Flattened, hidden, `mu`, `logvar`, sampled latent, decoder, and reconstruction
  shapes, including batch size one and output range.
- Gradient paths through both reparameterization inputs.
- Zero KL for a standard-normal posterior and finite scalar loss terms.
- Known KL values with latent summation, batch averaging, and non-unit variance.
- Preservation of the input MSE and correct reconstruction/KL gradient scaling.
- The actual training function updating parameters on one real eight-image
  batch, with gradients matching an independently constructed objective. This
  catches clearing gradients before the step and missing/double pixel scaling.
- Prior-sampled decoding without an input image.

## Recorded Result And Limits

All **8 tests passed on 2026-09-08**. Full output, environment versions, command,
and source hashes are saved in `../results/session2_smoke_test.txt`.

These checks establish model and training mechanics on CPU. They do not measure
convergence, validation quality, or recognizable generation. The current loss
contract assumes mean-reduced MSE and single-channel 28-by-28 images.

## Measured-Experiment Checks

`test_experiment_checks.py` adds five plain-function checks on tiny synthetic
data. They check image-weighted metrics with an unequal final batch, loss
accounting, separate mean-code decoding, evaluation preserving model state and
training RNG, fixed-seed repeatability, and sampling in evaluation mode.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 phase01_research_engineering/week07_variational_autoencoders/tests/test_experiment_checks.py
```

All five checks pass; output is in `../results/experiment_checks.txt`.
Together, the scripts cover 13 checks. Measured Session 3–5 results are stored
separately under `../results/completed_20260908/`.
