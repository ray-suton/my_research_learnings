# Run Log

Use this file to record commands, settings, and outcomes. Do not record
training results until the run has actually happened.

## Documentation And Setup

| Date | Task | Status | Notes |
| --- | --- | --- | --- |
| 2026-09-04 | Week 7 documentation scaffold | complete | Created README, experiment plan, run log, results placeholder, Session 1 concept note, and scaffold docs for configs/tests/results. |
| 2026-09-04 | Session 1 concept contract | complete | Defined the first VAE target around Fashion-MNIST, `mu`, `logvar`, reparameterization, reconstruction/KL loss accounting, Week 6 comparison anchors, and falsification conditions. |
| 2026-09-07 | Session 1 Socratic wrap-up | complete | Guided discussion clarified deterministic autoencoder versus VAE latents, image-conditioned sampling versus prior sampling, `epsilon`, KL as pressure toward `Normal(0, I)`, label-free unconditional training, and why VAE total loss must be logged alongside separate reconstruction and KL terms. |
| 2026-09-08 | Session 2 completion and Session 3 handoff | complete | Added and ran the eight-test CPU smoke suite, retained its output, reconciled status docs, and prepared `session3.md`. Learner implementation files were preserved. No measured multi-epoch VAE run was performed. |
| 2026-09-08 | Sessions 3–5 technical completion | complete | User authorized all remaining Week 7 work. Extended training/evaluation, added the run sequence and matched AE comparison, completed experiments and notes. Intuitive understanding review remains pending. |
| 2026-09-10 | Basic VAE intuition review | complete | Guided discussion covered prior versus image-conditioned sampling, KL pressure and information loss, fixed-code decoding, reparameterization feedback, and the encoder’s training role. Corrections and assessment limits are recorded in `session5.md`. No code or experiment changes. |

## Smoke Checks

| Date | Purpose | Command | Status | Notes |
| --- | --- | --- | --- | --- |
| 2026-09-08 | Loss scaling and one-batch training integration | `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -u -` (temporary inline diagnostic) | pass | Ran the current training function on eight cached Fashion-MNIST training images, CPU, mean MSE, beta 1, Adam LR 0.001, seed 42. Per-pixel MSE 0.16478354; KL 0.33274001; total 129.52302551. Every parameter had finite nonzero gradients and changed after one step. See `.omx/reports/task-2026-09-08-200510.md` from the repository root. The inline diagnostic is not a saved executable test; a project-local smoke script remains to be written. |
| 2026-09-08 | Saved Session 2 smoke suite | `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 phase01_research_engineering/week07_variational_autoencoders/tests/test_session2_smoke.py` (repository root) | pass: 8 tests | Full output, source hashes, and environment are in `results/session2_smoke_test.txt`. Cached data only, CPU, seed 42, eight-image training batch. Verifies shapes, output range, KL values, gradient scaling, input-loss preservation, training gradients/parameter updates, and prior decoding. This saved command supersedes the temporary diagnostic as the reproducible Session 2 check. |
| 2026-09-08 | Session 2 smoke-code style pass | Same saved script command above | pass: 8 checks | Replaced the unittest class with plain check functions, direct assertions, and `main()`, following earlier learning scripts. All eight groups still pass; refreshed `results/session2_smoke_test.txt`. Learner implementation files are unchanged. |
| 2026-09-08 | Measured-experiment checks | `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 phase01_research_engineering/week07_variational_autoencoders/tests/test_experiment_checks.py` | pass: 5 checks | Unequal-batch weighting, loss identity, mean decoding, evaluation state/RNG preservation, repeatability, and stochastic forward behavior. Output: `results/experiment_checks.txt`. |
| 2026-09-08 | Final artifact audit | Read-only JSON/PNG/source-hash validation | pass | All 11 measured JSONs and 16 PNGs verified; candidate test flags false, validation selection locked before final test, selected checkpoint hash matched. |

## Training Runs

| Run ID | Date | Command | Status | Key Result | Notes |
| --- | --- | --- | --- | --- | --- |
| session3_latent32 | 2026-09-08 | [Recorded command](results/completed_20260908/session3_latent32/command.txt) | complete | Validation sampled MSE 0.025294; KL 7.970769 | Five epochs; no test evaluation. |
| session4_latent8 | 2026-09-08 | [Recorded command](results/completed_20260908/session4_latent8/command.txt) | complete | Validation sampled MSE 0.024347; KL 7.984677 | Selected by validation MSE before test use. |
| session4_latent16 | 2026-09-08 | [Recorded command](results/completed_20260908/session4_latent16/command.txt) | complete | Validation sampled MSE 0.025318; KL 8.129920 | Five epochs; no test evaluation. |
| session4_latent64 | 2026-09-08 | [Recorded command](results/completed_20260908/session4_latent64/command.txt) | complete | Validation sampled MSE 0.026851; KL 7.694735 | Five epochs; no test evaluation. |
| session5_matched_autoencoder_comparison | 2026-09-08 | Command field in [comparison metrics](results/completed_20260908/session5_comparison/session5_comparison_metrics.json) | complete | Test MSE: AE 0.020632, VAE draw 0.024081, VAE decode(mu) 0.020514 | Matched latent size 8; selection already frozen. |

The full sequence ran from the repository root as:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/week07_vae_matplotlib python3 -u phase01_research_engineering/week07_variational_autoencoders/run_sessions.py --output-dir phase01_research_engineering/week07_variational_autoencoders/results/completed_20260908 --checkpoint-dir /tmp/week07_vae_20260908
```

Choose fresh output/checkpoint paths for a rerun. Existing results are protected
from overwrite. Checkpoints under `/tmp` are a temporary writable exception and
can expire. Downloads are disabled. Small stdout logs are present locally but
ignored by the root Git rules.

## Issues

| Date | Issue | Evidence | Resolution |
| --- | --- | --- | --- |
|  |  |  |  |
