# Run Log

Use this file to record commands, settings, and outcomes. Do not record results
until the run has actually happened.

## Documentation And Setup

| Date | Task | Status | Notes |
| --- | --- | --- | --- |
| 2026-08-31 | Session 1 experiment contract | complete | Chose Fashion-MNIST, `(1, 28, 28)` inputs, flattened dimension 784, latent dimension 32, MSE reconstruction loss, validation reconstruction MSE, average-image baseline, and original-versus-reconstruction visual output. |
| 2026-09-01 | Session 2 plain autoencoder learning path | complete | Built Fashion-MNIST loading, a fully connected `784 -> 128 -> 32 -> 128 -> 784` autoencoder, MSE training, and a manual original-versus-reconstruction visual check. |
| 2026-09-02 | Session 3 measured-run path | complete | Added seed-controlled train/validation/test dataloaders, CLI defaults, average-image baseline, final test-loss printout, and project-local reconstruction-grid output. |
| 2026-09-03 | Session 4 denoising path | complete | Added CLI-controlled Gaussian noise, noisy-input baseline measurement, metrics JSON output, and a three-row original/noisy/denoised reconstruction grid. |
| 2026-09-04 | Session 5 lab note and carry-forward | complete | Wrote `LAB_NOTE.md`, summarized the latent bottleneck conservatively, compared supervised learning with reconstruction-based representation learning, recorded failure modes, and carried the key cautions into Week 7 VAEs. |

## Checks

| Date | Purpose | Command | Status | Notes |
| --- | --- | --- | --- | --- |
| 2026-09-02 | Python syntax check | `python3 -m py_compile phase01_research_engineering/week06_autoencoders_representation_learning/dataloader.py phase01_research_engineering/week06_autoencoders_representation_learning/model.py phase01_research_engineering/week06_autoencoders_representation_learning/train.py` | passed | Dataloader, model, and training script compile. |
| 2026-09-02 | Split and batch check | `python3 -c "import sys; sys.path.insert(0, 'phase01_research_engineering/week06_autoencoders_representation_learning'); from dataloader import get_datasets, get_dataloaders; datasets = get_datasets(download=False, seed=42); loaders = get_dataloaders(batch_size=128, download=False, seed=42); print('datasets', [len(item) for item in datasets]); print('loaders', [len(loader.dataset) for loader in loaders]); xb, yb = next(iter(loaders[1])); print('val_batch', tuple(xb.shape), tuple(yb.shape))"` | passed | Dataset and loader lengths were `[55000, 5000, 10000]`; validation batch shape was `(128, 1, 28, 28)`. |
| 2026-09-03 | Python syntax check after Session 4 CLI changes | `python3 -m py_compile phase01_research_engineering/week06_autoencoders_representation_learning/dataloader.py phase01_research_engineering/week06_autoencoders_representation_learning/model.py phase01_research_engineering/week06_autoencoders_representation_learning/train.py` | passed | Dataloader, model, and training script compile. |
| 2026-09-03 | Dataloader split and validation batch check | `python3 -c "import sys; sys.path.insert(0, 'phase01_research_engineering/week06_autoencoders_representation_learning'); from dataloader import get_dataloaders; loaders = get_dataloaders(batch_size=128, download=False, seed=42); print([len(loader.dataset) for loader in loaders]); xb, yb = next(iter(loaders[1])); print(tuple(xb.shape), tuple(yb.shape))"` | passed | Loader lengths were `[55000, 5000, 10000]`; validation batch shape was `(128, 1, 28, 28)`. |
| 2026-09-03 | Session 4 metrics JSON validation | `python3 -m json.tool phase01_research_engineering/week06_autoencoders_representation_learning/results/session4_denoising_metrics.json` | passed | Metrics JSON parses and records `noise_std: 0.3`, final validation MSE 0.018124, test MSE 0.018131, noisy-input validation MSE 0.052527, and runtime 21.559s. |
| 2026-09-03 | Session 4 visual artifact check | `file phase01_research_engineering/week06_autoencoders_representation_learning/results/session4_denoising_reconstructions.png` | passed | Output is a PNG image, 1200 x 480. Manual visual inspection showed original/noisy/denoised rows with noise reduced in the denoised outputs. |
| 2026-09-03 | Session 3 versus Session 4 side-by-side metrics graph | `MPLCONFIGDIR=/tmp/matplotlib-codex python3 -c "...read session3/session4 metrics JSON and save comparison PNG..."` | passed | Saved `results/session3_vs_session4_metrics.png` with train/validation curves and final loss/baseline bars. |
| 2026-09-03 | Two-real-image latent interpolation probe | `MPLCONFIGDIR=/tmp/matplotlib-codex python3 -c "...train plain AE, encode two test images, decode latent interpolation path..."` | passed | Saved `results/latent_interpolation_two_real_images.png`; visual inspection shows an ankle boot-to-trouser interpolation plus pixel and reconstruction difference maps. |

## Training Runs

| Run ID | Date | Command | Status | Key Result | Notes |
| --- | --- | --- | --- | --- | --- |
| `session3_plain_autoencoder` | 2026-09-02 | `MPLCONFIGDIR=/tmp/matplotlib-codex python3 phase01_research_engineering/week06_autoencoders_representation_learning/train.py --epochs 5 --input 784 --hidden 128 --latent 32 --batch 128 --lr 0.001 --seed 42` | passed | Final validation MSE 0.014738; test MSE 0.014750; average-image validation MSE 0.088029. | CPU runtime was 20.764s. Reconstruction grid: `results/session3_reconstructions.png`. Metrics record: `results/session3_metrics.json`. |
| `session4_denoising_autoencoder` | 2026-09-03 | `MPLCONFIGDIR=/tmp/matplotlib-codex python3 phase01_research_engineering/week06_autoencoders_representation_learning/train.py --epochs 5 --input 784 --hidden 128 --latent 32 --batch 128 --lr 0.001 --seed 42 --noise-std 0.3` | passed | Final validation MSE 0.018124; test MSE 0.018131; noisy-input validation MSE 0.052527; average-image validation MSE 0.088029. | CPU runtime was 21.559s. Reconstruction grid: `results/session4_denoising_reconstructions.png`. Metrics record: `results/session4_denoising_metrics.json`. |

## Issues

| Date | Issue | Evidence | Resolution |
| --- | --- | --- | --- |
| 2026-09-02 | Earlier Session 2 exploratory root-level plot was not a clean final figure. | Root `Compare and Contrast.png` had overlapping axes/titles/images during inspection. | Session 3 now saves a clean project-local grid to `results/session3_reconstructions.png`. |
