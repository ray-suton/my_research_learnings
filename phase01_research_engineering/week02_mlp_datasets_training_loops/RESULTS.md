# Results

The three planned Week 2 Fashion-MNIST MLP runs are recorded as completed: the baseline, the learning-rate variant, and the hidden-size variant.

The baseline epoch stdout is recorded below. The learning-rate and hidden-size variant summaries were added from user-reported completed runs. Three loss plots now exist; the two variant plots preserve five-epoch train/validation curves and test loss, but their raw stdout, underlying numeric metric files, exact plot-generation commands, checkpoints, and populated per-run directories were not retained. The recorded test accuracy values therefore remain summary evidence rather than independently reproducible outputs. Do not reuse the test set for further tuning.

## Summary Table

| Run ID | Hidden Size | Learning Rate | Epochs | Validation Accuracy | Test Accuracy | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 128 | 0.001 | 5 | 0.8722 | 0.8644 | Completed; epoch stdout recorded below. |
| lr_variant | 128 | 0.003 | 5 | 0.8608 | 0.8548 | Completed; final metrics recorded, full epoch stdout not captured. |
| hidden_size_variant | 256 | 0.001 | 5 | 0.8822 | 0.8734 | Completed; final metrics recorded, full epoch stdout not captured. |

## Run Details

### Baseline

```bash
python3 phase01_research_engineering/week02_mlp_datasets_training_loops/train.py --epochs 5 --learning-rate 0.001 --hidden-size 128 --batch-size 64
```

```text
epoch 1: train_loss=0.4850 val_loss=0.4126 val_acc = 0.8485
epoch 2: train_loss=0.3650 val_loss=0.3696 val_acc = 0.8653
epoch 3: train_loss=0.3291 val_loss=0.3473 val_acc = 0.8720
epoch 4: train_loss=0.3037 val_loss=0.3397 val_acc = 0.8770
epoch 5: train_loss=0.2845 val_loss=0.3405 val_acc = 0.8722
```

Final recorded test accuracy: `0.8644`.

Baseline loss plot: `results/baseline_losses.png`. The plot was generated from the recorded baseline losses without retraining.

### Learning-Rate Variant

```bash
python3 phase01_research_engineering/week02_mlp_datasets_training_loops/train.py --epochs 5 --learning-rate 0.003 --hidden-size 128 --batch-size 64
```

Final recorded validation accuracy: `0.8608`.
Final recorded test accuracy: `0.8548`.

Full epoch stdout was not captured in this file.

Learning-rate-variant loss plot: `results/rl_variant_losses.png`. The plot preserves five train/validation epochs and test loss `0.4073`; its raw numeric series, stdout, and generation command were not captured.

### Hidden-Size Variant

```bash
python3 phase01_research_engineering/week02_mlp_datasets_training_loops/train.py --epochs 5 --learning-rate 0.001 --hidden-size 256 --batch-size 64
```

Final recorded validation accuracy: `0.8822`.
Final recorded test accuracy: `0.8734`.

Full epoch stdout was not captured in this file.

Hidden-size-variant loss plot: `results/hs_variant_losses.png`. The plot preserves five train/validation epochs and test loss `0.3524`; its raw numeric series, stdout, and generation command were not captured.

## Recorded Smoke Evidence

| Check | Status | Notes |
| --- | --- | --- |
| Dataloader/model/loss one-batch smoke test | recorded passed on 2026-07-06 | `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. pytest -q -p no:cacheprovider tests/test_smoke.py` passed from the Week 2 folder. |
| Training loop smoke run | recorded passed on 2026-07-06 | `PYTHONDONTWRITEBYTECODE=1 python train.py --epochs 1 --max-train-batches 2 --max-val-batches 2 --batch-size 64`; output: `train_loss=2.1816`, `val_loss=1.7975`, `val_acc=0.4844`. |
| Model-only forward recheck | passed on 2026-07-13 | `PYTHONDONTWRITEBYTECODE=1 python3 model.py` produced logits with shape `(64, 10)`. |
| Dataset-backed smoke recheck | blocked on 2026-07-13 | Both the fallback home root and documented scratch root were absent. The test stopped during dataset construction with downloads disabled, before model construction. |
| Final test evaluation | summary metrics recorded; raw command/output not captured | Test accuracy values are recorded in the summary table, but the exact evaluation command/output is unavailable for independent reproduction. |

## Interpretation

The baseline learned steadily over five epochs, with validation accuracy rising from `0.8485` after epoch 1 to `0.8722` after epoch 5. Validation loss improved through epoch 4 and then ticked up slightly at epoch 5, so the baseline may already be near the useful limit for this simple configuration.

The higher learning-rate variant (`0.003`) underperformed the baseline on both validation and test accuracy. For this setup, increasing the learning rate did not help.

The hidden-size variant (`256`) performed best on both validation and test accuracy. The improvement over the baseline is modest but consistent in the recorded summary metrics, so the larger hidden layer is the strongest of the three tested configurations.

The main modeling lesson is that a plain MLP reaches reasonable recorded Fashion-MNIST performance, but architecture matters. After the compact pure-PyTorch basic-ML bridge, moving to CNNs is justified because Fashion-MNIST has spatial structure that the MLP ignores.

## Limitations

- Full epoch stdout is missing for the learning-rate and hidden-size variants.
- The exact final test-evaluation command/output was not captured.
- Loss plots exist at `results/baseline_losses.png`, `results/rl_variant_losses.png`, and `results/hs_variant_losses.png`. The baseline plot was generated from recorded values; the exact commands, raw stdout, and structured numeric inputs for the variant plots were not retained.
- No checkpoint artifacts are recorded under `results/runs/` yet.
- The external Fashion-MNIST cache is not visible in the current 2026-07-13 environment, so the dataset-backed smoke test cannot presently be rerun without restoring or re-preparing the external data.
- Because test accuracies have now been recorded for all three runs, avoid further tuning against the test set.
