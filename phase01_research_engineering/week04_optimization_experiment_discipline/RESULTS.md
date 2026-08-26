# Results

Week 4 has measured config-driven CIFAR-10 optimization ablations using the
Week 3 MiniResNet pipeline. The selected final run is
`selected_final_seed7`, which uses AdamW, cosine annealing, batch size 64, and
seed 7.

Final selected result:

- best validation accuracy: 0.8930
- final validation accuracy: 0.8906
- final test accuracy: 0.8781
- final test loss: 0.3967
- test-set evaluation: performed only for the selected final config

## Infrastructure Smoke Checks

| Check | Status | Notes |
| --- | --- | --- |
| Config loading and validation | passed | All Week 4 config files load through `config_check.py`. |
| Optimizer and scheduler construction | passed | Configs construct AdamW+cosine, SGD+cosine, and AdamW without scheduler as intended. |
| Fake-batch training path | passed | `train(config)` runs one fake train/validation step and skips test evaluation when `evaluate_test: false`. |
| Run-record writing | passed | Tests verify `config.yaml`, `metrics.json`, and `notes.md` are created under a temporary run directory. |

## Summary Table

| Run ID | Optimizer | LR | Scheduler | Batch Size | Seed | Best Val Acc | Final Val Acc | Test Acc | Runtime | Notes |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `adamw_cosine_baseline` | AdamW | 0.001 | cosine | 64 | 42 | 0.8874 | 0.8820 | n/a | 389.9s | Validation-only baseline. |
| `sgd_momentum` | SGD momentum | 0.001 | cosine | 64 | 42 | 0.8300 | 0.8290 | n/a | 390.2s | Much weaker at this LR/schedule. |
| `lr_low` | AdamW | 0.0003 | cosine | 64 | 42 | 0.8664 | 0.8654 | n/a | 389.1s | Too conservative for 50 epochs. |
| `lr_high` | AdamW | 0.003 | cosine | 64 | 42 | 0.8868 | 0.8842 | n/a | 392.6s | Competitive but not clearly better than baseline. |
| `no_scheduler` | AdamW | 0.001 | none | 64 | 42 | 0.8748 | 0.8642 | n/a | 390.2s | Weaker and less stable late in training. |
| `batch_size_variant` | AdamW | 0.001 | cosine | 128 | 42 | 0.8742 | 0.8730 | n/a | 362.9s | Faster wall time, lower validation accuracy. |
| `seed_repeat_a` | AdamW | 0.001 | cosine | 64 | 7 | 0.8918 | 0.8888 | n/a | 387.3s | Strongest validation-only repeat. |
| `seed_repeat_b` | AdamW | 0.001 | cosine | 64 | 123 | 0.8770 | 0.8738 | n/a | 386.7s | Shows meaningful seed variance. |
| `selected_final_seed7` | AdamW | 0.001 | cosine | 64 | 7 | 0.8930 | 0.8906 | 0.8781 | 385.9s | Selected final config; reserved test set evaluated. |

## Interpretation

AdamW with cosine annealing remains the best-supported training recipe. The
baseline seed 42 run reached 0.8874 best validation accuracy, and the selected
seed 7 repeat reached 0.8930 best validation accuracy before final test
evaluation.

SGD with momentum underperformed badly at the chosen learning rate and schedule:
0.8300 best validation accuracy versus 0.8874 for the AdamW baseline. This does
not prove SGD is generally bad; it proves this SGD setting was not competitive
without further LR/momentum tuning.

Learning-rate scale mattered. Lowering AdamW from 0.001 to 0.0003 made the run
too slow for the 50-epoch budget. Raising it to 0.003 was competitive but not
clearly better than the baseline, and the final selected seed-7 baseline still
won.

The cosine schedule helped. Disabling it reached only 0.8748 best validation
accuracy and ended lower at 0.8642 final validation accuracy, while the
cosine baseline variants ended stronger.

Batch size 128 reduced runtime by about 27 seconds compared with the batch-size
64 baseline, but validation accuracy fell to 0.8742 best. That is a useful
speed/quality tradeoff, not a free improvement.

Seed variance was large enough to matter. The same AdamW+cosine recipe ranged
from 0.8770 to 0.8930 best validation accuracy across seeds/repeats. Small
single-run differences below that range should not be treated as reliable
algorithmic improvements.

## Reproducibility Notes

- Exact config used for each run is stored in each run directory as
  `config.yaml`.
- Structured metrics are stored as `metrics.json`.
- Notes stubs are stored as `notes.md`.
- Run directories are under `results/runs/YYYYMMDD_HHMMSS_run_name/`.
- Hardware for measured runs: NVIDIA RTX 5000 Ada Generation through
  `/home/rui.gao/anaconda3/bin/python`.
- The Codex sandboxed `python3` could not see CUDA, so real runs were launched
  outside the sandbox with the Anaconda interpreter.
- Test accuracy was checked only for `selected_final_seed7`.

## Limitations

- No best-validation checkpoint is saved yet, so final test accuracy is from
  the final epoch, not the best validation epoch.
- SGD was tested only with one learning rate and momentum value.
- Batch-size comparison did not adjust learning rate for batch size.
- Seed repeats were run only for the AdamW+cosine baseline recipe.
- Memory was observed through `nvidia-smi`, but exact peak GPU memory was not
  captured into each metrics file.
