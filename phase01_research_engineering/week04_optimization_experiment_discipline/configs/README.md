# Configs

This folder holds Week 4 experiment configs.

## Recorded Configs

- `adamw_cosine_baseline.yaml`: config-driven version of the Week 3 final setup.
- `sgd_momentum.yaml`: optimizer ablation that swaps AdamW for SGD with
  momentum.
- `no_scheduler.yaml`: scheduler ablation that keeps AdamW but disables cosine
  annealing.
- `lr_low.yaml`: learning-rate ablation using `0.0003`.
- `lr_high.yaml`: learning-rate ablation using `0.003`.
- `batch_size_variant.yaml`: batch-size ablation using `128`.
- `seed_repeat_a.yaml` and `seed_repeat_b.yaml`: baseline repeats with
  different seeds for variance estimation.
- `selected_final_seed7.yaml`: selected final AdamW+cosine configuration with
  test evaluation enabled after validation-only ablations identified seed `7`
  as the strongest validation repeat.

## Target Fields

Each config makes the training run explicit:

```yaml
run_name:
model_name:
epochs:
batch_size:
seed:
optimizer:
learning_rate:
weight_decay:
scheduler:
scheduler_min_lr:
data_root:
num_workers:
evaluate_test:
```

SGD configs also include `momentum`.

## Rules

- Change one major variable per ablation when possible.
- Give each config a meaningful `run_name`.
- Keep final test evaluation disabled unless the run is the selected final
  configuration.
- Record the config path in `RUN_LOG.md` when a run actually happens.

## Check Command

```bash
python3 phase01_research_engineering/week04_optimization_experiment_discipline/config_check.py \
  --config phase01_research_engineering/week04_optimization_experiment_discipline/configs/adamw_cosine_baseline.yaml
```
