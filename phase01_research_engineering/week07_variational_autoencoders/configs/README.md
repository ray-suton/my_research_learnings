# Configs

Measured configurations are saved beside their outputs as JSON:

```text
../results/completed_20260908/session3_latent32/config.json
../results/completed_20260908/session4_latent8/config.json
../results/completed_20260908/session4_latent16/config.json
../results/completed_20260908/session4_latent64/config.json
```

The final AE comparison settings are embedded in
`../results/completed_20260908/session5_comparison/session5_comparison_metrics.json`.

The shared first-run settings are five epochs, hidden size 128, batch 128,
Adam LR 0.001, beta 1, seed 42, validation seed 123, and one CPU thread.
Latent size is the Session 4 ablation variable. Each run also records data
paths, split sizes, loss reduction, sampling policy, and environment versions.

Use `train.py --help` for individual runs or the root-relative command in
`../README.md` for the full fixed sequence. Choose new output and checkpoint
paths for a rerun; existing result files are protected from overwrite.
