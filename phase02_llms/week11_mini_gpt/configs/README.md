# Configuration Record

No executable configuration exists yet. Use
[EXPERIMENT_PLAN.md](../EXPERIMENT_PLAN.md) as the proposed contract and record
resolved settings beside each actual run before reporting measurements.

Required fields per run:

- Run ID, date, Python and PyTorch versions, hardware, dtype, and seed.
- Model: `vocab`, `T`, `d_model`, `n_heads`, `n_layers`, `d_ff`, dropout,
  positional-embedding type, weight-tying on/off, final-norm on/off.
- Initialization scheme and the residual scaling factor.
- Optimizer: name, learning rate, betas, eps, weight decay, schedule, warmup,
  gradient clip.
- Data: split paths, window counts, `B`, steps, and the tokenizer artifact.
- Checkpoint interval and destination path under `AI_ROADMAP_DATA`.
- Exact command, exit status, and the failing check when one fails.

One frozen config file is expected once Session 1 closes; avoid maintaining a
second independent copy of the same defaults.
