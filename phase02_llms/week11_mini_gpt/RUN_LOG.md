# Run Log

## Initialization - 2026-10-05

Documentation scaffold. No model code exists, no training ran, and nothing was
measured. This entry records the initialization only.

What was created: [README.md](README.md) with the five-session plan and the
decision table to freeze, [session1.md](session1.md) as the conceptual starter,
[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) with the proposed model, loss, and
optimizer contract, this log, [RESULTS.md](RESULTS.md), and the
`configs/`, `tests/`, and `results/` guides.

What was **not** done: no `mini_gpt.py`, no training loop, no overfit run, no
HPC job, no loss curve, no samples. Every number in `EXPERIMENT_PLAN.md` is a
proposed default or an arithmetic prediction.

Inputs confirmed available before Session 2:

- Week 9 tokenizer artifact (vocab 2048, PAD 0 / EOS 1) and the per-split
  streams, 9/9 fixture checks green on 2026-10-05.
- Week 10 `TransformerBlock`, 15/15 + 9/9 + 11/11 fixture checks green, re-run
  2026-10-05.
- Prepared LM dataset: train 28,767 windows, validation 3,011, test 3,409.

## Future Run Record

For each actual run, retain:

1. Date, purpose, exact command, working directory, and exit status.
2. Seed, model sizes, `T`, `B`, dtype, dropout, and optimizer settings.
3. Hardware, wall time, steps/second, tokens/second, and peak memory.
4. Step-0 loss, final train loss, validation loss/perplexity, and the loss curve.
5. Checkpoint path and reload settings.
6. Generated samples with the exact decoding rule and prompt.
7. Failures and interpretation, without overwriting earlier results.

Add commands only once their scripts exist. Label proposals as unexecuted.
