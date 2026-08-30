# Results

Keep only small summaries and generated samples in Git.

Large checkpoints, bulky logs, and large corpora should stay outside Git under
the roadmap data/artifact policy.

## Expected Run Layout

Each meaningful run should have a folder like:

```text
results/runs/YYYYMMDD_HHMM_run_name/
  config.yaml
  metrics.json
  samples.txt
  notes.md
```

## Recorded Runs

| Run Directory | Status | Notes |
| --- | --- | --- |
| `results/runs/20260830_170359_lstm_readme_tiny/` | selected | Current README-corpus LSTM run after final documentation updates. |
| `results/runs/20260830_170204_lstm_readme_tiny/` | superseded | Earlier measured README-corpus run before README documentation updates changed the toy corpus text. |
