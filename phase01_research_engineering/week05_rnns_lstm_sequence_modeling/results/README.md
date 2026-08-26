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
