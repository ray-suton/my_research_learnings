# Results

Keep only small summaries in Git.

Large checkpoints, datasets, caches, and bulky stdout should stay outside Git
under the roadmap data/artifact policy.

## Expected Run Layout

Each meaningful run should have a folder like:

```text
results/runs/YYYYMMDD_HHMM_run_name/
  config.yaml
  metrics.json
  notes.md
```

`stdout.log` is optional; the completed Week 4 run records currently preserve
the config, structured metrics, and notes stub. Add plots only when they are
small and useful for comparison.
