# Results Directory

This directory is reserved for Week 2 experiment outputs.

Do not commit large artifacts. Keep raw data, processed caches, and large checkpoints outside Git under `AI_ROADMAP_DATA`.

Expected future layout:

```text
results/
  runs/
    YYYYMMDD_HHMM_baseline/
      metrics.json
      notes.md
      plots/
      stdout.log
```

Checkpoints should be saved only if implementation is authorized later, and large checkpoint files must remain out of Git.
