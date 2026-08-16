# Completed Basic-ML Reference Archive

`completed_reference_2026_07_15/` contains the prior completed bridge as an
untouched nested `basic_ml_mini_projects` package. It includes all source,
tests, notes, metrics, and figures that previously lived at the learning
workspace root.

Keep this archive read-only while rebuilding the material. Its purpose is
comparison and verification after an independent attempt, not solution
discovery before one.

## Verify the archived reference

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=basic_ml_mini_projects/archive/completed_reference_2026_07_15 \
.venv/bin/python -m unittest \
  basic_ml_mini_projects.test_day01_supervised_basic \
  basic_ml_mini_projects.test_day02_clustering_em \
  basic_ml_mini_projects.test_day03_margins_trees -v
```

The expected result is 26 passing tests.
