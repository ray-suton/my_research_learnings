# Tests

This folder holds Week 4 smoke tests.

## Implemented Tests

- config-loading test
- optimizer and scheduler construction test
- model forward-pass shape test
- tiny fake-batch training run that finishes quickly
- run-record writing test using a temporary directory

Current test files:

- `test_config_utils.py`
- `test_model_smoke.py`
- `test_training_config.py`

## Test Intent

These are not accuracy tests. They are trust tests:

- Does the data path return tensors with expected shapes?
- Does the selected model produce logits with shape `(batch, num_classes)`?
- Does the config parser load required fields?
- Can the training loop run end to end on a tiny fake or bounded dataset?
- Does a recorded run create `config.yaml`, `metrics.json`, and `notes.md`?

## Command

```bash
python3 -m pytest phase01_research_engineering/week04_optimization_experiment_discipline/tests -q
```
