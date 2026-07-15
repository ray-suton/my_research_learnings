# Run Log

Use this file to record commands, settings, and outcomes. Do not record results here until the run has actually happened.

## Dataset Preparation

| Date | Command | Data Root | Status | Notes |
| --- | --- | --- | --- | --- |
| 2026-07-02 | `python scripts/data/prepare_standard_datasets.py --dataset fashion_mnist --dry-run` | `/home/rui.gao/data/ai_research_roadmap` | passed | Dry run created a manifest. |
| 2026-07-02 | `python scripts/data/prepare_standard_datasets.py --dataset fashion_mnist --dry-run` | `/scratch/rui.gao/ai_research_roadmap_data` | passed | Dry run target was scratch storage. |
| 2026-07-02 | `python scripts/data/prepare_standard_datasets.py --dataset fashion_mnist` | `/scratch/rui.gao/ai_research_roadmap_data` | user-reported passed | Exact command output was not pasted into this file. |

## Training Runs

| Run ID | Date | Purpose | Command | Status | Key Result | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 2026-07-06 | Baseline MLP | `python3 phase01_research_engineering/week02_mlp_datasets_training_loops/train.py --epochs 5 --learning-rate 0.001 --hidden-size 128 --batch-size 64` | passed | `val_acc=0.8722`, `test_acc=0.8644` | Epoch stdout is recorded in `RESULTS.md`. Exact test-evaluation command was not captured in this log. Plot: `results/baseline_losses.png`, generated later from the recorded loss values. |
| lr_variant | 2026-07-06 | Learning-rate comparison | `python3 phase01_research_engineering/week02_mlp_datasets_training_loops/train.py --epochs 5 --learning-rate 0.003 --hidden-size 128 --batch-size 64` | passed | `val_acc=0.8608`, `test_acc=0.8548` | User completed the run and recorded final metrics in `RESULTS.md`; full epoch stdout and exact test-evaluation command were not captured. Plot: `results/rl_variant_losses.png`, preserving five-epoch curves and `test_loss=0.4073`; its generation command was not captured. |
| hidden_size_variant | 2026-07-06 | Hidden-size comparison | `python3 phase01_research_engineering/week02_mlp_datasets_training_loops/train.py --epochs 5 --learning-rate 0.001 --hidden-size 256 --batch-size 64` | passed | `val_acc=0.8822`, `test_acc=0.8734` | User completed the run and recorded final metrics in `RESULTS.md`; full epoch stdout and exact test-evaluation command were not captured. Plot: `results/hs_variant_losses.png`, preserving five-epoch curves and `test_loss=0.3524`; its generation command was not captured. |

## Code and Smoke Checks

| Date | Purpose | Command | Status | Key Result | Notes |
| --- | --- | --- | --- | --- | --- |
| 2026-07-05 | One-batch dataloader/model/loss smoke test | `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. pytest -q -p no:cacheprovider phase01_research_engineering/week02_mlp_datasets_training_loops/tests/test_smoke.py` | passed | `1 passed` | Requires repo root on `PYTHONPATH` in the current layout. |
| 2026-07-05 | Bare pytest import check | `pytest --collect-only -vv phase01_research_engineering/week02_mlp_datasets_training_loops/tests/test_smoke.py` | failed | `ModuleNotFoundError: No module named 'phase01_research_engineering'` | Import-path issue, not missing dataset/model files. |
| 2026-07-06 | One-batch dataloader/model/loss smoke test from Week 2 folder | `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. pytest -q -p no:cacheprovider tests/test_smoke.py` | passed | `1 passed in 2.13s` | Current test imports local `dataset.py` and `model.py`; run from the Week 2 folder with `PYTHONPATH=.`. |
| 2026-07-06 | Tiny limited-batch training smoke run | `PYTHONDONTWRITEBYTECODE=1 python train.py --epochs 1 --max-train-batches 2 --max-val-batches 2 --batch-size 64` | passed | `train_loss=2.1816`, `val_loss=1.7975`, `val_acc=0.4844` | Smoke check only; not a baseline run. |
| 2026-07-13 | Model-only forward recheck | `PYTHONDONTWRITEBYTECODE=1 python3 model.py` | passed | `logits_shape=(64, 10)` | Does not require Fashion-MNIST. |
| 2026-07-13 | Dataset-backed smoke recheck with default root | `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. pytest -q -p no:cacheprovider tests/test_smoke.py` | blocked | `RuntimeError: Dataset not found` | `AI_ROADMAP_DATA` was unset, so the loader used the absent fallback `/home/rui.gao/data/ai_research_roadmap`. The test stopped before model construction. |
| 2026-07-13 | Dataset-backed smoke recheck with documented scratch root | `AI_ROADMAP_DATA=/scratch/rui.gao/ai_research_roadmap_data PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. pytest -q -p no:cacheprovider tests/test_smoke.py` | blocked | `RuntimeError: Dataset not found` | `/scratch/rui.gao/ai_research_roadmap_data` was absent in the current session. Downloads remained disabled; the test stopped before model construction. |

## Issues

Record failed commands, environment problems, or confusing results here.

| Date | Issue | Evidence | Resolution |
| --- | --- | --- | --- |
| 2026-07-05 | Bare `pytest` cannot import repo-root package | `pytest --collect-only -vv phase01_research_engineering/week02_mlp_datasets_training_loops/tests/test_smoke.py` failed with `ModuleNotFoundError: No module named 'phase01_research_engineering'`. | Use `PYTHONPATH=.` or `python -m pytest` from the repo root until pytest/package configuration is added. |
| 2026-07-05 | `train.py` was a partial draft, not a verified training loop | Earlier draft referenced undefined names and optimizer mistakes. | Resolved for Tuesday smoke scope on 2026-07-06: limited-batch training smoke run passed. Full baseline and planned comparison runs were later completed and summarized in `RESULTS.md`. |
| 2026-07-13 | External Fashion-MNIST cache is not visible | Both the default home data root and documented scratch data root were absent; cache-disabled smoke tests failed during dataset construction, while a model-only forward passed. | Environment/data-availability issue, not a demonstrated code regression. Restore or re-prepare the external cache and set `AI_ROADMAP_DATA` only if a Week 2 rerun is needed. |
