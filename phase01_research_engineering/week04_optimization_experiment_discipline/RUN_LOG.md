# Run Log

Use this file to record commands, settings, and outcomes. Do not record results
until the run has actually happened.

## Documentation And Setup

| Date | Task | Status | Notes |
| --- | --- | --- | --- |
| 2026-08-22 | Week 4 documentation scaffold | complete | Created README, experiment plan, run log, results placeholder, lab-note template, and scaffold docs for configs/tests/results. |
| 2026-08-25 | Config-driven training infrastructure | complete | Added shared config loading/validation, config check CLI, config-driven training entrypoint, optimizer/scheduler builders, fake-batch smoke tests, and run-record writing. |

## Smoke Checks

| Date | Purpose | Command | Status | Notes |
| --- | --- | --- | --- | --- |
| 2026-08-25 | Python syntax check | `python3 -m py_compile phase01_research_engineering/week04_optimization_experiment_discipline/config_utils.py phase01_research_engineering/week04_optimization_experiment_discipline/config_check.py phase01_research_engineering/week04_optimization_experiment_discipline/dataset.py phase01_research_engineering/week04_optimization_experiment_discipline/model.py phase01_research_engineering/week04_optimization_experiment_discipline/train.py phase01_research_engineering/week04_optimization_experiment_discipline/tests/test_config_utils.py phase01_research_engineering/week04_optimization_experiment_discipline/tests/test_training_config.py phase01_research_engineering/week04_optimization_experiment_discipline/tests/test_model_smoke.py` | passed | All Week 4 config/training/test Python files compile. |
| 2026-08-25 | Config checks | `python3 phase01_research_engineering/week04_optimization_experiment_discipline/config_check.py --config <config>` | passed | Passed for all planned Week 4 configs, including the selected final config. |
| 2026-08-25 | Week 4 pytest smoke tests | `python3 -m pytest phase01_research_engineering/week04_optimization_experiment_discipline/tests -q` | passed | 9 tests passed after adding the run-record writing test. |
| 2026-08-25 | CUDA-visible Python check | `/home/rui.gao/anaconda3/bin/python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"` | passed | CUDA was visible through the Anaconda interpreter; GPU was NVIDIA RTX 5000 Ada Generation. |

## Training Runs

| Run ID | Date | Command | Status | Key Result | Notes |
| --- | --- | --- | --- | --- | --- |
| `adamw_cosine_baseline` | 2026-08-25 | `/home/rui.gao/anaconda3/bin/python phase01_research_engineering/week04_optimization_experiment_discipline/run_ablation_suite.py` | passed | Best val 0.8874; final val 0.8820. | Validation-only baseline. |
| `sgd_momentum` | 2026-08-25 | same suite command | passed | Best val 0.8300; final val 0.8290. | SGD at LR 0.001 was much weaker than AdamW. |
| `lr_low` | 2026-08-25 | same suite command | passed | Best val 0.8664; final val 0.8654. | Lower LR was too conservative for 50 epochs. |
| `lr_high` | 2026-08-25 | same suite command | passed | Best val 0.8868; final val 0.8842. | Competitive but not clearly better than baseline. |
| `no_scheduler` | 2026-08-25 | same suite command | passed | Best val 0.8748; final val 0.8642. | Fixed LR was weaker than cosine. |
| `batch_size_variant` | 2026-08-25 | same suite command | passed | Best val 0.8742; final val 0.8730. | Faster runtime but weaker validation result. |
| `seed_repeat_a` | 2026-08-25 | same suite command | passed | Best val 0.8918; final val 0.8888. | Strongest validation-only repeat. |
| `seed_repeat_b` | 2026-08-25 | same suite command | passed | Best val 0.8770; final val 0.8738. | Shows seed variance. |
| `selected_final_seed7` | 2026-08-25 | `/home/rui.gao/anaconda3/bin/python phase01_research_engineering/week04_optimization_experiment_discipline/train.py --config phase01_research_engineering/week04_optimization_experiment_discipline/configs/selected_final_seed7.yaml` | passed | Best val 0.8930; final val 0.8906; test acc 0.8781. | Selected final config with test evaluation enabled. |

## Issues

| Date | Issue | Evidence | Resolution |
| --- | --- | --- | --- |
| 2026-08-25 | Current sandboxed Python cannot see CUDA | `python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"` reported `False` and `0`, with an NVML initialization warning. | Do not launch real 50-epoch CIFAR-10 ablations from this sandbox. Use the existing host/HPC GPU path when running measured Week 4 ablations. |
| 2026-08-25 | First ablation-suite attempt was sandboxed | `nvidia-smi` did not show the run as a CUDA compute process during the initial sandboxed launch. | Interrupted the sandboxed run and relaunched with `/home/rui.gao/anaconda3/bin/python` outside the sandbox. Ignore incomplete directory `results/runs/20260825_222753_adamw_cosine_baseline/`; it has no `metrics.json`. |
