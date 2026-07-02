# Project Description

This workspace contains a Codex-ready AI research learning roadmap for building practical AI research skill over a 52-week schedule. It starts from a student who has completed the Machine Learning and Deep Learning Specializations and is preparing to become a stronger incoming AI student researcher.

The roadmap source of truth is `ai_research_roadmap_codex.md`. Phase-specific roadmap files under `phaseXX_*` mirror the relevant sections of the root roadmap.

The current repo-work stage is ready to begin Week 2: MLPs, datasets, and training loops. Week 1 bootstrap is complete from a learning-work perspective:

- repository scaffold and environment files exist
- local PyTorch smoke test exists
- SLURM GPU smoke-test script exists
- GPU smoke test passed on an NVIDIA RTX 5000 Ada Generation through SLURM
- autograd sanity test exists and passes
- tiny linear-regression training script exists and passes
- Week 1 training log, loss plot, and prediction-vs-actual plot exist
- Week 1 project README exists
- Week 1 note records the verified results and learning notes

The active branch is `main`. The worktree contains local Week 1 and Week 2 preparation files that may still need a human Git review before staging or committing; repository Git history is intentionally read-only for agents.

Data planning for Week 2 is present:

- `DATA_PLAN.md` describes the external data-root policy
- `data_registry/datasets.yaml` records dataset cards
- `scripts/data/validate_data_registry.py` validates the registry
- `scripts/data/prepare_standard_datasets.py` supports standard dataset preparation

Downloaded datasets, processed caches, checkpoints, and large generated artifacts should stay outside Git under `AI_ROADMAP_DATA`.
