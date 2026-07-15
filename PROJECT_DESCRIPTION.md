# Project Description

This workspace contains a Codex-ready AI research learning roadmap for building practical AI research skill over a 52-week schedule. It starts from a student who has completed the Machine Learning and Deep Learning Specializations and is preparing to become a stronger incoming AI student researcher.

The roadmap source of truth is `ai_research_roadmap_codex.md`. Phase-specific roadmap files under `phaseXX_*` mirror the relevant sections of the root roadmap.

The planned Week 2 three-run MLP comparison and the complete three-day pure-PyTorch basic-ML mini-project bridge are finished from a learning-work perspective. The roadmap is ready to move into Week 3 CNNs. Week 1 bootstrap is complete from a learning-work perspective:

- repository scaffold and environment files exist
- local PyTorch smoke test exists
- SLURM GPU smoke-test script exists
- GPU smoke test passed on an NVIDIA RTX 5000 Ada Generation through SLURM
- autograd sanity test exists and passes
- tiny linear-regression training script exists and passes
- Week 1 training log, loss plot, and prediction-vs-actual plot exist
- Week 1 project README exists
- Week 1 note records the verified results and learning notes

The active branch is `main`. The worktree contains local Week 2 project files, basic-ML bridge plans, and roadmap updates that may still need a human Git review before staging or committing; repository Git history is intentionally read-only for agents.

Data planning and run comparison work for Week 2 are present:

- `DATA_PLAN.md` describes the external data-root policy
- `data_registry/datasets.yaml` records dataset cards
- `scripts/data/validate_data_registry.py` validates the registry
- `scripts/data/prepare_standard_datasets.py` supports standard dataset preparation
- `phase01_research_engineering/week02_mlp_datasets_training_loops/RESULTS.md` records the completed baseline, learning-rate variant, and hidden-size variant summaries
- the hidden-size variant is currently the best recorded Week 2 MLP run
- baseline, learning-rate-variant, and hidden-size-variant loss plots are present
- baseline epoch stdout is preserved, but raw variant stdout, structured per-run metrics, exact plot-generation commands, checkpoints, and populated per-run directories were not retained
- an early 2026-07-13 dataset-backed smoke recheck could not see the external Fashion-MNIST cache, but the user later confirmed that the dataset and paths work after returning to the intended HPC node/session; the separate model-only forward recheck also passes

The pure-PyTorch basic-ML bridge has verified implementations for all three days. Day 1 covers linear regression with gradient descent, ordinary least squares, closed-form ridge regression, binary logistic regression, k-nearest neighbors, and Gaussian Naive Bayes. Day 2 covers k-means with inertia/silhouette comparison, density-based DBSCAN with parameter sensitivity, and a diagonal Gaussian mixture trained with expectation-maximization. Day 3 covers a linear SVM, CART decision tree, bootstrapped random forest, logistic gradient boosting with regression stumps, and regularized LDA. Twenty-six deterministic tests pass across the bridge. Three metrics tables, four plots, measured per-day notes, and a final assumptions/metrics/failure-case comparison are stored under `basic_ml_mini_projects/`.

Downloaded datasets, processed caches, checkpoints, and large generated artifacts should stay outside Git under `AI_ROADMAP_DATA`.
