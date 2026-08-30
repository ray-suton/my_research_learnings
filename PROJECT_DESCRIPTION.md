# Project Description

This workspace contains a Codex-ready AI research learning roadmap for building practical AI research skill over a 52-week schedule. It starts from a student who has completed the Machine Learning and Deep Learning Specializations and is preparing to become a stronger incoming AI student researcher.

The roadmap source of truth is `ai_research_roadmap_codex.md`. Phase-specific roadmap files under `phaseXX_*` mirror the relevant sections of the root roadmap.

The planned Week 2 three-run MLP comparison, the original three-day pure-PyTorch basic-ML mini-project bridge, Phase 1 Week 3 CNNs/mini-ResNets on CIFAR-10, Phase 1 Week 4 optimization/experiment discipline, and Phase 1 Week 5 RNNs/LSTMs/sequence modeling are finished from a learning-work perspective. The completed basic-ML bridge is preserved as a runnable reference under `basic_ml_mini_projects/archive/completed_reference_2026_07_15/`. A learner-owned Basic ML rebuild now has draft implementations for the main bridge algorithms and is documented in `basic_ml_mini_projects/BASIC_ML_REBUILD_REPORT.md`; ridge regression and regularized LDA are intentionally deferred. The next active learning area is Phase 1 Week 6: autoencoders and representation learning. Week 1 bootstrap is complete from a learning-work perspective:

- repository scaffold and environment files exist
- local PyTorch smoke test exists
- SLURM GPU smoke-test script exists
- GPU smoke test passed on an NVIDIA RTX 5000 Ada Generation through SLURM
- autograd sanity test exists and passes
- tiny linear-regression training script exists and passes
- Week 1 training log, loss plot, and prediction-vs-actual plot exist
- Week 1 project README exists
- Week 1 note records the verified results and learning notes

The active branch is `main`. The current worktree contains learner-owned Basic ML drafts, completed Week 3 CNN/mini-ResNet work, a Week 3 dimension-mechanics bridge, completed Week 4 config-driven ablations, and completed Week 5 sequence-modeling work. These changes still need human Git review before any staging or commit; repository Git history is intentionally read-only for agents.

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

The archived pure-PyTorch basic-ML bridge has verified implementations for all three days. Day 1 covers linear regression with gradient descent, ordinary least squares, closed-form ridge regression, binary logistic regression, k-nearest neighbors, and Gaussian Naive Bayes. Day 2 covers k-means with inertia/silhouette comparison, density-based DBSCAN with parameter sensitivity, and a diagonal Gaussian mixture trained with expectation-maximization. Day 3 covers a linear SVM, CART decision tree, bootstrapped random forest, logistic gradient boosting with regression stumps, and regularized LDA. Twenty-six deterministic tests pass across the archive. Three metrics tables, four plots, measured per-day notes, and a final assumptions/metrics/failure-case comparison remain under the dated reference path.

The completed Week 5 learning area is `phase01_research_engineering/week05_rnns_lstm_sequence_modeling/`: recurrence, LSTM gates, sequence batching, character-level language modeling, sample generation, and comparison against a simple sequence baseline. It includes an inspectable manual LSTM gate mini-lab, a config-driven tiny README-corpus LSTM run, smoke tests, generated samples, and a bigram baseline comparison. The next roadmap area is Week 6 autoencoders.

Downloaded datasets, processed caches, checkpoints, and large generated artifacts should stay outside Git under `AI_ROADMAP_DATA`.
