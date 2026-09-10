# Project Description

This workspace contains a Codex-ready AI research learning roadmap for building practical AI research skill over a 52-week schedule. It starts from a student who has completed the Machine Learning and Deep Learning Specializations and is preparing to become a stronger incoming AI student researcher.

The roadmap source of truth is `ai_research_roadmap_codex.md`. Phase-specific roadmap files under `phaseXX_*` mirror the relevant sections of the root roadmap.

The planned Week 2 three-run MLP comparison, the original three-day pure-PyTorch basic-ML mini-project bridge, Phase 1 Week 3 CNNs/mini-ResNets on CIFAR-10, Phase 1 Week 4 optimization/experiment discipline, Phase 1 Week 5 RNNs/LSTMs/sequence modeling, and Phase 1 Week 6 autoencoders/representation learning are finished from a learning-work perspective. The completed basic-ML bridge is preserved as a runnable reference under `basic_ml_mini_projects/archive/completed_reference_2026_07_15/`. A learner-owned Basic ML rebuild now has draft implementations for the main bridge algorithms and is documented in `basic_ml_mini_projects/BASIC_ML_REBUILD_REPORT.md`; ridge regression and regularized LDA are intentionally deferred. The next active learning area is Week 7: variational autoencoders. Week 1 bootstrap is complete from a learning-work perspective:

- repository scaffold and environment files exist
- local PyTorch smoke test exists
- SLURM GPU smoke-test script exists
- GPU smoke test passed on an NVIDIA RTX 5000 Ada Generation through SLURM
- autograd sanity test exists and passes
- tiny linear-regression training script exists and passes
- Week 1 training log, loss plot, and prediction-vs-actual plot exist
- Week 1 project README exists
- Week 1 note records the verified results and learning notes

At the 2026-09-07 inspection, the active branch is `main`, tracking `origin/main`, with latest commit `6d325bfc3a` (`week05 done`, 2026-08-30). The existing uncommitted work consists of this modified overview, the untracked Week 6 and Week 7 project directories, and the root images `Compare and Contrast.png` and `Final_plot.png`. Earlier Basic ML and Week 3–5 work is already represented in Git history. Repository Git history is intentionally read-only for agents.

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

The completed Week 5 learning area is `phase01_research_engineering/week05_rnns_lstm_sequence_modeling/`: recurrence, LSTM gates, sequence batching, character-level language modeling, sample generation, and comparison against a simple sequence baseline. It includes an inspectable manual LSTM gate mini-lab, a config-driven tiny README-corpus LSTM run, smoke tests, generated samples, and a bigram baseline comparison.

The completed Week 6 learning area is `phase01_research_engineering/week06_autoencoders_representation_learning/`: autoencoder bottlenecks, reconstruction loss, validation split discipline, average-image baseline comparison, denoising behavior, and reconstruction visual inspection. Sessions 1 and 2 established the Fashion-MNIST shape contract and first plain fully connected autoencoder. Session 3 added CLI-controlled measured runs, a deterministic validation split, train/validation/test reconstruction losses, an average-image baseline, and a project-local reconstruction grid. Session 4 added CLI-controlled Gaussian-noise corruption, noisy-input baseline measurement, metrics JSON output, and a three-row original/noisy/denoised grid. Session 5 added the final lab note, conservative latent-bottleneck interpretation, supervised-vs-representation comparison, failure modes, and Week 7 VAE carry-forward.

The active Week 7 roadmap area is variational autoencoders under `phase01_research_engineering/week07_variational_autoencoders/`. Session 1 concepts are complete, and Session 2's minimum trainable VAE checkpoint completed on 2026-09-08. The learner-owned loader, model, loss helper, and training function pass all eight saved CPU tests in `tests/test_session2_smoke.py`, including cached Fashion-MNIST splits, tensor shapes, KL values, reconstruction scaling, gradient paths, one parameter-update step, and prior decoding. The reproduction command is in `session2.md` and `RUN_LOG.md`; full output and source hashes are in `results/session2_smoke_test.txt`.

Week 7 Sessions 3–5 completed their technical work on 2026-09-08. The measured-run CLI, validation, and artifact pipeline now support a five-epoch VAE baseline, fixed latent sizes 8/16/32/64, and a fresh matched plain-autoencoder comparison. Latent size 8 was selected by validation sampled-reconstruction MSE before test evaluation. Its test MSE was 0.024081 with posterior samples and 0.020514 with decode(mu), compared with 0.020632 for the matched AE. All 13 saved checks pass, and the measured artifacts are under `results/completed_20260908/` within the Week 7 project. Sessions 3/4/5 and `LAB_NOTE.md` record the figures, exact settings, and one-seed limitations. Checkpoints under `/tmp/week07_vae_20260908/` are temporary and can be regenerated from saved commands.

The basic VAE intuition review completed on 2026-09-10 through guided discussion of prior sampling, KL pressure, information loss, sampling variability, reparameterization, and the encoder’s training role. This is a practical learning checkpoint, not an independent formal-derivation assessment. Week 8 GANs is the next roadmap area; no Week 8 implementation has started.

Downloaded datasets, processed caches, checkpoints, and large generated artifacts should stay outside Git under `AI_ROADMAP_DATA`.
