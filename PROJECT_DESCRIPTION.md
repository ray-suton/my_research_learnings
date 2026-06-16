# Project Description

This workspace contains a Codex-ready AI research learning roadmap for building practical AI research skill over a 52-week schedule. The roadmap starts from a student who has completed the Machine Learning and Deep Learning Specializations and is preparing to become a stronger incoming AI student researcher.

The current source of truth is `ai_research_roadmap_codex.md`. The main roadmap remains in the repo root, and phase-named roadmap files mirror the relevant phase plan inside each `phaseXX_*` folder. Week 1 is a bootstrap task: create the AI research repo skeleton, environment files, PyTorch smoke tests, SLURM GPU template, Week 1 project folder, and Week 1 lab-note template.

The repository is a valid Git repo on `master` with two commits: the initial scaffold commit and a follow-up `.gitignore` cleanup commit. The tracked scaffold currently includes `README.md`, `.gitignore`, `requirements.txt`, `environment.yml`, `scripts/smoke_tests/gpu_smoke_test.py`, phase directories, `notes/`, and `results/`. `TODO.md` exists locally but is ignored by `.gitignore`, so it is useful as local status evidence but is not part of the committed project state.

Tuesday, June 16 setup is mostly complete for the local scaffold: repo initialization, environment files, folder structure, `.gitignore`, and the GPU smoke-test script exist. Later Week 1 items remain unimplemented or unverified: no SLURM template, no autograd test, no linear-regression script, no Week 1 lab note, and no saved Week 1 result artifacts.

Latest local verification: `python3 scripts/smoke_tests/gpu_smoke_test.py` runs successfully on CPU with PyTorch 2.8.0+cu126. CUDA is unavailable in the current environment, so GPU/HPC verification remains unproven.
