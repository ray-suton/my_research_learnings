# Project Description

This workspace contains a Codex-ready AI research learning roadmap for building practical AI research skill over a 52-week schedule. The roadmap starts from a student who has completed the Machine Learning and Deep Learning Specializations and is preparing to become a stronger incoming AI student researcher.

The current source of truth is `ai_research_roadmap_codex.md`. Week 1 is a bootstrap task: create the AI research repo skeleton, environment files, PyTorch smoke tests, SLURM GPU template, Week 1 project folder, and Week 1 lab-note template.

The repository is now a valid Git repo with no commits yet. The Tuesday, June 16 scaffold is partially present: the phase directories, notes/results directories, `requirements.txt`, and `scripts/smoke_tests/gpu_smoke_test.py` exist. `README.md`, `TODO.md`, `.gitignore`, and `environment.yaml` are currently empty, and the roadmap calls for `environment.yml`. No SLURM template, autograd or linear-regression script, Week 1 lab note, or initial commit exists yet.

Latest local verification: `python3 scripts/smoke_tests/gpu_smoke_test.py` runs successfully on CPU with PyTorch 2.8.0+cu126. CUDA is unavailable in the current environment, so GPU/HPC verification remains unproven.
