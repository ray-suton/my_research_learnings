# Data Plan

This repository should treat data as a reproducible external dependency, not as source code.

The repo tracks:

- dataset choices
- source and license notes
- preprocessing steps
- split decisions
- small hand-written fixtures when needed for tests
- scripts that can recreate local data state

The repo should not track:

- raw downloaded datasets
- processed dataset caches
- Hugging Face or torchvision caches
- model checkpoints
- large logs, tensor dumps, or generated artifacts

## Data Root

Set one external data root before running data scripts:

```bash
export AI_ROADMAP_DATA="$HOME/data/ai_research_roadmap"
```

On the HPC, prefer scratch or project storage:

```bash
export AI_ROADMAP_DATA="/scratch/$USER/ai_research_roadmap_data"
```

If `AI_ROADMAP_DATA` is not set, the data scripts default to:

```text
~/data/ai_research_roadmap
```

Do not put the data root inside this Git repo.

## Repo Data Surfaces

```text
data_registry/
  datasets.yaml              # dataset source, license, usage, and preparation status
scripts/data/
  prepare_standard_datasets.py
  validate_data_registry.py
```

The dataset registry is the first place to check before adding a new dataset.

## Phase Strategy

### Phase 1: Research Engineering

Use tiny, standard, auto-download datasets:

- synthetic linear-regression data for Week 1
- MNIST or Fashion-MNIST for MLPs, autoencoders, and GAN sanity checks
- CIFAR-10 for CNN and training-loop practice
- a small public text corpus for toy language modeling

### Phase 2: LLMs

Use small text corpora and user-owned notes first:

- WikiText-2 or another small public corpus for tokenization and mini-GPT work
- `notes/` and paper summaries for domain-corpus and RAG practice
- small hand-authored prompt, instruction, and preference sets before using external instruction data

Record contamination risk for every evaluation set.

### Phase 3: Efficiency and Interpretability

Reuse Phase 1 and Phase 2 datasets where possible. This phase is about controlled comparisons, so stable repeated data is more important than dataset variety.

### Phase 4: Vision, Multimodal, and Generation

Start with small public vision data:

- CIFAR-10 for architecture comparisons
- a tiny custom image-question-answer manifest for VQA format practice
- generated outputs stored outside Git unless they are tiny report figures

Track whether images and generated outputs are safe to redistribute.

### Phase 5: Graphs, Audio, World Models, and Embodied AI

Use small canonical datasets only:

- Cora or Citeseer for graph learning
- a tiny speech dataset for audio tooling
- a small robot/behavior-cloning dataset only after its storage and license are understood

These are later-roadmap items, so keep them in `planned` status until needed.

### Phase 6: Capstone

The capstone dataset should be chosen with:

- a clear baseline
- a fixed metric
- known license terms
- a small reproducible subset
- a documented train/validation/test split
- leakage checks

Do not start the capstone by collecting a large dataset.

## Required Dataset Card Fields

Every nontrivial dataset should have enough information to answer:

- What is the source?
- What is the license or usage constraint?
- Which roadmap phase and week uses it?
- Where is it stored under `AI_ROADMAP_DATA`?
- What exact preprocessing produced the training/eval files?
- What split was used?
- What could leak into evaluation?
- Can this data be redistributed?

Treat registry license fields as working notes until the exact upstream dataset/version has been checked. Before publishing data, derived data, or model outputs that may reproduce source content, verify the current upstream terms and record the review in the relevant project note.

## Commands

Validate the registry:

```bash
python scripts/data/validate_data_registry.py
```

List supported standard dataset downloads:

```bash
python scripts/data/prepare_standard_datasets.py --list
```

Dry-run preparation:

```bash
python scripts/data/prepare_standard_datasets.py --dataset fashion_mnist --dry-run
```

Prepare the first Week 2 dataset:

```bash
python scripts/data/prepare_standard_datasets.py --dataset fashion_mnist
```

Prepare core early-roadmap datasets:

```bash
python scripts/data/prepare_standard_datasets.py --dataset fashion_mnist --dataset cifar10 --dataset wikitext2
```

## Git Rule

If a file is expensive to recreate but too large for Git, store it under `AI_ROADMAP_DATA` and document the reproduction command in `data_registry/datasets.yaml` or the project README.

If a file is tiny and used for tests, keep it under a clearly named fixture path and document that it is intentionally committed.
