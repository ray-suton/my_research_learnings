# Week 5 - RNNs, LSTMs, and Sequence Modeling

## Purpose

Week 5 moves from image models into sequence models. The goal is to understand
recurrence before Transformers: hidden state, sequence batching, vanishing
gradients, LSTM gates, teacher forcing, and sample generation.

## Scope

- Study simple RNNs, LSTMs, vanishing gradients, teacher forcing, and sequence
  loss.
- Choose or create a tiny character-level text corpus.
- Build a character vocabulary and sequence dataset.
- Implement a simple RNN or LSTM language model.
- Add config-driven training controls, smoke tests, and run records following
  the Week 4 pattern.
- Train a tiny model and generate samples.
- Compare against a simple baseline such as unigram or bigram sampling.

## Working Rules

- Keep the first corpus tiny and auditable.
- Define train/validation split and metric before training.
- Track config, metrics, generated samples, runtime, and limitations.
- Do not use the validation set as a generation prompt source unless recorded.
- Keep large artifacts outside Git.

## Folder Guide

- `configs/`: planned config files and config-field notes.
- `tests/`: smoke tests for tokenizer, batches, model forward pass, and tiny
  training.
- `results/`: small tracked summaries only.
- `results/runs/`: per-run metadata layout.
- `RUN_LOG.md`: commands and outcomes after they actually run.
- `RESULTS.md`: measured results and generated sample comparison.
- `LAB_NOTE.md`: final Week 5 learning note.

## Current Checkpoint

Week 5 is scaffolded. Next step: study recurrence/LSTM concepts and choose the
tiny character-level corpus.
