# Week 5 - RNNs, LSTMs, and Sequence Modeling

## Purpose

Week 5 moves from image models into sequence models. The goal is to understand
recurrence before Transformers: hidden state, sequence batching, vanishing
gradients, LSTM gates, teacher forcing, and sample generation.

## Scope

- Study simple RNNs, LSTMs, vanishing gradients, teacher forcing, and sequence
  loss.
- Hand-code one artificial LSTM timestep to expose the gates that
  `torch.nn.LSTM` hides.
- Choose or create a tiny character-level text corpus.
- Build a character vocabulary and sequence dataset.
- Implement a simple RNN or LSTM language model.
- Add config-driven training controls, smoke tests, and run records following
  the Week 4 pattern.
- Train a tiny model and generate samples.
- Compare against a simple baseline such as unigram or bigram sampling.

## Flexible Session Plan

This plan is artifact-driven, not a strict five-day schedule. Combine coding
and training when the data path is clear, and split a session only if the tensor
shapes or tests are not trustworthy yet.

### Session 1 - Concepts, Corpus, and Shape Contract

- Study simple RNN recurrence, vanishing/exploding gradients, LSTM gates,
  teacher forcing, and sequence loss just enough to explain the training setup.
- Choose or create one tiny character-level corpus.
- Record the corpus source, split rule, metric, and any redistribution limits.
- Write the expected tensor-shape contract:
  `(batch, sequence_length)` inputs, `(batch, sequence_length)` targets, and
  `(batch, sequence_length, vocab_size)` logits.

Deliverable: corpus decision plus shape notes.

### Session 2 - Minimum Trainable Loop

- Build the character vocabulary with encode/decode helpers.
- Create fixed-length input/target sequence batches.
- Implement the first simple recurrent model, preferably an LSTM unless there
  is a reason to start with a vanilla RNN.
- Add config-driven controls for corpus path, model type, sequence length,
  hidden size, batch size, learning rate, seed, and sample length.
- Add smoke tests for vocabulary round trip, batch shapes, model output shape,
  and one tiny train step.
- Run the smoke tests, then run a very small training sanity check if they pass.

Deliverable: runnable tokenizer, dataset, model, config path, tests, and tiny
training sanity check.

### Session 3 - Measured Run, Samples, and Baseline

- Run one small measured LSTM training job on the tiny corpus.
- Save run config, train/validation loss, validation perplexity if stable,
  runtime notes, and generated samples.
- Implement a simple non-neural unigram or bigram baseline.
- Evaluate the baseline on the same validation split where practical.
- Compare baseline samples against recurrent-model samples without overclaiming
  small or subjective differences.
- Record commands and outcomes in `RUN_LOG.md` only after runs happen.

Deliverable: first measured recurrent-model run plus baseline comparison in
`RESULTS.md`.

### Session 4 - Lab Note and Carry-Forward

- Write the Week 5 lab note: recurrence intuition, LSTM gate intuition,
  batching mistakes, training behavior, sample quality, and limitations.
- Update the results table with the final selected Week 5 run.
- Record overfitting, memorization, unstable generation, or data leakage risks.
- List what should carry into Week 6, especially config discipline, smoke
  tests, and checkpointing if it is added.

Deliverable: completed Week 5 lab note and next-step notes for Week 6.

## Working Rules

- Keep the first corpus tiny and auditable.
- Define train/validation split and metric before training.
- Track config, metrics, generated samples, runtime, and limitations.
- Do not use the validation set as a generation prompt source unless recorded.
- Keep large artifacts outside Git.

## Folder Guide

- `configs/`: planned config files and config-field notes.
- `DESIGN.md`: Session 1 design decisions for corpus, special tokens,
  input/target shapes, model target, generation, baseline, and smoke tests.
- `manual_lstm_block.py`: one artificial LSTM timestep with printed forget,
  input, candidate, output, cell, and hidden values.
- `tests/`: smoke tests for tokenizer, batches, model forward pass, and tiny
  training.
- `results/`: small tracked summaries only.
- `results/runs/`: per-run metadata layout.
- `RUN_LOG.md`: commands and outcomes after they actually run.
- `RESULTS.md`: measured results and generated sample comparison.
- `LAB_NOTE.md`: final Week 5 learning note.

## Current Checkpoint

Week 5 is complete from a learning-work perspective. The project has a manual
LSTM gate mini-lab, a config-driven character-level LSTM training path, smoke
tests, one measured README-corpus run, generated samples, and a bigram baseline
comparison recorded in `RESULTS.md`.
