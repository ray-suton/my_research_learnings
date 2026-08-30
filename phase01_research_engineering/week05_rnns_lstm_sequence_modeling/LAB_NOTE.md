# Week 5 Lab Note - RNNs, LSTMs, and Sequence Modeling

## Goal

Understand recurrence, hidden state, vanishing gradients, LSTM gates, sequence
batching, teacher forcing, and character-level sample generation.

## Reading Completed

- Simple RNN recurrence and why one shared transition is reused across
  timesteps.
- Vanishing/exploding gradients and why a more direct cell-state path helps
  long-term memory survive.
- LSTM input, forget, candidate, and output gates.
- Teacher-forced next-character prediction.
- Character-level language modeling with fixed-length batches.

## What I Implemented

- Manual one-timestep LSTM mini-lab in `manual_lstm_block.py`.
- README corpus loader and sentence/example splitter.
- Character vocabulary with `<PAD>` and `<EOS>` handling.
- Fixed-length input/target tensors for next-character prediction.
- Small `torch.nn.LSTM` language model with embedding and output projection.
- Config-driven training script with run records.
- Generated samples from fixed prompts.
- Bigram character baseline.
- Smoke tests for the manual block, tokenizer, batch shape, model shape, and
  tiny train step.

## Experiment Setup

- Corpus: `phase01_research_engineering/week05_rnns_lstm_sequence_modeling/README.md`
- Split: 78 train examples and 20 validation examples from 98 sentence-level
  README examples
- Vocabulary size: 65
- Model: 1-layer LSTM, embedding size 32, hidden size 64
- Sequence length: 128
- Baseline: smoothed character bigram model
- Metric: validation cross-entropy / perplexity over next-character targets
- Hardware: CPU for the recorded tiny run

## Results

- Run ID: `lstm_readme_tiny`
- Best validation loss: 1.8677 at epoch 22
- Best validation perplexity: 6.4736
- Final validation loss: 1.9465
- Final validation perplexity: 7.0042
- Bigram validation loss: 2.7505
- Bigram validation perplexity: 15.6508
- Runtime: 2.328s

The LSTM beat the bigram baseline on validation loss and generated more
README-like fragments, but it still produced noisy text.

## What Failed

- The first manual LSTM attempt used the wrong `torch.concat` call shape; the
  fixed version uses `torch.cat((x_t, h_prev), dim=0)`.
- The measured run overfit after the best validation epoch. Training loss kept
  improving, but validation loss worsened by the final epoch.
- Generated samples were locally structured but not coherent long-form text.

## What I Understand Better Now

- RNN/LSTM weights are shared across timesteps; what changes each timestep is
  the input and carried state.
- The LSTM cell state is the memory stream, while the hidden state is the
  exposed view used for prediction and the next timestep.
- The forget gate controls old memory, the input gate controls write strength,
  the candidate proposes new signed memory content, and the output gate controls
  how much updated memory is exposed.
- Teacher forcing trains every non-pad input position to predict the next token.
- Small text generation can look better than a baseline while still being too
  noisy and overfit to trust broadly.

## Next Steps

- Carry forward the habit of writing a tiny shape/operation mini-lab before
  trusting a larger PyTorch module.
- Add checkpointing in future sequence experiments when best-validation samples
  matter.
- Start Week 6 autoencoders with the same pattern: small concept demo, smoke
  tests, measured run, and honest limitations.
