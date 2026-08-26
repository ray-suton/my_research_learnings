# Tests

This folder will hold Week 5 smoke tests.

## Planned Tests

- corpus text loads
- vocabulary contains expected special tokens or characters
- encode/decode round trip works
- sequence batch shape is `(batch, sequence_length)`
- model forward pass returns logits with shape `(batch, sequence_length, vocab_size)`
- tiny train step runs without crashing

## Test Intent

These are trust tests, not sample-quality tests. They prove the sequence data
and model dimensions are wired correctly before long runs.
