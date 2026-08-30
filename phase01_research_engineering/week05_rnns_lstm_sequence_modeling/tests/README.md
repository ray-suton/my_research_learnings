# Tests

This folder holds Week 5 smoke tests.

## Implemented Tests

- manual LSTM block returns inspectable gate and state tensors
- manual LSTM `c_t` and `h_t` match the textbook update formulas
- README corpus splitting produces examples
- vocabulary contains expected special tokens
- encode/decode round trip works
- sequence batch shape is `(batch, sequence_length)`
- encoded examples include `<EOS>` and `<PAD>` where expected
- model forward pass returns logits with shape `(batch, sequence_length, vocab_size)`
- tiny train step writes a run record without crashing

## Test Intent

These are trust tests, not sample-quality tests. They prove the sequence data
and model dimensions are wired correctly before long runs.
