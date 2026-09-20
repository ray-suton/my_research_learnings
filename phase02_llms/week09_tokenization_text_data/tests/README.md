# Planned Behavioral Checks

No executable tests exist yet. The checks below are acceptance criteria for
the forthcoming learner implementation, not recorded passes.

## Tokenizer And Data

- Training receives only declared training rows; a held-out sentinel fixture
  must not enter the fitting iterator.
- Ordinary fixtures preserve case, accents, whitespace, emoji, and non-Latin
  text through encode/decode under the declared reversible settings.
- Token IDs are in range; PAD and EOS are distinct; special-token literal
  matching is checked separately from ordinary text.
- Saving/reloading preserves vocabulary, settings, IDs, and fixture outputs.
- Repeated deterministic encoding agrees; any sampled segmentation is disabled
  for the comparison.
- Corpus/subset identities and counts survive filtering. Empty rows and
  overlapping source content are reported explicitly.

## LM Examples

- Hand-written token streams verify exact next-token targets, shapes, and
  the stride-`T` overlap; include a full block and a short final block.
- Padding attention and loss masks reflect actual lengths; real EOS targets
  remain included. The padding mask is not tested as if it were a causal mask.
- Chunking never mixes train/validation/test; packing within a split follows
  the declared record-boundary policy.
- Changing preprocessing batch size preserves the resulting stream/examples;
  tails are not silently discarded per processing batch.
- Empty input, one-token input, and a block ending at EOS have explicit outcomes.

## Comparison And Test Execution

Both tokenizers must see identical comparison text without padding or truncation.
Check token/byte totals on a tiny fixture with known counts. Avoid assertions
that the custom tokenizer must outperform the pretrained reference.

Use small local fixtures for routine tests; prohibit network downloads during
tests. Check real-cache access separately, with a missing prerequisite clearly
reported. Record the eventual commands and complete outputs in
[RUN_LOG.md](../RUN_LOG.md), including skips. No training job is required to
validate a tensor/data contract.
