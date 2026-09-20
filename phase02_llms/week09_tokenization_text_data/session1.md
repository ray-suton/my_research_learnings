# Session 1 - Tokens, Vocabulary, and the Text Data Contract

## Status

Prepared on **2026-09-17**. The learning discussion has not yet taken place.
Completion should be recorded from the learner's responses, including any
guided corrections.

## From Week 5 To Week 9

The Week 5 LSTM used characters as its units. Here we ask how choosing different
units changes the sequence the model receives:

```text
text -> tokenizer -> integer token IDs -> embedding lookup -> sequence model
```

A token can represent a word, part of a word, punctuation, or bytes. Its ID is
a vocabulary index, not a numerical measure of meaning. The embedding is the
vector looked up for that ID; fitting a tokenizer and training those vectors
are separate operations.

For illustration only, a word such as `replaying` could be split into
`re` and `playing`, or into smaller pieces. An actual segmentation must be
measured with a particular tokenizer; we have not run one yet.

## Three Subword Approaches

| Approach | Core idea |
| --- | --- |
| BPE | Begin with small units and repeatedly merge frequent adjacent pairs. |
| WordPiece | Learn a subword vocabulary using a different pair-selection objective; encoding commonly uses longest matching vocabulary pieces. |
| Unigram | Start with candidate pieces, fit probabilities, and prune pieces; choose a high-probability segmentation at encoding time. |

Byte-level BPE starts from a complete byte alphabet, so an unseen ordinary
word can still be represented. Character-based vocabularies need a policy for
unseen characters. These distinctions are introduced in the official
[tokenization overview](https://huggingface.co/docs/transformers/tokenizer_summary).

The first practical implementation will focus on BPE. The learner should be
able to explain the alternatives without writing three tokenizer libraries.

## The Entire Pipeline Matters

Normalization may change the text before segmentation; pre-tokenization may
limit where pieces can merge. Vocabulary rules map pieces to IDs, and
post-processing may insert special tokens. Decoding has its own policy.
Consequently, an encode/decode round trip must be checked against the declared
normalization and special-token behavior. See the official
[tokenization pipeline](https://huggingface.co/docs/tokenizers/pipeline).

Our initial custom-tokenizer proposal preserves case, accents, whitespace,
and newlines. Its ordinary-text round-trip tests will disable added special
tokens and decoder cleanup. Literal strings that resemble special tokens need
an explicit policy and separate fixtures.

## Special Tokens And Batches

For this project, reserve separate padding and end-of-record IDs. Padding is
batch filler; an end marker is a meaningful boundary that can be a target.
Registering a token and inserting it into examples are separate decisions.

Padding adds filler to equalize lengths. Truncation removes text beyond a
limit. Packing combines shorter sequences into token blocks; it needs explicit
boundary rules. Read the official
[padding/truncation guide](https://huggingface.co/docs/transformers/main/en/pad_truncation)
for the library's controls.

Three controls serve different purposes in the future model path:

- A padding attention mask identifies real input positions and padding.
- A causal attention mask prevents a position from using future tokens.
- A loss mask excludes invalid targets, such as padding, from the objective.

An all-ones padding mask does not make an attention layer causal. An EOS token
alone also does not prevent attention across a packed record boundary. The
[causal LM guide](https://huggingface.co/docs/transformers/tasks/language_modeling)
provides context for next-token prediction; our exact interface is declared
in the [experiment plan](EXPERIMENT_PLAN.md).

## Corpus And Comparison

The working corpus is WikiText-2 raw, using its original splits. Learn the
custom vocabulary/merges from training text only. Validation text measures
segmentation behavior; reserve test text for a final fixed-protocol audit.

Compare token counts on identical text. Fewer tokens can fit more text into
a fixed token budget, but that alone does not establish a better language
model. Vocabulary size, text domain, and tokenizer training data differ.

## First Discussion Prompt

Why might we choose subword tokens instead of keeping the character-level
representation from Week 5? Discuss sequence length, vocabulary size, and
unfamiliar words rather than treating one representation as universally best.

Use one focused discussion question at a time. Explain routine shape and
index arithmetic directly. Later discussion should cover train-only fitting,
information lost by normalization, and the three masks above.

## Completion Evidence

Session 1 is ready to close when the learner can describe the text-to-ID-to-
embedding path, the main tokenization tradeoffs, and the proposed split,
boundary, and target conventions. Note which ideas required explanation.
Then proceed to Session 2's smallest tokenizer implementation; no assessed
understanding or successful code is implied by this document's existence.
