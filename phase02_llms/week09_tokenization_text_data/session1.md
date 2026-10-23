# Session 1 - Tokens, Vocabulary, and the Text Data Contract

## Status

Prepared on **2026-09-17**. The conceptual discussion took place on
**2026-09-20–21** and closed on **2026-09-21**, followed by the learner's
request to document and wrap up this session.

**Session 1 is complete as a guided conceptual checkpoint.** The
[discussion record](#discussion-record---2026-09-20-21) distinguishes the
learner's reasoning from instructor explanations. This does not establish
independent mastery of every subtopic or a passing implementation.

No corpus preparation, tokenizer fitting, executable test, or measured
comparison has run for Week 9. [Session 2](#session-2-handoff) is next.

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

## Opening Discussion Prompt (Historical)

Why might we choose subword tokens instead of keeping the character-level
representation from Week 5? Discuss sequence length, vocabulary size, and
unfamiliar words rather than treating one representation as universally best.

Use one focused discussion question at a time. Explain routine shape and
index arithmetic directly. Later discussion should cover train-only fitting,
information lost by normalization, and the three masks above.

## Completion Evidence

The discussion covered the text-to-ID-to-embedding path, tokenization
tradeoffs, and the proposed split, boundary, and target conventions. The
learner supplied reasoning about ID compatibility, shared pieces, causal
leakage, document overlap across splits, and validation-based selection.
Normalization/decoding and loss masking required explicit explanations.

The closure is based on this guided discussion, not an independently passed
exam or executed code. WordPiece/unigram details and some pipeline distinctions
were introduced without a full independent demonstration. Revisit them when
relevant to practical work; no additional Session 1 quiz is scheduled.

## Discussion Record - 2026-09-20-21

| Topic | Learner evidence | Guidance and assessment limit |
| --- | --- | --- |
| Subwords and unfamiliar words | Identified reuse of familiar pieces across common and rare words. | Shared pieces do not automatically receive greater attention; representing an unfamiliar word does not prove understanding its meaning. |
| Attention and context length | Initially suggested attention could remove the character-token context limitation. | Attention accesses visible tokens only. Shorter representations of identical text can fit more content into a fixed window. |
| Tokenizer/model compatibility | Explained that a pretrained model still interprets new IDs using its old embeddings. | A later explanation reinforced that reversible encoding does not establish compatibility with the model's input and output mappings. |
| Tokenizer algorithms | Restated BPE's frequency-based merging. | WordPiece's longest-match encoding and unigram's probability-based segmentation were explained; no independent implementation or complete algorithm comparison was assessed. |
| Multilingual token counts | Identified token count as a processing-cost factor. | The claim that Chinese always needs more tokens was qualified: results depend on the text, byte encoding, vocabulary, and learned merges. Alphabet size alone is insufficient. |
| Normalization and round trips | Recognized loss of distinction when different inputs map to the same IDs; suggested checking more input pairs. | Tokenizer decoding was initially confused with probabilistic language-model generation. The instructor distinguished reconstruction from prediction and explained how to isolate preprocessing changes. |
| Special tokens and loss masking | Recognized that removing EOS supervision can impair stopping; after a worked example, identified loss masking as selecting which prediction errors count and PAD as an invalid target. | Excluding a target from loss does not remove its input token. The input attention mask and shifted-target loss mask are not interchangeable. |
| Causality and packing | Explained that attending to future tokens violates autoregressive inference; recognized that B can still attend to earlier A across EOS. | Separate batch rows, simple causal packing, and attention restricted by record boundaries were distinguished. EOS itself does not enforce isolation. |
| Chunking and evaluation | Identified lost context across paragraph chunks and suggested a controlled evaluation. | No concrete long-range evaluation was executed or independently designed. Longer or overlapping chunks were discussed as tradeoffs. |
| Data splits and selection | Recognized tokenizer fitting as learning, identified related chunks crossing splits, and explained that choosing on validation makes the selected result dependent on validation. | Familiar words/topics may appear in held-out documents. Preserve the original splits, audit overlap, and reserve test text until the protocol is fixed. |

The discussion also separated tokenization efficiency from generation length.
The learner's BLEU question concerned short predictions, while the tokenizer
comparison keeps the underlying text fixed. BLEU's n-gram precision component
can favor incomplete output; its brevity penalty addresses that issue. This
was an explanatory detour, not a computed BLEU experiment. See the
[original BLEU paper, Section 2.2](https://aclanthology.org/P02-1040.pdf).

## Clarifications To Carry Forward

- Token IDs identify vocabulary entries; the model learns embedding vectors
  and output predictions associated with those entries. A new tokenizer can
  round-trip perfectly while disagreeing with a pretrained model's mappings.
- Byte coverage permits representation of ordinary input bytes. It cannot
  undo lossy normalization or decoder cleanup. Tokenizer decoding reconstructs
  the supplied IDs; language-model generation predicts new IDs.
- BPE applies learned merge priorities. WordPiece uses greedy longest-match
  encoding; exact vocabulary-training recipes vary. Unigram scores complete
  segmentations using learned piece probabilities. Probabilistic segmentation
  does not mean the decoder guesses a different original word.
- An advertised context window, the sequence length selected for training,
  and batch size are different quantities. More batch rows do not give each
  row access to the other rows. Independent chunks can lose useful context
  even when every original token is retained somewhere in the dataset.
- Train-only fitting applies to the tokenizer as well as the later model.
  Validation can guide choices; the resulting selected validation score is
  not an untouched final evaluation. Concatenation is allowed within a split.

Algorithm references used in the discussion:
[WordPiece](https://huggingface.co/learn/llm-course/en/chapter6/6),
[unigram](https://huggingface.co/learn/llm-course/en/chapter6/7), and
[byte-level BPE](https://huggingface.co/docs/transformers/en/tokenizer_summary#byte-level-bpe).
The WordPiece course identifies its original training reconstruction as
approximate; its pair-scoring formula was not treated as a universal exact rule.
For practical packing, see the
[NVIDIA overview](https://docs.nvidia.com/nemo/megatron-bridge/0.6.2/training/packed-sequences.html).

## Worked Example - Input Attention And Prediction Loss

This is a hand-written teaching example, not output from an executed tokenizer.
The real sequence is `cat, sleeps, EOS`; PAD fills the remaining positions.

```text
Source:                cat     sleeps   EOS   PAD   PAD
Input:                 cat     sleeps   EOS   PAD
Raw next-token target: sleeps  EOS      PAD   PAD
Input attention mask:  1       1        1     0
Loss mask:             1       1        0     0
```

At the input EOS position, the input is real but the next target is padding.
Ignore that prediction error. The earlier `sleeps -> EOS` prediction remains
trainable and teaches stopping. The learner identified that the loss mask
selects a position's prediction error and that PAD is not a real training target.

Under our proposed causal packing across records, `EOS -> first real token of
the next record` has a valid target and receives loss weight 1. `EOS -> PAD`
receives 0. The instructor supplied the real-next-record case after the learner
correctly excluded PAD. Do not decide loss validity solely from the input's
EOS identity, and do not copy the input attention mask as the loss mask.

## Working Contract And Remaining Practical Work

The [experiment plan](EXPERIMENT_PLAN.md) remains the detailed proposal. These
rules were carried forward in the wrap-up; exact versions, revisions, runtime
access, and hyperparameters still need to be resolved before execution.

- Use WikiText-2 raw and preserve the original train/validation/test membership.
  Fit the custom byte-level BPE vocabulary/merges on training text only.
- Preserve case, accents, spaces, and newlines for ordinary-text round trips.
  Control added special tokens and decoder cleanup; define literal special-token
  handling separately before implementing its tests.
- Treat each nonempty source row as a record and append one EOS. A source row
  is not assumed to be an entire article. Keep PAD and EOS IDs distinct.
- Pack within each split. Ordinary causal attention may use earlier records
  in the same block; the prototype does not claim document-isolated attention.
- Form next-token targets with one shift, retain real EOS targets, and ignore
  targets without a valid input/next-token pair. Check the eventual model API
  to avoid shifting an already shifted target a second time.
- Compare the custom tokenizer and frozen GPT-2 reference on identical
  validation text. This is a descriptive comparison with different vocabulary
  sizes and training histories, not proof of superior language understanding.
  Reserve test text for a fixed-protocol final audit.

Corpus preparation, implementation, serialization checks, executable tests,
measured comparisons, prepared LM data, and the Week 9 lab note remain pending.

## Session 2 Handoff

1. Trace a tiny BPE example to connect pair counts, merges, and vocabulary entries.
2. Before fitting, inspect actual package/cache access and record dataset
   revision, source splits, and the training-only fitting subset. Earlier absent
   paths do not rule out another cache; this closure did not recheck the environment.
3. Build the smallest learner-owned tokenizer path, then save/reload it and
   check ordinary-text encoding and decoding with small fixtures. Record actual
   commands and outcomes in [RUN_LOG.md](RUN_LOG.md).

Session 2 has not started. The learner owns normal lesson implementation;
the instructor explains and reviews. This closure authorizes documentation,
not automatic package installation, downloads, or implementation.
