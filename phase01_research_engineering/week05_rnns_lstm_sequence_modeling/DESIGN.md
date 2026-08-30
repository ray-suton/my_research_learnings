# Week 5 Design - Character-Level LSTM Generation

## Session 1 Decision

Use the Week 5 README as the first toy corpus for a character-level sequence
generation project.

This is intentionally small. The goal is not to build a useful language model;
the goal is to understand recurrent sequence modeling, LSTM state flow,
teacher-forced next-token prediction, batching, stopping tokens, sampling, and
baseline comparison.

## Learning Target

Train a small LSTM to predict the next character in short README-derived text
examples. After training, generate short README-like samples from a prompt and
stop generation when the model emits an end token.

The model should learn local character patterns such as:

- Markdown headings and bullets
- technical terms such as `RNN`, `LSTM`, `sequence`, and `generation`
- common word fragments and spacing
- sentence endings and short continuation patterns

It is not expected to reproduce the whole README or produce polished long-form
text.

## Corpus Design

Initial corpus:

```text
phase01_research_engineering/week05_rnns_lstm_sequence_modeling/README.md
```

Example unit:

- split the README into sentence-level examples
- keep useful Markdown fragments such as headings and bullets
- append `<EOS>` after each real example
- pad shorter examples with `<PAD>` for batching

Reason for sentence-level examples:

- gives more examples than treating each paragraph as one sequence
- makes stopping behavior easy to inspect through `<EOS>`
- keeps the first implementation simple

Tradeoff:

- sentence-level examples reduce long-range structure, so the model may learn
  local README style better than full section flow

## Special Tokens

Use two explicit special tokens:

```text
<EOS>  end of sentence/example
<PAD>  padding for fixed-length batches
```

`<EOS>` is a real target. It teaches the model when to stop.

`<PAD>` is not real text. It exists only so examples of different lengths can
fit into the same tensor batch. `<PAD>` positions must be ignored by the loss.

During generation:

- stop when `<EOS>` is sampled
- do not intentionally sample `<PAD>`
- also stop at a maximum generated length to avoid infinite output

## Input and Target Rule

Each sentence/example is encoded as characters plus `<EOS>`.

The model input is the sequence up to the final token. The target is the same
sequence shifted one step left, so every input position has exactly one
next-token label.

Example:

```text
full:       c1 c2 c3 ... cN <EOS>
input_ids:  c1 c2 c3 ... cN
target_ids: c2 c3 c4 ... <EOS>
```

After padding:

```text
input_ids:  (batch_size, sequence_length)
target_ids: (batch_size, sequence_length)
logits:     (batch_size, sequence_length, vocab_size)
```

The first implementation should use:

```text
sequence_length = 128
```

Examples longer than the usable length can be skipped at first. Truncation can
be added later if needed, but skipping is easier to reason about for the first
pass.

## Model Design

Start with one small LSTM language model:

```text
model_name = lstm
embedding_dim = 64
hidden_size = 128
num_layers = 1
```

Data flow:

```text
character ids -> embedding -> LSTM -> linear projection -> logits over vocab
```

Expected tensor flow:

```text
input_ids:  (batch_size, sequence_length)
embeddings: (batch_size, sequence_length, embedding_dim)
lstm_out:   (batch_size, sequence_length, hidden_size)
logits:     (batch_size, sequence_length, vocab_size)
```

## Manual LSTM Mini-Lab

Before relying on `torch.nn.LSTM`, use `manual_lstm_block.py` to inspect one
fully artificial timestep. This mini-lab keeps the input and hidden vectors
tiny and prints:

- `z_t = [x_t, h_prev]`
- forget gate `f_t`
- input/write gate `i_t`
- candidate memory `g_t`
- output gate `o_t`
- updated cell state `c_t`
- updated hidden state `h_t`

This is a learning artifact, not a training implementation. Its purpose is to
make the gate arithmetic concrete before using PyTorch's optimized LSTM module.

## Training Objective

Use teacher-forced next-character prediction.

At every non-pad position, optimize cross-entropy between:

- model logits for the next token
- the true next token from `target_ids`

Ignore `<PAD>` positions in the loss. Keep `<EOS>` positions in the loss.

Track:

- train loss
- validation loss
- validation perplexity if stable
- runtime
- generated samples

## Generation Design

Generate one character at a time from a short prompt.

Recommended first settings:

```text
temperature = 0.8 or 1.0
max_generated_chars = 300
```

Use temperature sampling rather than always taking the highest-probability next
character. Low temperature should produce safer but more repetitive samples.
High temperature should produce more varied but noisier samples.

Generation stops when:

- `<EOS>` is sampled
- or `max_generated_chars` is reached

## Baseline

Use a simple bigram character baseline first.

The bigram baseline predicts the next character from only the immediately
previous character. It has no hidden state and no learned long-range memory.

The LSTM should be compared against this baseline using:

- validation negative log likelihood or perplexity if practical
- generated sample quality
- obvious failure modes such as repetition, malformed Markdown, or failure to
  stop

## First Smoke Tests

Before running a measured training job, add tests for:

- README corpus text loads
- sentence/example splitting produces at least one example
- vocabulary can encode and decode a simple string
- encoded examples include `<EOS>`
- padded batches have shape `(batch_size, sequence_length)`
- target batches have the same shape as input batches
- model logits have shape `(batch_size, sequence_length, vocab_size)`
- one tiny train step runs and ignores `<PAD>` positions in the loss

## Session 2 Starting Point

Implement the minimum trainable loop:

1. corpus loading and sentence/example splitting
2. vocabulary with `<PAD>` and `<EOS>`
3. fixed-length dataset and batch creation
4. LSTM forward pass
5. smoke tests
6. tiny training sanity check

Do not run a measured experiment until the shape tests pass.
