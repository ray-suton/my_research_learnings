# Experiment Plan

## Learning Question

Can a small recurrent language model learn local character-level structure from
a tiny corpus, and how does it compare with a simple non-neural sequence
baseline?

## Session 1 Design Decision

The first corpus is the Week 5 README:

```text
phase01_research_engineering/week05_rnns_lstm_sequence_modeling/README.md
```

Use sentence-level examples with `<EOS>` appended after each real example.
Use `<PAD>` only for fixed-length batching. Train a character-level LSTM with
teacher-forced next-token prediction:

```text
full:       c1 c2 c3 ... cN <EOS>
input_ids:  c1 c2 c3 ... cN
target_ids: c2 c3 c4 ... <EOS>
```

The first implementation should use fixed sequence length `128`, ignore
`<PAD>` positions in the loss, keep `<EOS>` as a real target, and generate
until `<EOS>` or a maximum sample length.

Detailed design notes are in `DESIGN.md`.

## Baseline

Start with a simple character-level baseline:

- unigram or bigram sampling from the training corpus
- no neural hidden state
- evaluate with validation negative log likelihood or perplexity if practical
- compare generated text qualitatively against the neural model

## Model Target

Implement one small sequence model first:

- model type: `rnn` or `lstm`
- embedding dimension
- hidden size
- number of layers
- sequence length
- batch size
- learning rate
- seed

## Metrics

- train loss
- validation loss
- validation perplexity if stable
- runtime
- generated samples from fixed prompts or fixed random seed

## Planned Config Fields

- `run_name`
- `corpus_path`
- `model_name`
- `epochs`
- `batch_size`
- `sequence_length`
- `embedding_dim`
- `hidden_size`
- `num_layers`
- `learning_rate`
- `seed`
- `num_workers`
- `generate_num_chars`

## Falsification Conditions

- If batch shapes are unclear, do not train yet.
- If train loss decreases but generated text is only memorized fragments, record
  that limitation.
- If validation loss worsens while train loss improves, treat it as overfitting.
- If baseline generation is comparable to the neural model, do not overclaim the
  neural result.

## Expected Output

- config-driven character-level training script
- tokenizer/vocabulary and batch smoke tests
- one small recurrent-model run
- generated samples
- comparison against a simple baseline
- Week 5 lab note
