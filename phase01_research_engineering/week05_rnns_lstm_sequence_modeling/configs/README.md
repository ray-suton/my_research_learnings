# Configs

This folder holds Week 5 sequence-model configs.

## Target Fields

Each config should make the training run explicit:

```yaml
run_name:
corpus_path:
model_name:
epochs:
batch_size:
sequence_length:
embedding_dim:
hidden_size:
num_layers:
learning_rate:
seed:
num_workers:
generate_num_chars:
```

## Recorded Configs

| Config | Purpose | Status |
| --- | --- | --- |
| `lstm_readme_tiny.yaml` | First measured README-corpus character LSTM run with fixed seed, sequence length 128, hidden size 64, and bigram baseline comparison. | passed |

## Rules

- Start with one tiny corpus and one simple model.
- Change one major variable per ablation.
- Keep generated samples reproducible with fixed seeds/prompts.
- Record the config path in `RUN_LOG.md` when a run actually happens.
