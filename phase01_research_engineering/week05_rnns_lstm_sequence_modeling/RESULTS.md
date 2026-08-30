# Results

Week 5 has one measured README-corpus character-level LSTM run and a simple
bigram baseline comparison. The run is intentionally tiny; it is a learning
exercise for sequence batching, LSTM hidden/cell state flow, generation, and
baseline comparison.

## Summary Table

| Run ID | Model | Seq Len | Hidden Size | Best Val Loss | Final Val Loss | Bigram Val Loss | Runtime | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `lstm_readme_tiny` | LSTM | 128 | 64 | 1.8677 | 1.9465 | 2.7505 | 2.328s | Best validation loss occurred at epoch 22; final epoch shows overfitting. |

Run artifact:

```text
results/runs/20260830_170359_lstm_readme_tiny/
  config.yaml
  metrics.json
  notes.md
  samples.txt
```

## Sample Comparison

Fixed prompts were generated with seed 7 and temperature 0.8.

LSTM samples:

```text
# -> ## Scope
- -> - `tets, the the text or forward sunity inting tiny and sequects outa lease training recistwing
S -> SIG`: taadion nabuts/`: smoke tests, recurrentth, tests, and sample RNN omplain the the fidath source or cherpeck modelgingth, melidden training before training
```

Bigram samples:

```text
# -> #:t )ma tercos1vAo_Pk le tsesSNPKsesterWNuCh tab Bk- tBymforsterLmOSUDhSksaten, f`s R
- -> - Stht_qw1Do5xhex3# Ck3sifKe dek6Ns,dOjjA4j,:kvtileaberainitpchy ow-d 521` d.
S -> Sen`:ptfautelWqa co)`HC.
```

## Interpretation

The LSTM learned noticeably more local README-style structure than the bigram
baseline. Its generated samples include Markdown-like starts and fragments such
as `## Scope`, `training`, and `RNN`, while the bigram samples are much more
fragmented.

The validation-loss comparison also supports the LSTM over the baseline on this
toy split: best LSTM validation loss 1.8677 versus bigram validation loss
2.7505.

The run should not be interpreted as a useful language model. The corpus has
only 98 sentence-level examples, and final validation loss worsened after epoch
22 while training loss kept improving.

## Limitations

- The corpus is the Week 5 README, so the task is intentionally small and
  stylistic.
- Vocabulary was built from the tiny local corpus; this does not test
  generalization to unseen text domains.
- Final test-set evaluation is not meaningful here because the corpus is too
  small for a robust train/validation/test split.
- No checkpointing is implemented, so the recorded final sample is from the
  final epoch, not the best validation epoch.
- The generated text remains noisy and partly memorized.
