# Run Log

Use this file to record commands, settings, and outcomes. Do not record results
until the run has actually happened.

## Documentation And Setup

| Date | Task | Status | Notes |
| --- | --- | --- | --- |
| 2026-08-25 | Week 5 documentation scaffold | complete | Created README, experiment plan, run log, results placeholder, lab-note template, and scaffold docs for configs/tests/results. |
| 2026-08-26 | Session 1 design | complete | Chose the Week 5 README as the toy corpus, sentence-level examples with `<EOS>`, fixed sequence length 128, `<PAD>` ignored in loss, LSTM target, bigram baseline, and smoke-test requirements. |
| 2026-08-30 | Manual LSTM mini-lab and training infrastructure | complete | Added an inspectable one-step LSTM gate walkthrough, config validation, tokenizer/batching helpers, bigram baseline, config-driven training, and smoke tests. |

## Smoke Checks

| Date | Purpose | Command | Status | Notes |
| --- | --- | --- | --- | --- |
| 2026-08-30 | Python syntax check | `python3 -m py_compile phase01_research_engineering/week05_rnns_lstm_sequence_modeling/manual_lstm_block.py phase01_research_engineering/week05_rnns_lstm_sequence_modeling/corpus.py phase01_research_engineering/week05_rnns_lstm_sequence_modeling/tokenizer.py phase01_research_engineering/week05_rnns_lstm_sequence_modeling/model.py phase01_research_engineering/week05_rnns_lstm_sequence_modeling/baseline.py phase01_research_engineering/week05_rnns_lstm_sequence_modeling/config_utils.py phase01_research_engineering/week05_rnns_lstm_sequence_modeling/config_check.py phase01_research_engineering/week05_rnns_lstm_sequence_modeling/train.py phase01_research_engineering/week05_rnns_lstm_sequence_modeling/tests/test_manual_lstm_block.py phase01_research_engineering/week05_rnns_lstm_sequence_modeling/tests/test_tokenizer_and_corpus.py phase01_research_engineering/week05_rnns_lstm_sequence_modeling/tests/test_training_smoke.py` | passed | All Week 5 implementation and test Python files compile. |
| 2026-08-30 | Manual LSTM block check | `python3 phase01_research_engineering/week05_rnns_lstm_sequence_modeling/manual_lstm_block.py` | passed | Printed `z_t`, all four gates, updated `c_t`, updated `h_t`, and confirmed the update equations. |
| 2026-08-30 | Config check | `python3 phase01_research_engineering/week05_rnns_lstm_sequence_modeling/config_check.py --config phase01_research_engineering/week05_rnns_lstm_sequence_modeling/configs/lstm_readme_tiny.yaml` | passed | The first Week 5 measured-run config loaded and validated. |
| 2026-08-30 | Week 5 pytest smoke tests | `python3 -m pytest phase01_research_engineering/week05_rnns_lstm_sequence_modeling/tests -q` | passed | 9 tests passed. |

## Training Runs

| Run ID | Date | Command | Status | Key Result | Notes |
| --- | --- | --- | --- | --- | --- |
| `lstm_readme_tiny` | 2026-08-30 | `python3 phase01_research_engineering/week05_rnns_lstm_sequence_modeling/train.py --config phase01_research_engineering/week05_rnns_lstm_sequence_modeling/configs/lstm_readme_tiny.yaml` | passed | Best val loss 1.7582 at epoch 21; final val loss 1.9239; bigram val loss 2.6812. | Superseded by the later current-corpus run because README documentation updates changed the toy corpus text. Run record: `results/runs/20260830_170204_lstm_readme_tiny/`; CPU runtime was 2.127s. |
| `lstm_readme_tiny` | 2026-08-30 | `python3 phase01_research_engineering/week05_rnns_lstm_sequence_modeling/train.py --config phase01_research_engineering/week05_rnns_lstm_sequence_modeling/configs/lstm_readme_tiny.yaml` | passed | Best val loss 1.8677 at epoch 22; final val loss 1.9465; bigram val loss 2.7505. | Selected current-corpus run after final documentation updates. Run record: `results/runs/20260830_170359_lstm_readme_tiny/`; CPU runtime was 2.328s. |

## Issues

| Date | Issue | Evidence | Resolution |
| --- | --- | --- | --- |
| 2026-08-30 | Tiny README-corpus LSTM overfits after the best validation epoch | In the selected run, training loss kept falling to 0.6940 while validation loss rose from the best 1.8677 at epoch 22 to 1.9465 at epoch 34. | Recorded as a limitation; do not overclaim sample quality from this tiny corpus. |
