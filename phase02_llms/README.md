# Phase 2 - LLM Foundations and Full Pipeline

## Current Checkpoint

Phase 2 started on **2026-09-17** after Phase 1's learning-work closure.
[Week 9: tokenization and text data](week09_tokenization_text_data/README.md)
is initialized with documentation and a five-session plan. Its first learning
session is ready; implementation, data preparation, and experiments are pending.

## Learning Sequence

| Weeks | Focus |
| --- | --- |
| 9–12 | Tokenization, attention, mini-GPT, decoding, and perplexity |
| 13–16 | Masked language modeling, encoder-decoder models, pretraining, and domain adaptation |
| 17–20 | LoRA, QLoRA, preference tuning, and evaluation |
| 21–24 | Inference, retrieval, test-time computation, and tool-using agents |

The [Phase 2 roadmap](phase02_llms_roadmap.md) gives the full sequence; the
[root roadmap](../ai_research_roadmap_codex.md) remains the curriculum authority.
Weeks are learning units, not a claim that their original calendar dates apply.

## Accepted Training-Coverage Additions - 2026-09-20

| Placement | Required practical addition |
| --- | --- |
| Week 15 | Controlled data-mixture pretraining comparison with recorded token/compute budgets |
| Week 16 | Base/general-only/domain-only/replay comparison with retention measurements |
| Week 17 | Chat templates, multi-turn examples, assistant-only loss masks, and executable data checks |
| Extension A, Weeks 53–60 | Teacher-response distillation with a matched student baseline |
| Extension A, Weeks 53–60 | Actual GRPO/RLVR model or adapter updates with a tested verifier and held-out evaluation |

The first three are specified in the [Phase 2 roadmap](phase02_llms_roadmap.md).
The two advanced labs use Extension A's existing window in the
[root roadmap](../ai_research_roadmap_codex.md). They are planned work, not
completed experiments. Use small budgets and flexible sessions; Week 9
Session 1 remains the active learning checkpoint.

## Working Pattern

Start with the concept and a small data/model contract, then implement, test,
measure, and write a lab note. The learner owns ordinary lesson implementation;
Codex supports explanation and review. Record guided understanding separately
from independently demonstrated behavior and executed experiments.

Use the existing [data policy](../DATA_PLAN.md). Keep corpus files, tokenizer
artifacts, and tokenized datasets under `AI_ROADMAP_DATA`; retain small run
summaries and provenance in the relevant week folder.
