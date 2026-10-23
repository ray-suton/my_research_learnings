# Phase 2 - LLM Foundations and Full Pipeline

## Current Checkpoint

Phase 2 started on **2026-09-17** after Phase 1's learning-work closure.
[Week 9: tokenization and text data](week09_tokenization_text_data/README.md)
has a five-session plan. [Session 1](week09_tokenization_text_data/session1.md)
closed on **2026-09-21** with guided corrections and documented assessment
limits. [Session 2](week09_tokenization_text_data/session2.md) closed on
**2026-09-24** on a working README-trained byte-level BPE prototype (vocabulary
651, PAD 0 / EOS 1, reload equivalence, all checks passing).
[Session 3](week09_tokenization_text_data/session3.md) then ran the first
measured Week 9 experiment on 2026-09-24: WikiText-2 was fetched at revision
`b08601e0`, a custom tokenizer was refitted on training text only (2,222 records)
to the full 2,048-token target, and both tokenizers encoded the same 2,461
validation records. The custom tokenizer used 382,916 tokens versus GPT-2's
247,289 (about 1.55x).

**Week 9 closed on 2026-10-05.** Its deliverable was redrawn on 2026-09-28 to
*a training-only tokenizer and the per-split token streams it produces*. The
tokenizer side was complete and measured in Session 3 (2,048 tokens, PAD 0 /
EOS 1; 382,916 custom tokens versus GPT-2's 247,289 on the same 2,461 validation
records). The stream half (`build_split_stream`) was implemented on 2026-10-05
and its fixture suite passes **9/9**. The
[Week 9 lab note](week09_tokenization_text_data/LAB_NOTE.md) records the results,
the context-length implications, the limitations, and the reproduction steps, and
[Session 5](week09_tokenization_text_data/session5.md) closed the week's
discussion on 2026-09-25.

**Week 10 closed on 2026-10-05.**
[Week 10: the attention block](week10_attention_block/README.md) was initialized
on 2026-09-28 with a five-session plan, a frozen tensor and mask contract, and a
Session 1 starter, and it owned the prepared LM dataset as its second
deliverable. Session 1 closed on 2026-09-28 as a guided conceptual checkpoint;
Session 2 on 2026-09-29 with **15/15** attention checks; Session 3 on 2026-09-30
with **9/9** multi-head checks; Session 4 on 2026-10-01 with **11/11** block
checks. Session 5 re-ran all four suites on 2026-10-05 - **15/15, 9/9, 11/11**,
and **9/9** on the prepared LM dataset - wrote the
[lab note](week10_attention_block/LAB_NOTE.md) and
[session record](week10_attention_block/session5.md), and handed off to Week 11.

**Week 11: mini-GPT was initialized on 2026-10-05.**
[Week 11: the mini-GPT](week11_mini_gpt/README.md) has a five-session plan, a
decision table to freeze in Session 1, and a proposed model/loss/optimizer
contract in its experiment plan. Nothing is implemented or trained yet; every
number there is a proposed default.

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
completed experiments. Use small budgets and flexible sessions; Week 11 -
the mini-GPT - is the active learning checkpoint.

## Working Pattern

Start with the concept and a small data/model contract, then implement, test,
measure, and write a lab note. The learner owns ordinary lesson implementation;
Codex supports explanation and review. Record guided understanding separately
from independently demonstrated behavior and executed experiments.

Use the existing [data policy](../DATA_PLAN.md). Keep corpus files, tokenizer
artifacts, and tokenized datasets under `AI_ROADMAP_DATA`; retain small run
summaries and provenance in the relevant week folder.
