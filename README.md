# AI Research Roadmap

This repo tracks my year-long AI research preparation roadmap.

Current learning checkpoint (2026-10-05): Phase 1 is complete from a
learning-work perspective, [Phase 2](phase02_llms/README.md) is active, and
**Weeks 9 and 10 are closed. Week 11 - the mini-GPT - is initialized and is the
active checkpoint.**

**Week 9** delivers *a training-only tokenizer and the per-split token streams it
produces*. The training-only WikiText-2 tokenizer (vocab 2,048, PAD 0 / EOS 1)
was measured in Session 3: on the same 2,461 validation records the custom
tokenizer used 382,916 tokens versus GPT-2's 247,289 (about 1.55x), with no
unknown tokens for either. The stream half, `build_split_stream` in
[lm_data.py](phase02_llms/week09_tokenization_text_data/lm_data.py), was
implemented on 2026-10-05 and its fixture suite passes **9/9**.

**Week 10** delivers the tested Transformer block and the prepared LM dataset.
The hand-written `stable_softmax`, additive causal and padding masks, scaled
dot-product attention, `MultiHeadAttention`, and the pre-norm residual block with
hand-written layer norm and GELU MLP pass **15/15, 9/9, and 11/11** fixture
checks. Session 5 re-ran every suite on 2026-10-05, closed the prepared LM
dataset at **9/9**, and wrote
[LAB_NOTE.md](phase02_llms/week10_attention_block/LAB_NOTE.md) and
[session5.md](phase02_llms/week10_attention_block/session5.md). No model has been
trained, so no quality claim is made.

**Week 11** was initialized on 2026-10-05 as a documentation scaffold with a
five-session plan and a decision table to freeze before coding; nothing is
implemented or trained yet.

Starting background:
- Completed Machine Learning Specialization
- Completed Deep Learning Specialization

Time commitment:
- 1.5 hours per weekday
- No planned weekend work

Hardware:
- One school HPC GPU

Data policy:
- Keep raw and processed datasets outside Git under `AI_ROADMAP_DATA`.
- Track dataset sources, licenses, and preparation commands in `DATA_PLAN.md` and `data_registry/datasets.yaml`.
