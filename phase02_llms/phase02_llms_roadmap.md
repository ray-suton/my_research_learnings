# Phase 2 Roadmap - LLM Foundations and Full Pipeline

Parent roadmap: `../ai_research_roadmap_codex.md`

Scope note: this file is the Phase 2 slice of the Year 1 core roadmap. The root roadmap remains the source of truth for continual-learning extensions, including post-training, coding agents, long-context/memory, safety/evals, and systems tracks.

Phase goal: build the LLM pipeline from tokenization through pretraining, mid-training, post-training, evaluation, inference, RAG, test-time compute, and agents.

Research-process overlay:

- Define eval sets and metrics before comparing model variants.
- Track tokenization, data source, contamination risk, prompt set, seed, runtime, and memory.
- For LLM outputs, keep fixed prompts plus a failure-case table.
- Treat agents as systems above model inference, not as model classes.

## Weeks

### Week 9 - Tokenization and text data

Active project: [Week 9 workspace](week09_tokenization_text_data/README.md),
initialized on 2026-09-17. [Session 1](week09_tokenization_text_data/session1.md)
closed with guided corrections on 2026-09-21.
[Session 2](week09_tokenization_text_data/session2.md) closed on 2026-09-24 on a
working README-trained byte-level BPE prototype. Continue with
[Session 3](week09_tokenization_text_data/session3.md), executed on 2026-09-24:
WikiText-2 was fetched at a pinned revision and verified, the custom tokenizer
was refitted on the training subset only, and both tokenizers were compared on
identical validation text. The custom tokenizer needed about 55% more tokens
than the frozen GPT-2 reference.

**Deliverable redrawn 2026-09-28 and completed 2026-10-05.** [Session 5](week09_tokenization_text_data/session5.md) and the
[Week 9 lab note](week09_tokenization_text_data/LAB_NOTE.md) record the results,
limitations, reproduction steps, and handoff. The deliverable is now *a
training-only tokenizer and the per-split token streams it produces*: the
original "tokenizer and prepared LM dataset" line claimed the LM example builder,
which never fitted a Week 9 session. The tokenizer is complete and measured; the
stream half (`build_split_stream`) was implemented on 2026-10-05 with **9/9
fixture checks green**; the prepared LM dataset (`T + 1` windows, one-shifted
targets, both masks) is [Session 4](week09_tokenization_text_data/session4.md)
and closed under Week 10. See Week 10 below.

Goal: prepare language-modeling data correctly.

- Monday: Learn BPE, WordPiece, unigram tokenization, special tokens, padding, truncation, packing.
- Tuesday: Train a tiny tokenizer on a small corpus.
- Wednesday: Compare the custom tokenizer with a pretrained tokenizer.
- Thursday: Build the per-split token streams (one EOS per nonempty record) and the train/validation/test split boundary.
- Friday: Write how tokenization affects context length and model behavior.

Deliverable: a training-only tokenizer and the per-split token streams it produces.

### Week 10 - Transformer from scratch I: attention block

Goal: implement the core Transformer block.

Active project: [Week 10 workspace](week10_attention_block/README.md),
initialized on 2026-09-28 as a documentation scaffold - five-session plan, frozen
tensor/mask contract, and a Session 1 starter. Session 1 closed the same day as a
guided conceptual checkpoint, with the discussion record in
[session1.md](week10_attention_block/session1.md#discussion-record). Session 2
closed on 2026-09-29: [attention.py](week10_attention_block/attention.py) now
implements `stable_softmax`, both additive masks, scaled dot-product attention,
and the smoke run, and **15/15 fixture checks pass**. Multi-head attention
(Session 3) closed on 2026-09-30: `MultiHeadAttention` is implemented in
[multihead.py](week10_attention_block/multihead.py) and **9/9 fixture checks
pass**, including equivalence against an independently written per-head
reference. Session 4 - residuals, layer norm, MLP, and the assembled block -
closed on 2026-10-01 with **11/11 fixture checks passing** and a recorded smoke
run. **Session 5 closed the week on 2026-10-05:** it re-ran all four suites
(**15/15, 9/9, 11/11, 9/9**), closed the prepared LM dataset, and wrote
[LAB_NOTE.md](week10_attention_block/LAB_NOTE.md) and
[session5.md](week10_attention_block/session5.md). Nothing was trained, so no
quality claim is made.

Redrawn 2026-09-28: this week **owns the prepared LM dataset** - the LM example
half of [Session 4](week09_tokenization_text_data/session4.md), `T + 1` windows,
one-shifted targets, `attention_mask`, and `loss_mask` - beside the attention
block. Week 9's own remaining item, the per-split token stream
(`build_split_stream`), closed first, because the windows are built from that
stream, and this week's example half closed with it on 2026-10-05. Week 11
training is therefore unblocked. See
[lm_data.py](week09_tokenization_text_data/lm_data.py) for the implementation.

- Monday: Read self-attention, multi-head attention, positional encodings, residuals, FFN.
- Tuesday: Implement scaled dot-product attention and causal masking.
- Wednesday: Implement multi-head attention and shape tests.
- Thursday: Add residual connections, layer norm, and MLP block.
- Friday: Write a line-by-line explanation of the block.

Deliverable: tested Transformer block and the prepared LM dataset.

### Week 11 - Transformer from scratch II: mini-GPT

Goal: train a tiny decoder-only language model.

Active project: [Week 11 workspace](week11_mini_gpt/README.md), initialized on
2026-10-05 as a documentation scaffold - five-session plan, a decision table to
freeze in [Session 1](week11_mini_gpt/session1.md), and a proposed
model/loss/optimizer contract in the
[experiment plan](week11_mini_gpt/EXPERIMENT_PLAN.md). Nothing is implemented or
trained yet; every number there is a proposed default. Inputs are the Week 9
tokenizer and per-split streams and the Week 10 block and prepared LM dataset.

- Monday: Study embeddings, positional embeddings, decoder blocks, LM head, next-token loss.
- Tuesday: Implement mini-GPT using the Transformer block.
- Wednesday: Overfit on a tiny batch to verify correctness.
- Thursday: Train on a small corpus on HPC.
- Friday: Plot loss and generate samples.

Deliverable: mini-GPT trained from scratch.

### Week 12 - Sampling, decoding, and perplexity

Goal: understand generation behavior.

- Monday: Learn greedy, temperature, top-k, top-p, repetition penalty, beam search, perplexity.
- Tuesday: Implement greedy, temperature, top-k, and top-p sampling manually.
- Wednesday: Evaluate validation loss and perplexity.
- Thursday: Compare generations across decoding methods.
- Friday: Write why low loss does not always mean good generation.

Deliverable: decoding playground and comparison note.

### Week 13 - BERT-style masked language modeling

Goal: understand encoder-only Transformers.

- Monday: Read masked LM, bidirectionality, classification heads.
- Tuesday: Create a masked-token dataset.
- Wednesday: Implement a tiny encoder-only Transformer.
- Thursday: Train and inspect masked predictions.
- Friday: Write causal LM vs masked LM comparison.

Deliverable: tiny BERT-style model.

### Week 14 - T5 and encoder-decoder models

Goal: understand text-to-text modeling.

- Monday: Read encoder-decoder Transformers and task prefixes.
- Tuesday: Use a small pretrained T5 model for summarization or QA.
- Wednesday: Fine-tune on a tiny dataset.
- Thursday: Compare pretrained vs fine-tuned results.
- Friday: Write decoder-only vs encoder-only vs encoder-decoder notes.

Deliverable: small T5 fine-tuning experiment.

### Week 15 - Toy-scale LLM pretraining

Goal: understand what pretraining teaches and how a controlled data mixture affects a small model.

- Monday: Study data/compute scale and optimizer stability; declare a baseline and one data-mixture variant with fixed evaluation text.
- Tuesday: Clean two small training sources, check split leakage, and save source identities and sampling proportions.
- Wednesday: Train the two mini-GPT runs from identical initial weights with the same tokenizer, architecture, context length, optimizer, schedule, and processed-token budget.
- Thursday: Compare held-out loss by source and overall; record actual source-token counts, throughput, peak memory, and training time. Retain the context-length and model-width ablations as separately controlled comparisons.
- Friday: Write the result, failed hypotheses, and small-scale limitations.

Minimum comparison: change only source-mixture proportions in the primary pair. Freeze learning rate, warmup/decay, clipping, and batch settings before running; log them rather than attributing their effects to data. Match non-padding training tokens and record actual compute instead of claiming identical wall time. Start with a tiny smoke budget and extend sessions if needed; a negative result is valid.

Deliverable: toy pretraining report with a reproducible two-mixture comparison and explicit compute/data limits.

### Week 16 - Mid-training / continued pretraining

Goal: adapt a base LM while measuring domain gains and retention of general capability.

- Monday: Learn domain-adaptive/task-adaptive pretraining, forgetting, and general-data replay; declare four evaluation arms.
- Tuesday: Prepare separate domain and general training/held-out text. Freeze the replay ratio, tokenizer, optimizer, context length, and evaluation protocol.
- Wednesday: From the same base checkpoint, run general-only continuation, domain-only continuation, and domain-plus-general replay; retain the untouched base as the fourth arm.
- Thursday: Compare domain and general held-out loss/perplexity plus fixed prompts across all four arms; report gains, regressions, runtime, and actual domain/general tokens.
- Friday: Write the adaptation-versus-retention tradeoff, overfitting limits, and whether replay helped.

Match total non-padding training-token budgets and optimizer settings across the three continuation runs. The mixed arm sees fewer domain tokens at this budget; report that difference explicitly. Use a tiny common model/subset first, keep evaluation text out of all training sources, and do not claim that lower domain loss proves improved general capability.

Deliverable: domain-adaptation report with base/general-only/domain-only/replay results and a retention comparison.

### Week 17 - Supervised fine-tuning with LoRA

Goal: learn parameter-efficient instruction tuning and verify the exact text/targets used for learning.

- Monday: Learn LoRA/PEFT, chat templates, role boundaries, and assistant-only loss.
- Tuesday: Prepare a small instruction dataset with single-turn and multi-turn examples; inspect rendered text, token IDs, truncation/packing, and trainable label positions.
- Wednesday: Pass small data-contract checks, then run LoRA SFT on a small model with held-out conversations excluded from training.
- Thursday: Compare base vs SFT on fixed single-turn and multi-turn prompts; inspect formatting failures and capability regressions.
- Friday: Write pretraining vs SFT vs LoRA notes, including the template and masking decisions.

Required checks: prompt/user and padding targets are excluded from loss under the declared assistant-only policy; real assistant end markers remain valid targets; the model still receives the intended conversational context. Confirm the chosen trainer's label-shift convention, special tokens, and packing boundaries. Record and reject examples whose answer was entirely truncated. These are executable acceptance checks when the lesson is implemented, not passes inferred from configuration names.

Deliverable: LoRA adapter, inspected chat/label examples, passing data-contract checks, and a held-out comparison table.

### Week 18 - QLoRA and memory-efficient fine-tuning

Goal: adapt models under limited GPU memory.

- Monday: Learn 4-bit quantization plus LoRA adapters.
- Tuesday: Check exact GPU VRAM and choose model size.
- Wednesday: Run QLoRA smoke test.
- Thursday: Run one real small QLoRA job.
- Friday: Record GPU memory, batch size, sequence length, training speed.

Deliverable: QLoRA run log.

### Week 19 - Post-training and preference tuning with DPO

Goal: understand post-training as the stage after pretraining and mid-training, then run a small preference-tuning experiment.

- Monday: Learn the post-training stack: instruction SFT, reward modeling conceptually, RLHF/PPO conceptually, and DPO practically.
- Tuesday: Prepare prompt/chosen/rejected preference dataset.
- Wednesday: Run DPO on a small model or adapter.
- Thursday: Compare SFT-only vs SFT+DPO outputs.
- Friday: Write pretraining vs mid-training vs post-training notes, including what preference tuning changes and what it cannot guarantee.

Deliverable: post-training mini-report plus small DPO experiment.

### Week 20 - LLM evaluation

Goal: evaluate models repeatably.

- Monday: Learn perplexity, multiple-choice eval, exact match, GSM-style math, win rate, human eval.
- Tuesday: Install and run lm-evaluation-harness on a tiny model.
- Wednesday: Create a 20-50 prompt personal evaluation set.
- Thursday: Evaluate base, SFT, and DPO/QLoRA variants, including failure categories and contamination concerns.
- Friday: Write model/method/score/failure table and start `capstone_candidates.md` with at least five possible long-term project ideas.

Deliverable: LLM evaluation report plus first capstone candidate list.

### Week 21 - Inference, serving, and KV cache

Goal: understand LLM serving and performance.

- Monday: Learn prefill, decode, KV cache, batching, latency, throughput, memory.
- Tuesday: Serve a small model with vLLM or equivalent.
- Wednesday: Benchmark tokens/sec, latency, batch size, sequence length.
- Thursday: Compare Hugging Face generation with vLLM serving.
- Friday: Write why inference engineering differs from training.

Deliverable: inference benchmark.

### Week 22 - Retrieval-augmented generation

Goal: build a RAG system over personal notes or papers.

- Monday: Learn chunking, embeddings, vector search, reranking, context injection.
- Tuesday: Build a small document corpus from notes or papers.
- Wednesday: Implement retrieval and prompt construction.
- Thursday: Evaluate with and without retrieval.
- Friday: Write when RAG helps and fails.

Deliverable: RAG assistant over notes.

### Week 23 - Test-time compute: CoT, self-consistency, Tree of Thoughts

Goal: compare reasoning methods.

- Monday: Read chain-of-thought, self-consistency, and Tree of Thoughts conceptually.
- Tuesday: Build a small reasoning test set.
- Wednesday: Compare direct answer, CoT-style prompting, and self-consistency.
- Thursday: Implement a simple Tree-of-Thoughts-style search on one toy task.
- Friday: Write when extra test-time compute helps.

Deliverable: reasoning-method comparison.

### Week 24 - Agents and tool use

Goal: build a small tool-using loop.

- Monday: Read ReAct, tool-use concepts, and the idea of an agent as a system around model inference rather than a model itself.
- Tuesday: Build a tool-calling loop with calculator, local search, or Python execution.
- Wednesday: Add action/observation logging and a minimal short-term memory state.
- Thursday: Evaluate on 20 tasks requiring tools.
- Friday: Write agent failure modes and explain why agents sit above inference, tool use, test-time compute, memory, evaluation, and system design.

Deliverable: small tool-using agent.

## Agent Concept Note

An agent is not usually a new model class. In this roadmap, treat an agent as a system layer built on top of one or more model inference calls. The model predicts tokens or actions; the agent system decides how to wrap those calls with prompts, tools, retrieval, memory, test-time search, action/observation loops, guardrails, logging, and evaluation. A stronger model can make an agent better, but agent quality also depends on tool design, memory design, task decomposition, control flow, latency, reliability, and failure handling.

Agent stack to learn:

```text
model weights
  -> inference engine and decoding
  -> prompting and tool schemas
  -> test-time compute / search / reflection
  -> memory and retrieval
  -> planner-controller loop
  -> tools and environment actions
  -> logging, evals, safety checks, and system design
```

## Phase Outputs

- Tokenizer, prepared LM dataset, tested Transformer block, and mini-GPT.
- Controlled pretraining data-mixture and continued-pretraining retention/replay experiments.
- SFT, LoRA, QLoRA, DPO, and post-training comparison notes, with explicit chat-template and assistant-loss-mask checks.
- LLM evaluation report, inference benchmark, RAG system, test-time compute comparison, and tool-using agent.

## Continual Extension Connections

Phase 2 feeds directly into the highest-priority extension tracks:

- Modern post-training and agentic RL: RLHF, RLAIF, verifiers, rejection sampling, and tool-use rewards. Extension A now requires a response-distillation comparison and one actual small GRPO/RLVR model-training run; see the root roadmap for the shared acceptance contract.
- Coding agents and computer-use agents: repository agents, SWE-bench-style evals, program repair, browser/UI action loops.
- Long-context models and memory systems: long-context eval, attention vs retrieval, memory stores, lost-in-the-middle, summarization memory.
- AI safety, security, and robustness: prompt injection, jailbreaks, hallucination, RAG security, agent sandboxing.
- Data-centric AI and evaluation design: contamination, custom eval sets, failure taxonomies, human vs automated evals.
