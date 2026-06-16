# Phase 2 Roadmap - LLM Foundations and Full Pipeline

Parent roadmap: `../ai_research_roadmap_codex.md`

Phase goal: build the LLM pipeline from tokenization through pretraining, mid-training, post-training, evaluation, inference, RAG, test-time compute, and agents.

## Weeks

### Week 9 - Tokenization and text data

Goal: prepare language-modeling data correctly.

- Monday: Learn BPE, WordPiece, unigram tokenization, special tokens, padding, truncation, packing.
- Tuesday: Train a tiny tokenizer on a small corpus.
- Wednesday: Compare the custom tokenizer with a pretrained tokenizer.
- Thursday: Build tokenized chunks, attention masks, and train/val split.
- Friday: Write how tokenization affects context length and model behavior.

Deliverable: tokenizer and prepared LM dataset.

### Week 10 - Transformer from scratch I: attention block

Goal: implement the core Transformer block.

- Monday: Read self-attention, multi-head attention, positional encodings, residuals, FFN.
- Tuesday: Implement scaled dot-product attention and causal masking.
- Wednesday: Implement multi-head attention and shape tests.
- Thursday: Add residual connections, layer norm, and MLP block.
- Friday: Write a line-by-line explanation of the block.

Deliverable: tested Transformer block.

### Week 11 - Transformer from scratch II: mini-GPT

Goal: train a tiny decoder-only language model.

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

Goal: understand what pretraining teaches and what it cannot teach at small scale.

- Monday: Study data scale, compute scale, optimizer stability, and evaluation.
- Tuesday: Clean a larger small corpus.
- Wednesday: Train mini-GPT longer.
- Thursday: Run context-length and model-width ablations.
- Friday: Write toy pretraining limitations.

Deliverable: toy pretraining report.

### Week 16 - Mid-training / continued pretraining

Goal: adapt a base LM to a domain corpus.

- Monday: Learn domain-adaptive and task-adaptive pretraining.
- Tuesday: Choose domain corpus: AI abstracts, class notes, lab text, code docs.
- Wednesday: Continue pretraining a small pretrained model.
- Thursday: Compare base vs continued-pretrained model on domain prompts/perplexity.
- Friday: Write when continued pretraining helps or overfits.

Deliverable: domain-adapted small LM.

### Week 17 - Supervised fine-tuning with LoRA

Goal: learn parameter-efficient instruction tuning.

- Monday: Learn LoRA and PEFT.
- Tuesday: Prepare a small instruction dataset.
- Wednesday: Run LoRA SFT on a small model.
- Thursday: Compare base vs SFT outputs on fixed prompts.
- Friday: Write pretraining vs SFT vs LoRA notes.

Deliverable: LoRA adapter and comparison table.

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
- Thursday: Evaluate base, SFT, and DPO/QLoRA variants.
- Friday: Write model/method/score/failure table.

Deliverable: LLM evaluation report.

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
- Toy pretraining and continued-pretraining experiments.
- SFT, LoRA, QLoRA, DPO, and post-training comparison notes.
- LLM evaluation report, inference benchmark, RAG system, test-time compute comparison, and tool-using agent.
