# Phase 3 Roadmap - Efficiency, Architecture Variants, and Interpretability

Parent roadmap: `../ai_research_roadmap_codex.md`

Phase goal: understand modern efficiency techniques, alternative sequence architectures, hybrid model designs, and interpretability workflows.

## Weeks

### Week 25 - FlashAttention and attention efficiency

Goal: understand why attention becomes expensive.

- Monday: Read FlashAttention conceptually.
- Tuesday: Profile mini-GPT attention memory/time as context length increases.
- Wednesday: Compare regular attention with optimized attention if available.
- Thursday: Run context-length benchmark.
- Friday: Write why attention is memory-bound.

Deliverable: attention efficiency benchmark.

### Week 26 - Quantization and memory

Goal: understand precision and memory.

- Monday: Learn fp32, fp16, bf16, int8, int4, activation memory, optimizer state.
- Tuesday: Load a small model in full precision and quantized form.
- Wednesday: Compare memory usage and output quality.
- Thursday: Run inference benchmarks by precision setting.
- Friday: Write training memory vs inference memory notes.

Deliverable: quantization comparison.

### Week 27 - Mixture of Experts

Goal: understand sparse expert routing.

- Monday: Read MoE and Mixtral concepts.
- Tuesday: Implement a toy MoE feed-forward layer.
- Wednesday: Add top-1 or top-2 routing and load-balancing diagnostics.
- Thursday: Compare dense FFN vs toy MoE on a small task.
- Friday: Write capacity vs compute notes.

Deliverable: toy MoE layer and comparison.

### Week 28 - Mamba and state-space alternatives

Goal: understand alternatives to dense attention.

- Monday: Read selective state-space model concepts.
- Tuesday: Use an existing small Mamba-style implementation or model.
- Wednesday: Compare with a tiny Transformer on a sequence task.
- Thursday: Benchmark longer sequence lengths.
- Friday: Write why alternatives to attention exist.

Deliverable: Transformer vs Mamba-style comparison.

### Week 29 - Hybrid architectures

Goal: understand why models combine Transformer, SSM, and MoE ideas.

- Monday: Read hybrid architecture concepts.
- Tuesday: Sketch possible hybrid block designs.
- Wednesday: Build a tiny hybrid toy model or block simulation.
- Thursday: Run one small synthetic sequence experiment.
- Friday: Write why hybrid architectures are attractive.

Deliverable: hybrid architecture note.

### Week 30 - Interpretability I: attention and activations

Goal: inspect model internals.

- Monday: Read Transformer Circuits-style interpretability introduction.
- Tuesday: Add hooks to mini-GPT to save activations and attention maps.
- Wednesday: Visualize attention heads on short sequences.
- Thursday: Try activation patching on a toy behavior.
- Friday: Write what attention maps can and cannot show.

Deliverable: activation-inspection notebook.

### Week 31 - Interpretability II: sparse autoencoders

Goal: learn feature discovery from activations.

- Monday: Read sparse autoencoder interpretability concepts.
- Tuesday: Collect hidden activations from a small model.
- Wednesday: Train a small sparse autoencoder on those activations.
- Thursday: Inspect top-activating tokens/examples for a few features.
- Friday: Write what makes a feature interpretable or not.

Deliverable: tiny SAE experiment.

### Week 32 - Infra consolidation

Goal: make the LLM pipeline reproducible.

- Monday: Review training, inference, evaluation, memory, HPC notes.
- Tuesday: Clean the LLM pipeline repo.
- Wednesday: Add README instructions to reproduce one experiment.
- Thursday: Rerun one experiment from scratch using only the README.
- Friday: Write phase retrospective.

Deliverable: reproducible LLM mini-pipeline.

## Phase Outputs

- FlashAttention or attention-memory benchmark.
- Quantization comparison.
- Toy MoE, Mamba-style, and hybrid architecture experiments.
- Activation-inspection and sparse-autoencoder interpretability notebooks.
- Reproducible LLM mini-pipeline.
