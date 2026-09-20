# Phase 3 Roadmap - Efficiency, Architecture Variants, and Interpretability

Parent roadmap: `../ai_research_roadmap_codex.md`

Scope note: this file is the Phase 3 slice of the Year 1 core roadmap. The root roadmap remains the source of truth for continual-learning extensions, especially systems/scaling, long-context/memory, and research-process upgrades.

Phase goal: understand modern efficiency techniques, alternative sequence architectures, hybrid model designs, and interpretability workflows.

Research-process overlay:

- Record hardware, runtime, memory, precision, sequence length, and batch size for every systems comparison.
- Keep profiler traces or summary tables when profiling is the point of the experiment.
- Separate quality changes from speed/memory changes.
- For interpretability work, record what the visualization can and cannot prove.

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

### Bridge after Week 28 - Spiking neural networks

Placement: complete this bridge after the Mamba/state-space topic and before Week 29. Added on 2026-09-20. Allow four additional learning sessions of roughly 60–90 minutes; preserve existing week numbers and extend the calendar as needed rather than adding these tasks to Week 28's existing workload. This is planned learning, not an implemented project.

Goal: understand networks of stateful spiking neurons and distinguish their dynamics and training from Mamba's selective state-space blocks. RNNs, temporal state, and basic gradient training are useful background.

- Session 1: Simulate one leaky integrate-and-fire neuron. Plot membrane potential and spikes; inspect leak, threshold, reset, and the effect of input timing. Explain the relationship between subthreshold integration and a simple state-space recurrence.
- Session 2: Study rate and timing/latency encoding, choose one explicit input/output convention, and define a tiny synthetic temporal-classification task. Record time bins, sequence length, readout, train/validation/test splits, and checks for quiet inputs, threshold crossing, and reset behavior.
- Session 3: Train a small simulated LIF network with surrogate gradients. Document the hard-spike forward computation and approximate backward derivative. Verify finite learning signals and actual parameter updates in a spiking layer; a conventional readout is allowed.
- Session 4: Compare with a small non-spiking RNN on the same temporal task. Report held-out accuracy, learning curves, firing activity, simulation steps, runtime, memory, failure cases, and what the comparison establishes.

Comparison contract: use the same encoded examples, split identities, and time horizon for both models. Keep parameter counts and training budgets comparable, record remaining differences, and choose settings on validation data. Record the encoding cost and simulated time steps; a sparse spike count alone does not establish lower energy use or faster execution on a GPU. Start with a tiny run and retain negative results.

Deliverable: neuron voltage/spike plots, a verified trainable SNN, a baseline comparison table, and a short SNN-versus-RNN/SSM explanation. Completion requires saved commands, settings, actual test/run evidence, and limitations.

Reading: [integrate-and-fire dynamics](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html) and [surrogate-gradient learning](https://arxiv.org/abs/1901.09948). Neuromorphic hardware, event-based sensing applications, and broader neuron/learning-rule families remain later extension options.

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
- SNN bridge after Week 28: neuron dynamics plots, surrogate-gradient training, and a temporal-task baseline comparison.
- Activation-inspection and sparse-autoencoder interpretability notebooks.
- Reproducible LLM mini-pipeline.

## Continual Extension Connections

Phase 3 is the main bridge into the systems-heavy extensions:

- AI systems, scaling, and ML infrastructure: `torch.profiler`, GPU memory accounting, mixed precision, checkpoint/resume, dataloader bottlenecks, tokens per dollar.
- Long-context models and memory systems: attention cost, context-length scaling, retrieval vs attention, cache and memory tradeoffs.
- Data-centric AI and evaluation design: repeated runs, confidence intervals, and failure categories for efficiency comparisons.
- Research methods and scientific writing: turn one systems benchmark into a mini-paper with clear limitations.
