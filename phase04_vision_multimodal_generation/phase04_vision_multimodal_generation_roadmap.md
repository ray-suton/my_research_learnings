# Phase 4 Roadmap - Vision, Multimodal Models, and Generation

Parent roadmap: `../ai_research_roadmap_codex.md`

Scope note: this file is the Phase 4 slice of the Year 1 core roadmap. The root roadmap remains the source of truth for continual-learning extensions and cross-cutting evaluation/safety/system standards.

Phase goal: move from classical vision to multimodal models and generative image modeling with both from-scratch and library-based experiments.

Research-process overlay:

- Define task metrics before running demos: accuracy, retrieval recall, VQA score, qualitative rubric, or failure taxonomy.
- Save representative successes and failures, not only sample images.
- Track dataset source, license, preprocessing, and whether generated outputs can be shared.
- For multimodal systems, record prompt format, image resolution, model version, and failure cases.

## Weeks

### Week 33 - CNN to Vision Transformer

Goal: compare image inductive biases.

- Monday: Read ViT concepts: image patches as tokens, Transformer encoder.
- Tuesday: Implement image patching and patch embeddings.
- Wednesday: Build a tiny ViT classifier.
- Thursday: Compare CNN, ResNet, and ViT on a small dataset.
- Friday: Write inductive bias notes.

Deliverable: CNN/ResNet/ViT comparison.

### Week 34 - Self-supervised vision

Goal: learn contrastive and self-supervised features.

- Monday: Learn SimCLR/DINO-style intuition.
- Tuesday: Implement a small contrastive image representation experiment.
- Wednesday: Train on a small unlabeled image subset.
- Thursday: Freeze features and train a linear classifier.
- Friday: Write supervised vs self-supervised feature notes.

Deliverable: contrastive representation experiment.

### Week 35 - CLIP-style image-text learning

Goal: connect image and text representations.

- Monday: Read CLIP concepts.
- Tuesday: Use pretrained CLIP for zero-shot image classification.
- Wednesday: Build a small image-text retrieval demo.
- Thursday: Evaluate retrieval on a small custom dataset.
- Friday: Write why language supervision improves flexibility.

Deliverable: CLIP retrieval demo.

### Week 36 - Segmentation and SAM-style models

Goal: understand segmentation foundation models.

- Monday: Read SAM concepts.
- Tuesday: Run a pretrained segmentation model on sample images.
- Wednesday: Compare point, box, and mask prompts if supported.
- Thursday: Build a small visual demo.
- Friday: Write classification vs detection vs segmentation notes.

Deliverable: segmentation demo.

### Week 37 - Multimodal LLMs and VLM architecture

Goal: understand visual features connected to LLMs.

- Monday: Read Flamingo/LLaVA-style concepts: vision encoder, connector, LLM, instruction tuning.
- Tuesday: Run a compact VLM for image captioning or VQA.
- Wednesday: Build a 20-image VQA evaluation set.
- Thursday: Compare VLM outputs with CLIP retrieval or image classifier outputs.
- Friday: Write VLMs vs pure vision models notes.

Deliverable: compact VLM evaluation.

### Week 38 - Small VLM adaptation

Goal: adapt a VLM or simulate multimodal adaptation.

- Monday: Learn visual instruction tuning and multimodal data format.
- Tuesday: Prepare a tiny image-question-answer dataset.
- Wednesday: Run LoRA-style VLM fine-tuning if possible; otherwise use frozen vision features plus a small head.
- Thursday: Compare before/after behavior.
- Friday: Write why multimodal fine-tuning is harder than text-only fine-tuning.

Deliverable: small VLM adaptation experiment.

### Week 39 - Diffusion models from scratch

Goal: implement a tiny DDPM.

- Monday: Read forward noising, reverse denoising, noise prediction, sampling loop.
- Tuesday: Implement forward noising.
- Wednesday: Train a tiny denoising network on small images.
- Thursday: Generate samples and run one ablation.
- Friday: Write diffusion vs VAE vs GAN comparison.

Deliverable: tiny DDPM.

### Week 40 - Diffusers and conditional generation

Goal: learn library-side diffusion workflows.

- Monday: Study Hugging Face Diffusers course/tutorials.
- Tuesday: Run a pretrained diffusion pipeline.
- Wednesday: Try conditioning: class labels, text prompt, or image-to-image.
- Thursday: Build or modify a small custom pipeline.
- Friday: Write what from-scratch DDPM taught that the library hides.

Deliverable: diffusion library demo.

## Phase Outputs

- CNN/ResNet/ViT comparison.
- Self-supervised vision and CLIP retrieval demos.
- Segmentation and compact VLM evaluation.
- Small VLM adaptation experiment.
- Tiny DDPM and diffusion pipeline demo.

## Continual Extension Connections

Phase 4 connects to several later samplers:

- Data-centric AI and evaluation design: image/text dataset quality, labeling noise, retrieval evaluation, custom VQA eval sets.
- AI safety, security, and robustness: multimodal hallucination, prompt/image injection, unsafe generation, distribution shift.
- AI systems, scaling, and infrastructure: inference cost, image resolution, batch size, memory use, pipeline latency.
- AI for science: vision and generative modeling are useful entry points for microscopy, molecules, materials, and climate imagery.
