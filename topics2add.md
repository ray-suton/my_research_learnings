Your topic coverage is already broad. The main issue is not “missing all of
  AI”; it is that some 2025-2026 frontier topics are underweighted.

  Already Covered Well

  - MLPs, CNNs, ResNets
  - RNNs/LSTMs
  - Autoencoders, VAEs, GANs, diffusion
  - Transformers from scratch
  - BERT/T5/GPT-style models
  - LoRA, QLoRA, SFT, DPO
  - RAG, agents, test-time compute
  - Quantization, FlashAttention, MoE, Mamba
  - Interpretability basics
  - Vision, CLIP, SAM, VLMs
  - Graphs, audio, RL, world models, VLA/robotics
  - Capstone reproduction

  That is a strong skeleton.

  Topics I Would Add Or Strengthen

  1. Data-centric AI
      - dataset quality
      - deduplication
      - contamination
      - labeling noise
      - train/val/test leakage
      - synthetic data
      - data filtering

  2. Evaluation and benchmark design
      - benchmark contamination
      - custom eval sets
      - adversarial evals
      - failure taxonomies
      - confidence intervals
      - repeated runs
      - human vs automated eval

  3. Scaling laws and compute economics
      - model/data/compute scaling
      - Chinchilla-style tradeoffs
      - inference cost
      - GPU memory accounting
      - tokens per dollar
      - why small experiments may not extrapolate

  4. Modern post-training
     You mention DPO, but add more conceptual coverage of:
      - RLHF
      - RLAIF
      - RL with verifiable rewards / RLVR
      - process reward models
      - verifier models
      - rejection sampling
      - preference data quality

  5. Coding agents and AI for software engineering
     This is now a major frontier. Add:
      - code generation
      - repository-level agents
      - SWE-bench-style evaluation
      - test generation
      - program repair
      - tool-using coding loops

  6. Computer-use and GUI agents
     Separate this from generic agents:
      - browser agents
      - desktop/UI agents
      - action grounding
      - observation/action loops
      - tool failure recovery
      - sandboxing

  7. Long-context models and memory
     Add more than RAG:
      - long-context evaluation
      - attention vs retrieval
      - context compression
      - memory stores
      - lost-in-the-middle behavior
      - summarization memory

  8. AI safety and security
     This deserves its own recurring thread:
      - prompt injection
      - jailbreaks
      - data exfiltration
      - hallucination
      - eval gaming
      - model misuse
      - agent sandboxing
      - red-teaming

  9. AI systems / ML infrastructure
     You have some efficiency topics, but add:
      - torch.profiler
      - mixed precision
      - checkpoint/resume
      - SLURM job recovery
      - dataloader bottlenecks
      - experiment tracking
      - inference benchmarking

  10. AI for science
     Worth at least one focused week or capstone option:

  - protein/genomics models
  - materials discovery
  - molecule generation
  - weather/climate models
  - theorem proving
  - algorithm discovery

  11. Causal ML and uncertainty
     Not central to frontier LLMs, but important for research maturity:

  - causality vs correlation
  - uncertainty calibration
  - Bayesian intuition
  - conformal prediction
  - OOD detection

  12. Recommender systems / ranking
     Huge in industry and underrepresented:

  - embeddings
  - retrieval + ranking
  - two-tower models
  - learning-to-rank
  - feedback loops
  - bandits

  What I Would Not Do

  Do not add 20 more weeks. The roadmap is already full. Instead, add these as
  recurring overlays:

  - Monthly evaluation/reproducibility checkpoint
  - Monthly paper-summary note
  - Monthly “failure analysis” note
  - One safety/security mini-lab per phase
  - One systems/profiling mini-lab per phase
  - Capstone candidate list starting around Week 20

  Highest-Value Additions

  If you only add five topic upgrades, add these:

  1. Data-centric AI
  2. Evaluation/benchmark design
  3. Modern post-training: RLHF/RLAIF/RLVR/verifiers
  4. Coding agents and computer-use agents
  5. AI safety/security for agents and LLMs

  That would make the roadmap much more current and research-grade.