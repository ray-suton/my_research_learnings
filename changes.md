Most Important Missing Pieces



  1. A recurring research-methods layer

     Add a small monthly checkpoint for:

      - hypothesis before experiment

      - baseline

      - ablation

      - error analysis

      - limitations

      - what result would falsify the idea



  2. Paper-reading discipline

     Each paper note should include:

      - problem

      - method

      - key equation or algorithm

      - strongest empirical result

      - weakest assumption

      - one reproduction idea

      - one extension idea



  3. Evaluation standards

     Every project should define metrics before coding. Add:

      - train/val/test split rules

      - seed control

      - at least 3 repeated runs for important comparisons

      - runtime and memory reporting

      - failure-case table

      - “do not trust one lucky run” rule



  4. Experiment tracking

     Add a lightweight standard:



     results/

       runs/

         YYYYMMDD_HHMM_experiment_name/

           config.yaml

           metrics.json

           stdout.log

           notes.md

           plots/



  5. Testing and CI

     Add pytest and simple tests by Week 2 or Week 4:

      - shape tests

      - dataset tests

      - model forward-pass tests

      - config loading tests

      - smoke tests that run in under 30 seconds



  6. Pinned environments

     requirements.txt is fine for Week 1, but later add one of:

      - requirements.lock.txt

      - uv.lock

      - conda-lock

      - exact CUDA/PyTorch install notes



  7. Data governance and licensing

     Add rules for:

      - dataset source

      - license

      - preprocessing steps

      - whether data can be redistributed

      - where large files live

      - Hugging Face tokens / gated models / API keys



  8. AI safety, security, and evals deserve more weight

     Your roadmap mentions safety indirectly, but current frontier AI research

     makes this central. Add a dedicated recurring strand for:

      - hallucination evaluation

      - prompt injection

      - jailbreaks

      - data leakage

      - eval contamination

      - agent sandboxing

      - misuse risk

      - model behavior under distribution shift



  9. AI systems depth

     You have inference, KV cache, FlashAttention, quantization. Good. Also add:

      - profiling with torch.profiler

      - GPU memory accounting

      - dataloader bottlenecks

      - mixed precision

      - gradient accumulation

      - checkpoint/resume

      - failed-job recovery on SLURM



  10. Research writing cadence

     Add one monthly “mini paper”:



  Abstract

  Introduction

  Method

  Experiments

  Results

  Limitations

  Future work

  Reproduction instructions



  11. Mentor / external feedback loop

     The student should get outside critique at least monthly: professor, PhD

     student, open-source issue, Discord/Slack research group, or classmate

     review.



  12. Capstone selection starts too late

     Week 47 is late to choose. Start a capstone_candidates.md file around Week

     20 and keep adding ideas.



  One More Topic Gap



  I would add a small dedicated unit on AI for software engineering / coding

  agents, because it is now a major research and industry area: code generation,

  repository-level agents, program repair, SWE-bench-style evaluation, test

  generation, and tool-using coding loops.



  Verdict



  This is a strong roadmap and absolutely enough for Codex to implement Week 1.

  To make it research-grade, add recurring infrastructure around evaluation,

  paper notes, experiment tracking, safety, testing, and monthly synthesis. Do

  not add many more model families; add stronger research discipline around the

  ones already listed.
