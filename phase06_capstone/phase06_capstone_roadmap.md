# Phase 6 Roadmap - Capstone Research Project

Parent roadmap: `../ai_research_roadmap_codex.md`

Scope note: this file is the Phase 6 slice of the Year 1 core roadmap. The root roadmap remains the source of truth for continual-learning extensions and open research cycles after Week 52.

Phase goal: choose one feasible research direction, reproduce a baseline and main method, run ablations, analyze failures, and package the result into a portfolio-quality report.

Research-process overlay:

- Start `capstone_candidates.md` in Week 20 and keep adding ideas before Phase 6 begins.
- Each candidate should include problem, importance, data, compute, baseline, metric, main risk, smallest reproduction, and possible extension.
- The final capstone should include hypothesis, baseline, method, ablations, failure analysis, limitations, and reproduction instructions.

## Weeks

### Week 47 - Choose a capstone area

Goal: choose a feasible small research reproduction.

- Monday: Review all Friday notes from the year.
- Tuesday: Choose three candidate papers/projects.
- Wednesday: Evaluate feasibility with GPU/time.
- Thursday: Choose one paper and define the smallest credible reproduction target.
- Friday: Write a capstone proposal.

Deliverable: capstone proposal.

### Week 48 - Reproduce baseline

Goal: get the baseline working.

- Monday: Read the chosen paper carefully.
- Tuesday: Set up baseline code and dataset.
- Wednesday: Run a tiny smoke test.
- Thursday: Run baseline experiment on HPC.
- Friday: Write baseline results and compare with paper numbers if available.

Deliverable: baseline reproduction.

### Week 49 - Reproduce main method

Goal: implement or adapt the core method.

- Monday: Identify the smallest version of the paper's main method.
- Tuesday: Implement or adapt the method.
- Wednesday: Debug on a tiny subset.
- Thursday: Run main experiment.
- Friday: Compare baseline vs method.

Deliverable: first reproduction attempt.

### Week 50 - Ablations

Goal: test which factors matter.

- Monday: Choose three ablations: model size, data size, objective, prompt, retrieval, routing, context length, or loss term.
- Tuesday: Run ablation 1.
- Wednesday: Run ablation 2.
- Thursday: Run ablation 3.
- Friday: Create results table and plot.

Deliverable: ablation report.

### Week 51 - Failure analysis and diagnostics

Goal: understand why the method succeeds or fails.

- Monday: Collect failure cases.
- Tuesday: Group failures by cause.
- Wednesday: Run one diagnostic experiment.
- Thursday: Write limitations honestly.
- Friday: Create final figures and tables.

Deliverable: failure-analysis report.

### Week 52 - Final report and portfolio packaging

Goal: produce a clean public-facing project.

- Monday: Write final report outline.
- Tuesday: Clean code and configs.
- Wednesday: Reproduce final key result.
- Thursday: Write README, setup instructions, and limitations.
- Friday: Package portfolio summary and next research questions.

Deliverable: final capstone repo and report.

## Phase Outputs

- Capstone proposal.
- Baseline and main-method reproduction.
- Ablation report.
- Failure-analysis report.
- Final portfolio-ready capstone repo and report.

## Continual Extension Connections

Phase 6 becomes the template for later open research cycles:

- Choose a paper, benchmark, or open problem.
- Reproduce the smallest credible baseline.
- Add one controlled modification.
- Run ablations and error analysis.
- Write a mini-paper.
- Ask for external critique.
- Decide whether to continue, pivot, or archive.

After the Year 1 capstone, use the extension tracks to decide whether the next deep cycle should focus on post-training/agentic RL, coding agents, long-context/memory, safety/evals, systems, data-centric AI, AI for science, or causal/ranking work.
