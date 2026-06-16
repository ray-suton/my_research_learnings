# Phase 5 Roadmap - Graphs, Audio, RL, World Models, and Embodied AI

Parent roadmap: `../ai_research_roadmap_codex.md`

Phase goal: sample important non-text AI areas, then connect RL, imitation, world models, VLA systems, and agentic RL.

## Weeks

### Week 41 - Graph neural networks

Goal: learn message passing and graph attention.

- Monday: Read GCN/GAT concepts.
- Tuesday: Use PyTorch Geometric to load Cora/Citeseer or another node-classification dataset.
- Wednesday: Implement GCN or GAT.
- Thursday: Run hidden size, heads, dropout ablations.
- Friday: Write why graphs need different assumptions.

Deliverable: GAT node-classification project.

### Week 42 - Audio and speech sampler

Goal: experience audio modeling.

- Monday: Learn wav2vec 2.0 and Hugging Face audio tooling.
- Tuesday: Load a small speech dataset.
- Wednesday: Run inference with a pretrained ASR model.
- Thursday: Fine-tune or evaluate on a tiny subset if feasible.
- Friday: Write why audio differs from text/images.

Deliverable: small audio/speech experiment.

### Week 43 - RL basics

Goal: learn model-free RL basics.

- Monday: Review MDPs, policy, value function, Q-learning, policy gradients, actor-critic.
- Tuesday: Train a simple RL agent on CartPole or equivalent.
- Wednesday: Plot reward curves and debug instability.
- Thursday: Compare random, heuristic, and learned policies.
- Friday: Write why RL is harder to evaluate than supervised learning.

Deliverable: basic RL agent.

### Week 44 - World models

Goal: understand learned environment models.

- Monday: Read Dreamer-style world model concepts.
- Tuesday: Implement a tiny learned dynamics model or run a small existing example.
- Wednesday: Compare model-free vs model-based intuition.
- Thursday: Visualize predicted next states or latent transitions.
- Friday: Write what world model means in practice.

Deliverable: toy world-model experiment.

### Week 45 - Imitation learning and behavior cloning

Goal: learn offline policy learning from demonstrations.

- Monday: Learn behavior cloning, offline data, distribution shift, compounding errors.
- Tuesday: Create or load expert trajectories in a simple environment.
- Wednesday: Train a behavior-cloning policy.
- Thursday: Evaluate rollout success and failure cases.
- Friday: Write imitation learning vs RL notes.

Deliverable: behavior-cloning policy.

### Week 46 - VLA and robot foundation models

Goal: distinguish VLMs, VLAs, and robot policies.

- Monday: Read RT-2 and OpenVLA concepts.
- Tuesday: Study LeRobot data format or another robot dataset format.
- Wednesday: Load or inspect a robot dataset.
- Thursday: Run a small policy inference or dataset visualization notebook.
- Friday: Write VLM vs VLA vs policy-model notes.

Deliverable: embodied-AI survey note plus small data demo.

## Agentic RL Bridge

The roadmap separates ordinary RL, LLM agents, and embodied policies for clarity, but the modern research frontier increasingly connects them. Treat agentic RL as the study of optimizing interactive behavior over multi-step environments, tools, or tasks. This can include classical RL policies, RLHF-style language-model post-training, RL with verifiable rewards, tool-use rewards, self-play or synthetic task environments, and robot/VLA policies.

Small add-on project if time permits:

- Define a toy agent environment with clear rewards, such as calculator use, web-free document search, code execution, or a gridworld.
- Compare a prompt-only tool-using agent against a simple reward-shaped or rule-improved agent loop.
- Track success rate, tool calls, failure modes, and reward hacking.
- Write what is being optimized: model weights, prompts, policies, tool-selection rules, search budget, memory, or system control flow.

## Phase Outputs

- GAT node-classification project.
- Audio/speech sampler.
- Basic RL agent.
- Toy world-model or behavior-cloning project.
- Embodied-AI/VLA survey note plus small data demo.
- Agentic RL bridge note or toy tool-use reward experiment.
