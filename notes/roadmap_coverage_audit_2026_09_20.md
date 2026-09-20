# Roadmap Coverage Audit - Training Stages And Missing Architectures

Date: 2026-09-20. Scope: the current LLM curriculum, with related vision,
systems, and extension work identified separately. The learner accepted the
five training-stage priorities and then an SNN bridge after Week 28. The inventory below reflects
those additions as planned work; no experiment or learning assessment is
marked complete.

## Accepted Additions

| Placement | Added commitment |
| --- | --- |
| Week 15 | Controlled two-mixture pretraining experiment |
| Week 16 | Base/general-only/domain-only/replay retention comparison |
| Week 17 | Explicit chat-template, multi-turn, and assistant-loss-mask checks |
| Bridge after Week 28 | Four-session SNN introduction and temporal-task comparison |
| Extension A, Weeks 53–60 | Matched teacher-response distillation comparison |
| Extension A, Weeks 53–60 | Tested verifier plus an actual small GRPO/RLVR model-training run |

The first three are in the [Phase 2 roadmap](../phase02_llms/phase02_llms_roadmap.md);
the advanced labs are in Extension A of the [root roadmap](../ai_research_roadmap_codex.md).
The SNN bridge is mirrored in the [Phase 3 roadmap](../phase03_efficiency_interpretability/phase03_efficiency_interpretability_roadmap.md).
Existing week numbering is preserved; the bridge adds learning sessions before
Week 29 and can extend calendar time. Week 9 Session 1 remains active.

## Spiking Neural Networks

**Spiking neural networks (SNNs) are now included as a bridge after Week 28.**
They were absent at the initial audit; the learner accepted their addition
on 2026-09-20. Four planned sessions cover leaky integrate-and-fire dynamics,
spike encoding, surrogate-gradient training, and a small temporal-task
comparison with a non-spiking RNN. Nothing has been implemented or assessed.

The bridge makes the shared dynamical-state viewpoint explicit while teaching
spike thresholds, resets, and learning separately from Mamba. Neuromorphic
hardware and event-based sensing remain later extension options.

BERT is planned in Week 13, Mamba in Week 28 with hybrids in Week 29, and
image diffusion in Weeks 39–40. Image diffusion coverage does not establish
coverage of diffusion language-model training.

## Baseline Ratings Before The Accepted Additions

These are the original, pre-addition subjective ratings of **planned curriculum
coverage**, retained as an audit baseline rather than rescored for new
documentation. The reference is a broad small-scale LLM research education.
They are not measured percentages,
claims of frontier-scale training readiness, or ratings of the learner.
Week 9 is initialized; the LLM-stage experiments are still future work.

| Stage | Core curriculum / 10 | Breadth including extensions / 10 | Main reason |
| --- | ---: | ---: | --- |
| Pretraining | 7 | 8 | Strong tokenizer, causal-LM, masked-LM, and toy experiment foundation; modern data-mixture and training-systems depth is limited. |
| Mid-training | 4 | 5 | One domain-adaptation week; capability mixtures, retention, and actual context-extension training are weakly specified or absent. |
| Post-training | 6 | 8 | Concrete SFT/LoRA/QLoRA/DPO work; extensions name modern reward methods, but many do not promise model-weight training. |

An 8 here means broad planned exposure with gaps. A 9–10 would require a more
complete set of explicit practical comparisons and evaluation contracts.
Extension breadth can be high while implementation depth remains low. The
scores are not calculated by counting rows: a data pipeline and a minor loss
variant do not deserve equal weight.

## Stage Boundaries And Reading Basis

- **Pretraining:** learn broad representations/capabilities, usually from a
  large self-supervised corpus; includes objectives, data, optimization, and
  the systems needed to run the learning process.
- **Mid-training:** further develop or adapt a base model before final behavior
  tuning, for example domain adaptation, capability-focused mixtures, or
  context extension. The label is not standardized. SmolLM3 explicitly includes
  long-context and reasoning adaptation; Olmo 3 emphasizes targeted capability
  data. These examples are broader than our current Week 16.
  [SmolLM3 training recipe](https://huggingface.co/blog/smollm3),
  [Olmo 3 report](https://arxiv.org/html/2512.13961v1).
- **Post-training:** shape instruction-following, preferences, reasoning,
  safety, or interactive behavior using demonstrations, preferences, rewards,
  or teacher signals. SFT, DPO, online RL, and distillation are distinct families
  in the official [TRL taxonomy](https://huggingface.co/docs/trl/index).

The same objective or dataset type can appear at different stages. LoRA is a
parameter-update method; RLVR describes a reward source; PPO/GRPO are optimizers.
They are not mutually exclusive alternatives. Model merging is a related
checkpoint-combination operation, not necessarily a training objective.

Legend for the inventory:

- **Core:** an explicit practical project is scheduled, including accepted additions; execution is not implied.
- **Partial:** conceptual coverage, a related exercise, or an underspecified recipe.
- **Extension:** explicitly present in later tracks; the row states promised depth.
- **Gap:** no explicit coverage of the named technique/recipe was found.

This inventory groups representative mainstream techniques and selected
advanced families. It is not an exhaustive list of every published variant.

## 1. Pretraining Inventory

Local evidence: [Phase 2 roadmap](../phase02_llms/phase02_llms_roadmap.md),
[Week 9 contract](../phase02_llms/week09_tokenization_text_data/EXPERIMENT_PLAN.md),
[Phase 3 roadmap](../phase03_efficiency_interpretability/phase03_efficiency_interpretability_roadmap.md),
and extension tracks E/F in the [root roadmap](../ai_research_roadmap_codex.md).

| Technique family | Status | Exact extent / missing depth |
| --- | --- | --- |
| Tokenization: BPE, WordPiece, unigram; vocabulary tradeoffs | Core / partial | W9 proposes practical byte-level BPE; WordPiece/unigram are conceptual, not three implementations. |
| Data splits, token packing, next-token labels, padding/causal/loss masks | Core | W9–11; the initialized W9 contract is explicit. |
| Train-only tokenizer fitting, provenance, contamination checks | Core / extension | W9 includes source identities and exact cross-split text overlap auditing; extension E broadens evaluation contamination. |
| Corpus extraction, cleaning, filtering, exact/near deduplication | Partial / extension | W15 cleans a corpus; E covers filtering/deduplication. No full scalable ingestion or near-duplicate pipeline is promised. |
| Autoregressive next-token prediction | Core | W11 mini-GPT and W15 longer toy pretraining. |
| Masked language modeling | Core | W13 tiny BERT-style encoder. |
| Span corruption / denoising encoder-decoder pretraining | Partial | W14 uses/fine-tunes T5; it does not implement T5's pretraining objective. |
| Prefix-LM and mixed denoising objectives, e.g. UL2 | Gap | No explicit mixed-objective pretraining experiment. |
| Fill-in-the-middle (FIM) / code infilling | Gap | Code generation and coding agents later do not constitute infilling pretraining. |
| Replaced-token detection / ELECTRA; permutation objectives / XLNet | Gap | Alternative-objective electives are not listed. |
| Multi-token prediction | Gap | No objective predicting several future tokens is specified. |
| Diffusion language-model objectives | Gap | W39–40 concern image generation. |
| Source-mixture weighting, data selection, curriculum scheduling | Core / partial | W15 now requires a controlled two-mixture comparison. Automated mixture optimization and curriculum scheduling remain gaps. |
| Code/math/multilingual corpus balancing and sampling | Gap | No explicit pretraining mixture or language-balancing recipe. |
| Synthetic pretraining text, filtering, real/synthetic mixing | Extension / partial | E names synthetic data; no end-to-end synthesis-to-pretraining comparison is promised. |
| Optimizer, learning-rate schedule, regularization, gradient/batch stability | Core / partial | W4 supplies foundations; W15 now freezes/logs learning rate, warmup/decay, clipping, and batch settings. A broader LLM optimizer ablation is not required. |
| High-quality final-stage annealing and changing data mixtures | Gap | Not the same as ordinary learning-rate scheduling; no late-stage mixture comparison. |
| Scaling laws and compute-optimal parameter/token budgets | Partial / extension | W15 studies data/compute scale; F names Chinchilla tradeoffs, without a required scaling-law fit. |
| Mixed precision, gradient accumulation, profiling, checkpoint recovery | Extension | F promises small practical comparisons and recovery tests; Phase 1 provides related foundations. |
| Activation checkpointing and distributed training: DDP, FSDP/ZeRO, tensor/pipeline/context/expert parallelism | Gap | No explicit distributed-training curriculum; ordinary checkpoint/resume is a different topic. |
| Modern dense/MoE training recipes: RoPE, RMSNorm, SwiGLU, routing/load balance | Partial | W27 plans toy routing/load-balancing diagnostics; W28–29 cover Mamba/hybrids. RoPE, RMSNorm, and SwiGLU are not explicitly listed. |
| Contrastive/multimodal representation pretraining | Partial | W34 has a small vision contrastive project; W35 uses pretrained CLIP. Joint text-image or multimodal LLM pretraining is not promised. |

Reference distinctions: [UL2](https://arxiv.org/abs/2205.05131) introduces a
mixture of denoising objectives; [FIM](https://arxiv.org/abs/2207.14255) studies
infilling; [multi-token prediction](https://arxiv.org/abs/2404.19737) and
[masked diffusion LMs](https://arxiv.org/abs/2406.07524) are separate objective
families. [Chinchilla](https://arxiv.org/abs/2203.15556) motivates explicit
compute/data tradeoffs. [Llama 3](https://arxiv.org/html/2407.21783v3) documents
late annealing with changed data quality/mix; [FSDP documentation](https://huggingface.co/docs/transformers/fsdp)
illustrates training-state sharding, distinct from saving a restart checkpoint.

Assessment: the core teaches how to build and train a small LM and now includes
a controlled data-mixture comparison. Automated data selection, scaling-law
experiments, and larger training systems remain limited. Some missing objective
families are elective, not prerequisites for the Week 11 mini-GPT.

## 2. Mid-Training Inventory

Local evidence: [Week 16](../phase02_llms/phase02_llms_roadmap.md), with related
long-context evaluation and data/systems topics in root extension tracks C/E/F.
The original [DAPT/TAPT paper](https://arxiv.org/abs/2004.10964) distinguishes
domain-adaptive training from adaptation on task-associated unlabeled text.

| Technique family | Status | Exact extent / missing depth |
| --- | --- | --- |
| Continued pretraining on a domain corpus (DAPT) | Core | W16 adapts a small base LM and compares domain prompts/perplexity. |
| Task-adaptive pretraining on unlabeled task text (TAPT) | Partial | Explicit W16 reading topic; no separate DAPT-versus-TAPT experiment promised. |
| Targeted code/math/science capability mixtures | Partial | W16 allows a domain corpus; no staged multi-capability mixture or transfer comparison. |
| High-quality annealing, mixture transitions, microanneal data ablations | Partial / gap | W16 now provides a matched-budget domain/general replay comparison; final-stage annealing, staged mixture transitions, and microanneal dataset selection remain unspecified. |
| Reasoning-trace exposure or teacher-generated capability data | Gap | W23 is inference-time reasoning, not training on reasoning demonstrations. |
| Progressive context-length extension with positional adaptation | Gap | Context ablations and extension C evaluate context use; no long-context adaptation training. |
| Long-document mixtures, position/boundary treatment, short/long balance | Gap | W9 packs short blocks, but no long-context training data recipe is defined. |
| New-language adaptation and vocabulary/embedding expansion | Gap | Not explicitly scheduled. |
| Knowledge refresh / time-sensitive continued learning | Gap | Domain adaptation offers related intuition, but there is no temporal update protocol. |
| General-data replay/interleaving to reduce forgetting | Core | W16 now compares general-only, domain-only, and a fixed domain/general replay mixture against the untouched base. |
| Forgetting mitigation via regularization, distillation, selective updates | Gap | LoRA exists elsewhere; its role in retaining base capabilities is not tested. |
| Domain gain versus general-capability retention evaluation | Core | W16 now requires domain and general held-out loss/perplexity plus fixed prompts for all four arms. |
| Full-parameter versus adapter-based continued pretraining | Partial | CPT and LoRA are both taught, but no matched comparison connects them. |
| Modality-bridging training, e.g. vision-language connector alignment | Partial | W37–38 introduce VLMs/adaptation; no explicit projector-alignment stage. |

Modern examples justify treating mid-training more broadly than one domain
corpus: SmolLM3 includes context/reasoning adaptation, while Olmo 3 uses
targeted high-quality data mixtures and assesses candidate data through
short continuation experiments. This is a comparison of published recipes,
not a universal stage definition.
[SmolLM3](https://huggingface.co/blog/smollm3),
[Olmo 3](https://arxiv.org/html/2512.13961v1).

Assessment: this remains the narrowest stage, but the accepted W16 experiment
now measures both adaptation gains and general-capability retention, with
replay as a controlled intervention. Context-extension training remains a
distinct omission; RAG or memory evaluation does not fill it.

## 3. Post-Training Inventory

Local evidence: [Weeks 17–20](../phase02_llms/phase02_llms_roadmap.md),
[Phase 5 agentic-RL bridge](../phase05_embodied_rl/phase05_embodied_rl_roadmap.md),
and root extension tracks A/D/E. The method families below are distinct even
when one training pipeline combines them.

| Technique family | Status | Exact extent / missing depth |
| --- | --- | --- |
| Instruction supervised fine-tuning (SFT) | Core | W17 small instruction dataset and base-versus-SFT comparison. |
| Full-parameter SFT | Partial | T5 fine-tuning is planned, but update scope and a full-versus-PEFT comparison are not explicit. |
| LoRA and QLoRA | Core | W17–18 practical adapter and memory-efficient jobs. |
| Other PEFT: bottleneck adapters, prompt/prefix tuning, IA3, DoRA | Gap | Generic PEFT reading does not explicitly schedule these alternatives. |
| Chat templates, multi-turn formatting, assistant-only loss masking | Core | W17 now explicitly requires rendered-example inspection, mask/shift/truncation checks, and held-out multi-turn evaluation. |
| Synthetic instructions, self-instruct, filtering and teacher demonstrations | Partial / extension | A now requires filtered teacher answers and matched student response-distillation training. Self-instruct/new-instruction synthesis remains unspecified; E covers synthetic data more broadly. |
| DPO and chosen/rejected preference datasets | Core | W19 practical SFT-versus-SFT+DPO experiment. |
| Other preference objectives: IPO, KTO, ORPO, SimPO, APO | Gap | DPO is the only explicit practical preference objective. |
| Online/iterative preference optimization, online DPO | Gap | No iterative generation, relabeling, and policy-update experiment. |
| Learned scalar/outcome reward-model training | Partial | W19 discusses reward modeling conceptually; no learned reward-model fit. |
| RLHF using PPO | Partial | W19 and A are conceptual; no LLM PPO weight-update loop promised. |
| RLAIF / Constitutional AI | Extension / partial | A names RLAIF; explicit constitutional critique/revision training is absent. |
| RL with verifiable rewards (RLVR) | Extension | A now requires a tested verifier, actual GRPO model/adapter updates, a retained checkpoint, and held-out evaluation. |
| GRPO reasoning optimization | Extension | A now requires one bounded practical run from an SFT baseline; the task/model must pass a reward-signal and memory preflight. |
| Other policy-gradient methods: REINFORCE/RLOO, Dr. GRPO, DAPO, GSPO | Gap | No explicit comparative objective or implementation work. |
| KL regularization, clipping, reward normalization, entropy/length control | Partial | The A lab now monitors reward/length and KL when applicable; a comparative optimizer-stability study is still unspecified. |
| Process reward models and intermediate-step credit | Extension | A names process reward models; step-labeled reward training is not promised. |
| Outcome verification: exact answers, unit tests, structured-output rules | Extension | A explicitly proposes a tiny runnable verifier. |
| Best-of-N / rejection sampling | Extension | A proposes a comparison with SFT/DPO; whether accepted outputs become training data is unspecified. |
| Rejection-sampling fine-tuning / iterative self-training | Partial | Sampling is present, but a generate-filter-retrain loop is not explicitly required. |
| Reasoning-trace SFT / cold-start reasoning demonstrations | Gap | W23 prompting/search is not a training exercise. |
| Teacher-student distillation: response, logit, reasoning, on-policy | Extension / partial | A now requires response distillation on fixed teacher answers with a matched student baseline. Logit/on-policy methods and a dedicated reasoning-distillation comparison remain gaps. |
| Tool-call and multi-turn trajectory SFT | Gap | W24 builds a tool loop; it does not train the LM on tool trajectories. |
| Agentic RL with environment feedback and multi-step credit | Partial / extension | Phase 5/A offer toy reward environments; optimization may target prompts/rules rather than LM weights. |
| Safety/refusal fine-tuning, adversarial training, unlearning | Partial / gap | D includes safety/refusal evaluations; explicit model-update exercises and unlearning are absent. |
| Multimodal instruction/post-training | Partial | W38 proposes VLM LoRA if feasible, otherwise a frozen-feature/head alternative; no multimodal preference/RL requirement. |
| Model merging, checkpoint soups, adapter composition | Gap | No explicit merge-and-retention comparison; merging need not perform gradient updates. |
| Preference/eval quality, judge bias, held-out capabilities, regressions | Core / partial / extension | W20 and A/D/E provide evaluation foundations; judge calibration and stage-by-stage regression contracts need detail. |

Primary references for these distinctions:
[DPO](https://arxiv.org/abs/2305.18290) describes direct preference optimization;
[Constitutional AI](https://arxiv.org/abs/2212.08073) describes AI feedback;
[DeepSeek-R1](https://arxiv.org/abs/2501.12948) documents a staged reasoning
pipeline and distillation. The official
[SFT guide](https://huggingface.co/docs/trl/sft_trainer) makes chat/loss-mask
choices explicit, and [TRL](https://huggingface.co/docs/trl/index) distinguishes
offline preference, online reward optimization, and distillation trainers.
Its [GRPO guide](https://huggingface.co/docs/trl/v1.9.2/en/grpo_trainer) documents
several objective variants. [MT-Bench/Chatbot Arena](https://arxiv.org/abs/2306.05685)
studies LLM-judge limitations. These sources inform the technique inventory;
coverage judgments come from our local curriculum.

Assessment: practical SFT/DPO foundations now lead to explicit response-
distillation and GRPO/RLVR labs in Extension A. Execution is still pending;
the accepted training deliverable cannot be satisfied by a verifier alone,
and measured improvement is not guaranteed.

## Coverage That Should Not Be Double-Counted

- CoT prompting, self-consistency, and Tree of Thoughts are inference-time
  methods in W23; they do not imply reasoning SFT or reasoning RL.
- A tool-using agent loop is not tool-use model training.
- RAG, context retrieval, and memory stores do not perform continued pretraining.
- Extension C's long-context evaluation is not positional/context adaptation.
- Safety benchmarking does not establish safety fine-tuning coverage.
- A toy MoE layer is not an expert-parallel pretraining system.
- Ordinary checkpoint saving is not activation checkpointing or optimizer sharding.
- The curriculum's continual-learning extension tracks are not evidence of
  continual model-learning/forgetting-mitigation techniques.
- LoRA/QLoRA can be used across training stages; they are not separate objectives.

## Accepted Priority Improvements

The learner approved these five additions on 2026-09-20. Their placement and
acceptance conditions are now part of the root roadmap and the Phase 2 mirror.

1. **Strengthen Week 15's training recipe:** compare two training-source
   mixtures at matched token budgets with fixed model/optimizer settings;
   retain clean validation and actual compute/throughput/memory measurements.
2. **Strengthen Week 16's retention experiment:** compare the base model,
   general-text continuation, domain-only continuation, and domain/general
   replay under matched budgets. Evaluate both domain gains and general loss.
3. **Strengthen Week 17's SFT data mechanics:** inspect chat templates, trainable
   assistant spans, truncation/packing, and fixed multi-turn evaluation.
4. **Extension A: one reward-based model update:** a tiny verified task,
   SFT baseline, and one feasible GRPO/RLVR run with held-out accuracy,
   reward-hacking checks, token budget, and unsuccessful runs retained.
   The actual lesson includes a memory/reward-signal preflight and a model update.
5. **Extension A: response-distillation comparison:** a fixed teacher-generated
   dataset and matched student SFT baseline before considering on-policy methods.

Actual context-extension training is a useful next elective after basic
attention, positional encoding, and memory profiling are understood. The SNN
introduction is now an accepted bridge after Week 28. Deeper neuromorphic work,
alternative objective families, and multiple preference-loss variants remain
extension candidates while the main learning path proceeds.

## Evidence And Audit Limits

Read the root roadmap, Phase 2–6 roadmaps, active Week 9 records, and existing
status documents. A separate native agent mapped local coverage and false
positives. External method references are primary papers or official docs,
checked on 2026-09-20. Unlisted means no explicit curricular commitment found,
not that a topic cannot arise during later instruction.

The accepted changes update curriculum documentation only. No implementation,
training, dependency, dataset, or Git mutation was performed. Week 9 Session 1
remains the current learning checkpoint.
