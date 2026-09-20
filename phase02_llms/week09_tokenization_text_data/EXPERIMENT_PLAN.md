# Experiment Plan

## Status And Learning Question

Proposed on **2026-09-17**; nothing has been executed yet.

How does a small tokenizer fitted on training text compare with a fixed
pretrained tokenizer in sequence length, text preservation, and failure cases,
and can its outputs become reproducible next-token examples?

Expected outcome: the custom tokenizer represents held-out ordinary text
without unknown-byte failures and reloads consistently. Sequence lengths may
be longer than the pretrained reference. No superiority claim is required.

## Working Defaults To Freeze Before Execution

| Item | Proposed choice |
| --- | --- |
| Corpus | `Salesforce/wikitext`, `wikitext-2-raw-v1` |
| Corpus identity | Record resolved revision, dataset fingerprints, and hashes before fitting |
| Splits | Preserve upstream train/validation/test; never split randomly after chunking |
| Tiny fitting subset | Up to 1 MiB of UTF-8 training text, whole nonempty records in source order; record included row IDs and actual bytes |
| Custom tokenizer | Byte-level BPE, target vocabulary 2,048 including reserved tokens; complete 256-byte alphabet |
| Text policy | Preserve case/accents/whitespace; no automatic prefix space or text normalization |
| Special tokens | Distinct `<pad>` and `<eos>`; record assigned IDs and literal-string policy |
| Reference | Frozen `openai-community/gpt2` tokenizer; record immutable revision; tokenizer files only |
| Initial block length | `T = 128`, fixture batch size `B = 2` |
| Compute | CPU; first smoke fit should fit within one session; measure rather than promise runtime |

The official [WikiText dataset card](https://huggingface.co/datasets/Salesforce/wikitext)
identifies the corpus/configurations. Its rows must not automatically be treated
as complete articles. For the initial prototype, define a record as one nonempty
row and append one EOS to that record; label this as a row boundary convention.
Preserve official splits, audit duplicate nonempty text across splits, and
record any overlap without silently changing the benchmark membership.

The [GPT-2 tokenizer reference](https://huggingface.co/openai-community/gpt2)
has a different vocabulary and training history. This is a descriptive
comparison, not an isolated algorithm ablation. Freeze the reference; do not
fit it on this corpus or load model weights for Week 9.

## Data And Provenance

Use the root [data policy](../../DATA_PLAN.md) and
[dataset registry](../../data_registry/datasets.yaml). Raw caches, trained
tokenizers, and token arrays belong under `AI_ROADMAP_DATA`, with proposed
processed outputs in `processed/week09_tokenization_text_data/<run_id>/`.
Save small settings, source hashes, metrics, and artifact references locally.

Before execution, record actual cache access, package versions, corpus revision,
source/retained counts, fitting-subset selection, preprocessing, and license
notes for the retrieved version. A successful preparation-helper invocation
alone will not supply every required field. The current helper neither pins a
revision nor creates Week 9 token chunks; its dry-run writes metadata too.

## Measurement Contract

Use the same fixed validation rows for both tokenizers, without padding,
truncation, or automatically inserted special tokens during the comparison.
Report validation text identity and UTF-8 byte counts.

- Actual vocabulary size, saved tokenizer size, and ordinary-text round-trip checks.
- Total payload tokens and tokens per UTF-8 byte: total tokens / total source bytes.
- Median and 95th-percentile token length per record, with the row convention noted.
- Fraction of records longer than 128 tokens; this is a context-budget diagnostic.
- Unsupported/unknown outputs if applicable, and examples involving spacing,
  punctuation, accents, non-Latin text, emoji, and unusual words.
- Measured fitting/encoding time with hardware, package versions, and run settings.

Record an empty denominator as not applicable, never zero by convenience.
Smaller token counts are not language-model accuracy or comparable perplexity.
Any vocabulary-size change is a separately recorded run. Keep test text out
of fitting and tuning; use it only after the comparison protocol is fixed.

## LM Example Contract

For each split independently, encode nonempty records in stable order, append
one EOS per record, and form a continuous token stream. Packing may cross row
boundaries within that split. The simple causal model may attend to earlier
records in its block; we are not claiming document-isolated attention.

Take windows of `T + 1` stream tokens with stride `T`. This one-token overlap
preserves successive prediction pairs across windows. Right-pad a final short
window if it contains at least one valid input/target pair; log a lone trailing
token with no target. Carry remainders across preprocessing batches rather
than silently dropping each batch's tail.

| Quantity | Shape | Rule |
| --- | --- | --- |
| Source window | `(B, T + 1)` | Split-specific real tokens plus optional right padding |
| `input_ids` | `(B, T)` | Source positions `0:T`; integer vocabulary IDs |
| `targets` | `(B, T)` | Source positions `1:T+1`; invalid targets replaced by `-100` |
| `attention_mask` | `(B, T)` | 1 at actual input positions, 0 at padding |
| `loss_mask` | `(B, T)` | 1 only where both input and next-token target are real |

Real EOS targets remain trainable. Padding is identified by position/length,
with dedicated PAD/EOS IDs. Week 10 supplies causal attention separately.
These targets are already shifted for a future manual training loop; a model
API that shifts labels internally needs a different adapter, not a second shift.

## Failure Conditions And Completion

Stop before declaring a measured success if vocabulary fitting touches held-out
text, split identities mix, serialization changes outputs, claimed round trips
fail, masks hide real EOS targets, or chunk boundaries misalign targets.
Retain failed outcomes and settings in the run log.

Completion requires implemented and passing behavioral checks, a saved/reloadable
tokenizer, a comparison table, prepared split-specific LM data with provenance,
and a lab note. See [tests/README.md](tests/README.md) for planned checks.
