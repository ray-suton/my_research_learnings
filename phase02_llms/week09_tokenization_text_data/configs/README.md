# Configuration Record

No executable configuration exists yet. Use the
[experiment plan](../EXPERIMENT_PLAN.md) as the proposed contract and record
resolved settings beside each actual run before reporting measurements.

Required fields:

- Run ID, source hashes, Python/library versions, hardware, and seed if used.
- Dataset ID/config/revision, fingerprints, cache path, split counts, fitting
  subset row IDs/hash, actual bytes, and selection order.
- Normalization, pre-tokenization, byte alphabet, BPE settings, target/actual
  vocabulary size, decoder policy, reserved tokens/IDs, and boundary insertion.
- Reference tokenizer ID/revision; validation text identity and metric definitions.
- Block length, stride, padding side, remainder policy, target-shift convention,
  and attention/loss mask semantics.
- External tokenizer/data output paths, checksums, and preparation commands.

Reuse the existing configuration style once the learner implementation is
chosen. Avoid maintaining a second independent copy of the same defaults.
