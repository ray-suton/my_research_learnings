# Results And Artifact Storage

No measured artifacts exist yet. For a meaningful run, retain a small dated
bundle here with resolved settings, source identities, metrics, stdout/test
summaries, example segmentations, and a note explaining limitations.

Raw corpora, tokenizer files, token arrays, and caches belong outside Git under
`AI_ROADMAP_DATA`, following the [data policy](../../../DATA_PLAN.md). Store
their paths, hashes, and reproduction commands in the run bundle. Tiny
hand-authored fixtures may be tracked when their purpose is explicit.

Report actual vocabulary size and split/record/token counts. Preserve failed
runs and keep estimates separate from measurements. A directory or placeholder
table does not establish that fitting or verification succeeded.
