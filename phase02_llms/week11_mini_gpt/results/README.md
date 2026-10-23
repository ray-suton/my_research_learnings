# Results And Artifact Storage

No measured artifacts exist yet. For a meaningful run, retain a small dated
bundle here with resolved settings, source identities, loss curves, parameter
counts, timings, and a note explaining limitations.

This week produces trained weights. Checkpoints are **not** tracked in Git: they
live under `AI_ROADMAP_DATA`, following the [data policy](../../../DATA_PLAN.md),
with the reproduction command recorded in [RUN_LOG.md](../RUN_LOG.md) and the
registry note updated if a new artifact path is introduced.

Small loss-curve figures and sample text may be tracked when their purpose is
explicit and their generating settings are recorded. Preserve failed runs and
keep estimates separate from measurements.
