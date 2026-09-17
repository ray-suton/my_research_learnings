# Results Directory

The latest completed extension is the [twenty-epoch Fashion-MNIST CNN run](runs/fashion_mnist_cnn_seed42_20epochs_20260917/notes.md).
It retains twenty external checkpoint references, four invocation logs,
class-based real-image diagnostics, and 28 figures, including full
[real/generated panels](runs/fashion_mnist_cnn_seed42_20epochs_20260917/real_vs_generated.png),
[progression](runs/fashion_mnist_cnn_seed42_20epochs_20260917/audit_epoch_05_10_20_progression.png),
and [losses](runs/fashion_mnist_cnn_seed42_20epochs_20260917/audit_d_g_losses.png).

The latest comparison is the completed
[twenty-epoch CNN run](runs/session4_mnist_cnn_seed42_20epochs_20260916/notes.md)
against the verified existing twenty-epoch MLP. It retains twenty external
CNN checkpoint references, twenty-one monitoring grids, final panels, curves,
audits, recovery provenance, and three comparison figures. The prior runs
below remain intact.

The completed baseline is [runs/session3_mnist_mlp_seed42_20260915](runs/session3_mnist_mlp_seed42_20260915/notes.md).

The completed five-epoch CNN comparison is
[runs/session4_mnist_cnn_seed42_20260916](runs/session4_mnist_cnn_seed42_20260916/notes.md).
It retains the same standard artifact set, five additional external
checkpoints, and a [full MLP/CNN comparison](runs/session4_mnist_cnn_seed42_20260916/mlp_vs_cnn_epoch05.png).
Its verification covers 25 checks and exact final-checkpoint sample replay.

Its retained artifacts are:

- config.json, metrics.json, stdout.txt, verification.txt, notes.md
- sources/: the five executed implementation/test source snapshots
- real_validation.png and fixed_noise_epoch_00.png through epoch_05.png
- final_noise_seed456.png and training_curves.png
- real_vs_generated.png and real_vs_generated_single.png
- discriminator_by_digit.csv

The real/fake panels show independent samples, not reconstructions. The
single pair uses index 0 from the declared validation/final-noise panels.
No examples were selected for being especially attractive.

Five checkpoints are stored outside Git under the data root; metrics.json
records their absolute paths and hashes. The final checkpoint reproduces both
generated panels exactly. Keep future attempts in new run directories and
retain failed results as well as successful executions.

See [RESULTS.md](../RESULTS.md) for interpretation and
[RUN_LOG.md](../RUN_LOG.md) for the executed commands.
