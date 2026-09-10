# Results

`completed_20260908/` contains the completed Session 3–5 experiment artifacts.
See `../RESULTS.md` for the numerical summary and `../session3.md`,
`../session4.md`, and `../session5.md` for interpretation.

- Small configuration and metrics JSONs, command text, CSVs, and PNGs stay here.
- The short `stdout.log` files are present locally but ignored by root Git rules.
- Model checkpoints are outside the repository at `/tmp/week07_vae_20260908/`;
  they can expire and can be regenerated from the saved commands.
- `session2_smoke_test.txt` and `experiment_checks.txt` contain verification
  output for the current source files, separate from measured model quality.

Do not overwrite completed experiments when trying a new setting. Run with a
new output directory and retain its actual configuration and command.
