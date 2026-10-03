# BENCHMARK02B — Aggregate input-feature profile

- Status: **PASS**
- Sources profiled: 8
- Candidate input features profiled: 218
- Snapshot-all anchor visits used for aggregate support: 9600
- Snapshot-all participants: 2839

## Scope

The profiler used the A06 crosschecked input overlay plus the unchanged supervised anchor from frozen_experiment_v1_1. It produced aggregate statistics only. It did not construct final model matrices, impute values, remove additional features, fit encoders, select a support threshold, or train models.

## Interpretation

- `native_train_*` columns describe canonical model-ready rows belonging only to training participants, including rows that may later serve as longitudinal history.
- `snapshot_train_*` columns describe support after alignment only to supervised target visits in the training split.
- Validation and test feature distributions are intentionally not profiled in this stage, so they cannot influence type decisions, encoding, or support thresholds.
- Type suggestions are heuristic diagnostics. Frozen A06 semantic directives take precedence whenever present.
- The support threshold remains frozen at 50 participants; BENCHMARK02 does not reselect it.
- No new automatic feature decision was made.

## Privacy

No RID, VISCODE2, individualized row, medical value, category value, minimum, maximum, mean, median, or example value was written to these reports. Feature profiling used training participants only.
