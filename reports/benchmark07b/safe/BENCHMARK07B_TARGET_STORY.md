# BENCHMARK07B target representation story

Target decision ID: `a35b0846c0755c3196ff`.

The 323 frozen UCSFFSX7 targets are standardized with parameters fitted only on the corresponding training target group. PCA90 is then fitted only on those standardized training targets. Validation is transform-only and test target values are not parsed or materialized. Snapshot-matched and longitudinal share the same matched target transformation because their current target rows are identical.

- snapshot_all: train=6707, validation=1445, Full=323, PCA90=116 components, cumulative=0.900326.
- matched_pair: train=1419, validation=299, Full=323, PCA90=109 components, cumulative=0.900425.
