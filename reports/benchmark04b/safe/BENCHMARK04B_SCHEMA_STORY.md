# BENCHMARK04B — Frozen corrected feature schema

- Status: **PASS**
- Schema decision ID: `af33ab072e44c35c04b442a5`
- Current crosschecked features: 218
- Derived later: 1
- Frozen exclusions represented in schema: 150
- Semantically included features before scenario-specific support: 212
- Support threshold frozen: 50 distinct fit participants

## Interpretation

A06 directives override legacy semantics only where the CrossCheck explicitly changed a feature. Unchanged crosschecked features inherit their previously frozen semantic decision, including prior exclusions. Physical presence in crosschecked_inputs_v1 does not by itself imply semantic inclusion. BENCHMARK02 heuristic suggestions remain diagnostic only.

Support 50 is frozen here but is **not applied here**. BENCHMARK05b must recompute support independently for each scenario and temporal block using fit participants only; low-support columns are removed, never visits.

`AGE_AT_TARGET` is frozen as a numeric visit-derived covariate but is materialized only in BENCHMARK05b from LOCAL_ONLY birth metadata plus the current target visit date.

No individualized records or MRI values were read by this stage.