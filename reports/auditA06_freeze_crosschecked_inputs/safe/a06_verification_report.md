# A06 cross-checked input verification

- Status: **PASS**
- Script version: `0.1.2`
- Upstream package: `frozen_experiment_v1_1`
- Temporal policy decision ID: `90fefe5ec34d49743612`
- Frozen ledger: `v1.0.0`
- Checks: 271 total / 0 failed

## Frozen invariants

- Participant split unchanged: **PASS**
- Participant counts: `{'train': 1987, 'validation': 426, 'test': 426}`
- Supervised population: `9600` visits / `2839` participants
- MRI targets: `323`
- Temporal policy unchanged: **PASS**
- Support threshold remains `50`

## Source-level changes

### ADAS
- Rows: 12542 → 12542
- Columns: 113 → 60
- Removed feature columns: 53
- Removed names: `NDREASON, Q10TASK, Q11TASK, Q12TASK, Q8WORD1, Q8WORD10, Q8WORD10R, Q8WORD11, Q8WORD11R, Q8WORD12, Q8WORD12R, Q8WORD13, Q8WORD13R, Q8WORD14, Q8WORD14R, Q8WORD15, Q8WORD15R, Q8WORD16, Q8WORD16R, Q8WORD17, Q8WORD17R, Q8WORD18, Q8WORD18R, Q8WORD19, Q8WORD19R, Q8WORD2, Q8WORD20, Q8WORD20R, Q8WORD21, Q8WORD21R, Q8WORD22, Q8WORD22R, Q8WORD23, Q8WORD23R, Q8WORD24, Q8WORD24R, Q8WORD3, Q8WORD3R, Q8WORD4, Q8WORD4R, Q8WORD5, Q8WORD5R, Q8WORD6, Q8WORD6R, Q8WORD7, Q8WORD7R, Q8WORD8, Q8WORD8R, Q8WORD9, Q8WORD9R, Q9TASK, SOURCE, WORDLIST`
- Added feature columns: `none`
- Output SHA-256: `f9f296465ee5e1790f9e95dc796e5dd434f844981515113ca797824ce5462d76`

### CDR
- Rows: 13074 → 13074
- Columns: 13 → 11
- Removed feature columns: 2
- Removed names: `CDSOURCE, CDVERSION`
- Added feature columns: `none`
- Output SHA-256: `536863c20a942784763d9690228d7c4a2c9d349d829794289dae3763d46c5d47`

### FAQ
- Rows: 12842 → 12842
- Columns: 15 → 14
- Removed feature columns: 1
- Removed names: `SOURCE`
- Added feature columns: `none`
- Output SHA-256: `1ff833844edb002090f29cc00ce1250251760da7843885f47109e429d8292de5`

### MMSE
- Rows: 12732 → 12732
- Columns: 47 → 32
- Removed feature columns: 16
- Removed names: `MMD, MML, MMLTR1, MMLTR2, MMLTR3, MMLTR4, MMLTR5, MMLTR6, MMLTR7, MMO, MMR, MMW, NDREASON, SOURCE, WORDLIST, WORLDSCORE`
- Added feature columns: `MM_WORLD_SCORE`
- Output SHA-256: `544f189026d3e3cb2e5eda04ed1dcf284f2a3dfeae893812f67328993399fd82`

### MOCA
- Rows: 8606 → 8606
- Columns: 47 → 46
- Removed feature columns: 1
- Removed names: `SOURCE`
- Added feature columns: `none`
- Output SHA-256: `2c9e25d255d86a94af169c9f5c62dce173819158a79b9f36d4ce7620059109da`

### NEUROBAT
- Rows: 15254 → 15254
- Columns: 72 → 63
- Removed feature columns: 9
- Removed names: `ANART, ANARTND, AVDELBEGAN, AVENDED, BNTND, LDELBEGIN, LIMMEND, LMSTORY, SOURCE`
- Added feature columns: `none`
- Output SHA-256: `bf76e097af4fcc58edafc2d629fa64e5f6d247adff3d340b560cbbe2932819ec`

### PTDEMOG
- Rows: 2772 → 2839
- Columns: 72 → 11
- Removed feature columns: 61
- Removed names: `PTADBEG, PTADDX, PTASIAN, PTBIRGR, PTBIRPL, PTBIRPR, PTBORN, PTCLANG, PTCOGBEG, PTDOB, PTDOBYY, PTENGSPK, PTENGSPKAGE, PTETHCATH, PTIDENT, PTIMMAGE, PTIMMWHY, PTLANGPR1, PTLANGPR2, PTLANGPR3, PTLANGPR4, PTLANGPR5, PTLANGPR6, PTLANGRD1, PTLANGRD2, PTLANGRD3, PTLANGRD4, PTLANGRD5, PTLANGRD6, PTLANGSP, PTLANGSP1, PTLANGSP2, PTLANGSP3, PTLANGSP4, PTLANGSP5, PTLANGSP6, PTLANGTTL, PTLANGUN1, PTLANGUN2, PTLANGUN3, PTLANGUN4, PTLANGUN5, PTLANGUN6, PTLANGWR, PTLANGWR1, PTLANGWR2, PTLANGWR3, PTLANGWR4, PTLANGWR5, PTLANGWR6, PTNLANG, PTOPI, PTORIENT, PTORIENTOT, PTRTYR, PTSOURCE, PTSPOTTIM, PTSPTIM, PTTLANG, PTWORK, PTWORKHS`
- Added feature columns: `none`
- Output SHA-256: `639c98ece1802264c664d24f7495172aa6e90f3a5f2b04a44c7524cfd5ceba21`

### APOERES
- Rows: 2588 → 2588
- Columns: 3 → 3
- Removed feature columns: 0
- Removed names: `none`
- Added feature columns: `none`
- Output SHA-256: `8f632fa47a1e29d5333a5ea7e9142b81b2e38a07029af421aef086493695708f`

## Special semantic checks

### ADAS_UNABLE
- Q1UNABLE: `{'implicit_conducted_filled': 8789, 'explicit_conducted_preserved': 3606, 'documented_reason_preserved': 54, 'unknown_missing_preserved': 93}`
- Q2UNABLE: `{'implicit_conducted_filled': 8799, 'explicit_conducted_preserved': 3616, 'documented_reason_preserved': 34, 'unknown_missing_preserved': 93}`
- Q3UNABLE: `{'implicit_conducted_filled': 8795, 'explicit_conducted_preserved': 3612, 'documented_reason_preserved': 42, 'unknown_missing_preserved': 93}`
- Q4UNABLE: `{'implicit_conducted_filled': 8742, 'explicit_conducted_preserved': 3599, 'documented_reason_preserved': 108, 'unknown_missing_preserved': 93}`
- Q5UNABLE: `{'implicit_conducted_filled': 8797, 'explicit_conducted_preserved': 3613, 'documented_reason_preserved': 39, 'unknown_missing_preserved': 93}`
- Q6UNABLE: `{'implicit_conducted_filled': 8786, 'explicit_conducted_preserved': 3608, 'documented_reason_preserved': 55, 'unknown_missing_preserved': 93}`
- Q7UNABLE: `{'implicit_conducted_filled': 8794, 'explicit_conducted_preserved': 3611, 'documented_reason_preserved': 44, 'unknown_missing_preserved': 93}`
- Q8UNABLE: `{'implicit_conducted_filled': 8764, 'explicit_conducted_preserved': 3597, 'documented_reason_preserved': 88, 'unknown_missing_preserved': 93}`
- Q13UNABLE: `{'implicit_conducted_filled': 8730, 'explicit_conducted_preserved': 3586, 'documented_reason_preserved': 133, 'unknown_missing_preserved': 93}`

### MMSE_WORLD
- legacy_complete_rows: `8882`
- legacy_partial_rows: `0`
- new_score_rows: `3748`
- both_representation_rows: `0`
- representation_conflicts: `0`
- invalid_legacy_values: `0`
- invalid_new_values: `0`
- derived_nonmissing_rows: `12630`

### PTDEMOG_BASELINE
- official_rids_with_any_ptdemog_source_row: `2839`
- baseline_groups_evaluated: `2839`
- groups_with_multiple_rows_in_earliest_wave_or_date: `216`
- groups_refined_by_earliest_date_within_wave: `216`
- resolved_baseline_participants: `2839`
- unresolved_baseline_participants: `0`
- old_model_ready_participants: `2772`
- newly_available_participants_vs_old_model_ready: `67`
- no_longer_available_participants_vs_old_model_ready: `0`
- birth_year_metadata_available: `2839`
- birth_year_conflicts: `0`
- selected_reason_counts: `{'unique_temporal_candidate': 2839}`

### PTRACCAT
- changed_cells: `16`
- multi_response_cells_collapsed_to_common_more_than_one_category: `14`
- unexpected_code_cells: `0`

### AGE_AT_TARGET_PREREQUISITES
- participant_rows_in_age_metadata: `2839`
- participants_with_birth_year_candidate: `2839`
- birth_year_conflicts: `0`
- age_derivation_execution_stage: `BENCHMARK05`

### APOERES_DOMAIN
- allowed_domain_size: `6`
- unexpected_genotype_cells: `0`

## Frozen directive policy checks

- ADAS_Q9SCORE_Q12SCORE_categorical: **PASS**
- CDR_six_boxes_categorical: **PASS**
- CDR_CDGLOBAL_categorical: **PASS**
- CDR_CDRSB_numeric: **PASS**
- FAQ_items_official_map_then_categorical: **PASS**
- FAQTOTAL_retained_numeric: **PASS**
- MM_WORLD_SCORE_numeric: **PASS**
- MMSCORE_retained_numeric: **PASS**
- MOCA_DELW1_DELW5_categorical: **PASS**
- MOCA_total_retained_numeric: **PASS**
- NEUROBAT_summary_scores_retained_numeric: **PASS**
- PTDEMOG_common_core_types: **PASS**
- AGE_AT_TARGET_derive_later_numeric: **PASS**
- APOERES_GENOTYPE_categorical: **PASS**
- all_removed_columns_marked_excluded: **PASS**
- retained_current_features_exist: **PASS**

## AGE_AT_TARGET

A06 does **not** derive participant ages. It prepares and validates individualized birth-year metadata only in `local_only/`. The visit-specific `AGE_AT_TARGET` derivation remains frozen for BENCHMARK05, where the current target visit date is available. Raw birth fields are not predictors in the crosschecked PTDEMOG table.

## Privacy

This Markdown file is SAFE to share. It contains no participant identifier values, visit identifier values, individualized clinical dates, individualized clinical/cognitive/genetic values, predictions, or participant-level errors.

The sibling `local_only/` directory is **NOT SAFE TO SHARE** and may contain individualized audit details.
