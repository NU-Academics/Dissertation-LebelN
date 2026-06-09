# `src/features/` — Tiered Feature Engineering and Working-Set Construction

**Status:** Planned for Phase 3 (Google) and (Backblaze). Modules are extracted after `notebooks/10_feature_engineering_google.py` and `notebooks/11_preprocessed_eda_google.py` validate the logic against the EDA-informed feature priority documented in `eda_decisions.csv` (`V13` for Google tier structure, `V19` for Backblaze feature engineering approach).

## Planned modules (Google)

- `historical.py` — Tier 1 historical signals derived from the lifecycle summary: `prior_fail_count`, `has_prior_fail`, `resubmission_count`, `prior_evict_count`, `first_resubmission`, `lifecycle_position`. Motivated by `V09` (rapid-onset failure model) and `V10` (resubmission history dominates).
- `scheduling.py` — Tier 1 scheduling and priority features: `priority_tier` (one-hot using the bands in `src/data/schemas.py`), `scheduling_class`, `platform_id`, `cpu_request`, `memory_request`, `request_ratio`. Motivated by `V07`.
- `temporal.py` — Tier 1 submit-time temporal features: `submit_hour_of_day`, `submit_day_of_week`, sin/cos cyclic encodings, `submit_is_business_hours_pdt`, `submit_is_weekend`. Derived from `submit_time` in the lifecycle summary using the PDT wall-clock convention. Motivated by `V26` (FAIL_LOST rate varies up to ~8x across hour buckets).
- `runtime.py` — Tier 2 early-runtime features: `cpu_slope_{5s,15s,30s}`, `memory_slope_*`, `initial_cpu_ramp`, `initial_memory_ramp`, `first_interval_util_ratio`, `cpi_value`, `mapi_value` (conditional on `has_hardware_counters_majority`). Motivated by `V12` (utilization inversion: rate of change carries the signal).
- `utilization.py` — Tier 3 windowed utilization features: `avg_cpu_5min` / `15min` / `60min`, `max_cpu_*`, `std_cpu_*`, and the memory variants. Included for ablation per `V13`; expected predictive value is low because the rapid-onset failure window (median 23s) is shorter than the aggregation windows.
- `sampling.py` — Collection-level working-set construction. `build_working_set_google(events_lf, target_size_M=75)` retains every collection containing at least one FAIL or LOST instance (full failure preservation) and stratifies successful collections by the joint `(priority_tier, scheduling_class)` distribution. Per the Phase 3 plan and `P01`.
- `conflict_labels.py` — Stub for Week 5 RQ2 conflict labeling; placed early so the module imports resolve.
- `episodes.py` — Scheduled-episode (per-attempt) regrain that removes the instance-grain lifecycle-history leakage (`V30`). SQL builders: `build_episode_history_sql` (segmentation + strictly-prior history via the `sched_seq` construction and an `UNBOUNDED PRECEDING AND 1 PRECEDING` window), `build_episode_intervals_sql`, `build_episode_usage_subset_sql` (interval-assigned Tier 2/3 usage), `build_episode_runtime_features_sql` (reuses `runtime.slope_sql`), `build_episode_tier3_features_sql` (reuses `utilization._windowed_agg_sql`). Pure-Polars: `segment_episodes_polars` (testable mirror), `cap_negative_episodes` and `group_train_test_split` (modeling-stage helpers, `V32`). Motivated by `V30`-`V33`; carries `V01`/`V08`/`V27` (label) and `V09`/`V12`/`V13` (mechanism, tiers). Validated in `notebooks/11b`, `10` Sections 11-12, and `11` Section 3.8.

## Planned modules (Backblaze)

- `backblaze_smart.py` — Per-drive sliding-window SMART features. Tier 1 primary attributes (`V14`), Tier 2 rolling statistics and rate-of-change (`V19`), Tier 3 drift-aware and era-gated features for SMART 187 and 188.

## Cross-cutting conventions

- Each module exposes one or more pure functions taking Polars LazyFrames and returning LazyFrames with the new columns appended. No I/O.
- Each function's docstring cites the `eda_decisions.csv` row that motivates the feature.
- The Tier 3 module exists specifically to confirm the V12 inversion finding under preprocessing. The Risk-of-regression guard lives in `src/data/validation.py::assert_tier3_inversion`.

Modules land here once the corresponding notebook (10 or 11 for Google, post-`07c` for Backblaze) validates the logic and the relevant V-rows are populated.
