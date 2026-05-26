"""Feature engineering modules.

Phase 3 (Data Preparation) deliverables. Tiered feature construction per the
EDA-validated priority structure (Tier 1 historical/scheduling > Tier 2
slope/runtime > Tier 3 windowed utilization).

Planned submodules (created during Weeks 3 and 7):
- historical: Tier 1 lifecycle-derived features for Google Cluster Traces
  (prior_fail_count, resubmission_count, has_prior_fail, prior_evict_count).
- scheduling: Tier 1 scheduling features (priority_tier, scheduling_class,
  platform_id, cpu_request, memory_request, queue_time).
- runtime: Tier 2 early-runtime features (cpu_slope, memory_slope,
  initial_cpu_ramp, CPI/MAPI value with availability indicators).
- utilization: Tier 3 windowed utilization features for ablation use only.
- backblaze_smart: Tier 1 SMART primary discriminators, Tier 2 rolling
  statistics and rate-of-change, Tier 3 drift-aware cohort features.
- sampling: collection-level and drive-day stratified samplers for working-set
  construction. Full failure preservation per the Ch. 3 Sampling Strategy.
- conflict_labels: RQ2 conflict-type labelers (resource contention, priority
  inversion, scheduling violations) and outcome labels.
"""
