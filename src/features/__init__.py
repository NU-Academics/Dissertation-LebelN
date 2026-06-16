"""Feature engineering modules.

Phase 3 (Data Preparation) deliverables. Tiered feature construction per the
EDA-validated priority structure (Tier 1 historical/scheduling > Tier 2
slope/runtime > Tier 3 windowed utilization).

Planned submodules:
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
- episodes: scheduled-episode (per-attempt) regrain for Google Cluster Traces.
  Removes the instance-grain lifecycle-history leakage (V30) via strictly-prior
  history, with episode Tier 2/3 SQL builders and the modeling-stage
  per-instance negative cap + instance-keyed group split (V32).

Google Tier 1/2/3 modules were extracted from
``notebooks/10_feature_engineering_google.py`` after the feature logic was
validated against the working set (35,133,137 instances). Each module exposes
pure ``LazyFrame -> LazyFrame`` transforms; the Tier 2/3 modules additionally
expose BigQuery SQL builders capturing the validated at-scale production path.
The ``episodes`` module was extracted from notebooks 11b, 10 (Sections 11-12),
and 11 (Section 3.8) after the per-attempt redesign removed the leakage and met
the RQ1 >0.90 target at early-runtime on the episode matrix.
"""

from src.features.conflict_labels import (
    CONFLICT_TYPES,
    LABEL_COLUMN,
    META_COLUMNS,
    build_conflict_dataset,
    label_priority_inversion,
    label_resource_contention,
    label_scheduling_violations,
)
from src.features.episodes import (
    build_episode_history_sql,
    build_episode_intervals_sql,
    build_episode_runtime_features_sql,
    build_episode_tier3_features_sql,
    build_episode_usage_subset_sql,
    cap_negative_episodes,
    group_train_test_split,
    segment_episodes_polars,
)
from src.features.historical import (
    add_historical_features,
    add_lifecycle_position,
)
from src.features.runtime import (
    add_runtime_features,
    build_runtime_features_sql,
    build_usage_working_set_sql,
)
from src.features.sampling import (
    SamplingManifest,
    build_working_set_google,
    build_working_set_sql,
)
from src.features.scheduling import (
    add_hardware_counter_flag,
    add_scheduling_features,
    platform_suffix_map,
    priority_tier_expr,
)
from src.features.temporal import add_temporal_features
from src.features.utilization import (
    add_windowed_utilization,
    build_windowed_utilization_sql,
)

__all__ = [
    "add_historical_features",
    "add_lifecycle_position",
    "add_scheduling_features",
    "add_hardware_counter_flag",
    "priority_tier_expr",
    "platform_suffix_map",
    "add_temporal_features",
    "add_runtime_features",
    "build_usage_working_set_sql",
    "build_runtime_features_sql",
    "add_windowed_utilization",
    "build_windowed_utilization_sql",
    "label_resource_contention",
    "label_priority_inversion",
    "label_scheduling_violations",
    "build_conflict_dataset",
    "CONFLICT_TYPES",
    "LABEL_COLUMN",
    "META_COLUMNS",
    "build_episode_history_sql",
    "build_episode_intervals_sql",
    "build_episode_usage_subset_sql",
    "build_episode_runtime_features_sql",
    "build_episode_tier3_features_sql",
    "segment_episodes_polars",
    "cap_negative_episodes",
    "group_train_test_split",
    "build_working_set_google",
    "build_working_set_sql",
    "SamplingManifest",
]
