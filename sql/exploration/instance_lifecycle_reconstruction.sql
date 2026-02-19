-- instance_lifecycle_reconstruction.sql
-- Resolves Open Question #3: Can we reconstruct clean instance lifecycles, and do
-- lifecycle-derived features (queue time, running duration, resubmission count)
-- carry predictive signal for failure?
--
-- Rationale: EDA (notebook 03) profiled event types in isolation, but the RQ1 model
-- needs instance-level features: how long did an instance wait between SUBMIT and
-- SCHEDULE? How long did it run before terminating? How many times was it
-- resubmitted after eviction? The Google trace documentation notes that evicted/failed
-- instances get automatically resubmitted, so resubmission count is itself a
-- degradation signal. We need to know how clean these lifecycle sequences are and
-- whether the derived durations carry signal.
--
-- Approach:
--   1. For a temporal sample (days 10-12), group events by (collection_id, instance_index)
--      and reconstruct the event sequence.
--   2. Extract lifecycle features: time-to-schedule, running duration, resubmission count,
--      terminal event type.
--   3. Compute summary statistics comparing FAIL/LOST vs. FINISH lifecycles.
--
-- Cost note: Scoped to a 3-day temporal sample of instance_events_full only (no join
-- to the large instance_usage table). Moderate cost.
--
-- Replace YOUR-PROJECT-ID-HERE with the value from your GCP_PROJECT_ID Colab Secret.

-- ============================================================================
-- Part A: Event sequence patterns — what do instance lifecycles look like?
-- ============================================================================

WITH trace_time_bounds AS (
    SELECT
        CAST((9 * 86400 + 600) AS INT64) * 1000000 AS sample_start,
        CAST((12 * 86400 + 600) AS INT64) * 1000000 AS sample_end
),

-- All events for instances that have at least one event in the sample window
-- We include events *outside* the window for the same instance to capture
-- complete lifecycles (an instance may have been submitted before day 10 and
-- terminate during days 10-12).
sampled_instances AS (
    SELECT DISTINCT collection_id, instance_index
    FROM `YOUR-PROJECT-ID-HERE.dissertation_lebel.instance_events_full`,
         trace_time_bounds
    WHERE time BETWEEN sample_start AND sample_end
),

instance_events_sampled AS (
    SELECT
        ie.collection_id,
        ie.instance_index,
        ie.time,
        ie.type,
        ie.machine_id,
        ie.priority,
        ie.scheduling_class
    FROM `YOUR-PROJECT-ID-HERE.dissertation_lebel.instance_events_full` ie
    INNER JOIN sampled_instances si
        ON ie.collection_id = si.collection_id
        AND ie.instance_index = si.instance_index
),

-- Build the event sequence per instance as a string (e.g., "0→2→3→5")
event_sequences AS (
    SELECT
        collection_id,
        instance_index,
        STRING_AGG(CAST(type AS STRING), '→' ORDER BY time) AS event_sequence,
        COUNT(*) AS n_events,
        -- Terminal event = last event in the sequence
        ARRAY_AGG(type ORDER BY time DESC LIMIT 1)[OFFSET(0)] AS terminal_type,
        -- First and last timestamps
        MIN(time) AS first_event_time,
        MAX(time) AS last_event_time
    FROM instance_events_sampled
    GROUP BY collection_id, instance_index
)

-- Count the most common lifecycle patterns
SELECT
    event_sequence,
    terminal_type,
    COUNT(*) AS instance_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 4) AS pct_of_total
FROM event_sequences
GROUP BY event_sequence, terminal_type
ORDER BY instance_count DESC
LIMIT 50;


-- ============================================================================
-- Part B: Lifecycle duration features — queue time, running time, resubmission count
-- ============================================================================

WITH trace_time_bounds AS (
    SELECT
        CAST((9 * 86400 + 600) AS INT64) * 1000000 AS sample_start,
        CAST((12 * 86400 + 600) AS INT64) * 1000000 AS sample_end
),

sampled_instances AS (
    SELECT DISTINCT collection_id, instance_index
    FROM `YOUR-PROJECT-ID-HERE.dissertation_lebel.instance_events_full`,
         trace_time_bounds
    WHERE time BETWEEN sample_start AND sample_end
),

instance_events_sampled AS (
    SELECT
        ie.collection_id,
        ie.instance_index,
        ie.time,
        ie.type,
        ie.machine_id,
        ie.priority,
        ie.scheduling_class
    FROM `YOUR-PROJECT-ID-HERE.dissertation_lebel.instance_events_full` ie
    INNER JOIN sampled_instances si
        ON ie.collection_id = si.collection_id
        AND ie.instance_index = si.instance_index
),

-- Per-instance lifecycle features
lifecycle_features AS (
    SELECT
        collection_id,
        instance_index,
        -- Resubmission count: how many SUBMIT (type 0) events
        COUNTIF(type = 0) AS submit_count,
        -- Schedule count: how many SCHEDULE (type 3) events
        COUNTIF(type = 3) AS schedule_count,
        -- Eviction count: how many EVICT (type 4) events
        COUNTIF(type = 4) AS evict_count,
        -- Total events
        COUNT(*) AS total_events,
        -- Terminal type (last event)
        ARRAY_AGG(type ORDER BY time DESC LIMIT 1)[OFFSET(0)] AS terminal_type,
        -- First SUBMIT timestamp
        MIN(IF(type = 0, time, NULL)) AS first_submit_time,
        -- First SCHEDULE timestamp
        MIN(IF(type = 3, time, NULL)) AS first_schedule_time,
        -- Last SCHEDULE timestamp (the scheduling before terminal event)
        MAX(IF(type = 3, time, NULL)) AS last_schedule_time,
        -- Terminal event timestamp
        MAX(time) AS terminal_time,
        -- Priority and scheduling class (from first event)
        ARRAY_AGG(priority ORDER BY time LIMIT 1)[OFFSET(0)] AS priority,
        ARRAY_AGG(scheduling_class ORDER BY time LIMIT 1)[OFFSET(0)] AS scheduling_class
    FROM instance_events_sampled
    GROUP BY collection_id, instance_index
),

-- Compute derived durations
lifecycle_with_durations AS (
    SELECT
        *,
        -- Time-to-first-schedule: SUBMIT → first SCHEDULE (microseconds → seconds)
        SAFE_DIVIDE(first_schedule_time - first_submit_time, 1000000) AS queue_time_sec,
        -- Running duration: last SCHEDULE → terminal event (microseconds → seconds)
        SAFE_DIVIDE(terminal_time - last_schedule_time, 1000000) AS running_duration_sec,
        -- Total lifecycle: first event → terminal event
        SAFE_DIVIDE(terminal_time - first_submit_time, 1000000) AS total_lifecycle_sec,
        -- Resubmission flag (submit_count > 1 means at least one resubmission)
        IF(submit_count > 1, submit_count - 1, 0) AS resubmission_count,
        -- Outcome label
        CASE
            WHEN terminal_type IN (5, 8) THEN 'FAIL_LOST'
            WHEN terminal_type = 6 THEN 'FINISH'
            WHEN terminal_type = 4 THEN 'EVICT'
            WHEN terminal_type = 7 THEN 'KILL'
            ELSE 'OTHER'
        END AS outcome
    FROM lifecycle_features
    WHERE terminal_type IN (4, 5, 6, 7, 8)  -- Only instances with a terminal event
)

-- Aggregate lifecycle features by outcome
SELECT
    outcome,
    COUNT(*) AS n_instances,

    -- Lifecycle structure
    AVG(total_events) AS avg_total_events,
    AVG(submit_count) AS avg_submit_count,
    AVG(resubmission_count) AS avg_resubmission_count,
    AVG(evict_count) AS avg_evict_count,

    -- Fraction with resubmissions
    COUNTIF(resubmission_count > 0) AS n_resubmitted,
    ROUND(COUNTIF(resubmission_count > 0) * 100.0 / COUNT(*), 2) AS pct_resubmitted,

    -- Queue time (SUBMIT → SCHEDULE) in seconds
    AVG(queue_time_sec) AS mean_queue_time_sec,
    APPROX_QUANTILES(queue_time_sec, 100)[OFFSET(50)] AS median_queue_time_sec,
    APPROX_QUANTILES(queue_time_sec, 100)[OFFSET(95)] AS p95_queue_time_sec,

    -- Running duration (last SCHEDULE → terminal) in seconds
    AVG(running_duration_sec) AS mean_running_duration_sec,
    APPROX_QUANTILES(running_duration_sec, 100)[OFFSET(50)] AS median_running_duration_sec,
    APPROX_QUANTILES(running_duration_sec, 100)[OFFSET(5)] AS p5_running_duration_sec,
    APPROX_QUANTILES(running_duration_sec, 100)[OFFSET(25)] AS p25_running_duration_sec,
    APPROX_QUANTILES(running_duration_sec, 100)[OFFSET(75)] AS p75_running_duration_sec,
    APPROX_QUANTILES(running_duration_sec, 100)[OFFSET(95)] AS p95_running_duration_sec,

    -- Total lifecycle duration in seconds
    AVG(total_lifecycle_sec) AS mean_total_lifecycle_sec,
    APPROX_QUANTILES(total_lifecycle_sec, 100)[OFFSET(50)] AS median_total_lifecycle_sec

FROM lifecycle_with_durations
WHERE queue_time_sec >= 0       -- Exclude nonsensical negatives
  AND running_duration_sec >= 0
GROUP BY outcome
ORDER BY outcome;


-- ============================================================================
-- Part C: Resubmission count vs. eventual failure rate
-- ============================================================================
-- Does resubmission count predict eventual failure? If instances resubmitted 3+
-- times fail at significantly higher rates than single-pass instances, then
-- resubmission count is a strong predictor.

WITH trace_time_bounds AS (
    SELECT
        CAST((9 * 86400 + 600) AS INT64) * 1000000 AS sample_start,
        CAST((12 * 86400 + 600) AS INT64) * 1000000 AS sample_end
),

sampled_instances AS (
    SELECT DISTINCT collection_id, instance_index
    FROM `YOUR-PROJECT-ID-HERE.dissertation_lebel.instance_events_full`,
         trace_time_bounds
    WHERE time BETWEEN sample_start AND sample_end
),

instance_events_sampled AS (
    SELECT
        ie.collection_id,
        ie.instance_index,
        ie.time,
        ie.type
    FROM `YOUR-PROJECT-ID-HERE.dissertation_lebel.instance_events_full` ie
    INNER JOIN sampled_instances si
        ON ie.collection_id = si.collection_id
        AND ie.instance_index = si.instance_index
),

lifecycle AS (
    SELECT
        collection_id,
        instance_index,
        COUNTIF(type = 0) AS submit_count,
        ARRAY_AGG(type ORDER BY time DESC LIMIT 1)[OFFSET(0)] AS terminal_type
    FROM instance_events_sampled
    GROUP BY collection_id, instance_index
),

lifecycle_labeled AS (
    SELECT
        *,
        IF(submit_count > 1, submit_count - 1, 0) AS resubmission_count,
        CASE
            WHEN terminal_type IN (5, 8) THEN 1
            ELSE 0
        END AS is_failure
    FROM lifecycle
    WHERE terminal_type IN (4, 5, 6, 7, 8)
)

SELECT
    CASE
        WHEN resubmission_count = 0 THEN '0 (single pass)'
        WHEN resubmission_count = 1 THEN '1'
        WHEN resubmission_count = 2 THEN '2'
        WHEN resubmission_count BETWEEN 3 AND 5 THEN '3-5'
        WHEN resubmission_count BETWEEN 6 AND 10 THEN '6-10'
        ELSE '11+'
    END AS resubmission_bucket,
    COUNT(*) AS n_instances,
    SUM(is_failure) AS n_failures,
    ROUND(SUM(is_failure) * 100.0 / COUNT(*), 2) AS failure_rate_pct,
    AVG(submit_count) AS avg_submit_count
FROM lifecycle_labeled
GROUP BY resubmission_bucket
ORDER BY
    CASE resubmission_bucket
        WHEN '0 (single pass)' THEN 0
        WHEN '1' THEN 1
        WHEN '2' THEN 2
        WHEN '3-5' THEN 3
        WHEN '6-10' THEN 6
        ELSE 11
    END;


-- ============================================================================
-- Part D: Running duration distribution for FAIL vs. FINISH
-- ============================================================================
-- If FAIL instances cluster at short running durations (seconds rather than hours),
-- that suggests OOM/crash early in execution — a different predictive pattern than
-- slow degradation. This directly informs feature engineering window sizes.

WITH trace_time_bounds AS (
    SELECT
        CAST((9 * 86400 + 600) AS INT64) * 1000000 AS sample_start,
        CAST((12 * 86400 + 600) AS INT64) * 1000000 AS sample_end
),

sampled_instances AS (
    SELECT DISTINCT collection_id, instance_index
    FROM `YOUR-PROJECT-ID-HERE.dissertation_lebel.instance_events_full`,
         trace_time_bounds
    WHERE time BETWEEN sample_start AND sample_end
),

instance_events_sampled AS (
    SELECT
        ie.collection_id,
        ie.instance_index,
        ie.time,
        ie.type
    FROM `YOUR-PROJECT-ID-HERE.dissertation_lebel.instance_events_full` ie
    INNER JOIN sampled_instances si
        ON ie.collection_id = si.collection_id
        AND ie.instance_index = si.instance_index
),

lifecycle AS (
    SELECT
        collection_id,
        instance_index,
        ARRAY_AGG(type ORDER BY time DESC LIMIT 1)[OFFSET(0)] AS terminal_type,
        MAX(IF(type = 3, time, NULL)) AS last_schedule_time,
        MAX(time) AS terminal_time
    FROM instance_events_sampled
    GROUP BY collection_id, instance_index
),

durations AS (
    SELECT
        CASE
            WHEN terminal_type IN (5, 8) THEN 'FAIL_LOST'
            WHEN terminal_type = 6 THEN 'FINISH'
        END AS outcome,
        SAFE_DIVIDE(terminal_time - last_schedule_time, 1000000) AS running_duration_sec
    FROM lifecycle
    WHERE terminal_type IN (5, 6, 8)
      AND last_schedule_time IS NOT NULL
      AND terminal_time > last_schedule_time
)

-- Bucketed running duration distribution
SELECT
    outcome,
    CASE
        WHEN running_duration_sec < 1 THEN '<1s'
        WHEN running_duration_sec < 10 THEN '1-10s'
        WHEN running_duration_sec < 60 THEN '10s-1min'
        WHEN running_duration_sec < 300 THEN '1-5min'
        WHEN running_duration_sec < 900 THEN '5-15min'
        WHEN running_duration_sec < 3600 THEN '15min-1hr'
        WHEN running_duration_sec < 86400 THEN '1hr-1day'
        ELSE '>1day'
    END AS duration_bucket,
    COUNT(*) AS n_instances,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(PARTITION BY outcome), 2) AS pct_within_outcome
FROM durations
GROUP BY outcome, duration_bucket
ORDER BY
    outcome,
    CASE duration_bucket
        WHEN '<1s' THEN 0
        WHEN '1-10s' THEN 1
        WHEN '10s-1min' THEN 2
        WHEN '1-5min' THEN 3
        WHEN '5-15min' THEN 4
        WHEN '15min-1hr' THEN 5
        WHEN '1hr-1day' THEN 6
        ELSE 7
    END;
