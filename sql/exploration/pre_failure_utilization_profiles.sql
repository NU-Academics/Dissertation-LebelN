-- pre_failure_utilization_profiles.sql
-- Resolves Open Question #4: Do failing instances exhibit detectably different
-- resource utilization behavior in the time window before the terminal event?
--
-- Rationale: EDA (notebook 03, Section 2.2) showed FAIL events have the highest
-- average resource *requests* among terminal types, but that snapshot is at the
-- moment of failure. RQ1's ensemble model needs to know whether the *trajectory*
-- of actual usage leading up to failure is distinctive — CPU bursts, memory ramps,
-- utilization-request gaps widening. If there is no signal in pre-failure windows,
-- the model must rely on static attributes (priority, scheduling class, platform).
--
-- Approach:
--   1. Identify terminal FAIL/LOST events and terminal FINISH events as two cohorts.
--   2. For each terminal event, look back into instance_usage for the same instance
--      in 3 time windows: 5 min, 15 min, and 60 min before the event.
--   3. Compute summary statistics of avg_cpu, max_cpu, avg_memory, max_memory in
--      each window, per cohort.
--   4. Also compute the utilization-to-request ratio (actual / requested) by joining
--      back to the SCHEDULE event's resource request.
--
-- Cost note: This joins two very large tables. The query uses a temporal sample
-- (days 10-12 of the trace period) to keep costs manageable. Adjust the date
-- window or use TABLESAMPLE if needed.
--
-- Replace YOUR-PROJECT-ID-HERE with the value from your GCP_PROJECT_ID Colab Secret.

-- ============================================================================
-- Part A: Pre-failure vs. pre-success utilization statistics by time window
-- ============================================================================

WITH trace_time_bounds AS (
    -- Google Cluster Traces v3 starts at 2019-05-01 00:00:00 PDT with a 600-second offset.
    -- Days 10-12 correspond roughly to timestamps for May 10-12, 2019.
    -- Timestamp in microseconds: (day_offset_seconds + 600) * 1_000_000
    SELECT
        CAST((9 * 86400 + 600) AS INT64) * 1000000 AS sample_start,  -- day 10 start
        CAST((12 * 86400 + 600) AS INT64) * 1000000 AS sample_end    -- day 12 end
),

-- Step 1: Terminal events in the sample window
terminal_events AS (
    SELECT
        collection_id,
        instance_index,
        machine_id,
        time AS terminal_time,
        type AS terminal_type,
        cpu_request,
        memory_request,
        CASE
            WHEN type IN (5, 8) THEN 'FAIL_LOST'
            WHEN type = 6 THEN 'FINISH'
        END AS outcome
    FROM `YOUR-PROJECT-ID-HERE.dissertation_lebel.instance_events_full`,
         trace_time_bounds
    WHERE type IN (5, 6, 8)            -- FAIL, FINISH, LOST
      AND time BETWEEN sample_start AND sample_end
      AND machine_id IS NOT NULL
      AND collection_id IS NOT NULL
),

-- Step 2: Join with instance_usage in 3 lookback windows
usage_with_window AS (
    SELECT
        te.collection_id,
        te.instance_index,
        te.machine_id,
        te.terminal_time,
        te.terminal_type,
        te.outcome,
        te.cpu_request,
        te.memory_request,
        u.start_time,
        u.avg_cpu,
        u.max_cpu,
        u.avg_memory,
        u.max_memory,
        -- Classify into lookback windows (not mutually exclusive — 5min is also in 15min and 60min)
        CASE
            WHEN u.start_time >= te.terminal_time - 5 * 60 * 1000000
                 THEN '5min'
        END AS in_5min,
        CASE
            WHEN u.start_time >= te.terminal_time - 15 * 60 * 1000000
                 THEN '15min'
        END AS in_15min,
        CASE
            WHEN u.start_time >= te.terminal_time - 60 * 60 * 1000000
                 THEN '60min'
        END AS in_60min
    FROM terminal_events te
    INNER JOIN `YOUR-PROJECT-ID-HERE.dissertation_lebel.instance_usage_full` u
        ON te.collection_id = u.collection_id
        AND te.instance_index = u.instance_index
        AND te.machine_id = u.machine_id
    WHERE u.start_time >= te.terminal_time - 60 * 60 * 1000000   -- 60-min lookback max
      AND u.start_time < te.terminal_time                        -- strictly before terminal
      AND u.avg_cpu IS NOT NULL
      AND u.avg_memory IS NOT NULL
)

-- Step 3: Aggregate by outcome and window
SELECT
    outcome,
    window_label,
    COUNT(*) AS n_usage_records,
    COUNT(DISTINCT CONCAT(CAST(collection_id AS STRING), '-', CAST(instance_index AS STRING))) AS n_instances,

    -- CPU utilization
    AVG(avg_cpu) AS mean_avg_cpu,
    STDDEV(avg_cpu) AS std_avg_cpu,
    APPROX_QUANTILES(avg_cpu, 100)[OFFSET(50)] AS median_avg_cpu,
    AVG(max_cpu) AS mean_max_cpu,

    -- Memory utilization
    AVG(avg_memory) AS mean_avg_memory,
    STDDEV(avg_memory) AS std_avg_memory,
    APPROX_QUANTILES(avg_memory, 100)[OFFSET(50)] AS median_avg_memory,
    AVG(max_memory) AS mean_max_memory,

    -- Utilization-to-request ratio (resource squeeze indicator)
    -- Ratio > 1.0 means usage exceeds request (contention signal)
    AVG(SAFE_DIVIDE(avg_cpu, cpu_request)) AS mean_cpu_util_ratio,
    STDDEV(SAFE_DIVIDE(avg_cpu, cpu_request)) AS std_cpu_util_ratio,
    APPROX_QUANTILES(SAFE_DIVIDE(avg_cpu, cpu_request), 100)[OFFSET(50)] AS median_cpu_util_ratio,
    APPROX_QUANTILES(SAFE_DIVIDE(avg_cpu, cpu_request), 100)[OFFSET(95)] AS p95_cpu_util_ratio,

    AVG(SAFE_DIVIDE(avg_memory, memory_request)) AS mean_mem_util_ratio,
    STDDEV(SAFE_DIVIDE(avg_memory, memory_request)) AS std_mem_util_ratio,
    APPROX_QUANTILES(SAFE_DIVIDE(avg_memory, memory_request), 100)[OFFSET(50)] AS median_mem_util_ratio,
    APPROX_QUANTILES(SAFE_DIVIDE(avg_memory, memory_request), 100)[OFFSET(95)] AS p95_mem_util_ratio

FROM usage_with_window
CROSS JOIN UNNEST([
    STRUCT('5min' AS window_label, in_5min AS flag),
    STRUCT('15min', in_15min),
    STRUCT('60min', in_60min)
]) AS w
WHERE w.flag IS NOT NULL
GROUP BY outcome, window_label
ORDER BY outcome, window_label;


-- ============================================================================
-- Part B: Rate-of-change (slope) of CPU/memory in the 15-minute pre-terminal window
-- ============================================================================
-- This measures whether utilization is *accelerating* before failure.
-- We compute the linear slope of avg_cpu and avg_memory over the 15-min window
-- per instance, then compare distributions between FAIL_LOST and FINISH.

WITH trace_time_bounds AS (
    SELECT
        CAST((9 * 86400 + 600) AS INT64) * 1000000 AS sample_start,
        CAST((12 * 86400 + 600) AS INT64) * 1000000 AS sample_end
),

terminal_events AS (
    SELECT
        collection_id,
        instance_index,
        machine_id,
        time AS terminal_time,
        CASE
            WHEN type IN (5, 8) THEN 'FAIL_LOST'
            WHEN type = 6 THEN 'FINISH'
        END AS outcome
    FROM `YOUR-PROJECT-ID-HERE.dissertation_lebel.instance_events_full`,
         trace_time_bounds
    WHERE type IN (5, 6, 8)
      AND time BETWEEN sample_start AND sample_end
      AND machine_id IS NOT NULL
      AND collection_id IS NOT NULL
),

-- Usage records in the 15-min window before terminal event
pre_terminal_usage AS (
    SELECT
        te.collection_id,
        te.instance_index,
        te.machine_id,
        te.terminal_time,
        te.outcome,
        u.start_time,
        u.avg_cpu,
        u.avg_memory,
        -- Normalize time to minutes-before-terminal for slope calculation
        (te.terminal_time - u.start_time) / 60000000.0 AS minutes_before_terminal
    FROM terminal_events te
    INNER JOIN `YOUR-PROJECT-ID-HERE.dissertation_lebel.instance_usage_full` u
        ON te.collection_id = u.collection_id
        AND te.instance_index = u.instance_index
        AND te.machine_id = u.machine_id
    WHERE u.start_time >= te.terminal_time - 15 * 60 * 1000000
      AND u.start_time < te.terminal_time
      AND u.avg_cpu IS NOT NULL
      AND u.avg_memory IS NOT NULL
),

-- Per-instance linear slope via least-squares: slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
-- x = minutes_before_terminal (higher = further in past), y = usage metric
-- Positive slope means usage *decreasing* toward terminal (further in past has higher value)
-- Negative slope means usage *increasing* toward terminal (ramp-up before event)
per_instance_slopes AS (
    SELECT
        outcome,
        collection_id,
        instance_index,
        machine_id,
        COUNT(*) AS n_points,
        -- CPU slope
        SAFE_DIVIDE(
            COUNT(*) * SUM(minutes_before_terminal * avg_cpu) - SUM(minutes_before_terminal) * SUM(avg_cpu),
            COUNT(*) * SUM(minutes_before_terminal * minutes_before_terminal) - SUM(minutes_before_terminal) * SUM(minutes_before_terminal)
        ) AS cpu_slope,
        -- Memory slope
        SAFE_DIVIDE(
            COUNT(*) * SUM(minutes_before_terminal * avg_memory) - SUM(minutes_before_terminal) * SUM(avg_memory),
            COUNT(*) * SUM(minutes_before_terminal * minutes_before_terminal) - SUM(minutes_before_terminal) * SUM(minutes_before_terminal)
        ) AS memory_slope
    FROM pre_terminal_usage
    GROUP BY outcome, collection_id, instance_index, machine_id
    HAVING COUNT(*) >= 3  -- Need at least 3 points for a meaningful slope
)

-- Aggregate slopes by outcome
SELECT
    outcome,
    COUNT(*) AS n_instances,

    -- CPU slope distribution
    AVG(cpu_slope) AS mean_cpu_slope,
    STDDEV(cpu_slope) AS std_cpu_slope,
    APPROX_QUANTILES(cpu_slope, 100)[OFFSET(25)] AS p25_cpu_slope,
    APPROX_QUANTILES(cpu_slope, 100)[OFFSET(50)] AS median_cpu_slope,
    APPROX_QUANTILES(cpu_slope, 100)[OFFSET(75)] AS p75_cpu_slope,

    -- Memory slope distribution
    AVG(memory_slope) AS mean_memory_slope,
    STDDEV(memory_slope) AS std_memory_slope,
    APPROX_QUANTILES(memory_slope, 100)[OFFSET(25)] AS p25_memory_slope,
    APPROX_QUANTILES(memory_slope, 100)[OFFSET(50)] AS median_memory_slope,
    APPROX_QUANTILES(memory_slope, 100)[OFFSET(75)] AS p75_memory_slope

FROM per_instance_slopes
GROUP BY outcome
ORDER BY outcome;
