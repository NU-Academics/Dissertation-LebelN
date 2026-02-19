-- cpi_mapi_missingness_structure.sql
-- Resolves Open Question #6: Is the 20.5% CPI/MAPI missingness in instance_usage
-- driven by platform (hardware doesn't report counters), by workload characteristics
-- (scheduling class, priority), or by outcome (failing instances disproportionately
-- lack these metrics)?
--
-- Rationale: cycles_per_instruction and memory_accesses_per_instruction have identical
-- null counts (1,556,335,864 rows) — that identity strongly suggests the missingness
-- is systematic, not random. If platform-dependent, we can drop CPI/MAPI (platform_id
-- captures the same info). If outcome-dependent, missingness itself is a predictor and
-- should be encoded as a feature. If the non-null values also differ by outcome, then
-- CPI/MAPI become high-value predictors worth imputation effort.
--
-- Approach:
--   1. Join instance_usage with machine_events (for platform_id) on machine_id.
--   2. Compute CPI/MAPI null rates by platform_id.
--   3. Join instance_usage with instance_events to get the terminal outcome per instance.
--   4. Compute CPI/MAPI null rates by outcome (FAIL_LOST vs. FINISH).
--   5. For non-null CPI/MAPI records, compare distributions by outcome.
--   6. Cross-tabulate missingness x outcome for a chi-square contingency table.
--
-- Cost note: These queries join against instance_usage (1.5 TB). The platform query
-- uses BigQuery aggregation only (no data pulled). The outcome queries use TABLESAMPLE
-- to manage costs.
--
-- Replace YOUR-PROJECT-ID-HERE with the value from your GCP_PROJECT_ID Colab Secret.

-- ============================================================================
-- Part A: CPI/MAPI null rates by platform_id
-- ============================================================================
-- This determines whether missingness is hardware-driven.

SELECT
    m.platform_id,
    COUNT(*) AS n_usage_records,
    COUNTIF(u.cycles_per_instruction IS NULL) AS n_cpi_null,
    ROUND(COUNTIF(u.cycles_per_instruction IS NULL) * 100.0 / COUNT(*), 2) AS pct_cpi_null,
    COUNTIF(u.memory_accesses_per_instruction IS NULL) AS n_mapi_null,
    ROUND(COUNTIF(u.memory_accesses_per_instruction IS NULL) * 100.0 / COUNT(*), 2) AS pct_mapi_null,
    -- Non-null CPI/MAPI stats per platform
    AVG(u.cycles_per_instruction) AS mean_cpi,
    AVG(u.memory_accesses_per_instruction) AS mean_mapi,
    COUNT(DISTINCT u.machine_id) AS n_machines
FROM `YOUR-PROJECT-ID-HERE.dissertation_lebel.instance_usage_full` u
INNER JOIN (
    -- Get most recent platform_id per machine
    SELECT machine_id, platform_id
    FROM (
        SELECT
            machine_id,
            platform_id,
            ROW_NUMBER() OVER (PARTITION BY machine_id ORDER BY time DESC) AS rn
        FROM `YOUR-PROJECT-ID-HERE.dissertation_lebel.machine_events_full`
        WHERE platform_id IS NOT NULL
    )
    WHERE rn = 1
) m ON u.machine_id = m.machine_id
WHERE u.machine_id IS NOT NULL
GROUP BY m.platform_id
ORDER BY n_usage_records DESC;


-- ============================================================================
-- Part B: CPI/MAPI null rates by scheduling_class and priority tier
-- ============================================================================
-- This checks whether workload type affects CPI/MAPI reporting.
-- Uses a 5% TABLESAMPLE of instance_usage joined to instance_events for scheduling info.

WITH usage_sample AS (
    SELECT
        collection_id,
        instance_index,
        machine_id,
        cycles_per_instruction,
        memory_accesses_per_instruction
    FROM `YOUR-PROJECT-ID-HERE.dissertation_lebel.instance_usage_full`
    TABLESAMPLE SYSTEM (5 PERCENT)
    WHERE machine_id IS NOT NULL
),

-- Get scheduling info from the most recent SCHEDULE event per instance
sched_info AS (
    SELECT
        collection_id,
        instance_index,
        machine_id,
        scheduling_class,
        priority,
        ROW_NUMBER() OVER (
            PARTITION BY collection_id, instance_index, machine_id
            ORDER BY time DESC
        ) AS rn
    FROM `YOUR-PROJECT-ID-HERE.dissertation_lebel.instance_events_full`
    WHERE type = 3  -- SCHEDULE
      AND scheduling_class IS NOT NULL
)

SELECT
    s.scheduling_class,
    CASE
        WHEN s.priority <= 99 THEN 'Free (<=99)'
        WHEN s.priority BETWEEN 100 AND 115 THEN 'Best-effort (100-115)'
        WHEN s.priority BETWEEN 116 AND 119 THEN 'Mid-tier (116-119)'
        WHEN s.priority BETWEEN 120 AND 359 THEN 'Production (120-359)'
        WHEN s.priority >= 360 THEN 'Monitoring (>=360)'
        ELSE 'NULL'
    END AS priority_tier,
    COUNT(*) AS n_records,
    COUNTIF(u.cycles_per_instruction IS NULL) AS n_cpi_null,
    ROUND(COUNTIF(u.cycles_per_instruction IS NULL) * 100.0 / COUNT(*), 2) AS pct_cpi_null,
    AVG(u.cycles_per_instruction) AS mean_cpi_when_present,
    AVG(u.memory_accesses_per_instruction) AS mean_mapi_when_present
FROM usage_sample u
INNER JOIN sched_info s
    ON u.collection_id = s.collection_id
    AND u.instance_index = s.instance_index
    AND u.machine_id = s.machine_id
    AND s.rn = 1
GROUP BY s.scheduling_class, priority_tier
ORDER BY s.scheduling_class, priority_tier;


-- ============================================================================
-- Part C: CPI/MAPI null rates by instance outcome (FAIL_LOST vs. FINISH)
-- ============================================================================
-- This is the critical test: if missingness correlates with outcome, the null
-- pattern is itself a predictor.
-- Uses a 5% TABLESAMPLE of instance_usage, joined to terminal events.

WITH usage_sample AS (
    SELECT
        collection_id,
        instance_index,
        machine_id,
        start_time,
        cycles_per_instruction,
        memory_accesses_per_instruction
    FROM `YOUR-PROJECT-ID-HERE.dissertation_lebel.instance_usage_full`
    TABLESAMPLE SYSTEM (5 PERCENT)
    WHERE machine_id IS NOT NULL
),

-- Get terminal event per instance (most recent terminal event)
terminal_events AS (
    SELECT
        collection_id,
        instance_index,
        machine_id,
        type AS terminal_type,
        ROW_NUMBER() OVER (
            PARTITION BY collection_id, instance_index, machine_id
            ORDER BY time DESC
        ) AS rn
    FROM `YOUR-PROJECT-ID-HERE.dissertation_lebel.instance_events_full`
    WHERE type IN (4, 5, 6, 7, 8)  -- Terminal event types
),

labeled_usage AS (
    SELECT
        u.cycles_per_instruction,
        u.memory_accesses_per_instruction,
        CASE
            WHEN te.terminal_type IN (5, 8) THEN 'FAIL_LOST'
            WHEN te.terminal_type = 6 THEN 'FINISH'
            WHEN te.terminal_type = 4 THEN 'EVICT'
            WHEN te.terminal_type = 7 THEN 'KILL'
        END AS outcome
    FROM usage_sample u
    INNER JOIN terminal_events te
        ON u.collection_id = te.collection_id
        AND u.instance_index = te.instance_index
        AND u.machine_id = te.machine_id
        AND te.rn = 1
)

SELECT
    outcome,
    COUNT(*) AS n_records,

    -- CPI null rates
    COUNTIF(cycles_per_instruction IS NULL) AS n_cpi_null,
    ROUND(COUNTIF(cycles_per_instruction IS NULL) * 100.0 / COUNT(*), 2) AS pct_cpi_null,

    -- MAPI null rates (should mirror CPI)
    COUNTIF(memory_accesses_per_instruction IS NULL) AS n_mapi_null,
    ROUND(COUNTIF(memory_accesses_per_instruction IS NULL) * 100.0 / COUNT(*), 2) AS pct_mapi_null,

    -- Non-null CPI/MAPI distributions by outcome
    AVG(cycles_per_instruction) AS mean_cpi,
    STDDEV(cycles_per_instruction) AS std_cpi,
    APPROX_QUANTILES(cycles_per_instruction, 100)[OFFSET(50)] AS median_cpi,
    APPROX_QUANTILES(cycles_per_instruction, 100)[OFFSET(95)] AS p95_cpi,

    AVG(memory_accesses_per_instruction) AS mean_mapi,
    STDDEV(memory_accesses_per_instruction) AS std_mapi,
    APPROX_QUANTILES(memory_accesses_per_instruction, 100)[OFFSET(50)] AS median_mapi,
    APPROX_QUANTILES(memory_accesses_per_instruction, 100)[OFFSET(95)] AS p95_mapi

FROM labeled_usage
WHERE outcome IS NOT NULL
GROUP BY outcome
ORDER BY outcome;


-- ============================================================================
-- Part D: Chi-square contingency table — CPI missingness x failure outcome
-- ============================================================================
-- Produces a 2x2 table: (cpi_null vs. cpi_present) x (FAIL_LOST vs. FINISH)
-- Run a chi-square test on these counts in Python to assess whether missingness
-- is statistically independent of outcome.

WITH usage_sample AS (
    SELECT
        collection_id,
        instance_index,
        machine_id,
        cycles_per_instruction
    FROM `YOUR-PROJECT-ID-HERE.dissertation_lebel.instance_usage_full`
    TABLESAMPLE SYSTEM (5 PERCENT)
    WHERE machine_id IS NOT NULL
),

terminal_events AS (
    SELECT
        collection_id,
        instance_index,
        machine_id,
        type AS terminal_type,
        ROW_NUMBER() OVER (
            PARTITION BY collection_id, instance_index, machine_id
            ORDER BY time DESC
        ) AS rn
    FROM `YOUR-PROJECT-ID-HERE.dissertation_lebel.instance_events_full`
    WHERE type IN (5, 6, 8)  -- FAIL, FINISH, LOST only (binary outcome)
)

SELECT
    CASE WHEN u.cycles_per_instruction IS NULL THEN 'cpi_null' ELSE 'cpi_present' END AS cpi_status,
    CASE WHEN te.terminal_type IN (5, 8) THEN 'FAIL_LOST' ELSE 'FINISH' END AS outcome,
    COUNT(*) AS n_records
FROM usage_sample u
INNER JOIN terminal_events te
    ON u.collection_id = te.collection_id
    AND u.instance_index = te.instance_index
    AND u.machine_id = te.machine_id
    AND te.rn = 1
GROUP BY cpi_status, outcome
ORDER BY cpi_status, outcome;


-- ============================================================================
-- Part E: Confirm CPI and MAPI null identity
-- ============================================================================
-- Quick verification that CPI and MAPI are always null/non-null together.
-- If this returns 0 for the mismatched rows, they're perfectly correlated.

SELECT
    CASE
        WHEN cycles_per_instruction IS NULL AND memory_accesses_per_instruction IS NULL THEN 'both_null'
        WHEN cycles_per_instruction IS NOT NULL AND memory_accesses_per_instruction IS NOT NULL THEN 'both_present'
        WHEN cycles_per_instruction IS NULL AND memory_accesses_per_instruction IS NOT NULL THEN 'cpi_null_mapi_present'
        ELSE 'cpi_present_mapi_null'
    END AS null_pattern,
    COUNT(*) AS n_records,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 4) AS pct_of_total
FROM `YOUR-PROJECT-ID-HERE.dissertation_lebel.instance_usage_full`
GROUP BY null_pattern
ORDER BY null_pattern;
