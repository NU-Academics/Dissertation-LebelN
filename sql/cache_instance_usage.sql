-- cache_instance_usage.sql
-- Purpose: Cache ALL resource utilization data
-- This is the largest table (~1.5 TB) — run ONCE
-- Run ONCE, then query cached table

CREATE OR REPLACE TABLE `YOUR-PROJECT-ID-HERE.dissertation_lebel.instance_usage_full` AS
SELECT
    start_time,
    end_time,
    collection_id,
    instance_index,
    machine_id,
    alloc_collection_id,
    average_usage.cpus as avg_cpu,
    average_usage.memory as avg_memory,
    maximum_usage.cpus as max_cpu,
    maximum_usage.memory as max_memory,
    random_sample_usage.cpus as sample_cpu,
    random_sample_usage.memory as sample_memory,
    assigned_memory,
    page_cache_memory,
    cycles_per_instruction,
    memory_accesses_per_instruction,
    sample_rate,
    cpu_usage_distribution,
    tail_cpu_usage_distribution
FROM `google.com:google-cluster-data`.clusterdata_2019_a.instance_usage;
