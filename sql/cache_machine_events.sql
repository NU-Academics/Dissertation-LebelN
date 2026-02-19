-- cache_machine_events.sql
-- Purpose: Cache ALL machine lifecycle events
-- Run ONCE, then query cached table

CREATE OR REPLACE TABLE `YOUR-PROJECT-ID-HERE.dissertation_lebel.machine_events_full` AS
SELECT
    time,
    machine_id,
    type,
    switch_id,
    `capacity`.cpus as capacity_cpus,
    `capacity`.memory as capacity_memory,
    platform_id
FROM `google.com:google-cluster-data`.clusterdata_2019_a.machine_events;
