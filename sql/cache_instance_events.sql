-- cache_instance_events.sql
-- Purpose: Cache ALL instance events — no filtering, no derived columns
-- Derived columns (failure_label, event_category) will be added AFTER EDA
-- Run ONCE, then query cached table

CREATE OR REPLACE TABLE `YOUR-PROJECT-ID-HERE.dissertation_lebel.instance_events_full` AS
SELECT
    time,
    type,
    collection_id,
    scheduling_class,
    collection_type,
    priority,
    instance_index,
    machine_id,
    alloc_collection_id,
    resource_request.cpus as cpu_request,
    resource_request.memory as memory_request,
    constraint
FROM `google.com:google-cluster-data`.clusterdata_2019_a.instance_events;
