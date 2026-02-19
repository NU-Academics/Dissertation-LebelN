-- cache_collection_events.sql
-- Purpose: Cache ALL job/collection scheduling events
-- Run ONCE, then query cached table

CREATE OR REPLACE TABLE `YOUR-PROJECT-ID-HERE.dissertation_lebel.collection_events_full` AS
SELECT
    time,
    type,
    collection_id,
    scheduling_class,
    collection_type,
    priority,
    user,
    collection_name,
    parent_collection_id,
    start_after_collection_ids,
    max_per_machine,
    max_per_switch,
    vertical_scaling,
    scheduler
FROM `google.com:google-cluster-data`.clusterdata_2019_a.collection_events;
