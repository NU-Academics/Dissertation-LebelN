-- cache_machine_attributes.sql
-- Purpose: Cache machine attribute data for hardware analysis
-- Run ONCE, then query cached table

CREATE OR REPLACE TABLE `YOUR-PROJECT-ID-HERE.dissertation_lebel.machine_attributes_full` AS
SELECT
    time,
    machine_id,
    name,
    value,
    deleted
FROM `google.com:google-cluster-data`.clusterdata_2019_a.machine_attributes;
