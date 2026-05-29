"""Schemas and data validation.

Phase 3 deliverable. Polars schemas for each cached BigQuery table and each
Parquet partition, plus assertion helpers used by tests/test_preprocessing.py
to confirm post-preprocessing distributions match the EDA-validated targets.

Planned submodules (created during Week 2):
- schemas: Polars schemas for cached Google tables (instance_events,
  instance_usage, collection_events, machine_events, machine_attributes) and
  for the Backblaze daily-observation Parquet.
- validation: assertion helpers (null-rate checks, row-count checks, class
  imbalance ratios) callable from notebooks and tests.
"""
