# `sql/` - BigQuery Cache and Exploration Queries

Two layers:

- The five `cache_*.sql` files at the top level are one-time caching queries that materialize the public Google Cluster Traces v3 cell `a` tables into your own BigQuery dataset.
- `exploration/` contains the follow-up queries that close Phase 2 Open Questions; see `exploration/README.md` for the deeper walkthrough.

All queries target the cached `{project_id}.dissertation_lebel.*_full` tables and reference the project ID via the `GCP_PROJECT_ID` Colab Secret. Local edits typically swap `YOUR-PROJECT-ID-HERE` for the actual project name before pasting into the BigQuery console.

## Cache queries

The five caching queries are run once each, in order, from `notebooks/01_bigquery_caching.py`. Each writes to `{project_id}.dissertation_lebel.{table_name}_full` and is idempotent (re-running replaces the cached table without affecting downstream notebooks).

| File | Output table | Approximate scale |
|------|--------------|-------------------|
| `cache_instance_events.sql` | `instance_events_full` | 1,717,317,922 rows (~387 GB) |
| `cache_instance_usage.sql` | `instance_usage_full` | 7,575,500,668 rows (~1,992 GB) |
| `cache_collection_events.sql` | `collection_events_full` | 20,807,441 rows (~3 GB) |
| `cache_machine_events.sql` | `machine_events_full` | 46,219 rows |
| `cache_machine_attributes.sql` | `machine_attributes_full` | 1,702,926 rows |

Caching is the first non-trivial BigQuery cost in the project. Subsequent notebooks read exclusively from the cached tables to keep query spend predictable.

## Exploration queries

`exploration/` contains the three Phase 2 follow-up queries that close Open Questions #3, #4, and #6:

- `instance_lifecycle_reconstruction.sql` (resolves `O03`, produces `V09`, `V10`, and informs `V29`).
- `pre_failure_utilization_profiles.sql` (resolves `O04`, produces `V12`).
- `cpi_mapi_missingness_structure.sql` (resolves `O06`, produces `V11`).

The detailed per-query walkthrough lives in `exploration/README.md`. Each query was originally scoped to a 3-day temporal sample of `instance_events_full` to keep costs manageable; the production reconstructions in Phase 3 widen the scope to the full 31-day trace.

## Conventions

- BigQuery is the heavy-lift compute. Polars in-process work happens only after BigQuery has reduced the data to a tractable size.
- DDL output tables are clustered by `(collection_id, instance_index)` so downstream joins stay cheap. Date partitioning is not used because the working set is filtered by collection rather than by date.
- Sentinel timestamps (`time = 0` or `time = 2^63 - 1`, per `V25`) are filtered inside any CTE that computes lifecycle features. The constants live in `src/data/schemas.py`.
- Replace `YOUR-PROJECT-ID-HERE` before pasting into the console. Notebook 08's `fqn()` helper and `utils/bq_client.py::table_ref` substitute the project ID automatically for Python-driven calls.
