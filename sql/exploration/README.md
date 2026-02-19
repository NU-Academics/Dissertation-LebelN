# Exploration Queries

Ad-hoc SQL queries for EDA against cached BigQuery tables.

All queries target YOUR cached tables (`dissertation_lebel.*`),
not the public Google dataset. Replace `YOUR-PROJECT-ID-HERE` with
the value from your `GCP_PROJECT_ID` Colab Secret.

## Follow-Up Queries from Notebook 03 EDA

### `pre_failure_utilization_profiles.sql` (Open Question #4)

**Purpose:** Determine whether failing instances exhibit detectably different
resource utilization behavior in the time window *before* the terminal event.

**Parts:**
- **A:** Pre-failure vs. pre-success utilization statistics in 5-min, 15-min,
  and 60-min lookback windows. Includes utilization-to-request ratio (resource
  squeeze indicator).
- **B:** Rate-of-change (linear slope) of CPU/memory in the 15-min pre-terminal
  window. Tests whether utilization is *accelerating* before failure.

**Decides:** Whether resource utilization time-series features belong in the RQ1
model (and how to window them). Also determines feasibility of RQ3's >15-minute
lead-time target.

**Cost:** Joins instance_events + instance_usage. Scoped to days 10-12 of the
trace period.

---

### `instance_lifecycle_reconstruction.sql` (Open Question #3)

**Purpose:** Reconstruct instance lifecycles from event sequences and extract
duration features (queue time, running duration, resubmission count).

**Parts:**
- **A:** Most common event sequence patterns (e.g., SUBMIT→ENABLE→SCHEDULE→FAIL).
- **B:** Lifecycle duration features by outcome (FAIL_LOST vs. FINISH vs. EVICT
  vs. KILL): queue time, running duration, resubmission count.
- **C:** Resubmission count vs. eventual failure rate (does resubmission predict
  failure?).
- **D:** Running duration distribution bucketed by time scale (<1s, 1-10s, ...
  >1day) for FAIL vs. FINISH (early crash vs. slow degradation?).

**Decides:** Whether lifecycle-derived features are extractable at scale and carry
signal. Directly shapes feature engineering (Section 3.5.4) and determines
computational feasibility of full lifecycle reconstruction across 1.7B events.

**Cost:** Queries instance_events only (no usage join). Scoped to days 10-12.

---

### `cpi_mapi_missingness_structure.sql` (Open Question #6)

**Purpose:** Diagnose whether the 20.5% CPI/MAPI missingness is platform-driven,
workload-driven, or outcome-driven.

**Parts:**
- **A:** CPI/MAPI null rates by platform_id (hardware hypothesis).
- **B:** CPI/MAPI null rates by scheduling_class and priority tier (workload
  hypothesis).
- **C:** CPI/MAPI null rates and non-null value distributions by instance outcome
  (outcome-dependence hypothesis).
- **D:** 2x2 contingency table (cpi_null/present x FAIL_LOST/FINISH) for
  chi-square test in Python.
- **E:** Confirm CPI and MAPI null identity (always null/non-null together).

**Decides:** Whether to drop CPI/MAPI (if platform-driven), encode missingness as
a feature (if outcome-dependent), or invest in imputation (if non-null values also
differ by outcome). Affects variable definitions (Section 3.5.2) and imputation
strategy (Section 3.6.3.2).

**Cost:** Parts A and E query instance_usage directly (full population aggregation).
Parts B-D use 5% TABLESAMPLE joined to instance_events or machine_events.
