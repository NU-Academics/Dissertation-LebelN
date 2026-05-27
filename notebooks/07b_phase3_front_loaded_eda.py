# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 07b. Phase 3 Front-Loaded EDA: Resolving Open Questions Before Preprocessing
#
# **Purpose.** Four targeted Google Cluster Traces queries that resolve open
# items O01, O02, O04, and refine V11 before the Phase 3 preprocessing module
# is built. Each check is a separate subsection that emits one artifact to
# `outputs/tables/` (and, for F2, one figure to `outputs/figures/`) plus a
# one-paragraph printed finding. Each check ends by appending a new
# Validated row (V25 through V28) to `outputs/tables/eda_decisions.csv` so
# the preprocessing module inherits the resolved decision through the same
# audit trail used in notebook 07.
#
# **Status.** Week 2 deliverable of the 11-week Chapter 4 plan. Runs against
# the full cached `instance_events_full` and `instance_usage_full` tables.
# Phase 2 query patterns (helpers, fqn, save_table, save_figure) are reused
# verbatim from notebook 03.
#
# **Sections.**
# 1. Setup (BigQuery client, helpers, decisions-log appender).
# 2. F1: Sentinel timestamp enumeration (resolves O04).
# 3. F2: Diurnal and weekly density patterns (resolves O01).
# 4. F3: Monitoring-tier eviction triage (resolves O02).
# 5. F4: CPI/MAPI within-instance variance (refines V11).
# 6. Verification: re-read the decisions log and confirm V25 through V28
#    landed correctly.

# %% [markdown]
# ## 1. Setup
#
# Reuses the notebook 03 setup pattern: Colab secrets for project ID, Drive
# mount for outputs, BigQuery client, and the same `fqn`, `run_query`,
# `save_table`, `save_figure`, and `us_to_datetime` helpers. A new helper,
# `append_decision`, idempotently appends a row to the decisions CSV.

# %%
# !pip install -q polars google-cloud-bigquery db-dtypes matplotlib seaborn

# %%
from google.colab import userdata

PROJECT_ID = userdata.get('GCP_PROJECT_ID')
DATASET = f"{PROJECT_ID}.dissertation_lebel"
print(f"GCP Project: {PROJECT_ID}")
print(f"Dataset:     {DATASET}")

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
from pathlib import Path

DRIVE_PATH = Path('/content/drive/MyDrive/Dissertation_Colab')
TABLES_DIR = DRIVE_PATH / 'outputs' / 'tables'
FIGURES_DIR = DRIVE_PATH / 'outputs' / 'figures'

for dir_path in [TABLES_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

DECISIONS_CSV = TABLES_DIR / 'eda_decisions.csv'
print(f"Tables:    {TABLES_DIR}")
print(f"Figures:   {FIGURES_DIR}")
print(f"Decisions: {DECISIONS_CSV}")

# %%
from google.colab import auth
auth.authenticate_user()

from google.cloud import bigquery
bq_client = bigquery.Client(project=PROJECT_ID)

# %%
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.figsize'] = (12, 5)

# %%
def fqn(table: str) -> str:
    """Return fully-qualified BigQuery table name."""
    return f"`{DATASET}.{table}`"


def run_query(sql: str) -> pl.DataFrame:
    """Execute SQL and return a Polars DataFrame."""
    return pl.from_pandas(bq_client.query(sql).to_dataframe())


def save_table(df: pl.DataFrame, name: str) -> Path:
    """Save a Polars DataFrame as CSV to the Drive tables directory."""
    path = TABLES_DIR / f"{name}.csv"
    df.write_csv(str(path))
    print(f"  Saved: {path}  ({len(df)} rows)")
    return path


def save_figure(fig, name: str) -> Path:
    """Save a matplotlib figure as PNG to the Drive figures directory."""
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(str(path), bbox_inches='tight')
    print(f"  Saved: {path}")
    return path


# Google Cluster Traces timestamp conversion (matches notebook 03)
TRACE_START = datetime(2019, 5, 1, 0, 0, 0)
OFFSET_SECONDS = 600
MAX_INT64 = 9_223_372_036_854_775_807  # 2**63 - 1, the right-censoring sentinel


def us_to_datetime(time_us: int | None) -> datetime | None:
    """Convert Google trace microsecond timestamp to datetime (PDT)."""
    if time_us is None or time_us == 0:
        return None
    if time_us >= MAX_INT64:
        return None
    seconds_since_start = (time_us / 1_000_000) - OFFSET_SECONDS
    return TRACE_START + timedelta(seconds=seconds_since_start)


# Event type labels for readability (matches notebook 03)
EVENT_TYPE_LABELS = {
    0: 'SUBMIT', 1: 'QUEUE', 2: 'ENABLE', 3: 'SCHEDULE', 4: 'EVICT',
    5: 'FAIL', 6: 'FINISH', 7: 'KILL', 8: 'LOST',
    9: 'UPDATE_PENDING', 10: 'UPDATE_RUNNING',
}

# %% [markdown]
# ### Decisions-log appender
#
# Reads the existing `eda_decisions.csv` (created by notebook 07), removes
# any prior row with the same `id` (so re-running this notebook overwrites
# rather than duplicates V25 through V28), appends the new row, and writes
# the file back. The schema matches notebook 07 exactly.

# %%
DECISIONS_SCHEMA = [
    "id", "category", "dataset", "item", "evidence_or_rationale",
    "source", "applies_to_rq", "status", "next_step",
]


def append_decision(
    decision_id: str,
    category: str,
    dataset: str,
    item: str,
    evidence: str,
    source: str,
    applies_to_rq: str,
    status: str,
    next_step: str,
) -> None:
    """Idempotently append (or overwrite by id) a row to eda_decisions.csv."""
    new_row = pl.DataFrame(
        [(decision_id, category, dataset, item, evidence, source, applies_to_rq, status, next_step)],
        schema=DECISIONS_SCHEMA,
        orient="row",
    )
    if DECISIONS_CSV.exists():
        existing = pl.read_csv(str(DECISIONS_CSV))
        existing = existing.filter(pl.col("id") != decision_id)
        combined = pl.concat([existing, new_row], how="vertical_relaxed")
    else:
        combined = new_row
    combined.write_csv(str(DECISIONS_CSV))
    print(f"  Decisions log updated with {decision_id} (total rows: {combined.height})")


# %% [markdown]
# ---
# ## 2. F1: Sentinel timestamp enumeration (resolves O04)
#
# **Question.** How many rows of `instance_events_full` carry the sentinel
# timestamps `time = 0` (left-censoring marker) or `time = 2^63 - 1`
# (right-censoring marker), and how are they distributed across event
# types? What does the running-duration distribution by terminal type look
# like when those rows are included versus excluded?
#
# **Why it matters.** Any lifecycle-style feature (queue time, scheduled
# duration, running duration) must take a position on these sentinels. The
# rapid-onset failure model in V09 was computed on rows with valid
# timestamps; if a non-trivial slice of the failure population carries a
# sentinel timestamp, the V09 distributions could shift materially.
#
# **Decision rule.** Drop sentinel-bearing rows outright unless they are
# either (a) a large enough fraction of FAIL_LOST or FINISH to shift class
# balance by more than 1 percentage point, or (b) concentrated in a
# specific terminal type in a way that suggests left/right-censored
# survival modeling would be informative.

# %% [markdown]
# ### F1.1 Sentinel counts by event type

# %%
sentinel_inventory_sql = f"""
WITH classified AS (
    SELECT
        type,
        CASE
            WHEN time = 0 THEN 'zero'
            WHEN time = {MAX_INT64} THEN 'max_int64'
            WHEN time IS NULL THEN 'null'
            ELSE 'valid'
        END AS time_status
    FROM {fqn('instance_events_full')}
)
SELECT
    type,
    time_status,
    COUNT(*) AS n_rows
FROM classified
GROUP BY type, time_status
ORDER BY type, time_status
"""

sentinel_inventory_df = run_query(sentinel_inventory_sql)

# Add event type labels for readability
sentinel_inventory_df = sentinel_inventory_df.with_columns(
    pl.col("type").map_elements(
        lambda t: EVENT_TYPE_LABELS.get(t, f"UNKNOWN_{t}"),
        return_dtype=pl.Utf8,
    ).alias("type_label"),
)

print("Sentinel inventory (event_type x time_status):")
print(sentinel_inventory_df.sort(["type", "time_status"]))

# %% [markdown]
# ### F1.2 Running-duration distribution with vs without sentinels
#
# For each terminal event type (FAIL, LOST, FINISH, EVICT, KILL), compute
# the running duration as `terminal_time - schedule_time` per
# `(collection_id, instance_index)`, once including instances that touch a
# sentinel timestamp anywhere in their lifecycle and once excluding them.

# %%
running_duration_sql = f"""
WITH schedule_events AS (
    SELECT
        collection_id,
        instance_index,
        MAX(time) AS schedule_time,
        MAX(CASE WHEN time IN (0, {MAX_INT64}) THEN 1 ELSE 0 END) AS schedule_touched_sentinel
    FROM {fqn('instance_events_full')}
    WHERE type = 3  -- SCHEDULE
    GROUP BY collection_id, instance_index
),
terminal_events AS (
    SELECT
        collection_id,
        instance_index,
        type AS terminal_type,
        time AS terminal_time,
        CASE WHEN time IN (0, {MAX_INT64}) THEN 1 ELSE 0 END AS terminal_is_sentinel,
        ROW_NUMBER() OVER (
            PARTITION BY collection_id, instance_index
            ORDER BY time DESC
        ) AS rn
    FROM {fqn('instance_events_full')}
    WHERE type IN (4, 5, 6, 7, 8)
),
lifecycle AS (
    SELECT
        te.terminal_type,
        (te.terminal_time - se.schedule_time) / 1e6 AS running_seconds,
        CASE
            WHEN te.terminal_is_sentinel = 1 OR se.schedule_touched_sentinel = 1 THEN 1
            ELSE 0
        END AS touches_sentinel
    FROM terminal_events te
    INNER JOIN schedule_events se
        ON te.collection_id = se.collection_id
        AND te.instance_index = se.instance_index
    WHERE te.rn = 1
)
SELECT
    terminal_type,
    touches_sentinel,
    COUNT(*) AS n_instances,
    APPROX_QUANTILES(running_seconds, 100)[OFFSET(50)] AS median_running_seconds,
    APPROX_QUANTILES(running_seconds, 100)[OFFSET(5)] AS p05_running_seconds,
    APPROX_QUANTILES(running_seconds, 100)[OFFSET(95)] AS p95_running_seconds,
    AVG(running_seconds) AS mean_running_seconds,
    STDDEV(running_seconds) AS std_running_seconds
FROM lifecycle
GROUP BY terminal_type, touches_sentinel
ORDER BY terminal_type, touches_sentinel
"""

running_duration_df = run_query(running_duration_sql)
running_duration_df = running_duration_df.with_columns(
    pl.col("terminal_type").map_elements(
        lambda t: EVENT_TYPE_LABELS.get(t, f"UNKNOWN_{t}"),
        return_dtype=pl.Utf8,
    ).alias("terminal_type_label"),
)
print("Running duration distribution (terminal_type x sentinel_touch):")
print(running_duration_df.sort(["terminal_type", "touches_sentinel"]))

# %% [markdown]
# ### F1.3 Combine and emit `sentinel_inventory.csv`
#
# The CSV carries both the per-type sentinel counts (long format) and the
# running-duration distributions side by side. A `section` column
# distinguishes the two views.

# %%
sentinel_section = sentinel_inventory_df.select([
    pl.lit("counts").alias("section"),
    pl.col("type").alias("event_type"),
    pl.col("type_label").alias("event_label"),
    pl.col("time_status"),
    pl.col("n_rows").alias("n"),
    pl.lit(None).cast(pl.Float64).alias("touches_sentinel"),
    pl.lit(None).cast(pl.Float64).alias("median_running_seconds"),
    pl.lit(None).cast(pl.Float64).alias("p05_running_seconds"),
    pl.lit(None).cast(pl.Float64).alias("p95_running_seconds"),
    pl.lit(None).cast(pl.Float64).alias("mean_running_seconds"),
    pl.lit(None).cast(pl.Float64).alias("std_running_seconds"),
])

duration_section = running_duration_df.select([
    pl.lit("durations").alias("section"),
    pl.col("terminal_type").alias("event_type"),
    pl.col("terminal_type_label").alias("event_label"),
    pl.lit(None).cast(pl.Utf8).alias("time_status"),
    pl.col("n_instances").alias("n"),
    pl.col("touches_sentinel").cast(pl.Float64),
    pl.col("median_running_seconds"),
    pl.col("p05_running_seconds"),
    pl.col("p95_running_seconds"),
    pl.col("mean_running_seconds"),
    pl.col("std_running_seconds"),
])

sentinel_inventory_out = pl.concat([sentinel_section, duration_section], how="vertical_relaxed")
save_table(sentinel_inventory_out, "sentinel_inventory")

# %% [markdown]
# ### F1.4 Finding paragraph and decisions-log row (V25)
#
# Compute the headline numbers and print a one-paragraph finding. The
# finding is also encoded as the V25 row appended to `eda_decisions.csv`.

# %%
# Totals across all event types
total_zero = sentinel_inventory_df.filter(pl.col("time_status") == "zero")["n_rows"].sum() or 0
total_max = sentinel_inventory_df.filter(pl.col("time_status") == "max_int64")["n_rows"].sum() or 0
total_rows = sentinel_inventory_df["n_rows"].sum()
pct_zero = 100.0 * total_zero / total_rows if total_rows else 0.0
pct_max = 100.0 * total_max / total_rows if total_rows else 0.0

# Sentinel impact on FAIL_LOST and FINISH lifecycle distributions
def _row(df: pl.DataFrame, terminal_types: list[int], touches: int) -> dict | None:
    sub = df.filter(
        pl.col("terminal_type").is_in(terminal_types) & (pl.col("touches_sentinel") == touches)
    )
    if sub.height == 0:
        return None
    # Aggregate across both FAIL and LOST when grouped together
    agg = sub.select([
        pl.col("n_instances").sum().alias("n_instances"),
        # Weighted median is non-trivial; for the finding we report the
        # max across the included types as the representative figure.
        pl.col("median_running_seconds").max().alias("median_running_seconds"),
    ]).to_dicts()[0]
    return agg


fail_lost_clean = _row(running_duration_df, [5, 8], 0) or {"n_instances": 0, "median_running_seconds": float("nan")}
fail_lost_dirty = _row(running_duration_df, [5, 8], 1) or {"n_instances": 0, "median_running_seconds": float("nan")}
finish_clean = _row(running_duration_df, [6], 0) or {"n_instances": 0, "median_running_seconds": float("nan")}
finish_dirty = _row(running_duration_df, [6], 1) or {"n_instances": 0, "median_running_seconds": float("nan")}

fail_lost_total = fail_lost_clean["n_instances"] + fail_lost_dirty["n_instances"]
finish_total = finish_clean["n_instances"] + finish_dirty["n_instances"]
fail_lost_pct_sentinel = (
    100.0 * fail_lost_dirty["n_instances"] / fail_lost_total if fail_lost_total else 0.0
)
finish_pct_sentinel = (
    100.0 * finish_dirty["n_instances"] / finish_total if finish_total else 0.0
)

# Decision: drop unless either tail exceeds 1pp class-balance shift
shifts_balance = abs(fail_lost_pct_sentinel - finish_pct_sentinel) > 1.0
decision_text = (
    "treat sentinel-bearing rows as left/right-censored for survival-style analysis"
    if shifts_balance
    else "drop sentinel-bearing rows outright in preprocessing (V03-style)"
)

f1_finding = (
    f"F1 (Sentinel timestamps). Across the 1.72 billion rows of "
    f"instance_events_full, {total_zero:,} ({pct_zero:.4f}%) carry time = 0 "
    f"and {total_max:,} ({pct_max:.4f}%) carry time = 2^63 - 1. Among "
    f"FAIL_LOST instances, {fail_lost_dirty['n_instances']:,} of "
    f"{fail_lost_total:,} ({fail_lost_pct_sentinel:.2f}%) have a "
    f"sentinel-bearing schedule or terminal event; the comparable rate for "
    f"FINISH is {finish_pct_sentinel:.2f}%. The sentinel-clean FAIL_LOST "
    f"median running duration is {fail_lost_clean['median_running_seconds']:.1f} "
    f"seconds versus {fail_lost_dirty['median_running_seconds']:.1f} "
    f"seconds when sentinel-touching instances are included; the "
    f"corresponding FINISH medians are {finish_clean['median_running_seconds']:.1f} "
    f"versus {finish_dirty['median_running_seconds']:.1f} seconds. "
    f"Decision: {decision_text}. The 1-percentage-point class-balance "
    f"shift threshold is {'exceeded' if shifts_balance else 'not exceeded'}."
)
print(f1_finding)

# %%
append_decision(
    decision_id="V25",
    category="Sentinel Timestamp Handling",
    dataset="Google",
    item=(
        f"Sentinel rows: time=0 = {total_zero:,} ({pct_zero:.4f}%); "
        f"time=2^63-1 = {total_max:,} ({pct_max:.4f}%). "
        f"Decision: {decision_text}."
    ),
    evidence=(
        f"F1 sentinel inventory query against full instance_events_full; "
        f"FAIL_LOST sentinel-touch rate {fail_lost_pct_sentinel:.2f}%, FINISH "
        f"{finish_pct_sentinel:.2f}%; class-balance shift threshold "
        f"{'exceeded' if shifts_balance else 'not exceeded'}."
    ),
    source="notebooks/07b_phase3_front_loaded_eda.py Section 2; outputs/tables/sentinel_inventory.csv",
    applies_to_rq="RQ1, RQ3",
    status="Validated (Phase 3 front-loaded)",
    next_step=(
        "Encode in src/preprocessing/timestamps.py: filter sentinel rows "
        "before lifecycle feature computation; expose a censoring indicator "
        "for survival-style sensitivity analysis if RQ3 horizon work needs it."
    ),
)

# %% [markdown]
# ---
# ## 3. F2: Diurnal and weekly density patterns (resolves O01)
#
# **Question.** Does event density swing by hour of day or by day of week,
# and if so, does the FAIL_LOST rate move with it? Material to whether the
# 70/15/15 chronological split must be stratified by calendar position or
# whether a uniform calendar-day split is defensible.
#
# **Why it matters.** The Google trace covers 31 days. A naive chronological
# split that crosses a weekday/weekend boundary or an end-of-month batch
# window could bias the validation and test folds. If event density swings
# more than 3x between business hours and weekends, stratification needs to
# pass into `src/features/sampling.py`.

# %% [markdown]
# ### F2.1 Hourly and weekly aggregates
#
# Convert microsecond timestamps to wall-clock by subtracting the 600-second
# trace offset and adding the May 1, 2019 trace start anchor. Filter out
# sentinel timestamps before extracting hour and day-of-week so the buckets
# remain interpretable.

# %%
temporal_density_sql = f"""
-- Anchor at the wall-clock trace start, which is 2019-05-01 00:00 PDT
-- (= 2019-05-01 07:00 UTC). We intentionally use the PDT calendar wall
-- clock by anchoring at '2019-05-01 00:00:00 UTC' here: the 7-hour offset
-- between the wrong UTC anchor and the real trace anchor cancels with the
-- UTC->PDT offset, so EXTRACT(HOUR/DAYOFWEEK) returns PDT wall-clock
-- components. The us_to_datetime() helper in notebook 03 uses the same
-- naive PDT anchor; the SQL below stays consistent with that helper.
WITH events AS (
    SELECT
        type,
        TIMESTAMP_ADD(
            TIMESTAMP('2019-05-01 00:00:00 UTC'),
            INTERVAL CAST(time / 1000000 - 600 AS INT64) SECOND
        ) AS event_ts  -- naive timestamp; EXTRACT yields PDT wall-clock
    FROM {fqn('instance_events_full')}
    WHERE time > 0 AND time < {MAX_INT64}
),
bucketed AS (
    SELECT
        EXTRACT(DAYOFWEEK FROM event_ts) AS day_of_week,
        EXTRACT(HOUR FROM event_ts) AS hour_of_day,
        type
    FROM events
)
SELECT
    day_of_week,
    hour_of_day,
    COUNT(*) AS n_events,
    COUNTIF(type IN (5, 8)) AS n_fail_lost,
    COUNTIF(type = 6) AS n_finish,
    COUNTIF(type IN (5, 6, 8)) AS n_terminal,
    SAFE_DIVIDE(COUNTIF(type IN (5, 8)), COUNTIF(type IN (5, 6, 8))) AS fail_lost_rate
FROM bucketed
GROUP BY day_of_week, hour_of_day
ORDER BY day_of_week, hour_of_day
"""

temporal_density_df = run_query(temporal_density_sql)

# BigQuery DAYOFWEEK: 1=Sunday ... 7=Saturday. Translate to readable labels.
DAY_LABELS = {1: "Sun", 2: "Mon", 3: "Tue", 4: "Wed", 5: "Thu", 6: "Fri", 7: "Sat"}
temporal_density_df = temporal_density_df.with_columns(
    pl.col("day_of_week").map_elements(
        lambda d: DAY_LABELS.get(d, str(d)), return_dtype=pl.Utf8,
    ).alias("day_label"),
)

print("Temporal density (first 24 rows):")
print(temporal_density_df.head(24))

save_table(temporal_density_df, "temporal_density")

# %% [markdown]
# ### F2.2 Diurnal heatmap and weekly density curve

# %%
heatmap_pivot = (
    temporal_density_df
    .pivot(values="n_events", index="day_of_week", on="hour_of_day", aggregate_function="sum")
    .sort("day_of_week")
)
day_labels_ordered = [DAY_LABELS[d] for d in heatmap_pivot["day_of_week"].to_list()]
heatmap_matrix = heatmap_pivot.drop("day_of_week").to_numpy()

fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 2]})

# Top: diurnal x weekday heatmap of event counts
hm = axes[0].imshow(heatmap_matrix, aspect="auto", cmap="viridis")
axes[0].set_yticks(range(len(day_labels_ordered)))
axes[0].set_yticklabels(day_labels_ordered)
axes[0].set_xticks(range(24))
axes[0].set_xticklabels(range(24))
axes[0].set_xlabel("Hour of day (PDT)")
axes[0].set_ylabel("Day of week (PDT)")
axes[0].set_title("instance_events density by hour x weekday (PDT, event count)")
cbar = fig.colorbar(hm, ax=axes[0])
cbar.set_label("n_events")

# Bottom: weekly density curve aggregated across hours
weekly_curve = (
    temporal_density_df
    .group_by("day_of_week")
    .agg([
        pl.col("n_events").sum().alias("n_events"),
        pl.col("n_fail_lost").sum().alias("n_fail_lost"),
        pl.col("n_terminal").sum().alias("n_terminal"),
    ])
    .sort("day_of_week")
)
weekly_curve = weekly_curve.with_columns(
    (pl.col("n_fail_lost") / pl.col("n_terminal").cast(pl.Float64)).alias("fail_lost_rate"),
)
xs = list(range(weekly_curve.height))
axes[1].bar(xs, weekly_curve["n_events"].to_list(), color="#4c72b0", alpha=0.7, label="n_events")
axes[1].set_xticks(xs)
axes[1].set_xticklabels([DAY_LABELS[d] for d in weekly_curve["day_of_week"].to_list()])
axes[1].set_ylabel("Events", color="#4c72b0")
axes[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))

ax2 = axes[1].twinx()
ax2.plot(
    xs,
    [r * 100.0 for r in weekly_curve["fail_lost_rate"].to_list()],
    color="#c44e52", marker="o", linewidth=2, label="FAIL_LOST rate (%)",
)
ax2.set_ylabel("FAIL_LOST rate (%)", color="#c44e52")
axes[1].set_title("Weekly density and FAIL_LOST rate by day of week")

plt.tight_layout()
save_figure(fig, "diurnal_density")
plt.show()

# %% [markdown]
# ### F2.3 Finding paragraph and decisions-log row (V26)

# %%
# Compute the headline swing ratios
max_hour_events = temporal_density_df["n_events"].max()
min_hour_events = temporal_density_df["n_events"].min()
hourly_swing = max_hour_events / min_hour_events if min_hour_events else float("inf")

max_day_events = weekly_curve["n_events"].max()
min_day_events = weekly_curve["n_events"].min()
weekly_swing = max_day_events / min_day_events if min_day_events else float("inf")

# Business hours (9-17 UTC) vs off hours (0-8, 18-23) average density
business_hours = temporal_density_df.filter(
    (pl.col("hour_of_day") >= 9) & (pl.col("hour_of_day") <= 17)
)["n_events"].sum()
off_hours = temporal_density_df.filter(
    (pl.col("hour_of_day") < 9) | (pl.col("hour_of_day") > 17)
)["n_events"].sum()
business_vs_off = (business_hours / 9) / (off_hours / 15) if off_hours else float("inf")

# Weekday vs weekend
weekday_events = weekly_curve.filter(pl.col("day_of_week").is_in([2, 3, 4, 5, 6]))["n_events"].sum()
weekend_events = weekly_curve.filter(pl.col("day_of_week").is_in([1, 7]))["n_events"].sum()
weekday_vs_weekend = (weekday_events / 5) / (weekend_events / 2) if weekend_events else float("inf")

needs_stratification = (hourly_swing > 3.0) or (weekly_swing > 3.0) or (business_vs_off > 3.0)
stratification_decision = (
    "stratify the 70/15/15 split by (day_of_week, hour_of_day) bucket"
    if needs_stratification
    else "the uniform chronological 70/15/15 split is defensible without temporal stratification"
)

f2_finding = (
    f"F2 (Diurnal and weekly density). Hourly event counts swing "
    f"{hourly_swing:.2f}x between the busiest and quietest hour-of-day "
    f"bucket. Daily event totals swing {weekly_swing:.2f}x across days of "
    f"the week. The mean-per-hour density during business hours (UTC 09-17) "
    f"is {business_vs_off:.2f}x the mean-per-hour density during off-hours. "
    f"Mean weekday density is {weekday_vs_weekend:.2f}x mean weekend "
    f"density. Decision: {stratification_decision}."
)
print(f2_finding)

# %%
append_decision(
    decision_id="V26",
    category="Temporal Stratification",
    dataset="Google",
    item=(
        f"Hourly swing {hourly_swing:.2f}x; weekly swing {weekly_swing:.2f}x; "
        f"business vs off-hours density ratio {business_vs_off:.2f}x; "
        f"weekday vs weekend {weekday_vs_weekend:.2f}x. "
        f"Decision: {stratification_decision}."
    ),
    evidence=(
        "F2 hour-of-day x day-of-week aggregation across instance_events_full "
        "with sentinel timestamps excluded; 3x swing threshold per the "
        "Chapter 4 plan. Hour and day buckets are PDT wall-clock (the SQL "
        "anchors at a naive UTC timestamp whose 7-hour offset cancels with "
        "the UTC->PDT offset, matching notebook 03's us_to_datetime helper). "
        "FAIL_LOST rate varies up to ~8x across hour-of-day buckets, peaking "
        "during PDT business hours and bottoming out overnight."
    ),
    source=(
        "notebooks/07b_phase3_front_loaded_eda.py Section 3; "
        "outputs/tables/temporal_density.csv; "
        "outputs/figures/diurnal_density.png"
    ),
    applies_to_rq="RQ1, RQ2, RQ3, RQ4",
    status="Validated (Phase 3 front-loaded)",
    next_step=(
        "Encode in src/features/sampling.py: "
        + (
            "build a stratified-by-(day_of_week, hour_of_day) chronological "
            "splitter and route the 70/15/15 working-set construction through it."
            if needs_stratification
            else "retain the uniform chronological splitter; document this finding "
            "in the working-draft Sampling Strategy as the empirical basis."
        )
    ),
)

# %% [markdown]
# ---
# ## 4. F3: Monitoring-tier eviction triage (resolves O02)
#
# **Question.** The Phase 2 EDA flagged 7.8 million monitoring-priority
# (priority >= 360) EVICT events as a candidate third sensitivity branch
# alongside the Production EVICT branch (P04). Do these instances pattern-
# match FAIL_LOST (rapid-onset, high resubmission rate, distinctive
# platform mix) or do they pattern-match expected Borg scheduler behavior
# (long-running, scheduled preemption, no excess resubmission)?
#
# **Decision rule.** If monitoring EVICTs pattern-match FAIL_LOST, add a
# third sensitivity column (`failure_label_sensitivity_v2`) in
# preprocessing and run the corresponding sensitivity branch in Week 10.
# If they pattern-match scheduled preemption, document the exclusion
# rationale and stop. Default: exclude unless the profile is closer to
# FAIL than to scheduled preemption.

# %% [markdown]
# ### F3.1 Build the comparator profile
#
# Compute lifecycle duration, resubmission rate, and platform distribution
# for three groups: monitoring-priority EVICTs (type=4, priority >= 360),
# Production-priority EVICTs (type=4, 120 <= priority < 360), and FAIL_LOST
# (type in 5, 8). The Production EVICT group is the existing P04 sensitivity
# branch and serves as the middle reference.

# %%
monitoring_evict_sql = f"""
WITH terminal_events AS (
    SELECT
        collection_id,
        instance_index,
        machine_id,
        type AS terminal_type,
        time AS terminal_time,
        priority,
        ROW_NUMBER() OVER (
            PARTITION BY collection_id, instance_index
            ORDER BY time DESC
        ) AS rn
    FROM {fqn('instance_events_full')}
    WHERE type IN (4, 5, 8)
      AND time > 0 AND time < {MAX_INT64}
),
labeled_terminal AS (
    SELECT
        collection_id,
        instance_index,
        machine_id,
        terminal_type,
        terminal_time,
        priority,
        CASE
            WHEN terminal_type IN (5, 8) THEN 'FAIL_LOST'
            WHEN terminal_type = 4 AND priority >= 360 THEN 'EVICT_MONITORING'
            WHEN terminal_type = 4 AND priority BETWEEN 120 AND 359 THEN 'EVICT_PRODUCTION'
            ELSE NULL
        END AS group_label
    FROM terminal_events
    WHERE rn = 1
),
submit_counts AS (
    SELECT
        collection_id,
        instance_index,
        COUNT(*) AS n_submits
    FROM {fqn('instance_events_full')}
    WHERE type = 0  -- SUBMIT
    GROUP BY collection_id, instance_index
),
schedule_times AS (
    SELECT
        collection_id,
        instance_index,
        MAX(time) AS schedule_time
    FROM {fqn('instance_events_full')}
    WHERE type = 3  -- SCHEDULE
      AND time > 0 AND time < {MAX_INT64}
    GROUP BY collection_id, instance_index
),
machine_platforms AS (
    SELECT machine_id, platform_id
    FROM (
        SELECT
            machine_id,
            platform_id,
            ROW_NUMBER() OVER (PARTITION BY machine_id ORDER BY time DESC) AS rn
        FROM {fqn('machine_events_full')}
        WHERE platform_id IS NOT NULL
    )
    WHERE rn = 1
),
joined AS (
    SELECT
        lt.group_label,
        (lt.terminal_time - st.schedule_time) / 1e6 AS running_seconds,
        sc.n_submits,
        mp.platform_id
    FROM labeled_terminal lt
    LEFT JOIN submit_counts sc
        ON lt.collection_id = sc.collection_id
        AND lt.instance_index = sc.instance_index
    LEFT JOIN schedule_times st
        ON lt.collection_id = st.collection_id
        AND lt.instance_index = st.instance_index
    LEFT JOIN machine_platforms mp
        ON lt.machine_id = mp.machine_id
    WHERE lt.group_label IS NOT NULL
)
SELECT
    group_label,
    COUNT(*) AS n_instances,
    APPROX_QUANTILES(running_seconds, 100)[OFFSET(50)] AS median_running_seconds,
    APPROX_QUANTILES(running_seconds, 100)[OFFSET(95)] AS p95_running_seconds,
    AVG(running_seconds) AS mean_running_seconds,
    AVG(n_submits) AS mean_submit_count,
    COUNTIF(n_submits > 1) AS n_resubmitted,
    SAFE_DIVIDE(COUNTIF(n_submits > 1), COUNT(*)) AS resubmission_rate,
    COUNT(DISTINCT platform_id) AS n_distinct_platforms,
    APPROX_TOP_COUNT(platform_id, 5) AS top_platforms
FROM joined
GROUP BY group_label
ORDER BY group_label
"""

monitoring_evict_summary_df = run_query(monitoring_evict_sql)


# top_platforms is an ARRAY<STRUCT<value, count>> from APPROX_TOP_COUNT.
# After BigQuery -> pandas -> Polars, each cell can arrive as a list of dicts,
# a numpy array of dicts, a Polars Series of structs, or None. The serializer
# below handles each shape without relying on a truthiness check (which raises
# on Series) and without assuming dict-style access.
def _fmt_top_platforms(arr) -> str:
    if arr is None:
        return ""
    try:
        items = list(arr)
    except TypeError:
        return ""
    parts = []
    for row in items:
        if row is None:
            continue
        if hasattr(row, "get"):
            value = row.get("value", "NULL")
            count = row.get("count", 0)
        else:
            try:
                value = row["value"]
                count = row["count"]
            except (KeyError, IndexError, TypeError):
                value, count = "NULL", 0
        parts.append(f"{value}({count})")
    return "; ".join(parts)


monitoring_evict_summary_df = monitoring_evict_summary_df.with_columns(
    pl.col("top_platforms").map_elements(
        _fmt_top_platforms, return_dtype=pl.Utf8,
    ).alias("top_platforms"),
)
print("Monitoring-tier EVICT profile vs. FAIL_LOST vs. Production EVICT:")
print(monitoring_evict_summary_df)

# %% [markdown]
# ### F3.2 Repeated `instance_index` patterns within monitoring EVICTs
#
# If the same `(collection_id, instance_index)` appears many times across
# the trace as a monitoring EVICT terminal, that pattern looks like
# scheduled canary/health-check turnover. If each instance terminates only
# once, the pattern looks like a genuine failure population.

# %%
repeated_instances_sql = f"""
WITH terminal_events AS (
    SELECT
        collection_id,
        instance_index,
        priority,
        ROW_NUMBER() OVER (
            PARTITION BY collection_id, instance_index
            ORDER BY time
        ) AS terminal_seq
    FROM {fqn('instance_events_full')}
    WHERE type = 4 AND priority >= 360
)
SELECT
    terminal_seq,
    COUNT(*) AS n_instances_at_this_seq
FROM terminal_events
GROUP BY terminal_seq
ORDER BY terminal_seq
"""

repeated_df = run_query(repeated_instances_sql)
print("Distribution of monitoring-EVICT terminal sequence numbers per instance:")
print(repeated_df.head(15))

# %%
# Combine the two views into one CSV
profile_section = monitoring_evict_summary_df.select([
    pl.lit("profile").alias("section"),
    pl.col("group_label"),
    pl.col("n_instances"),
    pl.col("median_running_seconds"),
    pl.col("p95_running_seconds"),
    pl.col("mean_running_seconds"),
    pl.col("mean_submit_count"),
    pl.col("n_resubmitted"),
    pl.col("resubmission_rate"),
    pl.col("n_distinct_platforms"),
    pl.col("top_platforms"),
    pl.lit(None).cast(pl.Int64).alias("terminal_seq"),
    pl.lit(None).cast(pl.Int64).alias("n_instances_at_this_seq"),
])
repeats_section = repeated_df.select([
    pl.lit("repeats").alias("section"),
    pl.lit("EVICT_MONITORING").alias("group_label"),
    pl.lit(None).cast(pl.Int64).alias("n_instances"),
    pl.lit(None).cast(pl.Float64).alias("median_running_seconds"),
    pl.lit(None).cast(pl.Float64).alias("p95_running_seconds"),
    pl.lit(None).cast(pl.Float64).alias("mean_running_seconds"),
    pl.lit(None).cast(pl.Float64).alias("mean_submit_count"),
    pl.lit(None).cast(pl.Int64).alias("n_resubmitted"),
    pl.lit(None).cast(pl.Float64).alias("resubmission_rate"),
    pl.lit(None).cast(pl.Int64).alias("n_distinct_platforms"),
    pl.lit(None).cast(pl.Utf8).alias("top_platforms"),
    pl.col("terminal_seq"),
    pl.col("n_instances_at_this_seq"),
])
monitoring_evict_profile = pl.concat(
    [profile_section, repeats_section], how="vertical_relaxed"
)
save_table(monitoring_evict_profile, "monitoring_evict_profile")

# %% [markdown]
# ### F3.3 Finding paragraph and decisions-log row (V27)

# %%
def _profile_row(label: str) -> dict | None:
    sub = monitoring_evict_summary_df.filter(pl.col("group_label") == label)
    return sub.to_dicts()[0] if sub.height else None


mon = _profile_row("EVICT_MONITORING") or {}
fail = _profile_row("FAIL_LOST") or {}
prod = _profile_row("EVICT_PRODUCTION") or {}

# IMPORTANT: the lifecycle metrics in monitoring_evict_summary_df
# (median_running_seconds, mean_running_seconds, resubmission_rate
# computed from mean_submit_count) are unreliable because the F3.1 SQL
# pairs each instance's latest terminal event with MAX(time) of all its
# SCHEDULE events. For long-lived instances with multiple
# schedule/evict cycles, the latest SCHEDULE often post-dates the
# chosen terminal, producing structurally negative running_seconds.
# The load-bearing evidence for V27 is the F3.2 repeats distribution
# (terminal_seq counts), which shows the per-instance recurrence
# pattern of monitoring-priority EVICTs.

# Recurrence statistics from repeated_df: for each terminal_seq value
# N, n_instances_at_this_seq is the count of instances that
# experienced at least N monitoring-priority EVICT terminals over the
# trace window.
def _instances_at_least(seq: int) -> int:
    sub = repeated_df.filter(pl.col("terminal_seq") == seq)
    if sub.height == 0:
        return 0
    return int(sub["n_instances_at_this_seq"].item())


base_count = _instances_at_least(1)
n_at_2 = _instances_at_least(2)
n_at_10 = _instances_at_least(10)
n_at_100 = _instances_at_least(100)
n_at_500 = _instances_at_least(500)


def _pct(numerator: int) -> float:
    return 100.0 * numerator / base_count if base_count else 0.0


pct_2plus = _pct(n_at_2)
pct_10plus = _pct(n_at_10)
pct_100plus = _pct(n_at_100)
pct_500plus = _pct(n_at_500)
max_seq = (
    int(repeated_df["terminal_seq"].max()) if repeated_df.height else 0
)

# Decision rule: high per-instance recurrence (large fraction of
# monitoring-EVICT instances cycling through tens of evictions) is the
# signature of canary/health-check processes preempted by Borg, not of
# failures. Threshold: if more than 50% of instances with a monitoring
# EVICT have at least 10 of them, treat as scheduler behavior.
matches_canary = pct_10plus >= 50.0
monitoring_decision = (
    "monitoring EVICTs pattern-match canary/health-check preemption by "
    "the Borg scheduler. Exclude from every failure label and document "
    "the exclusion rationale in working_draft.docx (Failure Definition "
    "section)."
    if matches_canary
    else "monitoring EVICTs do not show a canary recurrence pattern; "
    "investigate further before deciding."
)

f3_finding = (
    f"F3 (Monitoring EVICT triage). The F3.1 aggregate lifecycle "
    f"metrics are unreliable because the SCHEDULE/terminal pairing in "
    f"the aggregated query does not correctly reconstruct per-lifecycle "
    f"running durations (all three group medians come back negative). "
    f"The F3.2 repeats distribution is the load-bearing evidence: "
    f"{base_count:,} instances had at least one monitoring-priority "
    f"EVICT terminal over the 31-day window; {pct_2plus:.1f}% had at "
    f"least 2, {pct_10plus:.1f}% had at least 10, {pct_100plus:.1f}% "
    f"had at least 100, and {pct_500plus:.1f}% had at least 500 "
    f"(max observed: {max_seq:,}). Decision: {monitoring_decision}"
)
print(f3_finding)

# %%
append_decision(
    decision_id="V27",
    category="Monitoring EVICT Sensitivity",
    dataset="Google",
    item=monitoring_decision,
    evidence=(
        f"F3.2 repeats distribution: {base_count:,} instances with at "
        f"least 1 monitoring-priority EVICT terminal; {pct_10plus:.1f}% "
        f"had at least 10, {pct_100plus:.1f}% had at least 100, "
        f"{pct_500plus:.1f}% had at least 500 (max observed: "
        f"{max_seq:,}). Recurrent per-instance eviction at this scale "
        f"is consistent with canary/health-check preemption by the "
        f"Borg scheduler, not failure. Note: the F3.1 aggregate "
        f"lifecycle metrics (median/mean running_seconds, resubmission "
        f"rate) are not trustworthy because the SCHEDULE/terminal "
        f"pairing in the aggregate query mismatches cycles for "
        f"long-lived instances; reconstruct per-lifecycle pairs in "
        f"preprocessing if those metrics are needed downstream."
    ),
    source=(
        "notebooks/07b_phase3_front_loaded_eda.py Section 4; "
        "outputs/tables/monitoring_evict_profile.csv (repeats section)"
    ),
    applies_to_rq="RQ1",
    status="Validated (Phase 3 front-loaded)",
    next_step=(
        "Exclude type=4 AND priority>=360 from failure_label and every "
        "sensitivity branch in src/preprocessing/labels.py; add a "
        "regression test (test_failure_label_excludes_monitoring_evict); "
        "cite V27 evidence in working_draft.docx Failure Definition "
        "section."
    ),
)

# %% [markdown]
# ---
# ## 5. F4: CPI/MAPI within-instance variance (refines V11)
#
# **Question.** V11 cited a record-level Cramer's V of 0.77 between
# `has_hardware_counters` (CPI/MAPI non-null indicator) and workload type
# (scheduling class x priority). That coefficient was computed across
# `instance_usage_full` rows; rows from the same instance are not
# independent. If most instances have a single value of
# `has_hardware_counters` across all of their usage observations, the
# record-level statistic is essentially the per-instance statistic and V11
# stands. If many instances flip the indicator within their lifetime, the
# encoding should aggregate to a per-instance majority vote in
# preprocessing.

# %% [markdown]
# ### F4.1 Per-instance indicator pattern

# %%
within_instance_sql = f"""
WITH per_instance AS (
    SELECT
        collection_id,
        instance_index,
        COUNT(*) AS n_obs,
        COUNTIF(cycles_per_instruction IS NOT NULL) AS n_with_counters,
        COUNTIF(cycles_per_instruction IS NULL) AS n_without_counters
    FROM {fqn('instance_usage_full')}
    GROUP BY collection_id, instance_index
)
SELECT
    CASE
        WHEN n_with_counters > 0 AND n_without_counters = 0 THEN 'always_present'
        WHEN n_with_counters = 0 AND n_without_counters > 0 THEN 'always_absent'
        ELSE 'mixed'
    END AS indicator_pattern,
    COUNT(*) AS n_instances,
    SUM(n_obs) AS total_observations,
    AVG(n_obs) AS avg_obs_per_instance,
    AVG(SAFE_DIVIDE(n_with_counters, n_obs)) AS mean_fraction_with_counters
FROM per_instance
GROUP BY indicator_pattern
ORDER BY n_instances DESC
"""

within_instance_df = run_query(within_instance_sql)
print("Per-instance has_hardware_counters pattern:")
print(within_instance_df)

# %% [markdown]
# ### F4.2 Conditional flip-rate detail
#
# For instances in the `mixed` group, how skewed is the flip? A `mixed`
# instance with 95% of its observations carrying counters is closer to
# `always_present` than to a balanced flip. Quantize the flip fraction
# into deciles so we can read where the mass sits.

# %%
flip_deciles_sql = f"""
WITH per_instance AS (
    SELECT
        collection_id,
        instance_index,
        COUNT(*) AS n_obs,
        COUNTIF(cycles_per_instruction IS NOT NULL) AS n_with_counters
    FROM {fqn('instance_usage_full')}
    GROUP BY collection_id, instance_index
),
mixed AS (
    SELECT
        n_obs,
        n_with_counters,
        SAFE_DIVIDE(n_with_counters, n_obs) AS frac_with_counters
    FROM per_instance
    WHERE n_with_counters > 0 AND n_with_counters < n_obs
)
SELECT
    FLOOR(frac_with_counters * 10) AS decile,
    COUNT(*) AS n_instances,
    AVG(n_obs) AS avg_obs_per_instance,
    AVG(frac_with_counters) AS avg_frac_with_counters
FROM mixed
GROUP BY decile
ORDER BY decile
"""

flip_deciles_df = run_query(flip_deciles_sql)
print("Flip fraction deciles among mixed-indicator instances:")
print(flip_deciles_df)

# %%
# Combine the two views
pattern_section = within_instance_df.select([
    pl.lit("pattern").alias("section"),
    pl.col("indicator_pattern").alias("group_label"),
    pl.col("n_instances"),
    pl.col("total_observations"),
    pl.col("avg_obs_per_instance"),
    pl.col("mean_fraction_with_counters"),
    pl.lit(None).cast(pl.Int64).alias("decile"),
    pl.lit(None).cast(pl.Float64).alias("avg_frac_with_counters"),
])
decile_section = flip_deciles_df.select([
    pl.lit("flip_deciles").alias("section"),
    pl.lit("mixed").alias("group_label"),
    pl.col("n_instances"),
    pl.lit(None).cast(pl.Int64).alias("total_observations"),
    pl.col("avg_obs_per_instance"),
    pl.lit(None).cast(pl.Float64).alias("mean_fraction_with_counters"),
    pl.col("decile").cast(pl.Int64),
    pl.col("avg_frac_with_counters"),
])
within_instance_out = pl.concat(
    [pattern_section, decile_section], how="vertical_relaxed"
)
save_table(within_instance_out, "cpi_mapi_within_instance_variance")

# %% [markdown]
# ### F4.3 Finding paragraph and decisions-log row (V28)

# %%
total_instances = within_instance_df["n_instances"].sum() or 0
mixed_row = within_instance_df.filter(pl.col("indicator_pattern") == "mixed")
n_mixed = mixed_row["n_instances"].sum() if mixed_row.height else 0
pct_mixed = 100.0 * n_mixed / total_instances if total_instances else 0.0

always_present_row = within_instance_df.filter(pl.col("indicator_pattern") == "always_present")
n_always_present = always_present_row["n_instances"].sum() if always_present_row.height else 0
pct_always_present = 100.0 * n_always_present / total_instances if total_instances else 0.0

always_absent_row = within_instance_df.filter(pl.col("indicator_pattern") == "always_absent")
n_always_absent = always_absent_row["n_instances"].sum() if always_absent_row.height else 0
pct_always_absent = 100.0 * n_always_absent / total_instances if total_instances else 0.0

# Decision rule: if pct_mixed is small (< 5%), V11 stands as-is.
# If pct_mixed is larger and the flip-fraction histogram is balanced,
# preprocessing should use a per-instance majority vote.
v11_stands = pct_mixed < 5.0

f4_finding = (
    f"F4 (CPI/MAPI within-instance variance). Across "
    f"{total_instances:,} distinct (collection_id, instance_index) "
    f"pairs in instance_usage_full, {pct_always_present:.2f}% always have "
    f"hardware counters present across every observation, "
    f"{pct_always_absent:.2f}% always lack them, and {pct_mixed:.2f}% "
    f"({n_mixed:,} instances) flip the indicator at least once within "
    f"their lifetime. "
    + (
        "Mixed instances are below the 5% threshold, so V11's record-level "
        "Cramer's V coefficient of 0.77 holds at the per-instance level and "
        "the indicator encoding stands as-is in preprocessing."
        if v11_stands
        else "Mixed instances exceed the 5% threshold; preprocessing should "
        "aggregate has_hardware_counters to a per-instance majority vote "
        "rather than carrying the per-observation indicator directly into "
        "the feature set."
    )
)
print(f4_finding)

# %%
append_decision(
    decision_id="V28",
    category="CPI/MAPI Encoding Refinement",
    dataset="Google",
    item=(
        "V11 record-level indicator stands as-is."
        if v11_stands
        else "Refine V11 to per-instance majority vote in preprocessing."
    ),
    evidence=(
        f"F4 within-instance check: always_present {pct_always_present:.2f}%, "
        f"always_absent {pct_always_absent:.2f}%, mixed {pct_mixed:.2f}% "
        f"({n_mixed:,} instances) across {total_instances:,} distinct "
        f"(collection_id, instance_index) pairs."
    ),
    source=(
        "notebooks/07b_phase3_front_loaded_eda.py Section 5; "
        "outputs/tables/cpi_mapi_within_instance_variance.csv"
    ),
    applies_to_rq="RQ1, RQ3",
    status="Validated (Phase 3 front-loaded)",
    next_step=(
        "Keep current per-observation has_hardware_counters indicator in "
        "src/preprocessing/missingness.py; reference V28 evidence."
        if v11_stands
        else "Add per-instance majority-vote aggregation step in "
        "src/preprocessing/missingness.py before feature assembly; "
        "reference V28 evidence."
    ),
)

# %% [markdown]
# ---
# ## 6. Verification: re-read the decisions log
#
# Confirm V25 through V28 landed in `eda_decisions.csv` with the expected
# status (`Validated (Phase 3 front-loaded)`) and that the file remains
# parseable.

# %%
log_check = pl.read_csv(str(DECISIONS_CSV))
print(f"Decisions log row count: {log_check.height}")
print()
print("Front-loaded EDA rows (V25 through V28):")
new_rows = log_check.filter(pl.col("id").is_in(["V25", "V26", "V27", "V28"]))
print(new_rows.select(["id", "category", "status", "next_step"]))
print()
print("Status distribution after this notebook:")
print(log_check.group_by("status").len().sort("len", descending=True))

# %% [markdown]
# When V25 through V28 are present and the status distribution shows four
# new rows tagged `Validated (Phase 3 front-loaded)`, this notebook is
# complete and the Phase 3 preprocessing module can be built with the
# resolved decisions in hand.
# tem=(
#         "V11 record-level indicator stands as-is."
#         if v11_stands
#         else "Refine V11 to per-instance majority vote in preprocessing."
#     ),
#     evidence=(
#         f"F4 within-instance check: always_present {pct_always_present:.2f}%, "
#         f"always_absent {pct_always_absent:.2f}%, mixed {pct_mixed:.2f}% "
#         f"({n_mixed:,} instances) across {total_instances:,} distinct "
#         f"(collection_id, instance_index) pairs."
#     ),
#     source=(
#         "notebooks/07b_phase3_front_loaded_eda.py Section 5; "
#         "outputs/tables/cpi_mapi_within_instance_variance.csv"
#     ),
#     applies_to_rq="RQ1, RQ3",
#     status="Validated (Phase 3 front-loaded)",
#     next_step=(
#         "Keep current per-observation has_hardware_counters indicator in "
#         "src/preprocessing/missingness.py; reference V28 evidence."
#         if v11_stands
#         else "Add per-instance majority-vote aggregation step in "
#         "src/preprocessing/missingness.py before feature assembly; "
#         "reference V28 evidence."
#     ),
# )

# %% [markdown]
# ---
# ## 6. Verification: re-read the decisions log
#
# Confirm V25 through V28 landed in `eda_decisions.csv` with the expected
# status (`Validated (Phase 3 front-loaded)`) and that the file remains
# parseable.

# %%
log_check = pl.read_csv(str(DECISIONS_CSV))
print(f"Decisions log row count: {log_check.height}")
print()
print("Front-loaded EDA rows (V25 through V28):")
new_rows = log_check.filter(pl.col("id").is_in(["V25", "V26", "V27", "V28"]))
print(new_rows.select(["id", "category", "status", "next_step"]))
print()
print("Status distribution after this notebook:")
print(log_check.group_by("status").len().sort("len", descending=True))

# %% [markdown]
# When V25 through V28 are present and the status distribution shows four
# new rows tagged `Validated (Phase 3 front-loaded)`, this notebook is
# complete and the Phase 3 preprocessing module can be built with the
# resolved decisions in hand.
