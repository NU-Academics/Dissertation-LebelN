# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 05 — Backblaze Hard Drive Data: Deep Exploratory Data Analysis
#
# **Purpose:** Comprehensive EDA of the Backblaze Hard Drive dataset (2013–2025)
# to characterize failure patterns, identify predictive SMART attributes, assess
# temporal drift for RQ5, and compare failure dynamics with the Google Cluster
# Traces findings from notebook 03.
#
# **Inputs:** Parquet files on GCS produced by notebook 04.
# - `gs://{GCS_BUCKET}/backblaze_parquet/*.parquet` (one per zip: annual or quarterly)
#
# **Outputs (saved to Drive):**
# - Tables (CSV): `outputs/tables/bb_*.csv`
# - Figures (PNG): `outputs/figures/bb_*.png`
#
# **Sections:**
# 1. Dataset Overview — dimensions, years, models, failure rates
# 2. Failure Analysis — rates by year/model/capacity, time-to-failure, seasonality
# 3. SMART Attribute Profiling — availability, distributions, discriminative power
# 4. Drive Model Analysis — fleet composition, model-specific patterns
# 5. Temporal Patterns (RQ5) — concept drift, distribution shift, fleet evolution
# 6. Cross-Dataset Comparison — Google (rapid-onset) vs. Backblaze (gradual degradation)
# 7. Key Findings Summary — all-markdown synthesis
#
# **Prerequisites:**
# - Notebook 04 completed (Parquet files on GCS)
# - Colab Secrets: `GCP_PROJECT_ID`

# %% [markdown]
# ---
# ## 0. Colab Session Setup

# %%
# !pip install -q polars matplotlib seaborn scikit-learn google-cloud-storage

# %%
from google.colab import userdata

PROJECT_ID = userdata.get('GCP_PROJECT_ID')
print(f"GCP Project: {PROJECT_ID}")

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
from google.colab import auth
auth.authenticate_user()

# %%
from pathlib import Path

DRIVE_PATH = Path('/content/drive/MyDrive/Dissertation_Colab')
TABLES_DIR = DRIVE_PATH / 'outputs' / 'tables'
FIGURES_DIR = DRIVE_PATH / 'outputs' / 'figures'
CHECKPOINT_DIR = DRIVE_PATH / 'checkpoints'

# Local temp directory for Parquet files downloaded from GCS
BACKBLAZE_DIR = Path('/content/backblaze_parquet')

for dir_path in [TABLES_DIR, FIGURES_DIR, CHECKPOINT_DIR, BACKBLAZE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# %%
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import gc

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.figsize'] = (12, 5)

# %% [markdown]
# ### Download Parquet Files from GCS
#
# Parquet files live on GCS (written by notebook 04). We download them to
# Colab's local NVMe for fast `scan_parquet` access during the EDA session.

# %%
from google.cloud import storage

GCS_BUCKET = f'{PROJECT_ID}-dissertation-data'
GCS_PARQUET_PREFIX = 'backblaze_parquet/'

gcs_client = storage.Client(project=PROJECT_ID)
bucket = gcs_client.bucket(GCS_BUCKET)

parquet_blobs = sorted(
    [b for b in bucket.list_blobs(prefix=GCS_PARQUET_PREFIX)
     if b.name.endswith('.parquet')],
    key=lambda b: b.name,
)

print(f"Found {len(parquet_blobs)} Parquet files on "
      f"gs://{GCS_BUCKET}/{GCS_PARQUET_PREFIX}\n")

for b in parquet_blobs:
    name = b.name.split('/')[-1]
    local_path = BACKBLAZE_DIR / name
    if local_path.exists() and local_path.stat().st_size == b.size:
        print(f"  {name:35s}  {b.size/1024**2:>8.1f} MB  (cached)")
    else:
        print(f"  {name:35s}  {b.size/1024**2:>8.1f} MB  downloading...",
              end='', flush=True)
        b.download_to_filename(str(local_path))
        print("  done")

print(f"\nAll Parquet files available at {BACKBLAZE_DIR}")

# %% [markdown]
# ### Helper Functions

# %%
def save_table(df: pl.DataFrame, name: str) -> None:
    """Save a Polars DataFrame as CSV to the Drive tables directory."""
    path = TABLES_DIR / f"{name}.csv"
    df.write_csv(str(path))
    print(f"  Saved: {path}  ({len(df)} rows)")


def save_figure(fig, name: str) -> None:
    """Save a matplotlib figure as PNG to the Drive figures directory."""
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(str(path), bbox_inches='tight')
    print(f"  Saved: {path}")


def save_checkpoint(obj, name: str) -> None:
    """Pickle an object to Drive checkpoints directory."""
    import pickle
    path = CHECKPOINT_DIR / f"{name}.pkl"
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(name: str):
    """Load a pickled checkpoint, or return None if absent."""
    import pickle
    path = CHECKPOINT_DIR / f"{name}.pkl"
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


def fmt_millions(x, _):
    return f'{x/1e6:.1f}M'


def fmt_thousands(x, _):
    return f'{x/1e3:.0f}K'


# %%
# Discover local Parquet files (downloaded from GCS above)
parquet_files = sorted(BACKBLAZE_DIR.glob('*.parquet'))
print(f"Found {len(parquet_files)} Parquet files in {BACKBLAZE_DIR}\n")
for pf in parquet_files:
    size_mb = pf.stat().st_size / (1024 * 1024)
    print(f"  {pf.name:35s}  {size_mb:>8.1f} MB")

# %% [markdown]
# ---
# # Section 1: Dataset Overview
#
# Total records, unique drives, unique models, failure counts and rates by year.
# Also document which SMART attributes are available in which years.

# %% [markdown]
# ### 1.1 Yearly Summary Statistics
#
# Scan each Parquet file and compute per-year aggregates. We use `scan_parquet`
# (lazy) to keep memory usage minimal — only the aggregated results are
# collected.

# %%
yearly_stats_ckpt = load_checkpoint('bb_yearly_stats')

if yearly_stats_ckpt is not None:
    yearly_stats = yearly_stats_ckpt
    print("Loaded yearly stats from checkpoint")
else:
    yearly_records = []
    for pf in parquet_files:
        lf = pl.scan_parquet(pf)
        stats = lf.with_columns(
            pl.col('date').dt.year().alias('year')
        ).group_by('year').agg(
            pl.len().alias('total_records'),
            pl.col('serial_number').n_unique().alias('unique_drives'),
            pl.col('model').n_unique().alias('unique_models'),
            pl.col('failure').sum().alias('failure_count'),
        ).collect()

        for row in stats.to_dicts():
            row['source_file'] = pf.stem
            yearly_records.append(row)

    yearly_stats = (
        pl.DataFrame(yearly_records)
        .group_by('year')
        .agg(
            pl.col('total_records').sum(),
            pl.col('unique_drives').sum(),   # approximate — some drives span quarters
            pl.col('unique_models').max(),
            pl.col('failure_count').sum(),
        )
        .with_columns(
            (pl.col('failure_count') / pl.col('total_records') * 100)
            .alias('failure_rate_pct')
        )
        .sort('year')
    )
    save_checkpoint(yearly_stats, 'bb_yearly_stats')

print(yearly_stats.to_pandas().to_string(index=False))
save_table(yearly_stats, 'bb_yearly_summary')

# %% [markdown]
# *Table 1.1 shows the yearly breakdown of the Backblaze dataset. Key numbers to
# note: total records, failure counts, and the daily failure rate (failure_rate_pct
# represents the percentage of daily observations that are failure events). The
# unique_drives count is approximate because drives may span quarterly file
# boundaries.*

# %% [markdown]
# ### 1.2 Grand Totals

# %%
grand_total_records = yearly_stats['total_records'].sum()
grand_total_failures = yearly_stats['failure_count'].sum()
grand_failure_rate = grand_total_failures / grand_total_records * 100
year_min = yearly_stats['year'].min()
year_max = yearly_stats['year'].max()

print(f"Total records:    {grand_total_records:>14,}")
print(f"Total failures:   {grand_total_failures:>14,}")
print(f"Daily failure rate: {grand_failure_rate:.4f}%")
print(f"Years covered:    {year_min} to {year_max}")

# %% [markdown]
# ### 1.3 SMART Attribute Availability by Year
#
# Which SMART IDs are available in each Parquet file? This documents schema
# evolution and informs which attributes can be used across the full dataset.

# %%
schema_evolution = []
for pf in parquet_files:
    schema = pl.read_parquet_schema(pf)
    smart_ids = sorted(
        {col.split('_')[1] for col in schema if col.startswith('smart_')},
        key=int,
    )
    schema_evolution.append({
        'file': pf.stem,
        'total_cols': len(schema),
        'smart_count': len(smart_ids),
        'smart_ids': ', '.join(smart_ids),
    })

schema_evo_df = pl.DataFrame(schema_evolution)
save_table(schema_evo_df, 'bb_schema_evolution')
print(schema_evo_df.select(['file', 'total_cols', 'smart_count']).to_pandas().to_string(index=False))

# %%
# Universal vs. variable SMART IDs
all_id_sets = []
for pf in parquet_files:
    schema = pl.read_parquet_schema(pf)
    ids = {col.split('_')[1] for col in schema if col.startswith('smart_')}
    all_id_sets.append(ids)

universal_ids = sorted(set.intersection(*all_id_sets), key=int) if all_id_sets else []
all_ids = sorted(set.union(*all_id_sets), key=int) if all_id_sets else []
variable_ids = sorted(set(all_ids) - set(universal_ids), key=int)

print(f"SMART IDs in ALL files ({len(universal_ids)}): {', '.join(universal_ids)}")
print(f"SMART IDs in SOME files ({len(variable_ids)}): {', '.join(variable_ids)}")

# %% [markdown]
# ---
# # Section 2: Failure Analysis
#
# Overall failure rate, failure rates by year/model/capacity, time-to-failure
# distributions, and seasonal patterns.

# %% [markdown]
# ### 2.1 Failure Rate by Year (Line Chart)

# %%
fig, ax = plt.subplots(figsize=(14, 5))
years = yearly_stats['year'].to_list()
rates = yearly_stats['failure_rate_pct'].to_list()
counts = yearly_stats['failure_count'].to_list()

ax.plot(years, rates, 'o-', color='steelblue', linewidth=2, markersize=6)
ax.set_xlabel('Year')
ax.set_ylabel('Daily Failure Rate (%)')
ax.set_title('Figure 2.1a: Backblaze Daily Failure Rate by Year')
ax.set_xticks(years)
ax.set_xticklabels(years, rotation=45)

ax2 = ax.twinx()
ax2.bar(years, counts, alpha=0.2, color='coral', label='Failure count')
ax2.set_ylabel('Failure Count')
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_thousands))

fig.tight_layout()
save_figure(fig, 'bb_failure_rate_by_year')
plt.show()

# %% [markdown]
# *Figure 2.1a shows the daily failure rate (blue line) and absolute failure count
# (coral bars) over time. The daily failure rate is the fraction of daily drive
# observations that record a failure=1 event. Year-over-year trends reveal
# whether the fleet is becoming more or less reliable.*

# %% [markdown]
# ### 2.2 Failure Rate by Drive Model (Top 20 by Volume)
#
# Which models dominate the fleet, and what are their failure rates? We scan
# across all files, aggregate per model, and rank by observation count.

# %%
model_stats_ckpt = load_checkpoint('bb_model_stats')

if model_stats_ckpt is not None:
    model_stats = model_stats_ckpt
    print("Loaded model stats from checkpoint")
else:
    model_frames = []
    for pf in parquet_files:
        lf = pl.scan_parquet(pf)
        chunk = lf.group_by('model').agg(
            pl.len().alias('obs_count'),
            pl.col('failure').sum().alias('fail_count'),
            pl.col('serial_number').n_unique().alias('drive_count'),
        ).collect()
        model_frames.append(chunk)

    model_stats = (
        pl.concat(model_frames)
        .group_by('model')
        .agg(
            pl.col('obs_count').sum(),
            pl.col('fail_count').sum(),
            pl.col('drive_count').sum(),  # approximate — drives span files
        )
        .with_columns(
            (pl.col('fail_count') / pl.col('obs_count') * 100).alias('fail_rate_pct'),
            (pl.col('fail_count') / pl.col('drive_count') * 100).alias('drive_fail_pct'),
        )
        .sort('obs_count', descending=True)
    )
    del model_frames
    gc.collect()
    save_checkpoint(model_stats, 'bb_model_stats')

save_table(model_stats, 'bb_model_failure_rates')

top20 = model_stats.head(20)
print(top20.to_pandas().to_string(index=False))

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Left: observation count
models = top20['model'].to_list()
obs = top20['obs_count'].to_list()
y_pos = range(len(models))

ax1.barh(y_pos, obs, color='steelblue')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(models, fontsize=9)
ax1.invert_yaxis()
ax1.set_xlabel('Observation Count')
ax1.set_title('Top 20 Models by Volume')
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_millions))

# Right: failure rate
fail_rates = top20['fail_rate_pct'].to_list()
colors = ['coral' if r > np.median(fail_rates) else 'steelblue' for r in fail_rates]

ax2.barh(y_pos, fail_rates, color=colors)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(models, fontsize=9)
ax2.invert_yaxis()
ax2.set_xlabel('Daily Failure Rate (%)')
ax2.set_title('Failure Rate (Top 20 Models)')

fig.suptitle('Figure 2.2a: Fleet Composition and Failure Rates by Drive Model',
             fontsize=13, y=1.02)
fig.tight_layout()
save_figure(fig, 'bb_model_failure_rates')
plt.show()

# %% [markdown]
# *Figure 2.2a shows the top 20 drive models by observation count (left) and
# their daily failure rates (right). Models with failure rates above the median
# are highlighted in coral. High-volume models with elevated failure rates are
# the most important for training — they provide both sufficient data and enough
# positive cases for the minority class.*

# %% [markdown]
# ### 2.3 Failure Rate by Capacity Bucket

# %%
capacity_stats_ckpt = load_checkpoint('bb_capacity_stats')

if capacity_stats_ckpt is not None:
    capacity_stats = capacity_stats_ckpt
    print("Loaded capacity stats from checkpoint")
else:
    cap_frames = []
    for pf in parquet_files:
        lf = pl.scan_parquet(pf)
        chunk = lf.with_columns(
            (pl.col('capacity_bytes') / 1e12).round(1).alias('capacity_tb')
        ).group_by('capacity_tb').agg(
            pl.len().alias('obs_count'),
            pl.col('failure').sum().alias('fail_count'),
        ).collect()
        cap_frames.append(chunk)

    capacity_stats = (
        pl.concat(cap_frames)
        .group_by('capacity_tb')
        .agg(
            pl.col('obs_count').sum(),
            pl.col('fail_count').sum(),
        )
        .with_columns(
            (pl.col('fail_count') / pl.col('obs_count') * 100).alias('fail_rate_pct')
        )
        .sort('capacity_tb')
    )
    del cap_frames
    gc.collect()
    save_checkpoint(capacity_stats, 'bb_capacity_stats')

save_table(capacity_stats, 'bb_capacity_failure_rates')
print(capacity_stats.to_pandas().to_string(index=False))

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

caps = capacity_stats['capacity_tb'].to_list()
obs = capacity_stats['obs_count'].to_list()
rates = capacity_stats['fail_rate_pct'].to_list()

ax1.bar([str(c) for c in caps], obs, color='steelblue')
ax1.set_xlabel('Capacity (TB)')
ax1.set_ylabel('Observation Count')
ax1.set_title('Fleet Size by Capacity')
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_millions))
ax1.tick_params(axis='x', rotation=45)

ax2.bar([str(c) for c in caps], rates, color='coral')
ax2.set_xlabel('Capacity (TB)')
ax2.set_ylabel('Daily Failure Rate (%)')
ax2.set_title('Failure Rate by Capacity')
ax2.tick_params(axis='x', rotation=45)

fig.suptitle('Figure 2.3a: Failure Rate by Drive Capacity',
             fontsize=13, y=1.02)
fig.tight_layout()
save_figure(fig, 'bb_failure_rate_by_capacity')
plt.show()

# %% [markdown]
# *Figure 2.3a shows fleet composition and failure rates broken down by drive
# capacity. This reveals whether certain capacity tiers (typically older, smaller
# drives or newer, larger drives) have systematically different reliability.*

# %% [markdown]
# ### 2.4 Time-to-Failure Distribution
#
# For drives that failed: how many days of observation exist before the failure
# event? This is the "observation window" — the amount of temporal history
# available for building features. We scan each file for failure events and
# compute per-drive observation span.

# %%
ttf_ckpt = load_checkpoint('bb_time_to_failure')

if ttf_ckpt is not None:
    ttf_df = ttf_ckpt
    print("Loaded time-to-failure from checkpoint")
else:
    ttf_frames = []
    for pf in parquet_files:
        lf = pl.scan_parquet(pf)
        # Get drives that have at least one failure in this file
        failed_drives = lf.filter(pl.col('failure') == 1).select('serial_number').unique()
        # For those drives, get their full observation span in this file
        chunk = (
            lf.join(failed_drives, on='serial_number', how='semi')
            .group_by('serial_number')
            .agg(
                pl.col('date').min().alias('first_seen'),
                pl.col('date').max().alias('last_seen'),
                pl.len().alias('obs_days'),
                pl.col('failure').sum().alias('n_failures'),
            )
            .collect()
        )
        ttf_frames.append(chunk)

    # Combine — a drive may appear in multiple files
    ttf_combined = pl.concat(ttf_frames)
    ttf_df = (
        ttf_combined
        .group_by('serial_number')
        .agg(
            pl.col('first_seen').min(),
            pl.col('last_seen').max(),
            pl.col('obs_days').sum(),
            pl.col('n_failures').sum(),
        )
        .with_columns(
            (pl.col('last_seen') - pl.col('first_seen')).dt.total_days().alias('span_days')
        )
    )
    del ttf_frames, ttf_combined
    gc.collect()
    save_checkpoint(ttf_df, 'bb_time_to_failure')

print(f"Failed drives: {len(ttf_df):,}")
print(f"Span (days) — median: {ttf_df['span_days'].median():.0f}, "
      f"mean: {ttf_df['span_days'].mean():.0f}, "
      f"p25: {ttf_df['span_days'].quantile(0.25):.0f}, "
      f"p75: {ttf_df['span_days'].quantile(0.75):.0f}")

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

span_vals = ttf_df['span_days'].drop_nulls().to_list()

ax1.hist(span_vals, bins=100, color='steelblue', edgecolor='white', linewidth=0.3)
ax1.set_xlabel('Days from First Observation to Failure')
ax1.set_ylabel('Number of Drives')
ax1.set_title('Time-to-Failure Distribution (All)')

# Zoom into first 365 days
short = [s for s in span_vals if s <= 365]
ax2.hist(short, bins=50, color='coral', edgecolor='white', linewidth=0.3)
ax2.set_xlabel('Days from First Observation to Failure')
ax2.set_ylabel('Number of Drives')
ax2.set_title('Time-to-Failure (First Year Only)')

fig.suptitle('Figure 2.4a: Time-to-Failure Distributions for Failed Drives',
             fontsize=13, y=1.02)
fig.tight_layout()
save_figure(fig, 'bb_time_to_failure')
plt.show()

# %% [markdown]
# *Figure 2.4a shows how long failed drives were observed before their failure
# event. The left panel shows the full distribution; the right panel zooms into
# the first year. A long tail means many drives have extensive pre-failure
# history — ideal for building temporal features (rolling SMART statistics,
# rate-of-change). Drives with very short histories may have been installed
# near failure and provide less predictive signal.*

# %% [markdown]
# ### 2.5 Seasonal Patterns
#
# Are failures more common in certain months? Aggregate failure counts and rates
# by calendar month across all years.

# %%
seasonal_ckpt = load_checkpoint('bb_seasonal')

if seasonal_ckpt is not None:
    monthly_stats = seasonal_ckpt
    print("Loaded seasonal stats from checkpoint")
else:
    monthly_frames = []
    for pf in parquet_files:
        lf = pl.scan_parquet(pf)
        chunk = lf.with_columns(
            pl.col('date').dt.month().alias('month')
        ).group_by('month').agg(
            pl.len().alias('obs_count'),
            pl.col('failure').sum().alias('fail_count'),
        ).collect()
        monthly_frames.append(chunk)

    monthly_stats = (
        pl.concat(monthly_frames)
        .group_by('month')
        .agg(
            pl.col('obs_count').sum(),
            pl.col('fail_count').sum(),
        )
        .with_columns(
            (pl.col('fail_count') / pl.col('obs_count') * 100).alias('fail_rate_pct')
        )
        .sort('month')
    )
    del monthly_frames
    gc.collect()
    save_checkpoint(monthly_stats, 'bb_seasonal')

save_table(monthly_stats, 'bb_seasonal_failure_rates')
print(monthly_stats.to_pandas().to_string(index=False))

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

months = monthly_stats['month'].to_list()
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# Handle case where not all 12 months are present
labels = [month_labels[m - 1] for m in months]

ax1.bar(labels, monthly_stats['fail_count'].to_list(), color='coral')
ax1.set_xlabel('Month')
ax1.set_ylabel('Failure Count')
ax1.set_title('Absolute Failures by Month')
ax1.tick_params(axis='x', rotation=45)

ax2.plot(labels, monthly_stats['fail_rate_pct'].to_list(), 'o-', color='steelblue',
         linewidth=2, markersize=6)
ax2.set_xlabel('Month')
ax2.set_ylabel('Daily Failure Rate (%)')
ax2.set_title('Failure Rate by Month')
ax2.tick_params(axis='x', rotation=45)

fig.suptitle('Figure 2.5a: Seasonal Failure Patterns', fontsize=13, y=1.02)
fig.tight_layout()
save_figure(fig, 'bb_seasonal_patterns')
plt.show()

# %% [markdown]
# *Figure 2.5a shows failure counts (left) and failure rates (right) by
# calendar month, aggregated across all years. Seasonal patterns may reflect
# environmental factors (temperature, humidity) or operational patterns
# (hardware refresh cycles, deployment waves).*

# %% [markdown]
# ---
# # Section 3: SMART Attribute Profiling
#
# For each SMART attribute: availability, distribution differences between
# healthy and failed drives, and discriminative power. This section identifies
# which SMART attributes matter most — driven by data, not literature
# assumptions.

# %% [markdown]
# ### 3.1 SMART Attribute Availability
#
# Not all SMART attributes are populated for all drives. Compute the fraction
# of non-null values for each SMART column across the full dataset. We use
# only universal SMART IDs (present in ALL files) for consistency.

# %%
# Build the list of universal raw columns to profile
# (normalized columns are less informative for EDA — raw values are more expressive)
universal_raw_cols = [f'smart_{sid}_raw' for sid in universal_ids]
universal_norm_cols = [f'smart_{sid}_normalized' for sid in universal_ids]

print(f"Universal SMART IDs to profile: {len(universal_ids)}")

# %%
availability_ckpt = load_checkpoint('bb_smart_availability')

if availability_ckpt is not None:
    avail_df = availability_ckpt
    print("Loaded SMART availability from checkpoint")
else:
    total_rows_seen = 0
    nonnull_counts = {col: 0 for col in universal_raw_cols}

    for pf in parquet_files:
        # Only select columns that exist in this file
        schema = pl.read_parquet_schema(pf)
        cols_in_file = [c for c in universal_raw_cols if c in schema]

        lf = pl.scan_parquet(pf)
        stats = lf.select(
            pl.len().alias('n_rows'),
            *[pl.col(c).is_not_null().sum().alias(f'{c}_nn') for c in cols_in_file]
        ).collect()

        total_rows_seen += stats['n_rows'][0]
        for c in cols_in_file:
            nonnull_counts[c] += stats[f'{c}_nn'][0]

    avail_records = []
    for col in universal_raw_cols:
        smart_id = col.split('_')[1]
        nn = nonnull_counts[col]
        avail_records.append({
            'smart_id': int(smart_id),
            'column': col,
            'nonnull_count': nn,
            'nonnull_pct': round(nn / total_rows_seen * 100, 2) if total_rows_seen else 0,
        })

    avail_df = pl.DataFrame(avail_records).sort('smart_id')
    save_checkpoint(avail_df, 'bb_smart_availability')

save_table(avail_df, 'bb_smart_availability')

# Show attributes sorted by availability
avail_sorted = avail_df.sort('nonnull_pct', descending=True)
print(avail_sorted.to_pandas().to_string(index=False))

# %%
# Identify near-empty and fully-populated attributes
high_avail = avail_df.filter(pl.col('nonnull_pct') >= 50)
low_avail = avail_df.filter(pl.col('nonnull_pct') < 10)
mid_avail = avail_df.filter(
    (pl.col('nonnull_pct') >= 10) & (pl.col('nonnull_pct') < 50)
)

print(f"\nHigh availability (>=50%): {len(high_avail)} attributes")
print(f"Mid availability (10-50%): {len(mid_avail)} attributes")
print(f"Low availability (<10%):   {len(low_avail)} attributes — candidates for dropping")

# %% [markdown]
# ### 3.2 SMART Attribute Discrimination: Failed vs. Healthy Drives
#
# For each SMART attribute with reasonable availability (>=50%), compare the
# distribution of the _raw value on the last observation of failed drives
# vs. a random sample of healthy drives. Compute the AUC of a single-feature
# classifier (rank-based) to quantify discriminative power.
#
# **Strategy:** We cannot load all rows into memory. Instead, for each Parquet
# file we extract (a) the last observation of every drive that failed, and
# (b) a sample of healthy drive observations. Then we concatenate across files.

# %%
# Columns to profile — only high-availability raw columns
profile_cols = high_avail['column'].to_list()
print(f"Profiling {len(profile_cols)} SMART attributes with >=50% availability")

# %%
disc_ckpt = load_checkpoint('bb_smart_discrimination')

if disc_ckpt is not None:
    disc_df = disc_ckpt
    print("Loaded discrimination data from checkpoint")
else:
    failed_frames = []
    healthy_frames = []
    select_cols = ['serial_number', 'date', 'failure', 'model'] + profile_cols

    for pf in parquet_files:
        schema = pl.read_parquet_schema(pf)
        cols_in_file = [c for c in select_cols if c in schema]

        lf = pl.scan_parquet(pf).select(cols_in_file)

        # Failed drives: last observation (the day of failure)
        failed_chunk = (
            lf.filter(pl.col('failure') == 1)
            .collect()
        )
        if len(failed_chunk) > 0:
            failed_frames.append(failed_chunk)

        # Healthy sample: random 0.5% of non-failure observations
        healthy_chunk = (
            lf.filter(pl.col('failure') == 0)
            .collect()
            .sample(fraction=0.005, seed=42)
        )
        if len(healthy_chunk) > 0:
            healthy_frames.append(healthy_chunk)

    failed_all = pl.concat(failed_frames, how='diagonal_relaxed')
    healthy_all = pl.concat(healthy_frames, how='diagonal_relaxed')

    disc_df = pl.concat([
        failed_all.with_columns(pl.lit('failed').alias('status')),
        healthy_all.with_columns(pl.lit('healthy').alias('status')),
    ], how='diagonal_relaxed')

    del failed_frames, healthy_frames, failed_all, healthy_all
    gc.collect()
    save_checkpoint(disc_df, 'bb_smart_discrimination')

print(f"Discrimination dataset: {len(disc_df):,} rows")
print(f"  Failed:  {disc_df.filter(pl.col('status') == 'failed').height:,}")
print(f"  Healthy: {disc_df.filter(pl.col('status') == 'healthy').height:,}")

# %%
# Compute AUC for each SMART attribute (single-feature rank-based classifier)
from sklearn.metrics import roc_auc_score

auc_ckpt = load_checkpoint('bb_smart_auc')

if auc_ckpt is not None:
    auc_df = auc_ckpt
    print("Loaded SMART AUC ranking from checkpoint")
else:
    auc_records = []
    labels = (disc_df['status'] == 'failed').cast(pl.Int8).to_numpy()

    for col in profile_cols:
        if col not in disc_df.columns:
            continue
        values = disc_df[col].to_numpy()
        mask = ~np.isnan(values.astype(float))
        if mask.sum() < 100 or labels[mask].sum() < 10:
            continue
        try:
            auc = roc_auc_score(labels[mask], values[mask])
            # AUC < 0.5 means inverse relationship — flip to show magnitude
            auc_magnitude = max(auc, 1 - auc)
            auc_records.append({
                'smart_id': int(col.split('_')[1]),
                'column': col,
                'auc': round(auc, 4),
                'auc_magnitude': round(auc_magnitude, 4),
                'direction': 'higher=more_failure' if auc >= 0.5 else 'lower=more_failure',
                'n_valid': int(mask.sum()),
            })
        except Exception:
            pass

    auc_df = pl.DataFrame(auc_records).sort('auc_magnitude', descending=True)
    save_checkpoint(auc_df, 'bb_smart_auc')

save_table(auc_df, 'bb_smart_auc_ranking')

# top_smart is used by Sections 4, 5, and 6 — define it here so it's
# always available once this cell has run (even after checkpoint reload).
top_smart = auc_df.head(15)

print(auc_df.to_pandas().to_string(index=False))

# %% [markdown]
# *The AUC ranking shows the discriminative power of each SMART attribute
# acting as a standalone classifier. AUC_magnitude close to 1.0 means the
# attribute strongly separates failed from healthy drives. Direction indicates
# whether higher or lower raw values correlate with failure.*

# %%
# Plot top 15 most discriminative SMART attributes

fig, ax = plt.subplots(figsize=(12, 6))
y_pos = range(len(top_smart))
colors = ['coral' if d == 'higher=more_failure' else 'steelblue'
          for d in top_smart['direction'].to_list()]

ax.barh(y_pos, top_smart['auc_magnitude'].to_list(), color=colors)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"SMART {sid}" for sid in top_smart['smart_id'].to_list()], fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('AUC (magnitude)')
ax.set_title('Figure 3.2a: Top 15 Most Discriminative SMART Attributes')
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (AUC=0.5)')
ax.legend(['Random baseline', 'Higher=failure', 'Lower=failure'])

fig.tight_layout()
save_figure(fig, 'bb_smart_auc_ranking')
plt.show()

# %% [markdown]
# *Figure 3.2a ranks SMART attributes by their ability to separate failed from
# healthy drives. Coral bars indicate attributes where higher values predict
# failure; blue bars indicate attributes where lower values predict failure.
# Compare this data-driven ranking with the literature claims about SMART IDs
# 5, 187, 188, 197, 198.*

# %% [markdown]
# ### 3.3 Literature Comparison: SMART 5, 187, 188, 197, 198
#
# The literature (Cheng et al., 2022; Zhang et al., 2023) highlights these five
# SMART IDs as the most predictive. Check where they rank in our data-driven
# AUC analysis.

# %%
literature_ids = [5, 187, 188, 197, 198]
literature_names = {
    5: 'Reallocated Sectors Count',
    187: 'Reported Uncorrectable Errors',
    188: 'Command Timeout',
    197: 'Current Pending Sector Count',
    198: 'Offline Uncorrectable',
}

print("Literature-highlighted SMART attributes vs. our AUC ranking:\n")
for sid in literature_ids:
    match = auc_df.filter(pl.col('smart_id') == sid)
    avail_match = avail_df.filter(pl.col('smart_id') == sid)
    avail_pct = avail_match['nonnull_pct'][0] if len(avail_match) > 0 else 0

    if len(match) > 0:
        auc_val = match['auc_magnitude'][0]
        rank = auc_df.with_row_index('rank').filter(
            pl.col('smart_id') == sid
        )['rank'][0] + 1
        direction = match['direction'][0]
        print(f"  SMART {sid:>3} ({literature_names[sid]:40s}): "
              f"AUC={auc_val:.4f}  rank={rank:>2}/{len(auc_df)}  "
              f"avail={avail_pct:.1f}%  {direction}")
    else:
        print(f"  SMART {sid:>3} ({literature_names[sid]:40s}): "
              f"NOT IN RANKING (avail={avail_pct:.1f}%)")

# %% [markdown]
# ### 3.4 Distribution Plots for Top Discriminative SMART Attributes
#
# Side-by-side boxplots showing the distribution of failed vs. healthy drives
# for the top 8 most discriminative attributes.

# %%
top8_ids = top_smart.head(8)['smart_id'].to_list()
top8_cols = [f'smart_{sid}_raw' for sid in top8_ids]

n_attrs = len(top8_cols)
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, col in enumerate(top8_cols):
    ax = axes[idx]
    if col not in disc_df.columns:
        ax.set_visible(False)
        continue

    smart_id = col.split('_')[1]

    # Extract data, clip outliers at 99th percentile for visualization
    plot_data = disc_df.select([col, 'status']).drop_nulls()
    if len(plot_data) == 0:
        ax.set_visible(False)
        continue

    p99 = plot_data[col].quantile(0.99)
    plot_data = plot_data.filter(pl.col(col) <= p99)

    failed_vals = plot_data.filter(pl.col('status') == 'failed')[col].to_list()
    healthy_vals = plot_data.filter(pl.col('status') == 'healthy')[col].to_list()

    bp = ax.boxplot(
        [healthy_vals, failed_vals],
        labels=['Healthy', 'Failed'],
        patch_artist=True,
        widths=0.6,
    )
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('coral')
    bp['boxes'][1].set_alpha(0.7)

    ax.set_title(f'SMART {smart_id}', fontsize=11)
    ax.set_ylabel('Raw value')

# Hide any unused subplots
for idx in range(n_attrs, len(axes)):
    axes[idx].set_visible(False)

fig.suptitle('Figure 3.4a: Distribution of Top SMART Attributes (Failed vs. Healthy)',
             fontsize=13, y=1.02)
fig.tight_layout()
save_figure(fig, 'bb_smart_distributions_top8')
plt.show()

# %% [markdown]
# *Figure 3.4a shows boxplots of the top 8 most discriminative SMART attributes,
# comparing failed drives (coral) to healthy drives (blue). Values are clipped
# at the 99th percentile for readability. The visual separation between the two
# groups confirms the AUC-based ranking.*

# %% [markdown]
# ### 3.5 Near-Constant and All-Null SMART Attributes (Drop Candidates)

# %%
drop_candidates = avail_df.filter(pl.col('nonnull_pct') < 5)
print(f"SMART attributes with <5% availability ({len(drop_candidates)} attributes):")
print("  These are candidates for dropping.\n")
print(drop_candidates.to_pandas().to_string(index=False))

# %% [markdown]
# ---
# # Section 4: Drive Model Analysis
#
# Fleet composition, model-specific SMART attribute behavior, and whether
# modeling should be stratified by drive model.

# %% [markdown]
# ### 4.1 Fleet Composition (Top 20 Models)

# %%
top20_models = model_stats.head(20)
total_obs = model_stats['obs_count'].sum()
top20_obs = top20_models['obs_count'].sum()
top20_pct = top20_obs / total_obs * 100

print(f"Top 20 models account for {top20_pct:.1f}% of all observations")
print(f"Total models: {len(model_stats)}")

save_table(top20_models, 'bb_top20_models')

# %% [markdown]
# ### 4.2 Do Predictive SMART Attributes Vary by Model?
#
# For the top 5 models by volume, compute the AUC of the top 5 overall SMART
# attributes. If AUC varies significantly across models, stratified modeling
# may be warranted.

# %%
top5_model_names = model_stats.head(5)['model'].to_list()
top5_smart_cols = [f'smart_{sid}_raw' for sid in top_smart.head(5)['smart_id'].to_list()]

model_auc_records = []
for model_name in top5_model_names:
    model_data = disc_df.filter(pl.col('model') == model_name)
    if len(model_data) < 50:
        continue
    model_labels = (model_data['status'] == 'failed').cast(pl.Int8).to_numpy()
    if model_labels.sum() < 5:
        continue

    for col in top5_smart_cols:
        if col not in model_data.columns:
            continue
        vals = model_data[col].to_numpy()
        mask = ~np.isnan(vals.astype(float))
        if mask.sum() < 50 or model_labels[mask].sum() < 5:
            continue
        try:
            auc = roc_auc_score(model_labels[mask], vals[mask])
            model_auc_records.append({
                'model': model_name,
                'smart_id': int(col.split('_')[1]),
                'auc': round(auc, 4),
                'auc_magnitude': round(max(auc, 1 - auc), 4),
                'n_failed': int(model_labels[mask].sum()),
            })
        except Exception:
            pass

model_auc_df = pl.DataFrame(model_auc_records)
save_table(model_auc_df, 'bb_model_specific_auc')
print(model_auc_df.to_pandas().to_string(index=False))

# %%
if len(model_auc_df) > 0:
    # Pivot for heatmap: models × SMART attributes
    pivot_data = model_auc_df.pivot(
        on='smart_id',
        index='model',
        values='auc_magnitude',
    )
    pivot_np = pivot_data.drop('model').to_numpy()
    model_labels_plot = pivot_data['model'].to_list()
    smart_labels_plot = [str(c) for c in pivot_data.columns if c != 'model']

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot_np, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')

    ax.set_xticks(range(len(smart_labels_plot)))
    ax.set_xticklabels([f'SMART {s}' for s in smart_labels_plot], rotation=45)
    ax.set_yticks(range(len(model_labels_plot)))
    ax.set_yticklabels(model_labels_plot, fontsize=9)

    for i in range(len(model_labels_plot)):
        for j in range(len(smart_labels_plot)):
            val = pivot_np[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9)

    plt.colorbar(im, ax=ax, label='AUC (magnitude)')
    ax.set_title('Figure 4.2a: SMART Attribute AUC by Drive Model (Top 5 Models)')

    fig.tight_layout()
    save_figure(fig, 'bb_model_smart_auc_heatmap')
    plt.show()

# %% [markdown]
# *Figure 4.2a shows how the discriminative power of top SMART attributes varies
# across the 5 highest-volume drive models. Uniform AUC values across models
# suggest a global model is sufficient. Large variation indicates model-specific
# feature importance — a case for stratified modeling or per-model feature
# selection.*

# %% [markdown]
# ---
# # Section 5: Temporal Patterns (RQ5 — Concept Drift)
#
# This section directly supports RQ5 (online learning and concept drift). We
# examine whether SMART attribute distributions and failure rates shift over
# the years, which would motivate the online/incremental learning approach.

# %% [markdown]
# ### 5.1 Failure Rate Trend
#
# Already plotted in Section 2.1. Here we add a formal trend decomposition.

# %%
# Monthly failure rate for finer-grained trend analysis
monthly_rate_ckpt = load_checkpoint('bb_monthly_rate')

if monthly_rate_ckpt is not None:
    monthly_rate = monthly_rate_ckpt
    print("Loaded monthly rate from checkpoint")
else:
    monthly_rate_frames = []
    for pf in parquet_files:
        lf = pl.scan_parquet(pf)
        chunk = lf.with_columns(
            pl.col('date').dt.strftime('%Y-%m').alias('year_month')
        ).group_by('year_month').agg(
            pl.len().alias('obs_count'),
            pl.col('failure').sum().alias('fail_count'),
        ).collect()
        monthly_rate_frames.append(chunk)

    monthly_rate = (
        pl.concat(monthly_rate_frames)
        .group_by('year_month')
        .agg(
            pl.col('obs_count').sum(),
            pl.col('fail_count').sum(),
        )
        .with_columns(
            (pl.col('fail_count') / pl.col('obs_count') * 100).alias('fail_rate_pct')
        )
        .sort('year_month')
    )
    del monthly_rate_frames
    gc.collect()
    save_checkpoint(monthly_rate, 'bb_monthly_rate')

save_table(monthly_rate, 'bb_monthly_failure_rate')

fig, ax = plt.subplots(figsize=(18, 5))
ym = monthly_rate['year_month'].to_list()
fr = monthly_rate['fail_rate_pct'].to_list()
ax.plot(range(len(ym)), fr, color='steelblue', linewidth=1)
ax.fill_between(range(len(ym)), fr, alpha=0.2, color='steelblue')

# Label every 12th month for readability
tick_positions = list(range(0, len(ym), 12))
tick_labels = [ym[i] for i in tick_positions]
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=45)
ax.set_xlabel('Month')
ax.set_ylabel('Daily Failure Rate (%)')
ax.set_title('Figure 5.1a: Monthly Failure Rate Over Time')

fig.tight_layout()
save_figure(fig, 'bb_monthly_failure_rate_trend')
plt.show()

# %% [markdown]
# *Figure 5.1a shows the monthly failure rate across the full dataset. Abrupt
# level shifts or sustained trends are direct evidence of concept drift — the
# underlying failure distribution changes over time. This motivates the online
# learning approach in RQ5.*

# %% [markdown]
# ### 5.2 SMART Attribute Distribution Shift Over Years
#
# For the top 5 most discriminative SMART attributes, compare the distribution
# of raw values across years. We sample a fixed fraction from each file to keep
# memory manageable.

# %%
drift_ckpt = load_checkpoint('bb_smart_drift')

if drift_ckpt is not None:
    drift_df = drift_ckpt
    print("Loaded drift data from checkpoint")
else:
    top5_drift_cols = [f'smart_{sid}_raw' for sid in top_smart.head(5)['smart_id'].to_list()]
    drift_frames = []
    for pf in parquet_files:
        schema = pl.read_parquet_schema(pf)
        cols_in_file = ['date'] + [c for c in top5_drift_cols if c in schema]
        lf = pl.scan_parquet(pf).select(cols_in_file)
        chunk = (
            lf.with_columns(pl.col('date').dt.year().alias('year'))
            .drop('date')
            .collect()
            .sample(fraction=0.002, seed=42)
        )
        drift_frames.append(chunk)

    drift_df = pl.concat(drift_frames, how='diagonal_relaxed')
    del drift_frames
    gc.collect()
    save_checkpoint(drift_df, 'bb_smart_drift')

print(f"Drift sample: {len(drift_df):,} rows")

# %%
top5_drift_cols = [f'smart_{sid}_raw' for sid in top_smart.head(5)['smart_id'].to_list()]
valid_drift_cols = [c for c in top5_drift_cols if c in drift_df.columns]

fig, axes = plt.subplots(len(valid_drift_cols), 1,
                         figsize=(16, 4 * len(valid_drift_cols)))
if len(valid_drift_cols) == 1:
    axes = [axes]

years_in_data = sorted(drift_df['year'].unique().to_list())
palette = sns.color_palette("viridis", len(years_in_data))

for idx, col in enumerate(valid_drift_cols):
    ax = axes[idx]
    smart_id = col.split('_')[1]

    col_data = drift_df.select(['year', col]).drop_nulls()
    if len(col_data) == 0:
        ax.set_visible(False)
        continue

    # Clip at 99th percentile for visualization
    p99 = col_data[col].quantile(0.99)
    col_data = col_data.filter(pl.col(col) <= p99)

    # Compute yearly medians for overlay
    yearly_medians = (
        col_data.group_by('year')
        .agg(pl.col(col).median().alias('median'))
        .sort('year')
    )

    ax.plot(
        yearly_medians['year'].to_list(),
        yearly_medians['median'].to_list(),
        'o-', color='coral', linewidth=2, markersize=6, zorder=3,
    )
    ax.set_xlabel('Year')
    ax.set_ylabel(f'SMART {smart_id} median (raw)')
    ax.set_title(f'SMART {smart_id}: Yearly Median Trend')

fig.suptitle('Figure 5.2a: SMART Attribute Distribution Shift Over Years',
             fontsize=13, y=1.02)
fig.tight_layout()
save_figure(fig, 'bb_smart_drift_yearly')
plt.show()

# %% [markdown]
# *Figure 5.2a tracks the yearly median of the top 5 discriminative SMART
# attributes. Systematic trends (increasing or decreasing over time) indicate
# covariate shift — the feature distributions change even for healthy drives,
# potentially degrading static models. This is key evidence for RQ5's online
# learning hypothesis.*

# %% [markdown]
# ### 5.3 Fleet Evolution: New Models Over Time
#
# When new drive models are introduced into the fleet, the overall data
# distribution shifts. Track which models are active in each year.

# %%
fleet_evo_ckpt = load_checkpoint('bb_fleet_evolution')

if fleet_evo_ckpt is not None:
    fleet_evo = fleet_evo_ckpt
    print("Loaded fleet evolution from checkpoint")
else:
    fleet_frames = []
    for pf in parquet_files:
        lf = pl.scan_parquet(pf)
        chunk = lf.with_columns(
            pl.col('date').dt.year().alias('year')
        ).group_by(['year', 'model']).agg(
            pl.len().alias('obs_count'),
            pl.col('serial_number').n_unique().alias('drive_count'),
        ).collect()
        fleet_frames.append(chunk)

    fleet_evo = (
        pl.concat(fleet_frames)
        .group_by(['year', 'model'])
        .agg(
            pl.col('obs_count').sum(),
            pl.col('drive_count').sum(),
        )
        .sort(['year', 'obs_count'], descending=[False, True])
    )
    del fleet_frames
    gc.collect()
    save_checkpoint(fleet_evo, 'bb_fleet_evolution')

# Models per year
models_per_year = (
    fleet_evo.group_by('year')
    .agg(
        pl.col('model').n_unique().alias('n_models'),
        pl.col('drive_count').sum().alias('total_drives'),
    )
    .sort('year')
)
save_table(models_per_year, 'bb_models_per_year')
print(models_per_year.to_pandas().to_string(index=False))

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

years_fleet = models_per_year['year'].to_list()

ax1.bar(years_fleet, models_per_year['n_models'].to_list(), color='steelblue')
ax1.set_xlabel('Year')
ax1.set_ylabel('Number of Unique Models')
ax1.set_title('Drive Model Count by Year')
ax1.set_xticks(years_fleet)
ax1.set_xticklabels(years_fleet, rotation=45)

ax2.bar(years_fleet, models_per_year['total_drives'].to_list(), color='coral')
ax2.set_xlabel('Year')
ax2.set_ylabel('Total Active Drives')
ax2.set_title('Fleet Size by Year')
ax2.set_xticks(years_fleet)
ax2.set_xticklabels(years_fleet, rotation=45)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_thousands))

fig.suptitle('Figure 5.3a: Fleet Evolution Over Time', fontsize=13, y=1.02)
fig.tight_layout()
save_figure(fig, 'bb_fleet_evolution')
plt.show()

# %% [markdown]
# *Figure 5.3a shows fleet evolution: the number of unique drive models (left)
# and total active drives (right) per year. Fleet growth and model turnover are
# primary drivers of concept drift in the Backblaze dataset.*

# %% [markdown]
# ---
# # Section 6: Cross-Dataset Comparison with Google Cluster Traces
#
# **Critical context from Google EDA (notebook 03):** The Google Traces revealed
# that failures are **rapid-onset crashes** (22-second median running duration),
# NOT slow degradation. Failing instances use less CPU/memory (not more), and
# the dominant signal is pre-scheduling history (72x failure increase for
# resubmitted instances).
#
# Backblaze is expected to be fundamentally different: SMART degradation should
# be **gradual** over days/weeks. This section verifies that assumption and
# documents the architectural implications for modeling.

# %% [markdown]
# ### 6.1 Pre-Failure SMART Degradation Curves
#
# For drives that failed: track how SMART attributes evolve in the 90 days
# leading up to failure. We want to know if degradation is gradual (supporting
# sliding-window features) or abrupt (requiring a different approach).
#
# We focus on the top 3 discriminative SMART attributes and align observations
# by "days before failure."
#
# **Memory strategy:** Instead of loading the full history of sampled drives
# (which OOMs), we build a failure-date lookup first, then process each
# Parquet file one at a time — joining with failure dates and filtering to the
# 90-day window *inside* the loop so only the relevant rows survive.

# %%
degradation_ckpt = load_checkpoint('bb_degradation_curves')

if degradation_ckpt is not None:
    degrad_df = degradation_ckpt
    print("Loaded degradation curves from checkpoint")
else:
    DEGRAD_WINDOW_DAYS = 90
    DEGRAD_SAMPLE_N = 1000

    top3_smart_ids = top_smart.head(3)['smart_id'].to_list()
    top3_cols = [f'smart_{sid}_raw' for sid in top3_smart_ids]

    # Step 1: Get failed drive serial numbers and their failure dates
    failure_events = []
    for pf in parquet_files:
        lf = pl.scan_parquet(pf)
        chunk = (
            lf.filter(pl.col('failure') == 1)
            .select(['serial_number', 'date'])
            .collect()
        )
        failure_events.append(chunk)

    failed_drives = (
        pl.concat(failure_events)
        .group_by('serial_number')
        .agg(pl.col('date').max().alias('failure_date'))
    )
    del failure_events
    gc.collect()
    print(f"Total failed drives: {len(failed_drives):,}")

    # Step 2: Sample down to a tractable number
    if len(failed_drives) > DEGRAD_SAMPLE_N:
        failed_sample = failed_drives.sample(n=DEGRAD_SAMPLE_N, seed=42)
    else:
        failed_sample = failed_drives
    del failed_drives
    gc.collect()

    serial_list = failed_sample['serial_number'].to_list()
    print(f"Sampled {len(serial_list):,} drives for degradation analysis")

    # Step 3: Process each file — join with failure dates and keep only
    # the 90-day window INSIDE the loop (never accumulate full history)
    degrad_frames = []
    for pf in parquet_files:
        schema = pl.read_parquet_schema(pf)
        cols_in_file = ['serial_number', 'date'] + [
            c for c in top3_cols if c in schema
        ]

        chunk = (
            pl.scan_parquet(pf)
            .select(cols_in_file)
            .filter(pl.col('serial_number').is_in(serial_list))
            .collect()
        )
        if len(chunk) == 0:
            del chunk
            gc.collect()
            continue

        # Join with failure dates and filter to the window immediately
        chunk = (
            chunk.join(failed_sample, on='serial_number', how='inner')
            .with_columns(
                (pl.col('failure_date') - pl.col('date'))
                .dt.total_days()
                .alias('days_before_failure')
            )
            .filter(
                pl.col('days_before_failure').is_between(0, DEGRAD_WINDOW_DAYS)
            )
            .drop('failure_date')
        )
        if len(chunk) > 0:
            degrad_frames.append(chunk)
        del chunk
        gc.collect()

    degrad_df = pl.concat(degrad_frames, how='diagonal_relaxed')
    del degrad_frames, failed_sample
    gc.collect()
    save_checkpoint(degrad_df, 'bb_degradation_curves')

print(f"Degradation dataset: {len(degrad_df):,} rows, "
      f"{degrad_df['serial_number'].n_unique():,} drives")

# %%
# Plot degradation curves: median SMART value vs. days before failure
top3_smart_ids = top_smart.head(3)['smart_id'].to_list()
top3_cols = [f'smart_{sid}_raw' for sid in top3_smart_ids]
valid_top3 = [c for c in top3_cols if c in degrad_df.columns]

fig, axes = plt.subplots(1, len(valid_top3), figsize=(6 * len(valid_top3), 5))
if len(valid_top3) == 1:
    axes = [axes]

for idx, col in enumerate(valid_top3):
    ax = axes[idx]
    smart_id = col.split('_')[1]

    # Aggregate by days_before_failure (focus on last 90 days)
    curve_data = (
        degrad_df.filter(pl.col('days_before_failure') <= 90)
        .group_by('days_before_failure')
        .agg(
            pl.col(col).median().alias('median'),
            pl.col(col).quantile(0.25).alias('p25'),
            pl.col(col).quantile(0.75).alias('p75'),
        )
        .sort('days_before_failure', descending=True)
    )

    days = curve_data['days_before_failure'].to_list()
    medians = curve_data['median'].to_list()
    p25 = curve_data['p25'].to_list()
    p75 = curve_data['p75'].to_list()

    ax.plot(days, medians, color='coral', linewidth=2, label='Median')
    ax.fill_between(days, p25, p75, alpha=0.2, color='coral', label='IQR')
    ax.set_xlabel('Days Before Failure')
    ax.set_ylabel(f'SMART {smart_id} (raw)')
    ax.set_title(f'SMART {smart_id}')
    ax.invert_xaxis()
    ax.legend()

fig.suptitle('Figure 6.1a: SMART Degradation Curves (Last 90 Days Before Failure)',
             fontsize=13, y=1.02)
fig.tight_layout()
save_figure(fig, 'bb_degradation_curves')
plt.show()

# %% [markdown]
# *Figure 6.1a shows how the top 3 discriminative SMART attributes evolve in
# the 90 days before failure. The x-axis counts backward from failure day (0).
# **Gradual upward/downward trends** confirm that SMART degradation is slow
# and progressive — supporting the sliding-window feature approach. **Abrupt
# jumps** near day 0 would indicate that degradation is sudden, requiring a
# different detection strategy.*

# %% [markdown]
# ### 6.2 Onset Detection: When Does Degradation Become Detectable?
#
# For each of the top 3 SMART attributes, compare the median value at different
# lead times (90, 60, 30, 14, 7, 1 days before failure) to the healthy-drive
# baseline. This quantifies the "warning window."

# %%
lead_times = [90, 60, 30, 14, 7, 1]
onset_records = []

for col in valid_top3:
    smart_id = col.split('_')[1]

    # Healthy baseline: median from the healthy sample in disc_df
    if col in disc_df.columns:
        healthy_baseline = (
            disc_df.filter(pl.col('status') == 'healthy')[col]
            .median()
        )
    else:
        healthy_baseline = None

    for lead in lead_times:
        window = degrad_df.filter(
            pl.col('days_before_failure').is_between(lead - 1, lead + 1)
        )
        if len(window) == 0 or col not in window.columns:
            continue
        median_val = window[col].median()
        onset_records.append({
            'smart_id': int(smart_id),
            'days_before_failure': lead,
            'median_value': median_val,
            'healthy_baseline': healthy_baseline,
            'ratio_to_baseline': (
                round(median_val / healthy_baseline, 3)
                if healthy_baseline and healthy_baseline != 0 else None
            ),
        })

onset_df = pl.DataFrame(onset_records)
save_table(onset_df, 'bb_degradation_onset')
print(onset_df.to_pandas().to_string(index=False))

# %% [markdown]
# *This table shows how SMART values at various lead times compare to the
# healthy-drive baseline. A ratio significantly different from 1.0 at 30+ days
# indicates early warning potential — the core justification for temporal
# sliding-window features in the Backblaze modeling approach.*

# %% [markdown]
# ### 6.3 Cross-Dataset Comparison Table
#
# Formal comparison of the two datasets' failure dynamics.

# %%
# Build the comparison table dynamically from EDA results.
# Google column: validated findings from notebook 03 (fixed).
# Backblaze column: populated from variables computed above.
bb_imbalance_ratio = f'{(grand_total_records - grand_total_failures) / grand_total_failures:.0f}:1'

comparison = pl.DataFrame([
    {
        'dimension': 'Domain',
        'google_traces': 'Cloud compute (Borg scheduler)',
        'backblaze': 'Physical hard drives',
    },
    {
        'dimension': 'Failure type',
        'google_traces': 'Software crash / misconfiguration',
        'backblaze': 'Hardware degradation (SMART attributes)',
    },
    {
        'dimension': 'Failure onset',
        'google_traces': 'Rapid (22s median running duration)',
        'backblaze': '(see degradation curves in 6.1 above)',
    },
    {
        'dimension': 'Dominant signal',
        'google_traces': 'Pre-scheduling history (72x resubmission signal)',
        'backblaze': '(see SMART AUC ranking in Section 3)',
    },
    {
        'dimension': 'Class imbalance',
        'google_traces': '3.4:1 (FAIL+LOST vs FINISH)',
        'backblaze': f'{bb_imbalance_ratio} (healthy vs failed observations)',
    },
    {
        'dimension': 'Prediction architecture',
        'google_traces': 'At-submission / at-scheduling time',
        'backblaze': '(determine after reviewing Sections 3 and 6)',
    },
    {
        'dimension': 'Feature priority',
        'google_traces': 'Tier 1: historical > Tier 2: runtime > Tier 3: utilization',
        'backblaze': '(see SMART AUC ranking in Section 3)',
    },
    {
        'dimension': 'Concept drift concern',
        'google_traces': 'Low (31-day trace, single cluster)',
        'backblaze': f'{year_max - year_min} years of data (see Section 5)',
    },
])

save_table(comparison, 'bb_cross_dataset_comparison')
print(comparison.to_pandas().to_string(index=False))

# %% [markdown]
# *Table 6.3 shows the structural differences between the two datasets. The
# Google column reflects validated findings from notebook 03. The Backblaze
# column references this notebook's results — review the degradation curves
# (6.1), onset detection (6.2), SMART ranking (Section 3), and drift analysis
# (Section 5) to complete the comparison narrative in notebook 07.*

# %% [markdown]
# ---
# # Section 7: Key Findings Summary
#
# > **Notebook:** `05_backblaze_eda.py`
# > **Date:** February 2026
# > **Purpose:** Synthesize all EDA outputs from Sections 1–6 into actionable findings for Chapter 3 methodology and Phase 3 pipeline decisions.

# %% [markdown]
# ### 7.1 Dataset Dimensions
#
# The Backblaze Hard Drive dataset spans **13 calendar years** (2013–2025) and comprises **681,749,417 daily drive observations** across **31,062 confirmed failure events**. The dataset is distributed across 27 Parquet files (annual files for 2013–2015, quarterly files from Q1 2016 onward), totaling approximately 40 unique SMART attributes in the earliest files and expanding to 93 SMART attributes by Q1 2025.
#
# **Key dimensional statistics:**
#
# | Metric | Value |
# |--------|-------|
# | Total daily observations | 681,749,417 |
# | Total failure events | 31,062 |
# | Grand daily failure rate | 0.0046% |
# | Class imbalance ratio | 21,947:1 (healthy : failed) |
# | Year range | 2013–2025 (13 years) |
# | Peak unique models in a single year | 87 (2024) |
# | Peak fleet size | 1,315,504 active drives (2025) |
#
# **Schema evolution:** The number of tracked SMART attributes grew substantially over time — from 40 IDs in 2013–2014 to 93 IDs by 2024–2025. Only **40 SMART IDs are universal** (present in every Parquet file across all years). An additional set of IDs appears only in later files as Backblaze expanded monitoring to newer drive models. This schema evolution has direct implications for feature engineering: any model trained on the full temporal span must be restricted to universal attributes or employ imputation strategies for file-specific attributes.
#
# **Fleet growth trajectory:** The fleet grew from approximately 29,000 active drives in 2013 to over 1.3 million in 2025 — a **~45× increase**. Total records per year scaled proportionally, from 5.1M observations in 2013 to 117.4M in 2025. This growth means that later years dominate any aggregate statistics and that temporal stratification is essential for unbiased drift analysis.
#
# **Implication for modeling:** The 21,947:1 class imbalance is **extreme** — roughly three orders of magnitude more severe than the Google Traces dataset (3.4:1). Standard classification algorithms will be overwhelmed by the majority class. This necessitates aggressive resampling (SMOTE, undersampling), cost-sensitive learning, or anomaly-detection framing. Evaluation must rely on metrics robust to imbalance: MCC, PR-AUC, and F1 rather than accuracy or standard ROC-AUC.

# %% [markdown]
# ### 7.2 Failure Characteristics
#
# ### 7.2.1 Yearly Failure Rate Trends
#
# The daily failure rate shows meaningful variation across years, declining from early highs and stabilizing in later years:
#
# | Year | Total Records | Failure Count | Daily Failure Rate (%) |
# |------|--------------|---------------|----------------------|
# | 2013 | 5,091,501 | 724 | 0.0142 |
# | 2014 | 12,582,414 | 2,205 | 0.0175 |
# | 2015 | 17,509,251 | 1,417 | 0.0081 |
# | 2016 | 24,471,617 | 1,428 | 0.0058 |
# | 2017 | 30,471,787 | 1,547 | 0.0051 |
# | 2018 | 36,600,253 | 1,364 | 0.0037 |
# | 2019 | 40,627,013 | 2,218 | 0.0055 |
# | 2020 | 52,286,398 | 1,495 | 0.0029 |
# | 2021 | 67,294,340 | 2,158 | 0.0032 |
# | 2022 | 80,357,762 | 3,158 | 0.0039 |
# | 2023 | 91,683,133 | 4,357 | 0.0048 |
# | 2024 | 105,379,761 | 4,664 | 0.0044 |
# | 2025 | 117,394,187 | 4,327 | 0.0037 |
#
# The daily failure rate peaked in 2014 (0.0175%) during the fleet's early expansion, declined to a minimum around 2018–2020 (0.0029–0.0037%), and modestly increased again in 2022–2023 (0.0039–0.0048%) as the fleet scaled past 1 million drives. The monthly failure rate time series (Figure 5.1a) shows pronounced volatility with spikes that suggest fleet-level events (e.g., large batch deployments of models with elevated failure propensity). This non-stationarity is direct evidence supporting the concept drift investigation in Section 7.5.
#
# ### 7.2.2 Failure Rate by Drive Model
#
# The top 20 drive models by observation count account for **615,290,182 observations (90.3% of the total dataset)**, confirming that the fleet is heavily concentrated. Failure rates vary substantially across models:
#
# - **Highest failure rate among top-20 models:** ST4000DM000 at 0.0072% daily (5,790 failures across 80.4M observations; 0.69% cumulative drive-level failure rate). This is the single largest-volume model in the dataset.
# - **Lowest failure rate among top-20 models:** HGST HMS5C4040BLE640 at 0.0011% daily (448 failures across 41.1M observations; 0.10% drive-level failure rate).
# - **Spread:** The daily failure rate across top-20 models ranges from 0.0011% to 0.0072% — a **6.5× difference** between the most and least reliable high-volume models.
#
# This inter-model variability suggests that drive model identity is itself a meaningful predictive feature. Additionally, models with both high volume and high failure rates (e.g., ST4000DM000, ST8000NM0055, ST12000NM0008, ST12000NM0007) are the most valuable for training because they provide both sufficient data and adequate positive-class samples.
#
# ### 7.2.3 Failure Rate by Capacity
#
# Failure rates show a general trend by drive capacity, though the relationship is not strictly monotonic. Among capacity buckets with substantial sample sizes (>1M observations):
#
# - **Smaller drives (≤3 TB):** Higher and more variable failure rates, ranging from 0.0056% (2 TB) to 0.0185% (0.5 TB). These are generally older-generation models that are aging out of the fleet.
# - **Mid-range drives (4–12 TB):** Moderate rates around 0.0047–0.0051%. These represent the fleet's core during 2016–2022.
# - **Large drives (14–22 TB):** Lower rates of 0.0017–0.0032%. These are newer-generation enterprise drives deployed in more recent years.
#
# The capacity-failure relationship is confounded with age and model generation — smaller-capacity drives tend to be older. Capacity should be included as a feature but interpreted cautiously.
#
# ### 7.2.4 Seasonal Patterns
#
# Monthly failure rates aggregated across all years show a mild seasonal pattern:
#
# - **Peak months:** July (0.0051%) and August (0.0051%) — summer months show the highest failure rates.
# - **Trough month:** December (0.0035%) — the lowest failure rate of any month.
# - **Magnitude:** The peak-to-trough spread is approximately 46% relative difference (0.0051% vs 0.0035%).
#
# This seasonal signal is modest but consistent, potentially reflecting temperature effects on mechanical components. However, the effect is confounded with deployment cycles (batch hardware refreshes, decommissioning schedules). Seasonal indicators (month or quarter) may be worth including as features for marginal predictive gain, but they are unlikely to be among the top-tier predictors.

# %% [markdown]
# ### 7.3 SMART Attribute Ranking
#
# ### 7.3.1 Availability Profiling
#
# Of the 40 universal SMART attributes (present in all Parquet files), availability varies widely:
#
# - **High availability (≥50%):** 17 attributes. These form the candidate feature set for modeling across the full temporal span.
# - **Mid availability (10–50%):** 6 attributes (IDs 183, 184, 187, 188, 189, 190, 191, 195, 196, 200, and others in this range). Notably, **SMART 187 (49.55%) and SMART 188 (49.44%)** fall just below the 50% threshold — these are frequently cited in the literature as top predictors but are not available for nearly half of all observations.
# - **Low availability (<10%):** Several attributes (IDs 11, 13, 15) with <1% non-null rates — effectively useless for modeling.
#
# The highest-availability attributes (all >98%) include: SMART 1 (Read Error Rate, 99.83%), SMART 9 (Power-On Hours, 99.88%), SMART 194 (Temperature, 99.88%), SMART 5 (Reallocated Sectors, 99.36%), SMART 197 (Current Pending Sectors, 98.01%), SMART 198 (Offline Uncorrectable, 98.40%), SMART 12 (Power Cycle Count, 98.95%), and SMART 192 (Power-Off Retract, 98.57%).
#
# ### 7.3.2 Discriminative Power (AUC Ranking)
#
# Single-feature AUC analysis (rank-based classifier) on the high-availability attributes yields the following ranking of the top 17 profiled attributes:
#
# | Rank | SMART ID | Description | AUC (magnitude) | Direction |
# |------|----------|-------------|-----------------|-----------|
# | 1 | 197 | Current Pending Sector Count | 0.7367 | Higher = more failure |
# | 2 | 5 | Reallocated Sectors Count | 0.7323 | Higher = more failure |
# | 3 | 198 | Offline Uncorrectable | 0.6815 | Higher = more failure |
# | 4 | 4 | Start/Stop Count | 0.6303 | Higher = more failure |
# | 5 | 12 | Power Cycle Count | 0.6201 | Higher = more failure |
# | 6 | 193 | Load Cycle Count | 0.6104 | Higher = more failure |
# | 7 | 240 | Head Flying Hours | 0.6077 | Higher = more failure |
# | 8 | 1 | Read Error Rate | 0.5897 | Higher = more failure |
# | 9 | 7 | Seek Error Rate | 0.5828 | Higher = more failure |
# | 10 | 9 | Power-On Hours | 0.5684 | Higher = more failure |
# | 11 | 242 | Total LBAs Read | 0.5673 | Higher = more failure |
# | 12 | 3 | Spin-Up Time | 0.5543 | **Lower** = more failure |
# | 13 | 194 | Temperature (Celsius) | 0.5266 | **Lower** = more failure |
# | 14 | 241 | Total LBAs Written | 0.5252 | Higher = more failure |
# | 15 | 199 | UDMA CRC Error Count | 0.5156 | Higher = more failure |
# | 16 | 192 | Power-Off Retract Count | 0.5142 | **Lower** = more failure |
# | 17 | 10 | Spin Retry Count | 0.5003 | Higher = more failure |
#
# **Key observations:**
#
# 1. **SMART 197 (Pending Sectors) and SMART 5 (Reallocated Sectors) are the top two discriminators**, with AUC values of 0.7367 and 0.7323 respectively. Both indicate active media degradation — sectors the drive has flagged as questionable or already remapped.
#
# 2. **SMART 198 (Offline Uncorrectable) ranks third** at 0.6815, completing a cluster of media-integrity indicators at the top of the ranking.
#
# 3. **SMART 4 (Start/Stop Count) and SMART 12 (Power Cycle Count) are surprisingly discriminative** (ranks 4–5, AUC 0.63 and 0.62). These are cumulative wear indicators rather than active defect counters, suggesting that mechanical wear history has predictive value independent of active defects.
#
# 4. **All top attributes predict failure with higher values**, except SMART 3 (Spin-Up Time, lower = failure) and SMART 194 (Temperature, lower = failure). The temperature finding is counterintuitive — failing drives may run cooler because they are already partially failed and handling less I/O. This warrants further investigation but the AUC is weak (0.5266).
#
# 5. **No single attribute exceeds AUC 0.75**, confirming that hard drive failure is a multi-signal phenomenon requiring ensemble feature engineering rather than single-attribute thresholding.
#
# ### 7.3.3 Literature Comparison: SMART 5, 187, 188, 197, 198
#
# The literature (Cheng et al., 2022; Zhang et al., 2023) consistently highlights five SMART attributes as most predictive: **5, 187, 188, 197, and 198.** Our data-driven analysis partially confirms this:
#
# | SMART ID | Literature Claim | Our AUC Rank | Our AUC | Availability | Status |
# |----------|-----------------|--------------|---------|-------------|--------|
# | 197 | Top predictor | **#1** | 0.7367 | 98.01% | ✅ Confirmed |
# | 5 | Top predictor | **#2** | 0.7323 | 99.36% | ✅ Confirmed |
# | 198 | Top predictor | **#3** | 0.6815 | 98.40% | ✅ Confirmed |
# | 187 | Top predictor | **Not ranked** | — | 49.55% | ⚠️ Excluded (below 50% availability threshold) |
# | 188 | Top predictor | **Not ranked** | — | 49.44% | ⚠️ Excluded (below 50% availability threshold) |
#
# **Critical finding:** SMART 187 (Reported Uncorrectable Errors) and SMART 188 (Command Timeout) — two of the five literature-standard attributes — are available for less than half of observations in this dataset. Studies that rely on all five attributes without documenting availability rates may produce results that do not generalize to heterogeneous, multi-year fleets. This is a meaningful methodological gap identified through our EDA.
#
# **Decision:** The modeling pipeline will use SMART 197, 5, and 198 as primary predictive features (confirmed by both data and literature). SMART 187 and 188 will be included as conditional features where available, with indicator encoding for missingness. The pipeline will also evaluate SMART 4, 12, and 193 as supplementary features based on their data-driven AUC ranking — these are not traditionally highlighted in the literature but show meaningful discriminative power (AUC > 0.60).

# %% [markdown]
# ### 7.4 Drive Model Analysis
#
# ### 7.4.1 Fleet Composition
#
# The fleet is dominated by a small number of high-volume models. The top 5 models by observation count are:
#
# | Model | Observations | Failures | Daily Fail Rate (%) | Drive Fail Rate (%) |
# |-------|-------------|----------|-------------------|-------------------|
# | ST4000DM000 | 80,403,688 | 5,790 | 0.0072 | 0.69 |
# | TOSHIBA MG07ACA14TA | 71,673,140 | 2,020 | 0.0028 | 0.25 |
# | ST8000NM0055 | 43,926,091 | 2,364 | 0.0054 | 0.48 |
# | ST12000NM0008 | 41,348,262 | 2,337 | 0.0057 | 0.50 |
# | HGST HMS5C4040BLE640 | 41,145,273 | 448 | 0.0011 | 0.10 |
#
# These five models together account for approximately 278.5M observations (40.9% of the dataset). The ST4000DM000 alone contributes 18.7% of all failure events.
#
# ### 7.4.2 Model-Specific SMART Importance (Stratification Analysis)
#
# The model-specific AUC heatmap reveals **substantial variation** in how SMART attributes perform across models:
#
# | Attribute | ST4000DM000 | TOSHIBA MG07 | ST8000NM0055 | ST12000NM0008 | HGST HMS5C4040 | Spread |
# |-----------|-------------|--------------|--------------|---------------|----------------|--------|
# | SMART 197 | 0.7729 | 0.7620 | 0.6364 | 0.6970 | 0.6274 | 0.1455 |
# | SMART 5 | 0.6444 | 0.6459 | **0.8717** | **0.8247** | 0.6071 | **0.2646** |
# | SMART 198 | 0.7744 | 0.5787 | 0.6364 | 0.6970 | 0.5000 | **0.2744** |
# | SMART 4 | 0.5865 | 0.6759 | 0.6138 | 0.6082 | 0.5125 | 0.1634 |
# | SMART 12 | 0.5889 | 0.6402 | 0.6072 | 0.5997 | 0.5163 | 0.1239 |
#
# **Key observations:**
#
# 1. **SMART 5 (Reallocated Sectors) shows the most dramatic model-specific variation.** For Seagate 8 TB and 12 TB enterprise drives (ST8000NM0055, ST12000NM0008), SMART 5 is an exceptionally strong predictor (AUC 0.87 and 0.82). For the HGST model, SMART 5 is only marginally discriminative (AUC 0.61). This is a **0.2646 spread** — the same attribute ranges from "strong predictor" to "near-random" depending on the drive model.
#
# 2. **SMART 198 also varies widely** — from 0.7744 (ST4000DM000) to 0.5000 (HGST, effectively random). The 0.2744 spread is the largest of any attribute.
#
# 3. **SMART 197 is the most stable predictor** across models, with a tighter spread of 0.1455 (range: 0.6274–0.7729). This makes it the most robust single feature for a global (non-stratified) model.
#
# 4. **The HGST HMS5C4040BLE640 consistently shows the weakest SMART discrimination** across all attributes. This model may have a fundamentally different failure mechanism that SMART attributes do not capture well, or its very low baseline failure rate (0.0011%) makes discrimination inherently harder.
#
# **Decision: Stratified modeling is warranted.** The AUC variation across models (up to 0.27 for a single attribute) is too large to ignore. The pipeline will evaluate both a global model and per-model (or per-manufacturer) models. At minimum, model identity should be included as a categorical feature in the global model. For the Seagate enterprise drives, SMART 5 should receive elevated feature importance weighting.

# %% [markdown]
# ### 7.5 Temporal Drift (RQ5 Evidence)
#
# ### 7.5.1 Failure Rate Non-Stationarity
#
# The monthly failure rate time series (Figure 5.1a, Table `bb_monthly_failure_rate.csv`) shows clear non-stationarity:
#
# - **Early period (2013–2014):** Higher, volatile failure rates (0.01–0.03% daily), reflecting a smaller, less mature fleet.
# - **Stabilization period (2015–2018):** Declining rates as fleet professionalized and higher-reliability enterprise models were deployed.
# - **Growth period (2019–2025):** Rates fluctuate between 0.002–0.006%, with periodic spikes likely corresponding to batch introductions of new (or problematic) model lines.
#
# The monthly rate ranges from a minimum of 0.0014% to a maximum of 0.0298% — a **21× range** — confirming that the underlying failure distribution is non-stationary. This is the primary quantitative justification for RQ5's online learning hypothesis.
#
# ### 7.5.2 SMART Attribute Distribution Shift
#
# Figure 5.2a tracks the yearly median of the top 5 discriminative SMART attributes. The drift analysis (Section 5.2) shows systematic covariate shift — SMART attribute distributions change over time even for the general population, not just for failing drives. This shift is driven by two interacting mechanisms:
#
# 1. **Fleet composition change:** New drive models with different baseline SMART profiles are introduced each year, while older models are retired. The number of unique models per year fluctuated between 40 (2013) and 87 (2024), with significant model turnover at each transition.
#
# 2. **Drive aging:** The cumulative nature of many SMART attributes (e.g., SMART 9 Power-On Hours, SMART 193 Load Cycle Count) means that fleet-level distributions shift rightward as the average fleet age increases.
#
# ### 7.5.3 Fleet Evolution
#
# | Year | Unique Models | Total Active Drives | Year-over-Year Growth |
# |------|--------------|--------------------|-----------------------|
# | 2013 | 40 | 29,072 | — |
# | 2014 | 81 | 47,793 | +64% |
# | 2015 | 78 | 62,898 | +32% |
# | 2016 | 77 | 285,011 | +353% |
# | 2017 | 68 | 366,413 | +29% |
# | 2018 | 61 | 424,919 | +16% |
# | 2019 | 55 | 469,049 | +10% |
# | 2020 | 59 | 612,491 | +31% |
# | 2021 | 70 | 810,410 | +32% |
# | 2022 | 75 | 930,907 | +15% |
# | 2023 | 83 | 1,041,814 | +12% |
# | 2024 | 87 | 1,197,036 | +15% |
# | 2025 | 80 | 1,315,504 | +10% |
#
# The massive 353% growth in 2016 and the sustained expansion through 2025 represent fleet-level distribution shifts. Each new model cohort brings different baseline SMART profiles, different failure mechanisms, and different capacity tiers. A static model trained on 2013–2018 data would encounter substantially different feature distributions when applied to 2022–2025 data.
#
# **Drift severity assessment:** The combination of non-stationary failure rates, covariate shift in SMART distributions, and fleet composition turnover constitutes **moderate-to-severe concept drift** — sufficient to justify the online/incremental learning approach proposed in RQ5. The 12-year temporal span (vs. Google's 31-day snapshot) makes Backblaze the primary dataset for drift evaluation.

# %% [markdown]
# ### 7.6 Cross-Dataset Comparison
#
# ### 7.6.1 Degradation Dynamics
#
# The degradation curve analysis (Section 6.1, Figure 6.1a) tracked the top 3 discriminative SMART attributes (197, 5, 198) over the 90 days before failure for a sample of 1,000 failed drives. The onset detection table (Section 6.2) compares SMART medians at various lead times to the healthy-drive baseline.
#
# **Critical observation on zero-inflation:** The degradation onset analysis shows that the **median values for SMART 197, 5, and 198 are 0.0 at all lead times** (90, 60, 30, 14, 7, and 1 day before failure), and the healthy baseline is also 0.0. This does not mean these attributes lack predictive power (their AUCs of 0.74, 0.73, and 0.68 demonstrate otherwise). Rather, it reveals that these SMART attributes are **heavily zero-inflated**: the majority of drives (including many that eventually fail) report zero for these attributes. Discrimination comes from the *tail* — the minority of drives that report non-zero values are disproportionately likely to be pre-failure.
#
# This zero-inflation pattern has important modeling implications:
# - Simple median-based onset detection is insufficient; the signal resides in the upper quantiles and non-zero frequency
# - Binary indicator features ("has non-zero SMART 197") may be as informative as the raw values
# - Rate-of-change features (how quickly an attribute transitions from zero to non-zero) may capture the degradation onset more effectively than static snapshots
#
# ### 7.6.2 Structural Comparison: Google vs. Backblaze
#
# | Dimension | Google Cluster Traces | Backblaze Hard Drives |
# |-----------|----------------------|----------------------|
# | **Domain** | Cloud compute (Borg scheduler) | Physical hard drives |
# | **Failure type** | Software crash / misconfiguration | Hardware degradation (SMART) |
# | **Failure onset** | Rapid (22s median running duration) | Gradual (zero-inflated, tail-driven) |
# | **Dominant signal** | Pre-scheduling history (72× resubmission) | SMART 197, 5, 198 (media integrity) |
# | **Class imbalance** | 3.4:1 (moderate) | 21,947:1 (extreme) |
# | **Prediction architecture** | At-submission / at-scheduling time | Sliding-window with temporal features |
# | **Feature priority** | Historical > Runtime > Utilization | Sector health > Wear indicators > Operational |
# | **Concept drift** | Low (31-day trace, single cluster) | High (12 years, fleet turnover) |
#
# **Architectural implications:**
#
# 1. **Different prediction timing:** Google's rapid-onset crashes require at-submission prediction. Backblaze's gradual degradation enables a **sliding-window approach** with rolling statistics and rate-of-change features computed over days or weeks of SMART history.
#
# 2. **Different imbalance strategies:** Google's moderate imbalance (3.4:1) is manageable with standard techniques. Backblaze's extreme imbalance (21,947:1) may require anomaly detection framing, severe undersampling of the majority class, or two-stage prediction (first detect "at-risk" drives, then predict time-to-failure).
#
# 3. **Different evaluation regimes:** Google's single-cluster, 31-day snapshot supports a standard temporal train/test split. Backblaze's 12-year span with fleet turnover requires **expanding-window or sliding-window evaluation** to assess model degradation over time.
#
# 4. **Complementary research value:** The two datasets represent opposing poles of failure prediction: rapid, software-driven crashes vs. slow, hardware-driven degradation. Demonstrating that an ensemble framework can adapt to both failure modes (via domain-specific feature engineering and architecture selection) strengthens the generalizability claim of the dissertation.

# %% [markdown]
# ### 7.7 Open Questions for Phase 3
#
# 1. **Non-zero frequency features:** Given the zero-inflation in SMART 197, 5, and 198, should binary indicators ("has_nonzero_smart_197") be engineered alongside raw values and rolling statistics? Are the upper quantiles (p90, p95, p99) more discriminative than medians or means for these attributes?
#
# 2. **Optimal temporal window for sliding features:** The degradation curves suggest change is detectable within 90 days, but the zero-inflation complicates interpretation. What window lengths (7, 14, 30, 60 days) maximize predictive power for rate-of-change features?
#
# 3. **Model stratification depth:** The heatmap shows AUC variation of up to 0.27 across models. Should stratification be at the model level (potentially dozens of sub-models), the manufacturer level (3–4 groups: Seagate, HGST/WDC, Toshiba, Hitachi), or handled via model-identity features in a global classifier?
#
# 4. **SMART 187 and 188 recovery:** These literature-standard attributes have ~49.5% availability. Can availability be improved by restricting analysis to post-2015 files (where schema expanded)? Does their discriminative power justify maintaining separate model branches for drives with and without these attributes?
#
# 5. **Time-to-failure modeling:** The EDA focused on binary failure prediction. For RQ3 (lead time analysis), should Backblaze support survival analysis (Cox PH or accelerated failure time models) using the days-from-first-observation-to-failure variable?
#
# 6. **Seasonal feature engineering:** The 46% relative seasonal variation is modest but consistent. Should month/quarter indicators be included, or is the signal too weak to justify the additional features?
#
# 7. **HGST anomaly:** The HGST HMS5C4040BLE640 shows consistently weak SMART discrimination. Is this a genuine failure-mechanism difference, or an artifact of the model's very low failure rate (0.0011% daily) making discrimination statistically harder? Should this model be excluded from training, or does its behavior represent an important edge case?

# %% [markdown]
# ### 7.8 Preliminary Decisions Log
#
# | # | Decision | Choice | Evidence | Literature Support |
# |---|----------|--------|----------|-------------------|
# | D1 | **Primary SMART features** | SMART 197, 5, 198 | AUC: 0.7367, 0.7323, 0.6815 (top 3) | Cheng et al. (2022); Zhang et al. (2023) confirm 5, 197, 198 |
# | D2 | **Secondary SMART features** | SMART 4, 12, 193, 240, 1, 7, 9 | AUC > 0.56; cumulative wear indicators | Data-driven; extends beyond literature standard |
# | D3 | **SMART 187/188 handling** | Conditional inclusion with indicator encoding | 49.5% availability; literature priority | Literature expects these; data shows availability gap |
# | D4 | **Imbalance strategy** | Cost-sensitive learning + severe undersampling; evaluate anomaly detection | 21,947:1 ratio | Li et al. (2021); standard SMOTE may be insufficient at this ratio |
# | D5 | **Model stratification** | Include model identity as feature; evaluate per-manufacturer sub-models | AUC spread up to 0.2744 across top-5 models | Zhang et al. (2023) use model-specific pipelines |
# | D6 | **Feature engineering approach** | Sliding-window temporal features: rolling mean, rolling std, rate-of-change, binary non-zero indicators | Gradual degradation pattern; zero-inflation | Supports sliding-window over point-in-time snapshot |
# | D7 | **Temporal evaluation strategy** | Expanding-window or sliding-window cross-validation over years | Non-stationary failure rates; fleet composition drift | Bergmeir & Benítez (2012); Campos et al. (2023) |
# | D8 | **Concept drift approach (RQ5)** | Online/incremental learning justified | 21× monthly rate range; schema evolution; 12-year fleet turnover | AlShafeey & Csaki (2024); direct support for RQ5 |
# | D9 | **Prediction framing** | Daily sliding-window prediction (multi-day lookahead) | Gradual degradation supports temporal features | Contrasts with Google at-submission architecture |
# | D10 | **Evaluation metrics** | MCC, PR-AUC, F1 (not accuracy or standard ROC-AUC) | Extreme class imbalance makes accuracy meaningless | Chicco & Jurman (2023)
