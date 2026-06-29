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
# # 07c. Backblaze Schema-Evolution Era Census
#
# **Purpose.** The Backblaze SMART schema is not stable across the 2013-2025
# window. Notebook 05 established the headline facts (40 universal SMART IDs out
# of 93 unique IDs; SMART 187 and 188 at roughly 49.5% availability overall;
# top predictors 197 / 5 / 198), but a single overall availability number hides
# *when* an attribute was collected. This notebook builds the per-quarter
# availability catalog needed before any Backblaze preprocessing: without an
# era catalog, the decision to drop, indicator-encode, or condition a column is
# arbitrary, and rolling-window features silently cross schema discontinuities.
#
# **What it produces.**
# 1. A long-format availability census: one row per `(year_quarter, smart_id)`
#    with the non-null fraction and the number of drives observed that quarter.
#    Saved to `outputs/tables/backblaze_schema_era_census.csv`.
# 2. Detected era boundaries, by clustering the per-SMART-ID availability time
#    series and locating the quarters where the available-attribute set shifts.
#    The working hypothesis (from notebook 05) is three eras: a sparse early
#    period, a standard middle period carrying the 40-universal set plus SMART
#    187 / 188, and an extended recent period. This notebook validates or
#    refines that hypothesis from the data.
# 3. A schema-evolution heatmap (`smart_id` rows by `year_quarter` columns) with
#    era boundaries marked. Saved to `outputs/figures/backblaze_schema_evolution.png`.
# 4. A `BACKBLAZE_ERAS` constants fragment, written to Drive for transfer into
#    `src/data/schemas.py`. The era constants drive every downstream Backblaze
#    preprocessing decision (column drop, availability indicator, conditional
#    inclusion).
#
# **Decisions operationalized.** This notebook supplies the empirical basis for
# the era-aware handling recorded against `eda_decisions.csv` rows V16 (SMART
# 187 / 188 conditional inclusion) and V19 (sliding-window feature approach),
# and resolves the open schema-evolution census item carried from notebook 05.
#
# **Scale discipline.** The Backblaze daily data is large (about 682M
# drive-day rows). Each Parquet file is scanned once with a lazy aggregation
# that returns only per-quarter summary rows, so the row-level data never lands
# in memory. Backblaze publishes one zip per year (2013-2014) or per quarter
# (2015 onward), so every calendar quarter lives in exactly one Parquet file;
# per-file per-quarter aggregates therefore compose exactly across files.

# %% [markdown]
# ## 0. Colab session setup

# %%
# !pip install -q polars scikit-learn pandas pyarrow google-cloud-storage matplotlib seaborn

# %%
import os
import sys
from pathlib import Path

from google.colab import userdata

GITHUB_PAT = userdata.get('GITHUB_PAT')
PROJECT_ID = userdata.get('GCP_PROJECT_ID')
REPO_OWNER = 'NU-Academics'
REPO_NAME = 'Dissertation-LebelN'
REPO_DIR = f'/content/{REPO_NAME}'
REPO_URL = f'https://{GITHUB_PAT}@github.com/{REPO_OWNER}/{REPO_NAME}.git'

if os.path.exists(REPO_DIR):
    # !cd {REPO_DIR} && git pull --quiet
    print(f"Repo already present at {REPO_DIR}; pulled latest.")
else:
    # !git clone --quiet {REPO_URL} {REPO_DIR}
    print(f"Cloned repo to {REPO_DIR}.")

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# Colab caches imported modules in sys.modules, so a later `git pull` has no
# effect until the runtime restarts. Drop any previously imported repo modules
# here so the freshly pulled source is what gets imported in the cells below.
for _m in [m for m in list(sys.modules)
           if m == "src" or m.startswith("src.") or m == "utils" or m.startswith("utils.")]:
    del sys.modules[_m]

# %%
from google.colab import auth
auth.authenticate_user()

# %%
import re
import gc

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 150

# %%
from google.colab import drive
drive.mount('/content/drive')

DRIVE_PATH = Path('/content/drive/MyDrive/Dissertation_Colab')
TABLES_DIR = DRIVE_PATH / 'outputs' / 'tables'
FIGURES_DIR = DRIVE_PATH / 'outputs' / 'figures'
FRAGMENT_DIR = DRIVE_PATH / 'outputs' / 'fragments'

# Local NVMe cache for the Parquet files (downloaded from GCS by notebook 04).
BACKBLAZE_DIR = Path('/content/backblaze_parquet')

for dir_path in [TABLES_DIR, FIGURES_DIR, FRAGMENT_DIR, BACKBLAZE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


def save_table(df: pl.DataFrame, name: str) -> None:
    """Write a Polars DataFrame as CSV to the Drive tables directory."""
    path = TABLES_DIR / f'{name}.csv'
    df.write_csv(path)
    print(f"Saved table: {path}  ({df.height:,} rows)")


def save_figure(fig, name: str) -> None:
    """Write a Matplotlib figure as PNG to the Drive figures directory."""
    path = FIGURES_DIR / f'{name}.png'
    fig.savefig(path, bbox_inches='tight')
    print(f"Saved figure: {path}")


# %% [markdown]
# ### Download Parquet files from GCS
#
# The Parquet files were written by notebook 04. They are pulled to local NVMe
# for fast `scan_parquet` access, skipping any already-cached file.

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
        print(f"  {name:35s}  {b.size / 1024**2:>8.1f} MB  (cached)")
    else:
        print(f"  {name:35s}  {b.size / 1024**2:>8.1f} MB  downloading...",
              end='', flush=True)
        b.download_to_filename(str(local_path))
        print("  done")

parquet_files = sorted(BACKBLAZE_DIR.glob('*.parquet'))
print(f"\n{len(parquet_files)} Parquet files available at {BACKBLAZE_DIR}")

# %% [markdown]
# ## 1. Per-quarter SMART availability census
#
# For every Parquet file, scan once and compute, per calendar quarter: the row
# count (drive-days), the number of distinct drives observed, and the non-null
# count of each `smart_{id}_raw` column present in that file. Raw columns are
# the modeling columns (notebook 05 Section 3.1), so availability is measured on
# them; the `_normalized` sibling tracks the same presence pattern.
#
# Results accumulate into dictionaries keyed by `(year_quarter, smart_id)`.
# Because each quarter resides in exactly one file, the accumulation is a direct
# composition with no cross-file double counting.

# %%
RAW_COL_RE = re.compile(r'^smart_(\d+)_raw$')


def year_quarter_expr() -> pl.Expr:
    """Map the `date` column to a sortable `YYYYQn` quarter label."""
    return (
        pl.col('date').dt.year().cast(pl.Utf8)
        + pl.lit('Q')
        + pl.col('date').dt.quarter().cast(pl.Utf8)
    )


# Per-quarter row and drive counts, and per-(quarter, smart_id) non-null counts.
rows_per_q: dict[str, int] = {}
drives_per_q: dict[str, int] = {}
nonnull_per_q_smart: dict[tuple[str, int], int] = {}
all_smart_ids: set[int] = set()

for pf in parquet_files:
    schema = pl.read_parquet_schema(pf)
    raw_cols = [c for c in schema if RAW_COL_RE.match(c)]
    smart_ids_here = {int(RAW_COL_RE.match(c).group(1)) for c in raw_cols}
    all_smart_ids |= smart_ids_here

    agg_exprs = [
        pl.len().alias('n_rows'),
        pl.col('serial_number').n_unique().alias('n_drives'),
    ]
    agg_exprs += [pl.col(c).is_not_null().sum().alias(c) for c in raw_cols]

    out = (
        pl.scan_parquet(pf)
        .with_columns(year_quarter_expr().alias('year_quarter'))
        .group_by('year_quarter')
        .agg(agg_exprs)
        .collect()
    )

    for row in out.iter_rows(named=True):
        yq = row['year_quarter']
        rows_per_q[yq] = rows_per_q.get(yq, 0) + int(row['n_rows'])
        drives_per_q[yq] = drives_per_q.get(yq, 0) + int(row['n_drives'])
        for c in raw_cols:
            sid = int(RAW_COL_RE.match(c).group(1))
            nonnull_per_q_smart[(yq, sid)] = (
                nonnull_per_q_smart.get((yq, sid), 0) + int(row[c])
            )

    print(f"  {pf.name:35s}  quarters: {out.height}")
    gc.collect()

quarters = sorted(rows_per_q)
smart_ids = sorted(all_smart_ids)
print(f"\nQuarters observed: {len(quarters)}  ({quarters[0]} .. {quarters[-1]})")
print(f"Distinct SMART IDs seen (raw columns): {len(smart_ids)}")

# %%
# Build the long-format census. A (quarter, smart_id) pair with no recorded
# non-null count had the column absent that quarter, so availability is 0.
census_records = []
for yq in quarters:
    n_rows = rows_per_q[yq]
    n_drives = drives_per_q[yq]
    for sid in smart_ids:
        nn = nonnull_per_q_smart.get((yq, sid), 0)
        census_records.append({
            'year_quarter': yq,
            'smart_id': sid,
            'availability_pct': round(100.0 * nn / n_rows, 4) if n_rows else 0.0,
            'n_drives_observed': n_drives,
        })

census = pl.DataFrame(census_records).sort(['year_quarter', 'smart_id'])

# Coverage diagnostics, printed first so they are visible even if a guard trips.
# Note on availability: it is the non-null fraction at the drive-day grain. Most
# SMART attributes are null for most drives, so per-quarter non-null counts run
# well below the count of columns merely present. The "40 universal SMART IDs"
# from notebook 05 is a column-presence claim, not a non-null claim; expect far
# fewer attributes to clear a 50% or 99% non-null bar in any given quarter.
total_rows = sum(rows_per_q.values())
print(f"Files scanned:        {len(parquet_files)}")
print(f"Quarters covered:     {len(quarters)}  ({quarters[0]} .. {quarters[-1]})")
print(f"Distinct SMART IDs:   {len(smart_ids)} (raw columns seen across files)")
print(f"Total drive-day rows: {total_rows:,}")

diag = (
    census.group_by('year_quarter')
    .agg(
        pl.col('n_drives_observed').first().alias('n_drives'),
        (pl.col('availability_pct') >= 50.0).sum().alias('n_attrs_ge50'),
        (pl.col('availability_pct') >= 99.0).sum().alias('n_attrs_ge99'),
    )
    .sort('year_quarter')
)
print("\nPer-quarter drive count and attributes clearing 50% / 99% availability:")
print(diag.to_pandas().to_string(index=False))

# Fail-loud guard against a stale module or a truncated/empty Parquet read. The
# robust truncation checks are on raw volume and coverage, not on an attribute-
# availability threshold (which reflects per-drive sparsity, not completeness).
# An empty or partial read collapses these counts; the full dataset is roughly
# 682M drive-day rows across about 50 quarters with 93 distinct SMART IDs.
expected_cols = {'year_quarter', 'smart_id', 'availability_pct', 'n_drives_observed'}
assert expected_cols.issubset(set(census.columns)), \
    f"census missing columns: {expected_cols - set(census.columns)}"
assert census.height == len(quarters) * len(smart_ids), \
    "census row count does not equal quarters x smart_ids"
assert total_rows > 100_000_000, \
    f"only {total_rows:,} drive-day rows scanned; suspect a truncated read"
assert len(smart_ids) >= 40, \
    f"only {len(smart_ids)} SMART raw columns seen; suspect a partial read"
assert len(quarters) >= 30, \
    f"only {len(quarters)} quarters covered; suspect a partial read"
assert drives_per_q[quarters[-1]] > 0, f"latest quarter {quarters[-1]} has zero drives"
print("\nGuard passed: row volume and schema coverage consistent with the full dataset.")

save_table(census, 'backblaze_schema_era_census')

# %% [markdown]
# ## 2. Era-boundary detection
#
# Pivot the census to a quarters-by-SMART-ID availability matrix. Two
# complementary views drive the boundaries:
#
# - **Composition shift.** Binarize availability at a presence threshold to get
#   the set of attributes collected each quarter, then measure the Jaccard
#   distance between consecutive quarters. Large jumps mark schema transitions.
# - **Clustering.** Agglomerative clustering on the per-quarter availability
#   vectors groups similar quarters; contiguous runs of a cluster become eras.
#
# The two views are reconciled into a small set of contiguous eras, then
# compared with the three-era hypothesis from notebook 05.

# %%
PRESENCE_THRESHOLD = 50.0   # availability_pct at/above which an attribute counts as collected
JACCARD_THRESHOLD = 0.10    # consecutive-quarter set distance flagged as a transition
N_CLUSTERS = 3              # hypothesis from notebook 05; refine if the data disagrees

# Quarters-by-smart_id availability matrix (rows ordered by calendar quarter).
avail_wide = (
    census.pivot(values='availability_pct', index='year_quarter', on='smart_id')
    .sort('year_quarter')
)
matrix = avail_wide.drop('year_quarter').to_numpy()
matrix = np.nan_to_num(matrix, nan=0.0)
q_index = avail_wide['year_quarter'].to_list()
sid_cols = [int(c) for c in avail_wide.columns if c != 'year_quarter']

# Present-attribute set per quarter.
present_sets = [
    frozenset(sid for sid, val in zip(sid_cols, row) if val >= PRESENCE_THRESHOLD)
    for row in matrix
]


def jaccard_distance(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


consecutive_dist = [0.0] + [
    jaccard_distance(present_sets[i - 1], present_sets[i])
    for i in range(1, len(present_sets))
]
transition_quarters = [
    q_index[i] for i in range(len(q_index)) if consecutive_dist[i] >= JACCARD_THRESHOLD
]

print("Consecutive-quarter composition shifts (Jaccard distance):")
for q, d, s in zip(q_index, consecutive_dist, present_sets):
    flag = '  <-- transition' if d >= JACCARD_THRESHOLD else ''
    print(f"  {q:8s}  d={d:0.3f}  |present|={len(s):3d}{flag}")
print(f"\nFlagged transition quarters: {transition_quarters}")

# %%
# Clustering view: group quarters by availability vector, then read off the
# dominant (contiguous) era label sequence.
cluster_labels = AgglomerativeClustering(n_clusters=N_CLUSTERS).fit_predict(matrix)

print("Cluster label per quarter:")
for q, lab in zip(q_index, cluster_labels):
    print(f"  {q:8s}  cluster={lab}")

# Collapse the label sequence into contiguous runs (era segments). A single
# isolated off-label quarter inside a longer run is absorbed into the run.
segments: list[list[int]] = []
for i, lab in enumerate(cluster_labels):
    if segments and cluster_labels[segments[-1][-1]] == lab:
        segments[-1].append(i)
    elif (segments and len(segments[-1]) >= 2 and i + 1 < len(cluster_labels)
          and cluster_labels[i + 1] == cluster_labels[segments[-1][-1]]):
        segments[-1].append(i)  # absorb a single-quarter blip
    else:
        segments.append([i])

print(f"\nContiguous era segments detected: {len(segments)}")

# %% [markdown]
# ### Era definitions
#
# For each contiguous segment, the era spans the first day of its first quarter
# to the last day of its last quarter. The era's `available_smart_ids` are the
# attributes present (at or above the presence threshold) in a majority of the
# era's quarters, which is robust to a single noisy quarter.

# %%
def quarter_start(yq: str) -> str:
    year = int(yq[:4])
    q = int(yq[-1])
    month = (q - 1) * 3 + 1
    return f"{year}-{month:02d}-01"


def quarter_end(yq: str) -> str:
    year = int(yq[:4])
    q = int(yq[-1])
    end_month = q * 3
    last_day = {3: 31, 6: 30, 9: 30, 12: 31}[end_month]
    return f"{year}-{end_month:02d}-{last_day}"


era_rows = []
for seg in segments:
    seg_quarters = [q_index[i] for i in seg]
    # Attributes present in a majority of the segment's quarters.
    counts: dict[int, int] = {}
    for i in seg:
        for sid in present_sets[i]:
            counts[sid] = counts.get(sid, 0) + 1
    majority = len(seg) / 2.0
    avail_ids = sorted(sid for sid, c in counts.items() if c >= majority)
    era_rows.append({
        'start_quarter': seg_quarters[0],
        'end_quarter': seg_quarters[-1],
        'start_date': quarter_start(seg_quarters[0]),
        'end_date': quarter_end(seg_quarters[-1]),
        'n_quarters': len(seg),
        'n_available_smart_ids': len(avail_ids),
        'available_smart_ids': avail_ids,
    })

print("Detected eras:")
for i, er in enumerate(era_rows, 1):
    print(f"  Era {i}: {er['start_date']} .. {er['end_date']}  "
          f"({er['n_quarters']} quarters, {er['n_available_smart_ids']} attributes)")
    print(f"         available_smart_ids = {er['available_smart_ids']}")

# Hypothesis check: are SMART 187 and 188 confined to a middle era as expected?
print("\nSMART 187 / 188 presence by era (notebook 05 hypothesis: middle era only):")
for i, er in enumerate(era_rows, 1):
    has187 = 187 in er['available_smart_ids']
    has188 = 188 in er['available_smart_ids']
    print(f"  Era {i} ({er['start_date']}..{er['end_date']}): 187={has187}, 188={has188}")

# %% [markdown]
# ## 3. Schema-evolution heatmap
#
# Availability of every SMART ID (rows) across calendar quarters (columns), with
# detected era boundaries marked. Reading left to right shows when attributes
# entered and left the schema.

# %%
heat = matrix.T  # smart_id rows, quarter columns
fig, ax = plt.subplots(figsize=(max(12, len(q_index) * 0.28), max(8, len(sid_cols) * 0.16)))
im = ax.imshow(heat, aspect='auto', cmap='viridis', vmin=0, vmax=100)

ax.set_xticks(range(len(q_index)))
ax.set_xticklabels(q_index, rotation=90, fontsize=7)
ax.set_yticks(range(len(sid_cols)))
ax.set_yticklabels(sid_cols, fontsize=6)
ax.set_xlabel('Calendar quarter')
ax.set_ylabel('SMART ID')
ax.set_title('Figure 07c.1: Backblaze SMART schema evolution (availability %, era boundaries marked)')

# Mark era boundaries with vertical lines at each segment start (after the first).
boundary_positions = [seg[0] for seg in segments[1:]]
for pos in boundary_positions:
    ax.axvline(pos - 0.5, color='red', linewidth=1.8, linestyle='--')

cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
cbar.set_label('availability %')
fig.tight_layout()
save_figure(fig, 'backblaze_schema_evolution')
plt.show()

# %% [markdown]
# ## 4. BACKBLAZE_ERAS constants fragment
#
# Emit the era definitions in the exact form expected by `src/data/schemas.py`:
# a list of `(start_date, end_date, era_name, available_smart_ids)` tuples. The
# fragment is written to Drive for transfer into the module; it is not the
# committed source itself. Era names are descriptive and adjustable after a
# human review of the detected boundaries.

# %%
def era_name(idx: int, er: dict) -> str:
    start_year = er['start_date'][:4]
    end_year = er['end_date'][:4]
    return f"era{idx}_{start_year}_{end_year}"


lines = ["BACKBLAZE_ERAS: list[tuple[str, str, str, tuple[int, ...]]] = ["]
for i, er in enumerate(era_rows, 1):
    ids_tuple = ", ".join(str(s) for s in er['available_smart_ids'])
    start = er['start_date']
    end = er['end_date']
    name = era_name(i, er)
    lines.append(f'    ("{start}", "{end}", "{name}", ({ids_tuple})),')
lines.append("]")
fragment = "\n".join(lines)

print(fragment)

fragment_path = FRAGMENT_DIR / 'backblaze_eras_fragment.py'
fragment_path.write_text(fragment + "\n")
print(f"\nWrote constants fragment: {fragment_path}")

# Also persist the era table as CSV for the decisions-log evidence trail.
era_table = pl.DataFrame([
    {
        'start_quarter': er['start_quarter'],
        'end_quarter': er['end_quarter'],
        'start_date': er['start_date'],
        'end_date': er['end_date'],
        'n_quarters': er['n_quarters'],
        'n_available_smart_ids': er['n_available_smart_ids'],
        'available_smart_ids': str(er['available_smart_ids']),
    }
    for er in era_rows
])
save_table(era_table, 'backblaze_eras')

# %% [markdown]
# ## 5. Summary and handoff
#
# Report back, for transfer into `src/data/schemas.py` and the decisions log:
#
# - the number of eras detected and their start/end dates;
# - whether the three-era hypothesis (sparse early, standard middle with SMART
#   187 / 188, extended recent) held or was refined;
# - the per-era available-attribute counts and the SMART 187 / 188 placement;
# - the printed `BACKBLAZE_ERAS` fragment.

# %%
print("ERA CENSUS SUMMARY")
print("=" * 60)
print(f"Quarters covered:        {q_index[0]} .. {q_index[-1]} ({len(q_index)} quarters)")
print(f"Distinct SMART IDs:      {len(sid_cols)}")
print(f"Eras detected:           {len(era_rows)}")
print(f"Presence threshold:      {PRESENCE_THRESHOLD}%")
print(f"Jaccard threshold:       {JACCARD_THRESHOLD}")
for i, er in enumerate(era_rows, 1):
    print(f"  Era {i}: {er['start_date']} .. {er['end_date']}  "
          f"{er['n_available_smart_ids']} attributes")
print("=" * 60)
print("Artifacts written:")
print(f"  - {TABLES_DIR / 'backblaze_schema_era_census.csv'}")
print(f"  - {TABLES_DIR / 'backblaze_eras.csv'}")
print(f"  - {FIGURES_DIR / 'backblaze_schema_evolution.png'}")
print(f"  - {FRAGMENT_DIR / 'backblaze_eras_fragment.py'}")
