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
# # 04 — Backblaze Hard Drive Data: Ingestion Pipeline
#
# **Purpose:** Download Backblaze quarterly/annual zip files from GCS, extract
# daily CSVs, convert to Parquet using Polars, and upload back to GCS. This is
# a pure data-pipeline notebook — no analysis. The output Parquet files feed into
# notebook 05 (Backblaze deep EDA).
#
# **Data layout on GCS:**
# - Raw zips: `gs://{PROJECT_ID}-dissertation-data/backblaze_raw/`
# - Annual zips for 2013–2015 (e.g., `data_2013.zip`)
# - Quarterly zips for 2016–2025 (e.g., `data_Q1_2016.zip`)
# - Each zip contains daily CSVs: `YYYY-MM-DD.csv`
#
# **Processing strategy:**
# - One zip at a time to stay within Colab's ~12 GB RAM
# - Each zip → single consolidated Parquet file uploaded to GCS
# - Checkpoint after each zip so interrupted sessions can resume
# - Schema evolution handled: SMART columns vary across years
#
# **Outputs:**
# - Parquet files: `gs://{GCS_BUCKET}/backblaze_parquet/{zip_stem}.parquet`
# - Ingestion log: `gs://{GCS_BUCKET}/backblaze_parquet/ingestion_log.csv`
# - Summary report printed at the end
#
# **Prerequisites:**
# - Backblaze zips uploaded to the GCS bucket listed above
# - Colab Secrets: `GCP_PROJECT_ID`

# %% [markdown]
# ---
# ## 1. Colab Session Setup

# %%
# !pip install -q polars google-cloud-storage

# %%
from google.colab import userdata

PROJECT_ID = userdata.get('GCP_PROJECT_ID')
print(f"GCP Project: {PROJECT_ID}")

# %%
from pathlib import Path

LOCAL_TMP = Path('/content/backblaze_tmp')
LOCAL_TMP.mkdir(parents=True, exist_ok=True)

print(f"Local temp:  {LOCAL_TMP}")

# %%
from google.colab import auth
auth.authenticate_user()

# %%
import polars as pl
import time
import zipfile
import shutil
import gc

# %% [markdown]
# ---
# ## 2. GCS Configuration & Zip File Discovery
#
# List all Backblaze zip files in the GCS bucket. We expect annual zips for
# 2013–2015 and quarterly zips for 2016–2025.

# %%
from google.cloud import storage

GCS_BUCKET = f'{PROJECT_ID}-dissertation-data'
GCS_RAW_PREFIX = 'backblaze_raw/'
GCS_PARQUET_PREFIX = 'backblaze_parquet/'

gcs_client = storage.Client(project=PROJECT_ID)
bucket = gcs_client.bucket(GCS_BUCKET)

print(f"GCS Bucket:  gs://{GCS_BUCKET}/")
print(f"  Raw zips:  {GCS_RAW_PREFIX}")
print(f"  Parquet:   {GCS_PARQUET_PREFIX}\n")

blobs = list(bucket.list_blobs(prefix=GCS_RAW_PREFIX))
zip_blobs = [b for b in blobs if b.name.endswith('.zip')]
zip_blobs.sort(key=lambda b: b.name)

print(f"Found {len(zip_blobs)} zip files in gs://{GCS_BUCKET}/{GCS_RAW_PREFIX}\n")
for b in zip_blobs:
    size_mb = b.size / (1024 * 1024)
    print(f"  {b.name.split('/')[-1]:30s}  {size_mb:>8.1f} MB")

# %% [markdown]
# ---
# ## 3. Core Data Types
#
# Define the 5 fixed columns that every Backblaze CSV contains. SMART columns
# vary by year/quarter and are detected dynamically during ingestion.

# %%
FIXED_COLUMNS = ['date', 'serial_number', 'model', 'capacity_bytes', 'failure']

FIXED_DTYPES = {
    'date': pl.Utf8,
    'serial_number': pl.Utf8,
    'model': pl.Utf8,
    'capacity_bytes': pl.Int64,
    'failure': pl.Int8,
}

# %% [markdown]
# ---
# ## 4. Checkpoint Management
#
# Track which zips have already been processed. On session restart, we skip
# completed zips and pick up where we left off.

# %%
GCS_LOG_BLOB_NAME = f'{GCS_PARQUET_PREFIX}ingestion_log.csv'
_LOCAL_LOG = LOCAL_TMP / 'ingestion_log.csv'


def load_completed_zips() -> dict[str, dict]:
    """Load the ingestion log from GCS. Returns {zip_name: metadata}."""
    blob = bucket.blob(GCS_LOG_BLOB_NAME)
    if not blob.exists():
        return {}
    blob.download_to_filename(str(_LOCAL_LOG))
    log_df = pl.read_csv(_LOCAL_LOG)
    return {
        row['zip_name']: {k: row[k] for k in row if k != 'zip_name'}
        for row in log_df.to_dicts()
    }


def save_ingestion_entry(zip_name: str, metadata: dict):
    """Append one entry to the ingestion log on GCS."""
    entry = {'zip_name': zip_name, **metadata}
    entry_df = pl.DataFrame([entry])

    blob = bucket.blob(GCS_LOG_BLOB_NAME)
    if blob.exists():
        blob.download_to_filename(str(_LOCAL_LOG))
        existing = pl.read_csv(_LOCAL_LOG)
        combined = pl.concat([existing, entry_df], how='diagonal_relaxed')
    else:
        combined = entry_df

    combined.write_csv(_LOCAL_LOG)
    blob.upload_from_filename(str(_LOCAL_LOG))


# %%
completed = load_completed_zips()
print(f"Already processed: {len(completed)} zip(s)")
if completed:
    for name in sorted(completed):
        rows = completed[name].get('total_rows', '?')
        print(f"  {name}: {int(rows):,} rows")

# %% [markdown]
# ---
# ## 5. Ingestion Pipeline
#
# For each zip file:
# 1. **Download** from GCS to Colab local storage (fast NVMe)
# 2. **Extract** daily CSVs from the zip (filtering out `__MACOSX` artifacts)
# 3. **Convert** each CSV to a small Parquet shard on local disk, sorted by
#    `serial_number` within that day (avoids OOM — only one day in memory)
# 4. **Merge** shards in filename (date) order via PyArrow incremental writer
#    → single sorted Parquet file, then **upload** to GCS
# 5. **Log** metadata (rows, columns, file size, duration)
# 6. **Clean up** local temp files and shards to free disk space

# %%
def infer_smart_dtypes(header_cols: list[str]) -> dict[str, pl.DataType]:
    """Build a dtype override dict for SMART columns.

    Normalized SMART values are Int16 (0–255 range).
    Raw SMART values are Int64 (can be very large counters).
    """
    dtypes = dict(FIXED_DTYPES)
    for col in header_cols:
        if col in FIXED_DTYPES:
            continue
        if '_normalized' in col:
            dtypes[col] = pl.Int16
        elif '_raw' in col:
            dtypes[col] = pl.Int64
    return dtypes


def _file_looks_sane(csv_path: Path, expected_ncols: int) -> bool:
    """Fast byte-level sanity check — catches corrupted line endings.

    Reads only the first 64 KB of the file.  If the first line contains
    more commas than 2× expected_ncols, the file is almost certainly
    corrupted (e.g., \\r-only line endings cause Polars to treat the
    entire file as a single row of thousands of columns).
    """
    try:
        chunk = csv_path.read_bytes()[:65_536]
        # Find the first proper newline (\\n).  If there is none in the
        # first 64 KB of a daily CSV (~120 KB), something is very wrong.
        nl_pos = chunk.find(b'\n')
        if nl_pos == -1:
            # No newline at all — could be \\r-only or one giant line
            header_bytes = chunk
        else:
            header_bytes = chunk[:nl_pos]
        comma_count = header_bytes.count(b',')
        if comma_count > expected_ncols * 2:
            print(f"\n    SKIP {csv_path.name}: header has ~{comma_count+1} "
                  f"fields (expected ~{expected_ncols}) — likely corrupted")
            return False
        # Also flag suspiciously large files with no newlines
        if nl_pos == -1 and csv_path.stat().st_size > 500_000:
            print(f"\n    SKIP {csv_path.name}: no newline in first 64 KB "
                  f"and file is {csv_path.stat().st_size/1e6:.1f} MB "
                  f"— likely corrupted line endings")
            return False
    except Exception:
        pass  # if the byte-check itself fails, fall through to normal read
    return True


def read_csv_safe(
    csv_path: Path,
    dtypes: dict,
    expected_ncols: int = 0,
) -> pl.DataFrame | None:
    """Read a single daily CSV, handling type mismatches gracefully.

    Strategy:
      1. Byte-level header check — catches corrupted files (wrong line
         endings that produce thousands of columns) *without* loading
         them into a DataFrame at all.
      2. Strict read with schema_overrides — fast, type-safe.
      3. Fallback: same overrides but ignore_errors=True so corrupt
         rows get nulls instead of crashing.
    """
    # --- Pre-flight: byte-level sanity check (never OOMs) ---
    if expected_ncols > 0 and not _file_looks_sane(csv_path, expected_ncols):
        return None

    try:
        return pl.read_csv(
            csv_path,
            schema_overrides=dtypes,
            null_values=['', 'NA', 'na'],
            ignore_errors=False,
            low_memory=True,
        )
    except Exception:
        try:
            return pl.read_csv(
                csv_path,
                schema_overrides=dtypes,
                null_values=['', 'NA', 'na'],
                ignore_errors=True,  # corrupt values → null
                low_memory=True,
            )
        except Exception as e:
            print(f"    SKIP {csv_path.name}: {e}")
            return None


def process_one_zip(blob, local_tmp: Path) -> dict:
    """Download, extract, convert one zip → Parquet. Returns metadata dict."""
    zip_name = blob.name.split('/')[-1]
    zip_stem = zip_name.replace('.zip', '')
    parquet_path = local_tmp / f'{zip_stem}.parquet'

    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"Processing: {zip_name} ({blob.size / 1024**2:.1f} MB)")
    print(f"{'='*70}")

    # --- Download ---
    zip_local = local_tmp / zip_name
    print("  Downloading from GCS...", end=' ', flush=True)
    blob.download_to_filename(str(zip_local))
    t_download = time.time() - t0
    print(f"done ({t_download:.1f}s)")

    # --- Extract ---
    extract_dir = local_tmp / zip_stem
    extract_dir.mkdir(exist_ok=True)
    print("  Extracting...", end=' ', flush=True)
    with zipfile.ZipFile(zip_local, 'r') as zf:
        zf.extractall(str(extract_dir))
    t_extract = time.time() - t0
    print(f"done ({t_extract - t_download:.1f}s)")

    # Remove zip to free disk space
    zip_local.unlink()

    # --- Find CSVs (may be nested in a subdirectory) ---
    # Filter out __MACOSX resource fork artifacts that some zips contain
    csv_files = sorted(
        p for p in extract_dir.rglob('*.csv')
        if '__MACOSX' not in p.parts
    )
    print(f"  Found {len(csv_files)} CSV files")

    if not csv_files:
        print(f"  WARNING: No CSV files found in {zip_name}")
        shutil.rmtree(extract_dir, ignore_errors=True)
        return {
            'total_rows': 0, 'total_cols': 0, 'csv_count': 0,
            'parquet_size_mb': 0, 'duration_s': time.time() - t0,
            'min_date': '', 'max_date': '', 'smart_columns': 0,
        }

    # --- Detect schema from first CSV header ---
    first_df_peek = pl.read_csv(csv_files[0], n_rows=0)
    all_columns = first_df_peek.columns
    dtypes = infer_smart_dtypes(all_columns)
    smart_cols = [c for c in all_columns if c.startswith('smart_')]
    print(f"  Schema: {len(all_columns)} columns ({len(smart_cols)} SMART)")

    # --- Convert CSVs → per-day Parquet shards (avoids OOM on large zips) ---
    # Each daily CSV is a single date, so we sort within each shard by
    # serial_number and then concatenate shards in filename order (which IS
    # date order since filenames are YYYY-MM-DD). This avoids a global sort
    # that would require materializing the full dataset in memory.
    shard_dir = local_tmp / f'{zip_stem}_shards'
    shard_dir.mkdir(exist_ok=True)

    expected_ncols = len(all_columns)

    print("  Reading CSVs → Parquet shards...", end=' ', flush=True)
    skipped = 0
    shard_count = 0
    total_rows = 0
    all_dates: list[str] = []  # track date range from CSV filenames

    MAX_CSV_BYTES = 500 * 1024 * 1024  # 500 MB — no single day should exceed this

    for csv_path in csv_files:
        # Guard: skip abnormally large files that would OOM during read
        try:
            fsize = csv_path.stat().st_size
            if fsize > MAX_CSV_BYTES:
                print(f"\n    SKIP {csv_path.name}: {fsize/1e6:.0f} MB "
                      f"exceeds {MAX_CSV_BYTES/1e6:.0f} MB limit — likely corrupted")
                csv_path.unlink(missing_ok=True)
                skipped += 1
                continue
        except OSError:
            pass

        df = read_csv_safe(csv_path, dtypes, expected_ncols)
        if df is None or len(df) == 0:
            skipped += 1
            csv_path.unlink(missing_ok=True)  # free disk even on skip
            continue
        # Sanity check: if the column count is wildly wrong, the file
        # likely has corrupted line endings (entire file parsed as one
        # row with thousands of columns).  Drop it before it eats RAM.
        if len(df.columns) > expected_ncols * 2 or len(df) < 2:
            print(f"\n    SKIP {csv_path.name}: "
                  f"bad shape ({len(df)} rows × {len(df.columns)} cols)")
            del df
            skipped += 1
            csv_path.unlink(missing_ok=True)
            gc.collect()
            continue
        # Parse date while we have the small DataFrame in memory
        if df['date'].dtype == pl.Utf8:
            df = df.with_columns(pl.col('date').str.to_date('%Y-%m-%d'))
        # Sort within this day by serial_number — the inter-shard order
        # (by date) is guaranteed by processing shards in filename order
        df = df.sort('serial_number')
        shard_path = shard_dir / f'{csv_path.stem}.parquet'
        df.write_parquet(shard_path, compression='snappy')
        total_rows += len(df)
        all_dates.append(csv_path.stem)  # filename is YYYY-MM-DD
        del df
        shard_count += 1
        # Delete source CSV immediately — no need to keep it alongside
        # the shard, and this prevents disk from holding both full sets
        csv_path.unlink(missing_ok=True)
        # Force gc after EVERY shard.  By 2018+ each CSV expands to
        # 200-300 MB as a DataFrame; Polars/Arrow buffers are not always
        # freed by `del` alone.  The ~50 ms cost per call is negligible.
        gc.collect()

    gc.collect()
    t_read = time.time() - t0
    print(f"done ({t_read - t_extract:.1f}s, {shard_count} shards)")

    if skipped:
        print(f"    Skipped {skipped}/{len(csv_files)} non-CSV artifacts")

    if shard_count == 0:
        print(f"  WARNING: No valid CSVs processed for {zip_name}")
        shutil.rmtree(extract_dir, ignore_errors=True)
        shutil.rmtree(shard_dir, ignore_errors=True)
        return {
            'total_rows': 0, 'total_cols': 0, 'csv_count': 0,
            'parquet_size_mb': 0, 'duration_s': time.time() - t0,
            'min_date': '', 'max_date': '', 'smart_columns': 0,
        }

    min_date = min(all_dates)
    max_date = max(all_dates)

    # Remove extracted CSVs — shards replace them
    shutil.rmtree(extract_dir, ignore_errors=True)

    # --- Merge shards into final Parquet one day at a time ---
    # Read each daily shard (already sorted by serial_number), append to
    # a Parquet file on local disk via pyarrow's incremental writer. This
    # keeps peak memory at one day's data (~60K rows × 95 cols ≈ 50-80 MB).
    # The merged file is then uploaded to GCS.
    print("  Merging shards → local Parquet...", end=' ', flush=True)

    import pyarrow.parquet as pq

    shard_paths = sorted(shard_dir.glob('*.parquet'))  # date order
    writer = None
    target_schema = None
    for i, shard_path in enumerate(shard_paths):
        tbl = pq.read_table(str(shard_path))
        if writer is None:
            target_schema = tbl.schema
            writer = pq.ParquetWriter(
                str(parquet_path),
                schema=target_schema,
                compression='zstd',
                compression_level=3,
            )
        elif tbl.schema != target_schema:
            # Rare edge case: a CSV within the same zip has a different
            # column set.  Select only the columns the writer expects.
            target_names = set(target_schema.names)
            tbl = tbl.select([c for c in tbl.column_names if c in target_names])
        writer.write_table(tbl)
        del tbl
        # Free shard from disk once written to reduce disk pressure
        shard_path.unlink()
        if i % 10 == 9:
            gc.collect()

    if writer is not None:
        writer.close()

    t_write = time.time() - t0

    # Read back the schema to get total_cols (from the written file)
    final_schema = pl.read_parquet_schema(parquet_path)
    total_cols = len(final_schema)
    parquet_size_mb = parquet_path.stat().st_size / (1024 * 1024)
    print(f"done ({t_write - t_read:.1f}s)")

    # --- Clean up local temp shards ---
    shutil.rmtree(shard_dir, ignore_errors=True)

    # --- Upload merged Parquet to GCS ---
    print("  Uploading to GCS...", end=' ', flush=True)
    output_blob = bucket.blob(f'{GCS_PARQUET_PREFIX}{zip_stem}.parquet')
    output_blob.upload_from_filename(str(parquet_path))
    t_upload = time.time() - t0
    print(f"done ({t_upload - t_write:.1f}s)")

    # Free local disk — the Parquet is safely on GCS now
    parquet_path.unlink(missing_ok=True)
    gc.collect()

    duration = time.time() - t0
    print(f"  {zip_name}: {total_rows:>12,} rows | {total_cols} cols | "
          f"{parquet_size_mb:.1f} MB | {min_date} to {max_date} | {duration:.0f}s")

    return {
        'total_rows': total_rows,
        'total_cols': total_cols,
        'csv_count': shard_count,
        'parquet_size_mb': round(parquet_size_mb, 2),
        'duration_s': round(duration, 1),
        'min_date': min_date,
        'max_date': max_date,
        'smart_columns': len(smart_cols),
    }

# %% [markdown]
# ---
# ## 6. Run Ingestion
#
# Process each zip file sequentially. Already-completed zips (present in the
# ingestion log) are skipped automatically.

# %%
pipeline_start = time.time()

for blob in zip_blobs:
    zip_name = blob.name.split('/')[-1]

    if zip_name in completed:
        print(f"SKIP (already done): {zip_name}")
        continue

    metadata = process_one_zip(blob, LOCAL_TMP)
    save_ingestion_entry(zip_name, metadata)
    completed[zip_name] = metadata

pipeline_duration = time.time() - pipeline_start
print(f"\n{'='*70}")
print(f"Pipeline complete in {pipeline_duration/60:.1f} minutes")
print(f"{'='*70}")

# %% [markdown]
# ---
# ## 7. Ingestion Summary
#
# Report total rows, total size, years covered, and per-file statistics.

# %%
# Re-download the log from GCS (may have been updated across retries)
bucket.blob(GCS_LOG_BLOB_NAME).download_to_filename(str(_LOCAL_LOG))
log_df = pl.read_csv(_LOCAL_LOG)
print(log_df.to_pandas().to_string(index=False))

# %%
total_rows = log_df['total_rows'].sum()
total_parquet_mb = log_df['parquet_size_mb'].sum()
total_files = len(log_df)
total_csvs = log_df['csv_count'].sum()

# Extract year range from date columns
all_min = log_df.filter(pl.col('min_date') != '')['min_date'].min()
all_max = log_df.filter(pl.col('max_date') != '')['max_date'].max()

# SMART column evolution
smart_range = log_df.select([
    pl.col('smart_columns').min().alias('min_smart'),
    pl.col('smart_columns').max().alias('max_smart'),
]).row(0)

print(f"\n{'='*70}")
print("BACKBLAZE INGESTION SUMMARY")
print(f"{'='*70}")
print(f"  Zip files processed:     {total_files:>10,}")
print(f"  Daily CSVs read:         {total_csvs:>10,}")
print(f"  Total rows:              {total_rows:>10,}")
print(f"  Total Parquet size:      {total_parquet_mb:>10,.1f} MB "
      f"({total_parquet_mb/1024:.2f} GB)")
print(f"  Date range:              {all_min} to {all_max}")
print(f"  SMART columns:           {smart_range[0]} to {smart_range[1]} "
      f"(schema evolution)")
print(f"{'='*70}")

# %%
# Verify all Parquet files are present on GCS
print("Verifying Parquet files on GCS...\n")
parquet_blobs = sorted(
    [b for b in bucket.list_blobs(prefix=GCS_PARQUET_PREFIX)
     if b.name.endswith('.parquet')],
    key=lambda b: b.name,
)
for b in parquet_blobs:
    name = b.name.split('/')[-1]
    size_mb = b.size / (1024 * 1024)
    print(f"  {name:35s}  {size_mb:>8.1f} MB")

print(f"\n{len(parquet_blobs)} Parquet files on GCS.")

# %% [markdown]
# ---
# ## 8. Schema Evolution Overview
#
# Show which SMART columns are present in each zip's Parquet file. This is
# essential context for notebook 05 (Backblaze EDA) — some SMART attributes
# only appear in newer data, and some models report different subsets.

# %%
# Collect column sets per file by reading Parquet metadata from GCS
import pyarrow.parquet as pq
from pyarrow.fs import GcsFileSystem

gcs_fs = GcsFileSystem()

schema_records = []
all_smart_sets = []
for b in parquet_blobs:
    name = b.name.split('/')[-1]
    stem = name.replace('.parquet', '')
    pa_schema = pq.read_schema(f'{GCS_BUCKET}/{b.name}', filesystem=gcs_fs)
    col_names = pa_schema.names
    smart_cols = sorted([c for c in col_names if c.startswith('smart_')])
    smart_ids = sorted(set(c.split('_')[1] for c in smart_cols), key=int)
    schema_records.append({
        'file': stem,
        'total_cols': len(col_names),
        'smart_cols': len(smart_cols),
        'smart_ids': ', '.join(smart_ids),
    })
    all_smart_sets.append(set(smart_ids))

schema_df = pl.DataFrame(schema_records)
print(schema_df.to_pandas().to_string(index=False))

# %%
# Identify SMART IDs present across ALL files vs. those that appear/disappear
if all_smart_sets:
    universal = set.intersection(*all_smart_sets)
    ever_present = set.union(*all_smart_sets)
    variable = ever_present - universal

    print(f"SMART IDs present in ALL files ({len(universal)}):")
    print(f"  {', '.join(sorted(universal, key=int))}")
    print(f"\nSMART IDs present in SOME files only ({len(variable)}):")
    print(f"  {', '.join(sorted(variable, key=int))}")

# %% [markdown]
# ---
# ## 9. Clean Up
#
# Remove any leftover temporary files from local Colab storage.

# %%
if LOCAL_TMP.exists():
    shutil.rmtree(LOCAL_TMP, ignore_errors=True)
    print(f"Cleaned up {LOCAL_TMP}")

print(f"\nParquet files saved to: gs://{GCS_BUCKET}/{GCS_PARQUET_PREFIX}")
print("Ready for notebook 05 (Backblaze EDA).")
