# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 00 — Environment Setup & Integration Checks
#
# Run this notebook at the start of every Colab session.
# It installs dependencies, mounts Drive, sets up paths, and verifies connectivity.

# %% [markdown]
# ## Install Dependencies

# %%
# !pip install -q polars google-cloud-bigquery db-dtypes

# %% [markdown]
# ## Mount Google Drive

# %%
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown]
# ## Clone / Update Repository

# %%
import os

REPO_URL = 'https://github.com/YOUR-USERNAME/Dissertation-LebelN.git'
REPO_DIR = '/content/Dissertation-LebelN'

if os.path.exists(REPO_DIR):
    # !cd {REPO_DIR} && git pull
    pass
else:
    # !git clone {REPO_URL} {REPO_DIR}
    pass

import sys
sys.path.insert(0, REPO_DIR)

# %% [markdown]
# ## Path Constants

# %%
from pathlib import Path

DRIVE_PATH = Path('/content/drive/MyDrive/Dissertation_Colab')
DATA_DIR = DRIVE_PATH / 'data'
CHECKPOINT_DIR = DRIVE_PATH / 'checkpoints'
OUTPUT_DIR = DRIVE_PATH / 'outputs'

for dir_path in [DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

print(f"Data directory:       {DATA_DIR}")
print(f"Checkpoint directory: {CHECKPOINT_DIR}")
print(f"Output directory:     {OUTPUT_DIR}")

# %% [markdown]
# ## Authenticate & Create BigQuery Client

# %%
from google.colab import auth
auth.authenticate_user()

from google.cloud import bigquery

PROJECT_ID = 'YOUR-PROJECT-ID-HERE'
bq_client = bigquery.Client(project=PROJECT_ID)

# %% [markdown]
# ## Integration Checks

# %%
def test_bigquery_connection():
    """Verify BigQuery connection works."""
    result = bq_client.query("SELECT 1 as test").to_dataframe()
    assert result['test'][0] == 1
    print("BigQuery connection OK")


def test_cached_tables_exist():
    """Verify cached tables are accessible."""
    tables = [
        'instance_events_full',
        'machine_events_full',
        'instance_usage_full',
        'collection_events_full',
        'machine_attributes_full',
    ]
    for table in tables:
        query = f"SELECT COUNT(*) as n FROM `{PROJECT_ID}.dissertation_lebel.{table}` LIMIT 1"
        try:
            result = bq_client.query(query).to_dataframe()
            print(f"  {table}: accessible ({result['n'][0]:,} rows)")
        except Exception as e:
            print(f"  {table}: not yet cached — {e}")


def test_drive_access():
    """Verify Drive is mounted and writable."""
    test_file = DRIVE_PATH / 'test_write.txt'
    test_file.write_text('test')
    assert test_file.exists()
    test_file.unlink()
    print("Drive access OK")


# %%
print("Running integration checks...\n")
test_bigquery_connection()
test_drive_access()
print()
test_cached_tables_exist()
print("\nSetup complete!")
