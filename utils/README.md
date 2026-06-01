# `utils/` — Shared Session and I/O Helpers

Minimal shared utilities. Three modules, all small enough to read in one sitting.

## `colab_setup.py`

Drive mount and output-directory creation for Google Colab sessions.

- `DRIVE_PATH = Path('/content/drive/MyDrive/Dissertation_Colab')` is the canonical Drive root for the project. Subdirectories `data/`, `checkpoints/`, and `outputs/` are exported as `DATA_DIR`, `CHECKPOINT_DIR`, and `OUTPUT_DIR`.
- `setup_drive()` mounts the Drive and creates the three working directories. Called by every Colab notebook in its setup section.

## `bq_client.py`

Authenticated BigQuery client factory backed by Colab Secrets.

- `PROJECT_ID` is loaded from the `GCP_PROJECT_ID` Colab Secret with a hardcoded placeholder fallback for local linting.
- `get_client(project_id=PROJECT_ID)` returns a `google.cloud.bigquery.Client` ready to query the cached `dissertation_lebel.*_full` tables.
- `table_ref(table_name)` returns the fully qualified backtick-quoted table reference for use inside SQL strings. Notebook 08's `fqn()` helper wraps this.

The dataset name `dissertation_lebel` is hardcoded as `DATASET`. There is one dataset per project.

## `checkpointing.py`

Pickle-based checkpoint helpers backed by `utils.colab_setup.CHECKPOINT_DIR`. Used to persist intermediate state across Colab session restarts (Colab sessions time out at 12 hours).

- `save_checkpoint(obj, name, epoch=None)` writes `obj` to `CHECKPOINT_DIR/{name}[_epoch{epoch}].pkl`.
- `load_checkpoint(name, epoch=None)` returns the pickled object or `None` if the checkpoint is missing.

The `epoch` argument is reserved for the Phase 4 training loops; Phase 2 and 3 callers do not use it.

## When to extend

Add a helper here when at least two notebooks share the same setup or I/O concern. Keep helpers single-purpose. I/O specifically intended to be testable belongs in `src/`, not `utils/`; the rule of thumb is that anything called from a `pytest` test should live under `src/` so the test does not have to clone Colab to import it.
