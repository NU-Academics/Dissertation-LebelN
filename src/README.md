# `src/` - Extracted Modules

Modules in `src/` are extracted from notebooks after the notebook has validated the logic against the EDA-confirmed statistics. This is the second-stage of the "let the data shape the code" principle that governs the repository: notebooks come first because they capture the analytical narrative; `src/` modules exist to make that narrative reusable, testable, and committee-auditable.

## Subpackages

```
src/
├── data/             Polars schemas, sentinel and priority constants, assertion helpers
├── preprocessing/    Per-event Polars transforms + BigQuery-backed lifecycle reconstruction
├── features/         Tier 1 / 2 / 3 feature engineering, working-set sampler (Phase 3)
├── models/           Ensemble training, conflict classifiers, online learners (Phase 4 onward)
└── evaluation/       Metrics, bootstrap confidence intervals, hypothesis tests (Phase 4 onward)
```

Each subpackage has its own README detailing the modules it carries and the decisions log entries that motivate them.

## Naming and extraction conventions

The seven-phase enhanced CRISP-DM lifecycle (Bokrantz et al., 2024) determines which subpackage a module belongs in:

| Subpackage | Phase | When code lands here |
|------------|-------|----------------------|
| `data/` | Phase 3 (Data Preparation) | Schemas, constants, validation helpers used everywhere downstream. |
| `preprocessing/` | Phase 3 (Data Preparation) | Cleaning, labeling, MNAR encoding, lifecycle reconstruction. Per-dataset modules. |
| `features/` | Phase 3 (Data Preparation) | Tier 1 / 2 / 3 features, working-set sampler, sliding windows. Per-dataset modules. |
| `models/` | Phase 4 (Modeling) and Phase 7 (Operation) | Ensemble factories, online learners, conflict classifiers. |
| `evaluation/` | Phase 5 (Evaluation) | Metric computations, bootstrap confidence intervals, hypothesis test wrappers, calibration. |

A module is allowed to enter `src/` once two conditions hold: (1) a notebook has executed the same logic end to end and verified the output against the EDA-confirmed numbers, and (2) the assertion suite in `src/data/validation.py` covers the regression-relevant invariants for that module.

## Imports

Modules import from `src` as a top-level package, for example `from src.preprocessing.google_traces import filter_sentinel_timestamps`. This works because the repo root sits on `sys.path` after the standard Colab clone-and-prepend pattern in `notebooks/00_setup_environment.py` and is the default when running `pytest` from the repo root.

## Pure functions, no I/O

Functions in `src/preprocessing/*.py` and `src/features/*.py` take Polars LazyFrames in and return LazyFrames out. They do not read files, query BigQuery, write to Drive, or mutate global state. I/O happens in notebooks and in the small set of helpers under `utils/`. The only exception is `src/preprocessing/lifecycle.py::reconstruct_instance_lifecycle`, which is deliberately BigQuery-backed because the SUBMIT / SCHEDULE / terminal join runs across the full 1.72B-row events table and would not fit in Colab memory; the function is documented and tested accordingly.

## Test coverage

`tests/test_smoke.py` verifies every subpackage imports cleanly. Per-module tests (`tests/test_preprocessing.py`, `tests/test_lifecycle.py`, future `test_features.py`, etc.) verify the algorithmic semantics against synthetic Polars LazyFrames hand-crafted to hit each branch.

See `tests/README.md` for the test inventory and how to run the suite.
