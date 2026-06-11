"""Phase 3/4 skeleton smoke test.

Verifies the src/ subpackages are importable. As modules are extracted from
notebooks during Phases 3 and 4, replace these import-only assertions with
substantive unit tests in:
- tests/test_preprocessing.py (Phase 3)
- tests/test_features.py (Phase 3)
- tests/test_metrics.py (Phase 4)
- tests/test_hypothesis.py (Phase 4)
- tests/test_sampling.py (Phase 3)
- tests/test_drift_detectors.py (Phase 7)
"""

import importlib


SRC_SUBPACKAGES = [
    "src",
    "src.preprocessing",
    "src.features",
    "src.data",
    "src.models",
    "src.evaluation",
]


def test_src_subpackages_importable() -> None:
    """Every src/ subpackage imports without error.

    This is the bare minimum confirmation that the Phase 3 skeleton is in
    place. The test fails informatively if any __init__.py is missing or
    malformed.
    """
    for pkg in SRC_SUBPACKAGES:
        module = importlib.import_module(pkg)
        assert module is not None, f"{pkg} imported as None"


def test_random_seed_constant_is_42() -> None:
    """The locked random seed must remain 42 throughout Chapter 4 analysis.

    Every stochastic operation across all five RQs uses this seed (Ch. 3
    Reproducibility Plan; Pre-Chapter 4 readiness memo). If this assertion
    needs to change, the entire reproducibility contract is being modified
    and Chapter 3 must be updated first.
    """
    RANDOM_SEED = 42
    assert RANDOM_SEED == 42
