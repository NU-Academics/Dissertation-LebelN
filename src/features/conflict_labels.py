"""RQ2 conflict labeling for Google Cluster Traces (stub).

TODO: implement the RQ2 conflict-type labelers. This module is a placeholder so
that imports across ``src.features`` resolve for now; the substantive logic is
implemented in the RQ2 conflict labeling notebook.

Planned scope (per the Chapter 3 RQ2 Study Procedures and ``P09`` in
``outputs/tables/eda_decisions.csv``): derive supervised conflict-resolution
targets from scheduling interactions - resource contention, priority
inversion, and scheduling-constraint violations - together with the
resolution-outcome label used to evaluate the >80% automated resolution
success target.

When implemented, :func:`label_conflicts` will be a pure
``LazyFrame -> LazyFrame`` transform consistent with the other modules in this
package (appends label columns, performs no I/O).
"""

from __future__ import annotations

import polars as pl

# Planned conflict taxonomy for RQ2 (P09 in outputs/tables/eda_decisions.csv).
CONFLICT_TYPES: tuple[str, ...] = (
    "resource_contention",
    "priority_inversion",
    "scheduling_violation",
)


def label_conflicts(events_lf: pl.LazyFrame, conflict_type: str) -> pl.LazyFrame:
    """Append RQ2 conflict-type and resolution-outcome labels (NOT IMPLEMENTED).

    Args:
        events_lf: instance/collection event LazyFrame to label.
        conflict_type: one of :data:`CONFLICT_TYPES`.

    Returns:
        A LazyFrame with the conflict-type and resolution-outcome label columns
        appended.

    Raises:
        NotImplementedError: always, until the RQ2 implementation lands.
    """
    raise NotImplementedError(
        "label_conflicts is an RQ2 deliverable; see CONFLICT_TYPES and "
        "P09 in outputs/tables/eda_decisions.csv for the planned taxonomy."
    )
