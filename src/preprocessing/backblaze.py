"""Per-row preprocessing transforms for the Backblaze Hard Drive dataset.

Pure Polars LazyFrame functions. No I/O. Each function takes a LazyFrame in
and returns a LazyFrame with one preprocessing transform applied, so the
calling notebook composes them in order and keeps all reads and writes outside
the module.

The transforms encode the Phase 2 and schema-census decisions captured in
``outputs/tables/eda_decisions.csv``:

- V14 / V15: primary and secondary SMART attributes.
- V16: SMART 187 / 188 conditional inclusion with an availability indicator.
- V18: drive-model identity and manufacturer as features.
- the era-census row: three SMART schema eras (notebook 07c), materialized as
  ``src.data.schemas.BACKBLAZE_ERAS``.

The dataset records one observation per drive per day. SSDs are excluded to
preserve a consistent 13-year HDD temporal axis for the drift analysis. The
schema is not stable across that span, so reconciliation guarantees a fixed
column set and availability indicators mark where a non-universal SMART column
was actually collected, rather than imputing it. Rolling and lag features built
downstream depend on a single deterministic sort by ``(serial_number, date)``,
established once in the calling notebook after row filtering.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.data.schemas import BACKBLAZE_ERAS

# Manufacturer prefixes follow the Backblaze model-naming conventions surfaced
# in notebook 05. The order matters: more specific prefixes are tested before
# shorter ones (WDC before WD). Drive-model identity and manufacturer are V18
# features.
_MANUFACTURER_PREFIXES: tuple[tuple[str, str], ...] = (
    ("ST", "Seagate"),
    ("TOSHIBA", "Toshiba"),
    ("TOSH", "Toshiba"),
    ("HGST", "HGST"),
    ("WDC", "WDC"),
    ("WD", "WDC"),
    ("Hitachi", "Hitachi"),
    ("HDS", "Hitachi"),
    ("HUH", "HGST"),
    ("HUS", "HGST"),
    ("Samsung", "Samsung"),
    ("SAMSUNG", "Samsung"),
)


def filter_hdds_only(
    lf: pl.LazyFrame,
    ssd_models: frozenset[str] | set[str] | None = None,
    model_column: str = "model",
) -> pl.LazyFrame:
    """Drop SSD rows so the modeling axis is HDD-only.

    SSDs are excluded to keep the 13-year temporal axis consistent for the
    drift analysis (RQ5). The Backblaze daily schema carries no drive-type
    flag, so SSDs are identified by model name. The set of SSD model strings is
    sourced from notebook 05 and passed in by the caller rather than hard-coded
    here, so the exclusion stays data-driven and auditable.

    Args:
        lf: Backblaze daily LazyFrame with a model column.
        ssd_models: model strings to exclude. ``None`` or an empty set is a
            no-op (the caller asserts the resulting row count downstream).
        model_column: name of the drive-model column.

    Returns:
        LazyFrame with SSD rows removed.
    """
    if not ssd_models:
        return lf
    return lf.filter(~pl.col(model_column).is_in(list(ssd_models)))


def assign_era(
    lf: pl.LazyFrame,
    era_constants: list[tuple[str, str, str, tuple[int, ...]]] | None = None,
    date_column: str = "date",
    era_column: str = "era",
) -> pl.LazyFrame:
    """Annotate each row with the SMART schema era of its observation date.

    Era boundaries come from the schema-evolution census (notebook 07c),
    materialized as ``BACKBLAZE_ERAS``. Each era is an inclusive
    ``[start_date, end_date]`` day range with an era name. Dates outside every
    era range receive ``"unknown"`` so the caller can detect and triage them.
    Every subsequent transform is era-aware.

    Args:
        lf: Backblaze daily LazyFrame with a date column.
        era_constants: list of ``(start_date, end_date, era_name,
            available_smart_ids)`` tuples. Defaults to ``BACKBLAZE_ERAS``.
        date_column: name of the observation-date column (a ``pl.Date``).
        era_column: name of the era label column to add.

    Returns:
        LazyFrame with the era label column appended.
    """
    if era_constants is None:
        era_constants = BACKBLAZE_ERAS

    expr = pl.when(pl.lit(False)).then(pl.lit(None, dtype=pl.Utf8))
    for start, end, name, _ids in era_constants:
        lo = date.fromisoformat(start)
        hi = date.fromisoformat(end)
        expr = expr.when(
            pl.col(date_column).is_between(lo, hi, closed="both")
        ).then(pl.lit(name))
    expr = expr.otherwise(pl.lit("unknown")).alias(era_column)
    return lf.with_columns(expr)


def reconcile_smart_schema(
    lf: pl.LazyFrame,
    smart_ids: tuple[int, ...] | list[int],
    keep_normalized: bool = True,
) -> pl.LazyFrame:
    """Guarantee a fixed SMART column set across the schema-evolving years.

    Different annual or quarterly files carry different SMART columns. To let
    downstream code rely on a stable schema, this transform ensures a
    ``smart_{id}_raw`` column (and, when ``keep_normalized`` is True, a
    ``smart_{id}_normalized`` column) exists for every requested SMART ID. Any
    column absent in the input is added as an all-null ``Float64`` column rather
    than imputed; the availability indicators (see
    :func:`encode_smart_availability_indicators`) then capture where a column
    was genuinely collected.

    Args:
        lf: Backblaze daily LazyFrame.
        smart_ids: SMART IDs that must be present after reconciliation (for
            example the union of IDs across all eras).
        keep_normalized: also guarantee the ``_normalized`` sibling columns.

    Returns:
        LazyFrame with every requested SMART column present.
    """
    existing = set(lf.collect_schema().names())
    additions: list[pl.Expr] = []
    for sid in smart_ids:
        raw = f"smart_{sid}_raw"
        if raw not in existing:
            additions.append(pl.lit(None, dtype=pl.Float64).alias(raw))
        if keep_normalized:
            norm = f"smart_{sid}_normalized"
            if norm not in existing:
                additions.append(pl.lit(None, dtype=pl.Float64).alias(norm))
    if not additions:
        return lf
    return lf.with_columns(additions)


def encode_smart_availability_indicators(
    lf: pl.LazyFrame,
    smart_ids: tuple[int, ...] | list[int],
) -> pl.LazyFrame:
    """Add a ``has_smart_{id}`` indicator for each requested SMART ID (V16).

    For non-universal SMART attributes (notably 187 and 188, which the census
    confines to the middle era), whether the attribute was collected is itself
    signal. The indicator is 1 where ``smart_{id}_raw`` is non-null and 0
    otherwise. This is the encoding that lets the model use era-gated columns
    without imputing the periods that did not collect them.

    Args:
        lf: Backblaze daily LazyFrame with the requested ``smart_{id}_raw``
            columns present (run :func:`reconcile_smart_schema` first).
        smart_ids: SMART IDs to encode an availability indicator for.

    Returns:
        LazyFrame with one ``has_smart_{id}`` column per requested ID.
    """
    indicators = [
        pl.col(f"smart_{sid}_raw").is_not_null().cast(pl.Int8).alias(f"has_smart_{sid}")
        for sid in smart_ids
    ]
    return lf.with_columns(indicators)


def canonicalize_drive_model(
    lf: pl.LazyFrame,
    aliases: dict[str, str] | None = None,
    model_column: str = "model",
) -> pl.LazyFrame:
    """Add canonical drive-model and manufacturer columns (V18).

    Some manufacturers rename a model mid-series. The optional ``aliases`` map
    (validated in notebook 05) folds known aliases onto a canonical model
    identity before the manufacturer is derived; the default is an identity map
    (no folding). The manufacturer is read from the canonical model-name prefix
    using the Backblaze naming conventions; unrecognized prefixes map to
    ``"Other"``.

    Args:
        lf: Backblaze daily LazyFrame with a model column.
        aliases: optional mapping from raw model string to canonical model
            string. Defaults to no folding.
        model_column: name of the raw drive-model column.

    Returns:
        LazyFrame with ``model_canonical`` and ``manufacturer`` columns added.
    """
    canonical = pl.col(model_column)
    if aliases:
        canonical = canonical.replace(aliases)
    out = lf.with_columns(canonical.alias("model_canonical"))

    manufacturer = pl.when(pl.lit(False)).then(pl.lit(None, dtype=pl.Utf8))
    for prefix, name in _MANUFACTURER_PREFIXES:
        manufacturer = manufacturer.when(
            pl.col("model_canonical").str.starts_with(prefix)
        ).then(pl.lit(name))
    manufacturer = manufacturer.otherwise(pl.lit("Other")).alias("manufacturer")
    return out.with_columns(manufacturer)


def mark_censoring(
    lf: pl.LazyFrame,
    serial_column: str = "serial_number",
    date_column: str = "date",
    failure_column: str = "failure",
) -> pl.LazyFrame:
    """Mark each drive's final observation as a failure or a censoring event.

    For the survival framing of the lead-time analysis (RQ3), every drive's
    last observation is classified. A drive whose terminal observation carries
    ``failure = 1`` is an observed failure; a drive that simply stops appearing
    in the dataset is right-censored. Adds three columns:

    - ``is_last_obs``: 1 on the drive's maximum-date row, else 0.
    - ``failure_observed``: 1 on a terminal row with ``failure = 1``, else 0.
    - ``censored``: 1 on a terminal row with ``failure = 0`` (the drive left the
      fleet without a recorded failure), else 0.

    Args:
        lf: Backblaze daily LazyFrame.
        serial_column: per-drive identifier column.
        date_column: observation-date column.
        failure_column: binary failure column (1 on the day of failure removal).

    Returns:
        LazyFrame with the three censoring columns appended.
    """
    is_last = (
        pl.col(date_column) == pl.col(date_column).max().over(serial_column)
    )
    return lf.with_columns(
        is_last.cast(pl.Int8).alias("is_last_obs")
    ).with_columns(
        (pl.col("is_last_obs") == 1)
        .and_(pl.col(failure_column) == 1)
        .cast(pl.Int8)
        .alias("failure_observed"),
        (pl.col("is_last_obs") == 1)
        .and_(pl.col(failure_column) != 1)
        .cast(pl.Int8)
        .alias("censored"),
    )
