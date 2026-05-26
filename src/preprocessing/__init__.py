"""Preprocessing modules.

Phase 3 (Data Preparation) deliverables. Each submodule provides pure functions
that take Polars LazyFrames as input and return LazyFrames with the
preprocessing transforms applied. No I/O happens inside these functions; that
stays in the calling notebook.

Planned submodules (created during Weeks 2 and 7):
- google_traces: sentinel filtering, failure labeling, MNAR hardware-counter
  encoding, instance lifecycle reconstruction, machine-attribute joins.
- backblaze: HDD-only filtering, SMART schema reconciliation, availability
  indicators for non-universal SMART IDs, drive-model canonicalization.
"""
