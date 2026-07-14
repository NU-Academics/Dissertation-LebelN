"""Concept-drift detectors for the online-learning research question.

Four detectors behind one interface, matching the four drift subtypes the study
commits to (Lu et al., 2018; Castano et al., 2025):

- ``ADWINDetector`` and ``PageHinkleyDetector`` consume a per-observation
  *performance* signal (the 0/1 loss, or any error stream) and are the sudden-drift
  detectors. Both wrap River, which is the committed online-learning library.
- ``KSDriftDetector`` and ``PSIDriftDetector`` consume a per-observation *feature*
  value and compare a sliding window against a reference window. They are the
  gradual-drift detectors, and they see covariate shift that a performance detector
  cannot (a feature distribution can move while the loss stays flat).

All four share the interface below, so the simulation harness can hold a
heterogeneous list of detectors and drive them identically:

    detector.update(value)      # one scalar per observation
    detector.drift_detected     # True on the update that raised the signal
    detector.warning_detected   # True in the pre-drift warning zone (ADWIN/PH)
    detector.reset()            # clear all state, including any reference window

Both flags are latched to the most recent ``update`` call, so the harness reads
them immediately after each update and records the observation index. Each
detector also tracks ``n_updates`` and ``drift_indices``, which is what the
detection-latency metric is computed from: latency is the gap between the known
onset index (for example the schema-era boundary) and the first entry in
``drift_indices`` after that onset.

**Scalar in, by design.** Each detector handles one signal. Multivariate covariate
drift is monitored by holding one window detector per feature and aggregating the
per-feature flags (for example, drift when more than k features flag in the same
window), which keeps the per-feature attribution the drift narrative needs.

**Reference windows.** The window detectors take their reference either explicitly
via ``set_reference`` (the usual case: a fixed pre-drift baseline period) or
automatically from the first ``reference_size`` observations of the stream. Until a
reference exists and the sliding window is full, ``update`` accumulates and never
flags. Windows are counted in observations, not days: at fleet scale a single day
carries hundreds of thousands of drive-days, so a window of a few hundred
observations is a small slice of one day, and a day-scale window means tens of
thousands of observations.

**Cost.** The window detectors run their two-sample test every ``test_every``
updates rather than on every observation. At stream scale, testing on every
observation is both expensive and near-duplicative, since consecutive windows
overlap in all but one point. The default of 1 is the conservative choice for
correctness; the simulation sets it to a daily or weekly stride.
"""

from __future__ import annotations

from collections import deque
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.stats import ks_2samp

# Conventional PSI bands (Yurdakul, 2018, and standard credit-risk practice):
# below 0.10 stable, 0.10 to 0.25 moderate shift, above 0.25 significant shift.
PSI_WARNING: float = 0.10
PSI_DRIFT: float = 0.25

# Window defaults are large on purpose. PSI's sampling-noise floor scales with
# (n_bins - 1) * (1/n_ref + 1/n_win), so small windows put the null value of the
# statistic above the 0.25 drift band and the detector flags stationary noise.
# See expected_null_psi below.
DEFAULT_REFERENCE_SIZE: int = 500
DEFAULT_WINDOW_SIZE: int = 500
DEFAULT_KS_ALPHA: float = 0.01
DEFAULT_PSI_BINS: int = 10
_EPS: float = 1e-6


@runtime_checkable
class DriftDetector(Protocol):
    """Structural interface every detector in this module satisfies."""

    def update(self, observation: float) -> None: ...

    @property
    def drift_detected(self) -> bool: ...

    @property
    def warning_detected(self) -> bool: ...

    def reset(self) -> None: ...


class _BaseDetector:
    """Shared bookkeeping: update counter, latched flags, and the drift index log."""

    def __init__(self) -> None:
        self._n_updates: int = 0
        self._drift: bool = False
        self._warning: bool = False
        self._drift_indices: list[int] = []

    @property
    def n_updates(self) -> int:
        """Observations consumed since construction or the last ``reset``."""
        return self._n_updates

    @property
    def drift_detected(self) -> bool:
        """True when the most recent ``update`` raised a drift signal."""
        return self._drift

    @property
    def warning_detected(self) -> bool:
        """True when the most recent ``update`` sits in a warning zone."""
        return self._warning

    @property
    def drift_indices(self) -> list[int]:
        """0-based stream indices at which drift was signalled."""
        return list(self._drift_indices)

    def _record(self, drift: bool, warning: bool) -> None:
        self._drift = drift
        self._warning = warning
        if drift:
            self._drift_indices.append(self._n_updates)
        self._n_updates += 1

    def reset(self) -> None:
        self._n_updates = 0
        self._drift = False
        self._warning = False
        self._drift_indices = []


# ---------------------------------------------------------------------------
# Performance-signal detectors (sudden drift)
# ---------------------------------------------------------------------------
def _river_drift():
    """Import ``river.drift`` lazily so the module loads without River installed."""
    try:
        from river import drift
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "river is required for ADWINDetector and PageHinkleyDetector. "
            "Install it with `pip install river`."
        ) from exc
    return drift


class ADWINDetector(_BaseDetector):
    """ADWIN over a per-observation loss stream (wraps ``river.drift.ADWIN``).

    ADWIN keeps a variable-length window of recent values and splits it whenever
    two sub-windows have significantly different means, which makes it sensitive to
    an abrupt change in error rate. Feed it the 0/1 loss (``1 - correct``) or any
    bounded error signal, one value per observation.

    ``delta`` is the confidence bound: smaller means fewer false alarms and slower
    detection. River's ADWIN exposes no warning zone, so ``warning_detected``
    mirrors ``drift_detected``, keeping the interface uniform.
    """

    def __init__(self, delta: float = 0.002) -> None:
        super().__init__()
        self.delta = delta
        self._adwin = _river_drift().ADWIN(delta=delta)

    def update(self, observation: float) -> None:
        self._adwin.update(float(observation))
        drift = bool(self._adwin.drift_detected)
        self._record(drift, drift)

    @property
    def width(self) -> int:
        """Current ADWIN window width (shrinks sharply on a detection)."""
        return int(self._adwin.width)

    def reset(self) -> None:
        super().reset()
        self._adwin = _river_drift().ADWIN(delta=self.delta)


class PageHinkleyDetector(_BaseDetector):
    """Page-Hinkley over a per-observation loss stream (wraps ``river.drift.PageHinkley``).

    Page-Hinkley accumulates the deviation of each value from the running mean and
    signals when the cumulative deviation exceeds ``threshold``. It is the
    complementary sudden-drift test to ADWIN: a sequential change-point statistic
    rather than a windowing one, so the two disagree in informative ways.

    ``min_instances`` is the burn-in before any signal can fire, ``delta`` the
    tolerated magnitude of drift, ``threshold`` the detection bound. River's
    Page-Hinkley exposes no warning zone, so ``warning_detected`` mirrors
    ``drift_detected``.

    **Defaults are tuned against the false-alarm rate on a 0/1 loss stream, not
    taken from the library.** The PH statistic on a Bernoulli loss with error rate
    p is a random walk of per-step variance ``p(1-p)`` carrying a downward drift of
    ``delta``, so its excursions above the running minimum are exponential with
    scale ``p(1-p) / (2 * delta)``. At a 5% error rate and the library-typical
    ``delta=0.005``, that scale is about 4.75, so any threshold near 5 is crossed by
    noise alone within a few hundred observations, and the drift log fills with
    events that mean nothing. ``delta=0.01`` with ``threshold=15`` puts a noise
    crossing far into the tail while still detecting a jump from a 5% to a 60%
    error rate within roughly 30 observations. Retune deliberately if the loss
    signal changes scale, and check the quiet-stream test when doing so.
    """

    def __init__(
        self,
        min_instances: int = 30,
        delta: float = 0.01,
        threshold: float = 15.0,
        alpha: float = 1 - 1e-4,
    ) -> None:
        super().__init__()
        self.min_instances = min_instances
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self._ph = self._build()

    def _build(self):
        return _river_drift().PageHinkley(
            min_instances=self.min_instances,
            delta=self.delta,
            threshold=self.threshold,
            alpha=self.alpha,
        )

    def update(self, observation: float) -> None:
        self._ph.update(float(observation))
        drift = bool(self._ph.drift_detected)
        self._record(drift, drift)

    def reset(self) -> None:
        super().reset()
        self._ph = self._build()


# ---------------------------------------------------------------------------
# Window detectors (gradual drift, covariate shift)
# ---------------------------------------------------------------------------
class _WindowDetector(_BaseDetector):
    """Reference window versus sliding window, tested every ``test_every`` updates."""

    def __init__(
        self,
        reference_size: int = DEFAULT_REFERENCE_SIZE,
        window_size: int = DEFAULT_WINDOW_SIZE,
        test_every: int = 1,
    ) -> None:
        super().__init__()
        if reference_size < 2 or window_size < 2:
            raise ValueError("reference_size and window_size must be at least 2")
        if test_every < 1:
            raise ValueError("test_every must be at least 1")
        self.reference_size = reference_size
        self.window_size = window_size
        self.test_every = test_every
        self._reference: np.ndarray | None = None
        self._auto_ref: deque[float] = deque(maxlen=reference_size)
        self._window: deque[float] = deque(maxlen=window_size)
        self._statistic: float = float("nan")

    @property
    def statistic(self) -> float:
        """Test statistic from the most recent evaluation (NaN before the first one)."""
        return self._statistic

    @property
    def has_reference(self) -> bool:
        return self._reference is not None

    def set_reference(self, values) -> None:
        """Fix the reference window explicitly (the usual case: a baseline period)."""
        ref = np.asarray(values, dtype=np.float64).ravel()
        ref = ref[np.isfinite(ref)]
        if ref.size < 2:
            raise ValueError("reference needs at least 2 finite values")
        self._reference = ref

    def update(self, observation: float) -> None:
        value = float(observation)
        if not np.isfinite(value):
            # Missingness is a signal elsewhere in the study, but it is not a
            # distributional value; skip it rather than poison the window.
            self._record(False, False)
            return

        if self._reference is None:
            self._auto_ref.append(value)
            if len(self._auto_ref) == self.reference_size:
                self._reference = np.asarray(self._auto_ref, dtype=np.float64)
            self._record(False, False)
            return

        self._window.append(value)
        ready = len(self._window) == self.window_size
        due = (self._n_updates % self.test_every) == 0
        if not (ready and due):
            self._record(False, False)
            return

        drift, warning, stat = self._evaluate(
            self._reference, np.asarray(self._window, dtype=np.float64)
        )
        self._statistic = stat
        self._record(drift, warning)

    def _evaluate(self, reference: np.ndarray, window: np.ndarray) -> tuple[bool, bool, float]:
        raise NotImplementedError

    def reset(self) -> None:
        super().reset()
        self._reference = None
        self._auto_ref.clear()
        self._window.clear()
        self._statistic = float("nan")


class KSDriftDetector(_WindowDetector):
    """Two-sample Kolmogorov-Smirnov test between a reference and a sliding window.

    KS is distribution-free and sensitive to any change in shape, not only in the
    mean, which is what gradual covariate shift looks like as fleet composition
    turns over. Drift is signalled when the KS p-value falls below ``alpha``;
    the warning band is a p-value below ``10 * alpha``.

    ``statistic`` carries the KS D statistic from the most recent test.
    """

    def __init__(
        self,
        reference_size: int = DEFAULT_REFERENCE_SIZE,
        window_size: int = DEFAULT_WINDOW_SIZE,
        alpha: float = DEFAULT_KS_ALPHA,
        test_every: int = 1,
    ) -> None:
        super().__init__(reference_size, window_size, test_every)
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self._p_value: float = float("nan")

    @property
    def p_value(self) -> float:
        """KS p-value from the most recent test (NaN before the first one)."""
        return self._p_value

    def _evaluate(self, reference: np.ndarray, window: np.ndarray) -> tuple[bool, bool, float]:
        stat, p = ks_2samp(reference, window)
        self._p_value = float(p)
        return bool(p < self.alpha), bool(p < min(1.0, 10 * self.alpha)), float(stat)


def expected_null_psi(n_bins: int, n_reference: int, n_window: int) -> float:
    """Approximate PSI under the null (no drift), from sampling noise alone.

    PSI is asymptotically ``chi2(n_bins - 1) / n`` in scale, giving an expected null
    value of about ``(n_bins - 1) * (1 / n_reference + 1 / n_window)`` (Yurdakul,
    2018). The consequence is easy to miss and easy to be burned by: with 10 bins, a
    100-point reference, and a 50-point window, the expected null PSI is about 0.27,
    which is *above* the conventional 0.25 "significant drift" band. At those sizes
    the bands are noise, and a detector will flag drift on a perfectly stationary
    stream.

    ``PSIDriftDetector`` uses this to refuse window sizes at which its own thresholds
    cannot be trusted. The remedy when it refuses is a larger window, a larger
    reference, or fewer bins.
    """
    if n_bins < 2 or n_reference < 1 or n_window < 1:
        raise ValueError("n_bins >= 2 and positive sample sizes are required")
    return float((n_bins - 1) * (1.0 / n_reference + 1.0 / n_window))


class PSIDriftDetector(_WindowDetector):
    """Population Stability Index between a reference and a sliding window.

    PSI bins both samples on quantile edges taken from the reference and sums
    ``(actual - expected) * ln(actual / expected)`` across bins. It is the
    industry-standard covariate-shift monitor and is reported on the conventional
    bands: below 0.10 stable, 0.10 to 0.25 moderate (warning), above 0.25
    significant (drift).

    Quantile edges are taken from the reference and deduplicated, which matters
    here because the primary SMART attributes are zero-inflated: a reference window
    of mostly zeros collapses to few distinct edges, and PSI is then computed over
    the bins that actually exist rather than over empty ones.

    **Small-window guard.** The conventional bands are only meaningful when the
    windows are large relative to the bin count, because PSI has a sampling-noise
    floor of about ``(n_bins - 1) * (1 / n_reference + 1 / n_window)``. The
    constructor computes that floor and refuses any configuration where it exceeds
    ``max_null_fraction`` of the drift threshold, rather than emitting drift events
    that are pure noise. Raise the window sizes or lower ``n_bins`` to satisfy it.
    """

    def __init__(
        self,
        reference_size: int = DEFAULT_REFERENCE_SIZE,
        window_size: int = DEFAULT_WINDOW_SIZE,
        n_bins: int = DEFAULT_PSI_BINS,
        drift_threshold: float = PSI_DRIFT,
        warning_threshold: float = PSI_WARNING,
        test_every: int = 1,
        max_null_fraction: float = 0.25,
    ) -> None:
        super().__init__(reference_size, window_size, test_every)
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2")
        null_floor = expected_null_psi(n_bins, reference_size, window_size)
        if null_floor > max_null_fraction * drift_threshold:
            raise ValueError(
                f"PSI is unreliable at these sizes: the expected null PSI is "
                f"{null_floor:.3f} against a drift threshold of {drift_threshold:.2f} "
                f"(n_bins={n_bins}, reference_size={reference_size}, "
                f"window_size={window_size}). Increase the window sizes, reduce "
                f"n_bins, or raise max_null_fraction deliberately."
            )
        self.n_bins = n_bins
        self.drift_threshold = drift_threshold
        self.warning_threshold = warning_threshold
        self.null_floor = null_floor

    def _evaluate(self, reference: np.ndarray, window: np.ndarray) -> tuple[bool, bool, float]:
        psi = population_stability_index(reference, window, n_bins=self.n_bins)
        return bool(psi > self.drift_threshold), bool(psi > self.warning_threshold), psi


def population_stability_index(reference, window, n_bins: int = DEFAULT_PSI_BINS) -> float:
    """PSI between two samples, binned on quantile edges from ``reference``.

    Returns 0.0 when the reference is degenerate (a single distinct value) and the
    window matches it, which is the correct reading for an all-zero SMART attribute
    that stays all-zero. Proportions are floored at a small epsilon so an empty bin
    contributes a finite penalty rather than an infinity.
    """
    ref = np.asarray(reference, dtype=np.float64).ravel()
    win = np.asarray(window, dtype=np.float64).ravel()
    ref = ref[np.isfinite(ref)]
    win = win[np.isfinite(win)]
    if ref.size == 0 or win.size == 0:
        return float("nan")

    edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        # Degenerate reference: one distinct value. PSI reduces to the mismatch in
        # the share of the window that sits on that value.
        expected = 1.0
        actual = float(np.mean(win == ref[0]))
        actual = min(max(actual, _EPS), 1.0)
        return float((actual - expected) * np.log(actual / expected))

    # Open the outer edges so values beyond the reference range fall in the end bins.
    edges[0], edges[-1] = -np.inf, np.inf
    expected = np.histogram(ref, bins=edges)[0] / ref.size
    actual = np.histogram(win, bins=edges)[0] / win.size
    expected = np.clip(expected, _EPS, None)
    actual = np.clip(actual, _EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))
