"""Unit tests for ``src/evaluation/drift_detectors.py``.

Each detector is driven with a synthetic stream whose drift point is known by
construction, and asserted on three things: it stays silent before the change, it
flags within a bounded latency after the change, and it stays silent on a matched
stationary stream (the false-alarm guard). PSI is additionally anchored on
analytically known cases, including the zero-inflated one that the primary SMART
attributes exhibit, and on its small-window guard: at small window sizes the
sampling-noise floor of PSI exceeds the conventional 0.25 drift band, so the
detector must refuse the configuration rather than emit noise as drift.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.drift_detectors import (
    ADWINDetector,
    DriftDetector,
    KSDriftDetector,
    PageHinkleyDetector,
    PSIDriftDetector,
    expected_null_psi,
    population_stability_index,
)

SEED = 42
CHANGE_POINT = 500
N = 1000

# PSI needs larger windows than KS to keep its null floor under the drift band.
PSI_CHANGE_POINT = 3000
PSI_N = 6000
PSI_WINDOW = 500


def _loss_stream(seed: int = SEED, drifting: bool = True) -> np.ndarray:
    """0/1 loss: 5% error rate, rising to 60% at the change point when drifting."""
    rng = np.random.default_rng(seed)
    pre = rng.binomial(1, 0.05, CHANGE_POINT)
    post = rng.binomial(1, 0.60 if drifting else 0.05, N - CHANGE_POINT)
    return np.concatenate([pre, post]).astype(float)


def _feature_stream(
    seed: int = SEED,
    drifting: bool = True,
    n: int = N,
    change_point: int = CHANGE_POINT,
) -> np.ndarray:
    """Feature values: N(0, 1), shifting to N(3, 1) at the change point when drifting."""
    rng = np.random.default_rng(seed)
    pre = rng.normal(0.0, 1.0, change_point)
    post = rng.normal(3.0 if drifting else 0.0, 1.0, n - change_point)
    return np.concatenate([pre, post])


def _run(detector, stream) -> list[int]:
    for value in stream:
        detector.update(value)
    return detector.drift_indices


# ---------------------------------------------------------------------------
# Interface conformance
# ---------------------------------------------------------------------------
def test_all_detectors_satisfy_the_protocol() -> None:
    pytest.importorskip("river")
    for detector in (
        ADWINDetector(),
        PageHinkleyDetector(),
        KSDriftDetector(),
        PSIDriftDetector(),
    ):
        assert isinstance(detector, DriftDetector)
        assert detector.n_updates == 0
        assert detector.drift_detected is False
        assert detector.warning_detected is False


def test_reset_clears_state() -> None:
    detector = KSDriftDetector(reference_size=50, window_size=50, alpha=0.01, test_every=25)
    _run(detector, _feature_stream())
    assert detector.n_updates == N
    assert detector.drift_indices
    detector.reset()
    assert detector.n_updates == 0
    assert detector.drift_indices == []
    assert detector.has_reference is False
    assert detector.drift_detected is False


# ---------------------------------------------------------------------------
# Performance-signal detectors (sudden drift)
# ---------------------------------------------------------------------------
def test_adwin_detects_a_sudden_error_rate_jump() -> None:
    pytest.importorskip("river")
    detector = ADWINDetector(delta=0.002)
    indices = _run(detector, _loss_stream())
    assert indices, "ADWIN did not flag a 5% to 60% error-rate jump"
    first = indices[0]
    assert first >= CHANGE_POINT, "ADWIN flagged before the change point"
    assert first - CHANGE_POINT <= 200, f"detection latency {first - CHANGE_POINT} too long"


def test_adwin_is_quiet_on_a_stationary_stream() -> None:
    pytest.importorskip("river")
    detector = ADWINDetector(delta=0.002)
    assert _run(detector, _loss_stream(drifting=False)) == []


def test_page_hinkley_detects_a_sudden_error_rate_jump() -> None:
    pytest.importorskip("river")
    detector = PageHinkleyDetector()
    indices = _run(detector, _loss_stream())
    assert indices, "Page-Hinkley did not flag a 5% to 60% error-rate jump"
    first = indices[0]
    assert first >= CHANGE_POINT, "Page-Hinkley flagged before the change point"
    assert first - CHANGE_POINT <= 200, f"detection latency {first - CHANGE_POINT} too long"


def test_page_hinkley_is_quiet_on_a_stationary_stream() -> None:
    """False-alarm guard. The defaults exist because of this test.

    A threshold near 5 with delta=0.005 is crossed by noise alone on a 5% error
    stream, since the PH excursion scale is p(1-p) / (2 * delta), about 4.75 there.
    A false positive here means the defaults have been loosened and the drift event
    log will fill with events that carry no signal.
    """
    pytest.importorskip("river")
    detector = PageHinkleyDetector()
    assert _run(detector, _loss_stream(drifting=False)) == []


def test_page_hinkley_defaults_keep_the_noise_scale_below_the_threshold() -> None:
    """The tuning relationship itself, so a future default change trips a test.

    Excursions of the PH statistic above its running minimum are exponential with
    scale p(1-p) / (2 * delta) on a Bernoulli(p) loss. The threshold must sit well
    into that tail (here, at least five scale lengths) for the detector to be quiet
    under stationarity.
    """
    pytest.importorskip("river")
    detector = PageHinkleyDetector()
    p = 0.05
    noise_scale = p * (1 - p) / (2 * detector.delta)
    assert detector.threshold >= 5 * noise_scale


# ---------------------------------------------------------------------------
# Window detectors (gradual drift / covariate shift)
# ---------------------------------------------------------------------------
def test_ks_detects_a_covariate_shift_within_two_windows() -> None:
    window = 50
    detector = KSDriftDetector(reference_size=100, window_size=window, alpha=0.001, test_every=10)
    indices = _run(detector, _feature_stream())
    assert indices, "KS did not flag a shift from N(0,1) to N(3,1)"
    first = indices[0]
    assert first >= CHANGE_POINT, "KS flagged before the change point"
    assert first - CHANGE_POINT <= 2 * window, f"detection latency {first - CHANGE_POINT} too long"


def test_ks_is_quiet_on_a_stationary_stream() -> None:
    detector = KSDriftDetector(reference_size=100, window_size=50, alpha=0.001, test_every=10)
    assert _run(detector, _feature_stream(drifting=False)) == []


def test_psi_detects_a_covariate_shift_within_two_windows() -> None:
    detector = PSIDriftDetector(
        reference_size=PSI_WINDOW, window_size=PSI_WINDOW, n_bins=10, test_every=50
    )
    stream = _feature_stream(n=PSI_N, change_point=PSI_CHANGE_POINT)
    indices = _run(detector, stream)
    assert indices, "PSI did not flag a shift from N(0,1) to N(3,1)"
    first = indices[0]
    assert first >= PSI_CHANGE_POINT, "PSI flagged before the change point"
    latency = first - PSI_CHANGE_POINT
    assert latency <= 2 * PSI_WINDOW, f"detection latency {latency} too long"
    assert detector.statistic > detector.drift_threshold


def test_psi_is_quiet_on_a_stationary_stream() -> None:
    detector = PSIDriftDetector(
        reference_size=PSI_WINDOW, window_size=PSI_WINDOW, n_bins=10, test_every=50
    )
    stream = _feature_stream(drifting=False, n=PSI_N, change_point=PSI_CHANGE_POINT)
    assert _run(detector, stream) == []


def test_psi_refuses_windows_where_its_own_bands_are_noise() -> None:
    # 10 bins with a 100-point reference and a 50-point window puts the expected
    # null PSI near 0.27, above the 0.25 drift band. The detector must refuse.
    assert expected_null_psi(10, 100, 50) > 0.25
    with pytest.raises(ValueError, match="unreliable"):
        PSIDriftDetector(reference_size=100, window_size=50, n_bins=10)
    # Fewer bins over the same windows brings the floor back under control.
    PSIDriftDetector(reference_size=100, window_size=50, n_bins=2)


def test_expected_null_psi_falls_with_sample_size() -> None:
    assert expected_null_psi(10, 500, 500) < expected_null_psi(10, 100, 100)
    assert expected_null_psi(10, 500, 500) == pytest.approx(9 * (1 / 500 + 1 / 500))


def test_window_detectors_stay_silent_until_the_reference_is_built() -> None:
    detector = PSIDriftDetector(reference_size=PSI_WINDOW, window_size=PSI_WINDOW, test_every=1)
    for value in _feature_stream(n=PSI_N, change_point=PSI_CHANGE_POINT)[: PSI_WINDOW - 1]:
        detector.update(value)
        assert detector.drift_detected is False
    assert detector.has_reference is False


def test_explicit_reference_is_used_instead_of_the_first_observations() -> None:
    rng = np.random.default_rng(SEED)
    detector = KSDriftDetector(reference_size=100, window_size=50, alpha=0.001, test_every=10)
    detector.set_reference(rng.normal(0.0, 1.0, 500))
    assert detector.has_reference is True
    # The stream is shifted from the first observation, so drift must be flagged
    # roughly one window in, with no reference burn-in.
    indices = _run(detector, rng.normal(3.0, 1.0, 200))
    assert indices and indices[0] <= 100


def test_non_finite_values_are_skipped_not_windowed() -> None:
    detector = PSIDriftDetector(reference_size=PSI_WINDOW, window_size=PSI_WINDOW, test_every=1)
    for value in [np.nan] * 20:
        detector.update(value)
        assert detector.drift_detected is False
    assert detector.has_reference is False
    assert detector.n_updates == 20


# ---------------------------------------------------------------------------
# PSI analytic cases
# ---------------------------------------------------------------------------
def test_psi_is_zero_for_identical_samples() -> None:
    rng = np.random.default_rng(SEED)
    sample = rng.normal(size=1000)
    assert population_stability_index(sample, sample) == pytest.approx(0.0, abs=1e-9)


def test_psi_bands_are_ordered_by_shift_magnitude() -> None:
    rng = np.random.default_rng(SEED)
    reference = rng.normal(0.0, 1.0, 5000)
    small = population_stability_index(reference, rng.normal(0.1, 1.0, 5000))
    large = population_stability_index(reference, rng.normal(2.0, 1.0, 5000))
    assert small < 0.10 < 0.25 < large


def test_psi_handles_a_degenerate_all_zero_reference() -> None:
    zeros = np.zeros(500)
    # Stays all-zero: no shift.
    assert population_stability_index(zeros, np.zeros(200)) == pytest.approx(0.0, abs=1e-9)
    # Half the window leaves zero: a real shift, and finite rather than infinite.
    psi = population_stability_index(zeros, np.concatenate([np.zeros(100), np.ones(100)]))
    assert np.isfinite(psi) and psi > 0.25
