from __future__ import annotations

import numpy as np
import pytest

from point_tracking_filter.core.analysis import deviation_stats
from point_tracking_filter.core.filtering import (
    FilterError,
    SplineParams,
    apply_spline_filter,
    optimize_spline,
)
from point_tracking_filter.core.model import Track

RNG = np.random.default_rng(4711)
NOISE = 0.2


def _clean(n: int = 500) -> Track:
    t = np.arange(n) / 75.0
    xyz = np.column_stack([np.sin(t), np.cos(t * 0.7) * 2.0, t * 0.5])
    return Track(t=t, xyz=xyz, name="clean", source="optitrack")


def _noisy(clean: Track, scale: float = NOISE) -> Track:
    track = clean.copy()
    track.xyz = clean.xyz + RNG.normal(scale=scale, size=clean.xyz.shape)
    track.name = "noisy"
    track.source = "oak-d"
    track.confidence = np.full(len(clean), 0.9)
    return track


def _rmse(track: Track, reference: Track) -> float:
    return float(np.sqrt(np.nanmean(np.sum((track.xyz - reference.xyz) ** 2, axis=1))))


def test_filtering_reduces_the_noise():
    clean = _clean()
    noisy = _noisy(clean)
    filtered = apply_spline_filter(noisy, SplineParams(smoothing=NOISE**2 * 3))

    assert filtered.source == "filtered"
    assert len(filtered) == len(noisy)
    np.testing.assert_allclose(filtered.t, noisy.t)
    assert _rmse(filtered, clean) < 0.5 * _rmse(noisy, clean)


def test_gaps_are_not_bridged():
    clean = _clean(n=300)
    noisy = _noisy(clean)
    noisy.xyz[100:150] = np.nan

    filtered = apply_spline_filter(noisy)
    assert np.all(np.isnan(filtered.xyz[100:150]))
    assert filtered.n_valid == noisy.n_valid


def test_short_segments_pass_through_unchanged():
    t = np.arange(10) / 10.0
    xyz = np.tile(np.arange(10.0)[:, None], (1, 3))
    xyz[2:] = np.nan
    track = Track(t=t, xyz=xyz, name="short", source="oak-d")

    filtered = apply_spline_filter(track)
    np.testing.assert_allclose(filtered.xyz[:2], xyz[:2])
    assert filtered.meta["n_passed_through"] == 2
    assert filtered.meta["n_smoothed"] == 0


def test_zero_smoothing_interpolates_the_samples():
    clean = _clean(n=100)
    filtered = apply_spline_filter(clean, SplineParams(smoothing=0.0))
    np.testing.assert_allclose(filtered.xyz, clean.xyz, atol=1e-6)


def test_confidence_weighting_is_accepted():
    clean = _clean(n=200)
    noisy = _noisy(clean)
    noisy.confidence = RNG.uniform(0.5, 1.0, size=len(noisy))

    filtered = apply_spline_filter(
        noisy, SplineParams(smoothing=NOISE**2 * 3, use_confidence_weights=True)
    )
    assert np.all(np.isfinite(filtered.xyz))
    assert _rmse(filtered, clean) < _rmse(noisy, clean)


def test_resampling_produces_a_uniform_grid():
    clean = _clean(n=300)
    filtered = apply_spline_filter(clean, SplineParams(smoothing=1e-3, resample_hz=30.0))

    assert filtered.confidence is None
    steps = np.diff(filtered.t)
    np.testing.assert_allclose(steps, steps[0], atol=1e-9)
    assert filtered.t[0] == pytest.approx(clean.t[0])
    assert filtered.t[-1] == pytest.approx(clean.t[-1])


def test_invalid_parameters_are_rejected():
    with pytest.raises(FilterError, match="smoothing"):
        SplineParams(smoothing=-1.0)
    with pytest.raises(FilterError, match="resample_hz"):
        SplineParams(resample_hz=0.0)


def test_filtering_an_empty_track_raises():
    track = Track(t=np.arange(3.0), xyz=np.full((3, 3), np.nan), name="empty", source="oak-d")
    with pytest.raises(FilterError, match="no valid samples"):
        apply_spline_filter(track)


def test_optimizer_beats_the_default():
    clean = _clean(n=400)
    noisy = _noisy(clean)

    default = SplineParams()
    default_cost = deviation_stats(apply_spline_filter(noisy, default), clean).euclidean_mean

    best, cost = optimize_spline(noisy, clean, default)
    assert cost <= default_cost
    assert cost < deviation_stats(noisy, clean).euclidean_mean

    recomputed = deviation_stats(apply_spline_filter(noisy, best), clean).euclidean_mean
    assert recomputed == pytest.approx(cost, rel=1e-9)


def test_optimizer_without_ground_truth_raises():
    noisy = _noisy(_clean(n=50))
    empty = Track(
        t=np.arange(3.0), xyz=np.full((3, 3), np.nan), name="empty", source="optitrack"
    )
    with pytest.raises(FilterError, match="no valid samples"):
        optimize_spline(noisy, empty)


def test_gcv_automatic_smoothing_reduces_the_noise():
    clean = _clean()
    noisy = _noisy(clean)
    filtered = apply_spline_filter(noisy, SplineParams(auto_smoothing=True))
    assert _rmse(filtered, clean) < 0.5 * _rmse(noisy, clean)


def test_gcv_requires_at_least_five_samples_per_segment():
    t = np.arange(4) / 10.0
    xyz = np.tile(np.arange(4.0)[:, None], (1, 3))
    track = Track(t=t, xyz=xyz, name="short", source="oak-d")

    filtered = apply_spline_filter(track, SplineParams())
    assert filtered.meta["n_passed_through"] == 4
    assert filtered.meta["n_smoothed"] == 0


def test_optimizer_disables_auto_smoothing_on_the_search():
    clean = _clean(n=400)
    noisy = _noisy(clean)

    best, cost = optimize_spline(noisy, clean, SplineParams(auto_smoothing=True))
    assert best.auto_smoothing is False
    assert cost < deviation_stats(noisy, clean).euclidean_mean
