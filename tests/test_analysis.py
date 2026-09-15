from __future__ import annotations

import numpy as np
import pytest

from point_tracking_filter.core.analysis import (
    AnalysisError,
    apply_transform,
    best_fit_transform,
    consistency_report,
    deviation_stats,
    resample_to,
    transformed_track,
)
from point_tracking_filter.core.align import rotation_from_euler
from point_tracking_filter.core.model import Track

RNG = np.random.default_rng(20260904)


def _curve(n: int = 200, start: float = 0.0, step: float = 0.01) -> Track:
    t = start + step * np.arange(n)
    xyz = np.column_stack([np.sin(t * 3), np.cos(t * 2) * 2.0, t * 1.5 + np.sin(t * 5)])
    return Track(t=t, xyz=xyz, name="curve", source="oak-d")


def test_deviation_against_itself_is_zero():
    track = _curve()
    stats = deviation_stats(track, track)

    assert stats.n_samples == len(track)
    np.testing.assert_allclose(stats.axis_mean, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(stats.axis_std, np.zeros(3), atol=1e-12)
    assert stats.euclidean_mean == pytest.approx(0.0, abs=1e-12)
    assert stats.rmse == pytest.approx(0.0, abs=1e-12)
    assert stats.p95 == pytest.approx(0.0, abs=1e-12)


def test_constant_offset_is_recovered():
    truth = _curve()
    shifted = truth.copy()
    shifted.xyz = truth.xyz + np.array([0.5, -0.25, 1.0])

    stats = deviation_stats(shifted, truth)
    np.testing.assert_allclose(stats.axis_mean, [0.5, -0.25, 1.0], atol=1e-9)
    np.testing.assert_allclose(stats.axis_std, np.zeros(3), atol=1e-9)
    assert stats.euclidean_mean == pytest.approx(np.linalg.norm([0.5, -0.25, 1.0]))


def test_resample_does_not_bridge_gaps():
    truth = _curve(n=100)
    truth.xyz[40:60] = np.nan

    times = truth.t.copy()
    resampled = resample_to(truth, times)
    assert np.all(np.isnan(resampled[41:59]))
    np.testing.assert_allclose(resampled[:40], truth.xyz[:40], atol=1e-12)


def test_resample_outside_the_range_is_nan():
    truth = _curve(n=50)
    resampled = resample_to(truth, np.array([truth.t[0] - 1.0, truth.t[-1] + 1.0]))
    assert np.all(np.isnan(resampled))


def test_non_overlapping_tracks_report_zero_samples():
    truth = _curve(n=50, start=0.0)
    other = _curve(n=50, start=100.0)
    stats = deviation_stats(other, truth)

    assert stats.n_samples == 0
    assert stats.n_excluded == 50
    assert np.isnan(stats.euclidean_mean)


def test_best_fit_recovers_a_known_transform():
    points = RNG.normal(size=(300, 3)) * 10.0

    transform = np.eye(4)
    transform[:3, :3] = rotation_from_euler((7.0, -11.0, 23.0))
    transform[:3, 3] = [1.5, -2.0, 0.75]

    target = apply_transform(transform, points)
    fitted = best_fit_transform(points, target)
    np.testing.assert_allclose(apply_transform(fitted, points), target, atol=1e-6)


def test_rigid_fit_ignores_scale():
    points = RNG.normal(size=(100, 3))
    fitted = best_fit_transform(points, points * 2.0)
    assert np.linalg.det(fitted[:3, :3]) == pytest.approx(1.0)


def test_best_fit_requires_enough_points():
    with pytest.raises(AnalysisError, match="at least 4 point pairs"):
        best_fit_transform(np.zeros((3, 3)), np.zeros((3, 3)))


def test_consistency_report_never_gets_worse():
    truth = _curve(n=400)
    analyzed = truth.copy()
    analyzed.xyz = apply_transform(
        np.array(
            [
                [1.0, 0.02, 0.0, 3.0],
                [0.0, 1.0, 0.0, -1.0],
                [0.0, 0.0, 1.0, 0.5],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        truth.xyz,
    ) + RNG.normal(scale=0.01, size=(len(truth), 3))

    report = consistency_report(analyzed, truth)
    assert report.after.euclidean_mean <= report.before.euclidean_mean
    assert report.improvement >= 0.0
    assert report.after.n_samples == report.before.n_samples


def test_consistency_report_removes_a_systematic_offset():
    truth = _curve(n=300)
    analyzed = truth.copy()
    analyzed.xyz = truth.xyz + np.array([2.0, -1.0, 0.5])

    report = consistency_report(analyzed, truth)
    assert report.before.euclidean_mean == pytest.approx(np.linalg.norm([2.0, -1.0, 0.5]))
    assert report.after.euclidean_mean < 1e-6
    assert report.warnings == []


def test_consistency_report_with_too_few_samples():
    truth = _curve(n=50, start=0.0)
    other = _curve(n=50, start=100.0)
    report = consistency_report(other, truth)

    np.testing.assert_allclose(report.transform, np.eye(4))
    assert any("too few" in message for message in report.warnings)


def test_transformed_track_keeps_time_and_gaps():
    track = _curve(n=20)
    track.xyz[5] = np.nan
    transform = np.eye(4)
    transform[:3, 3] = [1.0, 2.0, 3.0]

    result = transformed_track(track, transform)
    np.testing.assert_allclose(result.t, track.t)
    assert not result.valid_mask[5]
    np.testing.assert_allclose(result.xyz[0], track.xyz[0] + [1.0, 2.0, 3.0])
    assert "corrected" in result.name
