from __future__ import annotations

import numpy as np
import pytest

from point_tracking_filter.core.analysis import AnalysisError, smoothness_stats
from point_tracking_filter.core.model import Track
from point_tracking_filter.core.prediction import (
    DEFAULT_CAMERA_LATENCY,
    ConstantVelocityKalman,
    NaiveDifference,
    PredictionError,
    Score,
    SmoothedOffset,
    SpeedScheduledKalman,
    StreamConfig,
    WindowedPolynomial,
    WindowedSplineRefit,
    ZeroOrderHold,
    compare,
    estimate_noise,
    holding_rmse,
    oracle,
    output_times,
    references_for,
    sample_latency,
    schedule_wobble,
    score,
    simulate,
    tune,
)


def _every_predictor() -> list:
    return [
        ZeroOrderHold(),
        NaiveDifference(),
        ConstantVelocityKalman(),
        SpeedScheduledKalman(),
        SmoothedOffset(ConstantVelocityKalman(), horizon=0.05),
        WindowedPolynomial(degree=1),
        WindowedPolynomial(degree=2),
        WindowedSplineRefit(backend="gcv"),
        WindowedSplineRefit(backend="parametric"),
    ]

RATE = 75.0


def _ramp(n: int = 400, velocity: float = 10.0) -> Track:
    """Straight line at constant speed: every predictor should nail it."""
    t = np.arange(n) / RATE
    xyz = np.column_stack([t * velocity, np.zeros(n), np.ones(n)])
    return Track(t=t, xyz=xyz, name="ramp", source="oak-d")


def _wobble(n: int = 600, freq: float = 1.0, amplitude: float = 5.0) -> Track:
    t = np.arange(n) / RATE
    xyz = np.column_stack(
        [amplitude * np.sin(2 * np.pi * freq * t), np.zeros(n), np.zeros(n)]
    )
    return Track(t=t, xyz=xyz, name="wobble", source="oak-d")


def test_prediction_never_looks_ahead():
    """Truncating the recording must not change what was already emitted.

    This is the one defect that would silently invalidate every number the
    evaluation produces, so it is asserted directly.
    """
    track = _wobble()
    # A fixed display rate, so both runs query at bit-identical instants and
    # any difference can only come from the predictor having seen more data.
    config = StreamConfig(
        horizon=0.05, camera_latency=DEFAULT_CAMERA_LATENCY, output_hz=RATE
    )

    full = simulate(track, NaiveDifference(), config)
    half = len(track) // 2
    truncated = simulate(
        Track(t=track.t[:half], xyz=track.xyz[:half], name="cut", source="oak-d"),
        NaiveDifference(),
        config,
    )

    shared = len(truncated)
    assert shared > 10
    np.testing.assert_array_equal(full.t[:shared], truncated.t[:shared])
    np.testing.assert_array_equal(full.xyz[:shared], truncated.xyz[:shared])


def test_output_grid_is_uniform_and_targets_the_horizon():
    track = _ramp()
    config = StreamConfig(horizon=0.05, camera_latency=0.025, output_hz=90.0)
    out = simulate(track, ZeroOrderHold(), config)

    steps = np.diff(out.t)
    np.testing.assert_allclose(steps, 1.0 / 90.0, atol=1e-12)
    # The first query happens once the first sample has arrived, and asks for
    # the position one horizon later.
    assert out.t[0] == pytest.approx(track.t[0] + 0.025 + 0.05)


def test_constant_velocity_is_predicted_exactly():
    track = _ramp(velocity=10.0)
    config = StreamConfig(horizon=0.05, camera_latency=0.025)
    out = simulate(track, NaiveDifference(), config)

    truth = np.column_stack([out.t * 10.0, np.zeros(len(out)), np.ones(len(out))])
    # The first two outputs are the warm-up: velocity needs two samples.
    np.testing.assert_allclose(out.xyz[2:], truth[2:], atol=1e-9)


def test_zero_order_hold_lags_by_the_whole_latency():
    track = _ramp(velocity=10.0)
    config = StreamConfig(horizon=0.05, camera_latency=0.025)
    out = simulate(track, ZeroOrderHold(), config)

    usable = out.valid_mask
    error = out.xyz[usable, 0] - out.t[usable] * 10.0
    # It reports where the target was one full motion-to-photon ago.
    assert np.median(error) == pytest.approx(-10.0 * 0.075, abs=0.2)


def test_dropped_detections_are_never_delivered():
    track = _ramp(n=300)
    track.xyz[100:150] = np.nan

    out = simulate(track, ZeroOrderHold(), StreamConfig(camera_latency=0.0))
    # Only the valid samples reach the client; the very last one may still be
    # in flight when the final query happens.
    assert track.n_valid - 1 <= out.meta["n_consumed"] <= track.n_valid
    # The client keeps the last good position through the dropout rather than
    # receiving nothing.
    assert np.all(np.isfinite(out.xyz[-1]))


def test_predictions_before_the_first_sample_are_nan():
    track = _ramp()
    out = simulate(track, ZeroOrderHold(), StreamConfig(camera_latency=0.0))
    assert np.all(np.isfinite(out.xyz))

    empty = Track(
        t=np.arange(5.0), xyz=np.full((5, 3), np.nan), name="empty", source="oak-d"
    )
    with pytest.raises(PredictionError, match="no valid samples"):
        simulate(empty, ZeroOrderHold())


def test_latency_is_measured_across_the_normalized_time_axis():
    """``detection_time`` stays on the raw clock while ``t`` is shifted."""
    n = 50
    raw_t = 1000.0 + np.arange(n) / RATE
    onset = raw_t[0]
    track = Track(
        t=raw_t - onset,
        xyz=np.zeros((n, 3)),
        name="shifted",
        source="oak-d",
        meta={"detection_time": raw_t + 0.025, "led_onset_raw_time": onset},
    )
    np.testing.assert_allclose(sample_latency(track), 0.025)
    np.testing.assert_allclose(sample_latency(track, 0.01), 0.01)


def test_latency_rejects_inconsistent_metadata():
    track = _ramp(n=10)
    track.meta["detection_time"] = np.arange(3.0)
    with pytest.raises(PredictionError, match="detection_time"):
        sample_latency(track)

    track.meta["detection_time"] = track.t - 1.0
    with pytest.raises(PredictionError, match="precedes"):
        sample_latency(track)


def test_invalid_stream_config_is_rejected():
    with pytest.raises(PredictionError, match="horizon"):
        StreamConfig(horizon=-0.01)
    with pytest.raises(PredictionError, match="camera_latency"):
        StreamConfig(camera_latency=-0.01)
    with pytest.raises(PredictionError, match="output_hz"):
        StreamConfig(output_hz=0.0)


def test_oracle_tracks_the_target_without_lag():
    track = _wobble()
    config = StreamConfig(horizon=0.05, camera_latency=0.025)
    reference = oracle(track, config)

    usable = reference.valid_mask
    truth = 5.0 * np.sin(2 * np.pi * reference.t[usable])
    # Non-causal, so it is not paying the latency the causal predictors pay.
    assert np.max(np.abs(reference.xyz[usable, 0] - truth)) < 0.05


def test_smoothness_of_a_still_target_is_zero():
    n = 200
    track = Track(
        t=np.arange(n) / RATE, xyz=np.ones((n, 3)), name="still", source="filtered"
    )
    stats = smoothness_stats(track)
    assert stats.jerk_rms == pytest.approx(0.0, abs=1e-9)
    assert stats.step_p99 == pytest.approx(0.0, abs=1e-9)


def test_smoothness_ratios_against_itself_are_one():
    track = _wobble()
    stats = smoothness_stats(track, track)
    assert stats.jerk_ratio == pytest.approx(1.0)
    assert stats.hf_ratio == pytest.approx(1.0)
    assert stats.step_ratio == pytest.approx(1.0)


def test_jerk_matches_the_analytic_value():
    amplitude, freq = 5.0, 1.0
    stats = smoothness_stats(_wobble(freq=freq, amplitude=amplitude))
    # d^3/dt^3 of A sin(wt) has amplitude A w^3, and RMS is that over sqrt(2).
    expected = amplitude * (2 * np.pi * freq) ** 3 / np.sqrt(2)
    assert stats.jerk_rms == pytest.approx(expected, rel=1e-2)


def test_smoothness_needs_a_uniform_time_axis():
    t = np.concatenate([np.arange(50) / RATE, [10.0]])
    track = Track(
        t=t, xyz=np.zeros((t.size, 3)), name="ragged", source="filtered"
    )
    with pytest.raises(AnalysisError, match="uniform"):
        smoothness_stats(track)


@pytest.mark.parametrize("predictor", _every_predictor(), ids=lambda p: p.name)
def test_no_predictor_looks_ahead(predictor):
    """The causality guarantee has to hold for every family, not just one."""
    track = _wobble(n=240)
    config = StreamConfig(horizon=0.05, camera_latency=0.025, output_hz=RATE)

    full = simulate(track, predictor, config)
    half = len(track) // 2
    truncated = simulate(
        Track(t=track.t[:half], xyz=track.xyz[:half], name="cut", source="oak-d"),
        predictor,
        config,
    )

    shared = len(truncated)
    assert shared > 10
    np.testing.assert_array_equal(full.xyz[:shared], truncated.xyz[:shared])


@pytest.mark.parametrize("predictor", _every_predictor(), ids=lambda p: p.name)
def test_every_predictor_handles_a_dropout(predictor):
    track = _ramp(n=300)
    track.xyz[100:150] = np.nan

    out = simulate(track, predictor, StreamConfig(horizon=0.05, camera_latency=0.0))
    assert np.all(np.isfinite(out.xyz[-1]))
    # Nothing may run away while the target is unobserved.
    assert np.nanmax(np.abs(out.xyz[:, 0])) < 10.0 * track.t[-1] * 10.0


@pytest.mark.parametrize("predictor", _every_predictor(), ids=lambda p: p.name)
def test_constant_velocity_is_recovered_by_every_family(predictor):
    track = _ramp(n=400, velocity=10.0)
    config = StreamConfig(horizon=0.05, camera_latency=0.025, output_hz=RATE)
    out = simulate(track, predictor, config)

    settled = out.xyz[60:, 0] - out.t[60:] * 10.0
    if isinstance(predictor, ZeroOrderHold):
        pytest.skip("holding the last sample cannot follow a moving target")
    assert np.max(np.abs(settled)) < 1e-6


def test_windowed_polynomial_fits_curvature_exactly():
    n = 300
    t = np.arange(n) / RATE
    track = Track(
        t=t,
        xyz=np.column_stack([3.0 * t**2 - t, np.zeros(n), np.zeros(n)]),
        name="parabola",
        source="oak-d",
    )
    config = StreamConfig(horizon=0.05, camera_latency=0.0, output_hz=RATE)

    quadratic = simulate(track, WindowedPolynomial(degree=2), config)
    truth = 3.0 * quadratic.t**2 - quadratic.t
    np.testing.assert_allclose(quadratic.xyz[20:, 0], truth[20:], atol=1e-8)

    # A straight-line model cannot, and must lag instead.
    linear = simulate(track, WindowedPolynomial(degree=1), config)
    assert np.max(np.abs(linear.xyz[20:, 0] - truth[20:])) > 0.05


def test_kalman_process_noise_trades_lag_against_jitter():
    rng = np.random.default_rng(5)
    track = _wobble(n=600)
    track.xyz = track.xyz + rng.normal(scale=0.1, size=track.xyz.shape)
    config = StreamConfig(horizon=0.05, camera_latency=0.025, output_hz=RATE)
    reference = oracle(track, config)

    calm = smoothness_stats(
        simulate(track, ConstantVelocityKalman(sigma_a=20.0), config), reference
    )
    twitchy = smoothness_stats(
        simulate(track, ConstantVelocityKalman(sigma_a=5000.0), config), reference
    )
    assert calm.jerk_ratio < twitchy.jerk_ratio


def test_spline_refit_records_its_knot_choices():
    rng = np.random.default_rng(7)
    track = _wobble(n=300)
    track.xyz = track.xyz + rng.normal(scale=0.1, size=track.xyz.shape)

    predictor = WindowedSplineRefit(backend="gcv")
    simulate(track, predictor, StreamConfig(horizon=0.05, camera_latency=0.025))

    # The diagnostic must exist; whether the knots actually move is the
    # question the benchmark answers, so no expectation is asserted here.
    assert len(predictor.knot_counts) > 100
    assert 0.0 <= predictor.knot_change_rate <= 1.0


def test_predictor_arguments_are_validated():
    with pytest.raises(PredictionError, match="sigma_a"):
        ConstantVelocityKalman(sigma_a=0.0)
    with pytest.raises(PredictionError, match="sigma_m"):
        ConstantVelocityKalman(sigma_m=-1.0)
    with pytest.raises(PredictionError, match="degree"):
        WindowedPolynomial(degree=9)
    with pytest.raises(PredictionError, match="window_seconds"):
        WindowedPolynomial(window_seconds=0.0)
    with pytest.raises(PredictionError, match="backend"):
        WindowedSplineRefit(backend="nope")


def test_compare_reports_every_pairing():
    track = _wobble(n=200)
    config = StreamConfig(horizon=0.05, camera_latency=0.025, output_hz=RATE)
    predictors = [ZeroOrderHold(), ConstantVelocityKalman(), WindowedPolynomial()]

    rows = compare([track], predictors, config)
    assert len(rows) == 3
    assert {row["predictor"] for row in rows} == {p.name for p in predictors}
    for row in rows:
        assert np.isfinite(row["rmse"])
        assert np.isfinite(row["jerk_ratio"])
        assert row["us_per_update"] > 0


def _noisy_wobble(n: int = 600, scale: float = 0.1, seed: int = 3) -> Track:
    rng = np.random.default_rng(seed)
    track = _wobble(n=n)
    track.xyz = track.xyz + rng.normal(scale=scale, size=track.xyz.shape)
    return track


def test_estimate_noise_recovers_the_measurement_scale():
    clean = _wobble(n=800)
    noisy = clean.copy()
    rng = np.random.default_rng(2)
    noisy.xyz = clean.xyz + rng.normal(scale=0.2, size=clean.xyz.shape)

    sigma_m, sigma_a = estimate_noise(noisy)
    assert sigma_m == pytest.approx(0.2, rel=0.35)
    assert sigma_a > 0


def test_smoothed_offset_calms_its_inner_predictor():
    track = _noisy_wobble()
    config = StreamConfig(horizon=0.05, camera_latency=0.025, output_hz=RATE)
    reference = oracle(track, config)

    bare = ConstantVelocityKalman(sigma_a=200.0)
    wrapped = SmoothedOffset(
        ConstantVelocityKalman(sigma_a=200.0), horizon=0.05, time_constant=0.12
    )
    plain = smoothness_stats(simulate(track, bare, config), reference)
    damped = smoothness_stats(simulate(track, wrapped, config), reference)

    # Damping only the extrapolated part, at identical process noise.
    assert damped.jerk_ratio < plain.jerk_ratio


def _mid_dropout(n: int = 400, lost: slice = slice(150, 260)) -> Track:
    """A ramp with a hole in the middle.

    The hole must not be at the end: the harness stops querying once the
    last valid sample has arrived, so a trailing dropout is never actually
    observed by the client.
    """
    track = _ramp(n=n, velocity=10.0)
    track.xyz[lost] = np.nan
    return track


def test_smoothed_offset_fades_out_when_samples_go_stale():
    track = _mid_dropout()
    config = StreamConfig(horizon=0.05, camera_latency=0.0)
    predictor = SmoothedOffset(
        ConstantVelocityKalman(), horizon=0.05, gate_seconds=0.05
    )
    out = simulate(track, predictor, config)

    during = (out.t > track.t[160]) & (out.t < track.t[255])
    assert np.count_nonzero(during) > 20
    assert np.all(np.isfinite(out.xyz[during]))
    # It stops short of the truth rather than extrapolating ever further.
    drift = out.xyz[during, 0] - track.t[149] * 10.0
    assert np.max(drift) < 10.0 * (track.t[255] - track.t[149])


def test_the_gate_is_what_stops_the_runaway():
    track = _mid_dropout()
    config = StreamConfig(horizon=0.05, camera_latency=0.0)

    def excursion(gate):
        out = simulate(
            track,
            SmoothedOffset(
                ConstantVelocityKalman(), horizon=0.05, gate_seconds=gate
            ),
            config,
        )
        during = (out.t > track.t[160]) & (out.t < track.t[255])
        return float(np.max(out.xyz[during, 0]))

    assert excursion(None) > excursion(0.05)


def test_the_gate_does_not_bite_during_normal_operation():
    """The lead always contains the camera latency; that is not staleness.

    Measuring staleness against the horizon rather than the nominal lead
    made the gate shrink every prediction by a few percent forever.
    """
    track = _ramp(n=400, velocity=10.0)
    config = StreamConfig(horizon=0.05, camera_latency=0.025, output_hz=RATE)

    gated = simulate(
        track,
        SmoothedOffset(ConstantVelocityKalman(), horizon=0.05, gate_seconds=0.1),
        config,
    )
    truth = gated.t * 10.0
    assert np.max(np.abs(gated.xyz[60:, 0] - truth[60:])) < 1e-6


def test_schedule_opens_up_with_speed():
    predictor = SpeedScheduledKalman(
        sigma_slow=2.0, sigma_fast=40.0, speed_lo=20.0, speed_hi=100.0
    )
    assert predictor._schedule(0.0) == pytest.approx(2.0)
    assert predictor._schedule(10.0) == pytest.approx(2.0)
    assert predictor._schedule(200.0) == pytest.approx(40.0)
    assert 2.0 < predictor._schedule(60.0) < 40.0

    # Smoothstep, so the ramp leaves both ends without a kink.
    lower = predictor._schedule(21.0) - predictor._schedule(20.0)
    middle = predictor._schedule(61.0) - predictor._schedule(60.0)
    assert lower < middle


def test_schedule_stays_inside_its_bounds_on_real_motion():
    track = _noisy_wobble(n=500)
    predictor = SpeedScheduledKalman(sigma_slow=2.0, sigma_fast=40.0)
    simulate(track, predictor, StreamConfig(horizon=0.05, camera_latency=0.025))

    history = np.asarray(predictor.sigma_history)
    assert history.size > 100
    assert np.all(history >= 2.0 - 1e-9)
    assert np.all(history <= 40.0 + 1e-9)


def test_schedule_wobble_is_bounded_and_reported():
    track = _noisy_wobble(n=400)
    predictor = SpeedScheduledKalman(sigma_slow=2.0, sigma_fast=40.0)
    wobble = schedule_wobble([track], predictor)
    assert 0.0 <= wobble < 1.0


def test_cost_is_driven_by_the_worst_recording():
    """Averaging would let an over-represented regime pick the parameter."""
    passing = Score(rmses=[1.0], jerk_ratios=[2.0], baseline_rmses=[2.0])
    failing = Score(rmses=[9.0], jerk_ratios=[2.0], baseline_rmses=[2.0])
    both = Score(
        rmses=[1.0, 9.0], jerk_ratios=[2.0, 2.0], baseline_rmses=[2.0, 2.0]
    )

    assert both.cost == pytest.approx(failing.cost)
    assert both.cost > np.mean([passing.cost, failing.cost])
    assert not both.beats_holding
    assert passing.beats_holding


def test_tuning_improves_on_a_bad_starting_point():
    tracks = [_noisy_wobble(n=500, seed=8), _noisy_wobble(n=500, seed=9)]
    config = StreamConfig(horizon=0.05, camera_latency=0.025, output_hz=RATE)
    references = references_for(tracks, config)
    baselines = holding_rmse(tracks, config, references)

    result = tune(
        tracks,
        lambda value: ConstantVelocityKalman(sigma_a=value),
        (1.0, 3000.0),
        config,
        references,
        steps=6,
    )
    awful = score(
        tracks, ConstantVelocityKalman(sigma_a=3000.0), config, references, baselines
    )
    assert result.score.cost < awful.cost
    assert 1.0 <= result.value <= 3000.0
    assert result.evaluated > 1


def test_naive_prediction_buys_accuracy_with_jitter():
    """The trap the smoothness metrics exist to expose.

    Naive extrapolation looks better on deviation alone while making the
    output stream far jerkier than the motion it is tracking.
    """
    rng = np.random.default_rng(11)
    track = _wobble(n=900)
    track.xyz = track.xyz + rng.normal(scale=0.1, size=track.xyz.shape)

    config = StreamConfig(horizon=0.05, camera_latency=0.025)
    reference = oracle(track, config)
    hold = simulate(track, ZeroOrderHold(), config)
    naive = simulate(track, NaiveDifference(), config)

    from point_tracking_filter.core.analysis import deviation_stats

    assert deviation_stats(naive, reference).rmse < deviation_stats(hold, reference).rmse
    assert smoothness_stats(naive, reference).jerk_ratio > 5.0 * smoothness_stats(
        hold, reference
    ).jerk_ratio
