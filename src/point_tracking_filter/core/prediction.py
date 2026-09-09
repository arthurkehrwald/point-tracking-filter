"""Causal prediction of future positions for real-time use.

Offline smoothing (:mod:`.filtering`) may look at the whole recording. A
real-time client cannot. At wall-clock time ``tau`` only those samples whose
detection has already finished are available, and the position the client
needs is the one the target will have once the photons reach the eye.

Everything in this module is therefore strictly causal, and
:func:`simulate` enforces that by construction: a predictor is only ever
handed samples that had become available at the time it is asked to predict.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import ClassVar

import numpy as np

from .model import Track

DEFAULT_HORIZON = 0.05
DEFAULT_CAMERA_LATENCY = 0.025


class PredictionError(ValueError):
    """Raised when a track cannot be replayed or predicted."""


@dataclass(frozen=True)
class StreamConfig:
    """How a recording is replayed as if it arrived in real time.

    ``horizon`` is how far beyond the current wall-clock instant the client
    needs the position, i.e. the latency still ahead of the prediction
    (transport plus display). The camera latency already spent before a
    sample becomes available is modelled separately, so the total
    motion-to-photon lead over the newest shutter is
    ``horizon + camera_latency``.

    ``camera_latency`` of ``None`` uses the per-sample latency measured in
    the recording (``detection_time - capture_time``) instead of a constant,
    which also reproduces its jitter.

    Outputs are emitted on a uniform grid at ``output_hz`` because the
    smoothness metrics differentiate the output stream and so require a
    constant step. ``None`` falls back to the recording's own median rate,
    which reads the whole recording to do so: the predictions stay causal
    either way, but pass the real display rate when the query timetable
    itself must not depend on data that had not arrived yet.
    """

    horizon: float = DEFAULT_HORIZON
    camera_latency: float | None = None
    output_hz: float | None = None

    def __post_init__(self) -> None:
        if self.horizon < 0:
            raise PredictionError(f"horizon must not be negative, got {self.horizon}")
        if self.camera_latency is not None and self.camera_latency < 0:
            raise PredictionError(
                f"camera_latency must not be negative, got {self.camera_latency}"
            )
        if self.output_hz is not None and self.output_hz <= 0:
            raise PredictionError(f"output_hz must be positive, got {self.output_hz}")


class Predictor(ABC):
    """A causal, streaming estimator of future positions.

    ``update`` is called once per arriving sample in arrival order, and
    ``predict`` may be called for any target time. Implementations must not
    retain a reference to future data; :func:`simulate` never offers any.
    """

    name: ClassVar[str] = "predictor"

    @abstractmethod
    def reset(self) -> None:
        """Forget all state, as if freshly constructed."""

    @abstractmethod
    def update(self, t: float, xyz: np.ndarray, confidence: float | None = None) -> None:
        """Take a sample captured at time ``t``."""

    @abstractmethod
    def predict(self, t_target: float) -> np.ndarray:
        """Position estimate for ``t_target``, or NaN while not yet usable."""


class ZeroOrderHold(Predictor):
    """Repeat the newest sample. The cost of not predicting at all."""

    name: ClassVar[str] = "Zero-order hold"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._xyz: np.ndarray | None = None

    def update(self, t: float, xyz: np.ndarray, confidence: float | None = None) -> None:
        self._xyz = np.asarray(xyz, dtype=float).reshape(3)

    def predict(self, t_target: float) -> np.ndarray:
        if self._xyz is None:
            return np.full(3, np.nan)
        return self._xyz.copy()


class NaiveDifference(Predictor):
    """Extrapolate along the velocity of the two most recent samples.

    Deliberately unguarded: this is the failure mode the smoothness metrics
    exist to detect, so it must be free to misbehave.
    """

    name: ClassVar[str] = "Naive difference"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._previous: tuple[float, np.ndarray] | None = None
        self._latest: tuple[float, np.ndarray] | None = None

    def update(self, t: float, xyz: np.ndarray, confidence: float | None = None) -> None:
        self._previous = self._latest
        self._latest = (float(t), np.asarray(xyz, dtype=float).reshape(3))

    def predict(self, t_target: float) -> np.ndarray:
        if self._latest is None:
            return np.full(3, np.nan)
        t_last, xyz_last = self._latest
        if self._previous is None:
            return xyz_last.copy()
        t_previous, xyz_previous = self._previous
        step = t_last - t_previous
        if step <= 0:
            return xyz_last.copy()
        velocity = (xyz_last - xyz_previous) / step
        return xyz_last + velocity * (t_target - t_last)


class ConstantVelocityKalman(Predictor):
    """Recursive constant-velocity estimator with white acceleration noise.

    The three axes share their dynamics and their measurement noise, so a
    single covariance describes all of them and only the state is per-axis.
    Propagation uses the elapsed time rather than a sample count, so a
    dropped frame simply integrates for longer.

    ``sigma_a`` is the acceleration noise density and sets the whole
    lag-versus-jitter tradeoff; ``sigma_m`` is the measurement noise, about
    0.1 cm on these recordings.
    """

    name: ClassVar[str] = "Constant-velocity Kalman"

    def __init__(
        self,
        sigma_a: float = 200.0,
        sigma_m: float = 0.1,
        use_confidence_weights: bool = False,
    ) -> None:
        if sigma_a <= 0:
            raise PredictionError(f"sigma_a must be positive, got {sigma_a}")
        if sigma_m <= 0:
            raise PredictionError(f"sigma_m must be positive, got {sigma_m}")
        self.sigma_a = float(sigma_a)
        self.sigma_m = float(sigma_m)
        self.use_confidence_weights = use_confidence_weights
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((3, 2))
        self._covariance = np.zeros((2, 2))
        self._t: float | None = None

    def _propagate(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        transition = np.array([[1.0, dt], [0.0, 1.0]])
        noise = self.sigma_a**2 * np.array(
            [[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]]
        )
        state = self._state @ transition.T
        covariance = transition @ self._covariance @ transition.T + noise
        return state, covariance

    def update(self, t: float, xyz: np.ndarray, confidence: float | None = None) -> None:
        measurement = np.asarray(xyz, dtype=float).reshape(3)
        if self._t is None:
            self._state = np.column_stack([measurement, np.zeros(3)])
            # The velocity is entirely unknown until a second sample arrives.
            self._covariance = np.diag([self.sigma_m**2, 1e6])
            self._t = float(t)
            return

        dt = float(t) - self._t
        if dt > 0:
            self._state, self._covariance = self._propagate(dt)
            self._t = float(t)

        sigma_m = self.sigma_m
        if self.use_confidence_weights and confidence is not None:
            sigma_m = sigma_m / max(float(confidence), 1e-3)

        innovation = measurement - self._state[:, 0]
        innovation_covariance = self._covariance[0, 0] + sigma_m**2
        gain = self._covariance[:, 0] / innovation_covariance
        self._state = self._state + np.outer(innovation, gain)
        self._covariance = self._covariance - np.outer(gain, self._covariance[0, :])

    def predict(self, t_target: float) -> np.ndarray:
        if self._t is None:
            return np.full(3, np.nan)
        dt = float(t_target) - self._t
        return self._state[:, 0] + self._state[:, 1] * dt


class _TrailingWindow(Predictor):
    """Shared bookkeeping for predictors that refit a trailing window."""

    def __init__(self, window_seconds: float) -> None:
        if window_seconds <= 0:
            raise PredictionError(
                f"window_seconds must be positive, got {window_seconds}"
            )
        self.window_seconds = float(window_seconds)
        self.reset()

    def reset(self) -> None:
        self._t: list[float] = []
        self._xyz: list[np.ndarray] = []
        self._confidence: list[float] = []

    def update(self, t: float, xyz: np.ndarray, confidence: float | None = None) -> None:
        t = float(t)
        if self._t and t <= self._t[-1]:
            return  # A repeated or out-of-order timestamp carries no new information.
        self._t.append(t)
        self._xyz.append(np.asarray(xyz, dtype=float).reshape(3))
        self._confidence.append(1.0 if confidence is None else float(confidence))
        cutoff = t - self.window_seconds
        keep = 0
        while keep < len(self._t) - 1 and self._t[keep] < cutoff:
            keep += 1
        if keep:
            del self._t[:keep], self._xyz[:keep], self._confidence[:keep]

    def _window(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray(self._t),
            np.asarray(self._xyz),
            np.asarray(self._confidence),
        )


class WindowedPolynomial(_TrailingWindow):
    """Least-squares polynomial over a trailing window, evaluated ahead.

    This is a spline stripped of the parts that make a spline: no interior
    knots and a fixed low order, so the basis cannot change from frame to
    frame. The window is measured in seconds rather than samples, so a
    dropped frame shortens it instead of silently stretching it.
    """

    name: ClassVar[str] = "Windowed polynomial"

    def __init__(
        self,
        window_seconds: float = 0.2,
        degree: int = 1,
        decay_seconds: float | None = None,
        use_confidence_weights: bool = False,
    ) -> None:
        if not 1 <= degree <= 3:
            raise PredictionError(f"degree must be between 1 and 3, got {degree}")
        if decay_seconds is not None and decay_seconds <= 0:
            raise PredictionError(
                f"decay_seconds must be positive, got {decay_seconds}"
            )
        self.degree = int(degree)
        self.decay_seconds = decay_seconds
        self.use_confidence_weights = use_confidence_weights
        super().__init__(window_seconds)

    def predict(self, t_target: float) -> np.ndarray:
        times, values, confidence = self._window()
        if times.size == 0:
            return np.full(3, np.nan)
        if times.size < self.degree + 1:
            return values[-1].copy()

        relative = times - times[-1]
        design = np.vander(relative, self.degree + 1, increasing=True)

        weights = np.ones(times.size)
        if self.decay_seconds is not None:
            weights *= np.exp(relative / self.decay_seconds)
        if self.use_confidence_weights:
            weights *= np.clip(confidence, 1e-6, None)
        root = np.sqrt(weights)[:, None]

        coefficients, *_ = np.linalg.lstsq(design * root, values * root, rcond=None)
        powers = (float(t_target) - times[-1]) ** np.arange(self.degree + 1)
        return powers @ coefficients


class WindowedSplineRefit(_TrailingWindow):
    """Refit a smoothing spline to the trailing window and extrapolate.

    The literal form of "fit the past and read off the future". Both
    backends place their knots automatically, so the number of knots is
    recorded on every fit: if it changes as the window slides, the basis
    itself is moving underneath the extrapolation and the output jumps for
    reasons unrelated to the target's motion.
    """

    name: ClassVar[str] = "Windowed spline refit"

    def __init__(
        self,
        window_seconds: float = 0.3,
        backend: str = "gcv",
        smoothing: float = 0.05,
        degree: int = 3,
        auto_smoothing: bool = False,
    ) -> None:
        if backend not in ("gcv", "parametric"):
            raise PredictionError(f"unknown backend {backend!r}")
        if smoothing < 0:
            raise PredictionError(f"smoothing must not be negative, got {smoothing}")
        self.backend = backend
        self.smoothing = float(smoothing)
        self.degree = int(degree)
        self.auto_smoothing = auto_smoothing
        super().__init__(window_seconds)

    def reset(self) -> None:
        super().reset()
        self.knot_counts: list[int] = []

    @property
    def knot_change_rate(self) -> float:
        """Fraction of fits whose knot count differed from the previous fit."""
        if len(self.knot_counts) < 2:
            return float("nan")
        counts = np.asarray(self.knot_counts)
        return float(np.mean(counts[1:] != counts[:-1]))

    def _minimum_samples(self) -> int:
        return 5 if self.backend == "gcv" else max(self.degree + 1, 4)

    def predict(self, t_target: float) -> np.ndarray:
        from scipy.interpolate import make_smoothing_spline, make_splprep

        times, values, _ = self._window()
        if times.size == 0:
            return np.full(3, np.nan)
        if times.size < self._minimum_samples():
            return values[-1].copy()

        try:
            if self.backend == "gcv":
                lam = None if self.auto_smoothing else self.smoothing
                spline = make_smoothing_spline(times, values, lam=lam)
                self.knot_counts.append(int(spline.t.size))
                return np.asarray(spline(float(t_target))).reshape(3)

            spline, _ = make_splprep(
                [values[:, 0], values[:, 1], values[:, 2]],
                u=times,
                k=self.degree,
                s=self.smoothing * times.size * 3,
            )
            self.knot_counts.append(int(spline.t.size))
            return np.asarray(spline(float(t_target))).reshape(3)
        except (ValueError, np.linalg.LinAlgError):
            # A degenerate window must not abort the run; hold instead.
            return values[-1].copy()


class SpeedScheduledKalman(ConstantVelocityKalman):
    """Constant-velocity filter whose process noise follows the speed.

    No fixed process noise serves both regimes: the recordings want
    ``sigma_a`` near 1-3 when the target crawls and near 20 when it moves,
    and the gap is several-fold in jitter. Jitter is worst when the target is
    slow, because there is no real motion to mask it, while lag is worst when
    it is fast -- so the two requirements do not actually conflict.

    The schedule is driven by a heavily smoothed speed, with a fast attack
    and a slow release, because a parameter that itself jitters becomes a
    noise source: re-picking the GCV penalty every frame costs a factor of
    nine in jerk on these recordings. ``smoothstep`` keeps the transition
    C1-continuous so nothing pops at the ends of the ramp.
    """

    name: ClassVar[str] = "Speed-scheduled Kalman"

    def __init__(
        self,
        sigma_slow: float = 2.0,
        sigma_fast: float = 20.0,
        speed_lo: float = 20.0,
        speed_hi: float = 100.0,
        attack_seconds: float = 0.05,
        release_seconds: float = 0.40,
        sigma_m: float = 0.1,
        use_confidence_weights: bool = False,
    ) -> None:
        if speed_hi <= speed_lo:
            raise PredictionError(
                f"speed_hi must exceed speed_lo, got {speed_hi} <= {speed_lo}"
            )
        if attack_seconds <= 0 or release_seconds <= 0:
            raise PredictionError("attack and release times must be positive")
        self.sigma_slow = float(sigma_slow)
        self.sigma_fast = float(sigma_fast)
        self.speed_lo = float(speed_lo)
        self.speed_hi = float(speed_hi)
        self.attack_seconds = float(attack_seconds)
        self.release_seconds = float(release_seconds)
        super().__init__(
            sigma_a=sigma_slow,
            sigma_m=sigma_m,
            use_confidence_weights=use_confidence_weights,
        )

    def reset(self) -> None:
        super().reset()
        self._speed = 0.0
        self.sigma_history: list[float] = []

    def _schedule(self, speed: float) -> float:
        span = (speed - self.speed_lo) / (self.speed_hi - self.speed_lo)
        span = min(max(span, 0.0), 1.0)
        blend = span * span * (3.0 - 2.0 * span)
        return self.sigma_slow + (self.sigma_fast - self.sigma_slow) * blend

    def update(self, t: float, xyz: np.ndarray, confidence: float | None = None) -> None:
        if self._t is not None:
            dt = max(float(t) - self._t, 0.0)
            observed = float(np.linalg.norm(self._state[:, 1]))
            tau = self.attack_seconds if observed > self._speed else self.release_seconds
            self._speed += (1.0 - np.exp(-dt / tau)) * (observed - self._speed)
            self.sigma_a = self._schedule(self._speed)
        self.sigma_history.append(self.sigma_a)
        super().update(t, xyz, confidence)


class SmoothedOffset(Predictor):
    """Wrap a predictor and damp only the part that extrapolates.

    The output is split into a level and an offset,
    ``p(t + h) = p(t) + delta``. Nearly all the injected jitter rides on
    ``delta``, because it carries the velocity estimate multiplied by the
    horizon. Smoothing ``delta`` therefore buys smoothness far more cheaply
    than filtering the output would: it delays only the lead term, whereas
    filtering the position adds lag to the tracked position itself.

    ``gate_seconds`` fades the offset out when the newest sample goes stale,
    so a dropout degrades continuously towards holding the last position
    instead of extrapolating into nowhere. Staleness is measured against
    ``nominal_lead`` rather than the horizon, because the lead between the
    newest capture and the requested time always includes the camera's own
    latency: measuring against the horizon would read that ~25 ms as a
    permanent dropout and quietly shrink every prediction. Left as ``None``
    the nominal lead is learned as the smallest lead seen so far, which is
    the one observed just after a sample lands.
    """

    name: ClassVar[str] = "Smoothed offset"

    def __init__(
        self,
        inner: Predictor,
        horizon: float = DEFAULT_HORIZON,
        time_constant: float = 0.05,
        gate_seconds: float | None = 0.1,
        nominal_lead: float | None = None,
    ) -> None:
        if horizon <= 0:
            raise PredictionError(f"horizon must be positive, got {horizon}")
        if time_constant < 0:
            raise PredictionError(
                f"time_constant must not be negative, got {time_constant}"
            )
        if gate_seconds is not None and gate_seconds <= 0:
            raise PredictionError(f"gate_seconds must be positive, got {gate_seconds}")
        if nominal_lead is not None and nominal_lead <= 0:
            raise PredictionError(
                f"nominal_lead must be positive, got {nominal_lead}"
            )
        self.inner = inner
        self.horizon = float(horizon)
        self.time_constant = float(time_constant)
        self.gate_seconds = gate_seconds
        self.nominal_lead = nominal_lead
        self.reset()

    @property
    def name_with_inner(self) -> str:
        return f"{self.inner.name} + smoothed offset"

    def reset(self) -> None:
        self.inner.reset()
        self._offset = np.zeros(3)
        self._t: float | None = None
        self._seen_lead = float("inf")

    def update(self, t: float, xyz: np.ndarray, confidence: float | None = None) -> None:
        self.inner.update(t, xyz, confidence)
        base = self.inner.predict(float(t))
        raw = self.inner.predict(float(t) + self.horizon) - base
        if not np.all(np.isfinite(raw)):
            return
        if self._t is None or self.time_constant == 0.0:
            self._offset = raw
        else:
            dt = max(float(t) - self._t, 0.0)
            self._offset += (1.0 - np.exp(-dt / self.time_constant)) * (
                raw - self._offset
            )
        self._t = float(t)

    def predict(self, t_target: float) -> np.ndarray:
        if self._t is None:
            return np.full(3, np.nan)
        base = self.inner.predict(self._t)
        lead = float(t_target) - self._t
        gain = lead / self.horizon
        if self.gate_seconds is not None:
            nominal = self.nominal_lead
            if nominal is None:
                self._seen_lead = min(self._seen_lead, lead)
                nominal = self._seen_lead
            staleness = max(lead - nominal, 0.0)
            gain /= 1.0 + (staleness / self.gate_seconds) ** 2
        return base + self._offset * gain


def estimate_noise(track: Track) -> tuple[float, float]:
    """Seed values for ``sigma_m`` and ``sigma_a``, measured off a recording.

    The measurement scale is the spread of the raw samples around the
    offline fit. The acceleration noise density follows from the velocity
    increments of that fit, since the constant-velocity model treats them as
    white: ``var(dv) = sigma_a^2 * dt``. Both are starting points for
    :func:`tune`, not final values -- the offline fit suppresses some of the
    very motion the second estimate is trying to measure.
    """
    from .filtering import SplineParams, apply_spline_filter

    smoothed = apply_spline_filter(
        track, SplineParams(method="gcv", auto_smoothing=True)
    )
    residual = (track.xyz - smoothed.xyz)[track.valid_mask]
    if residual.size == 0:
        raise PredictionError(f"{track.name}: nothing to estimate noise from")
    sigma_m = float(np.sqrt(np.nanmean(residual**2)))

    increments: list[np.ndarray] = []
    steps: list[np.ndarray] = []
    for indices in smoothed.segments():
        if indices.size < 4:
            continue
        times = smoothed.t[indices]
        velocity = np.diff(smoothed.xyz[indices], axis=0) / np.diff(times)[:, None]
        increments.append(np.diff(velocity, axis=0))
        steps.append(np.diff(times)[1:])
    if not increments:
        return sigma_m, 1.0
    stacked = np.vstack(increments)
    dt = float(np.median(np.concatenate(steps)))
    sigma_a = float(np.sqrt(np.mean(stacked**2) / dt))
    return sigma_m, max(sigma_a, 1e-6)


def sample_latency(track: Track, camera_latency: float | None = None) -> np.ndarray:
    """Per-sample delay from shutter until the position is available.

    With ``camera_latency=None`` the value measured in the recording is
    used. ``detection_time`` stays on the recording's raw clock while ``t``
    has been shifted to the LED onset, so the shift is undone before the
    difference is taken.
    """
    if camera_latency is not None:
        return np.full(len(track), float(camera_latency))

    detection = track.meta.get("detection_time")
    if detection is None:
        return np.full(len(track), DEFAULT_CAMERA_LATENCY)

    detection = np.asarray(detection, dtype=float).reshape(-1)
    if detection.shape[0] != len(track):
        raise PredictionError(
            f"{track.name}: detection_time has {detection.shape[0]} entries "
            f"for {len(track)} samples"
        )
    raw_t = track.t + float(track.meta.get("led_onset_raw_time", 0.0))
    latency = detection - raw_t
    if not np.all(np.isfinite(latency)):
        raise PredictionError(f"{track.name}: detection_time contains invalid entries")
    if np.any(latency < 0):
        raise PredictionError(f"{track.name}: detection_time precedes the capture time")
    return latency


def output_times(track: Track, config: StreamConfig) -> np.ndarray:
    """Uniform grid of wall-clock instants at which the client asks."""
    available = track.t + sample_latency(track, config.camera_latency)
    usable = available[track.valid_mask]
    if usable.size == 0:
        raise PredictionError(f"{track.name}: no valid samples to replay")

    rate = config.output_hz
    if rate is None:
        step = np.median(np.diff(track.t)) if len(track) > 1 else 0.0
        if not np.isfinite(step) or step <= 0:
            raise PredictionError(f"{track.name}: cannot infer the sample rate")
        rate = 1.0 / float(step)

    start, end = float(usable[0]), float(available[track.valid_mask][-1])
    count = max(int(np.floor((end - start) * rate)) + 1, 1)
    return start + np.arange(count) / rate


def simulate(track: Track, predictor: Predictor, config: StreamConfig | None = None) -> Track:
    """Replay ``track`` as a real-time stream and collect the predictions.

    The returned track is stamped with the *target* times, so it can be
    compared against a ground truth with the ordinary deviation analysis and
    lines up with the real trajectory when shown in the player.
    """
    config = config or StreamConfig()
    predictor.reset()

    latency = sample_latency(track, config.camera_latency)
    available = track.t + latency
    order = np.argsort(available, kind="stable")
    valid = track.valid_mask

    queries = output_times(track, config)
    predictions = np.full((queries.size, 3), np.nan)

    cursor = 0
    n_consumed = 0
    for index, now in enumerate(queries):
        while cursor < order.size and available[order[cursor]] <= now:
            sample = order[cursor]
            cursor += 1
            if not valid[sample]:
                continue  # A dropped detection never reaches the client.
            confidence = (
                float(track.confidence[sample]) if track.confidence is not None else None
            )
            predictor.update(float(track.t[sample]), track.xyz[sample], confidence)
            n_consumed += 1
        predictions[index] = predictor.predict(now + config.horizon)

    return Track(
        t=queries + config.horizon,
        xyz=predictions,
        name=(
            f"{track.name} "
            f"({getattr(predictor, 'name_with_inner', predictor.name)} "
            f"+{config.horizon * 1000:.0f} ms)"
        ),
        source="filtered",
        meta={
            **track.meta,
            "filter": "prediction",
            "predictor": getattr(predictor, "name_with_inner", predictor.name),
            "stream_config": config,
            "source_name": track.name,
            "n_consumed": n_consumed,
            "camera_latency_median": float(np.median(latency)),
        },
    )


@dataclass
class Score:
    """How a predictor did across a set of recordings.

    The per-recording figures are kept rather than only their averages,
    because the recordings sit in very different motion regimes and a mean
    lets a predictor fail badly on one while passing overall.
    """

    rmses: list[float]
    jerk_ratios: list[float]
    baseline_rmses: list[float]

    @property
    def rmse(self) -> float:
        return float(np.mean(self.rmses))

    @property
    def jerk_ratio(self) -> float:
        return float(np.mean(self.jerk_ratios))

    @property
    def baseline_rmse(self) -> float:
        return float(np.mean(self.baseline_rmses))

    @property
    def beats_holding(self) -> bool:
        """True only when every recording clears the bar, not the average."""
        return all(
            rmse <= baseline
            for rmse, baseline in zip(self.rmses, self.baseline_rmses)
        )

    @staticmethod
    def _one(rmse: float, jerk_ratio: float, baseline: float) -> float:
        if not (np.isfinite(rmse) and np.isfinite(jerk_ratio)) or jerk_ratio <= 0:
            return float("inf")
        excess = max(0.0, rmse / baseline - 1.0)
        return abs(np.log(jerk_ratio)) + 100.0 * excess

    @property
    def costs(self) -> list[float]:
        """Per-recording cost.

        ``|log(jerk_ratio)|`` is two-sided on purpose: smoothing real motion
        away is as wrong as adding noise. Accuracy enters as a penalty
        rather than a term, because the brief is that smoothness matters
        more -- accuracy only has to be no worse than not predicting at all.
        """
        return [
            self._one(rmse, jerk, baseline)
            for rmse, jerk, baseline in zip(
                self.rmses, self.jerk_ratios, self.baseline_rmses
            )
        ]

    @property
    def cost(self) -> float:
        """Worst recording's cost, not the average.

        Averaging lets whichever regime happens to be over-represented in
        the tuning set drag the parameter towards itself: with three slow
        recordings against two fast ones, the mean picks a process noise
        that then fails on held-out fast motion. The deployed filter has to
        work in every regime, so it is tuned against its hardest case.
        """
        return float(np.max(self.costs))


@dataclass
class TuningResult:
    """The value the search settled on, and how it did."""

    value: float
    score: Score
    evaluated: int


def references_for(tracks: list[Track], config: StreamConfig) -> list[Track]:
    """Offline bounds for each recording, computed once and reused."""
    return [oracle(track, config) for track in tracks]


def holding_rmse(
    tracks: list[Track], config: StreamConfig, references: list[Track]
) -> list[float]:
    """Accuracy of not predicting at all: the bar every candidate must clear."""
    from .analysis import deviation_stats

    return [
        deviation_stats(simulate(track, ZeroOrderHold(), config), reference).rmse
        for track, reference in zip(tracks, references)
    ]


def score(
    tracks: list[Track],
    predictor: Predictor,
    config: StreamConfig,
    references: list[Track] | None = None,
    baselines: list[float] | None = None,
) -> Score:
    """Average accuracy and jitter of ``predictor`` over ``tracks``."""
    from .analysis import deviation_stats, smoothness_stats

    references = references or references_for(tracks, config)
    if baselines is None:
        baselines = holding_rmse(tracks, config, references)
    errors, ratios = [], []
    for track, reference in zip(tracks, references):
        predicted = simulate(track, predictor, config)
        errors.append(deviation_stats(predicted, reference).rmse)
        ratios.append(smoothness_stats(predicted, reference).jerk_ratio)
    return Score(rmses=errors, jerk_ratios=ratios, baseline_rmses=list(baselines))


def tune(
    tracks: list[Track],
    factory,
    bounds: tuple[float, float],
    config: StreamConfig | None = None,
    references: list[Track] | None = None,
    steps: int = 12,
) -> TuningResult:
    """Search ``factory``'s single knob for the calmest usable predictor.

    A coarse sweep on a logarithmic grid rather than a gradient method: the
    cost has a hard constraint penalty in it and the parameter spans orders
    of magnitude, so a sweep is both more robust and gives the frontier for
    free.
    """
    from scipy.optimize import minimize_scalar

    config = config or StreamConfig()
    references = references or references_for(tracks, config)
    baselines = holding_rmse(tracks, config, references)
    low, high = np.log10(bounds[0]), np.log10(bounds[1])
    evaluated = 0

    def cost(exponent: float) -> float:
        nonlocal evaluated
        evaluated += 1
        try:
            return score(
                tracks, factory(10.0**exponent), config, references, baselines
            ).cost
        except (PredictionError, ValueError):
            return float("inf")

    grid = np.linspace(low, high, steps)
    costs = [cost(exponent) for exponent in grid]
    best = int(np.argmin(costs))
    if not np.isfinite(costs[best]):
        raise PredictionError("no usable parameter found in the given bounds")

    # Refine inside the bracket around the best grid point.
    left = grid[max(best - 1, 0)]
    right = grid[min(best + 1, steps - 1)]
    if right > left:
        refined = minimize_scalar(
            cost, bounds=(left, right), method="bounded", options={"xatol": 1e-2}
        )
        if refined.fun <= costs[best]:
            value = 10.0**refined.x
            return TuningResult(
                value=value,
                score=score(tracks, factory(value), config, references, baselines),
                evaluated=evaluated,
            )

    value = 10.0 ** grid[best]
    return TuningResult(
        value=value,
        score=score(tracks, factory(value), config, references, baselines),
        evaluated=evaluated,
    )


def schedule_wobble(
    tracks: list[Track],
    predictor: SpeedScheduledKalman,
    config: StreamConfig | None = None,
) -> float:
    """How much measurement noise alone moves the scheduled process noise.

    The risk with scheduling is that the knob follows the noise rather than
    the motion, and a knob that jitters injects jitter -- re-picking the GCV
    penalty every frame costs a factor of nine in jerk on these recordings.

    Comparing jitter against the same filter frozen at the mean process
    noise cannot detect that, because the schedule is *meant* to be more
    responsive during fast motion, where absolute jerk dominates. So instead
    the same recording is replayed twice, once as measured and once after
    offline smoothing, and the two ``sigma_a`` trajectories are compared.
    Any difference is caused by measurement noise, which is exactly the
    failure being tested for. Values near zero mean the schedule is
    following the motion; approaching one it is following the noise.
    """
    from .filtering import SplineParams, apply_spline_filter

    config = config or StreamConfig()
    deviations = []
    for track in tracks:
        clean = apply_spline_filter(
            track, SplineParams(method="gcv", auto_smoothing=True)
        )
        clean.meta = dict(track.meta)

        simulate(track, predictor, config)
        noisy_sigma = np.asarray(predictor.sigma_history)
        simulate(clean, predictor, config)
        clean_sigma = np.asarray(predictor.sigma_history)

        length = min(noisy_sigma.size, clean_sigma.size)
        if length == 0:
            continue
        noisy_sigma, clean_sigma = noisy_sigma[:length], clean_sigma[:length]
        scale = np.mean(clean_sigma)
        if scale <= 0:
            continue
        deviations.append(float(np.sqrt(np.mean((noisy_sigma - clean_sigma) ** 2)) / scale))
    return float(np.mean(deviations)) if deviations else float("nan")


def compare(
    tracks: list[Track],
    predictors: list[Predictor],
    config: StreamConfig | None = None,
) -> list[dict]:
    """Score every predictor on every recording against the offline bound.

    Returns one row per (recording, predictor) with the accuracy and the
    smoothness statistics side by side, plus the per-update cost, since a
    predictor that cannot keep up with the camera is not a candidate.
    """
    from .analysis import deviation_stats, smoothness_stats

    config = config or StreamConfig()
    rows: list[dict] = []
    for track in tracks:
        reference = oracle(track, config)
        for predictor in predictors:
            started = perf_counter()
            predicted = simulate(track, predictor, config)
            elapsed = perf_counter() - started

            accuracy = deviation_stats(predicted, reference)
            smoothness = smoothness_stats(predicted, reference)
            updates = max(predicted.meta["n_consumed"], 1)
            row = {
                "recording": track.name,
                "predictor": getattr(predictor, "name_with_inner", predictor.name),
                "rmse": accuracy.rmse,
                "p95": accuracy.p95,
                "jerk_ratio": smoothness.jerk_ratio,
                "hf_ratio": smoothness.hf_ratio,
                "step_ratio": smoothness.step_ratio,
                "us_per_update": elapsed / updates * 1e6,
            }
            if isinstance(predictor, WindowedSplineRefit):
                row["knot_change_rate"] = predictor.knot_change_rate
            rows.append(row)
    return rows


def oracle(track: Track, config: StreamConfig | None = None, **filter_kwargs) -> Track:
    """Non-causal upper bound: the offline smoother on the output grid.

    Not a predictor. It is what the causal predictors are giving up by
    being unable to look ahead, and it doubles as the reference trajectory
    when no independent ground truth is available.
    """
    from .analysis import resample_to
    from .filtering import SplineParams, apply_spline_filter

    config = config or StreamConfig()
    params = SplineParams(**filter_kwargs) if filter_kwargs else SplineParams(
        method="gcv", auto_smoothing=True
    )
    smoothed = apply_spline_filter(track, params)
    targets = output_times(track, config) + config.horizon
    return Track(
        t=targets,
        xyz=resample_to(smoothed, targets),
        name=f"{track.name} (oracle)",
        source="filtered",
        meta={
            **track.meta,
            "filter": "oracle",
            "spline_params": params,
            "source_name": track.name,
        },
    )
