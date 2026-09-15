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
DEFAULT_SPLINE_SMOOTHING = 1e-4


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
        smoothing: float = DEFAULT_SPLINE_SMOOTHING,
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
        auto_smoothing=True
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
