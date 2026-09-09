"""Spline based smoothing of tracking data."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from scipy.interpolate import UnivariateSpline, make_smoothing_spline, make_splprep
from scipy.optimize import minimize_scalar

from .analysis import deviation_stats
from .model import Track

MIN_SEGMENT_SAMPLES = 4
GCV_MIN_SEGMENT_SAMPLES = 5
DEFAULT_SMOOTHING = 1.0

SmoothingMethod = Literal["fitpack", "gcv", "parametric"]

METHOD_LABELS: dict[SmoothingMethod, str] = {
    "fitpack": "FITPACK (per-axis)",
    "gcv": "GCV smoothing spline",
    "parametric": "Parametric curve (joint xyz)",
}


class FilterError(ValueError):
    """Raised when a track cannot be filtered."""


@dataclass(frozen=True)
class SplineParams:
    """Parameters of the smoothing spline.

    ``method`` selects the smoothing technique:

    - ``"fitpack"`` fits each axis independently with a FITPACK smoothing
      spline (:func:`~scipy.interpolate.UnivariateSpline`), penalizing jumps
      of the ``degree``-th derivative. ``smoothing`` is the condition per
      sample; the effective value handed to the spline is
      ``smoothing * n_samples`` so it stays comparable across recordings of
      different length.
    - ``"gcv"`` fits x, y and z jointly as a cubic smoothing spline
      (:func:`~scipy.interpolate.make_smoothing_spline`), penalizing the
      integral of the squared second derivative. ``smoothing`` is the
      penalty parameter :math:`\\lambda` directly (no per-sample scaling,
      since its units already differ across recording lengths). If
      ``auto_smoothing`` is set, :math:`\\lambda` is instead chosen
      automatically via generalized cross-validation.
    - ``"parametric"`` fits x, y and z jointly as a single parametric curve
      over time (:func:`~scipy.interpolate.make_splprep`), sharing one knot
      vector across axes. ``smoothing`` is again a per-sample condition,
      scaled by ``smoothing * n_samples * 3`` to stay roughly comparable to
      the per-axis FITPACK budget.
    """

    method: SmoothingMethod = "fitpack"
    smoothing: float = DEFAULT_SMOOTHING
    degree: int = 3
    use_confidence_weights: bool = False
    resample_hz: float | None = None
    auto_smoothing: bool = False

    def __post_init__(self) -> None:
        if self.method not in METHOD_LABELS:
            raise FilterError(f"unknown smoothing method {self.method!r}")
        if self.smoothing < 0:
            raise FilterError(f"smoothing must not be negative, got {self.smoothing}")
        if not 1 <= self.degree <= 5:
            raise FilterError(f"spline degree must be between 1 and 5, got {self.degree}")
        if self.resample_hz is not None and self.resample_hz <= 0:
            raise FilterError(f"resample_hz must be positive, got {self.resample_hz}")

    def with_smoothing(self, smoothing: float) -> SplineParams:
        return replace(self, smoothing=float(smoothing))


def _segment_times(t: np.ndarray, resample_hz: float | None) -> np.ndarray:
    if resample_hz is None:
        return t
    count = max(int(np.floor((t[-1] - t[0]) * resample_hz)) + 1, 2)
    return np.linspace(t[0], t[-1], count)


def _min_segment_samples(params: SplineParams) -> int:
    if params.method == "gcv":
        return max(GCV_MIN_SEGMENT_SAMPLES, MIN_SEGMENT_SAMPLES)
    return max(params.degree + 1, MIN_SEGMENT_SAMPLES)


def _fit_segment(
    seg_t: np.ndarray,
    seg_xyz: np.ndarray,
    out_t: np.ndarray,
    weights: np.ndarray | None,
    params: SplineParams,
) -> np.ndarray:
    if params.method == "fitpack":
        out_xyz = np.empty((out_t.size, 3))
        smoothing = params.smoothing * seg_t.size
        for axis in range(3):
            spline = UnivariateSpline(
                seg_t, seg_xyz[:, axis], w=weights, k=params.degree, s=smoothing
            )
            out_xyz[:, axis] = spline(out_t)
        return out_xyz

    if params.method == "gcv":
        lam = None if params.auto_smoothing else params.smoothing
        spline = make_smoothing_spline(seg_t, seg_xyz, w=weights, lam=lam)
        return spline(out_t)

    # "parametric": fit x, y and z jointly, sharing one knot vector.
    smoothing = params.smoothing * seg_t.size * 3
    spline, _ = make_splprep(
        [seg_xyz[:, 0], seg_xyz[:, 1], seg_xyz[:, 2]],
        u=seg_t,
        w=weights,
        k=params.degree,
        s=smoothing,
    )
    return spline(out_t).T


def apply_spline_filter(track: Track, params: SplineParams | None = None) -> Track:
    """Fit a smoothing spline per axis and return the smoothed track.

    Each contiguous run of valid samples is fitted separately, so gaps are
    never bridged. Segments too short for the requested degree are copied
    through unchanged.
    """
    params = params or SplineParams()

    kept: list[np.ndarray] = []
    times: list[np.ndarray] = []
    values: list[np.ndarray] = []
    n_smoothed = 0
    n_passed_through = 0

    for indices in track.segments():
        seg_t = track.t[indices]
        seg_xyz = track.xyz[indices]

        unique = np.concatenate(([True], np.diff(seg_t) > 0))
        seg_t, seg_xyz = seg_t[unique], seg_xyz[unique]
        indices = indices[unique]

        if seg_t.size < _min_segment_samples(params):
            kept.append(indices)
            times.append(seg_t)
            values.append(seg_xyz)
            n_passed_through += int(seg_t.size)
            continue

        weights = None
        if params.use_confidence_weights and track.confidence is not None:
            weights = np.clip(track.confidence[indices], 1e-6, None)

        out_t = _segment_times(seg_t, params.resample_hz)
        out_xyz = _fit_segment(seg_t, seg_xyz, out_t, weights, params)

        kept.append(indices)
        times.append(out_t)
        values.append(out_xyz)
        n_smoothed += int(seg_t.size)

    if not times:
        raise FilterError(f"{track.name}: no valid samples to filter")

    if params.resample_hz is None:
        # Keep the original sampling, including the gaps as NaN.
        filtered_xyz = np.full_like(track.xyz, np.nan)
        for indices, segment in zip(kept, values):
            filtered_xyz[indices] = segment
        out_t = track.t.copy()
        out_xyz = filtered_xyz
        confidence = track.confidence
    else:
        out_t = np.concatenate(times)
        out_xyz = np.vstack(values)
        confidence = None

    result = Track(
        t=out_t,
        xyz=out_xyz,
        name=f"{track.name} (spline)",
        source="filtered",
        confidence=confidence,
        meta={
            **track.meta,
            "filter": "spline",
            "spline_params": params,
            "source_name": track.name,
            "n_smoothed": n_smoothed,
            "n_passed_through": n_passed_through,
        },
    )
    return result


def optimize_spline(
    track: Track,
    ground_truth: Track,
    params: SplineParams | None = None,
    bounds: tuple[float, float] = (1e-6, 1e3),
) -> tuple[SplineParams, float]:
    """Find the smoothing factor minimizing the mean deviation.

    The search runs on ``log10(smoothing)`` and returns the best parameters
    together with the achieved mean Euclidean deviation in centimeters.
    """
    base = params or SplineParams()
    if base.auto_smoothing:
        # A numeric search is meaningless while the smoothing value is
        # ignored in favor of an automatically chosen one.
        base = replace(base, auto_smoothing=False)
    if ground_truth.n_valid == 0:
        raise FilterError(f"{ground_truth.name}: the ground truth has no valid samples")

    def cost(log_smoothing: float) -> float:
        candidate = base.with_smoothing(10.0**log_smoothing)
        try:
            filtered = apply_spline_filter(track, candidate)
        except (FilterError, ValueError):
            return np.inf
        stats = deviation_stats(filtered, ground_truth)
        if stats.n_samples == 0 or not np.isfinite(stats.euclidean_mean):
            return np.inf
        return stats.euclidean_mean

    result = minimize_scalar(
        cost,
        bounds=(np.log10(bounds[0]), np.log10(bounds[1])),
        method="bounded",
        options={"xatol": 1e-3},
    )

    best = base.with_smoothing(10.0**result.x)
    best_cost = float(result.fun)

    baseline = cost(np.log10(base.smoothing)) if base.smoothing > 0 else np.inf
    if baseline <= best_cost:
        return base, float(baseline)
    return best, best_cost
