"""Deviation analysis between an analyzed recording and its ground truth."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.optimize import least_squares

from .model import Track

TransformModel = Literal["rigid", "similarity", "affine", "projective"]
TRANSFORM_MODELS: tuple[TransformModel, ...] = (
    "rigid",
    "similarity",
    "affine",
    "projective",
)

MAX_GAP_FACTOR = 3.0
PLANARITY_TOL = 1e-3


class AnalysisError(ValueError):
    """Raised when two tracks cannot be compared."""


@dataclass
class DeviationStats:
    """Deviation of an analyzed track from its ground truth."""

    n_samples: int
    n_excluded: int
    axis_mean: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    axis_std: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    euclidean_mean: float = float("nan")
    euclidean_std: float = float("nan")
    rmse: float = float("nan")
    median: float = float("nan")
    p95: float = float("nan")

    def as_rows(self) -> list[tuple[str, str]]:
        """Human readable ``(label, value)`` rows for display in a table."""
        axes = "XYZ"
        rows = [("Samples", f"{self.n_samples}"), ("Excluded", f"{self.n_excluded}")]
        for index, axis in enumerate(axes):
            rows.append(
                (
                    f"Mean / std {axis} [cm]",
                    f"{self.axis_mean[index]:+.3f} / {self.axis_std[index]:.3f}",
                )
            )
        rows += [
            ("Mean euclidean [cm]", f"{self.euclidean_mean:.3f}"),
            ("Std euclidean [cm]", f"{self.euclidean_std:.3f}"),
            ("RMSE [cm]", f"{self.rmse:.3f}"),
            ("Median [cm]", f"{self.median:.3f}"),
            ("95th percentile [cm]", f"{self.p95:.3f}"),
        ]
        return rows


@dataclass
class ConsistencyReport:
    """Before/after deviation for the single best-fit transform."""

    model: TransformModel
    before: DeviationStats
    after: DeviationStats
    transform: np.ndarray
    warnings: list[str] = field(default_factory=list)

    @property
    def improvement(self) -> float:
        """Reduction of the mean Euclidean deviation in centimeters."""
        return float(self.before.euclidean_mean - self.after.euclidean_mean)


def resample_to(reference: Track, times: np.ndarray, max_gap: float | None = None) -> np.ndarray:
    """Linearly interpolate ``reference`` onto ``times``.

    Samples that fall outside the reference range or inside a gap longer
    than ``max_gap`` seconds become NaN, so gaps are never bridged.
    """
    times = np.asarray(times, dtype=float)
    ref_t, ref_xyz = reference.valid()
    if ref_t.size < 2:
        return np.full((times.size, 3), np.nan)

    order = np.argsort(ref_t, kind="stable")
    ref_t, ref_xyz = ref_t[order], ref_xyz[order]

    if max_gap is None:
        step = np.median(np.diff(ref_t))
        max_gap = MAX_GAP_FACTOR * step if np.isfinite(step) and step > 0 else np.inf

    out = np.empty((times.size, 3))
    for axis in range(3):
        out[:, axis] = np.interp(times, ref_t, ref_xyz[:, axis], left=np.nan, right=np.nan)

    right = np.searchsorted(ref_t, times, side="left").clip(1, ref_t.size - 1)
    spanned = ref_t[right] - ref_t[right - 1]
    out[spanned > max_gap] = np.nan
    return out


def paired_points(
    track: Track, ground_truth: Track, max_gap: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return ``(times, analyzed, truth, n_excluded)`` on the common grid."""
    resampled = resample_to(ground_truth, track.t, max_gap)
    usable = track.valid_mask & np.all(np.isfinite(resampled), axis=1)
    n_excluded = int(len(track) - np.count_nonzero(usable))
    return track.t[usable], track.xyz[usable], resampled[usable], n_excluded


def deviation_stats(
    track: Track, ground_truth: Track, max_gap: float | None = None
) -> DeviationStats:
    """Statistics of the deviation of ``track`` from ``ground_truth``."""
    _, analyzed, truth, n_excluded = paired_points(track, ground_truth, max_gap)
    return stats_from_points(analyzed, truth, n_excluded)


def stats_from_points(
    analyzed: np.ndarray, truth: np.ndarray, n_excluded: int = 0
) -> DeviationStats:
    analyzed = np.asarray(analyzed, dtype=float).reshape(-1, 3)
    truth = np.asarray(truth, dtype=float).reshape(-1, 3)
    if analyzed.shape[0] == 0:
        return DeviationStats(n_samples=0, n_excluded=n_excluded)

    difference = analyzed - truth
    distance = np.linalg.norm(difference, axis=1)
    return DeviationStats(
        n_samples=int(analyzed.shape[0]),
        n_excluded=n_excluded,
        axis_mean=difference.mean(axis=0),
        axis_std=difference.std(axis=0),
        euclidean_mean=float(distance.mean()),
        euclidean_std=float(distance.std()),
        rmse=float(np.sqrt(np.mean(distance**2))),
        median=float(np.median(distance)),
        p95=float(np.percentile(distance, 95)),
    )


@dataclass
class SmoothnessStats:
    """How calm a trajectory is, next to how calm it ought to be.

    Deviation statistics cannot see this: a predictor that twitches around
    the right answer scores well on mean error yet is unusable on a display.
    The ratios compare against a reference trajectory, so ``1.0`` means the
    output moves exactly as much as the real motion does. Above ``1`` the
    predictor is inventing motion, below ``1`` it is smoothing real motion
    away.
    """

    n_samples: int
    dt: float = float("nan")
    jerk_rms: float = float("nan")
    jerk_ratio: float = float("nan")
    hf_energy: float = float("nan")
    hf_ratio: float = float("nan")
    step_p99: float = float("nan")
    step_ratio: float = float("nan")

    def as_rows(self) -> list[tuple[str, str]]:
        """Human readable ``(label, value)`` rows for display in a table."""
        return [
            ("Samples", f"{self.n_samples}"),
            ("Jerk RMS [cm/s^3]", f"{self.jerk_rms:.1f}"),
            ("Jerk ratio", f"{self.jerk_ratio:.2f}"),
            ("High frequency ratio", f"{self.hf_ratio:.2f}"),
            ("99th pct step [cm]", f"{self.step_p99:.4f}"),
            ("Step ratio", f"{self.step_ratio:.2f}"),
        ]


def _finite_runs(mask: np.ndarray) -> list[np.ndarray]:
    """Index arrays of the contiguous ``True`` runs of ``mask``."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) != 1)
    return [part for part in np.split(idx, breaks + 1) if part.size]


def _uniform_step(times: np.ndarray, tolerance: float = 0.1) -> float:
    steps = np.diff(times)
    if steps.size == 0:
        raise AnalysisError("a smoothness metric needs at least two samples")
    step = float(np.median(steps))
    if not np.isfinite(step) or step <= 0:
        raise AnalysisError("the time axis is not increasing")
    if np.max(np.abs(steps - step)) > tolerance * step:
        raise AnalysisError(
            "smoothness metrics differentiate the output stream and so need a "
            "uniform time axis"
        )
    return step


def _jerk_rms(xyz: np.ndarray, runs: list[np.ndarray], dt: float) -> float:
    pooled = [np.diff(xyz[run], n=3, axis=0) for run in runs if run.size >= 4]
    pooled = [d for d in pooled if d.size]
    if not pooled:
        return float("nan")
    stacked = np.vstack(pooled) / dt**3
    return float(np.sqrt(np.mean(np.sum(stacked**2, axis=1))))


def _step_p99(xyz: np.ndarray, runs: list[np.ndarray]) -> float:
    pooled = [np.diff(xyz[run], axis=0) for run in runs if run.size >= 2]
    pooled = [d for d in pooled if d.size]
    if not pooled:
        return float("nan")
    return float(np.percentile(np.linalg.norm(np.vstack(pooled), axis=1), 99))


def _hf_energy(xyz: np.ndarray, runs: list[np.ndarray], dt: float, cutoff: float) -> float:
    """Signal power above ``cutoff`` Hz, pooled over the usable runs."""
    longest = max((run for run in runs), key=len, default=None)
    if longest is None or longest.size < 8:
        return float("nan")
    values = xyz[longest]
    centered = values - values.mean(axis=0)
    power = (np.abs(np.fft.rfft(centered, axis=0)) ** 2).sum(axis=1)
    band = np.fft.rfftfreq(longest.size, dt) >= cutoff
    if not np.any(band):
        return float("nan")
    return float(power[band].sum() / longest.size**2)


def smoothness_stats(
    track: Track,
    reference: Track | None = None,
    hf_cutoff: float = 8.0,
    max_gap: float | None = None,
) -> SmoothnessStats:
    """Smoothness of ``track``, optionally relative to ``reference``.

    ``hf_cutoff`` sits above the band real hand motion occupies, so power
    beyond it is noise the predictor added rather than motion it tracked.
    """
    dt = _uniform_step(track.t)
    mask = track.valid_mask
    truth = None
    if reference is not None:
        truth = resample_to(reference, track.t, max_gap)
        mask = mask & np.all(np.isfinite(truth), axis=1)

    runs = _finite_runs(mask)
    n_samples = int(sum(run.size for run in runs))
    if n_samples == 0:
        return SmoothnessStats(n_samples=0, dt=dt)

    jerk = _jerk_rms(track.xyz, runs, dt)
    hf = _hf_energy(track.xyz, runs, dt, hf_cutoff)
    step = _step_p99(track.xyz, runs)

    stats = SmoothnessStats(
        n_samples=n_samples, dt=dt, jerk_rms=jerk, hf_energy=hf, step_p99=step
    )
    if truth is None:
        return stats

    reference_jerk = _jerk_rms(truth, runs, dt)
    reference_hf = _hf_energy(truth, runs, dt, hf_cutoff)
    reference_step = _step_p99(truth, runs)
    stats.jerk_ratio = jerk / reference_jerk if reference_jerk > 0 else float("nan")
    stats.hf_ratio = hf / reference_hf if reference_hf > 0 else float("nan")
    stats.step_ratio = step / reference_step if reference_step > 0 else float("nan")
    return stats


def apply_transform(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a homogeneous 4x4 transform to ``(N, 3)`` points."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    projected = homogeneous @ np.asarray(transform, dtype=float).T
    weight = projected[:, 3:4]
    weight = np.where(np.abs(weight) < 1e-12, np.nan, weight)
    return projected[:, :3] / weight


def _homogeneous(linear: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = linear
    transform[:3, 3] = translation
    return transform


def _fit_kabsch(source: np.ndarray, target: np.ndarray, allow_scale: bool) -> np.ndarray:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    a = source - source_mean
    b = target - target_mean

    u, singular, vt = np.linalg.svd(a.T @ b)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ correction @ u.T

    scale = 1.0
    if allow_scale:
        variance = float(np.sum(a**2))
        if variance > 0:
            scale = float(np.sum(singular * np.diag(correction)) / variance)
    linear = scale * rotation
    return _homogeneous(linear, target_mean - linear @ source_mean)


def _fit_affine(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = np.hstack([source, np.ones((source.shape[0], 1))])
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    return _homogeneous(solution[:3].T, solution[3])


def _planarity(points: np.ndarray) -> float:
    """Smallest relative singular value of the centered point cloud."""
    centered = points - points.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    if singular[0] <= 0:
        return 0.0
    return float(singular[-1] / singular[0])


def _fit_projective(
    source: np.ndarray, target: np.ndarray, warnings: list[str]
) -> np.ndarray:
    if _planarity(source) < PLANARITY_TOL:
        warnings.append(
            "The trajectory is nearly planar, the projective fit is ill-conditioned; "
            "falling back to the affine model."
        )
        return _fit_affine(source, target)

    initial = _fit_affine(source, target)

    def residual(parameters: np.ndarray) -> np.ndarray:
        transform = np.eye(4)
        transform.flat[:15] = parameters
        predicted = apply_transform(transform, source)
        difference = predicted - target
        return np.where(np.isfinite(difference), difference, 1e6).ravel()

    result = least_squares(residual, initial.ravel()[:15], method="lm", max_nfev=2000)
    transform = np.eye(4)
    transform.flat[:15] = result.x

    if not np.all(np.isfinite(apply_transform(transform, source))):
        warnings.append("The projective fit degenerated; falling back to the affine model.")
        return initial
    return transform


def best_fit_transform(
    source: np.ndarray,
    target: np.ndarray,
    model: TransformModel = "rigid",
    warnings: list[str] | None = None,
) -> np.ndarray:
    """Single 4x4 transform mapping ``source`` points onto ``target``."""
    source = np.asarray(source, dtype=float).reshape(-1, 3)
    target = np.asarray(target, dtype=float).reshape(-1, 3)
    if source.shape != target.shape:
        raise AnalysisError("source and target must have the same shape")
    if source.shape[0] < 4:
        raise AnalysisError(
            f"at least 4 point pairs are needed to fit a transform, got {source.shape[0]}"
        )
    warnings = warnings if warnings is not None else []

    if model == "rigid":
        return _fit_kabsch(source, target, allow_scale=False)
    if model == "similarity":
        return _fit_kabsch(source, target, allow_scale=True)
    if model == "affine":
        return _fit_affine(source, target)
    if model == "projective":
        return _fit_projective(source, target, warnings)
    raise AnalysisError(f"unknown transform model {model!r}")


def consistency_report(
    track: Track,
    ground_truth: Track,
    model: TransformModel = "rigid",
    max_gap: float | None = None,
) -> ConsistencyReport:
    """Deviation before and after the best single transform of ``track``.

    The residual after fitting is the frame-independent tracking error; the
    part removed by the transform is systematic frame misalignment.
    """
    _, analyzed, truth, n_excluded = paired_points(track, ground_truth, max_gap)
    before = stats_from_points(analyzed, truth, n_excluded)

    if analyzed.shape[0] < 4:
        return ConsistencyReport(
            model=model,
            before=before,
            after=before,
            transform=np.eye(4),
            warnings=[
                f"Only {analyzed.shape[0]} comparable samples, too few to fit a transform."
            ],
        )

    warnings: list[str] = []
    transform = best_fit_transform(analyzed, truth, model, warnings)
    corrected = apply_transform(transform, analyzed)
    after = stats_from_points(corrected, truth, n_excluded)

    if not np.isfinite(after.euclidean_mean) or after.euclidean_mean > before.euclidean_mean:
        warnings.append("The fitted transform did not improve the deviation; keeping identity.")
        transform = np.eye(4)
        after = before

    return ConsistencyReport(
        model=model, before=before, after=after, transform=transform, warnings=warnings
    )


def transformed_track(track: Track, transform: np.ndarray, suffix: str = "corrected") -> Track:
    """Return a copy of ``track`` with ``transform`` applied to every point."""
    result = track.copy()
    result.xyz = apply_transform(transform, track.xyz)
    result.name = f"{track.name} ({suffix})"
    result.meta["transform"] = np.asarray(transform, dtype=float)
    return result
