"""Regenerate every number, table and figure of the paper's evaluation.

Run with ``uv run python scripts/evaluate.py``. Tables are written to
``paper/tables`` and figures to ``paper/assets``; the values quoted in the
running text are printed to stdout.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from point_tracking_filter.core.align import align_recording, load_extrinsic
from point_tracking_filter.core.analysis import (
    consistency_report,
    deviation_stats,
    resample_to,
    smoothness_stats,
    transformed_track,
)
from point_tracking_filter.core.filtering import SplineParams, apply_spline_filter
from point_tracking_filter.core.io.oak_d import load_oak_d
from point_tracking_filter.core.io.optitrack import load_optitrack
from point_tracking_filter.core.model import Track
from point_tracking_filter.core.prediction import (
    ConstantVelocityKalman,
    StreamConfig,
    WindowedSplineRefit,
    ZeroOrderHold,
    sample_latency,
    simulate,
    tune,
)
from point_tracking_filter.core.sync import normalize_to_led_onset, pair_paths

ROOT = Path(__file__).resolve().parents[1]
RECORDINGS = ROOT / "recordings"
TABLES = ROOT / "paper" / "tables"
ASSETS = ROOT / "paper" / "assets"

# fast1 is excluded because its tracks cannot be aligned (see the paper).
USED = ["slow2", "slow3", "slow4", "slow5", "fast2", "fast3", "fast4"]
EXCLUDED = ["fast1"]
HORIZONS = [0.0, 0.05, 0.1]
CURVE_HORIZONS = [0.0, 0.025, 0.05, 0.075, 0.1]
TUNING_HORIZON = 0.05
SPLINE_WINDOW = 0.3
SIGMA_M = 0.1

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
COLORS = {
    "truth": "#555555",
    "raw": "#9ecae1",
    "gcv": "#08519c",
    "hold": "#999999",
    "kalman": "#d95f02",
    "spline": "#1b9e77",
}


@dataclass
class Case:
    token: str
    regime: str
    raw: Track
    truth: Track
    alignment_rmse: float


def load_case(token: str, oak_path: Path, opti_path: Path, extrinsic) -> Case:
    oak = load_oak_d(oak_path, name=f"{token} oak-d")
    oak.led = normalize_to_led_onset(oak.led)
    opti = load_optitrack(opti_path, name=f"{token} optitrack")
    align_recording(opti, extrinsic)
    opti.led = normalize_to_led_onset(opti.led)

    report = consistency_report(oak.led, opti.led)
    raw = transformed_track(oak.led, report.transform, suffix="aligned")
    return Case(
        token=token,
        regime=token.rstrip("0123456789"),
        raw=raw,
        truth=opti.led,
        alignment_rmse=report.after.rmse,
    )


def speed_stats(track: Track) -> np.ndarray:
    speeds = []
    for indices in track.segments():
        if indices.size < 2:
            continue
        xyz = track.xyz[indices]
        speeds.append(np.linalg.norm(np.diff(xyz, axis=0), axis=1) / np.diff(track.t[indices]))
    return np.concatenate(speeds)


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def write_table(name: str, header: list[str], rows: list[list[str]], align: str) -> None:
    lines = [
        f"\\begin{{tabular}}{{{align}}}",
        "\\hline",
        " & ".join(header) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        if row == ["\\hline"]:
            lines.append("\\hline")
        else:
            lines.append(" & ".join(row) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", ""]
    (TABLES / f"{name}.tex").write_text("\n".join(lines))


# ---------------------------------------------------------------- filtering


def evaluate_filtering(cases: list[Case]) -> dict:
    rows, numbers = [], {}
    previous = None
    for case in cases:
        if previous is not None and case.regime != previous:
            rows.append(["\\hline"])
        previous = case.regime
        smoothed = apply_spline_filter(case.raw, SplineParams(auto_smoothing=True))
        before = deviation_stats(case.raw, case.truth)
        after = deviation_stats(smoothed, case.truth)
        residual = (case.raw.xyz - smoothed.xyz)[case.raw.valid_mask]
        white = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
        numbers[case.token] = {
            "raw": before.__dict__ | {"axis_std": before.axis_std.tolist(), "axis_mean": before.axis_mean.tolist()},
            "gcv": after.__dict__ | {"axis_std": after.axis_std.tolist(), "axis_mean": after.axis_mean.tolist()},
            "removed_rms": white,
            "mean_depth": float(np.nanmean(case.raw.xyz[:, 2])),
            **gcv_spectrum(case),
        }
        rows.append(
            [
                case.token,
                f"{fmt(before.rmse)} / {fmt(after.rmse)}",
                f"{fmt(before.p95)} / {fmt(after.p95)}",
                f"{fmt(before.p99)} / {fmt(after.p99)}",
                " / ".join(fmt(v) for v in before.axis_std),
                fmt(white, 3),
            ]
        )
    write_table(
        "filtering",
        ["Rec.", "RMSE", "P95", "P99", "Std.\\ $x$ / $y$ / $z$ (raw)", "$\\hat\\sigma$"],
        rows,
        "lccccc",
    )
    return numbers


def gcv_spectrum(case: Case) -> dict:
    """GCV parameters per axis and how much raw error lies below their cutoff.

    SciPy does not return the λ it picked, so the GCV routine is wrapped for
    the duration of one filter run. The cutoff follows from the transfer
    function 1 / (1 + λΔ(2πf)^4) of the spline on a uniform grid.
    """
    import scipy.interpolate._bsplines as bsplines

    original = bsplines._compute_optimal_gcv_parameter
    picked: list[tuple[int, np.ndarray]] = []

    def spy(X, wE, y, w):
        lam = original(X, wE, y, w)
        picked.append((y.shape[0], np.atleast_1d(lam)))
        return lam

    bsplines._compute_optimal_gcv_parameter = spy
    try:
        apply_spline_filter(case.raw, SplineParams(auto_smoothing=True))
    finally:
        bsplines._compute_optimal_gcv_parameter = original

    dt = float(np.median(np.diff(case.raw.t)))
    counts = np.array([n for n, _ in picked], dtype=float)
    logs = np.log(np.array([lam for _, lam in picked]))
    lam = np.exp((logs * counts[:, None]).sum(axis=0) / counts.sum())
    cutoff = (lam * dt) ** -0.25 / (2 * np.pi)

    error = case.raw.xyz - resample_to(case.truth, case.raw.t)
    usable = np.flatnonzero(np.all(np.isfinite(error), axis=1))
    runs = np.split(usable, np.flatnonzero(np.diff(usable) != 1) + 1)
    longest = max(runs, key=len)
    below = []
    for axis in range(3):
        values = error[longest, axis] - error[longest, axis].mean()
        power = np.abs(np.fft.rfft(values)) ** 2
        freq = np.fft.rfftfreq(longest.size, dt)
        below.append(float(power[1:][freq[1:] < cutoff[axis]].sum() / power[1:].sum()))
    return {"lambda": lam.tolist(), "cutoff_hz": cutoff.tolist(), "error_power_below_cutoff": below}


def figure_filtering(cases: list[Case]) -> None:
    """Depth error against the ground truth, before and after smoothing.

    The coordinates themselves are useless here: at the scale of the motion
    raw, smoothed and true depth lie on top of each other.
    """
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.2), constrained_layout=True)
    for ax, token in zip(axes, ["slow3", "fast4"]):
        case = next(c for c in cases if c.token == token)
        smoothed = apply_spline_filter(case.raw, SplineParams(auto_smoothing=True))
        truth = resample_to(case.truth, case.raw.t)
        error_raw = case.raw.xyz[:, 2] - truth[:, 2]
        error_gcv = smoothed.xyz[:, 2] - truth[:, 2]
        span = 6.0
        finite = np.isfinite(error_raw)
        # Centre the window on the largest raw error so outliers, if any, show.
        centre = case.raw.t[np.nanargmax(np.where(finite, np.abs(error_raw), -np.inf))]
        start = max(float(case.raw.t[finite][0]), centre - span / 2)
        mask = (case.raw.t >= start) & (case.raw.t <= start + span)
        t = case.raw.t[mask] - start
        ax.plot(t, error_raw[mask], color=COLORS["raw"], lw=0.8, label="OAK-D raw")
        ax.plot(t, error_gcv[mask], color=COLORS["gcv"], lw=1.0, label="GCV spline")
        ax.axhline(0.0, color=COLORS["truth"], lw=0.5)
        limit = 1.1 * np.nanmax(np.abs(error_gcv[mask]))
        ax.set_ylim(-max(limit, 1.0), max(limit, 1.0))
        ax.set_title(f"{token}: depth error")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("$\\hat z - z$ [cm]")
    axes[0].legend(frameon=False)
    fig.savefig(ASSETS / "filtering-excerpt.pdf")
    plt.close(fig)


def busiest_window(track: Track, span: float) -> float:
    """Start time of the window with the largest depth range."""
    valid = track.valid_mask
    t, z = track.t[valid], track.xyz[valid, 2]
    best, best_start = -1.0, float(t[0])
    for start in np.arange(t[0], t[-1] - span, span / 4):
        mask = (t >= start) & (t <= start + span)
        if np.count_nonzero(mask) < 10:
            continue
        spread = float(np.ptp(z[mask]))
        if spread > best:
            best, best_start = spread, float(start)
    return best_start


# --------------------------------------------------------------- prediction


def predictors(sigma_a: float, lam: float) -> dict:
    return {
        "hold": lambda: ZeroOrderHold(),
        "kalman": lambda: ConstantVelocityKalman(sigma_a=sigma_a, sigma_m=SIGMA_M),
        "spline": lambda: WindowedSplineRefit(window_seconds=SPLINE_WINDOW, smoothing=lam),
    }


def run(case: Case, predictor, horizon: float) -> dict:
    predicted = simulate(case.raw, predictor, StreamConfig(horizon=horizon))
    accuracy = deviation_stats(predicted, case.truth)
    smoothness = smoothness_stats(predicted, case.truth)
    return {
        "rmse": accuracy.rmse,
        "p95": accuracy.p95,
        "p99": accuracy.p99,
        "jerk_ratio": smoothness.jerk_ratio,
        "step_ratio": smoothness.step_ratio,
    }


def tune_parameters(cases: list[Case]) -> tuple[float, float]:
    tracks = [c.raw for c in cases]
    truths = [c.truth for c in cases]
    config = StreamConfig(horizon=TUNING_HORIZON)
    kalman = tune(
        tracks,
        lambda v: ConstantVelocityKalman(sigma_a=v, sigma_m=SIGMA_M),
        (0.5, 1000.0),
        config,
        truths,
        steps=10,
    )
    spline = tune(
        tracks,
        lambda v: WindowedSplineRefit(window_seconds=SPLINE_WINDOW, smoothing=v),
        (1e-5, 10.0),
        config,
        truths,
        steps=8,
    )
    print(f"tuned sigma_a = {kalman.value:.3g}, costs {np.round(kalman.score.costs, 2)}")
    print(f"tuned lambda  = {spline.value:.3g}, costs {np.round(spline.score.costs, 2)}")
    return kalman.value, spline.value


def regime_mean(results: dict, cases: list[Case], regime: str, key: str) -> float:
    return float(np.mean([results[c.token][key] for c in cases if c.regime == regime]))


def evaluate_prediction(cases: list[Case], sigma_a: float, lam: float) -> dict:
    factories = predictors(sigma_a, lam)
    results: dict = {}
    for horizon in sorted(set(HORIZONS) | set(CURVE_HORIZONS)):
        for name, factory in factories.items():
            per_case = {c.token: run(c, factory(), horizon) for c in cases}
            results[f"{name}@{horizon * 1000:.0f}"] = per_case

    labels = {"hold": "Hold", "kalman": "Kalman", "spline": "Spline refit"}
    rows = []
    for regime in ["slow", "fast"]:
        if rows:
            rows.append(["\\hline"])
        for horizon in HORIZONS:
            for name in factories:
                data = results[f"{name}@{horizon * 1000:.0f}"]
                rows.append(
                    [
                        regime if name == "hold" and horizon == HORIZONS[0] else "",
                        f"{horizon * 1000:.0f}" if name == "hold" else "",
                        labels[name],
                        fmt(regime_mean(data, cases, regime, "rmse")),
                        fmt(regime_mean(data, cases, regime, "p95")),
                        fmt(regime_mean(data, cases, regime, "jerk_ratio"), 1),
                        fmt(regime_mean(data, cases, regime, "step_ratio")),
                    ]
                )
    write_table(
        "prediction",
        ["Regime", "$h$ [ms]", "Predictor", "RMSE", "P95", "Jerk ratio", "Step ratio"],
        rows,
        "lllcccc",
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.2), constrained_layout=True)
    for ax, regime in zip(axes, ["slow", "fast"]):
        for name in factories:
            values = [
                regime_mean(results[f"{name}@{h * 1000:.0f}"], cases, regime, "rmse")
                for h in CURVE_HORIZONS
            ]
            ax.plot(
                np.array(CURVE_HORIZONS) * 1000, values, "o-", ms=3, color=COLORS[name], label=labels[name]
            )
        ax.set_title(f"{regime} recordings")
        ax.set_xlabel("horizon $h$ [ms]")
        ax.set_ylabel("mean RMSE [cm]")
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False)
    fig.savefig(ASSETS / "prediction-horizon.pdf")
    plt.close(fig)
    return results


def sweep(cases: list[Case]) -> dict:
    config_h = TUNING_HORIZON
    grids = {
        "kalman": (np.logspace(np.log10(0.5), 3, 9), lambda v: ConstantVelocityKalman(sigma_a=v, sigma_m=SIGMA_M)),
        "spline": (np.logspace(-5, 1, 7), lambda v: WindowedSplineRefit(window_seconds=SPLINE_WINDOW, smoothing=v)),
    }
    out: dict = {}
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.4), constrained_layout=True)
    for ax, regime in zip(axes, ["slow", "fast"]):
        chosen = [c for c in cases if c.regime == regime]
        hold = [run(c, ZeroOrderHold(), config_h) for c in chosen]
        ax.plot(
            np.mean([r["jerk_ratio"] for r in hold]),
            np.mean([r["rmse"] for r in hold]),
            "s",
            color=COLORS["hold"],
            label="Hold",
        )
        for name, (grid, factory) in grids.items():
            points = []
            for value in grid:
                runs = [run(c, factory(value), config_h) for c in chosen]
                points.append(
                    (float(value), np.mean([r["jerk_ratio"] for r in runs]), np.mean([r["rmse"] for r in runs]))
                )
            out[f"{name}/{regime}"] = points
            xs, ys = [p[1] for p in points], [p[2] for p in points]
            ax.plot(xs, ys, "o-", ms=3, color=COLORS[name], label={"kalman": "Kalman ($\\sigma_a$)", "spline": "Spline refit ($\\lambda$)"}[name])
            for index in (0, len(points) - 1):
                ax.annotate(
                    f"{points[index][0]:.2g}",
                    (xs[index], ys[index]),
                    textcoords="offset points",
                    xytext=(3, 3),
                    fontsize=6,
                    color=COLORS[name],
                )
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(f"{regime} recordings, $h$ = {config_h * 1000:.0f} ms")
        ax.set_xlabel("mean jerk ratio")
        ax.set_ylabel("mean RMSE [cm]")
    axes[0].legend(frameon=False)
    fig.savefig(ASSETS / "prediction-sweep.pdf")
    plt.close(fig)
    return out


def figure_prediction_excerpt(cases: list[Case], sigma_a: float, lam: float) -> None:
    case = next(c for c in cases if c.token == "fast3")
    span = 1.0
    start = busiest_window(case.truth, span)
    fig, ax = plt.subplots(figsize=(6.0, 2.2), constrained_layout=True)
    mask = (case.truth.t >= start) & (case.truth.t <= start + span)
    ax.plot(case.truth.t[mask] - start, case.truth.xyz[mask, 2], color=COLORS["truth"], lw=1.2, label="OptiTrack")
    labels = {"hold": "Hold", "kalman": "Kalman", "spline": "Spline refit"}
    for name, factory in predictors(sigma_a, lam).items():
        predicted = simulate(case.raw, factory(), StreamConfig(horizon=TUNING_HORIZON))
        mask = (predicted.t >= start) & (predicted.t <= start + span)
        # Kalman and spline refit nearly coincide: the Kalman line is drawn wide
        # underneath and the spline thin and dashed on top, so both stay visible.
        ax.plot(
            predicted.t[mask] - start,
            predicted.xyz[mask, 2],
            "--" if name == "spline" else "-",
            color=COLORS[name],
            lw={"kalman": 3.0, "spline": 1.0}.get(name, 1.0),
            zorder={"kalman": 2, "spline": 3}.get(name, 1),
            label=labels[name],
        )
    ax.set_xlabel("target time [s]")
    ax.set_ylabel("$z$ [cm]")
    ax.set_title(f"{case.token}: depth predicted {TUNING_HORIZON * 1000:.0f} ms ahead")
    ax.legend(frameon=False, ncol=4)
    fig.savefig(ASSETS / "prediction-excerpt.pdf")
    plt.close(fig)


# --------------------------------------------------------------------- main


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    TABLES.mkdir(parents=True, exist_ok=True)
    extrinsic = load_extrinsic()
    paths = {token: (a, g) for token, a, g in pair_paths(RECORDINGS / "oak-d", RECORDINGS / "optitrack")}

    for token in EXCLUDED:
        case = load_case(token, *paths[token], extrinsic)
        print(f"excluded {token}: RMSE after LED-fitted alignment {case.alignment_rmse:.1f} cm")

    cases = [load_case(token, *paths[token], extrinsic) for token in USED]
    numbers: dict = {"cases": {}}
    for case in cases:
        speed = speed_stats(case.truth)
        latency = sample_latency(case.raw)
        numbers["cases"][case.token] = {
            "duration": case.raw.duration,
            "n_valid": case.raw.n_valid,
            "speed_median": float(np.median(speed)),
            "speed_p95": float(np.percentile(speed, 95)),
            "latency_median": float(np.median(latency)),
            "latency_max": float(np.max(latency)),
        }
    for regime in ["slow", "fast"]:
        speed = np.concatenate([speed_stats(c.truth) for c in cases if c.regime == regime])
        accel = []
        for c in cases:
            if c.regime != regime:
                continue
            for indices in c.truth.segments():
                if indices.size < 3:
                    continue
                t = c.truth.t[indices]
                v = np.diff(c.truth.xyz[indices], axis=0) / np.diff(t)[:, None]
                a = np.diff(v, axis=0) / np.diff(t)[1:, None]
                accel.append(np.linalg.norm(a, axis=1))
        numbers[f"speed_{regime}"] = [float(np.median(speed)), float(np.percentile(speed, 95))]
        # Acceleration from 120 Hz second differences is noise dominated; smooth over ~50 ms first.
        numbers[f"accel_{regime}_raw_p95"] = float(np.percentile(np.concatenate(accel), 95))

    numbers["filtering"] = evaluate_filtering(cases)
    figure_filtering(cases)

    sigma_a, lam = tune_parameters(cases)
    numbers["sigma_a"], numbers["lambda"] = sigma_a, lam
    numbers["prediction"] = evaluate_prediction(cases, sigma_a, lam)
    figure_prediction_excerpt(cases, sigma_a, lam)
    numbers["sweep"] = sweep(cases)

    (TABLES / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    print(json.dumps({k: v for k, v in numbers.items() if k not in ("prediction", "filtering", "sweep")}, indent=1))


if __name__ == "__main__":
    main()
