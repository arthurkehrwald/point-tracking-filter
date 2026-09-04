"""Loader for Luxonis Oak-D LED recordings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..model import Recording, Track

REQUIRED_COLUMNS = ("x", "y", "z", "capture_time")


def load_oak_d(path: str | Path, name: str | None = None) -> Recording:
    """Load an Oak-D CSV recording.

    The file has the header ``x,y,z,capture_time,detection_time,confidence``
    with one LED observation per row. ``capture_time`` is an absolute
    monotonic clock in seconds and is used as the time base.
    """
    path = Path(path)
    frame = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path.name}: missing column(s) {', '.join(missing)}")

    frame = frame.sort_values("capture_time", kind="stable").reset_index(drop=True)

    t = frame["capture_time"].to_numpy(dtype=float)
    xyz = frame[["x", "y", "z"]].to_numpy(dtype=float)
    confidence = (
        frame["confidence"].to_numpy(dtype=float) if "confidence" in frame else None
    )

    meta: dict = {"raw_time_base": "capture_time"}
    if "detection_time" in frame:
        meta["detection_time"] = frame["detection_time"].to_numpy(dtype=float)

    track = Track(
        t=t,
        xyz=xyz,
        name=name or path.stem,
        source="oak-d",
        confidence=confidence,
        meta=meta,
    )
    return Recording(
        led=track,
        path=path,
        frame_rate=_estimate_frame_rate(t),
        meta={"length_units": "Centimeters"},
    )


def _estimate_frame_rate(t: np.ndarray) -> float | None:
    if t.size < 2:
        return None
    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        return None
    return float(1.0 / dt)
