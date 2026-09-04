"""Loader for Optitrack marker exports.

The export starts with a single key/value metadata line, a blank line and
four descriptor rows (``Type``, ``Name``, ``ID`` and the measurement type),
followed by the ``Frame,Time (Seconds),TimeCode`` data header and the frames.

Every marker contributes one consecutive ``X,Y,Z`` column triplet. By
convention the first three triplets are the markers rigidly mounted on the
(stationary) Oak-D camera; every further triplet is an LED observation
fragment, because Optitrack starts a new unlabeled marker whenever tracking
is lost and re-acquired.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..model import Recording, Track

N_CAMERA_MARKERS = 3
DATA_HEADER_FIRST_CELL = "Frame"
UNIT_TO_CM = {
    "millimeters": 0.1,
    "centimeters": 1.0,
    "meters": 100.0,
    "inches": 2.54,
}

CAMERA_MOTION_TOL_CM = 1.0
LED_MOTION_MIN_CM = 1.0


@dataclass
class _Layout:
    metadata: dict[str, str]
    header_row: int
    marker_names: list[str]


def parse_metadata(line: str | list[str]) -> dict[str, str]:
    """Parse the leading comma separated key/value metadata line."""
    cells = next(csv.reader([line])) if isinstance(line, str) else list(line)
    return {
        cells[i].strip(): cells[i + 1].strip()
        for i in range(0, len(cells) - 1, 2)
        if cells[i].strip()
    }


def _read_layout(path: Path) -> _Layout:
    metadata: dict[str, str] = {}
    name_row: list[str] = []
    header_row = -1
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for index, cells in enumerate(csv.reader(handle)):
            if index == 0:
                metadata = parse_metadata(cells)
                continue
            if not cells:
                continue
            label = cells[1].strip() if len(cells) > 1 else ""
            if not cells[0].strip() and label == "Name":
                name_row = [c.strip() for c in cells]
            if cells[0].strip() == DATA_HEADER_FIRST_CELL:
                header_row = index
                break
    if header_row < 0:
        raise ValueError(f"{path.name}: no '{DATA_HEADER_FIRST_CELL},...' header row found")
    if not name_row:
        raise ValueError(f"{path.name}: no 'Name' descriptor row found")
    marker_names = [n for n in name_row[3:][::3] if n]
    return _Layout(metadata=metadata, header_row=header_row, marker_names=marker_names)

def unit_scale_to_cm(units: str | None) -> float:
    if not units:
        return 1.0
    try:
        return UNIT_TO_CM[units.strip().lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported length unit {units!r}") from exc


def merge_led_fragments(
    fragments: list[np.ndarray], names: list[str] | None = None
) -> tuple[np.ndarray, dict]:
    """Merge LED fragment triplets into a single trajectory.

    At most one fragment is expected to be populated per frame. When several
    are populated the one nearest to the previous valid position wins and the
    conflict is counted.
    """
    if not fragments:
        raise ValueError("no LED fragments to merge")

    stack = np.stack(fragments, axis=0)
    n_frames = stack.shape[1]
    valid = np.all(np.isfinite(stack), axis=2)

    merged = np.full((n_frames, 3), np.nan)
    chosen = np.full(n_frames, -1, dtype=int)
    conflicts = 0
    previous: np.ndarray | None = None

    for frame in range(n_frames):
        candidates = np.flatnonzero(valid[:, frame])
        if candidates.size == 0:
            continue
        if candidates.size == 1:
            pick = int(candidates[0])
        else:
            conflicts += 1
            points = stack[candidates, frame]
            if previous is None:
                pick = int(candidates[0])
            else:
                distances = np.linalg.norm(points - previous, axis=1)
                pick = int(candidates[int(np.argmin(distances))])
        merged[frame] = stack[pick, frame]
        chosen[frame] = pick
        previous = merged[frame]

    meta = {
        "n_fragments": len(fragments),
        "fragment_names": list(names) if names else [],
        "merge_conflicts": conflicts,
        "fragment_index": chosen,
    }
    return merged, meta


def load_optitrack(path: str | Path, name: str | None = None) -> Recording:
    """Load an Optitrack CSV export into a :class:`Recording`."""
    path = Path(path)
    layout = _read_layout(path)

    frame = pd.read_csv(
        path,
        skiprows=layout.header_row,
        header=0,
        encoding="utf-8-sig",
        low_memory=False,
    )
    if frame.shape[1] < 3 + 3 * N_CAMERA_MARKERS:
        raise ValueError(
            f"{path.name}: expected at least {N_CAMERA_MARKERS} marker triplets, "
            f"got {(frame.shape[1] - 3) // 3}"
        )

    data = frame.iloc[:, 3:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n_triplets = data.shape[1] // 3
    if n_triplets <= N_CAMERA_MARKERS:
        raise ValueError(
            f"{path.name}: no LED marker columns found "
            f"(only {n_triplets} triplets, the first {N_CAMERA_MARKERS} are the camera markers)"
        )
    triplets = [data[:, 3 * i : 3 * i + 3] for i in range(n_triplets)]

    scale = unit_scale_to_cm(layout.metadata.get("Length Units"))
    triplets = [tri * scale for tri in triplets]

    marker_triplets = triplets[:N_CAMERA_MARKERS]
    led_triplets = triplets[N_CAMERA_MARKERS:]
    led_names = layout.marker_names[N_CAMERA_MARKERS:]

    merged, merge_meta = merge_led_fragments(led_triplets, led_names)

    t = frame.iloc[:, 1].to_numpy(dtype=float)
    frame_numbers = frame.iloc[:, 0].to_numpy(dtype=float)

    camera_markers = np.stack([np.nanmean(tri, axis=0) for tri in marker_triplets])
    marker_spread = np.stack(
        [_ptp_nan(tri) for tri in marker_triplets]
    )

    meta = {
        "raw_time_base": "Time (Seconds)",
        "frames": frame_numbers,
        "length_units": layout.metadata.get("Length Units"),
        "unit_scale_to_cm": scale,
        "capture_start_time": layout.metadata.get("Capture Start Time"),
        "marker_names": layout.marker_names,
        "marker_spread": marker_spread,
        **merge_meta,
    }

    track = Track(
        t=t,
        xyz=merged,
        name=name or _short_name(path),
        source="optitrack",
        meta=meta,
    )
    recording = Recording(
        led=track,
        path=path,
        camera_markers=camera_markers,
        frame_rate=_frame_rate(layout.metadata),
        meta=meta,
    )
    recording.warnings = validate_recording(recording)
    return recording


def validate_recording(
    recording: Recording,
    camera_motion_tol: float = CAMERA_MOTION_TOL_CM,
    led_motion_min: float = LED_MOTION_MIN_CM,
) -> list[str]:
    """Sanity check: camera markers stationary, LED moving.

    Returns a list of human readable warnings; empty when everything is fine.
    """
    warnings: list[str] = []
    label = recording.path.name

    spread = recording.meta.get("marker_spread")
    if spread is not None:
        spread = np.asarray(spread, dtype=float)
        for index, marker in enumerate(spread):
            worst = float(np.nanmax(marker)) if np.any(np.isfinite(marker)) else np.nan
            if not np.isfinite(worst):
                warnings.append(f"{label}: camera marker {index + 1} has no valid samples")
            elif worst > camera_motion_tol:
                warnings.append(
                    f"{label}: camera marker {index + 1} moved by {worst:.2f} cm "
                    f"(tolerance {camera_motion_tol:.2f} cm) - was the camera stationary?"
                )

    led_spread = recording.led.max_spread()
    if not np.isfinite(led_spread):
        warnings.append(f"{label}: the LED track has no valid samples")
    elif led_spread < led_motion_min:
        warnings.append(
            f"{label}: the LED moved only {led_spread:.2f} cm "
            f"(expected more than {led_motion_min:.2f} cm) - are the columns swapped?"
        )
    return warnings


def _ptp_nan(values: np.ndarray) -> np.ndarray:
    if not np.any(np.all(np.isfinite(values), axis=1)):
        return np.full(3, np.nan)
    return np.nanmax(values, axis=0) - np.nanmin(values, axis=0)


def _frame_rate(metadata: dict[str, str]) -> float | None:
    raw = metadata.get("Capture Frame Rate")
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


def _short_name(path: Path) -> str:
    return path.stem
