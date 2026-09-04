"""Time synchronization between the two camera systems.

The tracking LED was switched on while both systems were already recording,
so the first appearance of the LED is the same instant in both files. Every
recording therefore gets a normalized time axis with ``t = 0`` at LED onset.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .model import Recording, Track

MIN_ONSET_RUN = 3


class SyncError(ValueError):
    """Raised when a recording cannot be synchronized."""


def led_onset_index(track: Track, min_run: int = MIN_ONSET_RUN) -> int:
    """Index of the first valid sample of the first sufficiently long run.

    Requiring ``min_run`` consecutive valid samples guards against a single
    spurious detection before the LED was actually switched on.
    """
    mask = track.valid_mask
    if not np.any(mask):
        raise SyncError(f"{track.name}: the LED track has no valid samples")

    run = 0
    for index, valid in enumerate(mask):
        run = run + 1 if valid else 0
        if run >= min_run:
            return index - min_run + 1
    return int(np.flatnonzero(mask)[0])


def led_onset_time(track: Track, min_run: int = MIN_ONSET_RUN) -> float:
    return float(track.t[led_onset_index(track, min_run)])


def normalize_to_led_onset(track: Track, min_run: int = MIN_ONSET_RUN) -> Track:
    """Return a copy of ``track`` whose LED onset sits at ``t = 0``."""
    onset = led_onset_time(track, min_run)
    normalized = track.shifted(-onset)
    normalized.meta = dict(track.meta)
    normalized.meta["led_onset_raw_time"] = onset
    return normalized


def normalize_recording(recording: Recording, min_run: int = MIN_ONSET_RUN) -> Recording:
    recording.led = normalize_to_led_onset(recording.led, min_run)
    return recording


def recording_token(path: str | Path) -> str:
    """Leading name token of a recording file, e.g. ``slow5`` or ``sweep``."""
    stem = Path(path).stem
    match = re.match(r"[A-Za-z]+\d*", stem)
    return (match.group(0) if match else stem).lower()


@dataclass
class RecordingPair:
    """An analyzed recording together with its ground truth counterpart."""

    analyzed: Recording
    ground_truth: Recording
    token: str = ""
    offset: float = 0.0
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.token:
            self.token = recording_token(self.analyzed.path)

    @property
    def name(self) -> str:
        return f"{self.analyzed.name} vs {self.ground_truth.name}"

    def analyzed_track(self) -> Track:
        """The analyzed track with the user offset applied."""
        return self.analyzed.led.shifted(self.offset)


def pair_recordings(
    analyzed: list[Recording], ground_truth: list[Recording]
) -> list[RecordingPair]:
    """Pair recordings from both systems by their leading filename token."""
    by_token: dict[str, Recording] = {}
    for recording in ground_truth:
        by_token.setdefault(recording_token(recording.path), recording)

    pairs = []
    for recording in analyzed:
        token = recording_token(recording.path)
        counterpart = by_token.get(token)
        if counterpart is not None:
            pairs.append(
                RecordingPair(analyzed=recording, ground_truth=counterpart, token=token)
            )
    return pairs


def pair_paths(
    analyzed_dir: str | Path, ground_truth_dir: str | Path
) -> list[tuple[str, Path, Path]]:
    """Match CSV files of two directories by leading filename token."""
    truth = {}
    for path in sorted(Path(ground_truth_dir).glob("*.csv")):
        truth.setdefault(recording_token(path), path)

    matches = []
    for path in sorted(Path(analyzed_dir).glob("*.csv")):
        token = recording_token(path)
        if token in truth:
            matches.append((token, path, truth[token]))
    return matches
