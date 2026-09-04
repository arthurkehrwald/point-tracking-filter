"""Shared data model for tracking recordings."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np

Source = Literal["oak-d", "optitrack", "filtered"]


@dataclass
class Track:
    """A single 3D point trajectory over time.

    ``t`` is in seconds, ``xyz`` in centimeters with NaN rows where no
    observation exists.
    """

    t: np.ndarray
    xyz: np.ndarray
    name: str
    source: Source
    confidence: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float).reshape(-1)
        self.xyz = np.asarray(self.xyz, dtype=float).reshape(-1, 3)
        if self.t.shape[0] != self.xyz.shape[0]:
            raise ValueError(
                f"t and xyz length mismatch: {self.t.shape[0]} != {self.xyz.shape[0]}"
            )
        if self.confidence is not None:
            self.confidence = np.asarray(self.confidence, dtype=float).reshape(-1)
            if self.confidence.shape[0] != self.t.shape[0]:
                raise ValueError("confidence length does not match t")

    def __len__(self) -> int:
        return int(self.t.shape[0])

    @property
    def valid_mask(self) -> np.ndarray:
        """Boolean mask of samples with a complete finite observation."""
        return np.all(np.isfinite(self.xyz), axis=1)

    @property
    def n_valid(self) -> int:
        return int(np.count_nonzero(self.valid_mask))

    @property
    def duration(self) -> float:
        if len(self) == 0:
            return 0.0
        return float(self.t[-1] - self.t[0])

    def valid(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(t, xyz)`` restricted to valid samples."""
        mask = self.valid_mask
        return self.t[mask], self.xyz[mask]

    def spread(self) -> np.ndarray:
        """Per-axis peak-to-peak extent of the valid samples."""
        _, xyz = self.valid()
        if xyz.size == 0:
            return np.full(3, np.nan)
        return np.ptp(xyz, axis=0)

    def max_spread(self) -> float:
        spread = self.spread()
        if not np.all(np.isfinite(spread)):
            return float("nan")
        return float(np.max(spread))

    def segments(self, max_gap: float | None = None) -> Iterator[np.ndarray]:
        """Yield index arrays of contiguous runs of valid samples.

        A run is also broken when the time step exceeds ``max_gap`` seconds.
        """
        mask = self.valid_mask
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return
        breaks = np.flatnonzero(np.diff(idx) != 1)
        if max_gap is not None and idx.size > 1:
            dt = np.diff(self.t[idx])
            breaks = np.union1d(breaks, np.flatnonzero(dt > max_gap))
        for part in np.split(idx, breaks + 1):
            if part.size:
                yield part

    def shifted(self, offset: float) -> Track:
        """Return a copy with ``offset`` seconds added to the time axis."""
        return replace(self, t=self.t + offset)

    def renamed(self, name: str) -> Track:
        return replace(self, name=name)

    def copy(self) -> Track:
        return Track(
            t=self.t.copy(),
            xyz=self.xyz.copy(),
            name=self.name,
            source=self.source,
            confidence=None if self.confidence is None else self.confidence.copy(),
            meta=dict(self.meta),
        )


@dataclass
class Recording:
    """A loaded recording: the LED track plus system specific extras."""

    led: Track
    path: Path
    camera_markers: np.ndarray | None = None
    frame_rate: float | None = None
    warnings: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.camera_markers is not None:
            self.camera_markers = np.asarray(self.camera_markers, dtype=float)
            if self.camera_markers.shape != (3, 3):
                raise ValueError(
                    f"camera_markers must have shape (3, 3), got {self.camera_markers.shape}"
                )

    @property
    def name(self) -> str:
        return self.led.name

    @property
    def source(self) -> Source:
        return self.led.source
