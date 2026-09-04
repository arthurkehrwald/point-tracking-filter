"""Reference-frame alignment between Optitrack and the Oak-D camera.

The three markers rigidly mounted on the Oak-D define a camera frame in the
Optitrack global space. The remaining offset from the marker triangle to the
camera's optical center is *not* fitted from data; it is a hand-specified
extrinsic stored in ``config/extrinsic.toml``.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .model import Recording, Track

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "extrinsic.toml"
COLLINEAR_TOL = 1e-6


class AlignmentError(ValueError):
    """Raised when a camera frame cannot be constructed."""


def marker_frame(markers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build an orthonormal frame from the three camera markers.

    Returns ``(rotation, origin)`` where ``origin`` is the marker centroid
    and the rows of ``rotation`` are the basis vectors of the marker frame
    expressed in the Optitrack global frame, so that

    ``local = rotation @ (global - origin)``.
    """
    markers = np.asarray(markers, dtype=float)
    if markers.shape != (3, 3):
        raise AlignmentError(f"expected 3 markers of 3 coordinates, got {markers.shape}")
    if not np.all(np.isfinite(markers)):
        raise AlignmentError("camera markers contain non-finite values")

    first, second, third = markers
    edge_a = second - first
    edge_b = third - first
    normal = np.cross(edge_a, edge_b)
    area = np.linalg.norm(normal)
    if area < COLLINEAR_TOL * max(np.linalg.norm(edge_a) * np.linalg.norm(edge_b), 1.0):
        raise AlignmentError("camera markers are collinear, cannot build a frame")

    axis_x = edge_a / np.linalg.norm(edge_a)
    axis_z = normal / area
    axis_y = np.cross(axis_z, axis_x)

    rotation = np.stack([axis_x, axis_y, axis_z])
    return rotation, markers.mean(axis=0)


def rotation_from_euler(angles_deg: tuple[float, float, float]) -> np.ndarray:
    """Intrinsic X-Y-Z rotation matrix from degrees."""
    rx, ry, rz = np.deg2rad(np.asarray(angles_deg, dtype=float))
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return mz @ my @ mx


@dataclass
class CameraExtrinsic:
    """Hand-specified marker-triangle to optical-center transform.

    Applied after the axis permutation and sign flips that map the marker
    frame onto the Oak-D camera convention.
    """

    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis_permutation: tuple[int, int, int] = (0, 1, 2)
    axis_signs: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        if sorted(self.axis_permutation) != [0, 1, 2]:
            raise ValueError(
                f"axis_permutation must be a permutation of (0, 1, 2), "
                f"got {self.axis_permutation}"
            )
        if any(sign not in (-1, 1, -1.0, 1.0) for sign in self.axis_signs):
            raise ValueError(f"axis_signs must be +1 or -1, got {self.axis_signs}")

    @property
    def matrix(self) -> np.ndarray:
        """Combined 3x3 linear part including permutation and signs."""
        permutation = np.zeros((3, 3))
        for row, (axis, sign) in enumerate(zip(self.axis_permutation, self.axis_signs)):
            permutation[row, axis] = sign
        return rotation_from_euler(self.rotation_deg) @ permutation

    def apply(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        return points @ self.matrix.T + np.asarray(self.translation, dtype=float)

    def to_dict(self) -> dict:
        return {
            "translation": list(self.translation),
            "rotation_deg": list(self.rotation_deg),
            "axis_permutation": list(self.axis_permutation),
            "axis_signs": list(self.axis_signs),
        }

    @classmethod
    def from_dict(cls, data: dict) -> CameraExtrinsic:
        return cls(
            translation=tuple(float(v) for v in data.get("translation", (0, 0, 0))),
            rotation_deg=tuple(float(v) for v in data.get("rotation_deg", (0, 0, 0))),
            axis_permutation=tuple(int(v) for v in data.get("axis_permutation", (0, 1, 2))),
            axis_signs=tuple(float(v) for v in data.get("axis_signs", (1, 1, 1))),
        )


def load_extrinsic(path: str | Path | None = None) -> CameraExtrinsic:
    """Load the extrinsic from TOML, falling back to identity."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        return CameraExtrinsic()
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return CameraExtrinsic.from_dict(data.get("extrinsic", data))


def save_extrinsic(extrinsic: CameraExtrinsic, path: str | Path | None = None) -> Path:
    """Persist the extrinsic to TOML."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    def floats(values) -> str:
        return "[" + ", ".join(repr(float(v)) for v in values) + "]"

    def ints(values) -> str:
        return "[" + ", ".join(str(int(v)) for v in values) + "]"

    lines = [
        "# Hand-specified transform from the Oak-D marker triangle to the",
        "# camera's optical center. Identity leaves the marker frame as is.",
        "[extrinsic]",
        f"translation = {floats(extrinsic.translation)}",
        f"rotation_deg = {floats(extrinsic.rotation_deg)}",
        f"axis_permutation = {ints(extrinsic.axis_permutation)}",
        f"axis_signs = {floats(extrinsic.axis_signs)}",
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def to_camera_frame(
    track: Track, markers: np.ndarray, extrinsic: CameraExtrinsic | None = None
) -> Track:
    """Express a track given in the Optitrack global frame in the camera frame."""
    rotation, origin = marker_frame(markers)
    local = (track.xyz - origin) @ rotation.T
    if extrinsic is not None:
        local = extrinsic.apply(local)

    aligned = track.copy()
    aligned.xyz = local
    aligned.meta["aligned"] = True
    aligned.meta["marker_frame_origin"] = origin
    aligned.meta["marker_frame_rotation"] = rotation
    return aligned


def align_recording(
    recording: Recording, extrinsic: CameraExtrinsic | None = None
) -> Recording:
    """Return ``recording`` with its LED track expressed in the camera frame."""
    if recording.camera_markers is None:
        raise AlignmentError(f"{recording.name}: no camera markers to align with")
    recording.led = to_camera_frame(recording.led, recording.camera_markers, extrinsic)
    return recording


@dataclass
class AlignmentCache:
    """Memoizes the camera frame per recording pair.

    The camera may have been repositioned between takes, so the transform is
    keyed by pair and never shared globally.
    """

    extrinsic: CameraExtrinsic = field(default_factory=CameraExtrinsic)
    _frames: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def frame_for(self, key: str, markers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if key not in self._frames:
            self._frames[key] = marker_frame(markers)
        return self._frames[key]

    def align(self, key: str, track: Track, markers: np.ndarray) -> Track:
        rotation, origin = self.frame_for(key, markers)
        aligned = track.copy()
        aligned.xyz = self.extrinsic.apply((track.xyz - origin) @ rotation.T)
        aligned.meta["aligned"] = True
        aligned.meta["marker_frame_origin"] = origin
        aligned.meta["marker_frame_rotation"] = rotation
        return aligned

    def invalidate(self, key: str | None = None) -> None:
        if key is None:
            self._frames.clear()
        else:
            self._frames.pop(key, None)
