"""Reference-frame alignment between Optitrack and the Oak-D camera.

The three markers rigidly mounted on the Oak-D are measured by hand in the
camera's own coordinate space: the origin is the optical center and, seen
from the camera, ``+x`` points right, ``+y`` up and ``+z`` forward. Those
positions live in ``config/extrinsic.toml``.

Because the markers form an irregular triangle there is exactly one way to
map the first three Optitrack column triplets onto them; the assignment is
recovered by comparing the triangle's side lengths. The rigid transform from
the Optitrack global frame into the camera frame then follows from a Kabsch
fit of the matched marker pairs.

The camera space (``+x`` right, ``+y`` up, ``+z`` forward) is left-handed,
while Optitrack's global frame is right-handed, so a *pure* rotation can
never perfectly reconcile the two: ``CameraExtrinsic.allow_reflection`` lets
the fit include a reflection instead of forcing a rotation-only solution.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path

import numpy as np

from .model import Recording, Track

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "extrinsic.toml"
COLLINEAR_TOL = 1e-6
#: Two side lengths closer together than this (in cm) make the triangle too
#: regular to be matched unambiguously.
SIDE_LENGTH_TOL_CM = 0.5
#: A marker assignment is only accepted when the runner-up is worse by at
#: least this factor of ``SIDE_LENGTH_TOL_CM``.
MATCH_MARGIN_CM = 0.5

#: Placeholder geometry: the triangle of the shipped recordings, laid flat
#: above the optical center. Replace with the measured marker positions.
DEFAULT_MARKERS = (
    (-3.118, 1.999, 0.0),
    (3.769, 1.999, 0.0),
    (-0.651, 5.002, 0.0),
)


class AlignmentError(ValueError):
    """Raised when a camera frame cannot be constructed."""


def marker_frame(markers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build an arbitrary orthonormal frame from the three camera markers.

    This is the fallback used when no measured marker geometry is available.
    Returns ``(rotation, origin)`` where ``origin`` is the marker centroid
    and the rows of ``rotation`` are the basis vectors of the marker frame
    expressed in the Optitrack global frame, so that

    ``local = rotation @ (global - origin)``.
    """
    markers = _check_markers(markers)

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


def _check_markers(markers: np.ndarray) -> np.ndarray:
    markers = np.asarray(markers, dtype=float)
    if markers.shape != (3, 3):
        raise AlignmentError(f"expected 3 markers of 3 coordinates, got {markers.shape}")
    if not np.all(np.isfinite(markers)):
        raise AlignmentError("camera markers contain non-finite values")
    return markers


def side_lengths(markers: np.ndarray) -> np.ndarray:
    """Triangle side lengths opposite to each marker.

    Entry ``i`` is the length of the side *not* touching marker ``i``, which
    makes the vector a per-marker signature that is invariant to the marker
    ordering.
    """
    markers = _check_markers(markers)
    return np.array(
        [
            np.linalg.norm(markers[1] - markers[2]),
            np.linalg.norm(markers[2] - markers[0]),
            np.linalg.norm(markers[0] - markers[1]),
        ]
    )


@dataclass
class MarkerMatch:
    """Result of matching observed markers against the measured geometry.

    ``permutation[i]`` is the index of the observed marker that corresponds
    to reference marker ``i``.
    """

    permutation: tuple[int, int, int]
    residual: float
    margin: float

    @property
    def ambiguous(self) -> bool:
        return self.margin < MATCH_MARGIN_CM


def match_markers(observed: np.ndarray, reference: np.ndarray) -> MarkerMatch:
    """Find the unique assignment of observed markers to reference markers.

    The triangle is irregular, so the six possible assignments are scored by
    how well the side lengths agree and the best one wins. The distance to
    the runner-up is reported as ``margin`` so callers can flag an ambiguous
    (too regular or mismeasured) triangle.
    """
    observed = _check_markers(observed)
    reference = _check_markers(reference)

    observed_sides = side_lengths(observed)
    reference_sides = side_lengths(reference)

    scored = sorted(
        (float(np.max(np.abs(observed_sides[list(perm)] - reference_sides))), perm)
        for perm in permutations(range(3))
    )
    best_score, best_perm = scored[0]
    runner_up = scored[1][0]
    return MarkerMatch(
        permutation=tuple(int(i) for i in best_perm),
        residual=best_score,
        margin=float(runner_up - best_score),
    )


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


def rigid_fit(
    source: np.ndarray, target: np.ndarray, allow_reflection: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Kabsch fit: rotation and translation with ``target ≈ source @ R.T + t``.

    By default the fit is restricted to a proper rotation (``det == +1``), as
    Kabsch normally guarantees. For three (always coplanar) marker points the
    fit is exact either way -- a mirrored triangle is equally well matched by
    a rotation -- so the sign only matters for points *off* that plane, such
    as the LED. When the two frames are known to have opposite handedness,
    ``allow_reflection`` picks the improper (``det == -1``) solution instead,
    which places those off-plane points correctly.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    covariance = (source - source_centroid).T @ (target - target_centroid)
    u, _, vt = np.linalg.svd(covariance)
    # For fewer than 4 non-coplanar points the covariance is rank-deficient,
    # so the sign of the last singular vector is not determined by the data:
    # it is exactly the reflection/rotation choice this flag controls.
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.eye(3)
    correction[2, 2] = -sign if allow_reflection else sign
    rotation = vt.T @ correction @ u.T
    return rotation, target_centroid - rotation @ source_centroid


@dataclass(eq=False)
class CameraExtrinsic:
    """Measured marker positions in the Oak-D camera coordinate space.

    The origin is the optical center; seen from the camera ``+x`` points
    right, ``+y`` up and ``+z`` forward. Values are in centimeters, in the
    order the markers were measured — which observed Optitrack triplet maps
    to which of them is recovered automatically by :func:`match_markers`.
    """

    markers: np.ndarray = DEFAULT_MARKERS
    #: Allow the marker fit to include a reflection instead of a pure
    #: rotation, for the case where Optitrack's and the camera's marker
    #: coordinates use opposite handedness.
    allow_reflection: bool = False

    def __post_init__(self) -> None:
        self.markers = _check_markers(self.markers)
        sides = side_lengths(self.markers)
        if np.min(sides) <= 0.0:
            raise ValueError("marker positions must be three distinct points")
        gaps = np.abs(np.diff(np.sort(sides)))
        if np.min(gaps) < SIDE_LENGTH_TOL_CM:
            raise ValueError(
                "the marker triangle is too regular to be matched unambiguously: "
                f"side lengths {np.round(sides, 3).tolist()} cm must differ by at "
                f"least {SIDE_LENGTH_TOL_CM} cm"
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CameraExtrinsic):
            return NotImplemented
        return bool(np.array_equal(self.markers, other.markers)) and (
            self.allow_reflection == other.allow_reflection
        )

    @property
    def side_lengths(self) -> np.ndarray:
        return side_lengths(self.markers)

    def frame_for(self, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray, MarkerMatch]:
        """Rotation and origin taking Optitrack global points to camera space.

        ``camera = rotation @ (global - origin)``.
        """
        observed = _check_markers(observed)
        match = match_markers(observed, self.markers)
        rotation, translation = rigid_fit(
            observed[list(match.permutation)], self.markers, self.allow_reflection
        )
        origin = -rotation.T @ translation
        return rotation, origin, match

    def to_dict(self) -> dict:
        return {
            "markers": [[float(v) for v in row] for row in self.markers],
            "allow_reflection": bool(self.allow_reflection),
        }

    @classmethod
    def from_dict(cls, data: dict) -> CameraExtrinsic:
        markers = data.get("markers")
        if markers is None:
            markers = [data[key] for key in ("marker_1", "marker_2", "marker_3")]
        return cls(
            markers=np.asarray(markers, dtype=float),
            allow_reflection=bool(data.get("allow_reflection", False)),
        )


def load_extrinsic(path: str | Path | None = None) -> CameraExtrinsic:
    """Load the marker geometry from TOML, falling back to the placeholder."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        return CameraExtrinsic()
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return CameraExtrinsic.from_dict(data.get("extrinsic", data))


def save_extrinsic(extrinsic: CameraExtrinsic, path: str | Path | None = None) -> Path:
    """Persist the marker geometry to TOML."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = ",\n".join(
        "    [" + ", ".join(repr(float(v)) for v in row) + "]" for row in extrinsic.markers
    )
    lines = [
        "# Positions of the three tracking markers in the Oak-D camera space.",
        "# The origin is the optical center and, seen from the camera,",
        "# +x points right, +y up and +z forward. Units are centimeters.",
        "#",
        "# The markers form an irregular triangle, so the first three coordinate",
        "# triplets of an Optitrack recording are assigned to them automatically",
        "# by matching the side lengths - only one arrangement fits.",
        "[extrinsic]",
        "markers = [",
        rows,
        "]",
        "# If the recorded markers use the opposite handedness from the camera",
        "# space above, a pure rotation can never fit them; set this to true to",
        "# let the fit include a reflection instead.",
        f"allow_reflection = {'true' if extrinsic.allow_reflection else 'false'}",
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def to_camera_frame(
    track: Track, markers: np.ndarray, extrinsic: CameraExtrinsic | None = None
) -> Track:
    """Express a track given in the Optitrack global frame in the camera frame."""
    if extrinsic is None:
        rotation, origin = marker_frame(markers)
        match = None
    else:
        rotation, origin, match = extrinsic.frame_for(markers)
    local = (track.xyz - origin) @ rotation.T

    aligned = track.copy()
    aligned.xyz = local
    aligned.meta["aligned"] = True
    aligned.meta["marker_frame_origin"] = origin
    aligned.meta["marker_frame_rotation"] = rotation
    if match is not None:
        aligned.meta["marker_match"] = match
    return aligned


def align_recording(
    recording: Recording, extrinsic: CameraExtrinsic | None = None
) -> Recording:
    """Return ``recording`` with its LED track expressed in the camera frame."""
    if recording.camera_markers is None:
        raise AlignmentError(f"{recording.name}: no camera markers to align with")
    recording.led = to_camera_frame(recording.led, recording.camera_markers, extrinsic)
    match = recording.led.meta.get("marker_match")
    if match is not None:
        recording.meta["marker_match"] = match
        if match.ambiguous:
            recording.warnings.append(
                f"{recording.path.name}: marker assignment is ambiguous "
                f"(residual {match.residual:.2f} cm, runner-up only "
                f"{match.margin:.2f} cm worse)"
            )
        elif match.residual > MATCH_MARGIN_CM:
            recording.warnings.append(
                f"{recording.path.name}: measured marker triangle deviates from the "
                f"recorded one by {match.residual:.2f} cm"
            )
    return recording


@dataclass
class AlignmentCache:
    """Memoizes the camera frame per recording pair.

    The camera may have been repositioned between takes, so the transform is
    keyed by pair and never shared globally.
    """

    extrinsic: CameraExtrinsic | None = field(default_factory=CameraExtrinsic)
    _frames: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def frame_for(self, key: str, markers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if key not in self._frames:
            if self.extrinsic is None:
                self._frames[key] = marker_frame(markers)
            else:
                rotation, origin, _ = self.extrinsic.frame_for(markers)
                self._frames[key] = (rotation, origin)
        return self._frames[key]

    def align(self, key: str, track: Track, markers: np.ndarray) -> Track:
        rotation, origin = self.frame_for(key, markers)
        aligned = track.copy()
        aligned.xyz = (track.xyz - origin) @ rotation.T
        aligned.meta["aligned"] = True
        aligned.meta["marker_frame_origin"] = origin
        aligned.meta["marker_frame_rotation"] = rotation
        return aligned

    def invalidate(self, key: str | None = None) -> None:
        if key is None:
            self._frames.clear()
        else:
            self._frames.pop(key, None)
