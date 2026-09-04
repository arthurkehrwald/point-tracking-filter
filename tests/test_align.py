from __future__ import annotations

import numpy as np
import pytest

from point_tracking_filter.core.align import (
    AlignmentCache,
    AlignmentError,
    CameraExtrinsic,
    align_recording,
    load_extrinsic,
    marker_frame,
    rotation_from_euler,
    save_extrinsic,
    to_camera_frame,
)
from point_tracking_filter.core.io.optitrack import load_optitrack
from point_tracking_filter.core.model import Track

MARKERS = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])


def _track(xyz) -> Track:
    xyz = np.atleast_2d(np.asarray(xyz, dtype=float))
    return Track(t=np.arange(len(xyz), dtype=float), xyz=xyz, name="t", source="optitrack")


def test_marker_frame_is_orthonormal_and_right_handed():
    rotation, origin = marker_frame(MARKERS)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    np.testing.assert_allclose(origin, MARKERS.mean(axis=0))


def test_marker_frame_on_real_recordings(optitrack_path):
    recording = load_optitrack(optitrack_path)
    rotation, origin = marker_frame(recording.camera_markers)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-10)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    np.testing.assert_allclose(origin, recording.camera_markers.mean(axis=0))


def test_collinear_markers_raise():
    collinear = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    with pytest.raises(AlignmentError, match="collinear"):
        marker_frame(collinear)


def test_marker_frame_rejects_nan():
    markers = MARKERS.copy()
    markers[1, 1] = np.nan
    with pytest.raises(AlignmentError, match="non-finite"):
        marker_frame(markers)


def test_to_camera_frame_moves_centroid_to_origin():
    aligned = to_camera_frame(_track(MARKERS), MARKERS)
    np.testing.assert_allclose(aligned.xyz.mean(axis=0), np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(aligned.xyz[:, 2], np.zeros(3), atol=1e-12)
    assert aligned.meta["aligned"] is True


def test_to_camera_frame_preserves_distances():
    points = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 2.0]])
    aligned = to_camera_frame(_track(points), MARKERS)
    original = np.linalg.norm(points[0] - points[1])
    assert np.linalg.norm(aligned.xyz[0] - aligned.xyz[1]) == pytest.approx(original)


def test_to_camera_frame_keeps_gaps():
    points = np.array([[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan]])
    aligned = to_camera_frame(_track(points), MARKERS)
    assert aligned.valid_mask.tolist() == [True, False]


def test_extrinsic_identity_is_a_no_op():
    points = np.array([[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(CameraExtrinsic().apply(points), points)


def test_extrinsic_permutation_and_signs():
    extrinsic = CameraExtrinsic(axis_permutation=(2, 0, 1), axis_signs=(1, -1, 1))
    np.testing.assert_allclose(extrinsic.apply([[1.0, 2.0, 3.0]]), [[3.0, -1.0, 2.0]])


def test_extrinsic_rotation_and_translation():
    extrinsic = CameraExtrinsic(translation=(1.0, 0.0, 0.0), rotation_deg=(0.0, 0.0, 90.0))
    np.testing.assert_allclose(extrinsic.apply([[1.0, 0.0, 0.0]]), [[1.0, 1.0, 0.0]], atol=1e-12)


def test_extrinsic_rejects_invalid_permutation():
    with pytest.raises(ValueError, match="permutation"):
        CameraExtrinsic(axis_permutation=(0, 0, 1))
    with pytest.raises(ValueError, match="axis_signs"):
        CameraExtrinsic(axis_signs=(1, 2, 1))


def test_rotation_from_euler_is_orthonormal():
    rotation = rotation_from_euler((10.0, -20.0, 30.0))
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)


def test_extrinsic_roundtrip(tmp_path):
    extrinsic = CameraExtrinsic(
        translation=(1.5, -2.0, 0.25),
        rotation_deg=(0.0, 90.0, 15.0),
        axis_permutation=(1, 2, 0),
        axis_signs=(1.0, -1.0, 1.0),
    )
    path = save_extrinsic(extrinsic, tmp_path / "extrinsic.toml")
    assert load_extrinsic(path) == extrinsic


def test_load_extrinsic_defaults_to_identity(tmp_path):
    assert load_extrinsic(tmp_path / "missing.toml") == CameraExtrinsic()


def test_shipped_config_loads():
    assert isinstance(load_extrinsic(), CameraExtrinsic)


def test_alignment_cache_reuses_the_frame():
    cache = AlignmentCache()
    first = cache.align("slow2", _track(MARKERS), MARKERS)
    second = cache.align("slow2", _track(MARKERS), MARKERS)
    np.testing.assert_allclose(first.xyz, second.xyz)
    assert cache.frame_for("slow2", MARKERS) is cache.frame_for("slow2", MARKERS)

    cache.invalidate("slow2")
    assert "slow2" not in cache._frames


def test_align_recording_without_markers_raises(optitrack_path):
    recording = load_optitrack(optitrack_path)
    recording.camera_markers = None
    with pytest.raises(AlignmentError, match="no camera markers"):
        align_recording(recording)
