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
    match_markers,
    rigid_fit,
    save_extrinsic,
    side_lengths,
    to_camera_frame,
)
from point_tracking_filter.core.io.optitrack import load_optitrack
from point_tracking_filter.core.model import Track

MARKERS = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
#: An irregular triangle in camera space: sides 3, 4 and 5 cm.
GEOMETRY = np.array([[-1.0, 2.0, 0.0], [3.0, 2.0, 0.0], [-1.0, 5.0, 0.0]])


def _rotation(angle_deg: float, axis: int = 1) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    cos, sin = np.cos(angle), np.sin(angle)
    matrix = np.eye(3)
    other = [i for i in range(3) if i != axis]
    matrix[np.ix_(other, other)] = [[cos, -sin], [sin, cos]]
    return matrix


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


def test_side_lengths_are_opposite_to_each_marker():
    np.testing.assert_allclose(side_lengths(GEOMETRY), [5.0, 3.0, 4.0])


def test_side_lengths_are_independent_of_marker_order():
    reordered = GEOMETRY[[2, 0, 1]]
    np.testing.assert_allclose(np.sort(side_lengths(reordered)), np.sort(side_lengths(GEOMETRY)))


def test_rigid_fit_recovers_a_known_transform():
    rotation = _rotation(37.0)
    translation = np.array([10.0, -3.0, 4.0])
    target = GEOMETRY @ rotation.T + translation
    fitted_rotation, fitted_translation = rigid_fit(GEOMETRY, target)
    np.testing.assert_allclose(fitted_rotation, rotation, atol=1e-10)
    np.testing.assert_allclose(fitted_translation, translation, atol=1e-10)


def test_rigid_fit_recovers_a_known_improper_transform():
    # An improper (reflecting) transform: det == -1.
    improper = _rotation(37.0) @ np.diag([1.0, 1.0, -1.0])
    translation = np.array([5.0, -1.0, 2.0])
    target = GEOMETRY @ improper.T + translation
    fitted_rotation, fitted_translation = rigid_fit(GEOMETRY, target, allow_reflection=True)
    assert np.linalg.det(fitted_rotation) == pytest.approx(-1.0)
    np.testing.assert_allclose(fitted_rotation, improper, atol=1e-10)
    np.testing.assert_allclose(fitted_translation, translation, atol=1e-10)


def test_rigid_fit_without_allow_reflection_still_fits_three_coplanar_markers():
    # Three points are always coplanar, so a mirrored triangle can equally be
    # matched by *some* proper rotation -- the marker residual alone never
    # distinguishes the two cases; only off-plane points (e.g. the LED) do.
    improper = _rotation(37.0) @ np.diag([1.0, 1.0, -1.0])
    translation = np.array([5.0, -1.0, 2.0])
    target = GEOMETRY @ improper.T + translation
    fitted_rotation, fitted_translation = rigid_fit(GEOMETRY, target)
    assert np.linalg.det(fitted_rotation) == pytest.approx(1.0)
    fitted = GEOMETRY @ fitted_rotation.T + fitted_translation
    np.testing.assert_allclose(fitted, target, atol=1e-9)


@pytest.mark.parametrize("permutation", [(0, 1, 2), (1, 2, 0), (2, 1, 0), (0, 2, 1)])
def test_match_markers_finds_the_only_fitting_arrangement(permutation):
    placed = GEOMETRY @ _rotation(20.0).T + np.array([50.0, 1.0, -7.0])
    observed = placed[list(permutation)]
    match = match_markers(observed, GEOMETRY)
    expected = tuple(int(np.flatnonzero(np.asarray(permutation) == i)[0]) for i in range(3))
    assert match.permutation == expected
    assert match.residual == pytest.approx(0.0, abs=1e-9)
    assert not match.ambiguous


def test_match_markers_flags_an_ambiguous_triangle():
    equilateral = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [2.0, 3.4641016, 0.0]])
    assert match_markers(equilateral, equilateral).ambiguous


def test_extrinsic_puts_the_optical_center_at_the_origin():
    extrinsic = CameraExtrinsic(markers=GEOMETRY)
    rotation = _rotation(-25.0)
    offset = np.array([-92.0, 314.0, 135.0])
    observed = (GEOMETRY @ rotation.T + offset)[[2, 0, 1]]

    aligned = to_camera_frame(_track(observed), observed, extrinsic)
    np.testing.assert_allclose(aligned.xyz, GEOMETRY[[2, 0, 1]], atol=1e-9)
    optical_center = to_camera_frame(_track(offset), observed, extrinsic)
    np.testing.assert_allclose(optical_center.xyz[0], np.zeros(3), atol=1e-9)


def test_extrinsic_rejects_degenerate_geometry():
    with pytest.raises(ValueError, match="distinct points"):
        CameraExtrinsic(markers=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="too regular"):
        CameraExtrinsic(markers=[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [2.0, 3.4641016, 0.0]])


def test_extrinsic_roundtrip(tmp_path):
    extrinsic = CameraExtrinsic(markers=GEOMETRY)
    path = save_extrinsic(extrinsic, tmp_path / "extrinsic.toml")
    assert load_extrinsic(path) == extrinsic


def test_extrinsic_allow_reflection_roundtrip(tmp_path):
    extrinsic = CameraExtrinsic(markers=GEOMETRY, allow_reflection=True)
    path = save_extrinsic(extrinsic, tmp_path / "extrinsic.toml")
    loaded = load_extrinsic(path)
    assert loaded == extrinsic
    assert loaded.allow_reflection is True


def test_allow_reflection_correctly_places_a_point_off_the_marker_plane():
    # The markers form a plane; a mirrored embedding of that plane into the
    # Optitrack world is indistinguishable from a rotated one *for the
    # markers themselves*, but flips the sign of anything off that plane
    # (like the LED). allow_reflection picks the physically correct solution.
    improper = _rotation(37.0) @ np.diag([1.0, 1.0, -1.0])
    translation = np.array([50.0, -13.0, 27.0])
    world_markers = GEOMETRY @ improper.T + translation

    led_in_camera_space = np.array([0.5, 0.5, 5.0])
    world_led = led_in_camera_space @ improper.T + translation

    with_reflection = CameraExtrinsic(markers=GEOMETRY, allow_reflection=True)
    rotation, origin, _ = with_reflection.frame_for(world_markers)
    recovered_led = (world_led - origin) @ rotation.T
    np.testing.assert_allclose(recovered_led, led_in_camera_space, atol=1e-9)

    without_reflection = CameraExtrinsic(markers=GEOMETRY, allow_reflection=False)
    bad_rotation, bad_origin, _ = without_reflection.frame_for(world_markers)
    bad_recovered_led = (world_led - bad_origin) @ bad_rotation.T
    assert not np.allclose(bad_recovered_led, led_in_camera_space, atol=0.5)


def test_load_extrinsic_defaults_to_the_placeholder(tmp_path):
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


def test_align_recording_matches_the_shipped_geometry(optitrack_path):
    recording = align_recording(load_optitrack(optitrack_path), load_extrinsic())
    match = recording.meta["marker_match"]
    assert sorted(match.permutation) == [0, 1, 2]
    assert not match.ambiguous
    assert match.residual < 0.1


def test_align_recording_reports_a_mismeasured_triangle(optitrack_path):
    recording = load_optitrack(optitrack_path)
    align_recording(recording, CameraExtrinsic(markers=GEOMETRY))
    assert any("deviates" in w or "ambiguous" in w for w in recording.warnings)


def test_align_recording_without_markers_raises(optitrack_path):
    recording = load_optitrack(optitrack_path)
    recording.camera_markers = None
    with pytest.raises(AlignmentError, match="no camera markers"):
        align_recording(recording)
