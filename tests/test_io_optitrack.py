from __future__ import annotations

import numpy as np
import pytest

from point_tracking_filter.core.io.optitrack import (
    CAMERA_MOTION_TOL_CM,
    load_optitrack,
    merge_led_fragments,
    parse_metadata,
    unit_scale_to_cm,
    validate_recording,
)

from .conftest import OPTITRACK_DIR


def test_load_every_recording(optitrack_path):
    recording = load_optitrack(optitrack_path)
    track = recording.led

    assert recording.source == "optitrack"
    assert recording.frame_rate == pytest.approx(120.0)
    assert recording.meta["length_units"] == "Centimeters"
    assert recording.meta["capture_start_time"]

    assert len(track) > 0
    assert np.all(np.diff(track.t) > 0)
    assert track.n_valid > 0

    assert recording.camera_markers.shape == (3, 3)
    assert np.all(np.isfinite(recording.camera_markers))
    assert np.nanmax(recording.meta["marker_spread"]) <= CAMERA_MOTION_TOL_CM

    assert recording.warnings == []


def test_fragment_counts():
    sweep = load_optitrack(next(OPTITRACK_DIR.glob("sweep*.csv")))
    assert sweep.meta["n_fragments"] == 11

    slow5 = load_optitrack(next(OPTITRACK_DIR.glob("slow5*.csv")))
    assert slow5.meta["n_fragments"] == 3

    slow2 = load_optitrack(next(OPTITRACK_DIR.glob("slow2*.csv")))
    assert slow2.meta["n_fragments"] == 1


def test_merged_track_covers_all_fragments(optitrack_path):
    recording = load_optitrack(optitrack_path)
    index = recording.meta["fragment_index"]
    assert np.count_nonzero(index >= 0) == recording.led.n_valid
    assert index.max() < recording.meta["n_fragments"]


def test_parse_metadata():
    line = "Format Version,1.24,Capture Frame Rate,120.000000,Length Units,Centimeters"
    metadata = parse_metadata(line)
    assert metadata["Format Version"] == "1.24"
    assert metadata["Capture Frame Rate"] == "120.000000"
    assert metadata["Length Units"] == "Centimeters"


def test_unit_scale():
    assert unit_scale_to_cm("Centimeters") == 1.0
    assert unit_scale_to_cm("Millimeters") == pytest.approx(0.1)
    assert unit_scale_to_cm("Meters") == 100.0
    with pytest.raises(ValueError):
        unit_scale_to_cm("Furlongs")


def test_merge_picks_nearest_on_conflict():
    nan = np.nan
    first = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [nan, nan, nan]])
    second = np.array([[nan, nan, nan], [50.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

    merged, meta = merge_led_fragments([first, second])

    assert meta["merge_conflicts"] == 1
    assert meta["n_fragments"] == 2
    np.testing.assert_allclose(merged[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(merged[1], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(merged[2], [3.0, 0.0, 0.0])


def test_merge_without_fragments_raises():
    with pytest.raises(ValueError, match="no LED fragments"):
        merge_led_fragments([])


def test_file_without_led_columns_raises(tmp_path, optitrack_path):
    lines = optitrack_path.read_text(encoding="utf-8-sig").splitlines()
    header_index = next(i for i, line in enumerate(lines) if line.startswith("Frame,"))
    keep = 3 + 3 * 3
    trimmed = []
    for index, line in enumerate(lines[: header_index + 3]):
        if index >= 2 and line.strip():
            trimmed.append(",".join(line.split(",")[:keep]))
        else:
            trimmed.append(line)
    path = tmp_path / "no_led.csv"
    path.write_text("\n".join(trimmed) + "\n")

    with pytest.raises(ValueError, match="no LED marker columns"):
        load_optitrack(path)


def test_validate_flags_moving_camera(optitrack_path):
    recording = load_optitrack(optitrack_path)
    warnings = validate_recording(recording, camera_motion_tol=1e-9)
    assert len(warnings) == 3
    assert all("moved by" in message for message in warnings)


def test_validate_flags_static_led(optitrack_path):
    recording = load_optitrack(optitrack_path)
    warnings = validate_recording(recording, led_motion_min=1e9)
    assert any("the LED moved only" in message for message in warnings)
