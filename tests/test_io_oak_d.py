from __future__ import annotations

import numpy as np
import pytest

from point_tracking_filter.core.io.oak_d import load_oak_d


def test_load_every_recording(oak_d_path):
    recording = load_oak_d(oak_d_path)
    track = recording.led

    assert recording.source == "oak-d"
    assert len(track) > 0
    assert np.all(np.diff(track.t) >= 0)
    assert np.all(np.isfinite(track.xyz))
    assert track.n_valid == len(track)

    assert track.confidence is not None
    assert np.all((track.confidence >= 0) & (track.confidence <= 1))

    assert recording.frame_rate is not None
    assert 50.0 < recording.frame_rate < 120.0
    assert recording.camera_markers is None
    assert "detection_time" in track.meta


def test_missing_column_raises(tmp_path):
    path = tmp_path / "broken.csv"
    path.write_text("x,y,capture_time\n1,2,3\n")
    with pytest.raises(ValueError, match="missing column"):
        load_oak_d(path)


def test_rows_are_sorted_by_capture_time(tmp_path):
    path = tmp_path / "unsorted.csv"
    path.write_text(
        "x,y,z,capture_time,detection_time,confidence\n"
        "1,1,1,2.0,2.1,0.9\n"
        "0,0,0,1.0,1.1,0.8\n"
    )
    track = load_oak_d(path).led
    assert track.t.tolist() == [1.0, 2.0]
    assert track.xyz[0].tolist() == [0.0, 0.0, 0.0]
