from __future__ import annotations

import numpy as np
import pytest

from point_tracking_filter.core.io.oak_d import load_oak_d
from point_tracking_filter.core.io.optitrack import load_optitrack
from point_tracking_filter.core.model import Track
from point_tracking_filter.core.sync import (
    SyncError,
    led_onset_index,
    normalize_to_led_onset,
    pair_paths,
    pair_recordings,
    recording_token,
)

from .conftest import OAK_D_DIR, OPTITRACK_DIR


def _track(valid: list[bool]) -> Track:
    xyz = np.array([[1.0, 2.0, 3.0] if v else [np.nan] * 3 for v in valid])
    return Track(t=np.arange(len(valid), dtype=float), xyz=xyz, name="t", source="optitrack")


def test_onset_skips_spurious_single_sample():
    track = _track([False, True, False, False, True, True, True, True])
    assert led_onset_index(track, min_run=3) == 4


def test_onset_falls_back_when_no_long_run():
    track = _track([False, True, False])
    assert led_onset_index(track, min_run=3) == 1


def test_onset_on_all_nan_track_raises():
    with pytest.raises(SyncError, match="no valid samples"):
        led_onset_index(_track([False, False]))


def test_normalize_puts_onset_at_zero():
    track = _track([False, False, True, True, True])
    normalized = normalize_to_led_onset(track)
    assert normalized.t[2] == 0.0
    assert normalized.meta["led_onset_raw_time"] == 2.0
    assert track.t[2] == 2.0


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("slow2_2026-08-04_15-28-00.csv", "slow2"),
        ("slow5 2min 2026-08-04 03.53.42 PM.csv", "slow5"),
        ("slow5_2_min_2026-08-04_15-56-02.csv", "slow5"),
        ("sweep 2026-08-04 04.11.10 PM.csv", "sweep"),
        ("fast1 2026-08-04 02.58.35 PM.csv", "fast1"),
    ],
)
def test_recording_token(filename, expected):
    assert recording_token(filename) == expected


def test_pair_paths_matches_the_sample_recordings():
    matches = pair_paths(OAK_D_DIR, OPTITRACK_DIR)
    tokens = [token for token, _, _ in matches]
    assert tokens == ["fast1", "fast2", "fast3", "fast4", "slow2", "slow3", "slow4", "slow5", "sweep"]
    for token, analyzed, truth in matches:
        assert recording_token(analyzed) == recording_token(truth) == token


def test_pair_recordings_and_onset_alignment():
    _, oak_path, opti_path = pair_paths(OAK_D_DIR, OPTITRACK_DIR)[0]
    oak = load_oak_d(oak_path)
    opti = load_optitrack(opti_path)

    pair = pair_recordings([oak], [opti])[0]
    assert pair.token == "fast1"

    oak.led = normalize_to_led_onset(oak.led)
    opti.led = normalize_to_led_onset(opti.led)
    assert oak.led.t[led_onset_index(oak.led)] == 0.0
    assert opti.led.t[led_onset_index(opti.led)] == 0.0


def test_pair_offset_shifts_the_analyzed_track():
    _, oak_path, opti_path = pair_paths(OAK_D_DIR, OPTITRACK_DIR)[0]
    pair = pair_recordings([load_oak_d(oak_path)], [load_optitrack(opti_path)])[0]
    pair.offset = 0.5
    np.testing.assert_allclose(pair.analyzed_track().t, pair.analyzed.led.t + 0.5)
