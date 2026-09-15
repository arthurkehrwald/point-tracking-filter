"""Headless smoke test of the Qt user interface."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from point_tracking_filter.ui.main_window import MainWindow  # noqa: E402
from point_tracking_filter.ui.player_widget import line_pairs  # noqa: E402

from .conftest import OAK_D_DIR, OPTITRACK_DIR  # noqa: E402


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture
def window(app):
    window = MainWindow()
    yield window
    window.close()


def test_line_pairs_skips_gaps():
    xyz = np.array([[0.0, 0, 0], [1.0, 0, 0], [np.nan] * 3, [3.0, 0, 0], [4.0, 0, 0]])
    pairs = line_pairs(xyz)
    assert pairs.shape == (4, 3)
    np.testing.assert_allclose(pairs[:, 0], [0.0, 1.0, 3.0, 4.0])


def test_browser_lists_all_recordings(window):
    groups = [
        window.browser.tree.topLevelItem(index)
        for index in range(window.browser.tree.topLevelItemCount())
    ]
    names = {group.text(0): group.childCount() for group in groups}
    assert names["oak-d"] == len(list(OAK_D_DIR.glob("*.csv")))
    assert names["optitrack"] == len(list(OPTITRACK_DIR.glob("*.csv")))


def test_full_workflow(window):
    oak_path = next(OAK_D_DIR.glob("slow2*.csv"))
    opti_path = next(OPTITRACK_DIR.glob("slow2*.csv"))

    oak = window.store.load(oak_path)
    opti = window.store.load(opti_path)
    window.player.add_track(oak.led)
    window.player.add_track(opti.led)
    window.refresh_panels()

    assert len(window.player.entries) == 2
    assert window.banner.isHidden()

    # Playback: scrubbing and stepping the clock.
    window.player.set_time(1.0)
    assert window.player.current_time == pytest.approx(1.0)
    window.player.play()
    window.player._advance()
    assert window.player.current_time > 1.0
    window.player.pause()
    window.player.full_trajectory_box.setChecked(True)
    window.player.refresh()

    # Visibility toggle through the legend.
    item = window.player.legend.item(0)
    item.setCheckState(Qt.CheckState.Unchecked)
    assert not window.player.entries[oak.led.name].visible

    # Analysis of the pair.
    panel = window.analysis_panel
    panel.analyzed_box.setCurrentIndex(panel.analyzed_box.findData(oak.led.name))
    panel.truth_box.setCurrentIndex(panel.truth_box.findData(opti.led.name))
    for model_index in range(panel.model_box.count()):
        panel.model_box.setCurrentIndex(model_index)
        panel.analyze()
        assert panel.before_table.rowCount() > 0
        assert "comparable samples" in panel.summary.text()

    # Filtering, fed back into the player.
    filter_panel = window.filter_panel
    filter_panel.source_box.setCurrentIndex(filter_panel.source_box.findData(oak.led.name))
    filter_panel.truth_box.setCurrentIndex(filter_panel.truth_box.findData(opti.led.name))
    filtered = filter_panel.apply_filter()

    assert filtered is not None
    assert filtered.source == "filtered"
    assert filtered.name in window.player.track_names()
    assert "Mean deviation" in filter_panel.status.text()


def test_extrinsic_round_trip_through_the_panel(window, tmp_path, monkeypatch):
    from point_tracking_filter.ui import analysis_panel as module

    saved = {}
    monkeypatch.setattr(module, "save_extrinsic", lambda e, p=None: saved.setdefault("e", e))

    opti = window.store.load(next(OPTITRACK_DIR.glob("slow2*.csv")))
    window.player.add_track(opti.led)
    window.refresh_panels()
    before = opti.led.xyz.copy()

    editor = window.analysis_panel.extrinsic_editor
    moved = window.store.extrinsic.markers + np.array([5.0, 0.0, 0.0])
    for spins, position in zip(editor.markers, moved):
        spins[0].setValue(float(position[0]))
    editor._save()

    np.testing.assert_allclose(saved["e"].markers, moved)
    np.testing.assert_allclose(window.store.extrinsic.markers, moved)

    reloaded = window.store.load(next(OPTITRACK_DIR.glob("slow2*.csv")))
    shift = np.nanmean(reloaded.led.xyz - before, axis=0)
    np.testing.assert_allclose(shift, [5.0, 0.0, 0.0], atol=1e-9)


def test_filter_panel_method_switch(window):
    oak = window.store.load(next(OAK_D_DIR.glob("slow2*.csv")))
    opti = window.store.load(next(OPTITRACK_DIR.glob("slow2*.csv")))
    window.player.add_track(oak.led)
    window.player.add_track(opti.led)
    window.refresh_panels()

    panel = window.filter_panel
    panel.live_box.setChecked(False)
    panel.source_box.setCurrentIndex(panel.source_box.findData(oak.led.name))
    panel.truth_box.setCurrentIndex(panel.truth_box.findData(opti.led.name))

    for index in range(panel.method_box.count()):
        panel.method_box.setCurrentIndex(index)
        method = panel.method_box.currentData()
        assert panel.degree_spin.isEnabled() == (method != "gcv")
        assert panel.auto_smoothing_box.isEnabled() == (method == "gcv")
        filtered = panel.apply_filter()
        assert filtered is not None
        assert filtered.meta["spline_params"].method == method

    panel.method_box.setCurrentIndex(panel.method_box.findData("gcv"))
    panel.auto_smoothing_box.setChecked(True)
    assert not panel.smoothing_spin.isEnabled()
    filtered = panel.apply_filter()
    assert filtered is not None
    assert filtered.meta["spline_params"].auto_smoothing is True


def test_show_confidence_toggle(window):
    oak = window.store.load(next(OAK_D_DIR.glob("slow2*.csv")))
    assert oak.led.confidence is not None
    window.player.add_track(oak.led)
    window.player.full_trajectory_box.setChecked(True)

    calls = []
    original = window.player.scene.update_entry

    def spy(entry, path, current, confidence=None, current_confidence=None):
        calls.append((confidence, current_confidence))
        return original(entry, path, current, confidence, current_confidence)

    window.player.scene.update_entry = spy

    window.player.show_confidence_box.setChecked(True)
    window.player.refresh()
    confidence, current_confidence = calls[-1]
    assert confidence is not None
    assert current_confidence is not None

    window.player.show_confidence_box.setChecked(False)
    confidence, current_confidence = calls[-1]
    assert confidence is None
    assert current_confidence is None


def test_prediction_panel_shows_variants_side_by_side(window):
    oak = window.store.load(next(OAK_D_DIR.glob("slow2*.csv")))
    window.player.add_track(oak.led)
    window.refresh_panels()

    panel = window.prediction_panel
    panel.source_box.setCurrentIndex(panel.source_box.findData(oak.led.name))
    panel.show_in_player()

    # Every ticked variant reaches the player next to the original recording.
    names = window.player.track_names()
    assert oak.led.name in names
    predicted = [name for name in names if "+50 ms" in name]
    assert len(predicted) == 2
    assert any("Constant-velocity" in name for name in predicted)

    assert panel.table.rowCount() == 2
    labels = [panel.table.item(row, 0).text() for row in range(2)]
    assert "Zero-order hold" in labels
    # Accuracy and smoothness are reported together, since either alone can
    # pick a predictor that is unusable on the other axis.
    for row in range(2):
        for column in range(1, panel.table.columnCount()):
            assert panel.table.item(row, column).text() not in ("", "nan")


def test_prediction_panel_estimates_and_tunes(window):
    oak = window.store.load(next(OAK_D_DIR.glob("slow2*.csv")))
    window.player.add_track(oak.led)
    window.refresh_panels()

    panel = window.prediction_panel
    panel.source_box.setCurrentIndex(panel.source_box.findData(oak.led.name))

    panel.estimate_from_recording()
    assert "measurement noise" in panel.status.text()
    assert panel.sigma_a_spin.value() > 0

    panel.tune_selected()
    assert "Tuned fixed process noise" in panel.status.text()
    assert 0.5 <= panel.sigma_a_spin.value() <= 300.0


def test_prediction_panel_reports_missing_input(window):
    panel = window.prediction_panel
    panel.tracks = {}
    panel.refresh({})
    panel.show_in_player()
    assert "Load a recording" in panel.status.text()


def test_optimizer_button_runs(window):
    oak = window.store.load(next(OAK_D_DIR.glob("slow2*.csv")))
    opti = window.store.load(next(OPTITRACK_DIR.glob("slow2*.csv")))
    window.player.add_track(oak.led)
    window.player.add_track(opti.led)
    window.refresh_panels()

    panel = window.filter_panel
    panel.live_box.setChecked(False)
    panel.source_box.setCurrentIndex(panel.source_box.findData(oak.led.name))
    panel.truth_box.setCurrentIndex(panel.truth_box.findData(opti.led.name))
    panel.optimize()

    assert "Optimal smoothing" in panel.status.text()
    assert panel.smoothing_spin.value() >= 0.0
