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
