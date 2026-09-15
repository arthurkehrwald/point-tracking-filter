"""Main application window: recording browser, player and panels."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.align import CameraExtrinsic, align_recording, load_extrinsic
from ..core.io.oak_d import load_oak_d
from ..core.io.optitrack import load_optitrack
from ..core.model import Recording, Track
from ..core.sync import normalize_to_led_onset, recording_token
from .player_widget import PlayerWidget

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RECORDINGS_DIR = PROJECT_ROOT / "recordings"
OAK_D_DIR = RECORDINGS_DIR / "oak-d"
OPTITRACK_DIR = RECORDINGS_DIR / "optitrack"


@dataclass
class RecordingStore:
    """Loads recordings on demand and keeps them prepared for display.

    Every recording is normalized to its LED onset; Optitrack recordings are
    additionally expressed in the Oak-D camera frame.
    """

    extrinsic: CameraExtrinsic = field(default_factory=CameraExtrinsic)
    recordings: dict[Path, Recording] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def load(self, path: Path) -> Recording:
        path = Path(path)
        if path not in self.recordings:
            self.recordings[path] = self._prepare(path)
        return self.recordings[path]

    def _prepare(self, path: Path) -> Recording:
        if path.parent.name == "optitrack":
            recording = load_optitrack(path, name=f"{recording_token(path)} optitrack")
            align_recording(recording, self.extrinsic)
        else:
            recording = load_oak_d(path, name=f"{recording_token(path)} oak-d")
        recording.led = normalize_to_led_onset(recording.led)
        self.warnings.extend(recording.warnings)
        return recording

    def reload_all(self, extrinsic: CameraExtrinsic) -> None:
        """Re-prepare every loaded recording with a new extrinsic."""
        self.extrinsic = extrinsic
        paths = list(self.recordings)
        self.recordings.clear()
        self.warnings.clear()
        for path in paths:
            self.load(path)

    def loaded(self) -> list[Recording]:
        return list(self.recordings.values())

    def by_source(self, source: str) -> list[Recording]:
        return [r for r in self.recordings.values() if r.source == source]


class RecordingBrowser(QDockWidget):
    """Tree of the CSV files found under ``recordings/``."""

    def __init__(self, root: Path, parent: QWidget | None = None) -> None:
        super().__init__("Recordings", parent)
        self.root = Path(root)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self.setWidget(self.tree)
        self.populate()

    def populate(self) -> None:
        self.tree.clear()
        for directory in sorted(p for p in self.root.iterdir() if p.is_dir()):
            group = QTreeWidgetItem([directory.name])
            group.setFlags(Qt.ItemFlag.ItemIsEnabled)
            for path in sorted(directory.glob("*.csv")):
                child = QTreeWidgetItem([path.name])
                child.setData(0, Qt.ItemDataRole.UserRole, str(path))
                group.addChild(child)
            self.tree.addTopLevelItem(group)
            group.setExpanded(True)

    def selected_paths(self) -> list[Path]:
        paths = []
        for item in self.tree.selectedItems():
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data:
                paths.append(Path(data))
        return paths


class MainWindow(QMainWindow):
    """Top level window tying the browser, the player and the panels together."""

    def __init__(self, recordings_dir: Path = RECORDINGS_DIR) -> None:
        super().__init__()
        self.setWindowTitle("Point tracking filter")
        self.resize(1400, 900)

        self.store = RecordingStore(extrinsic=load_extrinsic())
        self.player = PlayerWidget()

        self.banner = QLabel()
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet(
            "background: #ffe9c0; color: #7a4a00; padding: 6px; border: 1px solid #d0a860;"
        )
        self.banner.hide()

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.banner)
        layout.addWidget(self.player, 1)
        self.setCentralWidget(central)

        self.browser = RecordingBrowser(recordings_dir, self)
        self._add_browser_actions()
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.browser)

        self._build_panels()
        self.statusBar().showMessage("Select recordings and press 'Show in player'.")

    def _add_browser_actions(self) -> None:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.browser.tree, 1)

        show_button = QPushButton("Show in player")
        show_button.clicked.connect(self.show_selected)
        layout.addWidget(show_button)

        clear_button = QPushButton("Clear player")
        clear_button.clicked.connect(self.player.clear)
        layout.addWidget(clear_button)

        layout.addWidget(QLabel("In player"))
        layout.addWidget(self.player.legend, 1)

        self.browser.setWidget(container)

    def _build_panels(self) -> None:
        from .analysis_panel import AnalysisPanel
        from .filter_panel import FilterPanel
        from .prediction_panel import PredictionPanel

        self.analysis_panel = AnalysisPanel(self.store, self)
        self.analysis_panel.extrinsic_changed.connect(self.on_extrinsic_changed)
        self.analysis_panel.track_produced.connect(self.on_filtered_track)

        self.filter_panel = FilterPanel(self.store, self)
        self.filter_panel.track_produced.connect(self.on_filtered_track)

        self.prediction_panel = PredictionPanel(self.store, self)
        self.prediction_panel.track_produced.connect(self.on_filtered_track)

        self.side_tabs = QTabWidget()
        self.side_tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.side_tabs.addTab(self._scrollable(self.analysis_panel), "Analysis")
        self.side_tabs.addTab(self._scrollable(self.filter_panel), "Filter")
        self.side_tabs.addTab(self._scrollable(self.prediction_panel), "Prediction")

        self.side_dock = QDockWidget("Panels", self)
        self.side_dock.setWidget(self.side_tabs)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.side_dock)

    @staticmethod
    def _scrollable(widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        return scroll

    def show_selected(self) -> None:
        paths = self.browser.selected_paths()
        if not paths:
            QMessageBox.information(self, "No selection", "Select one or more recordings first.")
            return

        self.store.warnings.clear()
        failures = []
        for path in paths:
            try:
                recording = self.store.load(path)
            except Exception as error:  # noqa: BLE001 - surfaced in the banner
                failures.append(f"{path.name}: {error}")
                continue
            self.player.add_track(recording.led)

        self.show_warnings(self.store.warnings + failures)
        self.refresh_panels()

    def available_tracks(self) -> dict[str, Track]:
        """Every track the panels may work with, in display order."""
        tracks = {
            recording.led.name: recording.led for recording in self.store.loaded()
        }
        tracks.update({name: entry.track for name, entry in self.player.entries.items()})
        return tracks

    def on_filtered_track(self, track: Track) -> None:
        self.player.add_track(track)
        self.refresh_panels()

    def on_extrinsic_changed(self, extrinsic: CameraExtrinsic) -> None:
        self.store.reload_all(extrinsic)
        for recording in self.store.loaded():
            if recording.led.name in self.player.track_names():
                self.player.add_track(recording.led)
        self.player.reset_view()
        self.refresh_panels()
        self.statusBar().showMessage("Camera extrinsic updated.", 5000)

    def refresh_panels(self) -> None:
        tracks = self.available_tracks()
        self.analysis_panel.refresh(tracks)
        self.filter_panel.refresh(tracks)
        self.prediction_panel.refresh(tracks)

    def show_warnings(self, messages: list[str]) -> None:
        if not messages:
            self.banner.hide()
            return
        self.banner.setText("\n".join(dict.fromkeys(messages)))
        self.banner.show()


def run(argv: list[str] | None = None) -> int:
    app = QApplication(argv if argv is not None else sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
