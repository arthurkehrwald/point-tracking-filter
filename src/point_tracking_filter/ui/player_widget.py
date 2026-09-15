"""3D playback widget for one or several tracks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..core.model import Track

PALETTE = [
    (0.20, 0.60, 1.00),
    (1.00, 0.45, 0.20),
    (0.35, 0.85, 0.40),
    (0.95, 0.35, 0.65),
    (0.85, 0.80, 0.25),
    (0.60, 0.45, 0.95),
    (0.30, 0.85, 0.85),
    (0.90, 0.30, 0.30),
]
TIMER_INTERVAL_MS = 16
SLIDER_STEPS = 2000
SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)
DEFAULT_TAIL_SECONDS = 1.5


def line_pairs(
    xyz: np.ndarray, confidence: np.ndarray | None = None
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Consecutive valid point pairs, so gaps are never drawn as lines.

    If ``confidence`` is given, also returns the per-vertex confidence
    (averaged over each segment's endpoints) aligned with the returned pairs.
    """
    valid = np.all(np.isfinite(xyz), axis=1)
    if xyz.shape[0] < 2:
        pos = np.zeros((0, 3))
        return (pos, np.zeros(0)) if confidence is not None else pos
    linked = valid[:-1] & valid[1:]
    if not np.any(linked):
        pos = np.zeros((0, 3))
        return (pos, np.zeros(0)) if confidence is not None else pos
    starts = xyz[:-1][linked]
    ends = xyz[1:][linked]
    pos = np.stack([starts, ends], axis=1).reshape(-1, 3)
    if confidence is None:
        return pos
    segment_confidence = (confidence[:-1][linked] + confidence[1:][linked]) / 2.0
    return pos, np.repeat(segment_confidence, 2)


MIN_CONFIDENCE_ALPHA = 0.15
MIN_CONFIDENCE = 0.8


def confidence_alpha(confidence: np.ndarray | float) -> np.ndarray | float:
    """Map raw confidence values to an opacity in ``[MIN_CONFIDENCE_ALPHA, 1]``."""
    remapped_conf = np.clip((confidence - MIN_CONFIDENCE) / (1 - MIN_CONFIDENCE), 0, 1)
    return MIN_CONFIDENCE_ALPHA + (1.0 - MIN_CONFIDENCE_ALPHA) * remapped_conf


@dataclass
class TrackEntry:
    """A track plus its presentation state."""

    track: Track
    color: tuple[float, float, float]
    visible: bool = True
    visuals: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.track.name

    def qcolor(self) -> QColor:
        return QColor.fromRgbF(*self.color)


class GLScene(QWidget):
    """OpenGL scene drawing trajectories and current positions."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        import pyqtgraph.opengl as gl

        self._gl = gl
        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=200)
        self.grid = gl.GLGridItem()
        self.grid.setSize(400, 400)
        self.grid.setSpacing(20, 20)
        self.view.addItem(self.grid)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view)

    def create(self, entry: TrackEntry) -> None:
        color = (*entry.color, 1.0)
        line = self._gl.GLLinePlotItem(
            pos=np.zeros((0, 3)), color=color, width=2.0, mode="lines", antialias=True
        )
        point = self._gl.GLScatterPlotItem(
            pos=np.zeros((0, 3)), color=color, size=12.0, pxMode=True
        )
        self.view.addItem(line)
        self.view.addItem(point)
        entry.visuals = {"line": line, "point": point}

    def update_entry(
        self,
        entry: TrackEntry,
        path: np.ndarray,
        current: np.ndarray | None,
        confidence: np.ndarray | None = None,
        current_confidence: float | None = None,
    ) -> None:
        line = entry.visuals["line"]
        point = entry.visuals["point"]
        rgb = entry.color
        if entry.visible:
            if confidence is not None:
                pairs, vertex_confidence = line_pairs(path, confidence)
                colors = np.zeros((pairs.shape[0], 4))
                colors[:, :3] = rgb
                colors[:, 3] = confidence_alpha(vertex_confidence)
                line.setData(pos=pairs, color=colors)
            else:
                line.setData(pos=line_pairs(path), color=(*rgb, 1.0))
            if current is None:
                point.setData(pos=np.zeros((0, 3)))
            else:
                alpha = (
                    1.0 if current_confidence is None else float(confidence_alpha(current_confidence))
                )
                point.setData(pos=current.reshape(1, 3), color=(*rgb, alpha))
        else:
            line.setData(pos=np.zeros((0, 3)))
            point.setData(pos=np.zeros((0, 3)))

    def remove(self, entry: TrackEntry) -> None:
        for item in entry.visuals.values():
            self.view.removeItem(item)
        entry.visuals = {}

    def focus(self, center: np.ndarray, extent: float) -> None:
        self.view.opts["center"] = pg.Vector(*center)
        self.view.setCameraPosition(distance=max(extent * 2.0, 10.0))
        self.grid.setSize(extent * 4, extent * 4)
        self.grid.setSpacing(max(extent / 5.0, 1.0), max(extent / 5.0, 1.0))


class ProjectionScene(QWidget):
    """2D fallback showing the XY, XZ and ZY projections."""

    PROJECTIONS = (("X", "Y", 0, 1), ("X", "Z", 0, 2), ("Z", "Y", 2, 1))

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plots = []
        for horizontal, vertical, _, _ in self.PROJECTIONS:
            plot = pg.PlotWidget()
            plot.setLabel("bottom", f"{horizontal} [cm]")
            plot.setLabel("left", f"{vertical} [cm]")
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.setAspectLocked(True)
            layout.addWidget(plot)
            self.plots.append(plot)

    def create(self, entry: TrackEntry) -> None:
        pen = pg.mkPen(QColor.fromRgbF(*entry.color), width=2)
        brush = pg.mkBrush(QColor.fromRgbF(*entry.color))
        curves, markers, confidence_dots = [], [], []
        for plot in self.plots:
            curves.append(plot.plot([], [], pen=pen, connect="finite"))
            dots = pg.ScatterPlotItem(size=6, pen=None)
            plot.addItem(dots)
            confidence_dots.append(dots)
            marker = pg.ScatterPlotItem(size=10, brush=brush, pen=None)
            plot.addItem(marker)
            markers.append(marker)
        entry.visuals = {
            "curves": curves,
            "markers": markers,
            "confidence_dots": confidence_dots,
        }

    def update_entry(
        self,
        entry: TrackEntry,
        path: np.ndarray,
        current: np.ndarray | None,
        confidence: np.ndarray | None = None,
        current_confidence: float | None = None,
    ) -> None:
        rgb = entry.color
        for index, (_, _, horizontal, vertical) in enumerate(self.PROJECTIONS):
            curve = entry.visuals["curves"][index]
            marker = entry.visuals["markers"][index]
            dots = entry.visuals["confidence_dots"][index]
            if entry.visible and path.size:
                curve.setData(path[:, horizontal], path[:, vertical], connect="finite")
            else:
                curve.setData([], [])
            if entry.visible and confidence is not None and path.size:
                alphas = confidence_alpha(confidence)
                brushes = [
                    QColor.fromRgbF(*rgb, float(alpha)) for alpha in alphas
                ]
                dots.setData(path[:, horizontal], path[:, vertical], brush=brushes)
            else:
                dots.setData([], [])
            if entry.visible and current is not None:
                alpha = (
                    1.0 if current_confidence is None else float(confidence_alpha(current_confidence))
                )
                marker.setBrush(QColor.fromRgbF(*rgb, alpha))
                marker.setData([current[horizontal]], [current[vertical]])
            else:
                marker.setData([], [])

    def remove(self, entry: TrackEntry) -> None:
        for plot, curve, marker, dots in zip(
            self.plots,
            entry.visuals.get("curves", []),
            entry.visuals.get("markers", []),
            entry.visuals.get("confidence_dots", []),
        ):
            plot.removeItem(curve)
            plot.removeItem(marker)
            plot.removeItem(dots)
        entry.visuals = {}

    def focus(self, center: np.ndarray, extent: float) -> None:
        for index, (_, _, horizontal, vertical) in enumerate(self.PROJECTIONS):
            self.plots[index].setRange(
                xRange=(center[horizontal] - extent, center[horizontal] + extent),
                yRange=(center[vertical] - extent, center[vertical] + extent),
                padding=0.05,
            )


class PlayerWidget(QWidget):
    """Playback of several tracks with a shared normalized time axis."""

    time_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.entries: dict[str, TrackEntry] = {}
        self._time = 0.0
        self._range = (0.0, 1.0)
        self._updating_slider = False
        self._color_cursor = 0

        self.scene, self.using_opengl = self._build_scene()
        self._build_controls()

        self.timer = QTimer(self)
        self.timer.setInterval(TIMER_INTERVAL_MS)
        self.timer.timeout.connect(self._advance)

    def _build_scene(self) -> tuple[QWidget, bool]:
        try:
            return GLScene(self), True
        except Exception:
            return ProjectionScene(self), False

    def _build_controls(self) -> None:
        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self._on_play_toggled)

        self.speed_box = QComboBox()
        for speed in SPEEDS:
            self.speed_box.addItem(f"{speed:g}x", speed)
        self.speed_box.setCurrentIndex(SPEEDS.index(1.0))

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, SLIDER_STEPS)
        self.slider.valueChanged.connect(self._on_slider_moved)

        self.time_label = QLabel("0.000 s")
        self.time_label.setMinimumWidth(90)
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.full_trajectory_box = QCheckBox("Full trajectory")
        self.full_trajectory_box.toggled.connect(self.refresh)

        self.show_confidence_box = QCheckBox("Show confidence")
        self.show_confidence_box.setChecked(True)
        self.show_confidence_box.toggled.connect(self.refresh)

        self.legend = QListWidget()
        self.legend.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.legend.itemChanged.connect(self._on_legend_changed)

        controls = QHBoxLayout()
        controls.addWidget(self.play_button)
        controls.addWidget(QLabel("Speed"))
        controls.addWidget(self.speed_box)
        controls.addWidget(self.slider, 1)
        controls.addWidget(self.time_label)
        controls.addWidget(self.full_trajectory_box)
        controls.addWidget(self.show_confidence_box)

        layout = QVBoxLayout(self)
        layout.addWidget(self.scene, 1)
        layout.addLayout(controls)

        if not self.using_opengl:
            banner = QLabel(
                "OpenGL is unavailable, showing 2D projections instead of the 3D view."
            )
            banner.setStyleSheet("color: #a06000;")
            layout.insertWidget(0, banner)

    def add_track(self, track: Track, color: tuple[float, float, float] | None = None) -> None:
        """Add or replace a track in the player."""
        if track.name in self.entries:
            self.remove_track(track.name)

        if color is None:
            color = PALETTE[self._color_cursor % len(PALETTE)]
            self._color_cursor += 1

        entry = TrackEntry(track=track, color=color)
        self.scene.create(entry)
        self.entries[track.name] = entry

        label = track.name if track.confidence is None else f"{track.name} (confidence)"
        item = QListWidgetItem(label)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked)
        item.setForeground(entry.qcolor())
        item.setData(Qt.ItemDataRole.UserRole, track.name)
        self.legend.addItem(item)

        self._recompute_range()
        self.reset_view()
        self.refresh()

    def remove_track(self, name: str) -> None:
        entry = self.entries.pop(name, None)
        if entry is None:
            return
        self.scene.remove(entry)
        for row in range(self.legend.count() - 1, -1, -1):
            if self.legend.item(row).data(Qt.ItemDataRole.UserRole) == name:
                self.legend.takeItem(row)
        self._recompute_range()
        self.refresh()

    def clear(self) -> None:
        for name in list(self.entries):
            self.remove_track(name)

    def track_names(self) -> list[str]:
        return list(self.entries)

    @property
    def current_time(self) -> float:
        return self._time

    def set_time(self, value: float) -> None:
        start, end = self._range
        self._time = float(np.clip(value, start, end))
        self._sync_slider()
        self.time_label.setText(f"{self._time:.3f} s")
        self.refresh()
        self.time_changed.emit(self._time)

    def play(self) -> None:
        self.play_button.setChecked(True)

    def pause(self) -> None:
        self.play_button.setChecked(False)

    def reset_view(self) -> None:
        points = [entry.track.valid()[1] for entry in self.entries.values()]
        points = [p for p in points if p.size]
        if not points:
            return
        stacked = np.vstack(points)
        center = (stacked.min(axis=0) + stacked.max(axis=0)) / 2.0
        extent = float(np.max(stacked.max(axis=0) - stacked.min(axis=0))) / 2.0
        self.scene.focus(center, max(extent, 1.0))

    def refresh(self) -> None:
        full = self.full_trajectory_box.isChecked()
        show_confidence = self.show_confidence_box.isChecked()
        tail = DEFAULT_TAIL_SECONDS
        for entry in self.entries.values():
            track = entry.track
            if full:
                window = np.ones(len(track), dtype=bool)
            else:
                window = (track.t >= self._time - tail) & (track.t <= self._time)
            path = track.xyz[window]
            confidence = (
                track.confidence[window]
                if show_confidence and track.confidence is not None
                else None
            )
            current, current_confidence = self._sample_at(track, self._time)
            if not show_confidence:
                current_confidence = None
            self.scene.update_entry(entry, path, current, confidence, current_confidence)

    def _sample_at(
        self, track: Track, time: float
    ) -> tuple[np.ndarray | None, float | None]:
        if len(track) == 0:
            return None, None
        index = int(np.searchsorted(track.t, time, side="right")) - 1
        if index < 0 or index >= len(track):
            return None, None
        point = track.xyz[index]
        if not np.all(np.isfinite(point)):
            return None, None
        confidence = (
            float(track.confidence[index]) if track.confidence is not None else None
        )
        return point, confidence

    def _recompute_range(self) -> None:
        starts, ends = [], []
        for entry in self.entries.values():
            times, _ = entry.track.valid()
            if times.size:
                starts.append(times[0])
                ends.append(times[-1])
        self._range = (min(starts), max(ends)) if starts else (0.0, 1.0)
        if self._range[1] <= self._range[0]:
            self._range = (self._range[0], self._range[0] + 1.0)
        self.set_time(np.clip(self._time, *self._range))

    def _sync_slider(self) -> None:
        start, end = self._range
        fraction = (self._time - start) / (end - start)
        self._updating_slider = True
        self.slider.setValue(int(round(fraction * SLIDER_STEPS)))
        self._updating_slider = False

    def _on_slider_moved(self, value: int) -> None:
        if self._updating_slider:
            return
        start, end = self._range
        self.set_time(start + (end - start) * value / SLIDER_STEPS)

    def _on_play_toggled(self, playing: bool) -> None:
        self.play_button.setText("Pause" if playing else "Play")
        if playing:
            if self._time >= self._range[1]:
                self.set_time(self._range[0])
            self.timer.start()
        else:
            self.timer.stop()

    def _on_legend_changed(self, item: QListWidgetItem) -> None:
        entry = self.entries.get(item.data(Qt.ItemDataRole.UserRole))
        if entry is not None:
            entry.visible = item.checkState() == Qt.CheckState.Checked
            self.refresh()

    def _advance(self) -> None:
        speed = self.speed_box.currentData()
        self.set_time(self._time + speed * TIMER_INTERVAL_MS / 1000.0)
        if self._time >= self._range[1]:
            self.pause()
