"""Panel for the spline filter: manual tuning and automatic optimization."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.analysis import deviation_stats
from ..core.filtering import FilterError, SplineParams, apply_spline_filter, optimize_spline
from ..core.model import Track


class FilterPanel(QWidget):
    """Spline controls with live preview and ground-truth based optimization."""

    track_produced = Signal(object)

    def __init__(self, store, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.store = store
        self.tracks: dict[str, Track] = {}

        self.source_box = QComboBox()
        self.truth_box = QComboBox()

        self.smoothing_spin = QDoubleSpinBox()
        self.smoothing_spin.setDecimals(6)
        self.smoothing_spin.setRange(0.0, 1000.0)
        self.smoothing_spin.setSingleStep(0.05)
        self.smoothing_spin.setValue(SplineParams().smoothing)

        self.auto_smoothing_box = QCheckBox("Automatic (GCV)")
        self.auto_smoothing_box.toggled.connect(self.smoothing_spin.setDisabled)

        self.confidence_box = QCheckBox("Weight by confidence")
        self.resample_box = QCheckBox("Resample")
        self.resample_hz = QDoubleSpinBox()
        self.resample_hz.setRange(1.0, 1000.0)
        self.resample_hz.setValue(75.0)
        self.resample_hz.setSuffix(" Hz")
        self.resample_hz.setEnabled(False)
        self.resample_box.toggled.connect(self.resample_hz.setEnabled)

        self.live_box = QCheckBox("Live update")
        self.live_box.setChecked(True)

        apply_button = QPushButton("Apply filter")
        apply_button.clicked.connect(lambda: self.apply_filter())

        optimize_button = QPushButton("Optimize against ground truth")
        optimize_button.clicked.connect(self.optimize)

        self.status = QLabel("No filter applied yet.")
        self.status.setWordWrap(True)

        for widget in (self.smoothing_spin, self.resample_hz):
            widget.valueChanged.connect(self._on_parameter_changed)
        for widget in (self.confidence_box, self.resample_box, self.auto_smoothing_box):
            widget.toggled.connect(self._on_parameter_changed)

        form = QFormLayout()
        form.addRow("Source", self.source_box)
        form.addRow("Ground truth", self.truth_box)
        form.addRow("Smoothing", self.smoothing_spin)
        form.addRow(self.auto_smoothing_box)
        form.addRow(self.confidence_box)
        form.addRow(self.resample_box, self.resample_hz)
        form.addRow(self.live_box)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(apply_button)
        layout.addWidget(optimize_button)
        layout.addWidget(self.status)
        layout.addStretch(1)

    def refresh(self, tracks: dict[str, Track] | None = None) -> None:
        if tracks is not None:
            self.tracks = dict(tracks)
        for box, preferred in ((self.source_box, "oak-d"), (self.truth_box, "optitrack")):
            current = box.currentData()
            box.blockSignals(True)
            box.clear()
            for name, track in self.tracks.items():
                box.addItem(f"{name} [{track.source}]", name)
            index = box.findData(current)
            if index < 0:
                index = self._first_of_source(preferred)
            box.setCurrentIndex(max(index, 0))
            box.blockSignals(False)

    def _first_of_source(self, source: str) -> int:
        for index, (_, track) in enumerate(self.tracks.items()):
            if track.source == source:
                return index
        return 0

    def params(self) -> SplineParams:
        return SplineParams(
            smoothing=self.smoothing_spin.value(),
            use_confidence_weights=self.confidence_box.isChecked(),
            resample_hz=self.resample_hz.value() if self.resample_box.isChecked() else None,
            auto_smoothing=self.auto_smoothing_box.isChecked(),
        )

    def _source_track(self) -> Track | None:
        return self.tracks.get(self.source_box.currentData())

    def _truth_track(self) -> Track | None:
        return self.tracks.get(self.truth_box.currentData())

    def apply_filter(self, announce: bool = True) -> Track | None:
        track = self._source_track()
        if track is None:
            self.status.setText("Load a recording to filter first.")
            return None
        try:
            filtered = apply_spline_filter(track, self.params())
        except FilterError as error:
            self.status.setText(str(error))
            return None

        message = (
            f"Smoothed {filtered.meta['n_smoothed']} samples "
            f"({filtered.meta['n_passed_through']} passed through)."
        )
        truth = self._truth_track()
        if truth is not None and truth is not track:
            stats = deviation_stats(filtered, truth)
            before = deviation_stats(track, truth)
            message += (
                f" Mean deviation from {truth.name}: "
                f"{before.euclidean_mean:.3f} cm before, {stats.euclidean_mean:.3f} cm after."
            )
        if announce:
            self.status.setText(message)
        self.track_produced.emit(filtered)
        return filtered

    def optimize(self) -> None:
        track = self._source_track()
        truth = self._truth_track()
        if track is None or truth is None or track is truth:
            self.status.setText("Pick a recording and a different ground truth first.")
            return
        try:
            best, cost = optimize_spline(track, truth, self.params())
        except FilterError as error:
            self.status.setText(str(error))
            return

        self.auto_smoothing_box.blockSignals(True)
        self.auto_smoothing_box.setChecked(best.auto_smoothing)
        self.auto_smoothing_box.blockSignals(False)
        self.smoothing_spin.setDisabled(best.auto_smoothing)
        self.smoothing_spin.blockSignals(True)
        self.smoothing_spin.setValue(best.smoothing)
        self.smoothing_spin.blockSignals(False)
        self.apply_filter(announce=False)
        self.status.setText(
            f"Optimal smoothing {best.smoothing:.6g} gives a mean deviation of {cost:.3f} cm "
            f"from {truth.name}."
        )

    def _on_parameter_changed(self, *_) -> None:
        if self.live_box.isChecked() and self._source_track() is not None:
            self.apply_filter()
