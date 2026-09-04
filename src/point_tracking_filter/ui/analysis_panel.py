"""Panel driving the deviation analysis and the camera extrinsic."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.align import CameraExtrinsic, save_extrinsic
from ..core.analysis import TRANSFORM_MODELS, DeviationStats, consistency_report
from ..core.model import Track

AXIS_LABELS = ("X", "Y", "Z")


class StatsTable(QTableWidget):
    """Two column table showing one :class:`DeviationStats`."""

    def __init__(self, title: str) -> None:
        super().__init__(0, 2)
        self.setHorizontalHeaderLabels([title, "Value"])
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setStretchLastSection(True)

    def show_stats(self, stats: DeviationStats | None) -> None:
        rows = stats.as_rows() if stats is not None else []
        self.setRowCount(len(rows))
        for row, (label, value) in enumerate(rows):
            self.setItem(row, 0, QTableWidgetItem(label))
            self.setItem(row, 1, QTableWidgetItem(value))
        self.resizeColumnsToContents()


class ExtrinsicEditor(QGroupBox):
    """Editor for the hand-specified marker-to-optical-center transform."""

    changed = Signal(CameraExtrinsic)

    def __init__(self, extrinsic: CameraExtrinsic) -> None:
        super().__init__("Camera extrinsic")
        form = QFormLayout(self)

        self.translation = [self._spin(-500.0, 500.0, 0.1) for _ in AXIS_LABELS]
        self.rotation = [self._spin(-360.0, 360.0, 1.0) for _ in AXIS_LABELS]
        self.permutation = []
        self.signs = []

        form.addRow("Translation [cm]", self._row(self.translation))
        form.addRow("Rotation [deg]", self._row(self.rotation))

        axis_row = QHBoxLayout()
        for _ in AXIS_LABELS:
            box = QComboBox()
            for index, label in enumerate(AXIS_LABELS):
                box.addItem(label, index)
            self.permutation.append(box)
            axis_row.addWidget(box)
        form.addRow("Axis source", self._wrap(axis_row))

        sign_row = QHBoxLayout()
        for _ in AXIS_LABELS:
            box = QComboBox()
            box.addItem("+", 1.0)
            box.addItem("-", -1.0)
            self.signs.append(box)
            sign_row.addWidget(box)
        form.addRow("Axis sign", self._wrap(sign_row))

        buttons = QHBoxLayout()
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self._emit)
        save_button = QPushButton("Apply and save")
        save_button.clicked.connect(self._save)
        buttons.addWidget(apply_button)
        buttons.addWidget(save_button)
        form.addRow(self._wrap(buttons))

        self.set_extrinsic(extrinsic)

    @staticmethod
    def _spin(minimum: float, maximum: float, step: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(3)
        return spin

    @staticmethod
    def _wrap(layout) -> QWidget:
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _row(self, spins) -> QWidget:
        layout = QHBoxLayout()
        for spin in spins:
            layout.addWidget(spin)
        return self._wrap(layout)

    def set_extrinsic(self, extrinsic: CameraExtrinsic) -> None:
        for spin, value in zip(self.translation, extrinsic.translation):
            spin.setValue(value)
        for spin, value in zip(self.rotation, extrinsic.rotation_deg):
            spin.setValue(value)
        for box, value in zip(self.permutation, extrinsic.axis_permutation):
            box.setCurrentIndex(int(value))
        for box, value in zip(self.signs, extrinsic.axis_signs):
            box.setCurrentIndex(0 if value >= 0 else 1)

    def extrinsic(self) -> CameraExtrinsic:
        return CameraExtrinsic(
            translation=tuple(spin.value() for spin in self.translation),
            rotation_deg=tuple(spin.value() for spin in self.rotation),
            axis_permutation=tuple(box.currentData() for box in self.permutation),
            axis_signs=tuple(box.currentData() for box in self.signs),
        )

    def _emit(self) -> CameraExtrinsic | None:
        try:
            extrinsic = self.extrinsic()
        except ValueError as error:
            self.setToolTip(str(error))
            return None
        self.setToolTip("")
        self.changed.emit(extrinsic)
        return extrinsic

    def _save(self) -> None:
        extrinsic = self._emit()
        if extrinsic is not None:
            save_extrinsic(extrinsic)


class AnalysisPanel(QWidget):
    """Pair selection, sync nudge, transform model and the statistics tables."""

    extrinsic_changed = Signal(CameraExtrinsic)

    def __init__(self, store, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.store = store
        self.tracks: dict[str, Track] = {}

        self.analyzed_box = QComboBox()
        self.truth_box = QComboBox()

        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-60.0, 60.0)
        self.offset_spin.setSingleStep(0.01)
        self.offset_spin.setDecimals(3)
        self.offset_spin.setSuffix(" s")

        self.model_box = QComboBox()
        for model in TRANSFORM_MODELS:
            self.model_box.addItem(model.capitalize(), model)

        analyze_button = QPushButton("Analyze")
        analyze_button.clicked.connect(self.analyze)

        form = QFormLayout()
        form.addRow("Analyzed", self.analyzed_box)
        form.addRow("Ground truth", self.truth_box)
        form.addRow("Sync offset", self.offset_spin)
        form.addRow("Transform model", self.model_box)
        form.addRow(analyze_button)

        self.before_table = StatsTable("Before")
        self.after_table = StatsTable("After transform")
        self.summary = QLabel("No analysis yet.")
        self.summary.setWordWrap(True)

        self.transform_view = QPlainTextEdit()
        self.transform_view.setReadOnly(True)
        self.transform_view.setMaximumHeight(90)

        tables = QHBoxLayout()
        tables.addWidget(self.before_table)
        tables.addWidget(self.after_table)

        self.extrinsic_editor = ExtrinsicEditor(store.extrinsic)
        self.extrinsic_editor.changed.connect(self.extrinsic_changed)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(tables, 1)
        layout.addWidget(self.summary)
        layout.addWidget(QLabel("Fitted transform"))
        layout.addWidget(self.transform_view)
        layout.addWidget(self.extrinsic_editor)

    def refresh(self, tracks: dict[str, Track] | None = None) -> None:
        """Update the selectable tracks, keeping the current choices."""
        if tracks is not None:
            self.tracks = dict(tracks)
        for box, preferred in ((self.analyzed_box, "oak-d"), (self.truth_box, "optitrack")):
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
        self.extrinsic_editor.set_extrinsic(self.store.extrinsic)

    def _first_of_source(self, source: str) -> int:
        for index, (_, track) in enumerate(self.tracks.items()):
            if track.source == source:
                return index
        return 0

    def analyze(self) -> None:
        analyzed = self.tracks.get(self.analyzed_box.currentData())
        truth = self.tracks.get(self.truth_box.currentData())
        if analyzed is None or truth is None:
            self.summary.setText("Load an analyzed recording and a ground truth first.")
            return
        if analyzed is truth:
            self.summary.setText("Pick two different recordings.")
            return

        shifted = analyzed.shifted(self.offset_spin.value())
        report = consistency_report(shifted, truth, self.model_box.currentData())

        self.before_table.show_stats(report.before)
        self.after_table.show_stats(report.after)
        self.transform_view.setPlainText(
            "\n".join(
                "  ".join(f"{value:9.4f}" for value in row)
                for row in np.asarray(report.transform)
            )
        )

        messages = [
            f"{report.before.n_samples} comparable samples, "
            f"{report.before.n_excluded} excluded.",
            f"The best {report.model} transform reduces the mean deviation by "
            f"{report.improvement:.3f} cm to {report.after.euclidean_mean:.3f} cm, "
            "which is the frame-independent tracking error.",
        ]
        messages.extend(report.warnings)
        self.summary.setText(" ".join(messages))
