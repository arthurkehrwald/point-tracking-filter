"""Panel driving the deviation analysis and the camera extrinsic."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.align import CameraExtrinsic, save_extrinsic
from ..core.analysis import (
    TRANSFORM_MODELS,
    AnalysisError,
    ConsistencyReport,
    consistency_report,
    deviation_stats,
    smoothness_stats,
    transformed_track,
)
from ..core.model import Track

AXIS_LABELS = ("X", "Y", "Z")


class ComparisonTable(QTableWidget):
    """Table comparing any number of tracks against a shared ground truth."""

    def __init__(self) -> None:
        super().__init__(0, 1)
        self.setHorizontalHeaderLabels(["Metric"])
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setStretchLastSection(True)

    def show_comparison(self, ground_truth: Track, tracks: list[Track]) -> None:
        columns: list[dict[str, str]] = []
        for track in tracks:
            rows = deviation_stats(track, ground_truth).as_rows()
            try:
                rows += [
                    (label, value)
                    for label, value in smoothness_stats(track, ground_truth).as_rows()
                    if label != "Samples"
                ]
            except AnalysisError:
                pass
            columns.append(dict(rows))

        labels = list(columns[0]) if columns else []
        self.setColumnCount(1 + len(tracks))
        self.setHorizontalHeaderLabels(["Metric"] + [track.name for track in tracks])
        self.setRowCount(len(labels))
        for row, label in enumerate(labels):
            self.setItem(row, 0, QTableWidgetItem(label))
            for col, values in enumerate(columns, start=1):
                self.setItem(row, col, QTableWidgetItem(values.get(label, "")))
        self.resizeColumnsToContents()


class ExtrinsicEditor(QGroupBox):
    """Editor for the marker positions in the Oak-D camera space."""

    changed = Signal(CameraExtrinsic)

    def __init__(self, extrinsic: CameraExtrinsic) -> None:
        super().__init__("Marker positions in camera space")
        form = QFormLayout(self)

        hint = QLabel(
            "Origin at the optical center, seen from the camera "
            "+x right, +y up, +z forward. Centimeters."
        )
        hint.setWordWrap(True)
        form.addRow(hint)

        self.markers = [
            [self._spin(-500.0, 500.0, 0.1) for _ in AXIS_LABELS] for _ in range(3)
        ]
        for index, spins in enumerate(self.markers, start=1):
            form.addRow(f"Marker {index} [cm]", self._row(spins))

        self.sides = QLabel()
        self.sides.setWordWrap(True)
        form.addRow("Side lengths", self.sides)

        self.allow_reflection = QCheckBox("Allow reflection (opposite handedness)")
        self.allow_reflection.setToolTip(
            "Lets the marker fit mirror an axis instead of only rotating, for the "
            "case where the camera space and Optitrack's global frame have "
            "opposite handedness."
        )
        form.addRow(self.allow_reflection)

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
        for spins, position in zip(self.markers, extrinsic.markers):
            for spin, value in zip(spins, position):
                spin.setValue(float(value))
        self.allow_reflection.setChecked(extrinsic.allow_reflection)
        self._show_sides(extrinsic)

    def _show_sides(self, extrinsic: CameraExtrinsic) -> None:
        lengths = ", ".join(f"{value:.2f}" for value in extrinsic.side_lengths)
        self.sides.setText(f"{lengths} cm (opposite marker 1, 2, 3)")

    def extrinsic(self) -> CameraExtrinsic:
        return CameraExtrinsic(
            markers=np.array([[spin.value() for spin in spins] for spins in self.markers]),
            allow_reflection=self.allow_reflection.isChecked(),
        )

    def _emit(self) -> CameraExtrinsic | None:
        try:
            extrinsic = self.extrinsic()
        except ValueError as error:
            self.setToolTip(str(error))
            return None
        self.setToolTip("")
        self._show_sides(extrinsic)
        self.changed.emit(extrinsic)
        return extrinsic

    def _save(self) -> None:
        extrinsic = self._emit()
        if extrinsic is not None:
            save_extrinsic(extrinsic)


class AnalysisPanel(QWidget):
    """Transform fitting between a pair, and deviation comparison of any tracks."""

    extrinsic_changed = Signal(CameraExtrinsic)
    track_produced = Signal(object)

    def __init__(self, store, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.store = store
        self.tracks: dict[str, Track] = {}
        self.report: ConsistencyReport | None = None
        self._analyzed_track: Track | None = None

        self.truth_box = QComboBox()

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Ground truth"))
        layout.addWidget(self.truth_box)
        layout.addWidget(self._build_fit_group())
        layout.addWidget(self._build_compare_group(), 1)

        self.extrinsic_editor = ExtrinsicEditor(store.extrinsic)
        self.extrinsic_editor.changed.connect(self.extrinsic_changed)
        layout.addWidget(self.extrinsic_editor)

    def _build_fit_group(self) -> QGroupBox:
        group = QGroupBox("Fit a transform")
        self.analyzed_box = QComboBox()

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

        self.apply_transform_button = QPushButton("Apply transform to copy")
        self.apply_transform_button.setEnabled(False)
        self.apply_transform_button.clicked.connect(self.apply_transform)

        form = QFormLayout(group)
        form.addRow("Analyzed", self.analyzed_box)
        form.addRow("Sync offset", self.offset_spin)
        form.addRow("Transform model", self.model_box)
        form.addRow(analyze_button)
        form.addRow(self.apply_transform_button)

        self.summary = QLabel("No analysis yet.")
        self.summary.setWordWrap(True)
        form.addRow(self.summary)

        self.transform_view = QPlainTextEdit()
        self.transform_view.setReadOnly(True)
        self.transform_view.setMaximumHeight(90)
        form.addRow(QLabel("Fitted transform"))
        form.addRow(self.transform_view)
        return group

    def _build_compare_group(self) -> QGroupBox:
        group = QGroupBox("Compare tracks to the ground truth")
        self.compare_list = QListWidget()
        self.compare_list.setMaximumHeight(120)

        compare_button = QPushButton("Compare")
        compare_button.clicked.connect(self.compare_tracks)

        self.comparison_table = ComparisonTable()

        layout = QVBoxLayout(group)
        layout.addWidget(self.compare_list)
        layout.addWidget(compare_button)
        layout.addWidget(self.comparison_table, 1)
        return group

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
        self._refresh_compare_list()
        self.extrinsic_editor.set_extrinsic(self.store.extrinsic)

    def _refresh_compare_list(self) -> None:
        checked = {
            self.compare_list.item(row).data(Qt.ItemDataRole.UserRole)
            for row in range(self.compare_list.count())
            if self.compare_list.item(row).checkState() == Qt.CheckState.Checked
        }
        self.compare_list.clear()
        for name, track in self.tracks.items():
            item = QListWidgetItem(f"{name} [{track.source}]")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if name in checked else Qt.CheckState.Unchecked
            )
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.compare_list.addItem(item)

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
        self.report = report
        self._analyzed_track = shifted
        self.apply_transform_button.setEnabled(True)

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

    def apply_transform(self) -> None:
        """Create a transformed copy of the analyzed track and add it to the player."""
        if self.report is None:
            return
        corrected = transformed_track(
            self._analyzed_track, self.report.transform, suffix=f"{self.report.model} fit"
        )
        self.track_produced.emit(corrected)

    def compare_tracks(self) -> None:
        """Compare every checked track against the chosen ground truth."""
        truth = self.tracks.get(self.truth_box.currentData())
        if truth is None:
            return
        names = [
            self.compare_list.item(row).data(Qt.ItemDataRole.UserRole)
            for row in range(self.compare_list.count())
            if self.compare_list.item(row).checkState() == Qt.CheckState.Checked
        ]
        tracks = [self.tracks[name] for name in names if name in self.tracks]
        self.comparison_table.show_comparison(truth, tracks)
