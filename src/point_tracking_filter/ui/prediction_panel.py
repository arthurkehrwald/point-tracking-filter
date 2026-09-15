"""Panel for predictive filtering: run variants side by side.

The point of showing several variants at once is that no single number
decides between them. The player shows the trajectories on top of each
other, and the Analysis panel's track comparison can score any of them
(together with the reference this panel used) against a ground truth.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.analysis import AnalysisError
from ..core.model import Track
from ..core.prediction import (
    ConstantVelocityKalman,
    PredictionError,
    StreamConfig,
    WindowedSplineRefit,
    ZeroOrderHold,
    estimate_noise,
    oracle,
    simulate,
    tune,
)

ORACLE_LABEL = "Offline bound (non-causal)"


class PredictionPanel(QWidget):
    """Causal prediction with a live comparison of the candidate filters."""

    track_produced = Signal(object)

    def __init__(self, store, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.store = store
        self.tracks: dict[str, Track] = {}

        self.source_box = QComboBox()
        self.truth_box = QComboBox()

        self.horizon_spin = QDoubleSpinBox()
        self.horizon_spin.setRange(0.0, 500.0)
        self.horizon_spin.setValue(50.0)
        self.horizon_spin.setSingleStep(5.0)
        self.horizon_spin.setSuffix(" ms")
        self.horizon_spin.setToolTip(
            "Latency still ahead of the prediction. The camera's own latency "
            "is measured from the recording and added on top."
        )

        self.hold_box = QCheckBox("Hold (no prediction)")
        self.hold_box.setChecked(True)
        self.fixed_box = QCheckBox("Kalman, fixed")
        self.fixed_box.setChecked(True)
        self.spline_box = QCheckBox("Windowed spline refit")

        self.sigma_a_spin = self._noise_spin(7.0, "Process noise of the fixed filter.")

        self.spline_window_spin = QDoubleSpinBox()
        self.spline_window_spin.setRange(10.0, 2000.0)
        self.spline_window_spin.setValue(300.0)
        self.spline_window_spin.setSingleStep(10.0)
        self.spline_window_spin.setSuffix(" ms")
        self.spline_window_spin.setToolTip("Trailing window refit on every update.")

        self.spline_smoothing_spin = QDoubleSpinBox()
        self.spline_smoothing_spin.setDecimals(3)
        self.spline_smoothing_spin.setRange(0.0, 1000.0)
        self.spline_smoothing_spin.setValue(0.05)
        self.spline_smoothing_spin.setSingleStep(0.01)
        self.spline_smoothing_spin.setToolTip("Smoothing penalty passed to the GCV spline backend.")

        show_button = QPushButton("Show in player")
        show_button.clicked.connect(self.show_in_player)

        tune_button = QPushButton("Tune on this recording")
        tune_button.clicked.connect(self.tune_selected)

        estimate_button = QPushButton("Estimate noise from recording")
        estimate_button.clicked.connect(self.estimate_from_recording)

        self.status = QLabel("No prediction yet.")
        self.status.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Source", self.source_box)
        form.addRow("Reference", self.truth_box)
        form.addRow("Horizon", self.horizon_spin)

        variants = QGroupBox("Variants")
        variant_layout = QFormLayout(variants)
        variant_layout.addRow(self.hold_box)
        variant_layout.addRow(self.fixed_box, self.sigma_a_spin)
        variant_layout.addRow(self.spline_box, self.spline_window_spin)
        variant_layout.addRow("  smoothing", self.spline_smoothing_spin)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(variants)
        layout.addWidget(show_button)
        layout.addWidget(tune_button)
        layout.addWidget(estimate_button)
        layout.addWidget(self.status)
        layout.addStretch(1)

    @staticmethod
    def _noise_spin(value: float, tip: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(2)
        spin.setRange(0.01, 10000.0)
        spin.setValue(value)
        spin.setToolTip(tip)
        return spin

    def refresh(self, tracks: dict[str, Track] | None = None) -> None:
        if tracks is not None:
            self.tracks = dict(tracks)

        current = self.source_box.currentData()
        self.source_box.blockSignals(True)
        self.source_box.clear()
        for name, track in self.tracks.items():
            self.source_box.addItem(f"{name} [{track.source}]", name)
        index = self.source_box.findData(current)
        if index < 0:
            index = self._first_of_source("oak-d")
        self.source_box.setCurrentIndex(max(index, 0))
        self.source_box.blockSignals(False)

        chosen = self.truth_box.currentData()
        self.truth_box.blockSignals(True)
        self.truth_box.clear()
        self.truth_box.addItem(ORACLE_LABEL, None)
        for name, track in self.tracks.items():
            self.truth_box.addItem(f"{name} [{track.source}]", name)
        restored = self.truth_box.findData(chosen)
        self.truth_box.setCurrentIndex(max(restored, 0))
        self.truth_box.blockSignals(False)

    def _first_of_source(self, source: str) -> int:
        for index, (_, track) in enumerate(self.tracks.items()):
            if track.source == source:
                return index
        return 0

    def config(self) -> StreamConfig:
        return StreamConfig(horizon=self.horizon_spin.value() / 1000.0)

    def _source_track(self) -> Track | None:
        return self.tracks.get(self.source_box.currentData())

    def _reference(self, track: Track, config: StreamConfig) -> Track:
        """What the variants are scored against.

        Defaults to the offline smoother, which is the best estimate of
        where the target actually was and is available for every recording;
        an independently recorded track can be picked instead.
        """
        chosen = self.truth_box.currentData()
        if chosen is not None and chosen in self.tracks:
            return self.tracks[chosen]
        return oracle(track, config)

    def predictors(self) -> list:
        """The variants currently ticked, in table order."""
        sigma_m = 0.1
        chosen: list = []
        if self.hold_box.isChecked():
            chosen.append(ZeroOrderHold())
        if self.fixed_box.isChecked():
            chosen.append(
                ConstantVelocityKalman(
                    sigma_a=self.sigma_a_spin.value(), sigma_m=sigma_m
                )
            )
        if self.spline_box.isChecked():
            chosen.append(
                WindowedSplineRefit(
                    window_seconds=self.spline_window_spin.value() / 1000.0,
                    smoothing=self.spline_smoothing_spin.value(),
                )
            )
        return chosen

    def _run(self) -> tuple[list[Track], Track] | None:
        track = self._source_track()
        if track is None:
            self.status.setText("Load a recording to predict from first.")
            return None
        predictors = self.predictors()
        if not predictors:
            self.status.setText("Tick at least one variant.")
            return None
        config = self.config()
        try:
            reference = self._reference(track, config)
            return [simulate(track, p, config) for p in predictors], reference
        except (PredictionError, AnalysisError, ValueError) as error:
            self.status.setText(str(error))
            return None

    def show_in_player(self) -> None:
        result = self._run()
        if result is None:
            return
        predicted, reference = result
        for track in predicted:
            self.track_produced.emit(track)
        if self.truth_box.currentData() is None:
            # The offline bound isn't a stored recording; add it too, so the
            # Analysis panel's track comparison can score against it.
            self.track_produced.emit(reference)
        self.status.setText(
            "Shown in the player. Use the Analysis panel's track comparison "
            "to score these, and the reference, against a ground truth."
        )

    def tune_selected(self) -> None:
        """Fit the process noise of the fixed Kalman variant."""
        track = self._source_track()
        if track is None:
            self.status.setText("Load a recording to tune on first.")
            return
        if not self.fixed_box.isChecked():
            self.status.setText("Tick the Kalman variant to tune.")
            return

        config = self.config()
        try:
            reference = self._reference(track, config)
            result = tune(
                [track],
                lambda value: ConstantVelocityKalman(sigma_a=value),
                (0.5, 300.0),
                config,
                [reference],
                steps=8,
            )
            self.sigma_a_spin.setValue(result.value)
            label = f"fixed process noise {result.value:.2f}"
        except (PredictionError, AnalysisError, ValueError) as error:
            self.status.setText(str(error))
            return

        self.status.setText(
            f"Tuned {label} on {track.name} alone, so treat it as a starting "
            f"point rather than a setting that will hold across recordings."
        )

    def estimate_from_recording(self) -> None:
        track = self._source_track()
        if track is None:
            self.status.setText("Load a recording to measure first.")
            return
        try:
            sigma_m, sigma_a = estimate_noise(track)
        except (PredictionError, ValueError) as error:
            self.status.setText(str(error))
            return

        self.sigma_a_spin.setValue(min(max(sigma_a, 0.01), 10000.0))
        self.status.setText(
            f"Measured {sigma_m:.3f} cm of measurement noise and an acceleration "
            f"scale of {sigma_a:.1f} on {track.name}. These are seeds for tuning, "
            f"not tuned values."
        )
