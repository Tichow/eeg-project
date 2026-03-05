"""
Widget de sélection des options de traitement du signal.
"""

from __future__ import annotations

from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QCheckBox
from PyQt5.QtCore import pyqtSignal


class ProcessingSelector(QGroupBox):
    """
    QGroupBox avec checkboxes pour les options de traitement :
      - Passe-bande (1–50 Hz)
      - Notch (50 Hz)
      - CAR (Common Average Reference)
      - Affichage SNR Alpha

    Signal :
        processing_changed(dict) : {'bandpass': bool, 'notch': bool, 'car': bool, 'show_snr': bool}
    """

    processing_changed = pyqtSignal(dict)

    def __init__(
        self,
        initial_bandpass: bool = True,
        initial_notch: bool    = True,
        initial_car: bool      = False,
        initial_show_snr: bool = False,
        parent=None,
    ) -> None:
        super().__init__('Traitement', parent=parent)

        self._cb_bandpass = QCheckBox('Passe-bande (1–50 Hz)')
        self._cb_notch    = QCheckBox('Notch 50 Hz')
        self._cb_car      = QCheckBox('CAR')
        self._cb_snr      = QCheckBox('SNR Alpha')

        self._cb_bandpass.setChecked(initial_bandpass)
        self._cb_notch.setChecked(initial_notch)
        self._cb_car.setChecked(initial_car)
        self._cb_snr.setChecked(initial_show_snr)

        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.addWidget(self._cb_bandpass)
        layout.addWidget(self._cb_notch)
        layout.addWidget(self._cb_car)
        layout.addWidget(self._cb_snr)

        self._cb_bandpass.stateChanged.connect(self._emit)
        self._cb_notch.stateChanged.connect(self._emit)
        self._cb_car.stateChanged.connect(self._emit)
        self._cb_snr.stateChanged.connect(self._emit)

    def _emit(self, _: int = 0) -> None:
        self.processing_changed.emit(self.state)

    @property
    def bandpass(self) -> bool:
        return self._cb_bandpass.isChecked()

    @property
    def notch(self) -> bool:
        return self._cb_notch.isChecked()

    @property
    def car(self) -> bool:
        return self._cb_car.isChecked()

    @property
    def show_snr(self) -> bool:
        return self._cb_snr.isChecked()

    @property
    def state(self) -> dict:
        return {
            'bandpass': self.bandpass,
            'notch':    self.notch,
            'car':      self.car,
            'show_snr': self.show_snr,
        }
