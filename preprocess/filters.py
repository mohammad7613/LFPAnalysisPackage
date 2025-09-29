# lfp_analysis/data_access/filters.py
"""Example preprocessors (filters)."""
from typing import Any
import numpy as np
import scipy.signal as sps
from .base import Preprocessor
from lfp_analysis.registery import register

@register("preprocessors","bandpass_filter")
class BandpassFilter(Preprocessor):
    """
    Simple zero-phase Butterworth bandpass applied to each channel/epoch
    independently.
    Input: (sessions, channels, epochs, samples)
    Output: same shape
    """
    def __init__(self, low: float, high: float, sfreq: float, order: int = 4):
        self.low = low
        self.high = high
        self.sfreq = sfreq
        self.order = order
    def _filter_1d(self, signal: np.ndarray) -> np.ndarray:
        b, a = sps.butter(self.order, [self.low, self.high], btype="band",
                         fs=self.sfreq)
        return sps.filtfilt(b, a, signal)
    
    def process(self, data: np.ndarray, **kwargs: Any) -> np.ndarray:
        sessions, channels, epochs, samples = data.shape
        out = np.empty_like(data)
        for s in range(sessions):
            for ch in range(channels):
                for e in range(epochs):
                    out[s, ch, e, :] = self._filter_1d(data[s, ch, e, :])
        return out