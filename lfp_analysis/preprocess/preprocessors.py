# lfp_analysis/data_access/filters.py
"""Example preprocessors (filters)."""
from typing import Any
import numpy as np
import scipy.signal as sps
from .base import Preprocessor
from lfp_analysis.registry import register

@register("preprocessors","selectsession")
class sessionselect(Preprocessor):
    """
    selection sessions
    independently.
    Input: numpy array with shape (Number of sessions,...)
    Output: numpy array with shape(Numbr of selected sessions,....)
    """
    def __init__(self, session_indexs):
        self.session_indexes = session_indexs
    def process(self, signal: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        """
        return signal[self.session_indexes]

@register("preprocessors","zscore")
class zscore(Preprocessor):
    """
    selection sessions
    independently.
    Input: numpy array with shape (Number of sessions,...)
    Output: numpy array with shape(Numbr of selected sessions,....)
    """
    def __init__(self, dim):
        self.dim = dim
        
    def process(self, signal: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        """
        mean = signal.mean(axis=self.dim, keepdims=True)
        std = signal.std(axis=self.dim, keepdims=True)

        zscored = (signal - mean) / std
        return zscored


@register("preprocessors","selecttime")
class timeselect(Preprocessor):
    """
    selection sessions
    independently.
    Input: numpy array with shape (Number of sessions,...)
    Output: numpy array with shape(Numbr of selected sessions,....)
    """
    def __init__(self, start_index, end_index):
        self.start_index = start_index
        self.end_index = end_index
    def process(self, signal: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        """
        return signal[:,:,:,self.start_index:self.end_index]


@register("preprocessors","removesession")
class sessionselect(Preprocessor):
    """
    selection sessions
    independently.
    Input: numpy array with shape (Number of sessions,...)
    Output: numpy array with shape(Numbr of selected sessions,....)
    """
    def __init__(self, session_indexs):
        self.session_indexes = session_indexs
    def process(self, signal: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        """
        number_session = signal.shape[0]
        all_sessions = np.arange(number_session)
        sessinos_selected = np.setdiff1d(all_sessions,self.session_indexes)
        return signal[sessinos_selected]



@register("preprocessors","baselinecorrection")
class baselinecorrection(Preprocessor):
    """
    selection sessions
    independently.
    Input: numpy array with shape (Number of sessions,...)
    Output: numpy array with shape(Numbr of selected sessions,....)
    """
    def __init__(self, baseline_window, sfreq=2000):
        # baseline_window expects (start, end) in seconds or sample indices
        # freq_band expects (low, high) in Hz; optional
        self.baseline_window = baseline_window
        self.sfreq = sfreq
        
    def process(self, signal: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        """
        baseline_corrected = self.baseline_correct(
            signal,
            sfreq=self.sfreq,
            baseline=self.baseline_window,
            mode="mean",
            keep_baseline_stats=False,
            nan_policy="propagate",
        )
        return baseline_corrected
    
    def baseline_correct(self,
        signal: np.ndarray,
        sfreq: float,
        baseline: tuple = (-0.47, 0.0),
        mode: str = "mean",
        keep_baseline_stats: bool = False,
        nan_policy: str = "propagate",
    ):
        """
        Baseline-correct epoched LFP.

        Parameters
        ----------
        signal : np.ndarray
            Shape (n_sessions, n_channels, n_epochs, n_samples).
        trial_period : tuple
            (t_start, t_end) in seconds for each epoch, e.g. (-0.47, 1.0).
        sfreq : float
            Sampling frequency in Hz.
        baseline : tuple
            (b_start, b_end) in seconds, relative to trial_period.
            Example: (-0.47, 0) for pre-stim baseline.
        mode : str
            "mean" or "median".
        keep_baseline_stats : bool
            If True, return (corrected, baseline_center, baseline_scale).
        nan_policy : str
            "propagate" -> use np.mean/median (NaNs yield NaN baseline)
            "omit"      -> use np.nanmean/nanmedian

        Returns
        -------
        corrected : np.ndarray
            Baseline-corrected signal, same shape as input.
        (optional) baseline_center : np.ndarray
            Baseline mean/median used for correction, shape (n_sessions, n_channels, n_epochs, 1).
        """
        assert signal.ndim == 4, "signal should be 4D: (sessions, channels, epochs, samples)"
        b_start, b_end = baseline

        n_samples = signal.shape[-1]
        # Time vector for the epoch
        times = np.arange(n_samples) / sfreq + b_start

        # Baseline mask
        b_mask = (times >= b_start) & (times <= b_end)
        if not np.any(b_mask):
            raise ValueError(
                f"No samples found in baseline window {baseline} "
                f"within trial period"
            )

        # Choose reduction function
        if nan_policy == "omit":
            if mode == "mean":
                reducer = np.nanmean
            elif mode == "median":
                reducer = np.nanmedian
            else:
                raise ValueError("mode must be 'mean' or 'median'")
        else:
            if mode == "mean":
                reducer = np.mean
            elif mode == "median":
                reducer = np.median
            else:
                raise ValueError("mode must be 'mean' or 'median'")

        # Compute baseline center per session-channel-epoch
        # keep dims for broadcasting over time
        baseline_center = reducer(signal[..., b_mask], axis=-1, keepdims=True)

        corrected = signal - baseline_center

        if keep_baseline_stats:
            return corrected, baseline_center
        return corrected