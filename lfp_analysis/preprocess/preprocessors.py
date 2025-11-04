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