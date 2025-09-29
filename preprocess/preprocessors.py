# lfp_analysis/data_access/filters.py
"""Example preprocessors (filters)."""
from typing import Any
import numpy as np
import scipy.signal as sps
from .base import Preprocessor
from lfp_analysis.registery import register

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
    def process(self, data: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        """
        return data[self.session_indexes]

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
        
    def process(self, data: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        """
        mean = data.mean(axis=self.dim, keepdims=True)
        std = data.std(axis=self.dim, keepdims=True)

        zscored = (data - mean) / std
        return zscored