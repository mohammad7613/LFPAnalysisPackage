# lfp_analysis/data_access/base.py
"""
Data access base: defines Preprocessor ABC and a registry for plugins.
"""
from typing import Callable, Dict, Type
import numpy as np
# PREPROCESSOR_REGISTRY: Dict[str, Type["Preprocessor"]] = {}
# def register_preprocessor(name: str) -> Callable:
#     """Decorator to register a Preprocessor class with a name.
#     Usage:
#     @register_preprocessor("bandpass_filter")
#     class BandpassFilter(Preprocessor): ...
#     """
#     def decorator(cls: Type["Preprocessor"]) -> Type["Preprocessor"]:
#         PREPROCESSOR_REGISTRY[name] = cls
#         return cls
#     return decorator
class Preprocessor:
    """Abstract preprocessor.
    A preprocessor should ac>cept and return numpy arrays with the following
    canonical shapes:
    - Input: `np.ndarray` with shape (n_sessions, n_channels, n_epochs,
    n_samples)
    - Output: same shape
    """
    def process(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Apply preprocessing to the entire dataset.
        Parameters
        ----------
        data : np.ndarray
        LFP data with shape (sessions, channels, epochs, samples)
        Returns
        -------
        np.ndarray
        Processed data with the same shape.
        """
        raise NotImplementedError