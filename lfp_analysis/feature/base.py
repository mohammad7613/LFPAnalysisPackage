# lfp_analysis/business_logic/base.py
from typing import Callable, Dict, Type, Any
import numpy as np
from abc import ABC, abstractmethod

# FEATURE_REGISTRY: Dict[str, Type["FeatureFunction"]] = {}

# StatisticalTest_REGISTRY: Dict[str, Type["StatisticalTest"]] = {}


# def register_feature(name: str) -> Callable:
#     def decorator(cls: Type["FeatureFunction"]) -> Type["FeatureFunction"]:
#         FEATURE_REGISTRY[name] = cls
#         return cls
#     return decorator


# def register_statistical_test(name: str) -> Callable:
#     def decorator(cls: Type["StatisticalTest"]) -> Type["StatisticalTest"]:
#         StatisticalTest_REGISTRY[name] = cls
#         return cls
#     return decorator


class FeatureFunction(ABC):
    """
    Abstract feature function.

    Expected input shapes:
    - single signal: 1D array (n_samples,)
    - or full dataset: (n_sessions, n_channels, n_epochs, n_samples)

    Concrete implementations should document which inputs they accept.
    """

    @abstractmethod
    def compute(self, signal: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        Compute the feature.

        If `signal` is 1D: return a float (or 1-element array).
        If `signal` is a full dataset: return array shaped (n_sessions, n_epochs) or similar.
        """
        raise NotImplementedError


class StatisticalTest(ABC):
    """
    Abstract base class for statistical tests or analyses.

    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Dictionary containing the data for analysis.
        Keys depend on the test:
          - For group comparisons: group names mapped to 1D arrays.
          - For regression: 'X' and 'y' keys with corresponding arrays.
    **kwargs : Additional keyword arguments specific to the test.

    Returns
    -------
    Dict[str, Any]
        Dictionary with test results (e.g., statistic, p-value, model coefficients).
    """

    @abstractmethod
    def compare(self, data: Dict[str, np.ndarray], **kwargs) -> Dict[str, Any]:
        pass
