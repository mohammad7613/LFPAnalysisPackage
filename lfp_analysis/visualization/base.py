# visualization/plot_base.py
from abc import ABC, abstractmethod

from typing import Any
import matplotlib.pyplot as plt
from ..feature import FeatureFunction
from typing import Callable, Dict, Type
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d  # for moving mean
from scipy.stats import t

VISUALIZER_REGISTRY: Dict[str, Type["Visualizer"]] = {}


def register_visualizer(name: str) -> Callable:
    """Decorator to register a Visualizer class with a name."""
    def decorator(cls: Type["Visualizer"]) -> Type["Visualizer"]:
        VISUALIZER_REGISTRY[name] = cls
        return cls
    return decorator

class Visualizer(ABC):
    def __init__(self, arguments: Dict):
        super().__init__()
        self.arguments = arguments
    @abstractmethod
    def visualize(self,data : list):
        ### It contains a dictionary with "feature" 
        ### The additional arugment such as labels and 
        # channels or the figure sizes should be defined in arguments dictionary ###
        pass
    
    def input_checker(self,data: list)-> bool:
        ### It needs to implemented in each subclass ###
        return True

    def run(self,data:list):
        if self.input_checker(data):
            self.visualize(data)
        else:
            raise KeyError("The input list does not have a correct structure")


def plot_grand_average_with_ci(
    data: np.ndarray, x: np.ndarray,label: str, color: str=None):
    """
    Plots the grand average of a 2D array with a confidence interval.
    This is a utility function intended for use by Visualizer classes.

    Parameters
    ----------
    data : np.ndarray
        2D array of data, where rows are observations (e.g., sessions) and
        columns are data points (e.g., trial indices or time points).
    x : np.ndarray
        The x-axis values for the plot.
    color : str
        The color to use for the plot.
    label : str
        The label for the plot legend.
    """
    n_observations = data.shape[0]
    grand_avg = np.nanmean(data, axis=0)
    ci = np.nanstd(data, axis=0) / np.sqrt(n_observations) * 1.96
    if color == None:
        plt.plot(x, grand_avg, label=label)
        plt.fill_between(x, grand_avg - ci, grand_avg + ci, alpha=0.3)
    else:
        plt.plot(x, grand_avg, color=color, label=label)
        plt.fill_between(x, grand_avg - ci, grand_avg + ci, color=color, alpha=0.3)

def compute_error(data, axis=0, level=None):
    """
    Compute either SEM or confidence interval half-width along a given axis.

    Parameters
    ----------
    data : array-like
        Input data (e.g., subjects × timepoints).
    axis : int
        Axis along which to compute the mean and error (default: 0).
    level : float or None
        Confidence level (e.g., 0.95 for 95% CI). 
        If None → returns SEM.
        If 0 < level < 1 → returns CI half-width at that confidence level.

    Returns
    -------
    mean : ndarray
        Mean across the specified axis.
    err  : ndarray
        SEM or CI half-width, same shape as mean.
    """

    data = np.asarray(data, dtype=np.float64)
    n = np.sum(~np.isnan(data), axis=axis)
    mean = np.nanmean(data, axis=axis)
    sem = np.nanstd(data, axis=axis, ddof=1) / np.sqrt(n)

    if level is None:
        # Just SEM
        return mean, sem
    else:
        # Confidence interval (two-tailed)
        df = n - 1
        # For large n, t and z are nearly identical
        tcrit = t.ppf(1 - (1 - level) / 2, df)
        err = sem * tcrit
        return mean, err




def plot_grand_average_with_ci(
    data: np.ndarray,
    x: np.ndarray,
    label: str,
    color: str = None,
    ax=None,
    m: int = 20, #window size parameter 
    level: float = None # confidence interval
):
    """
    Plot the grand average of a 2D array with a 95% confidence interval.

    Parameters
    ----------
    data : np.ndarray
        2D array of data (n_observations × n_points), e.g., sessions × trials.
    x : np.ndarray
        X-axis values.
    label : str
        Label for the legend.
    color : str, optional
        Line color (default: automatic).
    ax : matplotlib.axes.Axes, optional
        The subplot axis to draw on. If None, uses the current global axis.
    """

    n_observations = data.shape[0]


 # ---- Added movmean filter ----
    if m > 1:
    # data shape: (n_observations, n_points), smooth along time axis (axis=1)
        data_smoothed = uniform_filter1d(data.astype(np.float64), size=m, axis=1, mode="nearest")
        temp = m//2
        data_smoothed = data_smoothed[:,temp:-temp]
        x = x[temp:-temp]
    else:
        data_smoothed = data.astype(np.float64)

    # # Now compute grand average and CI from the smoothed observations
    grand_avg, ci = compute_error(data=data_smoothed, level= level)
    # grand_avg = np.nanmean(data_smoothed, axis=0)
    # ci = np.nanstd(data_smoothed, axis=0, ddof=0) / np.sqrt(n_observations) * 1.96
    # alpha = 0.05
    # df = n_observations - 1
    # t_crit = t.ppf(1 - alpha/2, df)

    # sem = np.nanstd(data_smoothed, axis=0, ddof=1) / np.sqrt(n_observations)
    # ci = sem * 1

    # Choose the axis to draw on
    ax = ax or plt.gca()

    # Plot the mean and fill CI on that axis
    if color is None:
        ax.plot(x, grand_avg, label=label)
        ax.fill_between(x, grand_avg - ci, grand_avg + ci, alpha=0.3)
    else:
        ax.plot(x, grand_avg, color=color, label=label)
        ax.fill_between(x, grand_avg - ci, grand_avg + ci, color=color, alpha=0.3)

