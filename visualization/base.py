# visualization/plot_base.py
from abc import ABC, abstractmethod

from typing import Any
import matplotlib.pyplot as plt
from ..feature import FeatureFunction
from typing import Callable, Dict, Type
import numpy as np
import matplotlib.pyplot as plt

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
    def visualize(self,data : dict):
        ### It contains a dictionary with "feature" 
        ### The additional arugment such as labels and 
        # channels or the figure sizes should be defined in arguments dictionary ###
        pass
    
    def input_checker(self,data: dict)-> bool:
        return "features" in data

    def run(self,data:dict):
        if self.input_checker(data):
            self.visualize(data)
        else:
            raise KeyError("The input dictionary must contain some key feature")   
            

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