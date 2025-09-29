import numpy as np
import matplotlib.pyplot as plt

from .base import Visualizer, plot_grand_average_with_ci, register_visualizer
from lfp_analysis.registery import register

@register("visualizers","trial_dyanmic_feature_visualizer")
class TrialDynamicFeatureVisualizer(Visualizer):
    """
    Plots the grand average of a feature over trials for different stimulus types.

    X-axis: Trial index
    Y-axis: Feature value with confidence interval over sessions
    """
    def __init__(self, arguments):
        super().__init__(arguments)
    def visualize(self,data: dict):
        """
        Compute and plot the trial-by-trial feature dynamics.
        """
        features = data["features"]
        argument = self.arguments
        legends = argument["legends"]
        n_epochs = features.shape[2]
        plt.figure(figsize=(12, 6))
        plt.title(f'{argument["FeatureName"]} Grand Average over Trials')
        plt.xlabel('Trial Index')
        plt.ylabel(argument["FeatureName"])
        for k,label in enumerate(legends):
            plot_grand_average_with_ci(features[:,k,:],np.arange(n_epochs),label=label)
        plt.legend()
        plt.show()    


@register("visualizers","time_dyanmic_feature_visualizer")
class TimeDynamicFeatureVisualizer(Visualizer):
    """
    Plots the grand average of a feature over trials for different stimulus types.

    X-axis: Trial index
    Y-axis: Feature value with confidence interval over sessions
    """
    def __init__(self, arguments):
        super().__init__(arguments)
    def visualize(self,data: dict):
        """
        Compute and plot the trial-by-trial feature dynamics.
        """
        features = data["features"]
        argument = data["args"]
        n_sessions, n_window = features.shape
        plt.figure(figsize=(12, 6))
        plt.title(f'{argument["FeatureName"]} Grand Average over Trials')
        plt.xlabel('Trial Index')
        plt.ylabel(argument["FeatureName"])
        plot_grand_average_with_ci(features,np.arange(n_window))