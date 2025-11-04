import numpy as np
import matplotlib.pyplot as plt

from .base import Visualizer, plot_grand_average_with_ci, register_visualizer
from lfp_analysis.registry import register

@register("visualizers","trial_dynamic_feature_visualizer")
class TrialDynamicFeatureVisualizer(Visualizer):
    """
    Plots the grand average of a feature over trials for different stimulus types.

    X-axis: Trial index
    Y-axis: Feature value with confidence interval over sessions
    """
    def __init__(self, arguments):
        super().__init__(arguments)
    def visualize(self, data: list):
        """
        Visualize multiple features in a subplot grid.
        
        Parameters
        ----------
        data : list of dict
            Each dict has {"id": feature_id, "data": feature_array}.
            feature_array is expected to have shape (sessions, channels, epochs, ...)
        """

        # ----------------------------------------------------------------------
        # 1. Parse arguments
        # ----------------------------------------------------------------------
        args = self.arguments
        feature_name = args.get("FeatureName", "Feature")
        level = args.get("CI",None)
        m = args.get("movemeansize", 1)
        title = args.get("title","")
        
        legends = args.get("legends", [])
        shape_subplot = args.get("ShapeSubplot", (1, len(data)))

        # make sure shape is a tuple (YAML may parse as list)
        if isinstance(shape_subplot, list):
            shape_subplot = tuple(shape_subplot)

        n_rows, n_cols = shape_subplot
        n_total = n_rows * n_cols
        n_inputs = len(data)

        if n_inputs > n_total:
            raise ValueError(f"ShapeSubplot {shape_subplot} can show max {n_total} features, "
                            f"but {n_inputs} were given.")

        # ----------------------------------------------------------------------
        # 2. Create figure and axes
        # ----------------------------------------------------------------------
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        axes = np.array(axes).reshape(-1)  # flatten in case of 2D array of axes

        # ----------------------------------------------------------------------
        # 3. Plot each feature in its own subplot
        # ----------------------------------------------------------------------
        for i, feature in enumerate(data):
            ax = axes[i]
            features = feature["data"]
            fid = feature["id"]

            # assuming features has shape (sessions, channels, epochs, ...)
            n_epochs = features.shape[2] if features.ndim >= 3 else features.shape[-1]
            trial_number = np.arange(n_epochs)

            ax.set_title(f"{feature_name}: {title}")
            ax.set_xlabel("Trial Index")
            ax.set_ylabel(feature_name)

            for k, label in enumerate(legends):
                # plot_grand_average_with_ci is your helper
                plot_grand_average_with_ci(
                    features[:, k, :], trial_number, ax=ax, label=label, m= m, level= level
                )

            ax.legend()

        # Hide extra subplots (if ShapeSubplot bigger than inputs)
        for j in range(n_inputs, n_total):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()      

@register("visualizers", "time_dynamic_feature_visualizer")
class TimeDynamicFeatureVisualizer(Visualizer):
    """
    Plots the grand average of a feature evolving over time windows.

    X-axis: Window index (time)
    Y-axis: Feature value with confidence interval over sessions
    """

    def __init__(self, arguments):
        super().__init__(arguments)

    def visualize(self, data: list):
        """
        Visualize the temporal dynamics of one or more features.

        Parameters
        ----------
        data : list of dict
            Each dict must have:
                {
                    "id": <feature_id>,
                    "data": np.ndarray of shape (n_sessions, n_channels, n_windows)
                }
        """

        # ----------------------------------------------------------------------
        # 1. Parse arguments
        # ----------------------------------------------------------------------
        args = self.arguments
        feature_name = args.get("FeatureName", "Feature")
        level = args.get("CI", None)
        m = args.get("movemeansize", 1)
        legends = args.get("legends", [])
        shape_subplot = args.get("ShapeSubplot", (1, len(data)))

        if isinstance(shape_subplot, list):
            shape_subplot = tuple(shape_subplot)

        n_rows, n_cols = shape_subplot
        n_total = n_rows * n_cols
        n_inputs = len(data)

        if n_inputs > n_total:
            raise ValueError(
                f"ShapeSubplot {shape_subplot} can show max {n_total} features, "
                f"but {n_inputs} were given."
            )

        # ----------------------------------------------------------------------
        # 2. Create figure and axes
        # ----------------------------------------------------------------------
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        axes = np.array(axes).reshape(-1)

        # ----------------------------------------------------------------------
        # 3. Plot each feature in its own subplot
        # ----------------------------------------------------------------------
        for i, feature in enumerate(data):
            ax = axes[i]
            features = feature["data"]  # shape: (n_sessions, n_channels, n_windows)
            fid = feature["id"]

            n_sessions, n_channels, n_windows = features.shape
            window_idx = np.arange(n_windows)

            ax.set_title(f"{feature_name}: {fid}")
            ax.set_xlabel("Window Index")
            ax.set_ylabel(feature_name)

            # If legends not given, use channel indices
            if not legends:
                legends = [f"Ch{ch+1}" for ch in range(n_channels)]

            # Plot each channel’s grand average with CI
            for ch in range(n_channels):
                plot_grand_average_with_ci(
                    data=features[:, ch, :],
                    x=window_idx,
                    ax=ax,
                    label=legends[ch] if ch < len(legends) else f"Ch{ch+1}",
                    m=m,
                    level=level,
                )

            ax.legend()

        # Hide any unused subplots
        for j in range(n_inputs, n_total):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()