import numpy as np
import matplotlib.pyplot as plt

from .base import Visualizer, plot_grand_average_with_ci, register_visualizer
from lfp_analysis.registry import register


def register_method(action_registry):
    def deco(name: str):
        def wrap(func):
            action_registry[name] = func
            return func
        return wrap
    return deco


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
    visualize_scheme = {}
    reg = register_method(visualize_scheme)


    def __init__(self, arguments):
        super().__init__(arguments)
        
 
    
    def visualize(self, data: list):
        scheme = self.arguments["scheme"]
        func = self.visualize_scheme[scheme]
        # bind the function to this instance before calling
        bound = func.__get__(self, self.__class__)
        bound(data)

    # def visualize(self, data: list):
    #     self.visualize_scheme[self.arguments["scheme"]](data)

    @reg("separate_figure")
    def visualize1(self, data: list):
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
        x_range = args.get("xrange",[])
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
            # features = feature["data"]  # shape: (n_sessions, n_channels, n_windows)
            features = feature["data"][:,:,:,0]
            fid = feature["id"]
            if x_range == []:
                n_sessions, n_channels, n_windows = features.shape
                x_idx = np.arange(n_windows)
            else:
                n_sessions, n_channels, n_windows = features.shape
                x_idx1 = np.arange(x_range[0], x_range[1], (x_range[1]-x_range[0])/n_windows)
                x_idx = x_idx1[:-1]

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
                    x=x_idx,
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

    @reg("same_figure")
    def visualizer2(self,data:list):
        # ----------------------------------------------------------------------
        # 1. Parse arguments
        # ----------------------------------------------------------------------
        args = self.arguments
        feature_name = args.get("FeatureName", "Feature")
        level = args.get("CI", None)
        m = args.get("movemeansize", 1)
        legends = args.get("legends", [])
        x_range = args.get("xrange",[])
        shape_subplot = args.get("ShapeSubplot", (1, len(data)))

        if isinstance(shape_subplot, list):
            shape_subplot = tuple(shape_subplot)

        # n_rows, n_cols = shape_subplot
        # n_total = n_rows * n_cols
        # n_inputs = len(data)

        # if n_inputs > n_total:
        #     raise ValueError(
        #         f"ShapeSubplot {shape_subplot} can show max {n_total} features, "
        #         f"but {n_inputs} were given."
        #     )
        # ----------------------------------------------------------------------
        # 1) Single figure/axes for all features
        # ----------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 5))

        # Optional: overall title/labels (can be adjusted per your taste)
        ax.set_title(f"All {feature_name}s")
        ax.set_xlabel("Window Index")
        ax.set_ylabel(feature_name)

        # ----------------------------------------------------------------------
        # 2) Plot every feature & channel on the SAME axes
        # ----------------------------------------------------------------------
        for i, feature in enumerate(data):
            # features: (n_sessions, n_channels, n_windows[, ...])
            # Use the first trailing component if present
            arr = feature["data"]
            if arr.ndim == 4:
                arr = arr[:, :, :, 0]
            features = arr  # (n_sessions, n_channels, n_windows)
            fid = feature["id"]

            n_sessions, n_channels, n_windows = features.shape

            # Build x for this feature (works even if windows differ across features)
            x_idx = np.linspace(x_range[0], x_range[1], n_windows, endpoint=False)

            # If legends not given or empty, create default per-channel legends
            if not legends:
                legends_for_feature = [f"Ch{ch+1}" for ch in range(n_channels)]
            else:
                # Reuse provided legends but allow arbitrary length using modulo
                legends_for_feature = [legends[ch % len(legends)] for ch in range(n_channels)]

            # Plot each channel’s grand average with CI on the SAME axes
            for ch in range(n_channels):
                plot_grand_average_with_ci(
                    data=features[:, ch, :],   # shape: (n_sessions, n_windows)
                    x=x_idx,
                    ax=ax,
                    label=f"{fid} | {legends_for_feature[ch]}",
                    m=m,
                    level=level,
                )

        # One legend for everything
        ax.legend()
        fig.tight_layout()
        plt.show()
