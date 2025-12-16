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


@register("visualizers", "composite_visualizer")
class CompositeVisualizer(Visualizer):
    """Compose multiple child visualizers using a shared figure/layout."""

    def __init__(self, arguments):
        super().__init__(arguments)

    def visualize(self, data: list, payload=None):
        payload = payload or {}
        spec = payload.get("spec")
        children = getattr(spec, "children", []) if spec else []
        if not children:
            return

        runner = payload.get("run_child")
        if runner is None:
            raise RuntimeError("CompositeVisualizer requires access to run_child callback")

        layout = self.arguments.get("layout", {})
        rows = int(layout.get("rows", 1))
        cols = int(layout.get("cols", max(1, len(children))))
        share_figure = self.arguments.get("share_figure", True)
        figsize = tuple(self.arguments.get("figsize", (6 * cols, 4 * rows)))
        show = self.arguments.get("show", True)

        if share_figure:
            fig, axes = plt.subplots(rows, cols, figsize=figsize)
            axes = np.atleast_1d(axes).reshape(-1)
            title = self.arguments.get("title")
            if title:
                fig.suptitle(title)
        else:
            fig = None
            axes = []

        for idx, child in enumerate(children):
            child_payload = dict(payload)
            child_payload.pop("spec", None)
            child_payload["defer_show"] = share_figure
            if share_figure:
                target_axis = axes[min(idx, len(axes) - 1)]
                child_payload["axis"] = target_axis
                child_payload["figure"] = fig
            runner(child, child_payload)

        if share_figure:
            if self.arguments.get("tight_layout", True):
                fig.tight_layout()
            if show and not self.arguments.get("defer_show", False):
                plt.show()


@register("visualizers","trial_dynamic_feature_visualizer")
class TrialDynamicFeatureVisualizer(Visualizer):
    """
    Plots the grand average of a feature over trials for different stimulus types.

    X-axis: Trial index
    Y-axis: Feature value with confidence interval over sessions
    """
    def __init__(self, arguments):
        super().__init__(arguments)
    def visualize(self, data: list, payload=None):
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
        payload = payload or {}
        provided_axis = payload.get("axis")
        provided_fig = payload.get("figure")
        defer_show = payload.get("defer_show", False)
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
        n_inputs = len(data)

        if provided_axis is None:
            n_total = n_rows * n_cols
            if n_inputs > n_total:
                raise ValueError(
                    f"ShapeSubplot {shape_subplot} can show max {n_total} features, "
                    f"but {n_inputs} were given."
                )
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
            axes = np.array(axes).reshape(-1)
        else:
            fig = provided_fig or provided_axis.figure
            axes = np.array([provided_axis])

        axes_count = len(axes)

        # ----------------------------------------------------------------------
        # 3. Plot each feature in its own subplot
        # ----------------------------------------------------------------------
        for i, feature in enumerate(data):
            ax = axes[i] if i < axes_count else axes[-1]
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

        if provided_axis is None:
            n_total = n_rows * n_cols
            for j in range(n_inputs, n_total):
                fig.delaxes(axes[j])
            plt.tight_layout()
            if not defer_show:
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
        
 
    
    def visualize(self, data: list, payload=None):
        scheme = self.arguments.get("scheme","same_figure")
        func = self.visualize_scheme[scheme]
        # bind the function to this instance before calling
        bound = func.__get__(self, self.__class__)
        bound(data, payload or {})

    # def visualize(self, data: list):
    #     self.visualize_scheme[self.arguments["scheme"]](data)

    @reg("separate_figure")
    def visualize1(self, data: list, payload=None):
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
        payload = payload or {}
        feature_name = args.get("FeatureName", "Feature")
        level = args.get("CI", None)
        m = args.get("movemeansize", 1)
        legends = args.get("legends", [])
        x_range = args.get("xrange",[])
        shape_subplot = args.get("ShapeSubplot", (1, len(data)))

        if isinstance(shape_subplot, list):
            shape_subplot = tuple(shape_subplot)

        n_rows, n_cols = shape_subplot
        n_inputs = len(data)

        provided_axis = payload.get("axis")
        provided_fig = payload.get("figure")
        defer_show = payload.get("defer_show", False)

        if provided_axis is None:
            n_total = n_rows * n_cols
            if n_inputs > n_total:
                raise ValueError(
                    f"ShapeSubplot {shape_subplot} can show max {n_total} features, "
                    f"but {n_inputs} were given."
                )
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
            axes = np.array(axes).reshape(-1)
        else:
            fig = provided_fig or provided_axis.figure
            axes = np.array([provided_axis])
            n_total = len(axes)

        # ----------------------------------------------------------------------
        # 3. Plot each feature in its own subplot
        # ----------------------------------------------------------------------
        for i, feature in enumerate(data):
            ax = axes[i] if i < n_total else axes[-1]
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
            if legends:
                legend_labels = legends
            else:
                legend_labels = [f"Ch{ch+1}" for ch in range(n_channels)]

            # Plot each channel’s grand average with CI
            for ch in range(n_channels):
                plot_grand_average_with_ci(
                    data=features[:, ch, :],
                    x=x_idx,
                    ax=ax,
                    label=legend_labels[ch] if ch < len(legend_labels) else f"Ch{ch+1}",
                    m=m,
                    level=level,
                )

            ax.legend()

        if provided_axis is None:
            for j in range(n_inputs, n_total):
                fig.delaxes(axes[j])
            plt.tight_layout()
            if not defer_show:
                plt.show()

    @reg("same_figure")
    def visualizer2(self,data:list, payload=None):
        # ----------------------------------------------------------------------
        # 1. Parse arguments
        # ----------------------------------------------------------------------
        args = self.arguments
        payload = payload or {}
        feature_name = args.get("FeatureName", "Feature")
        level = args.get("CI", None)
        m = args.get("movemeansize", 1)
        title = args.get("title","")
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
        provided_axis = payload.get("axis")
        provided_fig = payload.get("figure")
        defer_show = payload.get("defer_show", False)

        # ----------------------------------------------------------------------
        # 1) Single figure/axes for all features
        # ----------------------------------------------------------------------
        if provided_axis is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            ax = provided_axis
            fig = provided_fig or ax.figure

        # Optional: overall title/labels (can be adjusted per your taste)
        ax.set_title(title)
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
        if provided_axis is None:
            fig.tight_layout()
            if not defer_show:
                plt.show()
