import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, ttest_rel
from typing import Optional, Tuple

from .base import Visualizer, plot_grand_average_with_ci, register_visualizer
from lfp_analysis.registry import register
from lfp_analysis.utils.arrays_ops import ArrayReducer, Selection


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
            width_ratios = layout.get("width_ratios")
            height_ratios = layout.get("height_ratios")
            fig = plt.figure(figsize=figsize)
            grid = fig.add_gridspec(rows, cols, width_ratios=width_ratios, height_ratios=height_ratios)
            axes = [fig.add_subplot(grid[row, col]) for row in range(rows) for col in range(cols)]
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


@register("visualizers", "box_comparison_visualizer")
class BoxComparisonVisualizer(Visualizer):
    """Render two feature distributions as box plots and annotate p-values."""

    def __init__(self, arguments):
        super().__init__(arguments)
        self.default_reducer = self._build_reducer(arguments.get("reducer"))
        if self.default_reducer is None:
            legacy_axis = arguments.get("aggregation_axis")
            self.default_reducer = ArrayReducer(default_aggregation_axes=legacy_axis)
        self.input_reducers = [
            self._build_reducer(cfg) or self.default_reducer for cfg in arguments.get("input_reducers", [])
        ]
        stats_cfg = arguments.get("stat_test", {})
        self.paired = bool(stats_cfg.get("paired", False))
        self.nan_policy = stats_cfg.get("nan_policy", "omit")
        self.equal_var = bool(arguments.get("equal_var", False))

    def visualize(self, data: list, payload=None):
        payload = payload or {}
        if len(data) != 2:
            raise ValueError("box_comparison_visualizer expects exactly two feature inputs")

        args = self.arguments
        provided_axis = payload.get("axis")
        provided_fig = payload.get("figure")
        defer_show = payload.get("defer_show", False)

        if provided_axis is None:
            figsize = tuple(args.get("figsize", (5, 5)))
            fig, ax = plt.subplots(figsize=figsize)
        else:
            ax = provided_axis
            fig = provided_fig or ax.figure

        labels = args.get("labels", [data[0]["id"], data[1]["id"]])
        colors = args.get("colors", ["#1f77b4", "#ff7f0e"])
        alpha = float(args.get("alpha", 0.05))

        plot_samples = []
        stat_samples = []
        for idx, feature in enumerate(data):
            reducer = self.input_reducers[idx] if idx < len(self.input_reducers) else self.default_reducer
            plot_vals, stat_vals = self._prepare_sample(feature["data"], reducer)
            plot_samples.append(plot_vals)
            stat_samples.append(stat_vals)

        if any(sample.size == 0 for sample in plot_samples):
            raise ValueError("Box plot inputs must contain at least one numeric value")

        box = ax.boxplot(plot_samples, tick_labels=labels, patch_artist=True, notch=args.get("notch", False))
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(args.get("box_alpha", 0.7))

        # Optional swarm jitter overlay for more detail
        if args.get("show_points", False):
            jitter = args.get("jitter", 0.08)
            for idx, sample in enumerate(plot_samples, start=1):
                x = np.random.normal(loc=idx, scale=jitter, size=sample.size)
                ax.scatter(x, sample, s=args.get("point_size", 15), color=colors[idx-1], alpha=0.5)

        if self.paired:
            stat, p_value = self._paired_test(stat_samples[0], stat_samples[1])
        else:
            stat, p_value = ttest_ind(stat_samples[0], stat_samples[1], equal_var=self.equal_var, nan_policy=self.nan_policy)
        annotation = self._format_significance(p_value)

        y_max = np.nanmax([plot_samples[0].max(), plot_samples[1].max()])
        y_min = np.nanmin([plot_samples[0].min(), plot_samples[1].min()])
        span = y_max - y_min if y_max > y_min else 1.0
        line_height = y_max + span * 0.1
        line_offset = span * 0.05

        ax.plot([1, 1, 2, 2], [line_height, line_height + line_offset, line_height + line_offset, line_height],
                color=args.get("annotation_color", "black"), linewidth=1.2)

        if p_value < alpha:
            ax.text(1.5, line_height + line_offset * 1.2, annotation, ha="center", va="bottom",
                    fontsize=args.get("annotation_size", 12))
        else:
            ax.text(1.5, line_height + line_offset * 1.2, f"p = {p_value:.3f}", ha="center", va="bottom",
                    fontsize=args.get("annotation_size", 10))

        ax.set_title(args.get("title", "Group comparison"))
        ax.set_ylabel(args.get("ylabel", "Value"))

        grid_flag = args.get("grid", False)
        if grid_flag:
            ax.grid(True, axis="y", alpha=0.3)

        if provided_axis is None:
            fig.tight_layout()
            if not defer_show:
                plt.show()

    def _prepare_sample(self, array_like, reducer: ArrayReducer):
        arr = np.asarray(array_like, dtype=float)
        reduced = reducer.prepare(arr, flatten=not self.paired) if reducer is not None else arr
        if self.paired:
            stat_values = reduced.reshape(reduced.shape[0], -1)
            plot_values = stat_values.reshape(-1)
        else:
            stat_values = plot_values = reduced.reshape(-1)
        if plot_values.size == 0:
            raise ValueError("Box plot inputs must contain at least one numeric value after reduction")
        if np.any(np.isnan(plot_values)):
            raise ValueError("Input data contains NaN values after reduction")
        return plot_values, stat_values

    def _paired_test(self, sample_a: np.ndarray, sample_b: np.ndarray):
        if sample_a.shape != sample_b.shape:
            raise ValueError("Paired comparison requires samples with identical shapes")
        # sample_a = sample_a.reshape(sample_a.shape[0], -1)
        # sample_b = sample_b.reshape(sample_b.shape[0], -1)
        assert sample_a.shape[1] == 1, "Paired samples must have the same number of observations"
        assert sample_b.shape[1] == 1, "Paired samples must have the same number of observations"   
        stat, p_values = ttest_rel(sample_a, sample_b, axis=0, nan_policy=self.nan_policy)
        aggregated_stat = np.mean(stat)
        aggregated_p = np.mean(p_values)
        return aggregated_stat, aggregated_p

    def _build_reducer(self, cfg: Optional[dict]) -> Optional[ArrayReducer]:
        if not cfg:
            return None
        selections = self._parse_selections(cfg.get("select", []))
        aggregate = cfg.get("aggregate")
        flatten = cfg.get("flatten", True)
        return ArrayReducer(default_selections=selections, default_aggregation_axes=aggregate, flatten=flatten)

    def _parse_selections(self, selection_cfg: list) -> Tuple[Selection, ...]:
        selections = []
        for spec in selection_cfg:
            axis = spec["axis"]
            if "slice" in spec:
                slice_args = spec["slice"]
                start = slice_args[0]
                stop = slice_args[1]
                step = slice_args[2] if len(slice_args) > 2 else None
                indices = slice(start, stop, step)
            else:
                indices = spec.get("indices")
                if indices is None:
                    indices = spec.get("index")
            if indices is None:
                raise ValueError("Selection specification must include 'indices' or 'slice'.")
            selections.append(Selection(axis=axis, indices=indices))
        return tuple(selections)

    def _format_significance(self, p_value: float) -> str:
        if np.isnan(p_value):
            return "n/a"
        if p_value < 0.001:
            return "***"
        if p_value < 0.01:
            return "**"
        if p_value < 0.05:
            return "*"
        return "n.s."


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
