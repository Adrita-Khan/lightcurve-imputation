"""Publication-quality figure generation module."""

from .plots import (
    plot_lightcurve_pipeline,
    plot_error_distribution,
    plot_metric_heatmap,
    plot_boxplots,
    plot_violin,
    plot_runtime_comparison,
    plot_period_recovery,
    plot_missingness_pattern,
    save_figure,
)

__all__ = [
    "plot_lightcurve_pipeline",
    "plot_error_distribution",
    "plot_metric_heatmap",
    "plot_boxplots",
    "plot_violin",
    "plot_runtime_comparison",
    "plot_period_recovery",
    "plot_missingness_pattern",
    "save_figure",
]
