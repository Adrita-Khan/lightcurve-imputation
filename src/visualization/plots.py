"""
Publication-quality figure generation for the thesis.

All figures use a consistent style (LaTeX-like fonts, MNRAS-compatible sizes).
Figures are saved in PDF, PNG, and SVG by default.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Use non-interactive backend for headless execution
matplotlib.use("Agg")

# ── Publication style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "text.usetex": False,         # use mathtext (no LaTeX required)
    "font.family": "DejaVu Serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

_METHOD_COLORS = {
    "Mean-Fill": "#e41a1c",
    "Forward-Fill": "#ff7f00",
    "Linear-Interp": "#377eb8",
    "Spline-Interp": "#4daf4a",
    "GP-Matern32": "#984ea3",
    "TS-MICE": "#a65628",
    "KNN-Impute": "#f781bf",
    "RF-Impute": "#999999",
    "RNN-Impute": "#66c2a5",
    "GAIN-Impute": "#fc8d62",
    "MF-Impute": "#8da0cb",
    "GB-MICE": "#e78ac3",
    "SAITS": "#a6d854",
}


def save_figure(
    fig: plt.Figure,
    path_stem: str | Path,
    formats: Sequence[str] = ("pdf", "png", "svg"),
    dpi: int = 300,
) -> None:
    """Save a matplotlib figure in multiple formats.

    Parameters
    ----------
    fig : plt.Figure
    path_stem : str or Path
        Path without extension (e.g. ``'figures/fig_rmse'``).
    formats : sequence of str
        File extensions to write.
    dpi : int
        Resolution for raster formats.
    """
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(str(path_stem) + f".{fmt}", dpi=dpi)


# ---------------------------------------------------------------------------
# Figure 1 – Light-curve pipeline demonstration
# ---------------------------------------------------------------------------


def plot_lightcurve_pipeline(
    t: np.ndarray,
    flux_clean: np.ndarray,
    flux_gapped: np.ndarray,
    flux_imputed: np.ndarray,
    missing_idx: np.ndarray,
    method_name: str = "Linear Interpolation",
    fraction: float = 0.30,
    title_prefix: str = "",
) -> plt.Figure:
    """Three-panel pipeline figure: clean / gapped / reconstructed.

    Reproduces Figure 1.1 from the thesis.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    gap_regions = _contiguous_blocks(missing_idx)

    # Panel 1: clean signal
    ax = axes[0]
    ax.plot(t, flux_clean, color="#2166ac", lw=0.8, label="Ground truth")
    ax.set_ylabel("Flux")
    ax.set_title(f"{title_prefix}Complete synthetic signal")
    ax.legend(loc="upper right", framealpha=0.7)

    # Panel 2: gapped signal
    ax = axes[1]
    obs_mask = ~np.isnan(flux_gapped)
    ax.plot(t[obs_mask], flux_gapped[obs_mask], "k.", ms=1.5, label="Observed")
    for start, end in gap_regions:
        ax.axvspan(t[start], t[end], alpha=0.2, color="grey")
    ax.set_ylabel("Flux")
    ax.set_title(f"Observed data with {fraction*100:.0f}% MCAR gaps")
    ax.legend(loc="upper right", framealpha=0.7)

    # Panel 3: reconstructed
    ax = axes[2]
    ax.plot(t[obs_mask], flux_gapped[obs_mask], "k.", ms=1.5, label="Observed", zorder=3)
    for start, end in gap_regions:
        seg_t = t[start : end + 1]
        seg_f = flux_imputed[start : end + 1]
        ax.plot(seg_t, seg_f, "--", color="#d73027", lw=1.2, alpha=0.85)
    _legend_handles = [
        mpatches.Patch(color="#d73027", label=f"Imputed ({method_name})"),
        plt.Line2D([0], [0], color="k", marker=".", ms=4, lw=0, label="Observed"),
    ]
    ax.legend(handles=_legend_handles, loc="upper right", framealpha=0.7)
    ax.set_ylabel("Flux")
    ax.set_xlabel("Time (days)")
    ax.set_title(f"Reconstructed signal – {method_name}")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure – Metric heatmap
# ---------------------------------------------------------------------------


def plot_metric_heatmap(
    summary_df: pd.DataFrame,
    metric: str = "rmse_mean",
    fractions: Sequence[float] = (0.10, 0.30, 0.50),
    figsize: tuple = (9, 5),
) -> plt.Figure:
    """Heatmap of mean metric across methods × missingness fractions."""
    methods = summary_df["method"].unique().tolist()
    matrix = np.full((len(methods), len(fractions)), np.nan)

    for i, m in enumerate(methods):
        for j, frac in enumerate(fractions):
            row = summary_df[(summary_df["method"] == m) & (summary_df["fraction"] == frac)]
            if not row.empty and metric in row.columns:
                matrix[i, j] = row[metric].values[0]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r")
    fig.colorbar(im, ax=ax, label=metric)

    ax.set_xticks(range(len(fractions)))
    ax.set_xticklabels([f"{f*100:.0f}%" for f in fractions])
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xlabel("Missingness fraction")
    ax.set_title(f"{metric} – all methods")

    # Annotate cells
    for i in range(len(methods)):
        for j in range(len(fractions)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i,j]:.4f}", ha="center", va="center", fontsize=7)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure – Boxplots
# ---------------------------------------------------------------------------


def plot_boxplots(
    raw_df: pd.DataFrame,
    metric: str = "rmse",
    fraction: float = 0.30,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Boxplot of ``metric`` across seeds for each method at ``fraction``."""
    sub = raw_df[raw_df["fraction"] == fraction]
    methods = sub["method"].unique().tolist()
    data = [sub[sub["method"] == m][metric].dropna().values for m in methods]

    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(data, patch_artist=True, notch=False, medianprops=dict(color="black", lw=1.5))

    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(_METHOD_COLORS.get(m, "#aaaaaa"))
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} distribution – {fraction*100:.0f}% missing")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure – Violin plots
# ---------------------------------------------------------------------------


def plot_violin(
    raw_df: pd.DataFrame,
    metric: str = "rmse",
    fraction: float = 0.30,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Violin plot of metric distributions."""
    sub = raw_df[raw_df["fraction"] == fraction]
    methods = sub["method"].unique().tolist()
    data = [sub[sub["method"] == m][metric].dropna().values for m in methods]

    fig, ax = plt.subplots(figsize=figsize)
    parts = ax.violinplot(data, showmedians=True)
    for i, (pc, m) in enumerate(zip(parts["bodies"], methods)):
        pc.set_facecolor(_METHOD_COLORS.get(m, "#aaaaaa"))
        pc.set_alpha(0.7)

    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} violin – {fraction*100:.0f}% missing")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure – Runtime comparison
# ---------------------------------------------------------------------------


def plot_runtime_comparison(
    summary_df: pd.DataFrame,
    fraction: float = 0.30,
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """Horizontal bar chart of mean runtime per method."""
    sub = summary_df[summary_df["fraction"] == fraction].copy()
    if "runtime_s_mean" not in sub.columns:
        raise KeyError("'runtime_s_mean' not found in summary_df.")
    sub = sub.sort_values("runtime_s_mean")

    fig, ax = plt.subplots(figsize=figsize)
    colors = [_METHOD_COLORS.get(m, "#aaaaaa") for m in sub["method"]]
    ax.barh(sub["method"], sub["runtime_s_mean"], color=colors, alpha=0.8)
    if "runtime_s_std" in sub.columns:
        ax.errorbar(sub["runtime_s_mean"], sub["method"], xerr=sub["runtime_s_std"],
                    fmt="none", color="black", capsize=3)
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_title(f"Runtime – {fraction*100:.0f}% missing")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure – Period recovery
# ---------------------------------------------------------------------------


def plot_period_recovery(
    summary_df: pd.DataFrame,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Line plot of PRR vs missingness fraction for each method."""
    fig, ax = plt.subplots(figsize=figsize)
    fractions_pct = sorted(summary_df["fraction"].unique())

    for method in summary_df["method"].unique():
        sub = summary_df[summary_df["method"] == method].sort_values("fraction")
        if "prr" not in sub.columns:
            continue
        ax.plot(
            [f * 100 for f in sub["fraction"]],
            sub["prr"],
            marker="o",
            label=method,
            color=_METHOD_COLORS.get(method, None),
        )

    ax.set_xlabel("Missingness fraction (%)")
    ax.set_ylabel("Period Recovery Rate (PRR)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.set_title("Period Recovery Rate vs Missingness Fraction")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure – Missingness pattern
# ---------------------------------------------------------------------------


def plot_missingness_pattern(
    t: np.ndarray,
    flux_gapped: np.ndarray,
    fraction: float,
    figsize: tuple = (10, 2.5),
) -> plt.Figure:
    """Visualise the gap pattern in a gapped flux vector."""
    fig, ax = plt.subplots(figsize=figsize)
    obs_mask = ~np.isnan(flux_gapped)
    miss_mask = np.isnan(flux_gapped)

    ax.fill_between(t, 0, 1, where=miss_mask, transform=ax.get_xaxis_transform(),
                    alpha=0.4, color="tomato", label="Missing")
    ax.plot(t[obs_mask], np.ones(obs_mask.sum()) * 0.5, "|", ms=4, color="#2166ac",
            label="Observed", alpha=0.7)
    ax.set_yticks([])
    ax.set_xlabel("Time (days)")
    ax.set_title(f"Gap pattern — {fraction*100:.0f}% MCAR missing")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure – Error distribution
# ---------------------------------------------------------------------------


def plot_error_distribution(
    raw_df: pd.DataFrame,
    metric: str = "rmse",
    fraction: float = 0.30,
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """KDE / histogram of metric distribution per method."""
    sub = raw_df[raw_df["fraction"] == fraction]
    methods = sub["method"].unique().tolist()

    fig, ax = plt.subplots(figsize=figsize)
    for m in methods:
        vals = sub[sub["method"] == m][metric].dropna().values
        if len(vals) < 2:
            continue
        ax.hist(vals, bins=10, alpha=0.4, color=_METHOD_COLORS.get(m, None), label=m, density=True)

    ax.set_xlabel(metric.upper())
    ax.set_ylabel("Density")
    ax.set_title(f"{metric.upper()} distribution – {fraction*100:.0f}% missing")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _contiguous_blocks(idx: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end) pairs of contiguous index blocks."""
    if len(idx) == 0:
        return []
    blocks = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i != prev + 1:
            blocks.append((start, prev))
            start = i
        prev = i
    blocks.append((start, prev))
    return blocks
