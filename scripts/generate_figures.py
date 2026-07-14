#!/usr/bin/env python3
"""
scripts/generate_figures.py
===========================
Regenerate all figures from saved CSV results without re-running
the full experiment.

Usage
-----
    python scripts/generate_figures.py
    python scripts/generate_figures.py --results data/results/raw_results.csv
"""

import argparse
import sys
from pathlib import Path

# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.io import load_results
from src.evaluation.aggregator import summarise_results
from src.visualization.plots import (
    plot_boxplots, plot_violin, plot_metric_heatmap,
    plot_period_recovery, plot_runtime_comparison, save_figure,
)
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate figures from saved results.")
    parser.add_argument("--results", default="data/results/raw_results.csv",
                        help="Path to raw_results.csv")
    parser.add_argument("--config", default="configs/experiment.yml",
                        help="Experiment configuration")
    parser.add_argument("--outdir", default="figures", help="Output directory")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"[ERROR] Results file not found: {results_path}")
        print("Run `python run_all.py` first.")
        sys.exit(1)

    print(f"Loading results from {results_path} …")
    raw_df = load_results(results_path)
    summary_df = summarise_results(raw_df)
    cfg = load_config(args.config)
    fractions = cfg["missingness"]["fractions"]
    outdir = Path(args.outdir)
    formats = cfg["output"].get("figure_formats", ["pdf", "png", "svg"])
    dpi = cfg["output"].get("dpi", 300)

    print(f"Generating figures in {outdir}/ …")
    outdir.mkdir(parents=True, exist_ok=True)

    # Boxplots and violins
    for frac in fractions:
        for metric in ["rmse", "mae"]:
            fig = plot_boxplots(raw_df, metric=metric, fraction=frac)
            save_figure(fig, outdir / f"fig_box_{metric}_{int(frac*100)}pct", formats, dpi)
            plt.close(fig)

            fig = plot_violin(raw_df, metric=metric, fraction=frac)
            save_figure(fig, outdir / f"fig_violin_{metric}_{int(frac*100)}pct", formats, dpi)
            plt.close(fig)

    # Heatmap
    for metric in ["rmse_mean", "mae_mean"]:
        if metric in summary_df.columns:
            fig = plot_metric_heatmap(summary_df, metric=metric, fractions=fractions)
            save_figure(fig, outdir / f"fig_heatmap_{metric}", formats, dpi)
            plt.close(fig)

    # Period recovery
    if "prr" in summary_df.columns:
        fig = plot_period_recovery(summary_df)
        save_figure(fig, outdir / "fig_period_recovery", formats, dpi)
        plt.close(fig)

    # Runtime
    if "runtime_s_mean" in summary_df.columns:
        for frac in fractions:
            fig = plot_runtime_comparison(summary_df, fraction=frac)
            save_figure(fig, outdir / f"fig_runtime_{int(frac*100)}pct", formats, dpi)
            plt.close(fig)

    print(f"Done. Figures saved to {outdir}/")


if __name__ == "__main__":
    main()
