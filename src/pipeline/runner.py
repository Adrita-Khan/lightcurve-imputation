"""
ExperimentPipeline: orchestrate the full corrupt-and-recover experiment.

The pipeline follows Algorithm 1 from the thesis:
1. Generate a clean synthetic light curve.
2. For each missingness fraction × imputation method × seed:
   a. Inject MCAR gaps.
   b. Estimate period from observed cadences.
   c. Apply imputation method.
   d. Evaluate reconstruction metrics.
3. Aggregate results, run statistical tests.
4. Generate all figures and tables.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from tqdm import tqdm

from src.evaluation.aggregator import aggregate_results, summarise_results
from src.evaluation.metrics import evaluate_imputation
from src.imputation.registry import get_imputer, list_imputers
from src.missingness.injector import inject_gaps
from src.simulation.generator import generate_synthetic_lightcurve
from src.statistics.tables import generate_stats_table
from src.utils.config import load_config
from src.utils.io import save_results, save_table
from src.utils.logging_utils import get_logger
from src.utils.seeds import make_seed_sequence
from src.visualization.plots import (
    plot_boxplots,
    plot_lightcurve_pipeline,
    plot_metric_heatmap,
    plot_missingness_pattern,
    plot_period_recovery,
    plot_runtime_comparison,
    plot_violin,
    save_figure,
)

logger = get_logger(__name__)


class ExperimentPipeline:
    """Full experiment pipeline.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML experiment configuration.
    """

    def __init__(self, config_path: str | Path = "configs/experiment.yml") -> None:
        self.cfg = load_config(config_path)
        self._setup_output_dirs()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """Execute the complete pipeline and return the raw results DataFrame."""
        logger.info("=" * 60)
        logger.info("Starting lightcurve-imputation experiment pipeline")
        logger.info("=" * 60)

        # Stage 1: generate ground-truth signal
        sig_cfg = self.cfg["signal"]
        t, flux, signal_params = generate_synthetic_lightcurve(**sig_cfg)
        logger.info(
            f"Signal generated: N={sig_cfg['N']}, P0={sig_cfg['P0']} d, "
            f"A={sig_cfg['A']}, sigma_eps={sig_cfg['sigma_eps']}"
        )

        # Stage 2–4: corrupt → impute → evaluate
        miss_cfg = self.cfg["missingness"]
        fractions = miss_cfg["fractions"]
        n_seeds = miss_cfg["n_seeds"]
        base_seed = self.cfg["reproducibility"]["base_seed"]
        seeds = make_seed_sequence(n_seeds, base_seed)

        method_cfg = self.cfg["methods"]
        param_cfg = self.cfg.get("method_params", {})
        eval_cfg = self.cfg["evaluation"]

        active_methods = [k for k, v in method_cfg.items() if v]
        logger.info(f"Active methods ({len(active_methods)}): {active_methods}")
        logger.info(f"Fractions: {fractions}, Seeds: {n_seeds}")

        all_results: list[dict] = []

        for frac in fractions:
            logger.info(f"\n── Missingness fraction: {frac*100:.0f}% ──")
            for method_key in active_methods:
                logger.info(f"  Method: {method_key}")
                for seed_idx, seed in enumerate(tqdm(seeds, desc=f"  {method_key}", leave=False)):
                    row = self._run_single(
                        t=t,
                        flux=flux,
                        signal_params=signal_params,
                        method_key=method_key,
                        params=param_cfg.get(method_key, {}),
                        fraction=frac,
                        seed=seed,
                        eval_cfg=eval_cfg,
                        miss_cfg=miss_cfg,
                    )
                    row["seed_idx"] = seed_idx
                    all_results.append(row)

        raw_df = aggregate_results(all_results)
        results_dir = Path(self.cfg["output"]["results_dir"])
        save_results(raw_df, results_dir / "raw_results.csv")
        logger.info(f"\nRaw results saved: {results_dir / 'raw_results.csv'}")

        # Stage 5: aggregate and generate outputs
        summary_df = summarise_results(raw_df)
        save_results(summary_df, results_dir / "summary_results.csv")

        self._generate_tables(raw_df, summary_df)
        self._generate_figures(t, flux, raw_df, summary_df)

        logger.info("\n" + "=" * 60)
        logger.info("Pipeline complete.")
        logger.info("=" * 60)
        return raw_df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_single(
        self,
        t: np.ndarray,
        flux: np.ndarray,
        signal_params: dict,
        method_key: str,
        params: dict,
        fraction: float,
        seed: int,
        eval_cfg: dict,
        miss_cfg: dict,
    ) -> dict:
        """Run one (method, fraction, seed) experiment and return a metric dict."""
        # Inject gaps
        gapped, missing_idx, true_vals = inject_gaps(
            flux=flux,
            p=fraction,
            seed=seed,
            block_ratio=miss_cfg.get("block_ratio", 0.5),
            block_min_len=miss_cfg.get("block_min_len", 5),
            block_max_len=miss_cfg.get("block_max_len", 50),
        )

        # Estimate period from observed cadences
        obs_mask = ~np.isnan(gapped)
        period_est = _estimate_period(t[obs_mask], gapped[obs_mask])

        # Build and run imputer
        try:
            imputer = get_imputer(method_key, params=params, seed=seed)
            imputed = imputer.fit_impute(t, gapped, missing_idx, period_est=period_est)
        except Exception as exc:
            warnings.warn(f"[{method_key}] seed={seed} frac={fraction}: {exc}")
            # Fallback: fill with observed mean
            imputed = gapped.copy()
            imputed[missing_idx] = float(np.nanmean(gapped))
            imputer_runtime, imputer_mem = 0.0, 0.0
        else:
            imputer_runtime = imputer.runtime_s
            imputer_mem = imputer.memory_mb

        # Evaluate
        metrics = evaluate_imputation(
            t=t,
            flux_imputed=imputed,
            missing_idx=missing_idx,
            true_vals=true_vals,
            true_period=signal_params["P0"],
            true_amplitude=signal_params["A"],
            true_phase=signal_params["phi0"],
            period_tol=eval_cfg.get("period_tol", 0.01),
            runtime_s=imputer_runtime,
            memory_mb=imputer_mem,
        )
        metrics["method"] = get_imputer(method_key, params={}).name  # use display name
        metrics["method_key"] = method_key
        metrics["fraction"] = fraction
        metrics["seed"] = seed
        return metrics

    def _setup_output_dirs(self) -> None:
        out = self.cfg["output"]
        for d in [out["results_dir"], out["figures_dir"], out["tables_dir"]]:
            Path(d).mkdir(parents=True, exist_ok=True)

    def _generate_tables(self, raw_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
        tables_dir = Path(self.cfg["output"]["tables_dir"])
        fractions = self.cfg["missingness"]["fractions"]

        # Main RMSE/MAE table
        for metric in ["rmse", "mae"]:
            rows = []
            for frac in fractions:
                sub_sum = summary_df[summary_df["fraction"] == frac]
                col_mean = f"{metric}_mean"
                col_std = f"{metric}_std"
                if col_mean in sub_sum.columns:
                    tbl = sub_sum[["method", col_mean, col_std]].copy()
                    tbl["fraction"] = frac
                    rows.append(tbl)
            if rows:
                combined = pd.concat(rows)
                save_table(combined, tables_dir / f"table_{metric}.tex",
                           fmt="latex", caption=f"{metric.upper()} by method and fraction",
                           label=f"tab:{metric}")
                save_table(combined, tables_dir / f"table_{metric}.csv", fmt="csv")

        # Statistical summary table
        for frac in fractions:
            stats_tbl = generate_stats_table(raw_df, metric="rmse", fraction=frac)
            save_table(stats_tbl, tables_dir / f"stats_rmse_{int(frac*100)}pct.tex",
                       fmt="latex", caption=f"Statistical summary RMSE ({int(frac*100)}%)",
                       label=f"tab:stats_{int(frac*100)}")
            save_table(stats_tbl, tables_dir / f"stats_rmse_{int(frac*100)}pct.csv", fmt="csv")

        logger.info(f"Tables written to {tables_dir}")

    def _generate_figures(
        self,
        t: np.ndarray,
        flux: np.ndarray,
        raw_df: pd.DataFrame,
        summary_df: pd.DataFrame,
    ) -> None:
        figs_dir = Path(self.cfg["output"]["figures_dir"])
        formats = self.cfg["output"].get("figure_formats", ["pdf", "png", "svg"])
        dpi = self.cfg["output"].get("dpi", 300)
        fractions = self.cfg["missingness"]["fractions"]
        miss_cfg = self.cfg["missingness"]
        base_seed = self.cfg["reproducibility"]["base_seed"]

        # Pipeline demonstration figure
        sig_cfg = self.cfg["signal"]
        gapped_demo, missing_idx_demo, _ = inject_gaps(
            flux=flux, p=0.30, seed=base_seed,
            block_ratio=miss_cfg["block_ratio"],
            block_min_len=miss_cfg["block_min_len"],
            block_max_len=miss_cfg["block_max_len"],
        )
        from src.imputation.deterministic import LinearInterpImputer
        lin = LinearInterpImputer()
        imputed_demo = lin.impute(t, gapped_demo, missing_idx_demo)
        fig = plot_lightcurve_pipeline(t, flux, gapped_demo, imputed_demo, missing_idx_demo,
                                       method_name="Linear Interpolation", fraction=0.30)
        save_figure(fig, figs_dir / "fig_pipeline_demo", formats, dpi)
        plt_close(fig)

        # Missingness pattern
        for frac in fractions:
            gapped, missing_idx, _ = inject_gaps(flux=flux, p=frac, seed=base_seed)
            fig = plot_missingness_pattern(t, gapped, frac)
            save_figure(fig, figs_dir / f"fig_missingness_{int(frac*100)}pct", formats, dpi)
            plt_close(fig)

        # Metric heatmap
        for metric in ["rmse_mean", "mae_mean"]:
            if metric in summary_df.columns:
                fig = plot_metric_heatmap(summary_df, metric=metric, fractions=fractions)
                save_figure(fig, figs_dir / f"fig_heatmap_{metric}", formats, dpi)
                plt_close(fig)

        # Boxplots and violin plots
        for frac in fractions:
            for metric in ["rmse", "mae"]:
                fig = plot_boxplots(raw_df, metric=metric, fraction=frac)
                save_figure(fig, figs_dir / f"fig_box_{metric}_{int(frac*100)}pct", formats, dpi)
                plt_close(fig)
                fig = plot_violin(raw_df, metric=metric, fraction=frac)
                save_figure(fig, figs_dir / f"fig_violin_{metric}_{int(frac*100)}pct", formats, dpi)
                plt_close(fig)

        # Runtime comparison
        if "runtime_s_mean" in summary_df.columns:
            for frac in fractions:
                fig = plot_runtime_comparison(summary_df, fraction=frac)
                save_figure(fig, figs_dir / f"fig_runtime_{int(frac*100)}pct", formats, dpi)
                plt_close(fig)

        # Period recovery
        if "prr" in summary_df.columns:
            fig = plot_period_recovery(summary_df)
            save_figure(fig, figs_dir / "fig_period_recovery", formats, dpi)
            plt_close(fig)

        logger.info(f"Figures written to {figs_dir}")


def plt_close(fig) -> None:
    import matplotlib.pyplot as plt
    plt.close(fig)


def _estimate_period(t_obs: np.ndarray, f_obs: np.ndarray) -> float:
    """Estimate dominant period from observed cadences via Lomb–Scargle."""
    if len(t_obs) < 10:
        return 1.0  # fallback
    try:
        ls = LombScargle(t_obs, f_obs)
        freq, power = ls.autopower(minimum_frequency=0.1, maximum_frequency=10.0)
        if len(freq) == 0:
            return 1.0
        return float(1.0 / freq[np.argmax(power)])
    except Exception:
        return 1.0


def run_experiment(config_path: str = "configs/experiment.yml") -> pd.DataFrame:
    """Convenience function for running the experiment from Python."""
    pipeline = ExperimentPipeline(config_path)
    return pipeline.run()


def main() -> None:
    """CLI entry point: ``python -m src.pipeline.runner`` or ``lc-impute``."""
    parser = argparse.ArgumentParser(
        description="Lightcurve missing-data imputation experiment pipeline."
    )
    parser.add_argument(
        "--config",
        default="configs/experiment.yml",
        help="Path to YAML configuration file (default: configs/experiment.yml).",
    )
    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
