#!/usr/bin/env python3
"""
scripts/generate_tables.py
==========================
Regenerate all publication tables from saved CSV results.

Usage
-----
    python scripts/generate_tables.py
    python scripts/generate_tables.py --results data/results/raw_results.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.io import load_results, save_table
from src.evaluation.aggregator import summarise_results
from src.statistics.tables import generate_stats_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate tables from saved results.")
    parser.add_argument("--results", default="data/results/raw_results.csv")
    parser.add_argument("--config", default="configs/experiment.yml")
    parser.add_argument("--outdir", default="tables")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"[ERROR] Results file not found: {results_path}")
        sys.exit(1)

    raw_df = load_results(results_path)
    summary_df = summarise_results(raw_df)
    cfg = load_config(args.config)
    fractions = cfg["missingness"]["fractions"]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Writing tables to {outdir}/ …")

    # RMSE and MAE summary tables
    for metric in ["rmse", "mae"]:
        col_mean = f"{metric}_mean"
        col_std = f"{metric}_std"
        if col_mean not in summary_df.columns:
            continue
        tbl = summary_df[["method", "fraction", col_mean, col_std]].copy()
        save_table(tbl, outdir / f"table_{metric}.tex", fmt="latex",
                   caption=f"{metric.upper()} by method and missingness fraction",
                   label=f"tab:{metric}")
        save_table(tbl, outdir / f"table_{metric}.csv", fmt="csv")
        print(f"  Saved table_{metric}.tex/.csv")

    # Statistical summary (Wilcoxon + CI) at each fraction
    for frac in fractions:
        tbl = generate_stats_table(raw_df, metric="rmse", fraction=frac)
        save_table(tbl, outdir / f"stats_rmse_{int(frac*100)}pct.tex", fmt="latex",
                   caption=f"Statistical summary: RMSE at {int(frac*100)}% missing",
                   label=f"tab:stats_{int(frac*100)}")
        save_table(tbl, outdir / f"stats_rmse_{int(frac*100)}pct.csv", fmt="csv")
        print(f"  Saved stats_rmse_{int(frac*100)}pct.tex/.csv")

    # PRR table
    if "prr" in summary_df.columns:
        prr_tbl = summary_df[["method", "fraction", "prr"]].copy()
        save_table(prr_tbl, outdir / "table_prr.tex", fmt="latex",
                   caption="Period Recovery Rate (PRR) by method and missingness fraction",
                   label="tab:prr")
        save_table(prr_tbl, outdir / "table_prr.csv", fmt="csv")
        print("  Saved table_prr.tex/.csv")

    print("Done.")


if __name__ == "__main__":
    main()
