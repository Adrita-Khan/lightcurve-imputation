#!/usr/bin/env python3
"""
run_all.py
==========
Single entry point to reproduce all thesis experiments, figures, and tables.

Usage
-----
    python run_all.py                        # use default experiment.yml
    python run_all.py --config configs/fast_test.yml   # quick smoke test
    make reproduce                           # equivalent via Makefile

The script runs the complete four-stage pipeline:
    1. Synthetic light-curve generation
    2. MCAR gap injection (10%, 30%, 50% at 30 seeds each)
    3. Thirteen imputation methods
    4. Evaluation, statistical tests, figures, and tables

Outputs are written to:
    data/results/   — raw_results.csv, summary_results.csv
    figures/        — all publication-quality figures (PDF, PNG, SVG)
    tables/         — LaTeX and CSV tables

All results are fully reproducible from the fixed random seeds in the config.
"""

import argparse
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce all thesis experiments, figures, and tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="configs/experiment.yml",
        metavar="PATH",
        help="Path to YAML experiment configuration.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate imports and configuration without running experiments.",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  Lightcurve Imputation – Master's Thesis Reproducibility Script")
    print("=" * 65)
    print(f"  Config : {args.config}")
    print(f"  Dry run: {args.dry_run}")
    print("=" * 65)

    # Validate configuration exists
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[ERROR] Configuration file not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        # Import everything to check for missing dependencies
        print("\n[DRY RUN] Validating imports …")
        from src.utils.config import load_config
        cfg = load_config(cfg_path)
        from src.simulation.generator import generate_synthetic_lightcurve
        from src.missingness.injector import inject_gaps
        from src.imputation.registry import list_imputers
        from src.evaluation.metrics import evaluate_imputation
        print(f"  Available imputers : {list_imputers()}")
        print(f"  Signal N           : {cfg['signal']['N']}")
        print(f"  Fractions          : {cfg['missingness']['fractions']}")
        print(f"  Seeds              : {cfg['missingness']['n_seeds']}")
        print("\n[DRY RUN] All imports OK. No experiments run.")
        return

    # Run the full pipeline
    from src.pipeline.runner import ExperimentPipeline

    t0 = time.perf_counter()
    pipeline = ExperimentPipeline(cfg_path)
    raw_df = pipeline.run()

    elapsed = time.perf_counter() - t0
    n_rows = len(raw_df)
    print(f"\n  Total experiments : {n_rows}")
    print(f"  Wall-clock time   : {elapsed:.1f} s ({elapsed/60:.1f} min)")
    print("\n  Figures   → figures/")
    print("  Tables    → tables/")
    print("  Results   → data/results/")
    print("\nDone. All outputs saved.")


if __name__ == "__main__":
    main()
