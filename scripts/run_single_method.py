#!/usr/bin/env python3
"""
scripts/run_single_method.py
============================
Run one imputation method at one missingness fraction for quick testing
or ablation analysis.

Usage
-----
    python scripts/run_single_method.py --method gp_matern --fraction 0.30
    python scripts/run_single_method.py --method linear_interp --fraction 0.10 --seeds 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.simulation.generator import generate_synthetic_lightcurve
from src.missingness.injector import inject_gaps
from src.imputation.registry import get_imputer, list_imputers
from src.evaluation.metrics import evaluate_imputation
from src.evaluation.aggregator import aggregate_results, summarise_results
from src.utils.config import load_config
from src.utils.seeds import make_seed_sequence
from astropy.timeseries import LombScargle


def estimate_period(t_obs, f_obs):
    ls = LombScargle(t_obs, f_obs)
    freq, power = ls.autopower(minimum_frequency=0.1, maximum_frequency=10.0)
    return float(1.0 / freq[np.argmax(power)]) if len(freq) > 0 else 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single imputation method.")
    parser.add_argument("--method", default="linear_interp",
                        help=f"Method key. Available: {list_imputers()}")
    parser.add_argument("--fraction", type=float, default=0.30,
                        help="Missingness fraction (e.g. 0.10, 0.30, 0.50)")
    parser.add_argument("--seeds", type=int, default=10, help="Number of random seeds")
    parser.add_argument("--config", default="configs/experiment.yml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    t, flux, params = generate_synthetic_lightcurve(**cfg["signal"])
    seeds = make_seed_sequence(args.seeds, cfg["reproducibility"]["base_seed"])
    method_params = cfg.get("method_params", {}).get(args.method, {})

    print(f"\nMethod    : {args.method}")
    print(f"Fraction  : {args.fraction*100:.0f}%")
    print(f"Seeds     : {args.seeds}")
    print("-" * 40)

    rows = []
    for seed in seeds:
        gapped, missing_idx, true_vals = inject_gaps(flux, p=args.fraction, seed=seed)
        obs_mask = ~np.isnan(gapped)
        period_est = estimate_period(t[obs_mask], gapped[obs_mask])
        imp = get_imputer(args.method, params=method_params, seed=seed)
        imputed = imp.fit_impute(t, gapped, missing_idx, period_est=period_est)
        m = evaluate_imputation(
            t, imputed, missing_idx, true_vals,
            true_period=params["P0"], true_amplitude=params["A"], true_phase=params["phi0"],
            runtime_s=imp.runtime_s, memory_mb=imp.memory_mb,
        )
        m.update({"method": imp.name, "method_key": args.method,
                  "fraction": args.fraction, "seed": seed})
        rows.append(m)
        print(f"  seed={seed}  RMSE={m['rmse']:.5f}  MAE={m['mae']:.5f}  "
              f"PRR={m['period_recovered']}  t={m['runtime_s']:.3f}s")

    raw_df = aggregate_results(rows)
    summary = summarise_results(raw_df)
    print("\n── Summary ──")
    for col in ["rmse_mean", "rmse_std", "mae_mean", "prr"]:
        if col in summary.columns:
            print(f"  {col:20s}: {summary[col].values[0]:.5f}")


if __name__ == "__main__":
    main()
