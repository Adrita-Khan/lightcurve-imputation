#!/usr/bin/env python3
"""
Run statistical significance tests on experiment results.

Produces:
  - Friedman test (Table 4.5)
  - Nemenyi post-hoc pairwise matrix (Table 4.6)
  - Wilcoxon signed-rank GP vs Spline at each fraction (Table 4.7)

Requires seed-level accuracy data saved during run_experiment.py.
Usage:
    python scripts/run_statistical_tests.py --results_dir results
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation.metrics import friedman_test, nemenyi_test, wilcoxon_test

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("stat_tests")

METHOD_ORDER = [
    "Mean_Fill", "Forward_Fill", "Linear", "Spline",
    "GP_Matern32", "TS_MICE",
    "KNN_Impute", "RF_Impute", "RNN_Impute",
    "GAIN_Impute", "MF_Impute", "GB_MICE", "SAITS",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    return p.parse_args()


def main():
    args = parse_args()
    tables_dir = Path(args.results_dir) / "tables"
    seed_path  = tables_dir / "seed_level_accuracy.csv"

    if not seed_path.exists():
        logger.warning(
            "Seed-level accuracy file not found at %s.\n"
            "Ensure run_experiment.py saves per-seed accuracies.\n"
            "Generating placeholder tables.", seed_path
        )
        _write_placeholder_tables(tables_dir)
        return

    seed_df = pd.read_csv(seed_path)
    fractions = sorted(seed_df["fraction"].unique())

    friedman_rows  = []
    wilcoxon_rows  = []

    for frac in fractions:
        sub = seed_df[seed_df["fraction"] == frac]
        # Build (S × M) accuracy matrix
        methods_present = [m for m in METHOD_ORDER if m in sub["method"].unique()]
        S_vals = sorted(sub["seed"].unique())

        acc_matrix = np.zeros((len(S_vals), len(methods_present)))
        for mi, method in enumerate(methods_present):
            m_sub = sub[sub["method"] == method].sort_values("seed")
            acc_matrix[:, mi] = m_sub["accuracy"].values[:len(S_vals)]

        # Friedman
        fr = friedman_test(acc_matrix)
        friedman_rows.append({
            "Missingness":  f"{100*frac:.0f}%",
            "Friedman χ²":  f"{fr['statistic']:.3f}",
            "p-value":      f"{fr['p_value']:.4g}",
            "Conclusion":   "Reject H₀" if fr["p_value"] < 0.05 else "Fail to reject H₀",
        })
        logger.info("Friedman %d%%: χ²=%.3f, p=%.4g", int(100*frac),
                    fr["statistic"], fr["p_value"])

        # Nemenyi at 30%
        if abs(frac - 0.3) < 1e-6 and len(methods_present) >= 2:
            try:
                p_matrix = nemenyi_test(acc_matrix, methods_present)
                p_matrix.to_csv(tables_dir / "table_nemenyi_30pct.csv")
                logger.info("Saved Nemenyi matrix.")
            except Exception as exc:
                logger.warning("Nemenyi test failed: %s", exc)

        # Wilcoxon: GP vs Spline
        if "GP_Matern32" in methods_present and "Spline" in methods_present:
            gp_idx = methods_present.index("GP_Matern32")
            sp_idx = methods_present.index("Spline")
            wres = wilcoxon_test(acc_matrix[:, gp_idx], acc_matrix[:, sp_idx])
            wilcoxon_rows.append({
                "Missingness":   f"{100*frac:.0f}%",
                "W":             f"{wres['statistic']:.1f}",
                "p-value":       f"{wres['p_value']:.4g}",
                "Effect size rW": f"{wres['effect_size']:.3f}",
                "Interpretation": "Significant" if wres["p_value"] < 0.05 else "Not significant",
            })

    pd.DataFrame(friedman_rows).to_csv(tables_dir / "table_friedman.csv", index=False)
    logger.info("Saved Friedman table.")

    if wilcoxon_rows:
        pd.DataFrame(wilcoxon_rows).to_csv(tables_dir / "table_wilcoxon_gp_vs_spline.csv", index=False)
        logger.info("Saved Wilcoxon table.")


def _write_placeholder_tables(tables_dir):
    """Write placeholder CSVs with headers only."""
    fracs = ["10%", "30%", "50%"]
    pd.DataFrame([{
        "Missingness": f, "Friedman χ²": "—", "p-value": "—", "Conclusion": "Pending"
    } for f in fracs]).to_csv(tables_dir / "table_friedman.csv", index=False)

    pd.DataFrame([{
        "Missingness": f, "W": "—", "p-value": "—",
        "Effect size rW": "—", "Interpretation": "Pending"
    } for f in fracs]).to_csv(tables_dir / "table_wilcoxon_gp_vs_spline.csv", index=False)


if __name__ == "__main__":
    main()
