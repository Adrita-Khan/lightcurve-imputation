#!/usr/bin/env python3
"""
Main experiment runner — reproduces all results in Chapter 4 of the thesis.

Implements the five-stage controlled corrupt-and-recover pipeline
(Algorithm 1 in the thesis):
  1. Data loading + preprocessing
  2. Train and freeze the RF classifier on gap-free features
  3. For each (method, fraction, seed): inject gaps → impute → extract features → classify
  4. Compute RMSE, MAE, accuracy loss, period recovery, feature distortion
  5. Save results to results/tables/

Usage:
    python scripts/run_experiment.py --config configs/experiment.yaml
    python scripts/run_experiment.py --config configs/experiment.yaml --methods GP_Matern32 SAITS
    python scripts/run_experiment.py --config configs/experiment.yaml --fractions 0.3 --n_seeds 5
"""

import argparse
import json
import logging
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import preprocess_light_curve, compute_completeness, truncate_to_window
from src.data.gap_injection import inject_gaps
from src.features.extraction import extract_features, FEATURE_NAMES
from src.classification.classifier import train_classifier, evaluate_classifier, load_classifier
from src.imputation.registry import get_all_imputers, ALL_METHOD_NAMES
from src.evaluation.metrics import (
    rmse as compute_rmse,
    mae as compute_mae,
    accuracy_loss,
    period_recovery_rate,
    feature_distortion,
    bootstrap_ci,
    safe_missingness_threshold,
)
from src.utils.period import lomb_scargle_peak_power

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("run_experiment")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run the full imputation benchmark.")
    p.add_argument("--config", default="configs/experiment.yaml", help="Path to config YAML.")
    p.add_argument("--methods", nargs="+", default=None,
                   help="Subset of methods to run (default: all 13).")
    p.add_argument("--fractions", nargs="+", type=float, default=None,
                   help="Missingness fractions to test (default: from config).")
    p.add_argument("--n_seeds", type=int, default=None,
                   help="Number of gap realisations per light curve (default: from config).")
    p.add_argument("--data_dir", default="data/processed",
                   help="Directory containing preprocessed light curves as .parquet files.")
    p.add_argument("--results_dir", default="results", help="Output directory.")
    p.add_argument("--clf_path", default=None,
                   help="Path to a pre-trained classifier .pkl (skip training if provided).")
    p.add_argument("--n_jobs", type=int, default=1, help="Parallel jobs (experimental).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- Load config ---
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results_dir = Path(args.results_dir)
    tables_dir  = results_dir / "tables"
    models_dir  = results_dir / "models"
    tables_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    fractions = args.fractions or cfg["gap"]["fractions"]
    n_seeds   = args.n_seeds   or cfg["gap"]["n_seeds"]
    methods   = args.methods   or ALL_METHOD_NAMES
    seed_base = cfg.get("seed", 42)

    logger.info("=" * 60)
    logger.info("Experiment: %d methods, %d fractions, %d seeds",
                len(methods), len(fractions), n_seeds)
    logger.info("=" * 60)

    # --- Load preprocessed data ---
    data_dir = Path(args.data_dir)
    light_curves = _load_dataset(data_dir, cfg)
    logger.info("Dataset: %d light curves", len(light_curves))

    if len(light_curves) == 0:
        logger.error(
            "No light curves found in '%s'. "
            "Run scripts/download_data.py first, or check the data directory.",
            data_dir,
        )
        sys.exit(1)

    # --- Train/test split ---
    from sklearn.model_selection import train_test_split
    kics   = [lc["kic"]   for lc in light_curves]
    labels = [lc["label"] for lc in light_curves]
    idx    = np.arange(len(light_curves))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
        stratify=labels,
    )
    train_lcs = [light_curves[i] for i in train_idx]
    test_lcs  = [light_curves[i] for i in test_idx]
    logger.info("Train: %d | Test: %d", len(train_lcs), len(test_lcs))

    # --- Extract gap-free training features ---
    logger.info("Extracting gap-free training features …")
    X_train, y_train = _extract_feature_matrix(train_lcs)

    # --- Train (or load) classifier ---
    clf_path = Path(models_dir / "rf_classifier.pkl")
    if args.clf_path and Path(args.clf_path).exists():
        logger.info("Loading pre-trained classifier from %s", args.clf_path)
        clf = load_classifier(args.clf_path)
    else:
        logger.info("Training RF classifier …")
        clf = train_classifier(
            X_train, y_train,
            n_estimators=cfg["classifier"]["n_estimators"],
            random_state=cfg["classifier"]["random_state"],
            cv_folds=cfg["classifier"]["cv_folds"],
            model_path=clf_path,
        )

    # --- Gap-free baseline on test set ---
    logger.info("Evaluating gap-free baseline …")
    X_test_gf, y_test = _extract_feature_matrix(test_lcs)
    baseline_result = evaluate_classifier(clf, X_test_gf, y_test)
    acc_baseline = baseline_result["accuracy"]
    logger.info("Gap-free baseline accuracy: %.4f", acc_baseline)

    # Save baseline features for distortion analysis
    phi_true_matrix = X_test_gf.copy()

    # --- Period recovery baseline ---
    periods_true = np.array([
        lomb_scargle_peak_power(lc["time"], lc["flux"])[0]
        for lc in test_lcs
    ])

    # --- Instantiate all imputers ---
    imputers = get_all_imputers(seed=seed_base, config=cfg)

    # --- Main experiment loop ---
    all_results = []

    for method_name in methods:
        if method_name not in imputers:
            logger.warning("Unknown method '%s', skipping.", method_name)
            continue

        imputer = imputers[method_name]
        logger.info("Method: %s", method_name)

        for p in fractions:
            logger.info("  Fraction: %.0f%%", 100 * p)

            seed_rmses = []
            seed_maes  = []
            seed_accs  = []
            seed_prrs  = []
            phi_imp_list = []

            for s in range(n_seeds):
                seed = seed_base + s
                imputer.seed = seed  # update seed per realisation

                batch_rmse = []
                batch_mae  = []
                batch_phi  = []
                periods_imp = []
                y_pred_all  = []
                phi_test_batch = []

                for lc in test_lcs:
                    time  = lc["time"]
                    flux  = lc["flux"]

                    # Inject gaps
                    gapped, mask, ground_truth = inject_gaps(
                        flux, p=p, seed=seed,
                        block_ratio=cfg["gap"]["block_ratio"],
                        block_min_len=cfg["gap"]["block_min_len"],
                        block_max_len=cfg["gap"]["block_max_len"],
                    )

                    # Impute
                    t0 = _time.perf_counter()
                    try:
                        flux_imp = imputer.impute(gapped, mask, time)
                    except Exception as exc:
                        logger.debug("Imputation failed for KIC %s: %s", lc.get("kic"), exc)
                        flux_imp = gapped.copy()
                        flux_imp[~mask] = np.nanmean(flux[mask])

                    missing_idx = np.where(~mask)[0]
                    if len(missing_idx) > 0:
                        batch_rmse.append(compute_rmse(ground_truth, flux_imp[missing_idx]))
                        batch_mae.append(compute_mae(ground_truth, flux_imp[missing_idx]))

                    # Feature extraction on imputed curve
                    phi = extract_features(time, flux_imp)
                    batch_phi.append(phi)

                    # Period from imputed curve
                    p_imp, _ = lomb_scargle_peak_power(time, flux_imp)
                    periods_imp.append(p_imp)

                # Batch classification
                X_imp_batch = np.stack(batch_phi)
                result = evaluate_classifier(clf, X_imp_batch, y_test)

                seed_rmses.append(np.nanmean(batch_rmse))
                seed_maes.append(np.nanmean(batch_mae))
                seed_accs.append(result["accuracy"])
                seed_prrs.append(period_recovery_rate(
                    periods_true, np.array(periods_imp),
                    threshold=cfg["evaluation"]["period_recovery_threshold"],
                ))
                phi_imp_list.append(X_imp_batch)

            # Aggregate across seeds
            phi_imp_matrix = np.mean(np.stack(phi_imp_list), axis=0)
            feat_dist = feature_distortion(phi_imp_matrix, phi_true_matrix)
            rmse_mean, rmse_std = np.nanmean(seed_rmses), np.nanstd(seed_rmses)
            mae_mean,  mae_std  = np.nanmean(seed_maes),  np.nanstd(seed_maes)
            acc_mean,  acc_std  = np.mean(seed_accs),     np.std(seed_accs)
            prr_mean            = np.mean(seed_prrs)

            ci_lo, ci_hi = bootstrap_ci(
                np.array(seed_accs),
                n_bootstrap=cfg["evaluation"]["n_bootstrap"],
                seed=seed_base,
            )

            acc_loss_val = accuracy_loss(acc_baseline, acc_mean)

            row = {
                "method":       method_name,
                "fraction":     p,
                "rmse_mean":    rmse_mean,
                "rmse_std":     rmse_std,
                "mae_mean":     mae_mean,
                "mae_std":      mae_std,
                "acc_mean":     acc_mean,
                "acc_std":      acc_std,
                "acc_ci_lo":    ci_lo,
                "acc_ci_hi":    ci_hi,
                "acc_loss":     acc_loss_val,
                "prr":          prr_mean,
                "feat_dist_mean": float(feat_dist.mean()),
            }
            all_results.append(row)

            logger.info(
                "    RMSE=%.4f±%.4f  MAE=%.4f±%.4f  Acc=%.4f [%.4f,%.4f]  ΔAcc=%.4f  PRR=%.4f",
                rmse_mean, rmse_std, mae_mean, mae_std,
                acc_mean, ci_lo, ci_hi, acc_loss_val, prr_mean,
            )

    # --- Save results ---
    results_df = pd.DataFrame(all_results)
    out_path = tables_dir / "experiment_results.csv"
    results_df.to_csv(out_path, index=False)
    logger.info("Results saved to %s", out_path)

    # Save baseline metadata
    baseline_meta = {
        "acc_baseline": acc_baseline,
        "f1_baseline":  baseline_result["f1_macro"],
        "mcc_baseline": baseline_result["mcc"],
        "n_train":      len(train_lcs),
        "n_test":       len(test_lcs),
        "methods":      methods,
        "fractions":    fractions,
        "n_seeds":      n_seeds,
    }
    with open(tables_dir / "baseline_meta.json", "w") as f:
        json.dump(baseline_meta, f, indent=2)

    logger.info("Done.")
    return results_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_dataset(data_dir: Path, cfg: dict) -> list[dict]:
    """Load preprocessed parquet files from data_dir."""
    light_curves = []
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        return []

    import pandas as pd

    for fp in parquet_files:
        df = pd.read_parquet(fp)
        if "flux" not in df.columns or "time" not in df.columns:
            continue
        if "label" not in df.columns and "class_label" not in df.columns:
            continue

        label_col = "label" if "label" in df.columns else "class_label"
        label = int(df[label_col].iloc[0])

        time  = df["time"].values.astype(float)
        flux  = df["flux"].values.astype(float)

        # Truncate to analysis window
        n_window = cfg.get("data", {}).get("analysis_window", 4000)
        if len(flux) > n_window:
            time = time[:n_window]
            flux = flux[:n_window]

        light_curves.append({
            "kic":   fp.stem,
            "time":  time,
            "flux":  flux,
            "label": label,
        })

    return light_curves


def _extract_feature_matrix(lcs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Extract 35-D feature vectors for a list of light curves."""
    X = np.stack([extract_features(lc["time"], lc["flux"]) for lc in lcs])
    y = np.array([lc["label"] for lc in lcs])
    return X, y


if __name__ == "__main__":
    main()
