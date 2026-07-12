#!/usr/bin/env python3
"""
Download and preprocess Kepler variable-star light curves.

Fetches PDCSAP flux from MAST via lightkurve, applies the preprocessing
pipeline (Algorithm 1), and saves each light curve as a parquet file in
data/processed/.

Usage:
    python scripts/download_data.py --config configs/experiment.yaml
    python scripts/download_data.py --n_workers 4 --max_per_class 200
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.download import build_label_catalogue, download_light_curve, load_light_curve
from src.data.preprocessing import (
    preprocess_light_curve,
    compute_completeness,
    truncate_to_window,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("download_data")


def parse_args():
    p = argparse.ArgumentParser(description="Download Kepler light curves.")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--n_workers", type=int, default=1, help="Parallel workers.")
    p.add_argument("--max_per_class", type=int, default=None,
                   help="Max targets per class (useful for quick testing).")
    p.add_argument("--raw_dir",  default="data/raw")
    p.add_argument("--proc_dir", default="data/processed")
    p.add_argument("--cache_dir", default="data/cache")
    return p.parse_args()


def process_one(row, raw_dir, proc_dir, cfg):
    """Download, preprocess, and save one light curve."""
    kic   = int(row["KIC"])
    label = int(row["class_label"])
    out_path = Path(proc_dir) / f"KIC{kic:09d}.parquet"

    if out_path.exists():
        return True  # already processed

    # Download
    raw_path = download_light_curve(
        kic, raw_dir,
        completeness_threshold=cfg["data"]["completeness_threshold"],
    )
    if raw_path is None:
        return False

    # Load raw data
    df_raw = load_light_curve(raw_path)

    # Preprocess
    try:
        df_clean = preprocess_light_curve(df_raw)
    except Exception as exc:
        logger.debug("KIC %d preprocessing failed: %s", kic, exc)
        return False

    # Completeness check after preprocessing
    if compute_completeness(df_clean) < cfg["data"]["completeness_threshold"]:
        return False

    # Truncate to analysis window
    window = cfg["data"]["analysis_window"]
    df_clean = truncate_to_window(df_clean, window_size=window)

    # Add label and save
    df_clean["class_label"] = label
    df_clean["KIC"]         = kic
    df_clean.to_parquet(out_path, index=False)
    return True


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    raw_dir  = Path(args.raw_dir)
    proc_dir = Path(args.proc_dir)
    cache_dir = Path(args.cache_dir)
    for d in [raw_dir, proc_dir, cache_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Build label catalogue
    logger.info("Building label catalogue …")
    catalogue = build_label_catalogue(cache_dir)
    logger.info("Catalogue: %d labelled targets", len(catalogue))

    # Optionally limit per class
    if args.max_per_class:
        catalogue = (
            catalogue.groupby("class_label", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), args.max_per_class), random_state=42))
            .reset_index(drop=True)
        )
        logger.info("After limiting: %d targets", len(catalogue))

    rows = catalogue.to_dict("records")

    # Download and process
    logger.info("Processing %d targets (workers=%d) …", len(rows), args.n_workers)

    if args.n_workers > 1:
        results = Parallel(n_jobs=args.n_workers)(
            delayed(process_one)(row, raw_dir, proc_dir, cfg)
            for row in tqdm(rows, desc="Downloading")
        )
    else:
        results = [
            process_one(row, raw_dir, proc_dir, cfg)
            for row in tqdm(rows, desc="Downloading")
        ]

    n_ok   = sum(results)
    n_fail = len(results) - n_ok
    logger.info("Done. Accepted: %d | Rejected/failed: %d", n_ok, n_fail)

    # Summary by class
    class_names = {v["label"]: k for k, v in cfg["data"]["classes"].items()}
    available = list(proc_dir.glob("*.parquet"))
    labels_found = []
    for fp in available:
        try:
            df = pd.read_parquet(fp, columns=["class_label"])
            labels_found.append(int(df["class_label"].iloc[0]))
        except Exception:
            pass

    from collections import Counter
    counts = Counter(labels_found)
    logger.info("Available per class:")
    for label, cnt in sorted(counts.items()):
        logger.info("  %s: %d", class_names.get(label, str(label)), cnt)


if __name__ == "__main__":
    main()
