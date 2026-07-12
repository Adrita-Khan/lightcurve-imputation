"""
Data acquisition for Kepler variable-star light curves.

Downloads PDCSAP flux from MAST via the lightkurve package and merges
class labels from three catalogues:
  - Kepler Eclipsing Binary Catalogue (Kirk et al. 2016)
  - Kepler RR Lyrae catalogue (Nemec et al. 2013)
  - Debosscher et al. (2011) classification table

Usage:
    python -m src.data.download --config configs/experiment.yaml
"""

import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Catalogue URLs (publicly available via MAST / VizieR)
# ---------------------------------------------------------------------------
_CATALOGUE_URLS = {
    "eb": "https://keplerEBs.villanova.edu/catalog/",   # Kirk et al. 2016
    "rrlyr": "https://vizier.cds.unistra.fr/viz-bin/VizieR",
    "debosscher": "https://vizier.cds.unistra.fr/viz-bin/VizieR",
}

_CLASS_MAP = {
    "RRLYR": 0,
    "DSCT":  1,
    "EB":    2,
    "GDOR":  3,
    "SOL":   4,
    "ROT":   5,
}


def build_label_catalogue(cache_dir: Path) -> pd.DataFrame:
    """
    Merge the three Kepler variable-star catalogues into a single
    DataFrame with columns [KIC, class_label, class_name].

    Parameters
    ----------
    cache_dir : Path
        Directory where downloaded catalogue CSVs are cached.

    Returns
    -------
    pd.DataFrame
        Merged label catalogue.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "kepler_labels.csv"

    if cache_file.exists():
        logger.info("Loading cached label catalogue from %s", cache_file)
        return pd.read_csv(cache_file)

    try:
        import lightkurve as lk  # noqa: F401
    except ImportError:
        raise ImportError(
            "lightkurve is required. Install with: pip install lightkurve"
        )

    logger.info("Building label catalogue from MAST/VizieR catalogues …")
    frames = []

    # --- Eclipsing Binaries (Kirk et al. 2016) ---
    try:
        eb_df = _fetch_eb_catalogue()
        eb_df["class_name"] = "EB"
        frames.append(eb_df[["KIC", "class_name"]])
        logger.info("EB catalogue: %d targets", len(eb_df))
    except Exception as exc:
        logger.warning("Could not fetch EB catalogue: %s", exc)

    # --- RR Lyrae (Nemec et al. 2013) ---
    try:
        rl_df = _fetch_rrlyr_catalogue()
        rl_df["class_name"] = "RRLYR"
        frames.append(rl_df[["KIC", "class_name"]])
        logger.info("RR Lyrae catalogue: %d targets", len(rl_df))
    except Exception as exc:
        logger.warning("Could not fetch RR Lyrae catalogue: %s", exc)

    # --- Debosscher et al. 2011 (DSCT, GDOR, SOL, ROT) ---
    try:
        db_df = _fetch_debosscher_catalogue()
        frames.append(db_df[["KIC", "class_name"]])
        logger.info("Debosscher catalogue: %d targets", len(db_df))
    except Exception as exc:
        logger.warning("Could not fetch Debosscher catalogue: %s", exc)

    if not frames:
        raise RuntimeError(
            "No catalogues could be fetched. Check your network connection."
        )

    catalogue = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset="KIC", keep="first")
        .reset_index(drop=True)
    )
    catalogue["class_label"] = catalogue["class_name"].map(_CLASS_MAP)
    catalogue.to_csv(cache_file, index=False)
    logger.info("Label catalogue saved to %s (%d targets)", cache_file, len(catalogue))
    return catalogue


def _fetch_eb_catalogue() -> pd.DataFrame:
    """Fetch Kepler EB catalogue (Kirk et al. 2016) via astroquery."""
    from astroquery.vizier import Vizier
    v = Vizier(columns=["KIC"], row_limit=-1)
    result = v.get_catalogs("J/AJ/151/68")
    if result:
        tbl = result[0]
        return pd.DataFrame({"KIC": tbl["KIC"].data.astype(int)})
    return pd.DataFrame(columns=["KIC"])


def _fetch_rrlyr_catalogue() -> pd.DataFrame:
    """Fetch Kepler RR Lyrae catalogue (Nemec et al. 2013) via astroquery."""
    from astroquery.vizier import Vizier
    v = Vizier(columns=["KIC"], row_limit=-1)
    result = v.get_catalogs("J/ApJ/773/181")
    if result:
        tbl = result[0]
        return pd.DataFrame({"KIC": tbl["KIC"].data.astype(int)})
    return pd.DataFrame(columns=["KIC"])


def _fetch_debosscher_catalogue() -> pd.DataFrame:
    """
    Fetch Debosscher et al. (2011) classification table via astroquery.
    Maps their class labels to our six-class scheme.
    """
    from astroquery.vizier import Vizier
    _debosscher_map = {
        "DSCT": "DSCT",
        "SX Phe": "DSCT",
        "GDOR": "GDOR",
        "solar-like": "SOL",
        "spotted": "ROT",
        "rotational": "ROT",
    }
    v = Vizier(columns=["KIC", "Class"], row_limit=-1)
    result = v.get_catalogs("J/A+A/534/A125")
    if not result:
        return pd.DataFrame(columns=["KIC", "class_name"])
    tbl = result[0]
    df = pd.DataFrame({"KIC": tbl["KIC"].data.astype(int),
                        "raw_class": [str(c) for c in tbl["Class"].data]})
    df["class_name"] = df["raw_class"].map(_debosscher_map)
    return df.dropna(subset=["class_name"])[["KIC", "class_name"]]


def download_light_curve(
    kic: int,
    raw_dir: Path,
    completeness_threshold: float = 0.95,
    cadence: str = "long",
) -> str | None:
    """
    Download all available Kepler quarters for a single KIC target and
    save as a parquet file.

    Parameters
    ----------
    kic : int
        Kepler Input Catalog identifier.
    raw_dir : Path
        Directory where individual light curve files are cached.
    completeness_threshold : float
        Minimum fraction of non-NaN cadences required for inclusion.
    cadence : str
        'long' (29.4 min) or 'short' (1 min).

    Returns
    -------
    str | None
        Path to the saved parquet file, or None if the target is rejected.
    """
    import lightkurve as lk

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / f"KIC{kic:09d}.parquet"

    if out_path.exists():
        return str(out_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            search = lk.search_lightcurve(
                f"KIC {kic}", mission="Kepler", cadence=cadence, author="Kepler"
            )
            if len(search) == 0:
                return None
            lc_coll = search.download_all(quality_bitmask="default")
            if lc_coll is None or len(lc_coll) == 0:
                return None
            lc = lc_coll.stitch()
        except Exception as exc:
            logger.debug("KIC %d download failed: %s", kic, exc)
            return None

    # Completeness check before preprocessing
    n_total = len(lc)
    n_valid = int(np.sum(np.isfinite(lc.flux.value)))
    if n_total == 0 or n_valid / n_total < completeness_threshold:
        logger.debug("KIC %d rejected: completeness %.3f", kic, n_valid / n_total)
        return None

    df = pd.DataFrame({
        "time": lc.time.value,
        "flux": lc.flux.value,
        "flux_err": lc.flux_err.value if hasattr(lc, "flux_err") else np.full(n_total, np.nan),
    })
    df.to_parquet(out_path, index=False)
    return str(out_path)


def load_light_curve(path: str | Path) -> pd.DataFrame:
    """Load a saved light curve parquet file."""
    return pd.read_parquet(path)
