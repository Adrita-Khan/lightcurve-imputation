"""YAML configuration loading and merging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    path : str or Path
        Path to the ``.yml`` or ``.yaml`` configuration file.

    Returns
    -------
    dict
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as fh:
        cfg = yaml.safe_load(fh)
    if cfg is None:
        cfg = {}
    return cfg


def merge_configs(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base``.

    Keys in ``override`` that are dicts are merged recursively;
    all other keys replace the base value.

    Parameters
    ----------
    base : dict
    override : dict

    Returns
    -------
    dict
    """
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = merge_configs(result[key], val)
        else:
            result[key] = val
    return result
