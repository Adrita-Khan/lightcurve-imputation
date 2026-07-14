"""Result serialisation and table export utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_results(df: pd.DataFrame, path: str | Path) -> None:
    """Serialise a results DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
    path : str or Path
        Destination path (parent directories created automatically).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_results(path: str | Path) -> pd.DataFrame:
    """Load a results CSV produced by :func:`save_results`."""
    return pd.read_csv(path)


def save_table(
    df: pd.DataFrame,
    path: str | Path,
    fmt: str = "latex",
    float_format: str = "%.4f",
    caption: str = "",
    label: str = "",
) -> None:
    """Export a DataFrame as a LaTeX or CSV table.

    Parameters
    ----------
    df : pd.DataFrame
    path : str or Path
        Output path (extension determines format if ``fmt`` is ``'auto'``).
    fmt : str
        ``'latex'``, ``'csv'``, or ``'markdown'``.
    float_format : str
        Printf-style format string for floats.
    caption, label : str
        LaTeX caption and label (used when ``fmt='latex'``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "latex":
        latex_str = df.to_latex(index=False, float_format=float_format, escape=True)
        # Inject caption and label if provided
        if caption or label:
            label_line = f"\\label{{{label}}}\n" if label else ""
            caption_line = f"\\caption{{{caption}}}\n" if caption else ""
            latex_str = (
                "\\begin{table}[ht]\n\\centering\n"
                + caption_line
                + label_line
                + latex_str
                + "\\end{table}\n"
            )
        path.write_text(latex_str)

    elif fmt == "csv":
        df.to_csv(path, index=False, float_format=float_format)

    elif fmt == "markdown":
        path.write_text(df.to_markdown(index=False, floatfmt=float_format))

    else:
        raise ValueError(f"Unknown format '{fmt}'. Choose from: latex, csv, markdown.")
