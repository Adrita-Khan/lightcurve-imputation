"""Statistical comparison and hypothesis testing module."""

from .tests import (
    wilcoxon_signed_rank,
    friedman_test,
    nemenyi_posthoc,
    bootstrap_ci,
    effect_size_cohens_d,
)
from .tables import generate_stats_table

__all__ = [
    "wilcoxon_signed_rank",
    "friedman_test",
    "nemenyi_posthoc",
    "bootstrap_ci",
    "effect_size_cohens_d",
    "generate_stats_table",
]
