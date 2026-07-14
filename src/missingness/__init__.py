"""Missing-data injection module."""

from .injector import inject_gaps, GapPattern

__all__ = ["inject_gaps", "GapPattern"]
