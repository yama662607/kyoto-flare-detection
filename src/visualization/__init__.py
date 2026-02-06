"""
Visualization module for flare detection analysis.

This module provides publication-quality plotting functions for stellar flare analysis.
"""

from src.visualization.flare_plots import (
    plot_cumulative_energy,
    plot_flare_frequency,
    plot_max_energy,
    plot_total_energy,
)
from src.visualization.paper_style import STAR_COLORS, STAR_MARKERS, apply_paper_style

__all__ = [
    "apply_paper_style",
    "STAR_COLORS",
    "STAR_MARKERS",
    "plot_flare_frequency",
    "plot_total_energy",
    "plot_max_energy",
    "plot_cumulative_energy",
]
