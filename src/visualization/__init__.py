"""
Visualization module for flare detection analysis.

This module provides publication-quality plotting functions for stellar flare analysis.
"""

from src.visualization.paper_style import apply_paper_style, STAR_COLORS, STAR_MARKERS
from src.visualization.flare_plots import (
    plot_flare_frequency,
    plot_total_energy,
    plot_max_energy,
    plot_cumulative_energy,
)

__all__ = [
    "apply_paper_style",
    "STAR_COLORS",
    "STAR_MARKERS",
    "plot_flare_frequency",
    "plot_total_energy",
    "plot_max_energy",
    "plot_cumulative_energy",
]
