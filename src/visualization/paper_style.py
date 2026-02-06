"""
Paper-style matplotlib configuration for publication-quality figures.
"""

import matplotlib.pyplot as plt

# Star color and marker definitions
STAR_COLORS = {
    "DS Tuc": "black",
    "EK Dra": (0.8, 0.3, 0.3),  # Muted red
    "V889 Her": (0.3, 0.3, 0.8),  # Muted blue
}

STAR_MARKERS = {"DS Tuc": "o", "EK Dra": "x", "V889 Her": "+"}

STAR_LINESTYLES = {"DS Tuc": "-", "EK Dra": "--", "V889 Her": ":"}

# Cumulative energy plot colors for individual stars
CUMULATIVE_COLORS = [
    "#000000",
    "#E41A1C",
    "#377EB8",
    "#4DAF4A",
    "#984EA3",
    "#FF7F00",
    "#A65628",
    "#F781BF",
    "#999999",
    "#66C2A5",
    "#FC8D62",
    "#8DA0CB",
]


def apply_paper_style():
    """
    Apply publication-quality matplotlib style settings.

    This function sets various rcParams to create figures suitable for
    academic publication, including:
    - Clean tick marks (direction='in')
    - Appropriate line widths
    - Arial font family
    - PDF-compatible font settings
    """
    plt.rcParams["xtick.major.width"] = 1.5
    plt.rcParams["ytick.major.width"] = 1.5
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["xtick.major.size"] = 7
    plt.rcParams["ytick.major.size"] = 7
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["ytick.minor.visible"] = True
    plt.rcParams["xtick.minor.width"] = 1.5
    plt.rcParams["ytick.minor.width"] = 1.5
    plt.rcParams["xtick.minor.size"] = 4
    plt.rcParams["ytick.minor.size"] = 4
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.size"] = 16


def get_star_style(star_name: str) -> dict:
    """
    Get the plotting style for a given star.

    Args:
        star_name: Name of the star (e.g., 'DS Tuc', 'EK Dra', 'V889 Her')

    Returns:
        Dictionary with 'color', 'marker', and 'linestyle' keys
    """
    return {
        "color": STAR_COLORS.get(star_name, "grey"),
        "marker": STAR_MARKERS.get(star_name, "o"),
        "linestyle": STAR_LINESTYLES.get(star_name, "-"),
    }
