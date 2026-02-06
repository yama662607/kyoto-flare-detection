"""
Flare analysis plotting functions.

This module provides publication-quality plotting functions for stellar flare analysis,
including scatter plots with power-law fits and cumulative energy distributions.
"""

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

from src.visualization.paper_style import (
    CUMULATIVE_COLORS,
    apply_paper_style,
    get_star_style,
)


def _power_law(x, a, b):
    """Power law function: y = a * x^b"""
    return a * x**b


def plot_flare_frequency(
    all_stars_detectors: dict[str, dict[str, Any]],
    output_filename: str = "analysis_result_freq_with_error_plot.pdf",
    show_fit: bool = True,
    show_legend: bool = True,
    figsize: tuple = (8, 6),
) -> plt.Figure | None:
    """
    Plot Flare Frequency vs Starspot Area with error bars and power-law fit.

    Args:
        all_stars_detectors: Dictionary with structure {star_name: {detector_name: detector_instance}}
        output_filename: Output PDF filename
        show_fit: Whether to show power-law fit lines
        show_legend: Whether to show legend
        figsize: Figure size as (width, height)

    Returns:
        matplotlib Figure object if successful, None otherwise
    """
    apply_paper_style()

    # Attribute names
    x_axis_attribute = "array_starspot"
    n_axis_attribute = "array_flare_number"
    t_axis_attribute = "array_precise_obs_time"

    MARKER_SIZE = 8
    CAPSIZE = 2
    ELINEWIDTH = 1.2

    fig, ax = plt.subplots(figsize=figsize)
    plot_successful = False

    for star_name, detectors_dict in all_stars_detectors.items():
        style = get_star_style(star_name)
        x_all, y_all, yerr_all = [], [], []

        for det in detectors_dict.values():
            if not all(
                hasattr(det, attr)
                for attr in [x_axis_attribute, n_axis_attribute, t_axis_attribute]
            ):
                continue

            x_arr = np.asarray(getattr(det, x_axis_attribute), dtype=float)
            N_arr = np.asarray(getattr(det, n_axis_attribute), dtype=float)
            T_arr = np.asarray(getattr(det, t_axis_attribute), dtype=float)

            if x_arr.size == 0 or N_arr.size == 0 or T_arr.size == 0:
                continue
            if not (len(x_arr) == len(N_arr) == len(T_arr)):
                continue

            # Calculate frequency and Poisson error
            y_calc = N_arr / T_arr
            yerr_arr = np.sqrt(N_arr + 1.0) / T_arr

            # Scale and mask
            x_scaled = x_arr / 1e17
            mask = (
                np.isfinite(x_scaled)
                & np.isfinite(y_calc)
                & np.isfinite(yerr_arr)
                & (x_scaled > 0)
                & (y_calc > 0)
                & (T_arr > 0)
                & (N_arr >= 0)
            )

            if np.any(mask):
                x_all.append(x_scaled[mask])
                y_all.append(y_calc[mask])
                yerr_all.append(yerr_arr[mask])

        if len(x_all) == 0:
            continue

        x_all = np.concatenate(x_all)
        y_all = np.concatenate(y_all)
        yerr_all = np.concatenate(yerr_all)

        if len(x_all) < 2:
            continue

        # Plot error bars
        ax.errorbar(
            x_all,
            y_all,
            yerr=yerr_all,
            fmt=style["marker"],
            linestyle="None",
            color=style["color"],
            markersize=MARKER_SIZE,
            capsize=CAPSIZE,
            elinewidth=ELINEWIDTH,
            alpha=0.9,
            label=f"{star_name} Data",
        )

        # Power-law fit
        if show_fit:
            try:
                popt, pcov = curve_fit(
                    _power_law,
                    x_all,
                    y_all,
                    sigma=yerr_all,
                    absolute_sigma=True,
                    maxfev=10000,
                )
                a, b = popt
                a_err, b_err = np.sqrt(np.diag(pcov))

                x_fit = np.linspace(x_all.min(), x_all.max(), 200)
                y_fit = _power_law(x_fit, a, b)

                ax.plot(
                    x_fit,
                    y_fit,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    label="_nolegend_",
                )
                plot_successful = True

                print(
                    f"{star_name}: a = {a:.2e} ± {a_err:.2e}, b = {b:.2f} ± {b_err:.2f}"
                )
            except Exception as e:
                print(f"Could not perform power-law curve_fit for {star_name}: {e}")
        else:
            plot_successful = True

    if plot_successful:
        ax.set_xlabel(r"Starspot Area [10$^{21}$ cm$^2$]", fontsize=17)
        ax.set_ylabel(r"Flare Frequency (>5×10$^{33}$ erg) [per day]", fontsize=17)

        ax.set_xlim(0.5, 8)
        ax.set_ylim(0.04, 1.0)
        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_xticks([0.5, 1.0, 2.0, 3.0, 5.0])
        ax.set_yticks([0.05, 0.1, 0.2, 0.3, 0.5, 1.0])

        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x:g}"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, pos: f"{y:g}"))

        ax.xaxis.set_minor_locator(
            mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1)
        )
        ax.yaxis.set_minor_locator(
            mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1)
        )
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())

        ax.tick_params(
            axis="both", which="both", direction="in", top=False, right=False
        )
        ax.tick_params(axis="both", which="major", length=7, width=1.5, labelsize=16)
        ax.tick_params(axis="both", which="minor", length=4, width=1.2)

        if show_legend:
            ax.legend(loc="lower right", fontsize=15, frameon=False)

        fig.tight_layout()
        plt.savefig(output_filename, format="pdf", bbox_inches="tight")
        print(f"\nPlot saved as '{output_filename}'")
        return fig
    else:
        print("\nNo data was plotted.")
        return None


def plot_total_energy(
    all_stars_detectors: dict[str, dict[str, Any]],
    output_filename: str = "analysis_result_totalene_plot.pdf",
    show_fit: bool = True,
    show_legend: bool = True,
    figsize: tuple = (8, 6),
) -> plt.Figure | None:
    """
    Plot Total Flare Energy vs Starspot Area with power-law fit.

    Args:
        all_stars_detectors: Dictionary with structure {star_name: {detector_name: detector_instance}}
        output_filename: Output PDF filename
        show_fit: Whether to show power-law fit lines
        show_legend: Whether to show legend
        figsize: Figure size as (width, height)

    Returns:
        matplotlib Figure object if successful, None otherwise
    """
    apply_paper_style()

    x_axis_attribute = "array_starspot"
    y_axis_attribute = "array_sum_energy"
    marker_size = 100

    fig, ax = plt.subplots(figsize=figsize)
    plot_successful = False

    for star_name, detectors_dict in all_stars_detectors.items():
        style = get_star_style(star_name)
        x_data_for_star = []
        y_data_for_star = []

        for det_instance in detectors_dict.values():
            if hasattr(det_instance, x_axis_attribute) and hasattr(
                det_instance, y_axis_attribute
            ):
                x_array = getattr(det_instance, x_axis_attribute)
                y_array = getattr(det_instance, y_axis_attribute)
                if (
                    isinstance(x_array, (list, np.ndarray))
                    and isinstance(y_array, (list, np.ndarray))
                    and len(x_array) == len(y_array)
                ):
                    x_data_for_star.extend(x_array)
                    y_data_for_star.extend(y_array)

        if len(x_data_for_star) > 1:
            x_data = np.array(x_data_for_star)
            y_data = np.array(y_data_for_star)
            x_data_scaled = x_data / 1e17
            y_data_scaled = y_data / 1e35

            ax.scatter(
                x_data_scaled,
                y_data_scaled,
                label=f"{star_name} Data",
                color=style["color"],
                alpha=0.9,
                marker=style["marker"],
                s=marker_size,
            )

            if show_fit:
                try:
                    mask = (x_data_scaled > 0) & (y_data_scaled > 0)
                    x_fit_data = x_data_scaled[mask]
                    y_fit_data = y_data_scaled[mask]

                    popt, pcov = curve_fit(_power_law, x_fit_data, y_fit_data)
                    a, b = popt
                    a_err, b_err = np.sqrt(np.diag(pcov))

                    x_fit = np.linspace(x_fit_data.min(), x_fit_data.max(), 200)
                    y_fit = _power_law(x_fit, a, b)

                    ax.plot(
                        x_fit, y_fit, color=style["color"], linestyle=style["linestyle"]
                    )
                    plot_successful = True

                    print(
                        f"{star_name}: a = {a:.2e} ± {a_err:.2e}, b = {b:.2f} ± {b_err:.2f}"
                    )
                except Exception as e:
                    print(f"Could not perform power-law curve_fit for {star_name}: {e}")
            else:
                plot_successful = True

    if plot_successful:
        ax.set_xlabel(r"Starspot Area [10$^{21}$ cm$^2$]", fontsize=17)
        ax.set_ylabel(
            r"Total Flare Energy (>5×10$^{33}$erg) [10$^{35}$ erg]", fontsize=17
        )

        ax.set_xlim(0.5, 8)
        ax.set_ylim(0.1, 7)
        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_xticks([0.5, 1.0, 2.0, 3.0, 5.0])
        ax.set_yticks([0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0])

        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x:g}"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, pos: f"{y:g}"))

        ax.xaxis.set_minor_locator(
            mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1)
        )
        ax.yaxis.set_minor_locator(
            mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1)
        )
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())

        ax.tick_params(
            axis="both", which="both", direction="in", top=False, right=False
        )
        ax.tick_params(axis="both", which="major", length=7, width=1.5, labelsize=16)
        ax.tick_params(axis="both", which="minor", length=4, width=1.2)

        if show_legend:
            ax.legend(loc="lower right", fontsize=15, frameon=False)

        fig.tight_layout()
        plt.savefig(output_filename, format="pdf", bbox_inches="tight")
        print(f"\nPlot saved as '{output_filename}'")
        return fig
    else:
        print("\nNo data was plotted.")
        return None


def plot_max_energy(
    all_stars_detectors: dict[str, dict[str, Any]],
    output_filename: str = "analysis_result_maxene_plot.pdf",
    show_pearson: bool = False,
    show_legend: bool = True,
    figsize: tuple = (8, 6),
) -> plt.Figure | None:
    """
    Plot Max Flare Energy vs Starspot Area.

    Args:
        all_stars_detectors: Dictionary with structure {star_name: {detector_name: detector_instance}}
        output_filename: Output PDF filename
        show_pearson: Whether to show Pearson correlation in legend
        show_legend: Whether to show legend
        figsize: Figure size as (width, height)

    Returns:
        matplotlib Figure object if successful, None otherwise
    """
    apply_paper_style()

    x_axis_attribute = "array_starspot"
    y_axis_attribute = "array_max_energy"
    marker_size = 100

    fig, ax = plt.subplots(figsize=figsize)
    plot_successful = False

    for star_name, detectors_dict in all_stars_detectors.items():
        style = get_star_style(star_name)
        x_data_for_star = []
        y_data_for_star = []

        for det_instance in detectors_dict.values():
            if hasattr(det_instance, x_axis_attribute) and hasattr(
                det_instance, y_axis_attribute
            ):
                x_array = getattr(det_instance, x_axis_attribute)
                y_array = getattr(det_instance, y_axis_attribute)
                if (
                    isinstance(x_array, (list, np.ndarray))
                    and isinstance(y_array, (list, np.ndarray))
                    and len(x_array) == len(y_array)
                ):
                    x_data_for_star.extend(x_array)
                    y_data_for_star.extend(y_array)

        if len(x_data_for_star) > 1:
            x_data = np.array(x_data_for_star, dtype=float)
            y_data = np.array(y_data_for_star, dtype=float)

            valid = np.isfinite(x_data) & np.isfinite(y_data)
            x_data = x_data[valid]
            y_data = y_data[valid]

            x_data_scaled = x_data / 1e17
            y_data_scaled = y_data / 1e34

            # Calculate Pearson correlation if requested
            label = f"{star_name} Data"
            if (
                show_pearson
                and len(x_data_scaled) >= 2
                and np.nanstd(x_data_scaled) > 0
                and np.nanstd(y_data_scaled) > 0
            ):
                try:
                    r, p = pearsonr(x_data_scaled, y_data_scaled)
                    label = (
                        f"{star_name} (r={r:.2f}, p={p:.2e}, N={len(x_data_scaled)})"
                    )
                except Exception:
                    pass

            ax.scatter(
                x_data_scaled,
                y_data_scaled,
                label=label,
                color=style["color"],
                alpha=0.9,
                marker=style["marker"],
                s=marker_size,
            )

            plot_successful = True

    if plot_successful:
        ax.set_xlabel(r"Starspot Area [10$^{21}$ cm$^2$]", fontsize=17)
        ax.set_ylabel(r"Max Flare Energy [10$^{34}$ erg]", fontsize=17)

        ax.set_xlim(0.5, 8)
        ax.set_ylim(0.5, 45)
        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_xticks([0.5, 1.0, 2.0, 3.0, 5.0])
        ax.set_yticks([0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0])

        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x:g}"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, pos: f"{y:g}"))

        ax.xaxis.set_minor_locator(
            mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1)
        )
        ax.yaxis.set_minor_locator(
            mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1)
        )
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())

        ax.tick_params(
            axis="both", which="both", direction="in", top=False, right=False
        )
        ax.tick_params(axis="both", which="major", length=7, width=1.5, labelsize=16)
        ax.tick_params(axis="both", which="minor", length=4, width=1.2)

        if show_legend:
            ax.legend(loc="lower right", fontsize=15, frameon=False)

        fig.tight_layout()
        plt.savefig(output_filename, format="pdf", bbox_inches="tight")
        print(f"\nPlot saved as '{output_filename}'")
        return fig
    else:
        print("\nNo data was plotted.")
        return None


def plot_cumulative_energy(
    detectors_dict: dict[str, Any],
    star_name: str,
    output_filename: str | None = None,
    show_threshold: bool = True,
    threshold_energy: float = 5.0,
    figsize: tuple = (8, 6),
    xlim: tuple | None = None,
    xticks: list[float] | None = None,
    yticks: list[float] | None = None,
    colors: list[str] | None = None,
) -> plt.Figure | None:
    """
    Plot cumulative flare energy distribution for a single star.

    Args:
        detectors_dict: Dictionary {detector_name: detector_instance} for a single star
        star_name: Name of the star for labeling
        output_filename: Output PDF filename. If None, defaults to f"flare_cumenergy_{star_name}.pdf"
        show_threshold: Whether to show energy threshold line
        threshold_energy: Energy threshold in units of 10^33 erg
        figsize: Figure size as (width, height)
        xlim: X-axis limits (min, max)
        xticks: List of major x-axis tick positions
        yticks: List of major y-axis tick positions
        colors: List of colors to cycle through for sectors

    Returns:
        matplotlib Figure object if successful, None otherwise
    """
    apply_paper_style()

    if output_filename is None:
        safe_name = star_name.replace(" ", "")
        output_filename = f"flare_cumenergy_{safe_name}.pdf"

    detector_items = list(detectors_dict.items())

    # Use provided colors or legacy-matching defaults per star
    if colors:
        plot_colors = colors
    else:
        star_key = star_name.replace("_", " ").strip()
        if star_key == "V889 Her":
            plot_colors = CUMULATIVE_COLORS[:4]
        elif star_key == "DS Tuc":
            plot_colors = CUMULATIVE_COLORS[:5]
        else:
            plot_colors = CUMULATIVE_COLORS

    fig, ax = plt.subplots(figsize=figsize)

    for i, (det_name, det) in enumerate(detector_items):
        if det.energy is None or len(det.energy) == 0:
            continue

        color = plot_colors[i % len(plot_colors)]

        energy_cor = np.sort(det.energy)
        cumenergy = np.array([len(energy_cor) - j for j in range(len(energy_cor))])
        rate = cumenergy / det.precise_obs_time

        # Parse sector number from detector name
        try:
            s = det_name.split("_")[-1]
            sector_num = int(s[1:])
            label = f"Sector {sector_num}"
        except Exception:
            label = det_name

        ax.step(
            energy_cor / 1e33,
            rate,
            where="post",
            color=color,
            linewidth=1.8,
            label=label,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"Flare Energy [10$^{33}$erg]", fontsize=17)
    ax.set_ylabel(r"Cumulative Number [day$^{-1}$]", fontsize=17)

    ax.tick_params(labelsize=16)

    if show_threshold:
        ax.axvline(
            x=threshold_energy,
            color="black",
            linestyle="dotted",
            linewidth=1.5,
            label="Energy threshold",
            zorder=0,
        )

    # Set custom or default X ticks
    if xticks:
        ax.set_xticks(xticks)
    else:
        ax.set_xticks([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100, 200])

    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    # Set custom or default Y ticks
    if yticks:
        ax.set_yticks(yticks)
    else:
        ax.set_yticks([0.05, 0.1, 0.2, 0.3, 0.5, 1])

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # Set X limits if provided
    if xlim:
        ax.set_xlim(xlim)

    leg = ax.legend(loc="lower left", fontsize=10, frameon=True)
    leg.get_frame().set_alpha(0)

    plt.tight_layout()
    plt.savefig(output_filename, format="pdf", bbox_inches="tight")
    print(f"\nPlot saved as '{output_filename}'")

    return fig
