import json
from pathlib import Path

def fix_notebook():
    nb_path = Path("/Users/daisukeyamashiki/Code/Research/kyoto-flare-detection/notebooks/flare_create_graphs.ipynb")

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Fix: Remove the incorrect attribute references and use correct single-value attributes
    # The issue is that 'array_flare_number' and 'array_precise_obs_time' don't exist
    # They are single values 'flare_number' and 'precise_obs_time' per detector instance

    # New approach: Use flare_number and precise_obs_time (single values) and aggregate them
    # across detectors, along with existing array_starspot which IS an array attribute

    old_code_patterns = [
        "n_axis_attribute = 'array_flare_number'",
        "t_axis_attribute = 'array_precise_obs_time'",
        # Fixed display_name pattern to use star_folder_name directly since keys are like DS_Tuc_A
        "display_name = star_key.replace('_', ' ').replace(' A', '')",
    ]

    new_code_patterns = [
        "# Note: flare_number and precise_obs_time are SINGLE values per detector, not arrays",
        "# We use per-detector values, not per-observation arrays",
        "display_name = star_key.replace('_', ' ').replace(' A', '')  # DS_Tuc_A -> DS Tuc",
    ]

    # More targeted fix: rewrite the plotting logic entirely
    new_freq_cell_source = [
        "from scipy.optimize import curve_fit\n",
        "import numpy as np\n",
        "\n",
        "def power_law(x, a, b):\n",
        "    return a * x**b\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(8, 6))\n",
        "plot_successful = False\n",
        "\n",
        "color_map = {'DS Tuc': 'black', 'EK Dra': (0.8, 0.3, 0.3), 'V889 Her': (0.3, 0.3, 0.8)}\n",
        "marker_map = {'DS Tuc': 'o', 'EK Dra': 'x', 'V889 Her': '+'}\n",
        "linestyle_map = {'DS Tuc': '-', 'EK Dra': '--', 'V889 Her': ':'}\n",
        "\n",
        "if 'all_stars_detectors' in locals() and all_stars_detectors:\n",
        "    for star_key, detectors_dict in all_stars_detectors.items():\n",
        "        display_name = star_key.replace('_', ' ').replace(' A', '')\n",
        "        current_color = color_map.get(display_name, 'grey')\n",
        "        current_marker = marker_map.get(display_name, 'o')\n",
        "        current_linestyle = linestyle_map.get(display_name, '-')\n",
        "\n",
        "        x_all, y_all, yerr_all = [], [], []\n",
        "\n",
        "        # Each detector represents ONE observation period\n",
        "        # Use 'starspot' (single value), 'flare_number' (single value), 'precise_obs_time' (single value)\n",
        "        for det in detectors_dict.values():\n",
        "            if hasattr(det, 'starspot') and hasattr(det, 'flare_number') and hasattr(det, 'precise_obs_time'):\n",
        "                x_val = float(det.starspot)\n",
        "                n_val = float(det.flare_number)\n",
        "                t_val = float(det.precise_obs_time)\n",
        "\n",
        "                if x_val > 0 and t_val > 0:\n",
        "                    y_val = n_val / t_val  # frequency\n",
        "                    yerr_val = np.sqrt(n_val + 1.0) / t_val  # Poisson error\n",
        "                    x_all.append(x_val / 1e17)  # scale\n",
        "                    y_all.append(y_val)\n",
        "                    yerr_all.append(yerr_val)\n",
        "\n",
        "        if len(x_all) > 1:\n",
        "            x_all, y_all, yerr_all = np.array(x_all), np.array(y_all), np.array(yerr_all)\n",
        "            # Filter for log plotting\n",
        "            mask = (x_all > 0) & (y_all > 0)\n",
        "            x_plot, y_plot, yerr_plot = x_all[mask], y_all[mask], yerr_all[mask]\n",
        "            \n",
        "            ax.errorbar(x_plot, y_plot, yerr=yerr_plot, fmt=current_marker, color=current_color,\n",
        "                        label=f'{display_name} Data', capsize=2, elinewidth=1.2, markersize=8)\n",
        "\n",
        "            try:\n",
        "                popt, pcov = curve_fit(power_law, x_plot, y_plot, sigma=yerr_plot, absolute_sigma=True)\n",
        "                x_fit = np.linspace(x_plot.min(), x_plot.max(), 200)\n",
        "                ax.plot(x_fit, power_law(x_fit, *popt), color=current_color, linestyle=current_linestyle)\n",
        "                print(f\"{display_name}: a={popt[0]:.2e}, b={popt[1]:.2f}\")\n",
        "                plot_successful = True\n",
        "            except:\n",
        "                plot_successful = True\n",
        "\n",
        "    if plot_successful:\n",
        "        ax.set_xlabel(r\"Starspot Area [10$^{21}$ cm$^2$]\", fontsize=17)\n",
        "        ax.set_ylabel(r\"Flare Frequency (>5×10$^{33}$ erg) [per day]\", fontsize=17)\n",
        "        ax.set_xlim(0.5, 8); ax.set_ylim(0.04, 1.0)\n",
        "        ax.set_xscale('log'); ax.set_yscale('log')\n",
        "        ax.set_xticks([0.5, 1.0, 2.0, 3.0, 5.0])\n",
        "        ax.set_yticks([0.05, 0.1, 0.2, 0.3, 0.5, 1.0])\n",
        "        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f\"{x:g}\"))\n",
        "        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, pos: f\"{y:g}\"))\n",
        "        ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)*0.1))\n",
        "        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)*0.1))\n",
        "        ax.legend(loc='lower right', fontsize=15, frameon=False)\n",
        "        plt.savefig('analysis_result_freq_plot_integrated.pdf', bbox_inches='tight')\n",
        "        plt.show()\n",
    ]

    new_total_energy_source = [
        "fig, ax = plt.subplots(figsize=(8, 6))\n",
        "plot_successful = False\n",
        "\n",
        "if 'all_stars_detectors' in locals() and all_stars_detectors:\n",
        "    for star_key, detectors_dict in all_stars_detectors.items():\n",
        "        display_name = star_key.replace('_', ' ').replace(' A', '')\n",
        "        current_color = color_map.get(display_name, 'grey')\n",
        "        current_marker = marker_map.get(display_name, 'o')\n",
        "        current_linestyle = linestyle_map.get(display_name, '-')\n",
        "\n",
        "        x_data, y_data = [], []\n",
        "        # Use single-value attributes: starspot & sum_flare_energy\n",
        "        for det in detectors_dict.values():\n",
        "            if hasattr(det, 'starspot') and hasattr(det, 'sum_flare_energy'):\n",
        "                x_val = float(det.starspot)\n",
        "                y_val = float(det.sum_flare_energy)\n",
        "                if x_val > 0 and y_val > 0:\n",
        "                    x_data.append(x_val / 1e17)\n",
        "                    y_data.append(y_val / 1e35)\n",
        "\n",
        "        if len(x_data) > 1:\n",
        "            x_arr, y_arr = np.array(x_data), np.array(y_data)\n",
        "            ax.scatter(x_arr, y_arr, color=current_color, marker=current_marker, s=100, label=f\"{display_name} Data\")\n",
        "            try:\n",
        "                popt, _ = curve_fit(power_law, x_arr, y_arr)\n",
        "                x_f = np.linspace(x_arr.min(), x_arr.max(), 200)\n",
        "                ax.plot(x_f, power_law(x_f, *popt), color=current_color, linestyle=current_linestyle)\n",
        "                print(f\"{display_name}: a={popt[0]:.2e}, b={popt[1]:.2f}\")\n",
        "                plot_successful = True\n",
        "            except: plot_successful = True\n",
        "\n",
        "    if plot_successful:\n",
        "        ax.set_xlabel(r\"Starspot Area [10$^{21}$ cm$^2$]\", fontsize=17)\n",
        "        ax.set_ylabel(r\"Total Flare Energy (>5×10$^{33}$erg)[10$^{35}$ erg]\", fontsize=17)\n",
        "        ax.set_xlim(0.5, 8); ax.set_ylim(0.1, 7)\n",
        "        ax.set_xscale('log'); ax.set_yscale('log')\n",
        "        ax.set_xticks([0.5, 1.0, 2.0, 3.0, 5.0])\n",
        "        ax.set_yticks([0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0])\n",
        "        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f\"{x:g}\"))\n",
        "        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, pos: f\"{y:g}\"))\n",
        "        ax.legend(loc='lower right', fontsize=15, frameon=False)\n",
        "        plt.savefig('analysis_result_totalene_plot_integrated.pdf', bbox_inches='tight')\n",
        "        plt.show()\n",
    ]

    new_max_energy_source = [
        "fig, ax = plt.subplots(figsize=(8, 6))\n",
        "plot_successful = False\n",
        "\n",
        "if 'all_stars_detectors' in locals() and all_stars_detectors:\n",
        "    for star_key, detectors_dict in all_stars_detectors.items():\n",
        "        display_name = star_key.replace('_', ' ').replace(' A', '')\n",
        "        current_color = color_map.get(display_name, 'grey')\n",
        "        current_marker = marker_map.get(display_name, 'o')\n",
        "\n",
        "        x_data, y_data = [], []\n",
        "        for det in detectors_dict.values():\n",
        "            if hasattr(det, 'starspot') and hasattr(det, 'energy'):\n",
        "                x_val = float(det.starspot)\n",
        "                energies = det.energy\n",
        "                if hasattr(energies, '__len__') and len(energies) > 0:\n",
        "                    y_val = float(max(energies))\n",
        "                    if x_val > 0 and y_val > 0:\n",
        "                        x_data.append(x_val / 1e17)\n",
        "                        y_data.append(y_val / 1e34)\n",
        "\n",
        "        if len(x_data) > 1:\n",
        "            x_arr, y_arr = np.array(x_data), np.array(y_data)\n",
        "            ax.scatter(x_arr, y_arr, color=current_color, marker=current_marker, s=100, label=f\"{display_name} Data\")\n",
        "            plot_successful = True\n",
        "\n",
        "    if plot_successful:\n",
        "        ax.set_xlabel(r\"Starspot Area [10$^{21}$ cm$^2$]\", fontsize=17)\n",
        "        ax.set_ylabel(r\"Max Flare Energy [10$^{34}$ erg]\", fontsize=17)\n",
        "        ax.set_xlim(0.5, 8); ax.set_ylim(0.5, 45)\n",
        "        ax.set_xscale('log'); ax.set_yscale('log')\n",
        "        ax.set_xticks([0.5, 1.0, 2.0, 3.0, 5.0])\n",
        "        ax.set_yticks([0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0])\n",
        "        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f\"{x:g}\"))\n",
        "        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, pos: f\"{y:g}\"))\n",
        "        ax.legend(loc='lower right', fontsize=15, frameon=False)\n",
        "        plt.savefig('analysis_result_maxene_plot_integrated.pdf', bbox_inches='tight')\n",
        "        plt.show()\n",
    ]

    # Find and replace the plotting cells
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code" and cell.get("id") == "flare_frequency_advanced":
            cell["source"] = new_freq_cell_source
            cell["outputs"] = []
            print("Replaced flare_frequency_advanced cell")
        elif cell.get("cell_type") == "code" and cell.get("id") == "total_energy_advanced":
            cell["source"] = new_total_energy_source
            cell["outputs"] = []
            print("Replaced total_energy_advanced cell")
        elif cell.get("cell_type") == "code" and cell.get("id") == "max_energy_advanced":
            cell["source"] = new_max_energy_source
            cell["outputs"] = []
            print("Replaced max_energy_advanced cell")

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("Done! Fixed plotting cells to use single-value attributes (starspot, flare_number, precise_obs_time, etc.)")

if __name__ == "__main__":
    fix_notebook()
