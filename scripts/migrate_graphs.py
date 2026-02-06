#!/usr/bin/env python3
"""
Migrate additional graphs from legacy/flare_notebook.ipynb to notebooks/flare_create_graphs.ipynb.
Adds:
1. Max Energy with Pearson correlation
2. Cumulative Energy plots for each star (EK Dra, DS Tuc, V889 Her)
"""
import json
from pathlib import Path


def create_max_energy_pearson_cell():
    """Create cell for Max Energy plot with Pearson correlation."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": "max_energy_pearson",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Max Flare Energy with Pearson Correlation\n",
            "from scipy.stats import pearsonr\n",
            "\n",
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
            "            \n",
            "            # Calculate Pearson correlation\n",
            "            r_str, p_str = 'NA', 'NA'\n",
            "            if len(x_arr) >= 2 and np.nanstd(x_arr) > 0 and np.nanstd(y_arr) > 0:\n",
            "                try:\n",
            "                    r, p = pearsonr(x_arr, y_arr)\n",
            "                    r_str, p_str = f'{r:.2f}', f'{p:.2e}'\n",
            "                except: pass\n",
            "            \n",
            "            label = f\"{display_name} (r={r_str}, p={p_str}, N={len(x_arr)})\"\n",
            "            ax.scatter(x_arr, y_arr, color=current_color, marker=current_marker, s=100, label=label)\n",
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
            "        ax.legend(loc='lower right', fontsize=12, frameon=False)\n",
            "        plt.savefig('analysis_result_maxene_plot_pearsonr.pdf', bbox_inches='tight')\n",
            "        plt.show()\n",
        ]
    }


def create_cumulative_energy_cell(star_key: str, star_display: str, output_filename: str):
    """Create cell for Cumulative Energy plot for a specific star."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": f"cumulative_energy_{star_key.lower().replace(' ', '_')}",
        "metadata": {},
        "outputs": [],
        "source": [
            f"# Cumulative Flare Energy - {star_display}\n",
            "\n",
            "colors_cum = [\n",
            "    '#000000', '#E41A1C', '#377EB8', '#4DAF4A',\n",
            "    '#984EA3', '#FF7F00', '#A65628', '#F781BF',\n",
            "    '#999999', '#66C2A5', '#FC8D62', '#8DA0CB',\n",
            "]\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(8, 6))\n",
            "\n",
            f"if '{star_key}' in all_stars_detectors:\n",
            f"    detector_items = list(all_stars_detectors['{star_key}'].items())\n",
            "    \n",
            "    for i, (det_name, det) in enumerate(detector_items):\n",
            "        if det.energy is None or len(det.energy) == 0:\n",
            "            continue\n",
            "\n",
            "        color = colors_cum[i % len(colors_cum)]\n",
            "        energy_cor = np.sort(det.energy)\n",
            "        cumenergy = np.array([len(energy_cor) - j for j in range(len(energy_cor))])\n",
            "        rate = cumenergy / det.precise_obs_time\n",
            "\n",
            "        # det_name is 'detector_s0001' -> 'Sector 1'\n",
            "        try:\n",
            "            s = det_name.split('_')[-1]  # 's0001'\n",
            "            sector_num = int(s[1:])  # 1\n",
            "            label = f'Sector {sector_num}'\n",
            "        except:\n",
            "            label = det_name\n",
            "\n",
            "        ax.step(energy_cor / 1e33, rate, where='post', color=color, linewidth=1.8, label=label)\n",
            "\n",
            "    ax.set_xscale('log'); ax.set_yscale('log')\n",
            "    ax.set_xlabel(r'Flare Energy [10$^{33}$erg]', fontsize=17)\n",
            "    ax.set_ylabel(r'Cumulative Number [day$^{-1}$]', fontsize=17)\n",
            "    ax.tick_params(labelsize=16)\n",
            "    ax.axvline(x=5, color='black', linestyle='dotted', linewidth=1.5, label='Energy threshold', zorder=0)\n",
            "    ax.set_xticks([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])\n",
            "    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))\n",
            "    ax.xaxis.set_minor_formatter(mticker.NullFormatter())\n",
            "    ax.set_yticks([0.05, 0.1, 0.2, 0.3, 0.5, 1])\n",
            "    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))\n",
            "    leg = ax.legend(loc='lower left', fontsize=10, frameon=True)\n",
            "    leg.get_frame().set_alpha(0)\n",
            "    plt.tight_layout()\n",
            f"    plt.savefig('{output_filename}', format='pdf', bbox_inches='tight')\n",
            "    plt.show()\n",
        ]
    }


def main():
    nb_path = Path("/Users/daisukeyamashiki/Code/Research/kyoto-flare-detection/notebooks/flare_create_graphs.ipynb")

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Find the position to insert new cells (after existing plotting cells)
    insert_pos = len(nb["cells"])

    # Add Max Energy with Pearson correlation
    nb["cells"].append(create_max_energy_pearson_cell())
    print("Added: Max Energy with Pearson Correlation")

    # Add Cumulative Energy plots for each star
    star_configs = [
        ("EK_Dra", "EK Dra", "flare_cumenergy_EKDra.pdf"),
        ("DS_Tuc_A", "DS Tuc", "flare_cumenergy_DSTuc.pdf"),
        ("V889_Her", "V889 Her", "flare_cumenergy_V889Her.pdf"),
    ]

    for star_key, star_display, output_filename in star_configs:
        nb["cells"].append(create_cumulative_energy_cell(star_key, star_display, output_filename))
        print(f"Added: Cumulative Energy for {star_display}")

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("\nMigration complete!")


if __name__ == "__main__":
    main()
