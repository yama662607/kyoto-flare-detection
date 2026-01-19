import json
import os

notebook_path = r"c:\Users\81803\Documents\フレア\kyoto-flare-detection\notebooks\flare_notebook_integrated.ipynb"

# The new code to inject into the cell
new_source_code = [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.ticker as mticker\n",
    "from scipy.stats import pearsonr\n",
    "\n",
    "# --- 論文用のスタイル設定 ---\n",
    "plt.rcParams['xtick.major.width'] = 1.5\n",
    "plt.rcParams['ytick.major.width'] = 1.5\n",
    "plt.rcParams['axes.linewidth'] = 1.5\n",
    "plt.rcParams['xtick.major.size'] = 7\n",
    "plt.rcParams['ytick.major.size'] = 7\n",
    "plt.rcParams[\"xtick.minor.visible\"] = True\n",
    "plt.rcParams[\"ytick.minor.visible\"] = True\n",
    "plt.rcParams['xtick.minor.width'] = 1.5\n",
    "plt.rcParams['ytick.minor.width'] = 1.5\n",
    "plt.rcParams['xtick.minor.size'] = 4\n",
    "plt.rcParams['ytick.major.size'] = 7\n",
    "plt.rcParams['ytick.minor.size'] = 4\n",
    "plt.rcParams['xtick.direction'] = 'in'\n",
    "plt.rcParams['ytick.direction'] = 'in'\n",
    "plt.rcParams['font.family'] = 'Arial'\n",
    "plt.rcParams['pdf.fonttype'] = 42\n",
    "plt.rcParams['ps.fonttype'] = 42\n",
    "plt.rcParams['font.size'] = 16\n",
    "\n",
    "# --- ここからグラフ作成処理 ---\n",
    "x_axis_attribute = 'array_starspot'\n",
    "y_axis_attribute = 'array_max_energy'\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(8, 6))\n",
    "plot_successful = False\n",
    "\n",
    "if 'all_stars_detectors' in locals() and all_stars_detectors:\n",
    "    color_map = {\n",
    "        'DS Tuc': 'black',\n",
    "        'EK Dra': (0.8, 0.3, 0.3),\n",
    "        'V889 Her': (0.3, 0.3, 0.8)\n",
    "    }\n",
    "    marker_map = {'DS Tuc': 'o', 'EK Dra': 'x', 'V889 Her': '+'}\n",
    "    marker_size = 100\n",
    "    linestyle_map = {'DS Tuc': '-', 'EK Dra': '--', 'V889 Her': ':'}  # 使ってないけど残してOK\n",
    "\n",
    "    for star_name, detectors_dict in all_stars_detectors.items():\n",
    "        current_color = color_map.get(star_name, 'grey')\n",
    "        current_marker = marker_map.get(star_name, 'x')\n",
    "\n",
    "        x_data_for_star = []\n",
    "        y_data_for_star = []\n",
    "\n",
    "        for det_instance in detectors_dict.values():\n",
    "            if hasattr(det_instance, x_axis_attribute) and hasattr(det_instance, y_axis_attribute):\n",
    "                x_array = getattr(det_instance, x_axis_attribute)\n",
    "                y_array = getattr(det_instance, y_axis_attribute)\n",
    "\n",
    "                if isinstance(x_array, (list, np.ndarray)) and isinstance(y_array, (list, np.ndarray)) and len(x_array) == len(y_array):\n",
    "                    x_data_for_star.extend(x_array)\n",
    "                    y_data_for_star.extend(y_array)\n",
    "\n",
    "        if len(x_data_for_star) > 1:\n",
    "            x_data = np.array(x_data_for_star, dtype=float)\n",
    "            y_data = np.array(y_data_for_star, dtype=float)\n",
    "\n",
    "            # NaN/inf を除去\n",
    "            valid = np.isfinite(x_data) & np.isfinite(y_data)\n",
    "            x_data = x_data[valid]\n",
    "            y_data = y_data[valid]\n",
    "\n",
    "            # スケーリング（表示用）\n",
    "            x_data_scaled = x_data / 1e17\n",
    "            y_data_scaled = y_data / 1e34\n",
    "\n",
    "            # Pearson（相関は「スケーリングしてもしなくても同じ」だけど、ここではスケーリング後で計算）\n",
    "            r_str, p_str = \"NA\", \"NA\"\n",
    "            if len(x_data_scaled) >= 2:\n",
    "                # 定数配列だと pearsonr がエラー/警告になるので弾く\n",
    "                if np.nanstd(x_data_scaled) > 0 and np.nanstd(y_data_scaled) > 0:\n",
    "                    try:\n",
    "                        r, p = pearsonr(x_data_scaled, y_data_scaled)\n",
    "                        r_str = f\"{r:.2f}\"\n",
    "                        # pは桁が小さくなりがちなので見やすく\n",
    "                        p_str = f\"{p:.2e}\"\n",
    "                    except Exception:\n",
    "                        pass\n",
    "\n",
    "            label = f\"{star_name} (r={r_str}, p={p_str}, N={len(x_data_scaled)})\"\n",
    "\n",
    "            ax.scatter(\n",
    "                x_data_scaled,\n",
    "                y_data_scaled,\n",
    "                label=label,\n",
    "                color=current_color,\n",
    "                alpha=0.9,\n",
    "                marker=current_marker,\n",
    "                s=marker_size\n",
    "            )\n",
    "\n",
    "            plot_successful = True\n",
    "\n",
    "    if plot_successful:\n",
    "        ax.set_xlabel(r\"Starspot Area [10$^{21}$ cm$^2$]\", fontsize=17)\n",
    "        ax.set_ylabel(\"Max Flare Energy [10$^{34}$ erg]\", fontsize=17)\n",
    "\n",
    "        ax.set_xlim(0.5, 8)\n",
    "        ax.set_ylim(0.5, 45)\n",
    "\n",
    "        ax.set_yscale('log')\n",
    "        ax.set_xscale('log')\n",
    "\n",
    "        ax.set_xticks([0.5, 1.0, 2.0, 3.0, 5.0])\n",
    "        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))\n",
    "        ax.xaxis.set_minor_formatter(mticker.NullFormatter())\n",
    "\n",
    "        ax.set_yticks([0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0])\n",
    "        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))\n",
    "\n",
    "        ax.legend(loc='lower right', fontsize=15, frameon=False)\n",
    "        ax.tick_params(axis='both', which='major', labelsize=16)\n",
    "\n",
    "        fig.tight_layout()\n",
    "        plt.savefig('analysis_result_maxene_plot_pearsonr.pdf', format='pdf', bbox_inches='tight')\n",
    "        print(\"\\nPlot saved as 'analysis_result_maxene_plot.pdf'\")\n",
    "        plt.show()\n",
    "    else:\n",
    "        print(\"\\nNo data was plotted.\")\n",
    "else:\n",
    "    print(\"\\nError: 'all_stars_detectors' dictionary not found or is empty.\")\n"
]

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb_data = json.load(f)

# Find validity check (it's cell 7 in 0-indexed terms, but let's find it by content)
target_cell_index = -1
for i, cell in enumerate(nb_data['cells']):
    if cell['cell_type'] == 'code':
        source_text = "".join(cell['source'])
        if "STAR_CLASS_MAP" in source_text and "plt.figure" in source_text:
            target_cell_index = i
            break

if target_cell_index != -1:
    print(f"Found target cell at index {target_cell_index}. Patching...")
    # Preserve metadata, just change source
    nb_data['cells'][target_cell_index]['source'] = new_source_code
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb_data, f, indent=1, ensure_ascii=False)
    print("Notebook patched successfully.")
else:
    print("Target cell not found. Please verify the notebook content.")
