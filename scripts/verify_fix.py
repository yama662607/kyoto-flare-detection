import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import pearsonr

# --- Mock Data Setup ---
class MockDetector:
    def __init__(self, spot, energy):
        self.array_starspot = spot
        self.array_max_energy = energy

# Create mock data
all_stars_detectors = {
    'EK Dra': {
        'det1': MockDetector(np.array([1e17, 2e17]), np.array([1e34, 2e34])),
        'det2': MockDetector(np.array([3e17]), np.array([3e34]))
    },
    'DS Tuc': {
        'det1': MockDetector(np.array([4e17, 5e17]), np.array([5e34, 10e34]))
    }
}

print("Mock data setup complete.")

# --- The Code from the Notebook Cell ---
# (Pasted below to verify it runs without error in this context)

# --- 論文用のスタイル設定 ---
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.size'] = 7
plt.rcParams['ytick.major.size'] = 7
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams['xtick.minor.width'] = 1.5
plt.rcParams['ytick.minor.width'] = 1.5
plt.rcParams['xtick.minor.size'] = 4
plt.rcParams['ytick.minor.size'] = 4
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 16

# --- ここからグラフ作成処理 ---
x_axis_attribute = 'array_starspot'
y_axis_attribute = 'array_max_energy'

fig, ax = plt.subplots(figsize=(8, 6))
plot_successful = False

if 'all_stars_detectors' in locals() and all_stars_detectors:
    color_map = {
        'DS Tuc': 'black',
        'EK Dra': (0.8, 0.3, 0.3),
        'V889 Her': (0.3, 0.3, 0.8)
    }
    marker_map = {'DS Tuc': 'o', 'EK Dra': 'x', 'V889 Her': '+'}
    marker_size = 100
    linestyle_map = {'DS Tuc': '-', 'EK Dra': '--', 'V889 Her': ':'}  # 使ってないけど残してOK

    for star_name, detectors_dict in all_stars_detectors.items():
        current_color = color_map.get(star_name, 'grey')
        current_marker = marker_map.get(star_name, 'x')

        x_data_for_star = []
        y_data_for_star = []

        for det_instance in detectors_dict.values():
            if hasattr(det_instance, x_axis_attribute) and hasattr(det_instance, y_axis_attribute):
                x_array = getattr(det_instance, x_axis_attribute)
                y_array = getattr(det_instance, y_axis_attribute)

                if isinstance(x_array, (list, np.ndarray)) and isinstance(y_array, (list, np.ndarray)) and len(x_array) == len(y_array):
                    x_data_for_star.extend(x_array)
                    y_data_for_star.extend(y_array)

        if len(x_data_for_star) > 1:
            x_data = np.array(x_data_for_star, dtype=float)
            y_data = np.array(y_data_for_star, dtype=float)

            # NaN/inf を除去
            valid = np.isfinite(x_data) & np.isfinite(y_data)
            x_data = x_data[valid]
            y_data = y_data[valid]

            # スケーリング（表示用）
            x_data_scaled = x_data / 1e17
            y_data_scaled = y_data / 1e34

            # Pearson（相関は「スケーリングしてもしなくても同じ」だけど、ここではスケーリング後で計算）
            r_str, p_str = "NA", "NA"
            if len(x_data_scaled) >= 2:
                # 定数配列だと pearsonr がエラー/警告になるので弾く
                if np.nanstd(x_data_scaled) > 0 and np.nanstd(y_data_scaled) > 0:
                    try:
                        r, p = pearsonr(x_data_scaled, y_data_scaled)
                        r_str = f"{r:.2f}"
                        # pは桁が小さくなりがちなので見やすく
                        p_str = f"{p:.2e}"
                    except Exception:
                        pass

            label = f"{star_name} (r={r_str}, p={p_str}, N={len(x_data_scaled)})"

            ax.scatter(
                x_data_scaled,
                y_data_scaled,
                label=label,
                color=current_color,
                alpha=0.9,
                marker=current_marker,
                s=marker_size
            )

            plot_successful = True

    if plot_successful:
        ax.set_xlabel(r"Starspot Area [10$^{21}$ cm$^2$]", fontsize=17)
        ax.set_ylabel("Max Flare Energy [10$^{34}$ erg]", fontsize=17)

        ax.set_xlim(0.5, 8)
        ax.set_ylim(0.5, 45)

        ax.set_yscale('log')
        ax.set_xscale('log')

        ax.set_xticks([0.5, 1.0, 2.0, 3.0, 5.0])
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())

        ax.set_yticks([0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0])
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

        ax.legend(loc='lower right', fontsize=15, frameon=False)
        ax.tick_params(axis='both', which='major', labelsize=16)

        fig.tight_layout()
        plt.savefig('verification_result.pdf', format='pdf', bbox_inches='tight')
        print("\nPlot saved as 'verification_result.pdf'")
        print("VERIFICATION SUCCESS: The code executed and generated a plot.")
    else:
        print("\nNo data was plotted.")
else:
    print("\nError: 'all_stars_detectors' dictionary not found or is empty.")
