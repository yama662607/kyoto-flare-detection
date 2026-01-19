import json
import os

notebook_path = r"c:\Users\81803\Documents\フレア\kyoto-flare-detection\notebooks\flare_notebook_integrated.ipynb"

# Data Loading Cell Content
data_loading_source = [
    "# =========================================================\n",
    "# フォルダ名（実際のディレクトリ名）と、表示名（ラベル）を分けて管理\n",
    "# folder: os.listdir(BASE_STARS_FOLDER) で出てくる名前と一致させる\n",
    "# label : 図やログで見せたい名前（スペース入りなど自由）\n",
    "# =========================================================\n",
    "STAR_INFO = {\n",
    "    \"EKDra\":    {\"class\": ek_module.FlareDetector_EK_Dra,   \"label\": \"EK Dra\"},\n",
    "    \"V889 Her\": {\"class\": v889_module.FlareDetector_V889_Her, \"label\": \"V889 Her\"},\n",
    "    \"DS Tuc\":   {\"class\": ds_module.FlareDetector_DS_Tuc_A,   \"label\": \"DS Tuc\"},\n",
    "}\n",
    "\n",
    "# Create legacy STAR_CLASS_MAP for compatibility with older cells if any\n",
    "STAR_CLASS_MAP = {k: v['class'] for k, v in STAR_INFO.items()}\n",
    "\n",
    "# 星のデータが格納されている親フォルダ\n",
    "BASE_STARS_FOLDER = \"C:/Users/81803/Documents/フレア/Kyoto_Student_Flare_Project/all_stars/\"\n",
    "\n",
    "# 全ての星の解析結果を格納する辞書\n",
    "all_stars_detectors = {}\n",
    "\n",
    "print(f\"Checking stars folder: {BASE_STARS_FOLDER}\")\n",
    "\n",
    "if not os.path.exists(BASE_STARS_FOLDER):\n",
    "    print(f\"Error: Base stars folder '{BASE_STARS_FOLDER}' not found.\")\n",
    "else:\n",
    "    folder_list = os.listdir(BASE_STARS_FOLDER)\n",
    "    print(f\"Found folders: {folder_list}\")\n",
    "    \n",
    "    for star_folder_name in folder_list:\n",
    "        star_folder_path = os.path.join(BASE_STARS_FOLDER, star_folder_name)\n",
    "\n",
    "        if not os.path.isdir(star_folder_path):\n",
    "            continue\n",
    "\n",
    "        # Normalize name for matching if needed, but here we try exact match first\n",
    "        if star_folder_name not in STAR_INFO:\n",
    "            # Try removing spaces or similar if simple match fails, or just warn\n",
    "            # For V889 Her, folder might be \"V889 Her\" or \"V889_Her\"\n",
    "            # Let's try to find a key that is contained in the folder name or vice versa?\n",
    "            # For now, precise match required based on my dict above.\n",
    "            print(f\"Warning: Folder '{star_folder_name}' not in STAR_INFO keys. Skipping.\")\n",
    "            continue\n",
    "\n",
    "        TargetFlareDetectorClass = STAR_INFO[star_folder_name][\"class\"]\n",
    "        star_label = STAR_INFO[star_folder_name][\"label\"]\n",
    "\n",
    "        print(f\"Processing star: {star_label} (folder='{star_folder_name}')\")\n",
    "\n",
    "        star_specific_detectors = {}\n",
    "        file_list = [os.path.join(star_folder_path, fname) for fname in os.listdir(star_folder_path)]\n",
    "\n",
    "        for file_path in file_list:\n",
    "            if not file_path.endswith(\".fits\"):\n",
    "                continue\n",
    "\n",
    "            file_name_only = os.path.basename(file_path)\n",
    "            s_number = None\n",
    "            for part in file_name_only.split(\"-\"):\n",
    "                if part.startswith(\"s\") and len(part) >= 5 and part[1:5].isdigit():\n",
    "                    s_number = part[:5]\n",
    "                    break\n",
    "\n",
    "            if s_number:\n",
    "                instance_name = f\"detector_{s_number}\"\n",
    "                try:\n",
    "                    # Common params\n",
    "                    detector_params = {\n",
    "                        \"ene_thres_high\": 1e39\n",
    "                    }\n",
    "                    # Instantiate\n",
    "                    detector = TargetFlareDetectorClass(\n",
    "                        file=file_path,\n",
    "                        process_data=True,\n",
    "                        ene_thres_low=5e33,\n",
    "                        **detector_params\n",
    "                    )\n",
    "\n",
    "                    # Only add if valid data\n",
    "                    # Assuming process_data populates tessBJD\n",
    "                    if hasattr(detector, 'tessBJD') and detector.tessBJD is not None and len(detector.tessBJD) > 0:\n",
    "                        star_specific_detectors[instance_name] = detector\n",
    "                    else:\n",
    "                        print(f\"  Warning: No valid data for {file_name_only}\")\n",
    "\n",
    "                except Exception as e:\n",
    "                    print(f\"  Error processing file {file_name_only}: {e}\")\n",
    "                    continue\n",
    "\n",
    "        if star_specific_detectors:\n",
    "            try:\n",
    "                sorted_star_detectors = dict(sorted(\n",
    "                    star_specific_detectors.items(),\n",
    "                    key=lambda x: x[1].tessBJD[0] if (hasattr(x[1], 'tessBJD') and x[1].tessBJD is not None and len(x[1].tessBJD) > 0) else float('inf')\n",
    "                ))\n",
    "                all_stars_detectors[star_label] = sorted_star_detectors\n",
    "                print(f\"  Loaded {len(sorted_star_detectors)} detectors for {star_label}\")\n",
    "            except Exception as e:\n",
    "                print(f\"  Error sorting detectors for {star_label}: {e}\")\n",
    "                all_stars_detectors[star_label] = star_specific_detectors\n",
    "        else:\n",
    "            print(f\"  No valid FITS files processed for star: {star_label}\")\n",
    "            \n",
    "print(f\"Total stars loaded: {len(all_stars_detectors)}\")\n"
]

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb_data = json.load(f)

# Insert after Cell 2 (Imports)
# List is 0-indexed.
# Cell 0: Root Setup
# Cell 1: Imports
# We want to be at index 2.
insertion_index = 2

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": data_loading_source
}

nb_data['cells'].insert(insertion_index, new_cell)
print(f"Inserted Data Loading cell at index {insertion_index}")

# Verify no duplicate STAR_INFO cells
count = 0
for cell in nb_data['cells']:
    if "STAR_INFO =" in "".join(cell['source']):
        count += 1
print(f"STAR_INFO cell count: {count}")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb_data, f, indent=1, ensure_ascii=False)

print("Fix complete.")
