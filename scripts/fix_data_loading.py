import json

notebook_path = r"c:\Users\81803\Documents\フレア\kyoto-flare-detection\notebooks\flare_notebook_integrated.ipynb"

# Complete Data Loading Cell - will be inserted after the imports cell
data_loading_source = [
    "# =========================================================\n",
    "# Star Configuration and Data Loading\n",
    "# =========================================================\n",
    "STAR_INFO = {\n",
    "    \"EKDra\":    {\"class\": ek_module.FlareDetector_EK_Dra,   \"label\": \"EK Dra\"},\n",
    "    \"V889 Her\": {\"class\": v889_module.FlareDetector_V889_Her, \"label\": \"V889 Her\"},\n",
    "    \"DS Tuc\":   {\"class\": ds_module.FlareDetector_DS_Tuc_A,   \"label\": \"DS Tuc\"},\n",
    "}\n",
    "\n",
    "# Legacy compatibility\n",
    "STAR_CLASS_MAP = {k: v['class'] for k, v in STAR_INFO.items()}\n",
    "\n",
    "BASE_STARS_FOLDER = \"C:/Users/81803/Documents/フレア/Kyoto_Student_Flare_Project/all_stars/\"\n",
    "all_stars_detectors = {}\n",
    "\n",
    "print(f\"Loading star data from: {BASE_STARS_FOLDER}\")\n",
    "\n",
    "if not os.path.exists(BASE_STARS_FOLDER):\n",
    "    print(f\"ERROR: Base folder not found!\")\n",
    "else:\n",
    "    for star_folder_name in os.listdir(BASE_STARS_FOLDER):\n",
    "        star_folder_path = os.path.join(BASE_STARS_FOLDER, star_folder_name)\n",
    "        if not os.path.isdir(star_folder_path):\n",
    "            continue\n",
    "        if star_folder_name not in STAR_INFO:\n",
    "            print(f\"Skipping folder: {star_folder_name}\")\n",
    "            continue\n",
    "\n",
    "        TargetClass = STAR_INFO[star_folder_name][\"class\"]\n",
    "        star_label = STAR_INFO[star_folder_name][\"label\"]\n",
    "        print(f\"Processing: {star_label}\")\n",
    "\n",
    "        star_detectors = {}\n",
    "        for fname in os.listdir(star_folder_path):\n",
    "            if not fname.endswith(\".fits\"):\n",
    "                continue\n",
    "            file_path = os.path.join(star_folder_path, fname)\n",
    "            s_number = None\n",
    "            for part in fname.split(\"-\"):\n",
    "                if part.startswith(\"s\") and len(part) >= 5 and part[1:5].isdigit():\n",
    "                    s_number = part[:5]\n",
    "                    break\n",
    "            if s_number:\n",
    "                try:\n",
    "                    det = TargetClass(file=file_path, process_data=True, ene_thres_low=5e33, ene_thres_high=1e39)\n",
    "                    if hasattr(det, 'tessBJD') and det.tessBJD is not None and len(det.tessBJD) > 0:\n",
    "                        star_detectors[f\"detector_{s_number}\"] = det\n",
    "                except Exception as e:\n",
    "                    print(f\"  Error: {fname}: {e}\")\n",
    "\n",
    "        if star_detectors:\n",
    "            sorted_dets = dict(sorted(star_detectors.items(), key=lambda x: x[1].tessBJD[0] if x[1].tessBJD is not None and len(x[1].tessBJD) > 0 else float('inf')))\n",
    "            all_stars_detectors[star_label] = sorted_dets\n",
    "            print(f\"  Loaded {len(sorted_dets)} sectors\")\n",
    "\n",
    "print(f\"\\nTotal stars loaded: {len(all_stars_detectors)}\")\n",
    "print(f\"Stars: {list(all_stars_detectors.keys())}\")\n"
]

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find insertion point: after cell with "Successfully imported star modules"
# That's the imports cell at index 1
# Cell 0: Path setup
# Cell 1: Imports 
# Cell 2: Should be data loading but currently is debug or graph cell

# Check if cell 2 already has STAR_INFO - if not, insert
cell2_source = "".join(nb['cells'][2]['source']) if len(nb['cells']) > 2 else ""

if "STAR_INFO" in cell2_source and "for star_folder_name" in cell2_source:
    print("Data loading cell already exists at index 2")
else:
    print("Inserting data loading cell at index 2")
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "data_loading_cell",
        "metadata": {},
        "outputs": [],
        "source": data_loading_source
    }
    nb['cells'].insert(2, new_cell)
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Cell inserted and notebook saved")

print("Done")
