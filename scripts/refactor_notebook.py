import json
import os

notebook_path = r"c:\Users\81803\Documents\フレア\kyoto-flare-detection\notebooks\flare_notebook_integrated.ipynb"

# 1. ROOT PATH SETUP CELL
root_setup_source = [
    "import os\n",
    "import sys\n",
    "import re\n",
    "from pathlib import Path\n",
    "from pprint import pprint\n",
    "\n",
    "# Set up Project Root\n",
    "NOTEBOOK_DIR = Path().resolve()\n",
    "PROJECT_ROOT = NOTEBOOK_DIR.parent\n",
    "if str(PROJECT_ROOT) not in sys.path:\n",
    "    sys.path.insert(0, str(PROJECT_ROOT))\n",
    "\n",
    "print('Project:', PROJECT_ROOT)\n"
]

# 2. MAIN IMPORTS CELL
main_imports_source = [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import plotly.graph_objects as go\n",
    "import plotly.express as px\n",
    "from plotly.subplots import make_subplots\n",
    "\n",
    "# Import star modules from src\n",
    "try:\n",
    "    from src import flarepy_EK_Dra as ek_module\n",
    "    from src import flarepy_V889_Her as v889_module\n",
    "    from src import flarepy_DS_Tuc_A as ds_module\n",
    "    print(\"Successfully imported star modules from src.\")\n",
    "except ImportError as e:\n",
    "    print(f\"Error importing modules from src: {e}\")\n",
    "    # Fallback if src is not reachable as package (though logic above ensures it should be)\n",
    "    import flarepy_EK_Dra as ek_module\n",
    "    import flarepy_V889_Her as v889_module\n",
    "    import flarepy_DS_Tuc_A as ds_module\n"
]

# 3. DATA LOADING CELL (Updated STAR_INFO)
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
    "    # 他の星も追加するなら同様に:\n",
    "    # \"FolderName\": {\"class\": some_module.FlareDetector, \"label\": \"Nice Label\"},\n",
    "}\n",
    "\n",
    "# 星のデータが格納されている親フォルダ\n",
    "BASE_STARS_FOLDER = \"C:/Users/81803/Documents/フレア/Kyoto_Student_Flare_Project/all_stars/\"\n",
    "\n",
    "# 全ての星の解析結果を格納する辞書\n",
    "all_stars_detectors = {}\n",
    "\n",
    "if not os.path.exists(BASE_STARS_FOLDER):\n",
    "    print(f\"Error: Base stars folder '{BASE_STARS_FOLDER}' not found.\")\n",
    "    # For debugging/verification environment where local path might not exist, we skip exit()\n",
    "    # exit() \n",
    "else:\n",
    "    for star_folder_name in os.listdir(BASE_STARS_FOLDER):\n",
    "        star_folder_path = os.path.join(BASE_STARS_FOLDER, star_folder_name)\n",
    "\n",
    "        if not os.path.isdir(star_folder_path):\n",
    "            continue\n",
    "\n",
    "        if star_folder_name not in STAR_INFO:\n",
    "            print(f\"Warning: No FlareDetector class mapped for star folder '{star_folder_name}'. Skipping.\")\n",
    "            continue\n",
    "\n",
    "        TargetFlareDetectorClass = STAR_INFO[star_folder_name][\"class\"]\n",
    "        star_label = STAR_INFO[star_folder_name][\"label\"]\n",
    "\n",
    "        print(f\"Processing star: {star_label} (folder='{star_folder_name}', using {TargetFlareDetectorClass.__module__}.{TargetFlareDetectorClass.__name__})\")\n",
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
    "                    if detector.tessBJD is not None and len(detector.tessBJD) > 0:\n",
    "                        star_specific_detectors[instance_name] = detector\n",
    "                    else:\n",
    "                        print(f\"  Warning: No data or empty BJD for {file_name_only}\")\n",
    "                except Exception as e:\n",
    "                    print(f\"  Error processing file {file_name_only}: {e}\")\n",
    "                    continue\n",
    "\n",
    "        if star_specific_detectors:\n",
    "            try:\n",
    "                sorted_star_detectors = dict(sorted(\n",
    "                    star_specific_detectors.items(),\n",
    "                    key=lambda x: x[1].tessBJD[0] if (x[1].tessBJD is not None and len(x[1].tessBJD) > 0) else float('inf')\n",
    "                ))\n",
    "                all_stars_detectors[star_label] = sorted_star_detectors\n",
    "            except Exception as e:\n",
    "                print(f\"  Error sorting detectors for {star_label}: {e}\")\n",
    "                all_stars_detectors[star_label] = star_specific_detectors\n",
    "        else:\n",
    "            print(f\"  No valid FITS files processed for star: {star_label}\")\n"
]

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb_data = json.load(f)

# Insert Root Setup at the beginning
new_cell_root = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": root_setup_source
}
nb_data['cells'].insert(0, new_cell_root)

# Find old import cells and replace/merge
# We look for cells containing imports.
# Strategy: Find the cell with "import plotly" or "import numpy" and replace it with main_imports_source.
# Find the cell with "STAR_INFO =" and replace it with data_loading_source.
# Remove the cell with "try:\n    import flarepy_EK_Dra..." as it's redundant now.

cells_to_remove_indices = []
main_import_index = -1
data_loading_index = -1

for i, cell in enumerate(nb_data['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Identify Main Import Cell (The broken one usually, or just one of the import ones)
        if "import plotly.express as px" in source and "flarepy_DS_Tuc" in source:
            main_import_index = i
        
        # Identify the separate Try-Import cell
        if "try:" in source and "import flarepy_EK_Dra as ek_module" in source:
             cells_to_remove_indices.append(i)
        
        # Identify Data Loading Cell
        if "STAR_INFO =" in source:
            data_loading_index = i

# Update Main Imports
if main_import_index != -1:
    print(f"Updating Main Imports at index {main_import_index}")
    nb_data['cells'][main_import_index]['source'] = main_imports_source
else:
    # If not found, insert after root setup
    print("Main imports cell not found, inserting new one.")
    new_cell_imports = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": main_imports_source
    }
    nb_data['cells'].insert(1, new_cell_imports)
    # Adjust indices
    data_loading_index += 1
    # cells_to_remove_indices = [x+1 for x in cells_to_remove_indices] # We handle removal by reverse index anyway

# Updates Data Loading
if data_loading_index != -1:
    print(f"Updating Data Loading at index {data_loading_index}")
    nb_data['cells'][data_loading_index]['source'] = data_loading_source
else:
    print("WARNING: Data Loading Cell (STAR_INFO) not found!")

# Remove redundant cells (reverse order to keep indices valid)
for idx in sorted(cells_to_remove_indices, reverse=True):
    print(f"Removing redundant cell at index {idx}")
    nb_data['cells'].pop(idx)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb_data, f, indent=1, ensure_ascii=False)

print("Refactoring complete.")
