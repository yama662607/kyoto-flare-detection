import json

notebook_path = r"c:\Users\81803\Documents\フレア\kyoto-flare-detection\notebooks\flare_notebook_integrated.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb_data = json.load(f)

print(f"Total cells: {len(nb_data['cells'])}")
for i, cell in enumerate(nb_data['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        print(f"--- Cell {i} (Code) ---")
        print(f"Start: {source[:50]!r}")
        print(f"End:   {source[-50:]!r}")
        
        if "Set up Project Root" in source:
            print("[TYPE] Root Setup")
        if "from src import" in source:
            print("[TYPE] New Imports from SRC")
        if "STAR_INFO =" in source:
            print("[TYPE] Star Info / Data Loading")
        if "import flarepy_DS_Tuc" in source and "from src" not in source:
             print("[TYPE] Old Bad Import")
        
        print("----------------")
