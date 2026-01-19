import json

notebook_path = r"c:\Users\81803\Documents\フレア\kyoto-flare-detection\notebooks\flare_notebook_integrated.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb_data = json.load(f)

print(f"Total cells: {len(nb_data['cells'])}")
for i, cell in enumerate(nb_data['cells'][:5]):
    print(f"--- Cell {i} ---")
    source = "".join(cell['source'])[:200]
    print(source)
    print("----------------")
