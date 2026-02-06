
import json
from pathlib import Path

def main():
    notebook_path = Path("notebooks/flare_notebook.ipynb")
    print(f"Loading notebook: {notebook_path}")

    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    # New source for Cell 1
    # Removes redundant/incorrect imports and uses PROJECT_ROOT for path
    new_source = [
        '# モジュールは Cell 0 でインポート済みのもの (ek_module, ds_module, v889_module) を使用します\n',
        '\n',
        '# =========================================================\n',
        '# フォルダ名（実際のディレクトリ名）と、表示名（ラベル）を分けて管理\n',
        '# folder: os.listdir(BASE_STARS_FOLDER) で出てくる名前と一致させる\n',
        '# label : 図やログで見せたい名前（スペース入りなど自由）\n',
        '# =========================================================\n',
        'STAR_INFO = {\n',
        '    "EKDra":    {"class": ek_module.FlareDetector,   "label": "EK Dra"},\n',
        '    "V889 Her": {"class": v889_module.FlareDetector, "label": "V889 Her"},\n',
        '    "DS Tuc":   {"class": ds_module.FlareDetector,   "label": "DS Tuc"},\n',
        '    # 他の星も追加するなら同様に:\n',
        '    # "FolderName": {"class": some_module.FlareDetector, "label": "Nice Label"},\n',
        '}\n',
        '\n',
        '# 星のデータが格納されている親フォルダ\n',
        '# data/TESS/(starname) の形に合わせてパスを設定\n',
        '# PROJECT_ROOT は Cell 0 で定義済み\n',
        'BASE_STARS_FOLDER = PROJECT_ROOT / "data" / "TESS"\n',
        '\n',
        'print(f"Base stars folder set to: {BASE_STARS_FOLDER}")\n'
    ]

    # Update Cell 1 (index 1)
    nb['cells'][1]['source'] = new_source
    nb['cells'][1]['outputs'] = []

    # Write back
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("Successfully updated Cell 1 (paths and imports)!")

if __name__ == "__main__":
    main()
