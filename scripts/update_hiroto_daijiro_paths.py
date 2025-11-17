from pathlib import Path

import nbformat

BOOTSTRAP_CODE = """import os
import sys
from pathlib import Path

NOTEBOOK_DIR = Path().resolve()
PROJECT_ROOT = NOTEBOOK_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print('Project:', PROJECT_ROOT)
"""

NOTEBOOKS = {
    Path("hiroto/flare_DS_Tuc_A.ipynb"): "DS_Tuc_A",
    Path("daijiro/flare_detect_DS_Tuc_A.ipynb"): "DS_Tuc_A",
    Path("daijiro/flare_detect_EK_Dra.ipynb"): "EK_Dra",
    Path("daijiro/flare_detect_V889 Her.ipynb"): "V889_Her",
}

MODULE_IMPORTS = {
    "DS_Tuc_A": "from src.flarepy_DS_Tuc_A import FlareDetector_DS_Tuc_A",
    "EK_Dra": "from src.flarepy_EK_Dra import FlareDetector_EK_Dra",
    "V889_Her": "from src.flarepy_V889_Her import FlareDetector_V889_Her",
}

EXTRA_REPLACEMENTS = {
    Path("daijiro/flare_notebook.ipynb"): [
        (
            'BASE_STARS_FOLDER = "C:/Users/81803/Documents/フレア/Kyoto_Student_Flare_Project/all_stars/"',
            "BASE_STARS_FOLDER = PROJECT_ROOT / 'data' / 'TESS'",
        ),
        (
            "star_folder_path = os.path.join(BASE_STARS_FOLDER, star_folder_name)",
            "star_folder_path = BASE_STARS_FOLDER / star_folder_name",
        ),
        (
            "if not os.path.isdir(star_folder_path): # フォルダでなければスキップ",
            "if not star_folder_path.is_dir(): # フォルダでなければスキップ",
        ),
        (
            "file_list = [os.path.join(star_folder_path, fname) for fname in os.listdir(star_folder_path)]",
            "file_list = [star_folder_path / fname for fname in os.listdir(star_folder_path)]",
        ),
    ],
}

FOLDER_REPLACEMENTS = {
    "DS_Tuc_A": [
        (
            'folder_path = "../data/TESS/DS_Tuc_A"',
            "folder_path = PROJECT_ROOT / 'data' / 'TESS' / 'DS_Tuc_A'",
        ),
        (
            'folder_path = "./DS Tuc A/"',
            "folder_path = PROJECT_ROOT / 'data' / 'TESS' / 'DS_Tuc_A'",
        ),
    ],
    "EK_Dra": [
        (
            'folder_path = "./EKDra/"',
            "folder_path = PROJECT_ROOT / 'data' / 'TESS' / 'EK_Dra'",
        ),
    ],
    "V889_Her": [
        (
            'folder_path = "./V889 Her/"',
            "folder_path = PROJECT_ROOT / 'data' / 'TESS' / 'V889_Her'",
        ),
    ],
}


def ensure_bootstrap(nb):
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.source = BOOTSTRAP_CODE
            return


def apply_replacements(nb, replacements):
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        source = cell.source
        for old, new in replacements:
            if old in source:
                source = source.replace(old, new)
        cell.source = source


def main():
    for path, subdir in NOTEBOOKS.items():
        if not path.exists():
            continue
        nb = nbformat.read(path, as_version=4)
        ensure_bootstrap(nb)
        import_stmt = MODULE_IMPORTS[subdir]
        if not any(
            import_stmt in cell.source for cell in nb.cells if cell.cell_type == "code"
        ):
            nb.cells.insert(1, nbformat.v4.new_code_cell(import_stmt))
        apply_replacements(nb, FOLDER_REPLACEMENTS[subdir])
        nbformat.write(nb, path)
        print("updated", path)

    for path, replacements in EXTRA_REPLACEMENTS.items():
        if not path.exists():
            continue
        nb = nbformat.read(path, as_version=4)
        ensure_bootstrap(nb)
        apply_replacements(nb, replacements)
        nbformat.write(nb, path)
        print("updated", path)


if __name__ == "__main__":
    main()
