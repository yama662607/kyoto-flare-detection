import nbformat
from pathlib import Path

COMMON_SETUP = """import os
import sys
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

NOTEBOOK_DIR = Path().resolve()
PROJECT_ROOT = NOTEBOOK_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print('Project:', PROJECT_ROOT)
"""

STAR_CONFIG = [
    {
        "name": "DS_Tuc_A",
        "path": Path("notebooks/flare_detect_DS_Tuc_A.ipynb"),
        "module_code": "from src.flarepy_DS_Tuc_A import FlareDetector_DS_Tuc_A",
        "data_dir": "data/TESS/DS_Tuc_A",
        "list_name": "detectors_DS_Tuc_A",
        "class_name": "FlareDetector_DS_Tuc_A",
    },
    {
        "name": "EK_Dra",
        "path": Path("notebooks/flare_detect_EK_Dra.ipynb"),
        "module_code": "from src.flarepy_EK_Dra import FlareDetector_EK_Dra",
        "data_dir": "data/TESS/EK_Dra",
        "list_name": "detectors_EK_Dra",
        "class_name": "FlareDetector_EK_Dra",
    },
    {
        "name": "V889_Her",
        "path": Path("notebooks/flare_detect_V889_Her.ipynb"),
        "module_code": "from src.flarepy_V889_Her import FlareDetector_V889_Her",
        "data_dir": "data/TESS/V889_Her",
        "list_name": "detectors_V889_Her",
        "class_name": "FlareDetector_V889_Her",
    },
]


def build_loop_code(cfg):
    return f"""{cfg['list_name']} = []\nfor file_path in sorted((PROJECT_ROOT / '{cfg['data_dir']}').glob('*.fits')):\n    detector = {cfg['class_name']}(file=str(file_path), process_data=True)\n    {cfg['list_name']}.append(detector)\n\nfor detector in {cfg['list_name']}:\n    print(detector.file, detector.flare_number, detector.sum_flare_energy)\n"""


def remove_existing_markers(cells, cfg):
    markers = [
        "NOTEBOOK_DIR = Path().resolve()",
        cfg["module_code"],
        f"(PROJECT_ROOT / '{cfg['data_dir']}').glob",
        f"Path('{cfg['data_dir']}').glob",
    ]
    filtered = []
    for cell in cells:
        if cell.cell_type == 'code' and any(marker in cell.source for marker in markers):
            continue
        filtered.append(cell)
    return filtered


def insert_bootstrap(nb, cfg):
    cells = remove_existing_markers(nb.cells, cfg)
    head = []
    rest_start = 0
    for i, cell in enumerate(cells):
        if cell.cell_type == 'markdown' and cell.source.startswith('### Source:'):
            head.append(cell)
            rest_start = i + 1
        else:
            break
    rest = cells[rest_start:]
    new_cells = head + [
        nbformat.v4.new_code_cell(COMMON_SETUP),
        nbformat.v4.new_code_cell(cfg['module_code']),
        nbformat.v4.new_code_cell(build_loop_code(cfg)),
    ] + rest
    nb.cells = new_cells


def main():
    for cfg in STAR_CONFIG:
        path = cfg['path']
        if not path.exists():
            continue
        nb = nbformat.read(path, as_version=4)
        insert_bootstrap(nb, cfg)
        nbformat.write(nb, path)
        print('updated', path)


if __name__ == '__main__':
    main()
