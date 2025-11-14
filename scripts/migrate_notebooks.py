import json
from pathlib import Path
import nbformat

STAR_CONFIG = [
    {
        "name": "DS_Tuc_A",
        "module": "flarepy_DS_Tuc_A",
        "class": "FlareDetector_DS_Tuc_A",
        "data_dir": "data/TESS/DS_Tuc_A",
    },
    {
        "name": "EK_Dra",
        "module": "flarepy_EK_Dra",
        "class": "FlareDetector_EK_Dra",
        "data_dir": "data/TESS/EK_Dra",
    },
    {
        "name": "V889_Her",
        "module": "flarepy_V889_Her",
        "class": "FlareDetector_V889_Her",
        "data_dir": "data/TESS/V889_Her",
    },
]

TEMPLATE_HEADER = """import os
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

CELL_DETECTOR_LOOP = """{class_name} = []
for file_path in sorted(Path('{data_dir}').glob('*.fits')):
    detector = {module}.{class_name}(file=file_path, process_data=True)
    {class_name}.append(detector)

for detector in {class_name}:
    print(detector.file, detector.flare_number, detector.sum_flare_energy)
"""

CELL_METADATA = {
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python", "version": "3.13"},
}


def build_notebook(config):
    nb = nbformat.v4.new_notebook()
    header = TEMPLATE_HEADER
    header += f"\n\nfrom src.{config['module']} import {config['class']}\n"
    nb.cells.append(nbformat.v4.new_code_cell(header))
    nb.cells.append(nbformat.v4.new_markdown_cell(f"# {config['name']} 軌跡データの処理"))
    nb.cells.append(nbformat.v4.new_code_cell(CELL_DETECTOR_LOOP.format(
        module=f"src.{config['module']}",
        class_name=config['class'],
        data_dir=config['data_dir'],
    )))
    nb.metadata.update(CELL_METADATA)
    return nb


def main():
    notebooks_dir = Path('notebooks')
    notebooks_dir.mkdir(exist_ok=True)
    for config in STAR_CONFIG:
        nb = build_notebook(config)
        target_path = notebooks_dir / f"flare_detect_{config['name']}.ipynb"
        nbformat.write(nb, target_path)
        print('wrote', target_path)


if __name__ == '__main__':
    main()
