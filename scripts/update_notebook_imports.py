#!/usr/bin/env python
"""
Script to update flare_notebook.ipynb imports to use the proper pattern.
"""
import json
from pathlib import Path


def main():
    notebook_path = Path(__file__).parent.parent / "notebooks" / "flare_notebook.ipynb"

    print(f"Loading notebook: {notebook_path}")

    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    # New import source with PROJECT_ROOT setup
    new_source = [
        'import sys\n',
        'from pathlib import Path\n',
        '\n',
        '# プロジェクトルートを設定\n',
        'PROJECT_ROOT = Path().resolve()\n',
        'if PROJECT_ROOT.name in ["notebooks", "src"]:\n',
        '    PROJECT_ROOT = PROJECT_ROOT.parent\n',
        '\n',
        'sys.path.insert(0, str(PROJECT_ROOT))\n',
        '\n',
        'import os\n',
        'from src.flarepy_DS_Tuc import FlareDetector\n',
        'import plotly.express as px\n',
        'import numpy as np\n',
        'import plotly.graph_objects as go\n',
        'from plotly.subplots import make_subplots\n',
        'from numpy.polynomial import Polynomial\n',
        'from src import flarepy_EK_Dra as EK_Dra\n',
        'from src import flarepy_DS_Tuc as DS_Tuc\n',
        'from src import flarepy_V889_Her as V889\n',
        'import matplotlib.pyplot as plt'
    ]

    # Update first cell
    nb['cells'][0]['source'] = new_source
    nb['cells'][0]['outputs'] = []  # Clear existing error outputs

    # Write back
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("Successfully updated notebook imports!")
    print("New imports:")
    for line in new_source:
        print(f"  {line.rstrip()}")


if __name__ == "__main__":
    main()
