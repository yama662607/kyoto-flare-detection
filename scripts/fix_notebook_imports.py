
import json
from pathlib import Path

def main():
    notebook_path = Path("notebooks/flare_notebook.ipynb")
    print(f"Loading notebook: {notebook_path}")

    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    # Correct Import Source with Legacy Aliases and Monkey Patching
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
        'import matplotlib.pyplot as plt\n',
        'import plotly.express as px\n',
        'import numpy as np\n',
        'import plotly.graph_objects as go\n',
        'from plotly.subplots import make_subplots\n',
        'from numpy.polynomial import Polynomial\n',
        '\n',
        '# Import modules using legacy aliases expected by later cells\n',
        'from src import flarepy_EK_Dra as ek_module\n',
        'from src import flarepy_DS_Tuc_A as ds_module\n',
        'from src import flarepy_V889_Her as v889_module\n',
        '\n',
        '# Monkey patch modules to expose FlareDetector class as expected\n',
        'ek_module.FlareDetector = ek_module.FlareDetector_EK_Dra\n',
        'ds_module.FlareDetector = ds_module.FlareDetector_DS_Tuc_A\n',
        'v889_module.FlareDetector = v889_module.FlareDetector_V889_Her\n',
        '\n',
        '# Also provide direct aliases if used elsewhere\n',
        'EK_Dra = ek_module\n',
        'DS_Tuc = ds_module\n',
        'V889 = v889_module\n',
        '\n',
        '# Legacy import of single class\n',
        'from src.flarepy_DS_Tuc_A import FlareDetector_DS_Tuc_A as FlareDetector\n',
        '\n',
        '# Migrated visualization module\n',
        'try:\n',
        '    from src.visualization import (\n',
        '        plot_flare_frequency,\n',
        '        plot_total_energy,\n',
        '        plot_max_energy,\n',
        '        plot_cumulative_energy,\n',
        '        STAR_COLORS\n',
        '    )\n',
        '    print("Successfully imported src.visualization module")\n',
        'except ImportError as e:\n',
        '    print(f"Warning: Could not import src.visualization: {e}")\n'
    ]

    # Update first cell
    nb['cells'][0]['source'] = new_source
    nb['cells'][0]['outputs'] = []  # Clear previous errors

    # Write back
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("Successfully fixed notebook imports with legacy compatibility!")

if __name__ == "__main__":
    main()
