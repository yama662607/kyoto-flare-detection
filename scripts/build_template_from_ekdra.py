import nbformat
from pathlib import Path

TEMPLATE_MARKDOWN = {
    "intro": "# Flare Detector Template (EK Dra)\nこのノートブックでは `src.flarepy_EK_Dra.FlareDetector_EK_Dra` の基本的な使い方を示します。",
    "setup": "## 1. Setup\nPlotly Express と Graph Objects の両方を使用できるように準備します。",
    "data": "## 2. FITS ファイルの確認",
    "detector": "## 3. FlareDetector_EK_Dra で処理",
    "summary": "## 4. サマリ",
    "px_line": "## 5. Plotly Express: Light Curve",
    "px_energy": "## 6. Plotly Express: Energy vs Time",
    "px_hist": "## 7. Plotly Express: Flare Energy Histogram",
    "go_line": "## 8. Plotly Graph Objects: Light Curve",
    "loop": "## 9. 複数ファイルの処理例",
    "analysis": "## 10. 追加解析",
}

CODE_SNIPPETS = {
    "setup": """import os\nimport sys\nfrom pathlib import Path\n\nimport numpy as np\nimport pandas as pd\nimport plotly.express as px\nimport plotly.graph_objects as go\nfrom plotly.subplots import make_subplots\n\nNOTEBOOK_DIR = Path().resolve()\nPROJECT_ROOT = NOTEBOOK_DIR.parent\nif str(PROJECT_ROOT) not in sys.path:\n    sys.path.insert(0, str(PROJECT_ROOT))\nprint('PROJECT_ROOT:', PROJECT_ROOT)\n""",
    "data": """DATA_DIR = PROJECT_ROOT / 'data' / 'TESS' / 'EK_Dra'\nfits_files = sorted(DATA_DIR.glob('*.fits'))\nprint('found', len(fits_files), 'files')\nfits_files[:3]\n""",
    "detector": """from src.flarepy_EK_Dra import FlareDetector_EK_Dra\n\nexample_file = fits_files[0] if fits_files else None\ndetector = None\nif example_file is not None:\n    detector = FlareDetector_EK_Dra(file=str(example_file), process_data=True)\ndetector\n""",
    "summary": """summary = {\n    'file': detector.file,\n    'flare_number': detector.flare_number,\n    'sum_flare_energy': detector.sum_flare_energy,\n    'precise_obs_time': detector.precise_obs_time,\n    'flare_ratio': getattr(detector, 'flare_ratio', None),\n}\nsummary\n""",
    "px_line": """px.line(\n    x=detector.tessBJD,\n    y=detector.mPDCSAPflux,\n    labels={'x': 'BJD - 2457000', 'y': 'Normalized Flux'},\n    title='EK Dra Light Curve (Plotly Express)'\n)\n""",
    "px_energy": """px.scatter(\n    x=detector.tessBJD,\n    y=detector.energy,\n    labels={'x': 'BJD - 2457000', 'y': 'Energy'},\n    title='Flare Energy vs Time (Plotly Express)'\n)\n""",
    "px_hist": """px.histogram(\n    x=detector.energy,\n    nbins=50,\n    labels={'x': 'Energy', 'y': 'Count'},\n    title='Flare Energy Distribution (Plotly Express)'\n)\n""",
    "go_line": """fig = go.Figure()\nfig.add_trace(go.Scatter(\n    x=detector.tessBJD,\n    y=detector.mPDCSAPflux,\n    mode='lines',\n    name='Normalized Flux'\n))\nfig.update_layout(\n    title='EK Dra Light Curve (Graph Objects)',\n    xaxis_title='BJD - 2457000',\n    yaxis_title='Normalized Flux'\n)\nfig.show()\n""",
    "loop": """detectors = []\nfor file_path in fits_files[:3]:\n    det = FlareDetector_EK_Dra(file=str(file_path), process_data=True)\n    detectors.append(det)\nprint('processed', len(detectors), 'files')\n""",
    "analysis": """print('Rotation period:', detector.per)\nprint('First 5 flare energies:', detector.energy[:5])\npx.box(\n    y=[d.sum_flare_energy for d in detectors],\n    labels={'y': 'Sum Flare Energy'},\n    title='Sum Flare Energy per File'\n)\n""",
}

def build_template():
    nb = nbformat.v4.new_notebook()
    nb.metadata = {
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
        'language_info': {'name': 'python', 'version': '3.13'},
    }
    nb.cells = [
        nbformat.v4.new_markdown_cell(TEMPLATE_MARKDOWN['intro']),
        nbformat.v4.new_markdown_cell(TEMPLATE_MARKDOWN['setup']),
        nbformat.v4.new_code_cell(CODE_SNIPPETS['setup']),
        nbformat.v4.new_markdown_cell(TEMPLATE_MARKDOWN['data']),
        nbformat.v4.new_code_cell(CODE_SNIPPETS['data']),
        nbformat.v4.new_markdown_cell(TEMPLATE_MARKDOWN['detector']),
        nbformat.v4.new_code_cell(CODE_SNIPPETS['detector']),
        nbformat.v4.new_markdown_cell(TEMPLATE_MARKDOWN['summary']),
        nbformat.v4.new_code_cell(CODE_SNIPPETS['summary']),
        nbformat.v4.new_markdown_cell(TEMPLATE_MARKDOWN['px_line']),
        nbformat.v4.new_code_cell(CODE_SNIPPETS['px_line']),
        nbformat.v4.new_markdown_cell(TEMPLATE_MARKDOWN['px_energy']),
        nbformat.v4.new_code_cell(CODE_SNIPPETS['px_energy']),
        nbformat.v4.new_markdown_cell(TEMPLATE_MARKDOWN['px_hist']),
        nbformat.v4.new_code_cell(CODE_SNIPPETS['px_hist']),
        nbformat.v4.new_markdown_cell(TEMPLATE_MARKDOWN['go_line']),
        nbformat.v4.new_code_cell(CODE_SNIPPETS['go_line']),
        nbformat.v4.new_markdown_cell(TEMPLATE_MARKDOWN['loop']),
        nbformat.v4.new_code_cell(CODE_SNIPPETS['loop']),
        nbformat.v4.new_markdown_cell(TEMPLATE_MARKDOWN['analysis']),
        nbformat.v4.new_code_cell(CODE_SNIPPETS['analysis']),
    ]
    output = Path('notebooks/flare_detect_template.ipynb')
    nbformat.write(nb, output)
    print('Template updated ->', output)

if __name__ == '__main__':
    build_template()
