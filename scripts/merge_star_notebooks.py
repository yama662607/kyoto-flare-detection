"""Merge notebooks from hiroto/daijiro/archive per star into notebooks/flare_detect_{star}.ipynb."""
import nbformat
from pathlib import Path

STAR_MAPPINGS = {
    "DS_Tuc_A": [
        Path("hiroto/flare_DS_Tuc_A.ipynb"),
        Path("daijiro/flare_detect_DS_Tuc_A.ipynb"),
        Path("notebooks/flare_detect_DS_Tuc_A.ipynb"),
    ],
    "EK_Dra": [
        Path("daijiro/flare_detect_EK_Dra.ipynb"),
        Path("notebooks/flare_detect_EK_Dra.ipynb"),
    ],
    "V889_Her": [
        Path("daijiro/flare_detect_V889 Her.ipynb"),
        Path("notebooks/flare_detect_V889_Her.ipynb"),
    ],
}

OUTPUT_DIR = Path("notebooks")
OUTPUT_DIR.mkdir(exist_ok=True)


def read_nb(path: Path):
    if not path.exists():
        return None
    return nbformat.read(path, as_version=4)


def merge_cells(nb_list):
    seen = set()
    cells = []
    for nb in nb_list:
        if nb is None:
            continue
        for cell in nb.cells:
            key = (cell.cell_type, cell.source.strip())
            if key in seen:
                continue
            seen.add(key)
            cells.append(cell)
    return cells


def merge_notebooks():
    for star, sources in STAR_MAPPINGS.items():
        notebooks = [read_nb(src) for src in sources]
        merged_cells = merge_cells(notebooks)
        metadata = next((nb.metadata for nb in notebooks if nb), None)
        nb_out = nbformat.v4.new_notebook()
        nb_out.metadata = metadata or {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python", "version": "3.13"},
        }
        nb_out.cells = merged_cells
        target = OUTPUT_DIR / f"flare_detect_{star}.ipynb"
        nbformat.write(nb_out, target)
        print("merged", star, "->", target)


if __name__ == "__main__":
    merge_notebooks()
