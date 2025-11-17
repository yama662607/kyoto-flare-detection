from pathlib import Path

import nbformat

STAR_NOTEBOOKS = {
    "DS_Tuc_A": [
        Path("hiroto/flare_DS_Tuc_A.ipynb"),
        Path("daijiro/flare_detect_DS_Tuc_A.ipynb"),
    ],
    "EK_Dra": [Path("daijiro/flare_detect_EK_Dra.ipynb")],
    "V889_Her": [Path("daijiro/flare_detect_V889 Her.ipynb")],
}

OUTPUT = Path("notebooks/flare_detect_template.ipynb")

nb_template = nbformat.v4.new_notebook()
nb_template.metadata = {
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python", "version": "3.13"},
}
nb_template.cells.append(
    nbformat.v4.new_markdown_cell(
        "# flare_detect_template\nこのテンプレートは hiroto/daijiro の notebook を星毎に順次結合したものです。"
    )
)

for star, sources in STAR_NOTEBOOKS.items():
    nb_template.cells.append(nbformat.v4.new_markdown_cell(f"\n---\n## {star}\n"))
    for src_path in sources:
        if not src_path.exists():
            continue
        nb_template.cells.append(
            nbformat.v4.new_markdown_cell(f"### {src_path} からの抜粋")
        )
        nb = nbformat.read(src_path, as_version=4)
        nb_template.cells.extend(nb.cells)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb_template, OUTPUT)
print("template written to", OUTPUT)
