#!/usr/bin/env python
"""
Extract full source code of each plotting cell for detailed analysis.
"""
import json
from pathlib import Path


def main():
    notebook_path = Path(__file__).parent.parent / "notebooks" / "flare_notebook.ipynb"
    output_path = Path(__file__).parent.parent / "outputs" / "legacy_plots_analysis.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    with open(output_path, 'w') as out:
        out.write(f"Legacy Notebook Plotting Analysis\n")
        out.write(f"=" * 80 + "\n\n")

        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                # Check for plotting code
                keywords = ['plt.', 'go.', '.plot', 'px.', 'fig', 'Figure', 'subplots', 'scatter', 'savefig']
                if any(keyword in source for keyword in keywords):
                    out.write(f"\n{'=' * 80}\n")
                    out.write(f"CELL {i}\n")
                    out.write(f"{'=' * 80}\n\n")
                    out.write(source)
                    out.write("\n\n")

    print(f"Analysis saved to: {output_path}")

    # Also print a summary
    print("\nQuick Summary:")
    print("-" * 40)
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            keywords = ['plt.', 'go.', '.plot', 'px.', 'fig', 'Figure', 'subplots', 'scatter', 'savefig']
            if any(keyword in source for keyword in keywords):
                # Find x_axis_attribute and y_axis_attribute values
                x_val = None
                y_val = None
                for line in source.split('\n'):
                    line = line.strip()
                    if line.startswith('x_axis_attribute ='):
                        x_val = line.split('=')[1].strip().strip("'\"")
                    if line.startswith('y_axis_attribute ='):
                        y_val = line.split('=')[1].strip().strip("'\"")

                if x_val and y_val:
                    print(f"Cell {i}: {x_val} vs {y_val}")
                else:
                    # Check for output filename
                    for line in source.split('\n'):
                        if 'output_filename' in line and '=' in line:
                            filename = line.split('=')[1].strip().strip("'\"")
                            print(f"Cell {i}: Output -> {filename}")
                            break
                    else:
                        print(f"Cell {i}: (need manual inspection)")


if __name__ == "__main__":
    main()
