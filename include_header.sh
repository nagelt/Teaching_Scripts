python - <<'EOF'
import nbformat
from pathlib import Path

header = Path("header.html").read_text()

for nb_path in Path(".").rglob("*.ipynb"):
    nb = nbformat.read(nb_path, as_version=4)

    # Replace first cell
    nb.cells[0] = nbformat.v4.new_markdown_cell(header)

    nbformat.write(nb, nb_path)
    print(f"Updated: {nb_path}")
EOF
