import json

def replace_double_backslashes_in_ipynb(file_path):
    # Load the notebook JSON
    with open(file_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # Recursively replace double backslashes in all strings
    def replace_in_cell(cell):
        if isinstance(cell, str):
            return cell.replace('\\\\', '\\')
        elif isinstance(cell, list):
            return [replace_in_cell(item) for item in cell]
        elif isinstance(cell, dict):
            return {key: replace_in_cell(value) for key, value in cell.items()}
        else:
            return cell

    # Apply the replacement to the notebook content
    notebook = replace_in_cell(notebook)

    # Save the modified notebook back
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)
    
    print("All double backslashes have been replaced with single backslashes.")