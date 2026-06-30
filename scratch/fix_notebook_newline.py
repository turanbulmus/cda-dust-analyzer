import json

notebook_path = "/workspaces/cda-dust-analyzer/Notebooks/07_3p_ablation_study.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

modified = False
for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        for i, line in enumerate(source):
            # Find the line that writes request to file
            if "f.write(req +" in line and "\\\\n" in line:
                print(f"Original line: {repr(line)}")
                # Replace the double-escaped literal \\n (which was '\\\\n' in JSON) with single-escaped \n
                new_line = line.replace("\\\\n", "\\n")
                print(f"Modified line: {repr(new_line)}")
                source[i] = new_line
                modified = True

if modified:
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated successfully!")
else:
    print("No matching lines found to update in the notebook.")
