import py3Dmol
import webbrowser
import os
from pathlib import Path

def visualize_pdb(pdb_file, style='cartoon', color='spectrum'):
    with open(pdb_file, 'r') as f:
        pdb_data = f.read()
    view = py3Dmol.view(width=800, height=600)
    view.addModel(pdb_data, 'pdb')
    if style == 'cartoon':
        view.setStyle({'cartoon': {'color': color}})
    elif style == 'stick':
        view.setStyle({'stick': {'colorscheme': color}})
    elif style == 'sphere':
        view.setStyle({'sphere': {'colorscheme': color}})
    view.zoomTo()
    return view

script_dir = os.path.dirname(os.path.abspath(__file__))
pdb_file = os.path.join(script_dir, "case_a_baseline.pdb")
view = visualize_pdb(pdb_file, style='cartoon', color='spectrum')


# Generate the full HTML document
html_content = view._make_html()

# Wrap it into a complete HTML page
full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Protein 3D Viewer</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Protein 3D Structure Viewer</h1>
        {html_content}
    </div>
</body>
</html>"""

# Save the file
html_file = 'protein_view.html'
with open(html_file, 'w', encoding='utf-8') as f:
    f.write(full_html)

# Open it automatically
file_path = os.path.abspath(html_file)
webbrowser.open(Path(file_path).resolve().as_uri())
print(f"Generated full HTML file: {file_path}")
print("If the browser still shows an error, check the network connection because 3Dmol.js is loaded from a CDN.")