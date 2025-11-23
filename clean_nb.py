import nbformat as nbf

NOTEBOOK_PATH = "Notebook/FT_Transformers_.ipynb"  # adjust if needed

nb = nbf.read(NOTEBOOK_PATH, as_version=4)

if "widgets" in nb.get("metadata", {}):
    print("Removing metadata.widgets …")
    nb["metadata"].pop("widgets")
else:
    print("No metadata.widgets found.")

nbf.write(nb, NOTEBOOK_PATH)
print("Notebook cleaned and saved.")