import json

nb_path = r"E:/casmi/lines/antigravity-exploit/scratch/baseline/casmi26-v22-rel150-e2r2-clean.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

c34 = "".join(nb["cells"][34]["source"])
with open("scratch_cell34.txt", "w", encoding="utf-8") as f:
    f.write(c34)
print(f"Cell 34 length: {len(c34)} characters. Written to scratch_cell34.txt")
