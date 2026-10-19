import json

nb_path = r"E:/casmi/lines/antigravity-exploit/scratch/baseline/casmi26-v22-rel150-e2r2-clean.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

c36 = "".join(nb["cells"][36]["source"])
with open("scratch_cell36.txt", "w", encoding="utf-8") as f:
    f.write(c36)
print(f"Cell 36 length: {len(c36)} characters. Written to scratch_cell36.txt")
