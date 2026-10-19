import json

nb_path = r"E:/casmi/lines/antigravity-exploit/scratch/baseline/casmi26-v22-rel150-e2r2-clean.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx in [24, 26, 30, 33, 34, 35, 36]:
    c = nb["cells"][idx]
    src = "".join(c.get("source", []))
    print(f"=== Cell {idx} ({c['cell_type']}) ===")
    print(src[:600])
    print("...\n")
