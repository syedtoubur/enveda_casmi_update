import json

nb_path = r"E:/casmi/lines/antigravity-exploit/scratch/baseline/casmi26-v22-rel150-e2r2-clean.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    if "_factorial_entry" in src:
        print(f"Cell {idx} contains _factorial_entry")
        for line in src.splitlines():
            if "_factorial_entry" in line:
                print("  ", line)
