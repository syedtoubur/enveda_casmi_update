import json

nb_path = r"E:/casmi/lines/antigravity-exploit/scratch/baseline/casmi26-v22-rel150-e2r2-clean.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    lines = src.splitlines()
    first_few = " | ".join(lines[:2])[:100]
    if any(k in src for k in ["REL_PROMOTE", "E2R2", "FRAG_GAP", "casmi_post", "e180", "dreams", "C12", "submission.to_csv", "submission.csv"]):
        print(f"Cell {i} ({c['cell_type']}): {first_few}")
