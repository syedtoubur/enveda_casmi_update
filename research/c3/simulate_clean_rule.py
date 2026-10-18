import json

with open(r"E:\ENVEDA_casmi\research\c3\20261018-iso-rank\stage_a_result.json", "r") as f:
    d = json.load(f)

print("Top keys:", list(d.keys()))
if "c12" in d:
    print("c12 keys:", list(d["c12"].keys()))
