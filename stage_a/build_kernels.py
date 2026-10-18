"""Stage the Stage A Kaggle dataset + 5 kernel dirs under E:/casmi/stage_a_kaggle (nothing here touches Kaggle)."""
import json, os, shutil, sys
SA = os.path.dirname(os.path.abspath(__file__)); BUILD = os.path.join(SA, "..", "build")
W = "E:/casmi/stage_a_kaggle"; D = f"{W}/data"; CODE = f"{D}/code"
os.makedirs(CODE, exist_ok=True)
for f in ("casmi_post.py", "iso_rank.py", "iso_nl.py", "isomer_mut.py", "frag_rescore.py"):
    shutil.copy(f"{BUILD}/{f}", CODE)
for f in ("common.py", "a3.py", "a4.py", "kit_setup.py"):
    shutil.copy(f"{SA}/{f}", CODE)
pass  # val.parquet: mounted from syedtoubur/validation-data-casmi-2026
if os.path.exists(f"{SA}/../../../../ENVEDA_casmi/research/c3/20261018-iso-rank/stage_a_frozen.json"):
    pass
fz = "E:/ENVEDA_casmi/research/c3/20261018-iso-rank/stage_a_frozen.json"
if os.path.exists(fz):
    shutil.copy(fz, D)
if os.path.exists(f"{W}/pseudo_spectra.pkl"):
    shutil.copy(f"{W}/pseudo_spectra.pkl", D)
json.dump({"title": "casmi26-c3iso-stagea-in", "id": "syedtoubur/casmi26-c3iso-stagea-in", "licenses": [{"name": "CC0-1.0"}]},
          open(f"{D}/dataset-metadata.json", "w"))
KIT, INP = "syedtoubur/casmi26-harness-kit", "syedtoubur/casmi26-c3iso-stagea-in"
VAL = "syedtoubur/validation-data-casmi-2026"
A3 = ["syedtoubur/c3iso-stagea-a3-s0", "syedtoubur/c3iso-stagea-a3-s1", "syedtoubur/c3iso-stagea-a3-pseudo"]
specs = [("a3-s0", "a3_cohorts_kernel.py", dict(__SHARD__="0"), [KIT, INP, VAL], []),
         ("a3-s1", "a3_cohorts_kernel.py", dict(__SHARD__="1"), [KIT, INP, VAL], []),
         ("a3-pseudo", "a3_pseudo_kernel.py", {}, [KIT, INP, "syedtoubur/casmi26-c3iso-i1b-measure", "syedtoubur/casmi26-c3iso-i1b1-ice-out"], []),
         ("a4-freeze", "a4_kernel.py", dict(__MODE__="freeze"), [KIT, INP, VAL], A3),
         ("a4-score", "a4_kernel.py", dict(__MODE__="score"), [KIT, INP, VAL], A3)]
for name, src, subs, ds, ks in specs:
    kd = f"{W}/kernels/{name}"; os.makedirs(kd, exist_ok=True)
    s = open(f"{SA}/{src}", encoding="utf-8").read()
    for a, b in subs.items():
        s = s.replace(a, b)
    code = f"run_{name.replace('-', '_')}.py"
    open(f"{kd}/{code}", "w", encoding="utf-8").write(s)
    json.dump({"id": f"syedtoubur/c3iso-stagea-{name}", "title": f"c3iso-stagea-{name}", "code_file": code, "language": "python",
               "kernel_type": "script", "is_private": "true", "enable_gpu": "false", "enable_tpu": "false", "enable_internet": "true",
               "dataset_sources": ds, "competition_sources": [], "kernel_sources": ks}, open(f"{kd}/kernel-metadata.json", "w"), indent=1)
print("staged", W)
