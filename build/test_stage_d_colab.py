"""Prove casmi_validate_ab_colab.ipynb differs from the Kaggle build ONLY where intended.

Stage D is the notebook that decides whether we ship. If the Colab port quietly drifted from
the Kaggle one, the A/B would be measuring a different pipeline than the leaderboard runs.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
load = lambda p: ["".join(c["source"]) for c in
                  json.loads((ROOT / p).read_text(encoding="utf-8"))["cells"]]

P = load("enveda-casmi-2026-phase1-enhanced-baseline.ipynb")
K = load("casmi_validate_ab.ipynb")
C = load("casmi_validate_ab_colab.ipynb")

assert len(K) == 19 and len(C) == 21, f"unexpected shapes: kaggle {len(K)}, colab {len(C)}"

# ---- phase1 cells: verbatim except the two that hard-code the Kaggle input root ------
for pi, ci in ((1, 3), (3, 5), (4, 6), (5, 7), (6, 8)):
    assert C[ci] == P[pi], f"colab cell {ci} is not phase1 cell {pi} verbatim"
for pi, ci in ((2, 4), (7, 9)):
    body = C[ci].split("\n", 1)[1]                      # drop the NOTE line
    assert body == P[pi].replace("/kaggle/input", "/content/casmi_stage"), \
        f"colab cell {ci} is not phase1 cell {pi} with only the path substituted"
print("  phase1 cells verbatim (2 and 7 path-substituted only)  [ok]")

# ---- Stage-D cells that must be byte-identical to the Kaggle build -------------------
IDENTICAL = {11: 13, 13: 15, 14: 16, 16: 18, 18: 20}    # kaggle index -> colab index
for ki, ci in IDENTICAL.items():
    assert C[ci] == K[ki], (
        f"colab cell {ci} has drifted from kaggle cell {ki} - Stage D must score the same "
        "pipeline in both environments")
print(f"  {len(IDENTICAL)} Stage-D cells byte-identical to the Kaggle build  [ok]")

# ---- the cells that are allowed to differ, and exactly how --------------------------
CFG_C, CH_C, P1_C, P3_C = C[11], C[14], C[17], C[19]

for frag in ("WORKERS = max(2, min(NCPU, 8))", "OUT = str(WORK)",
             "glob.glob(STAGE + '/**/fp_*.pt', recursive=True)",
             "glob.glob(STAGE + '/**/rank_train.npz', recursive=True)"):
    assert frag in CFG_C, f"config cell missing {frag!r}"
assert "MPool(SD.WORKERS)" in CH_C and "MPool(4)" not in CH_C, "MetFrag pool not resized"
for frag in ("P1CK", "[RESUME] pass 1", "pickle.dump(CACHE", "if mid in CACHE: continue"):
    assert frag in P1_C, f"pass 1 missing {frag!r}"
assert "lib_hits=lib_hits" not in P1_C, "lib_hits still cached; the checkpoint will be huge"
assert P1_C.count("lib_hits.get(") == 2, "lib_hits must still build lv for the window and tiers"
assert P3_C.count("to_csv(") == 1, "pass 3 should write the CSV after every config"
print("  the four ported cells carry exactly the intended changes  [ok]")

# ---- no Kaggle path survives anywhere except the two NOTE comments -------------------
CODE = {i for i, c in enumerate(json.loads(
    (ROOT / "casmi_validate_ab_colab.ipynb").read_text(encoding="utf-8"))["cells"])
    if c["cell_type"] == "code"}
leaks = [i for i, s in enumerate(C)
         if i in CODE and "/kaggle" in s and not s.lstrip().startswith("# NOTE:")]
assert not leaks, f"cells {leaks} still reference a Kaggle path"
print("  no Kaggle path outside the two NOTE comments  [ok]")

# ---- the pre-flight cell actually guards the two silent failures ---------------------
V = C[12]
assert "fp_bits_sha1" in V and "'G' in z.files" in V and "t1 < 0.85" in V, \
    "the pre-flight cell must check the fp_bits contract, the G column and the leak signature"
print("  pre-flight guards fp_bits sha1, the G column and v1's leak signature  [ok]")

for i in sorted(CODE):
    compile(C[i], f"colab_cell_{i}", "exec")

print("\nSTAGE-D COLAB OK: same pipeline, only the environment differs.")
