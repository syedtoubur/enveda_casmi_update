"""Build the Colab version of Stage D (4-way A/B), derived from the Kaggle build.

The nine Stage-D cells are taken from the generated casmi_validate_ab.ipynb rather than
duplicated here, so the two notebooks cannot drift apart. Run mk_stage_d.py first.

Environment changes only:
  * paths resolve under Google Drive instead of /kaggle/input | /kaggle/working
  * the MetFrag pool is sized from os.cpu_count() instead of a hard-coded 4
  * pass 1 checkpoints to Drive and resumes, because Colab sessions drop
  * one extra pre-flight cell that proves the artifacts are the ones you think they are
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE1 = HERE.parent / "enveda-casmi-2026-phase1-enhanced-baseline.ipynb"
KAGGLE = HERE.parent / "casmi_validate_ab.ipynb"
OUT = HERE.parent / "casmi_validate_ab_colab.ipynb"

P = ["".join(c["source"]) for c in json.loads(PHASE1.read_text(encoding="utf-8"))["cells"]]
K = ["".join(c["source"]) for c in json.loads(KAGGLE.read_text(encoding="utf-8"))["cells"]]
assert "Stage-D config" in K[10] and "SHIP / HOLD" in K[18], \
    "casmi_validate_ab.ipynb is not the expected shape - run build/mk_stage_d.py first"

split = P[8].index("# Fit Calibrated Rankers")
RANK_FEATURES = P[8][:split].rstrip()
assert "def rank_features" in RANK_FEATURES and "rank_train.npz" not in RANK_FEATURES

DRIVE = "/content/casmi_stage"
assert P[2].count("/kaggle/input") == 3 and P[7].count("/kaggle/input") == 1
assert all("/kaggle" not in P[i] for i in (1, 3, 4, 5, 6)), "unexpected Kaggle path"
P2 = P[2].replace("/kaggle/input", DRIVE).replace("find_file('test.parquet')", "find_file('train.parquet')")
P7 = P[7].replace("/kaggle/input", DRIVE)


def sub(s, pairs, tag):
    for old, new in pairs:
        n = s.count(old)
        assert n == 1, f"{tag}: expected 1 occurrence of {old!r}, found {n}"
        s = s.replace(old, new)
    assert "/kaggle" not in s, f"{tag}: a Kaggle path survived"
    return s


# ---- Stage-D cells, with the environment substituted --------------------------------
D10 = sub(K[10], [
    ("    TOPN = 25", "    TOPN = 25\n    WORKERS = max(2, min(NCPU, 8))"),
    ("OUT = '/kaggle/working'", "OUT = str(WORK)"),
    ("glob.glob('/kaggle/input/**/fp_*.pt', recursive=True)",
     "glob.glob(STAGE + '/**/fp_*.pt', recursive=True)"),
    ("glob.glob('/kaggle/input/**/rank_train.npz', recursive=True)",
     "glob.glob(STAGE + '/**/rank_train.npz', recursive=True)"),
], "D10")

D12 = sub(K[12], [("MPool(4)", "MPool(SD.WORKERS)")], "D12")

# pass 1: resume from Drive. Colab drops sessions and this pass is the expensive one.
D15 = sub(K[15], [
    # lib_hits is read only inside the iteration that builds it (for `lv`, and for each
    # tier's `lv`), never afterwards -- and it is by far the largest object in the entry,
    # one float per distinct library key in the window. Dropping it keeps the resume
    # checkpoint small enough to write to Drive without stalling the loop.
    ("               lib_hits=lib_hits, tiers={})",
     "               tiers={})"),
    ("t0 = time.time()\nCACHE = {}\nfor gi, (mid, sub) in enumerate(mols):",
     "t0 = time.time()\n"
     "P1CK = os.path.join(SD.OUT, 'stage_d_pass1.pkl')\n"
     "CACHE = pickle.load(open(P1CK, 'rb')) if os.path.exists(P1CK) else {}\n"
     "if CACHE: print(f\"[RESUME] pass 1: {len(CACHE):,}/{len(mols):,} molecules cached\")\n"
     "_n0 = len(CACHE)\n"
     "for gi, (mid, sub) in enumerate(mols):\n"
     "    if mid in CACHE: continue"),
    ("    if (gi + 1) % 50 == 0 or gi == len(mols) - 1:\n"
     "        el = time.time() - t0\n"
     "        print(f\"  pass1 {gi+1}/{len(mols)} | {el:.0f}s | "
     "eta {(len(mols)-gi-1)*el/(gi+1)/60:.0f} min\", flush=True)",
     "    if (gi + 1) % 50 == 0 or gi == len(mols) - 1:\n"
     "        el = time.time() - t0\n"
     "        rate = max(len(CACHE) - _n0, 1) / max(el, 1e-9)\n"
     "        print(f\"  pass1 {len(CACHE)}/{len(mols)} | {el/60:.1f} min | \"\n"
     "              f\"eta {(len(mols)-len(CACHE))/rate/60:.0f} min\", flush=True)\n"
     "    if (gi + 1) % 200 == 0 or gi == len(mols) - 1:\n"
     "        pickle.dump(CACHE, open(P1CK, 'wb'), protocol=4)"),
], "D15")

# pass 3: write the per-molecule CSV after every config, not only at the end
D17 = sub(K[17], [
    ('    print(f"  {name:12s} MRR {d.rr.mean():.4f}  ({time.time()-t0:.0f}s)")',
     '    print(f"  {name:12s} MRR {d.rr.mean():.4f}  ({time.time()-t0:.0f}s)")\n'
     '    pd.concat([v.assign(config=k) for k, v in RES.items()]).to_csv(\n'
     '        os.path.join(SD.OUT, "stage_d_results.csv"), index=False)'),
], "D17")

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.splitlines(True)})


md("""# CASMI 2026 — Stage D on Colab: does the new stack actually win?

Scores the full phase1 ranking pipeline over the 1,145 `val_scripts` molecules in four
configurations:

| config | FPNet | rank_train.npz | what it isolates |
|---|---|---|---|
| `base`       | old | old | your current leaderboard anchor (0.337) |
| `model_only` | **new** | old | the coupling trap — **expect flat or negative** |
| `matrix_only`| old | **new** | did regenerating the matrix alone help? |
| `both`       | **new** | **new** | the shipping candidate |

## Which runtime

Stage D is **CPU-bound**, like Stage C. The GPU runs `model_logits` twice over 1,145
molecules — about four minutes of its life. Everything else is numba entropy search
(pass 1), RDKit fragmentation, and 16 `HistGradientBoostingClassifier` fits (pass 3), all
of which scale with **vCPU**.

| runtime | vCPU | RAM | units/hr | est. wall | est. cost |
|---|---|---|---|---|---|
| T4 + High-RAM | 4 | 51 GB | ~2.0 | ~4 h | ~8 units |
| **L4 + High-RAM** | **8** | **53 GB** | **~4.8** | **~2.5 h** | **~12 units** |
| A100 | 12 | 83 GB | ~13 | ~1.8 h | ~24 units |

**Use L4 with High-RAM.** T4 is marginally cheaper but costs you an extra 1.5 h of the day;
A100 doubles the bill to save 40 minutes on work the GPU barely touches. High-RAM is not
optional — `load_library` holds all 2.54M peak lists, and `build_candidate_pool` holds
711,705 fingerprints on top of it.

> Runtime → Change runtime type → **L4 GPU**, and Runtime shape → **High-RAM**.

## Before you start

The new `rank_train.npz` must sit **next to the checkpoints it was built from**. The setup
cell copies `_stagec/rank_train.npz` into the staged `casmi26-fpnet-v3/` folder for you and
stops with a clear message if Stage C never wrote it. Twelve of the 31 ranker features are
that FPNet's logits; pairing a matrix with the wrong model is exactly what cost 0.05.

Pass 1 checkpoints to Drive every 200 molecules and resumes automatically, so a dropped
session costs minutes rather than the whole run.

## The decision rule

Ship `both` only if **C2 and C2_hard each gain more than 1 paired SE** and **C1_npx does not
regress by more than 1 SE**. At n=400 and n=200, 1 SE on MRR is roughly 0.02 and 0.03.
Anything smaller is noise.

Expect `model_only` to land flat-to-negative. That arm is the control: it pairs the new FPNet
with the old matrix, which is precisely the mismatch Stage C exists to remove.

Read these as **relative** deltas. Every val truth comes from `train.parquet`, which is inside
the candidate pool, so truth-in-pool is ~1.0 by construction and val MRR always reads above the
leaderboard. The gap is the pool miss rate, and nothing here can measure it.""")

code('''#@title Colab setup — mount Drive, check the runtime, stage inputs locally
import os, shutil, subprocess, sys, time, glob as _glob
from pathlib import Path

DRIVE_ROOT = "/content/drive/MyDrive/casmi_2026"
STAGE      = "''' + DRIVE + '''"     # local mirror; Drive I/O is far too slow for a 2.5M-row read

from google.colab import drive
drive.mount("/content/drive")

# ---- runtime sanity ---------------------------------------------------------------
import multiprocessing, psutil, torch
NCPU = multiprocessing.cpu_count()
RAM  = psutil.virtual_memory().total / 1e9
print(f"vCPU      : {NCPU}")
print(f"RAM       : {RAM:.0f} GB")
print(f"GPU       : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
print(f"free disk : {shutil.disk_usage('/content').free/1e9:.0f} GB")

if RAM < 40:
    raise SystemExit(
        f"Only {RAM:.0f} GB RAM. load_library holds all 2.54M peak lists and the candidate "
        "pool holds 711,705 fingerprints on top of it.\\n"
        "Runtime -> Change runtime type -> Runtime shape: High-RAM, then re-run.")
if NCPU <= 2:
    print("\\n!! 2 vCPU. Stage D is CPU-bound; this will take many hours.")
    print("   Switch to an L4 + High-RAM runtime (8 vCPU) before continuing.")

# ---- stage the inputs onto local disk ----------------------------------------------
#  Both model generations are staged on purpose: this notebook selects between them by
#  explicit path, never by glob, so having both present is what makes the A/B possible.
NEED = {
    "enveda-CASMI26-molecule-id-mass-spectra": ["train.parquet"],
    "coconut_dataset":        ["fp_bits.npy", "coco_fp.npy", "coco_mass.npy", "coco_meta.pkl"],
    "val_scripts":            ["val.parquet", "val_key.pkl"],
    "casmi26-fpnet-v3":       None,     # NEW checkpoints
    "casmi26-fp-models-v2":   None,     # OLD checkpoints
    "casmi26-ranker-features": None,    # OLD rank_train.npz
    "casmi26-offline-rdkit-2026033": None,
}
os.makedirs(STAGE, exist_ok=True)
t0 = time.time()
for sub_, files in NEED.items():
    src = Path(DRIVE_ROOT) / sub_
    if not src.exists():
        raise SystemExit(f"missing on Drive: {src}")
    dst = Path(STAGE) / sub_; dst.mkdir(parents=True, exist_ok=True)
    for f in (files if files is not None else [p.name for p in src.iterdir() if p.is_file()]):
        s, d = src / f, dst / f
        if not s.exists():
            raise SystemExit(f"missing on Drive: {s}")
        if d.exists() and d.stat().st_size == s.stat().st_size:
            print(f"  cached  {sub_}/{f}  ({s.stat().st_size/1e6:.0f} MB)"); continue
        print(f"  copying {sub_}/{f}  ({s.stat().st_size/1e6:.0f} MB) ...", flush=True)
        shutil.copy2(s, d)

# ---- the NEW matrix must sit beside the NEW checkpoints ----------------------------
#  Stage C writes it to _stagec/. The artifact-selection cell below identifies the new
#  matrix by DIRECTORY CO-LOCATION with the new checkpoints, which is also how you must
#  upload it to Kaggle. Doing the same thing here keeps the two environments honest.
_newmat = Path(DRIVE_ROOT) / "_stagec" / "rank_train.npz"
if not _newmat.exists():
    raise SystemExit(
        f"{_newmat} does not exist.\\n"
        "Stage C either has not run or its gate refused to write. Run casmi_rank_matrix_colab\\n"
        "to completion first - there is nothing to A/B without it.")
shutil.copy2(_newmat, Path(STAGE) / "casmi26-fpnet-v3" / "rank_train.npz")
print(f"\\n  NEW matrix placed beside the NEW checkpoints "
      f"({_newmat.stat().st_size/1e6:.0f} MB)")
print(f"inputs staged in {time.time()-t0:.0f}s -> {STAGE}")

WORK = Path(DRIVE_ROOT) / "_staged"
WORK.mkdir(parents=True, exist_ok=True)
print(f"outputs  -> {WORK}")''')

md(f"""## phase1 cells 1–8 — verbatim

Cells **2** and **7** have one substitution: the literal `/kaggle/input` becomes `{DRIVE}`
(3 occurrences in cell 2, 1 in cell 7). Everything else is byte-identical to
`enveda-casmi-2026-phase1-enhanced-baseline.ipynb`.

Stage D only means anything if it scores the code that will actually run on the leaderboard.""")

code(P[1])
code("# NOTE: '/kaggle/input' -> '" + DRIVE + "' (3 occurrences). Otherwise verbatim.\n" + P2)
code(P[3])
code(P[4])
code(P[5])
code(P[6])
code("# NOTE: '/kaggle/input' -> '" + DRIVE + "' (1 occurrence). Otherwise verbatim.\n" + P7)
code(RANK_FEATURES)

code(D10)

code('''#@title VERIFY — are these the artifacts you think they are?
#  Two failures are silent and expensive: a checkpoint trained against a different fp_bits
#  layout (the 6,930-bit vector would be permuted, and f.z becomes noise), and shipping v1's
#  rank_train.npz by mistake. Both are cheap to rule out and impossible to spot afterwards.
import hashlib
print("=" * 74); print("STAGE-D PRE-FLIGHT"); print("=" * 74)

FP_SHA1 = hashlib.sha1(
    np.ascontiguousarray(np.load(find_file('fp_bits.npy'))).tobytes()).hexdigest()
print(f"fp_bits sha1   : {FP_SHA1}")
for p in NEW_PT:
    got = torch.load(p, map_location='cpu', weights_only=False).get('fp_bits_sha1')
    assert got == FP_SHA1, (
        f"{os.path.basename(p)} was trained against a DIFFERENT fp_bits.npy ({got}) - the "
        "6,930-bit layout must match exactly or f.z is meaningless")
print(f"   all {len(NEW_PT)} new checkpoints match the staged fp_bits.npy   [ok]")

print()
for tag, path in (('OLD', OLD_NPZ[0]), ('NEW', NEW_NPZ[0])):
    z = np.load(path)
    t1 = float(z['X'][z['Y'] == 1, 20].mean())
    print(f"{tag} matrix     : {z['X'].shape[0]:>7,} x {z['X'].shape[1]}  "
          f"G:{'G' in z.files!s:5s} truth-is-model-top1 {t1*100:5.1f}%   "
          f"<- {os.path.basename(os.path.dirname(path))}")
    assert z['X'].shape[1] == NFEAT, \\
        f"{tag} matrix has {z['X'].shape[1]} features, rank_features makes {NFEAT}"
    if tag == 'NEW':
        assert 'G' in z.files, \\
            "the NEW matrix has no G column - that is v1, not the Stage-C v2 output"
        assert t1 < 0.85, (
            f"the NEW matrix has the truth as the model's top-1 in {t1*100:.0f}% of windows. "
            "That is v1's leak signature (90.2%); a healthy v2 matrix reads about 72%.")
print("   (forked 42.0%  |  v1 LEAKED 90.2%  |  Stage-C v2 ~72%)")
print("   -> PASS")''')

code(K[11])
code(D12)
code(K[13])
code(K[14])
code(D15)
code(K[16])
code(D17)
code(K[18])

out = {"cells": cells,
       "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                   "name": "python3"},
                    "language_info": {"name": "python", "version": "3.12"}},
       "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{OUT.name}: {len(cells)} cells")
