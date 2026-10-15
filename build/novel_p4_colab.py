# %% [markdown]
# # c3-gen-union (K1): decoder per predicted formula + mutator edits, on one leak-free scale
#
# Campaign `20261015-c3-gen-slot2`, gate K1 (`research/harness/gate_report.md`, block 2026-10-15 + amendment).
#
# Per pseudo_c3 molecule this notebook reads `p4_inputs.pkl` (written locally by `build/novel_p4_inputs.py` from the
# pool-banned pseudo_c3 dump) and produces:
#
# | leg | what | score |
# |---|---|---|
# | `dec` | `search_masked_beam` per **predicted** formula (no oracle), beam `BEAM` | the beam's own masked log p |
# | `dec256` | the same at beam 256 on a `SUB256` subset, for the beam-size cost of K4 | idem |
# | `mut` | c3v2 analog edits from the banned Engine 2 analogs | `teacher_logp` (equal to the beam score on a leaf) |
#
# **No FPNet anywhere** (leaky on pseudo_c3). Raw lists go to Drive `casmi_2026/_novel/p4/raw/`; ranking with the
# metric's tautomer-canonical IK14 and the K1 verdict run locally (`build/novel_p4_score.py`). Resumable.

# %%
import os, sys, json, time, hashlib, pickle, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

from google.colab import drive
drive.mount("/content/drive")

ROOT = Path("/content/drive/MyDrive/casmi_2026")
P0 = ROOT / "_novel" / "p0"
P1 = ROOT / "_novel" / "p1"
P3 = ROOT / "_novel" / "p3"
EMB = ROOT / "_novel" / "emb_dbec3a0_lx"
OUT = ROOT / "_novel" / "p4"
RAW = OUT / "raw"
RDKIT_WHL = next(iter(sorted((ROOT / "casmi26-offline-rdkit-2026033").glob("rdkit-*.whl"))), None)
SRC = Path("/content/novel_p4/src")
for p in (RAW, SRC):
    p.mkdir(parents=True, exist_ok=True)

SELFIES_PIN, RDKIT_PIN = "2.1.2", "2026.03.3"
N_CORPUS_ROWS = 1_323_878
BEAM = 64                 # production-feasible beam (K4); dec256 below measures what beam 256 would add
SUB256 = 150              # molecules re-run at beam 256 (first SUB256 by molecule_id)
CHUNK = 25                # flush to Drive every CHUNK molecules
assert (OUT / "p4_inputs.pkl").exists(), "upload p4_inputs.pkl (build/novel_p4_inputs.py) to casmi_2026/_novel/p4/"
for p in (P3 / "decoder.pt", P0 / "selfies_vocab.json", P1 / "pseudo_c3.parquet", EMB):
    assert p.exists(), p


def sh(cmd, check=True):
    print("$", cmd[:300], flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if r.returncode and check:
        print(r.stderr[-3000:])
        raise RuntimeError(f"command failed ({r.returncode})")
    return r.stdout

# %% [markdown]
# ## Kernel packages and GPU (pins as in P3)

# %%
def kernel_version(mod):
    r = subprocess.run([sys.executable, "-c", f"import {mod}; print({mod}.__version__)"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def v_match(v, target):
    return v is not None and (v == target or v.replace(".0", ".") == target.replace(".0", "."))


if not v_match(kernel_version("rdkit"), RDKIT_PIN):
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if RDKIT_WHL is not None and py_tag in RDKIT_WHL.name:
        sh(f"{sys.executable} -m pip install -q --no-deps {RDKIT_WHL}", check=False)
    if not v_match(kernel_version("rdkit"), RDKIT_PIN):
        sh(f"{sys.executable} -m pip install -q rdkit==2026.3.3")
sh(f"{sys.executable} -m pip install -q selfies=={SELFIES_PIN}", check=False)

import torch
import rdkit, selfies as sf
assert v_match(rdkit.__version__, RDKIT_PIN), rdkit.__version__
assert torch.cuda.is_available(), "P4 needs a GPU runtime"
DEV = torch.device("cuda")
print(dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0), rdkit=rdkit.__version__, selfies=sf.__version__))

# %% [markdown]
# ## Project sources (embedded verbatim from `build/` by `build/mk_novel_p4_colab.py`)

# %%
NOVEL_DECODER_SRC = None  # @@NOVEL_DECODER@@
SIMSIM_COMMON_SRC = None  # @@SIMSIM_COMMON@@
NOVEL_COHORT_SRC = None  # @@NOVEL_COHORT@@
NOVEL_P3_DATA_SRC = None  # @@NOVEL_P3_DATA@@
NOVEL_P3_EVAL_SRC = None  # @@NOVEL_P3_EVAL@@
NOVEL_P4_SRC = None  # @@NOVEL_P4@@
for fname, text in [("novel_decoder.py", NOVEL_DECODER_SRC), ("simsim_common.py", SIMSIM_COMMON_SRC),
                    ("novel_cohort.py", NOVEL_COHORT_SRC), ("novel_p3_data.py", NOVEL_P3_DATA_SRC),
                    ("novel_p3_eval.py", NOVEL_P3_EVAL_SRC), ("novel_p4.py", NOVEL_P4_SRC)]:
    (SRC / fname).write_text(text, encoding="utf-8")
sys.path.insert(0, str(SRC))
import novel_decoder as nd
import novel_p3_data as p3d
import novel_p3_eval as p3e
import novel_p4 as p4
nd.check_constraints()

# %% [markdown]
# ## Decoder, embeddings, queries, inputs

# %%
CK = torch.load(P3 / "decoder.pt", map_location=DEV, weights_only=False)
VOCAB = nd.SelfiesVocab.from_json(P0 / "selfies_vocab.json")
assert CK["vocab_sha1"] == hashlib.sha1((P0 / "selfies_vocab.json").read_bytes()).hexdigest()
assert tuple(CK["adducts"]) == nd.TEST_ADDUCTS and tuple(CK["ce_bins"]) == nd.CE_BINS
TABLE = nd.TokenTable(VOCAB)
MODEL = nd.DreamsSelfiesDecoder(**CK["hparams"]).to(DEV)
MODEL.load_state_dict(CK["state_dict"])
MODEL.eval()
MAX_LEN = CK["max_len"]
STORE = p3d.EmbeddingStore(EMB, n_rows=N_CORPUS_ROWS)
Q_C3, skipped = p3e.build_queries(pd.read_parquet(P1 / "pseudo_c3.parquet"))
Q = {q["molecule_id"]: q for q in Q_C3}
INP = pickle.load(open(OUT / "p4_inputs.pkl", "rb"))["rows"]
MIDS = sorted(m for m in INP if m in Q)
print(f"decoder epoch {CK['epoch']} | queries {len(Q_C3)} (skipped {len(skipped)}) | inputs {len(INP)} | run {len(MIDS)}")
print("mean predicted formulas", np.mean([len(INP[m]["formulas"]) for m in MIDS]),
      "| mean mutants", np.mean([len(INP[m].get("mutants") or []) for m in MIDS]))


def memory(q, formula):
    return p3e.query_memory(MODEL, dict(q, formula=formula), STORE, DEV, torch)

# %% [markdown]
# ## Generation (resumable: one pickle per CHUNK molecules under `raw/`)

# %%
def done_ids():
    out = set()
    for f in RAW.glob("chunk_*.pkl"):
        out |= set(pickle.load(open(f, "rb")))
    return out


def run_one(m, beam256):
    q, inp = Q[m], INP[m]
    rec = dict(dec=[], dec256=[], mut=[], secs={})
    t0 = time.time()
    for f in inp["formulas"]:
        mem, pad = memory(q, f)
        rec["dec"] += [(s, lp, f) for s, lp in nd.search_masked_beam(MODEL, mem, f, TABLE, mem_pad=pad, beam=BEAM,
                                                                      max_len=MAX_LEN, alpha=0.0)]
    rec["secs"]["dec"] = time.time() - t0
    if beam256:
        t0 = time.time()
        for f in inp["formulas"]:
            mem, pad = memory(q, f)
            rec["dec256"] += [(s, lp, f) for s, lp in nd.search_masked_beam(MODEL, mem, f, TABLE, mem_pad=pad,
                                                                             beam=256, max_len=MAX_LEN, alpha=0.0)]
        rec["secs"]["dec256"] = time.time() - t0
    t0 = time.time()
    by_f = {}
    for smi, f, ai, dec in inp.get("mutants") or []:
        by_f.setdefault(f, []).append((smi, ai, dec))
    for f, items in by_f.items():
        try:
            mem, pad = memory(q, f)
        except Exception:                                  # a formula outside the decoder's elements
            continue
        for smi, ai, dec in items:
            sel = p4.to_selfies(smi)
            lp = None if sel is None else p4.teacher_logp(MODEL, mem, pad, TABLE, sel[0], f, torch)
            rec["mut"].append((smi, lp, f, ai, dec))
    rec["secs"]["mut"] = time.time() - t0
    return rec


todo = [m for m in MIDS if m not in done_ids()]
sub256 = set(MIDS[:SUB256])
print(f"{len(MIDS) - len(todo)} done, {len(todo)} to go")
buf, t_all = {}, time.time()
for i, m in enumerate(todo, 1):
    try:
        buf[m] = run_one(m, m in sub256)
    except Exception as e:                                 # one bad molecule must not lose the run
        print(m, "failed:", type(e).__name__, e, flush=True)
        buf[m] = dict(dec=[], dec256=[], mut=[], secs={}, error=repr(e))
    if len(buf) >= CHUNK or i == len(todo):
        pickle.dump(buf, open(RAW / f"chunk_{int(time.time())}_{i}.pkl", "wb"), protocol=4)
        buf = {}
        el = time.time() - t_all
        print(f"{i}/{len(todo)}  {el / i:.1f}s/mol  eta {(len(todo) - i) * el / i / 60:.0f} min", flush=True)

# %% [markdown]
# ## Merge for download

# %%
ALL = {}
for f in sorted(RAW.glob("chunk_*.pkl")):
    ALL.update(pickle.load(open(f, "rb")))
pickle.dump(dict(raw=ALL, beam=BEAM, sub256=sorted(sub256), decoder_epoch=CK["epoch"]),
            open(OUT / "p4_raw.pkl", "wb"), protocol=4)
sec = pd.DataFrame([r["secs"] for r in ALL.values() if r.get("secs")])
print(f"p4_raw.pkl: {len(ALL)} molecules -> download it to E:/casmi/c3_gen/ and run build/novel_p4_score.py")
print("seconds per molecule (A100):", sec.describe().loc[["mean", "50%", "max"]].round(2).to_dict())
