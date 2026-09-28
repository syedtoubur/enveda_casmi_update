"""Build the Phase-4 submission notebook for specs_simple_ms_simulator.

    simple_sim_submit.py -> casmi_simple_sim_submission.ipynb  (Kaggle, GPU T4, internet off)
                         -> casmi_simple_sim_nosim.ipynb       (the same notebook with CFG.USE_SIM = False:
                                                               the LB control that isolates ICEBERG)

Substitutions (each tag exactly once):
  * `# @@PHASE1_KERNELS@@`  -> phase1's numba kernel cell (_clean / entropy_sim / search), verbatim
  * `# @@PHASE1_LIBRARY@@`  -> phase1's adduct-physics + library cell (the one defining load_library), verbatim
  * `# @@PHASE1_POOL@@`     -> phase1's candidate-pool cell (the one defining build_candidate_pool), verbatim
  * `# @@PHASE1_EVIDENCE@@` -> phase1's evidence cell (lib_sim, MetFrag-lite; defines frag_scores), verbatim
  * `# @@SIMSIM_COMMON@@`   -> build/simsim_common.py, verbatim
  * `WORKER_SRC = None  # @@ICEBERG_WORKER@@` -> build/iceberg_worker.py as a string literal

Usage:  python build/mk_simple_sim.py
"""
import json
from pathlib import Path

from mk_simple_sim_gate import KERNEL_TAG, PHASE1, WORKER, WORKER_TAG, phase1_kernels
from mk_simple_sim_grid import COMMON, COMMON_TAG
from py2nb import split_cells, to_nb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = HERE / "simple_sim_submit.py"
DST = ROOT / "casmi_simple_sim_submission.ipynb"
DST_NOSIM = ROOT / "casmi_simple_sim_nosim.ipynb"
USE_SIM_LINE = "    USE_SIM = True "

# tag -> the `def` that identifies the phase1 code cell to embed
PHASE1_CELLS = {
    "# @@PHASE1_LIBRARY@@": "def load_library(",
    "# @@PHASE1_POOL@@": "def build_candidate_pool(",
    "# @@PHASE1_EVIDENCE@@": "def frag_scores(",
}


def phase1_cell(marker):
    code = ["".join(c["source"]) for c in json.loads(PHASE1.read_text(encoding="utf-8"))["cells"]
            if c["cell_type"] == "code"]
    hits = [s for s in code if marker in s]
    assert len(hits) == 1, f"expected exactly one phase1 cell with {marker!r}, found {len(hits)}"
    assert "# %%" not in hits[0], "a phase1 cell would be split by py2nb"
    return hits[0]


def render():
    text = SRC.read_text(encoding="utf-8")
    subs = [(KERNEL_TAG, phase1_kernels()),
            *((tag, phase1_cell(marker)) for tag, marker in PHASE1_CELLS.items()),
            (COMMON_TAG, COMMON.read_text(encoding="utf-8")),
            (WORKER_TAG, "WORKER_SRC = " + repr(WORKER.read_text(encoding="utf-8")))]
    for tag, body in subs:
        assert text.count(tag) == 1, f"{SRC.name}: expected one {tag}"
        text = text.replace(tag, body)
    assert "@@" not in text, f"{SRC.name}: an unsubstituted placeholder survived"
    return text


def render_nosim():
    text = render()
    assert text.count(USE_SIM_LINE) == 1, "expected exactly one CFG.USE_SIM = True line"
    return text.replace(USE_SIM_LINE, "    USE_SIM = False")


def build():
    for dst, text in ((DST, render()), (DST_NOSIM, render_nosim())):
        nb = to_nb(split_cells(text))
        dst.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{dst.name}: {len(nb['cells'])} cells")


if __name__ == "__main__":
    build()
