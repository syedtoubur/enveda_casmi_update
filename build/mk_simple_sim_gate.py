"""Build the Phase-0 (ICEBERG gate) notebooks for specs_simple_ms_simulator.

    simple_sim_gate.py         -> casmi_simple_sim_gate_colab.ipynb   (Colab: G1-G4, G6, G5 Colab half)
    simple_sim_gate_kaggle.py  -> casmi_simple_sim_gate_kaggle.ipynb  (Kaggle, internet off: G5)

Two substitutions only:
  * `# @@PHASE1_KERNELS@@`  -> phase1 code cell 2 (the numba _clean / entropy_sim kernels), verbatim
  * `WORKER_SRC = None  # @@ICEBERG_WORKER@@` -> build/iceberg_worker.py as a string literal

Usage:  python build/mk_simple_sim_gate.py
"""
import json
from pathlib import Path

from py2nb import split_cells, to_nb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PHASE1 = ROOT / "enveda-casmi-2026-phase1-enhanced-baseline.ipynb"
WORKER = HERE / "iceberg_worker.py"

KERNEL_TAG = "# @@PHASE1_KERNELS@@"
WORKER_TAG = "WORKER_SRC = None  # @@ICEBERG_WORKER@@"
BUILDS = [
    (HERE / "simple_sim_gate.py", ROOT / "casmi_simple_sim_gate_colab.ipynb", True),
    (HERE / "simple_sim_gate_kaggle.py", ROOT / "casmi_simple_sim_gate_kaggle.ipynb", False),
]


def phase1_kernels():
    code = ["".join(c["source"]) for c in json.loads(PHASE1.read_text(encoding="utf-8"))["cells"]
            if c["cell_type"] == "code"]
    hits = [s for s in code if "def entropy_sim(" in s and "def _clean(" in s]
    assert len(hits) == 1, f"expected exactly one phase1 kernel cell, found {len(hits)}"
    assert "/kaggle" not in hits[0] and "CFG." not in hits[0], "kernel cell is not self-contained"
    return hits[0]


def build(src, dst, needs_worker):
    text = src.read_text(encoding="utf-8")
    assert text.count(KERNEL_TAG) == 1, f"{src.name}: expected one {KERNEL_TAG}"
    text = text.replace(KERNEL_TAG, phase1_kernels())
    if needs_worker:
        assert text.count(WORKER_TAG) == 1, f"{src.name}: expected one {WORKER_TAG}"
        text = text.replace(WORKER_TAG, "WORKER_SRC = " + repr(WORKER.read_text(encoding="utf-8")))
    assert "@@" not in text, f"{src.name}: an unsubstituted placeholder survived"
    nb = to_nb(split_cells(text))
    dst.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{dst.name}: {len(nb['cells'])} cells")


if __name__ == "__main__":
    for src, dst, w in BUILDS:
        build(src, dst, w)
