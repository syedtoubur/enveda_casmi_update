"""Build the Phase-1 (condition grid) notebooks for specs_simple_ms_simulator.

    simple_sim_grid.py       -> casmi_simple_sim_grid_colab.ipynb  (Colab: steps 1-4 and 6)
    simple_sim_t4_kaggle.py  -> casmi_simple_sim_t4_kaggle.ipynb   (Kaggle T4 x2, internet off: step 5)

Substitutions (each tag must appear exactly once where it is used):
  * `# @@SIMSIM_COMMON@@`  -> build/simsim_common.py, verbatim
  * `# @@PHASE1_KERNELS@@` -> phase1 code cell 2 (numba _clean / entropy_sim), verbatim  [Colab only]
  * `WORKER_SRC = None  # @@ICEBERG_WORKER@@` -> build/iceberg_worker.py as a string literal

Usage:  python build/mk_simple_sim_grid.py
"""
import json
from pathlib import Path

from mk_simple_sim_gate import KERNEL_TAG, WORKER, WORKER_TAG, phase1_kernels
from py2nb import split_cells, to_nb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
COMMON = HERE / "simsim_common.py"
COMMON_TAG = "# @@SIMSIM_COMMON@@"

# (source, notebook, embeds the phase1 kernels)
BUILDS = [
    (HERE / "simple_sim_grid.py", ROOT / "casmi_simple_sim_grid_colab.ipynb", True),
    (HERE / "simple_sim_t4_kaggle.py", ROOT / "casmi_simple_sim_t4_kaggle.ipynb", False),
]


def render(src, needs_kernels):
    text = src.read_text(encoding="utf-8")
    subs = [(COMMON_TAG, COMMON.read_text(encoding="utf-8")),
            (WORKER_TAG, "WORKER_SRC = " + repr(WORKER.read_text(encoding="utf-8")))]
    if needs_kernels:
        subs.append((KERNEL_TAG, phase1_kernels()))
    else:
        assert KERNEL_TAG not in text, f"{src.name}: unexpected {KERNEL_TAG}"
    for tag, body in subs:
        assert text.count(tag) == 1, f"{src.name}: expected one {tag}"
        text = text.replace(tag, body)
    assert "@@" not in text, f"{src.name}: an unsubstituted placeholder survived"
    return text


def build(src, dst, needs_kernels):
    nb = to_nb(split_cells(render(src, needs_kernels)))
    dst.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{dst.name}: {len(nb['cells'])} cells")


if __name__ == "__main__":
    for src, dst, k in BUILDS:
        build(src, dst, k)
