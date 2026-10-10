"""Build the Track 4b Colab notebook.

    np_signals_colab.py         -> casmi_np_signals_colab.ipynb          (Colab CPU High-RAM: dataset build + benchmark)
    np_signals_realism_colab.py -> casmi_np_signals_realism_colab.ipynb  (Colab CPU High-RAM: cohort P + ablation, R.*)

The placeholder line `NP_CORE_SRC = None  # @@NP_CORE@@` becomes the source of build/np_signals_core.py, which the
notebook writes back to a file so the multiprocessing workers can import it.

Usage:  python build/mk_np_signals.py
"""
import json
from pathlib import Path

from py2nb import split_cells, to_nb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BUILDS = [(HERE / "np_signals_colab.py", ROOT / "casmi_np_signals_colab.ipynb"),
          (HERE / "np_signals_realism_colab.py", ROOT / "casmi_np_signals_realism_colab.ipynb")]
LINE = "NP_CORE_SRC = None  # @@NP_CORE@@"


def build(src=BUILDS[0][0], dst=BUILDS[0][1]):
    text = src.read_text(encoding="utf-8")
    assert text.count(LINE) == 1, f"expected one `{LINE}`"
    core = (HERE / "np_signals_core.py").read_text(encoding="utf-8")
    text = text.replace(LINE, f"NP_CORE_SRC = {core!r}")
    assert "@@" not in text
    nb = to_nb(split_cells(text))
    dst.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{dst.name}: {len(nb['cells'])} cells")
    return nb


if __name__ == "__main__":
    for s, d in BUILDS:
        build(s, d)
