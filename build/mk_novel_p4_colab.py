"""build/novel_p4_colab.py -> casmi_novel_p4_colab.ipynb with the project sources embedded (no Drive copy of code).

    python build/mk_novel_p4_colab.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SUBS = {"NOVEL_DECODER": "novel_decoder.py", "SIMSIM_COMMON": "simsim_common.py", "NOVEL_COHORT": "novel_cohort.py",
        "NOVEL_P3_DATA": "novel_p3_data.py", "NOVEL_P3_EVAL": "novel_p3_eval.py", "NOVEL_P4": "novel_p4.py"}


def build(out=os.path.join(ROOT, "casmi_novel_p4_colab.ipynb")):
    text = open(os.path.join(HERE, "novel_p4_colab.py"), encoding="utf-8").read()
    for tag, fname in SUBS.items():
        line = f"{tag}_SRC = None  # @@{tag}@@"
        assert text.count(line) == 1, tag
        src = open(os.path.join(HERE, fname), encoding="utf-8").read()
        text = text.replace(line, f"{tag}_SRC = {src!r}")
    tmp = os.path.join(ROOT, "_novel_p4_colab_filled.py")
    open(tmp, "w", encoding="utf-8").write(text)
    try:
        subprocess.run([sys.executable, os.path.join(HERE, "py2nb.py"), tmp, out], check=True)
    finally:
        os.remove(tmp)
    return out


if __name__ == "__main__":
    print("wrote", build())
