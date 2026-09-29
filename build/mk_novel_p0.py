"""Build the specs_novel Phase-0 gate notebooks.

    novel_p0_colab.py   -> casmi_novel_p0_colab.ipynb    (Colab: P0.1a, P0.2-P0.5)
    novel_p0_kaggle.py  -> casmi_novel_p0_kaggle.ipynb   (Kaggle, internet off: P0.1b)
    novel_reembed_colab.py -> casmi_novel_reembed_colab.ipynb (Colab GPU: G0.1-R corpus re-embed)
    novel_lx_reembed_colab.py -> casmi_novel_lx_reembed_colab.ipynb (Colab GPU: G0.1-R2 fixed-code re-embed)
    novel_lx_cpu_colab.py  -> casmi_novel_lx_cpu_colab.ipynb   (Colab CPU: G0.1-R2 peak leg)
    novel_lx_kaggle.py     -> casmi_novel_lx_kaggle.ipynb      (Kaggle, internet off: G0.1-R2 leg)
    novel_p1_colab.py      -> casmi_novel_p1_colab.ipynb       (Colab CPU: P1 cohorts, G1)
    novel_p3_train_colab.py -> casmi_novel_p3_train_colab.ipynb (Colab GPU: P3 decoder training)
    novel_p3_eval_colab.py -> casmi_novel_p3_eval_colab.ipynb   (Colab GPU: P3 search bake-off)
    novel_p3_null_colab.py -> casmi_novel_p3_null_colab.ipynb   (Colab CPU: RandomDeNovo, MCES, G3)

Each placeholder line `NAME = None  # @@TAG@@` becomes `NAME = <repr of the source file>`: the notebooks
write the sources back to files, so the multiprocessing workers and the DreaMS venv can import them.

Usage:  python build/mk_novel_p0.py
"""
import json
from pathlib import Path

from py2nb import split_cells, to_nb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

SOURCES = {   # tag -> (variable, embedded file)
    "DREAMS_INFER": ("DREAMS_INFER_SRC", HERE / "dreams_infer.py"),
    "NOVEL_DECODER": ("NOVEL_DECODER_SRC", HERE / "novel_decoder.py"),
    "DREAMS_OFFICIAL_WORKER": ("OFFICIAL_WORKER_SRC", HERE / "dreams_official_worker.py"),
    "NOVEL_LX": ("NOVEL_LX_SRC", HERE / "novel_lx.py"),
    "NOVEL_COHORT": ("NOVEL_COHORT_SRC", HERE / "novel_cohort.py"),
    "SIMSIM_COMMON": ("SIMSIM_COMMON_SRC", HERE / "simsim_common.py"),
    "NOVEL_P3_DATA": ("NOVEL_P3_DATA_SRC", HERE / "novel_p3_data.py"),
    "NOVEL_P3_EVAL": ("NOVEL_P3_EVAL_SRC", HERE / "novel_p3_eval.py"),
}
BUILDS = [
    (HERE / "novel_p0_colab.py", ROOT / "casmi_novel_p0_colab.ipynb",
     ("DREAMS_INFER", "NOVEL_DECODER", "DREAMS_OFFICIAL_WORKER")),
    (HERE / "novel_p0_kaggle.py", ROOT / "casmi_novel_p0_kaggle.ipynb", ("DREAMS_INFER", "NOVEL_DECODER")),
    (HERE / "novel_reembed_colab.py", ROOT / "casmi_novel_reembed_colab.ipynb", ("DREAMS_INFER",)),
    (HERE / "novel_lx_reembed_colab.py", ROOT / "casmi_novel_lx_reembed_colab.ipynb", ("DREAMS_INFER", "NOVEL_LX")),
    (HERE / "novel_lx_cpu_colab.py", ROOT / "casmi_novel_lx_cpu_colab.ipynb", ("DREAMS_INFER", "NOVEL_LX")),
    (HERE / "novel_lx_kaggle.py", ROOT / "casmi_novel_lx_kaggle.ipynb", ("DREAMS_INFER", "NOVEL_LX")),
    (HERE / "novel_p1_colab.py", ROOT / "casmi_novel_p1_colab.ipynb", ("NOVEL_DECODER", "SIMSIM_COMMON", "NOVEL_COHORT")),
    (HERE / "novel_p3_train_colab.py", ROOT / "casmi_novel_p3_train_colab.ipynb",
     ("NOVEL_DECODER", "SIMSIM_COMMON", "NOVEL_COHORT", "NOVEL_P3_DATA")),
    (HERE / "novel_p3_eval_colab.py", ROOT / "casmi_novel_p3_eval_colab.ipynb",
     ("NOVEL_DECODER", "SIMSIM_COMMON", "NOVEL_COHORT", "NOVEL_P3_DATA", "NOVEL_P3_EVAL")),
    (HERE / "novel_p3_null_colab.py", ROOT / "casmi_novel_p3_null_colab.ipynb",
     ("NOVEL_DECODER", "SIMSIM_COMMON", "NOVEL_COHORT", "NOVEL_P3_DATA", "NOVEL_P3_EVAL")),
]


def placeholder(tag):
    var, _ = SOURCES[tag]
    return f"{var} = None  # @@{tag}@@"


def source_text(tag):
    # read_text normalises CRLF (git autocrlf on Windows), so the embedded text and its SHA1 are LF
    return SOURCES[tag][1].read_text(encoding="utf-8")


def build(src, dst, tags):
    text = src.read_text(encoding="utf-8")
    for tag in tags:
        line = placeholder(tag)
        assert text.count(line) == 1, f"{src.name}: expected one `{line}`"
        text = text.replace(line, f"{SOURCES[tag][0]} = {source_text(tag)!r}")
    assert "@@" not in text, f"{src.name}: an unsubstituted placeholder survived"
    nb = to_nb(split_cells(text))
    dst.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{dst.name}: {len(nb['cells'])} cells")


if __name__ == "__main__":
    for src, dst, tags in BUILDS:
        build(src, dst, tags)
