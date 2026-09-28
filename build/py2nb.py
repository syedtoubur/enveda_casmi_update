"""Convert a cell-marked .py file into a .ipynb notebook.

Markers (must start at column 0, on their own line):
    # %%                 -> code cell
    # %% [markdown]      -> markdown cell (leading '# ' stripped from each line)

Usage:  python py2nb.py source.py out.ipynb
"""
import json
import sys


def split_cells(text):
    cells, cur, kind = [], [], "code"
    for line in text.split("\n"):
        if line.startswith("# %%"):
            if cur:
                cells.append((kind, "\n".join(cur).strip("\n")))
            cur = []
            kind = "markdown" if "[markdown]" in line else "code"
        else:
            cur.append(line)
    if cur:
        cells.append((kind, "\n".join(cur).strip("\n")))
    return [(k, s) for k, s in cells if s.strip()]


def to_nb(cells):
    out = []
    for kind, src in cells:
        if kind == "markdown":
            src = "\n".join(
                ln[2:] if ln.startswith("# ") else (ln[1:] if ln.startswith("#") else ln)
                for ln in src.split("\n")
            )
            out.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)})
        else:
            out.append({
                "cell_type": "code", "metadata": {}, "execution_count": None,
                "outputs": [], "source": src.splitlines(True),
            })
    return {
        "cells": out,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "accelerator": "GPU",
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, encoding="utf-8") as f:
        nb = to_nb(split_cells(f.read()))
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"{dst}: {len(nb['cells'])} cells")
