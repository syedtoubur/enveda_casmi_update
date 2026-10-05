"""Generate a self-contained notebook from a driver: the casmi_scratch package is embedded as source strings.

    python scratch_pkg/tools/mk_notebook.py r0_kaggle                       -> scratch_pkg/notebooks/casmi_scratch_r0.ipynb
    python scratch_pkg/tools/mk_notebook.py r0_kaggle --name nosim --set use_sim=False

`--set k=v` writes CONFIG_OVERRIDES (Python literals), so a probe and its control differ in that one line.
"""
import argparse
import ast
import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT / "build"))
from py2nb import split_cells, to_nb  # noqa: E402

TAG = "PKG_FILES = None  # @@PKG_FILES@@"
OVR = "CONFIG_OVERRIDES = dict()"


def package_files():
    files = {}
    for p in sorted((PKG / "casmi_scratch").rglob("*.py")):
        rel = p.relative_to(PKG).as_posix()
        files[rel] = p.read_text(encoding="utf-8")
    return files


def render(driver, overrides=None, tag=None):
    text = (PKG / "drivers" / f"{driver}.py").read_text(encoding="utf-8")
    assert text.count(TAG) == 1, f"{driver}: expected one {TAG!r}"
    text = text.replace(TAG, "PKG_FILES = " + repr(package_files()))
    if overrides:
        assert text.count(OVR) == 1
        text = text.replace(OVR, "CONFIG_OVERRIDES = dict(" +
                            ", ".join(f"{k}={v!r}" for k, v in overrides.items()) + ")")
    if tag:
        assert text.count('RUN_TAG = "r0"') == 1
        text = text.replace('RUN_TAG = "r0"', f"RUN_TAG = {tag!r}")
    assert "@@" not in text
    return text


def parse_set(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = ast.literal_eval(v)
    return out


def build(driver, name=None, overrides=None):
    stem = driver.split("_")[0] + (f"-{name}" if name else "")
    dst = PKG / "notebooks" / f"casmi_scratch_{stem}.ipynb"
    dst.parent.mkdir(exist_ok=True)
    nb = to_nb(split_cells(render(driver, overrides, tag=stem)))
    dst.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{dst.relative_to(ROOT)}: {len(nb['cells'])} cells")
    return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("driver")
    ap.add_argument("--name")
    ap.add_argument("--set", nargs="*")
    a = ap.parse_args()
    build(a.driver, a.name, parse_set(a.set))
