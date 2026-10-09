"""Generate a self-contained notebook from a driver: the casmi_scratch package is embedded as source strings.

    python scratch_pkg/tools/mk_notebook.py --all                           -> every notebook in NOTEBOOKS
    python scratch_pkg/tools/mk_notebook.py r0_kaggle                       -> scratch_pkg/notebooks/casmi_scratch_r0.ipynb
    python scratch_pkg/tools/mk_notebook.py r0_kaggle --name nosim --set use_sim=False
    python scratch_pkg/tools/mk_notebook.py r0_kaggle --stem r1a --set pool_source='"r1"'   -> casmi_scratch_r1a.ipynb
    python scratch_pkg/tools/mk_notebook.py r1_pool_colab                   -> casmi_scratch_r1_pool_colab.ipynb

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
# every committed notebook: (driver, name, overrides, stem). `--all` regenerates them; test_notebook checks them.
NOTEBOOKS = [
    ("r0_kaggle", None, {}, None),
    ("r0_kaggle", "nosim", {"use_sim": False}, None),
    ("r0_kaggle", None, {"pool_source": "r1"}, "r1a"),
    ("r0_kaggle", None, {"pool_source": "r1", "pool_drop_e180": True}, "r1b"),
    ("r0_kaggle", None, {"pool_source": "r1", "sim_model": "glacier"}, "r1a_glacier"),
    ("r0_kaggle", None, {"pool_source": "r1", "pool_drop_e180": True, "sim_model": "glacier"}, "r1b_glacier"),
    ("r1_pool_colab", None, {}, None),
    ("r1_audit_colab", None, {}, None),
    ("r0_kaggle", None, {"pool_source": "r1", "pool_drop_e180": True, "class_vote": True}, "r2_ice"),
    ("r0_kaggle", None, {"pool_source": "r1", "pool_drop_e180": True, "sim_model": "glacier", "class_vote": True},
     "r2_gl"),
    ("r2_npc_colab", None, {}, None),
    ("r2_audit_colab", None, {}, None),
    ("r0_kaggle", None, {"pool_source": "r1", "pool_drop_e180": True, "sim_model": "glacier", "fp_channel": "own"},
     "r3_own_gl"),
    ("r0_kaggle", None, {"pool_source": "r1", "pool_drop_e180": True, "sim_model": "glacier", "fp_channel": "ahmed"},
     "r3_full1_gl"),
    ("r0_kaggle", None, {"pool_source": "r1", "pool_drop_e180": True, "sim_model": "glacier", "class_vote": True,
                         "fp_channel": "own"}, "r3_own_r2gl"),
    ("r0_kaggle", None, {"pool_source": "r1", "pool_drop_e180": True, "sim_model": "glacier", "class_vote": True,
                         "fp_channel": "ahmed"}, "r3_full1_r2gl"),
    ("r3_audit_colab", None, {}, None),
    ("r0_kaggle", None, {"pool_source": "r1", "pool_drop_e180": True, "sim_model": "glacier", "class_vote": True,
                         "fp_channel": "ahmed", "fp_banks": ("ahmed", "prv"), "fp_weights": (1.0, 0.6),
                         "fp_stagec": True}, "r4_famfuse"),
    *[("r0_kaggle", None, {"pool_source": "r1", "pool_drop_e180": True, "sim_model": "glacier", "class_vote": vote,
                           "fp_channel": "ahmed", "analog_channel": an, "analog_exclude_train": excl},
       f"r5_{an}{'' if vote else '_novote'}{'' if excl else '_inclt'}")
      for excl in (True, False) for an in ("pssm", "tani") for vote in (True, False)],
    ("r5_audit_colab", None, {}, None),
]
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
    if tag and 'RUN_TAG = "r0"' in text:
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


def notebook_stem(driver, name=None, stem=None):
    return (stem or (driver[:-len("_kaggle")] if driver.endswith("_kaggle") else driver)) + (f"-{name}" if name else "")


def notebook_json(driver, name=None, overrides=None, stem=None):
    nb = to_nb(split_cells(render(driver, overrides, tag=notebook_stem(driver, name, stem))))
    return json.dumps(nb, ensure_ascii=False, indent=1)


def build(driver, name=None, overrides=None, stem=None):
    dst = PKG / "notebooks" / f"casmi_scratch_{notebook_stem(driver, name, stem)}.ipynb"
    dst.parent.mkdir(exist_ok=True)
    text = notebook_json(driver, name, overrides, stem)
    dst.write_text(text, encoding="utf-8")
    nb = json.loads(text)
    print(f"{dst.relative_to(ROOT)}: {len(nb['cells'])} cells")
    return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("driver", nargs="?")
    ap.add_argument("--all", action="store_true", help="regenerate every notebook in NOTEBOOKS")
    ap.add_argument("--name")
    ap.add_argument("--set", nargs="*")
    ap.add_argument("--stem", help="notebook stem (default: the driver name without _kaggle)")
    a = ap.parse_args()
    if a.all:
        for d, n, o, st in NOTEBOOKS:
            build(d, n, o, st)
    else:
        build(a.driver, a.name, parse_set(a.set), a.stem)
