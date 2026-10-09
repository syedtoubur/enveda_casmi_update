"""Turn a Kaggle notebook into a Colab script that runs from Drive with the /kaggle/input layout.

    python research/bin/nb_to_colab.py research/exploit/<idea> probe.ipynb --kind smoke
    python research/bin/nb_to_colab.py research/exploit/<idea> probe.ipynb --kind val [--val-parquet val_scripts/val.parquet]

Writes <idea>/colab/<kind>.py, which you then run with colab_run.py.
- The shim links every Drive folder under casmi_2026/ to /kaggle/input/<name>,
  /kaggle/input/datasets/<owner>/<name> (owner from the catalog) and /kaggle/input/competitions/<competition>.
- smoke: KAGGLE_IS_COMPETITION_RERUN is left unset, so the notebook takes its own fast-commit path (10-20 molecules, L24).
- val: KAGGLE_IS_COMPETITION_RERUN=1 (full path), with the competition test.parquet replaced by the val parquet.
  Read the result only relative to the base on the same cohort (L07).
Notebook cells are concatenated in order; IPython magics (`!pip`, `%...`) are kept, because colab exec runs in an IPython kernel.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import REPO, idea_dir, load_config

SHIM = '''# ---- research/bin/nb_to_colab.py shim: Drive -> /kaggle/input ----
import os, json, shutil, time
_DRIVE = {drive_root!r}
_COMP = {competition!r}
_OWNERS = {owners!r}
_KIND = {kind!r}
_VAL = {val!r}
for _ in range(24):  # colab_run mounts Drive; the mount can lag the CLI by a minute
    if os.path.isdir(_DRIVE):
        break
    time.sleep(5)
assert os.path.isdir(_DRIVE), f'Drive not mounted: {{_DRIVE}} missing (colab_run.py mounts it)'
for _d in ('/kaggle/input/datasets', '/kaggle/input/competitions', '/kaggle/working', '/kaggle/tmp'):
    os.makedirs(_d, exist_ok=True)
for _name in sorted(os.listdir(_DRIVE)):
    _src = os.path.join(_DRIVE, _name)
    if not os.path.isdir(_src) or _name == _COMP:
        continue
    for _dst in [f'/kaggle/input/{{_name}}'] + ([f'/kaggle/input/datasets/{{_OWNERS[_name]}}/{{_name}}'] if _name in _OWNERS else []):
        os.makedirs(os.path.dirname(_dst), exist_ok=True)
        if not os.path.lexists(_dst):
            os.symlink(_src, _dst)
_comp_dst = f'/kaggle/input/competitions/{{_COMP}}'
os.makedirs(_comp_dst, exist_ok=True)
for _f in os.listdir(os.path.join(_DRIVE, _COMP)):
    _t = os.path.join(_comp_dst, _f)
    if not os.path.lexists(_t):
        os.symlink(os.path.join(_DRIVE, _COMP, _f), _t)
if not os.path.lexists(f'/kaggle/input/{{_COMP}}'):
    os.symlink(_comp_dst, f'/kaggle/input/{{_COMP}}')
if _KIND == 'val':
    os.remove(os.path.join(_comp_dst, 'test.parquet'))
    os.symlink(os.path.join(_DRIVE, _VAL), os.path.join(_comp_dst, 'test.parquet'))
    os.environ['KAGGLE_IS_COMPETITION_RERUN'] = '1'
else:
    os.environ.pop('KAGGLE_IS_COMPETITION_RERUN', None)
os.chdir('/kaggle/working')
print(f'[shim] kind={{_KIND}} inputs linked:', sorted(os.listdir('/kaggle/input')))
# ---- end shim ----
'''


def owners_from_catalog(cfg) -> dict[str, str]:
    cat = json.loads((REPO / cfg["catalog"]).read_text(encoding="utf-8"))
    out = {}
    for group in cat.get("datasets", {}).values():
        for entry in group.values():
            slug = entry.get("slug", "")
            if "/" in slug:
                owner, name = slug.split("/", 1)
                out[name] = owner
    return out


def notebook_code(nb_path: Path) -> str:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    parts = []
    for i, c in enumerate(cells):
        src = c.get("source", "")
        src = "".join(src) if isinstance(src, list) else src
        parts.append(f"# ---- cell {i} ----\n{src}\n")
    return "\n".join(parts)


def build(cfg, nb_path: Path, kind: str, val: str) -> str:
    shim = SHIM.format(drive_root=cfg["colab"]["drive_root"], competition=cfg["competition"],
                       owners=owners_from_catalog(cfg), kind=kind, val=val)
    return shim + "\n" + notebook_code(nb_path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("idea")
    ap.add_argument("notebook")
    ap.add_argument("--kind", choices=["smoke", "val"], required=True)
    ap.add_argument("--val-parquet", default="val_scripts/val.parquet", help="relative to the Drive root")
    a = ap.parse_args(argv)
    cfg = load_config()
    idea = idea_dir(a.idea)
    out = idea / "colab" / f"{a.kind}.py"
    out.parent.mkdir(exist_ok=True)
    out.write_text(build(cfg, Path(a.notebook), a.kind, a.val_parquet), encoding="utf-8")
    print(out.relative_to(REPO).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
