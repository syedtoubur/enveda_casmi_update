"""Create research/<line>/<YYYYMMDD-slug>/ from research/_template.

    python research/bin/new_idea.py exploit pop-tiebreak
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil

from _common import RESEARCH, load_config, write_status


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("line")
    ap.add_argument("slug")
    a = ap.parse_args(argv)
    if a.line not in load_config()["lines"]:
        raise SystemExit(f"unknown line {a.line!r}")
    slug = re.sub(r"[^a-z0-9-]+", "-", a.slug.lower()).strip("-")
    dest = RESEARCH / a.line / f"{dt.date.today():%Y%m%d}-{slug}"
    if dest.exists():
        raise SystemExit(f"exists: {dest}")
    shutil.copytree(RESEARCH / "_template", dest, ignore=shutil.ignore_patterns(".gitkeep"))
    (dest / "spike").mkdir(exist_ok=True)
    (dest / "tickets").mkdir(exist_ok=True)
    write_status(dest, "drafting", line=a.line)
    print(dest.relative_to(RESEARCH.parent).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
