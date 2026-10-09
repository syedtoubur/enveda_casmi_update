"""Match Kaggle submissions to idea folders and record scores.

    python research/bin/poll_lb.py [--write]

Reads `kaggle competitions submissions -c <competition> --csv`. A submission belongs to an idea
when its description contains the idea's kernel slug (STATUS `kernel:`): the human pastes the
slug from the submit card into Kaggle's submission description. Unmatched submissions among the
15 most recent are listed so the Reporter can ask.
Prints a table; with --write it sets STATUS `lb:` and moves `ready-to-submit`/`submitted` ideas
to `scored`. KEEP/REVERT and the EXPERIMENTS/LESSONS updates are the Reporter's job
(/daily-digest), because they need the paired control.
"""
from __future__ import annotations

import argparse
import csv
import io
import subprocess

from _common import RESEARCH, load_config, read_status, write_status


def fetch(cfg) -> list[dict]:
    r = subprocess.run(["kaggle", "competitions", "submissions", "-c", cfg["competition"], "--csv"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SystemExit(r.stderr or r.stdout)
    return list(csv.DictReader(io.StringIO(r.stdout)))


def match(subs: list[dict], ideas: dict[str, str]) -> list[tuple[str, dict]]:
    out = []
    for s in subs:
        text = " ".join(str(v) for v in s.values()).lower()
        for idea, slug in ideas.items():
            if slug and slug.split("/")[-1].lower() in text:
                out.append((idea, s))
    return out


def main(argv=None, fetcher=fetch) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config()
    ideas = {}
    for st in RESEARCH.glob("*/*/STATUS"):
        _, meta = read_status(st.parent)
        if meta.get("kernel"):
            ideas[st.parent.relative_to(RESEARCH).as_posix()] = meta["kernel"]
    subs = fetcher(cfg)
    matched = match(subs, ideas)
    seen = {id(s) for _, s in matched}
    for s in subs[:15]:
        if id(s) not in seen:
            print(f"UNMATCHED\t{s.get('date', '')}\t{s.get('description', '')}\tLB={s.get('publicScore', '')}")
    for idea, s in matched:
        score = s.get("publicScore") or s.get("public_score") or ""
        print(f"{idea}\t{s.get('date', '')}\t{s.get('status', '')}\tLB={score}")
        if a.write and score:
            d = RESEARCH / idea
            state, _ = read_status(d)
            write_status(d, "scored" if state in ("ready-to-submit", "submitted") else state, lb=score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
