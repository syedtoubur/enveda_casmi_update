ID = "R04"
TAGS = ["spec", "premise"]
LESSON = "Every cited file:line exists (repo or exploit worktree) and the file has that many lines."
SRC = "seed rule: premise claims must be checkable (judge found misquoted :635/:553 in fusion-depth)"
FLAGS = []

import re  # noqa: E402
from _util import REPO, WORKTREE, lines  # noqa: E402

CITE = re.compile(r"`?([\w./\-]+\.(?:py|md|yaml|json|ipynb)):(\d+)")


def check(idea):
    out = []
    for f, i, ln in lines(idea, "spec.md", "hypothesis.md", "attack.md"):
        for path, n in CITE.findall(ln):
            cands = [REPO / path, WORKTREE / path, idea / path]
            hit = next((p for p in cands if p.is_file()), None)
            if hit is None:
                out.append(f"{f}:{i}: cited file not found: {path}")
            elif int(n) > sum(1 for _ in open(hit, encoding="utf-8", errors="replace")):
                out.append(f"{f}:{i}: {path} has fewer than {n} lines")
    return out
