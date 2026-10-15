ID = "R04"
TAGS = ["spec", "premise"]
LESSON = "Every cited file:line exists (repo or a line worktree) and the file has that many lines."
SRC = ("seed rule: premise claims must be checkable (judge found misquoted :635/:553 in fusion-depth); "
       "2026-10-12 promo-soften: bare `casmi_post.py:230` was reported missing (loop audit W10)")
FLAGS = ["rules/fixtures/R04_line_past_end",
         # closed ideas: bare casmi/*.py cites (that package lives in .scratch/casmi_v4)
         "exploit/20261014-formula-window", "exploit/20261014-refspectra-lib"]

import re  # noqa: E402
from _util import REPO, WORKTREE, lines  # noqa: E402

CITE = re.compile(r"`?([\w./\-]+\.(?:py|md|yaml|json|ipynb)):(\d+)")
WORKTREES = [WORKTREE, WORKTREE.parent / "explore"]
# a bare file name (`casmi_post.py:230`) is looked up where the loop's code lives
CODE_DIRS = [d for wt in WORKTREES for d in (wt, wt / "build", wt / "scratch")] + \
            [REPO / "research" / "bin", REPO / "research" / "harness", REPO / "research" / "rules"]


def candidates(path: str, idea):
    yield from (REPO / path, idea / path, *(wt / path for wt in WORKTREES))
    if "/" not in path:
        yield from (d / path for d in CODE_DIRS)


def check(idea):
    out = []
    for f, i, ln in lines(idea, "spec.md", "hypothesis.md", "attack.md"):
        for path, n in CITE.findall(ln):
            hits = [p for p in candidates(path, idea) if p.is_file()]
            if not hits:
                out.append(f"{f}:{i}: cited file not found: {path}")
            elif all(int(n) > sum(1 for _ in open(h, encoding="utf-8", errors="replace")) for h in hits):
                out.append(f"{f}:{i}: {path} has fewer than {n} lines")
    return out
