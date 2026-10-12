ID = "R06"
TAGS = ["decision", "submit"]
LESSON = "A submit card names the paired own-account control kernel; KEEP is decided only by poll_lb.decide() (L05, L06)."
SRC = "20261010-adduct-iceberg-rerank: marked KEEP vs zenith 0.425 while the control was unscored"
FLAGS = ["rules/fixtures/R06_keep_vs_wrong_base"]

import re  # noqa: E402
from _util import text  # noqa: E402


def check(idea):
    card = text(idea, "submit_card.md")
    if not card or "<user/slug>" in card or "kind: control" in text(idea, "STATUS"):   # unfilled template
        return []
    out = []
    if not re.search(r"[Pp]aired control.*syedtoubur/[\w-]+", card):
        out.append("submit_card.md: no paired control kernel slug (syedtoubur/...)")
    if re.search(r"\bSOTA\b|\bDecision\b.*\bKEEP\b", card) and "decide" not in card:
        out.append("submit_card.md: claims KEEP/SOTA outside poll_lb.decide()")
    return out
