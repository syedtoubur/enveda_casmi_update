ID = "R11"
TAGS = ["spec", "c3"]
LESSON = ("A per-child trigger may not place a generated candidate above its parent (or at rank 1) unless its C1/C2 "
          "false-alarm rate is MEASURED at or below 4.5 %; every CPU child-vs-parent signal measured fires on 21-71 %.")
SRC = ("research/c3/20261018-signal-audit/result.md (kernel syedtoubur/c3iso-signal-audit v1): pooled C1/C2 fire "
       "S_nl 0.47, S_mult 0.53, S_frag 0.71, rule A 0.22, rule B 0.21; break-even FPR 4.5 % (research/c3/opus_audit_result.md §2)")
FLAGS = ["rules/fixtures/R11_above_parent_no_fpr"]

import re  # noqa: E402

DOCS = ("spec.md", "hypothesis.md", "research.md", "card.md")
ABOVE = re.compile(r"(above (its |the )?parent|\bat pos\(p\)(?!\s*\+)|"
                   r"\b(insert|push|plac|promot|mov)\w* (\S+ ){0,4}(to|at|into) (rank|slot)[- ]?1\b)", re.I)
NEG = re.compile(r"\b(never|not|no|dead|closed|forbid\w*|veto\w*|instead of)\b", re.I)
CHILD = re.compile(r"\b(child|children|mutant|generated|isomer|de novo|novel)\b", re.I)
FPR_OK = re.compile(r"C1/C2 (false[- ]alarm|FPR|fire)[^\n]*MEASURED", re.I)


def check(idea):
    if idea.parent.name not in ("c3", "fixtures"):          # Class 3 ideas only (child placement)
        return []
    txt = {n: (idea / n).read_text(encoding="utf-8", errors="replace") for n in DOCS if (idea / n).exists()}
    if any(FPR_OK.search(t) for t in txt.values()):
        return []
    out = []
    for n, t in txt.items():
        for i, ln in enumerate(t.splitlines(), 1):
            if ABOVE.search(ln) and CHILD.search(ln) and not NEG.search(ln):
                out.append(f"{n}:{i}: places a child above its parent / at rank 1 without a MEASURED C1/C2 "
                           f"false-alarm rate (bar <= 4.5 %, R11)")
    return out
