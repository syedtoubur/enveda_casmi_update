ID = "R12"
TAGS = ["spec", "revival"]
LESSON = ("A revival of a dead idea must name its cause and a fix that changes mechanism, data or target stratum; "
          "re-tuning a knob, reusing a signal already measured dead, or combining parts without distinct strata is refused.")
SRC = ("research/necropsy/dead_signals.yaml (S_nl/S_mult/S_frag fire 0.47-0.71 on C1/C2, research/c3/20261018-signal-audit/"
       "result.md); research/bin/necropsy.py; plan decision 13")
FLAGS = ["rules/fixtures/R12_knob_retune", "rules/fixtures/R12_dead_signal", "rules/fixtures/R12_shared_stratum"]

import re  # noqa: E402

import _util  # noqa: E402

DOCS = ("hypothesis.md", "spec.md")
REVIVES = re.compile(r"^\s*[-*]?\s*\**revives\**\s*:", re.I)
CAUSE = re.compile(r"^\s*[-*]?\s*\**cause\**\s*:\s*\S", re.I)
FIX = re.compile(r"^\s*[-*]?\s*\**fix\**\s*:\s*(\S.*)", re.I)
COMBINES = re.compile(r"^\s*[-*]?\s*\**combines\**\s*:(.*)", re.I)
KEY = re.compile(r"^\s*\**[A-Za-z_][\w ]*\**\s*:")
KNOB = re.compile(r"^(fix:\s*)?(raise|lower|tune|change|set|increase|decrease)\b.*"
                  r"\b(threshold|weight|margin|window|k|slot|cap|alpha|beta)\b", re.I)
MECH = re.compile(r"new signal|new feature|different cohort|stratum|model|data|detector|pool|generator", re.I)
STRATA = re.compile(r"stratum|cohort", re.I)
IDEA_REF = re.compile(r"\b(?:research/)?(?:[\w-]+/)?20\d{6}-[\w-]+|\b[\w-]+/20\d{6}-[\w-]+")


def dead_signals() -> list[str]:
    p = _util.REPO / "research" / "necropsy" / "dead_signals.yaml"
    if not p.exists():
        return []
    import yaml
    return [str(d["signal"]) for d in (yaml.safe_load(p.read_text(encoding="utf-8")) or []) if d.get("signal")]


def _combines_blocks(lines):
    """(lineno, block lines) for every combines: line plus its bullet / indented continuation."""
    for i, ln in enumerate(lines):
        m = COMBINES.match(ln)
        if not m:
            continue
        block = [m.group(1)]
        for nxt in lines[i + 1:]:
            if not nxt.strip():
                break
            if not nxt.lstrip().startswith(("-", "*")) and not nxt.startswith((" ", "\t")) and KEY.match(nxt):
                break
            block.append(nxt)
        yield i + 1, block


def check(idea):
    texts = {n: _util.text(idea, n) for n in DOCS}
    allines = [ln for t in texts.values() for ln in t.splitlines()]
    if not any(REVIVES.match(ln) for ln in allines):
        return []
    out = []
    if not any(CAUSE.match(ln) for ln in allines):
        out.append("revives: without a cause: line (R12; name the necropsy cause)")
    fixes = [m.group(1) for m in (FIX.match(ln) for ln in allines) if m]
    if not fixes:
        out.append("revives: without a fix: line (R12)")
    for f in fixes:
        if KNOB.match(f) and not MECH.search(f):
            out.append(f"fix only re-tunes a knob, no new mechanism/data/stratum: {f[:80]!r} (R12)")
    sigs = [(s, re.compile(r"(?<!\w)" + re.escape(s) + r"(?!\w)", re.I)) for s in dead_signals()]
    for n, t in texts.items():
        for i, ln in enumerate(t.splitlines(), 1):
            for s, rx in sigs:
                if rx.search(ln) and "MEASURED" not in ln:
                    out.append(f"{n}:{i}: names measured-dead signal {s!r} without MEASURED on the same line (R12)")
        for start, block in _combines_blocks(t.splitlines()):
            joined = "\n".join(block)
            if len(set(IDEA_REF.findall(joined))) >= 2 and not STRATA.search(joined):
                out.append(f"{n}:{start}: combines >= 2 ideas without naming distinct strata/cohorts (R12)")
    return out
