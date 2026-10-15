"""Promote human-approved lesson / rule proposals into LESSONS.md and research/rules/ (grill 2026-10-13).

    python research/bin/promote.py research/proposals/<file>.md [--dry-run]

The loop never edits LESSONS.md or a rule on its own: the report step drafts `research/proposals/*.md`, the human sets
`decision: approve` (or `reject`) under each item, and this script applies the approved ones:

  lesson add     -> new `- **Lnn** <text> Src: <proposal>` line (LESSONS.md stays append-only)
  lesson modify  -> new `- **Lnn** (supersedes Lxx) <text>` line
  lesson retire  -> new `- **Lnn** retires Lxx: <reasoning>` line
  rule add       -> research/rules/<ID>_<slug>.py from the ```python block, fixtures from ```file:<path> blocks
  rule modify    -> the rule file is replaced (the old one goes to research/rules/retired/)
  rule retire    -> the rule file and its fixtures move to research/rules/retired/

Every rule change must keep `preflight.py --selftest` green, or it is rolled back and marked `promotion: failed`.
An item is marked `decision: promoted <date>` so a rerun never applies it twice.

Item format (one `## ` section per item):
    ## P1: <title>
    - action: add | modify | retire
    - kind: lesson | rule
    - target: L12 | R04          (modify / retire only)
    - decision: pending          (the human writes approve / reject)
    ### Text            (lesson: the one-line lesson; rule: a ```python block with the whole module)
    ### Reasoning / ### Evidence
    ### Fixture         (rule add/modify: ```file:fixtures/R09_case/STATUS blocks, paths relative to research/rules/)
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import _common

FIELD = re.compile(r"^- (action|kind|target|decision):\s*(.*?)\s*$", re.M)


def items(md: str) -> list[dict]:
    out = []
    parts = re.split(r"(?m)^## ", md)
    for i, p in enumerate(parts[1:], 1):
        f = dict(FIELD.findall(p))
        title = p.splitlines()[0].strip()
        text = _section(p, "Text")
        code = re.search(r"```python\n(.*?)```", text, re.S)
        files = re.findall(r"```file:(\S+)\n(.*?)```", p, re.S)
        out.append(dict(index=i, title=title, raw=p, action=f.get("action", "").lower(), kind=f.get("kind", "").lower(),
                        target=f.get("target", ""), decision=f.get("decision", "").lower(),
                        text=re.sub(r"```.*?```", "", text, flags=re.S).strip(), code=code.group(1) if code else "",
                        files=files, reasoning=_section(p, "Reasoning")))
    return out


def _section(block: str, name: str) -> str:
    m = re.search(rf"^### {name}\s*\n(.*?)(?=^### |\Z)", block, re.S | re.M)
    return m.group(1).strip() if m else ""


def next_lesson_id(lessons: str) -> str:
    ids = [int(x) for x in re.findall(r"^- \*\*L(\d+)\*\*", lessons, re.M)]
    return f"L{max(ids, default=0) + 1:02d}"


def selftest() -> tuple[bool, str]:
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "preflight.py"), "--selftest"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode == 0, (r.stdout + r.stderr)[-1500:]


def failing(out: str) -> set[str]:
    """The `FAIL Rnn` lines of a selftest run."""
    return {ln.strip() for ln in out.splitlines() if ln.startswith("FAIL ")}


def apply_lesson(it: dict, src: str, lessons_path: Path) -> str:
    text = lessons_path.read_text(encoding="utf-8")
    lid = next_lesson_id(text)
    body = it["text"].splitlines()[0] if it["text"] else it["title"]
    if it["action"] == "add":
        line = f"- **{lid}** {body} Src: {src}."
    elif it["action"] == "modify":
        line = f"- **{lid}** (supersedes {it['target']}) {body} Src: {src}."
    elif it["action"] == "retire":
        line = f"- **{lid}** retires {it['target']}: {it['reasoning'].splitlines()[0] if it['reasoning'] else body} Src: {src}."
    else:
        raise ValueError(f"unknown action {it['action']!r}")
    with lessons_path.open("a", encoding="utf-8") as f:
        f.write(("" if text.endswith("\n") else "\n") + line + "\n")
    return lid


def apply_rule(it: dict, rules: Path, test=selftest) -> str:
    """Write / replace / retire one rule; roll back if the selftest gains a failure (a live idea that an existing
    rule already flags must not block every promotion)."""
    base_ok, base_out = test()
    before = set() if base_ok else failing(base_out)
    retired = rules / "retired"
    touched: list[tuple[Path, Path | None]] = []          # (path, backup or None = new file)

    def put(path: Path, content: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        bak = None
        if path.exists():
            retired.mkdir(exist_ok=True)
            bak = retired / f"{path.name}.{time.strftime('%Y%m%d%H%M%S')}"
            shutil.copy2(path, bak)
        touched.append((path, bak))
        path.write_text(content, encoding="utf-8")

    def rollback():
        for path, bak in reversed(touched):
            if bak is None:
                path.unlink(missing_ok=True)
            else:
                shutil.copy2(bak, path)

    if it["action"] in ("add", "modify"):
        m = re.search(r'^ID\s*=\s*"(R\d+)"', it["code"], re.M)
        if not m:
            raise ValueError("rule code has no ID = \"Rnn\" line")
        rid = it["target"] if it["action"] == "modify" else m.group(1)
        existing = sorted(rules.glob(f"{rid}_*.py"))
        if it["action"] == "add" and existing:
            raise ValueError(f"{rid} already exists ({existing[0].name})")
        if it["action"] == "modify" and not existing:
            raise ValueError(f"{rid} does not exist")
        slug = re.sub(r"\W+", "_", it["title"].split(":", 1)[-1].strip().lower())[:40].strip("_") or "rule"
        put(existing[0] if existing else rules / f"{rid}_{slug}.py", it["code"])
        for rel, content in it["files"]:
            put(rules / rel, content)
    elif it["action"] == "retire":
        retired.mkdir(exist_ok=True)
        for p in list(rules.glob(f"{it['target']}_*.py")) + list((rules / "fixtures").glob(f"{it['target']}_*")):
            dst = retired / p.name
            shutil.move(str(p), str(dst))
            touched.append((dst, None))           # a failed retire moves them back below
        if not touched:
            raise ValueError(f"nothing to retire for {it['target']}")
    ok, out = test()
    if not ok and failing(out) - before or not ok and not failing(out):
        if it["action"] == "retire":
            for dst, _ in touched:
                shutil.move(str(dst), str((rules / "fixtures" if dst.is_dir() else rules) / dst.name))
        else:
            rollback()
        raise RuntimeError(f"selftest failed, rolled back:\n{out}")
    return it["target"] or "rule"


def promote(path: Path, dry_run: bool = False, test=selftest) -> list[str]:
    md = path.read_text(encoding="utf-8")
    src = path.relative_to(_common.REPO).as_posix() if path.is_relative_to(_common.REPO) else path.name
    report = []
    for it in items(md):
        if it["decision"] != "approve":
            report.append(f"P{it['index']} {it['title']}: {it['decision'] or 'no decision'} (skipped)")
            continue
        if dry_run:
            report.append(f"P{it['index']} {it['title']}: would apply {it['kind']} {it['action']}")
            continue
        try:
            if it["kind"] == "lesson":
                res = apply_lesson(it, src, _common.RESEARCH / "LESSONS.md")
            elif it["kind"] == "rule":
                res = apply_rule(it, _common.RESEARCH / "rules", test)
            else:
                raise ValueError(f"unknown kind {it['kind']!r}")
            mark = f"- decision: promoted {time.strftime('%Y-%m-%d')} ({res})"
        except Exception as e:  # noqa: BLE001 - one bad item must not block the others
            mark = f"- decision: approve\n- promotion: failed {time.strftime('%Y-%m-%d')}: {str(e).splitlines()[0]}"
        new = re.sub(r"(?m)^- decision:\s*approve\s*$", mark, it["raw"], count=1)
        md = md.replace(it["raw"], new, 1)
        report.append(f"P{it['index']} {it['title']}: {mark.splitlines()[-1][2:]}")
    if not dry_run:
        path.write_text(md, encoding="utf-8")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("proposal")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    p = Path(a.proposal) if Path(a.proposal).is_absolute() else _common.REPO / a.proposal
    rep = promote(p, a.dry_run)
    print("\n".join(rep))
    return 1 if any("failed" in r for r in rep) else 0


if __name__ == "__main__":
    raise SystemExit(main())
