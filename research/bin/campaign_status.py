"""Read-only, deterministic summary of research campaigns (no model calls, never writes).

    python research/bin/campaign_status.py                       # 2 lines per campaign, newest first
    python research/bin/campaign_status.py --json                # structured, every campaign
    python research/bin/campaign_status.py --campaign <id>       # deep detail (markdown)
    python research/bin/campaign_status.py --campaign <id> --json
    python research/bin/campaign_status.py --repo <root>         # inspect another checkout

Paths resolve from this file (`parents[2]`, like `_common.REPO`) unless --repo / $CASMI_REPO is given.
Schemas vary between campaigns, so every read is defensive: a missing or odd file yields null, not a crash.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

NEEDS_STATUS = {"needs-human", "g4-timeout", "paused-quota"}
TERMINAL_STATUS = {"dead", "revert", "withdrawn", "killed", "wontfix"}
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


# ---------------------------------------------------------------- small helpers
def _section_fallback(md: str, title: str) -> str:
    m = re.search(rf"^## {re.escape(title)}\s*\n(.*?)(?=^## |\Z)", md, re.S | re.M)
    return m.group(1).strip() if m else ""


def _get_section():
    try:
        from brief import section  # research/bin/brief.py (needs yaml etc.; optional here)
        return section
    except Exception:
        return _section_fallback


section = _get_section()


def read_text(p: Path) -> str | None:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def read_json(p: Path):
    t = read_text(p)
    if t is None:
        return None
    try:
        return json.loads(t)
    except Exception:
        return None


def to_float(x):
    try:
        return float(str(x).strip())
    except Exception:
        return None


def read_status(idea: Path) -> tuple[str | None, dict]:
    t = read_text(idea / "STATUS")
    if t is None:
        return None, {}
    lines = t.splitlines()
    meta = {}
    for ln in lines[1:]:
        if ":" in ln:
            k, v = ln.split(":", 1)
            meta[k.strip()] = v.strip()
    return (lines[0].strip() if lines else None), meta


# ---------------------------------------------------------------- repo layout
class Repo:
    def __init__(self, root: Path):
        self.root = root
        self.research = root / "research"

    def campaign_ids(self) -> list[str]:
        d = self.research / "campaigns"
        if not d.is_dir():
            return []
        ids = [p.name for p in d.iterdir() if p.is_dir() and (p / "state.json").exists()]
        return sorted(ids, reverse=True)

    def idea_dirs(self) -> list[Path]:
        out = []
        if not self.research.is_dir():
            return out
        for line in self.research.iterdir():
            if not line.is_dir() or line.name in {"campaigns", "bin", "inbox", "harness", "conductor"}:
                continue
            for idea in line.iterdir():
                if idea.is_dir() and (idea / "STATUS").exists():
                    out.append(idea)
        return out

    def resolve_idea(self, ref: str, line: str | None = None) -> Path | None:
        ref = (ref or "").strip().replace("\\", "/")
        if not ref:
            return None
        p = self.root / ref
        if "/" in ref and p.is_dir():
            return p
        slug = ref.rstrip("/").split("/")[-1]
        cands = [d for d in self.idea_dirs() if d.name == slug]
        if not cands:
            cands = [d for d in self.idea_dirs() if d.name.endswith("-" + slug) or d.name.endswith(slug)]
        if line:
            pref = [d for d in cands if d.parent.name == line]
            cands = pref or cands
        return cands[0] if cands else None

    def rel(self, p: Path) -> str:
        try:
            return p.relative_to(self.root).as_posix()
        except Exception:
            return p.as_posix()


# ---------------------------------------------------------------- per-idea detail
def harness_summary(idea: Path) -> dict | None:
    d = read_json(idea / "harness_result.json")
    if not isinstance(d, dict):
        return None
    out = {"target": d.get("target"), "knobs": d.get("knobs"), "cohorts": {}}
    cohorts = d.get("cohorts")
    if isinstance(cohorts, dict):
        for name, c in cohorts.items():
            if not isinstance(c, dict):
                continue
            out["cohorts"][name] = {k: c.get(k) for k in
                                    ("n", "delta", "ci90", "delta_leak_hit", "delta_other", "displaced",
                                     "up", "down", "advisory") if k in c}
    return out


def last_gate_line(idea: Path) -> str | None:
    t = read_text(idea / "gates.md")
    if not t:
        return None
    rows = [ln.strip() for ln in t.splitlines() if ln.strip().startswith("|") and not set(ln.strip()) <= set("|-: ")]
    rows = [r for r in rows if not re.match(r"^\|\s*gate\s*\|", r, re.I)]
    return rows[-1] if rows else None


def necropsy_cause(idea: Path) -> str | None:
    t = read_text(idea / "necropsy.md")
    if not t:
        return None
    for ln in t.splitlines():
        if re.search(r"cause", ln, re.I) and ln.strip():
            return ln.strip()[:300]
    body = [ln.strip() for ln in t.splitlines() if ln.strip() and not ln.startswith("#")]
    return body[0][:300] if body else None


def open_questions(idea: Path) -> list[str]:
    t = read_text(idea / "questions.md")
    if not t:
        return []
    out = []
    for ln in t.splitlines():
        s = ln.strip()
        if re.match(r"^#{2,6}\s+\S", s):
            out.append(s.lstrip("# ").strip()[:160])
        elif re.match(r"^[-*]\s+\S", ln) and not ln.startswith((" ", "\t")):
            if not re.search(r"\b(resolved|answered|closed)\b", s, re.I):
                out.append(s[2:].strip()[:160])
    return out


def idea_detail(repo: Repo, ref: str, base: dict) -> dict:
    d = dict(base)
    d["idea"] = ref
    d.setdefault("status", None)
    d.setdefault("reason", None)
    folder = repo.resolve_idea(ref, base.get("_line"))
    d.pop("_line", None)
    d["folder"] = repo.rel(folder) if folder else None
    live, meta = read_status(folder) if folder else (None, {})
    d["status_live"] = live
    if live:
        d["status"] = live
    d["status_reason"] = meta.get("reason") or meta.get("note") or d.get("reason")
    if d.get("status_reason") and not d.get("reason"):
        d["reason"] = d["status_reason"]
    d["lb"] = meta.get("lb") or d.get("lb") or None
    d["gate_line"] = last_gate_line(folder) if folder else None
    d["harness"] = harness_summary(folder) if folder else None
    d["necropsy_cause"] = necropsy_cause(folder) if folder else None
    d["open_questions"] = open_questions(folder) if folder else []
    return d


# ---------------------------------------------------------------- campaign
def last_activity(log: str | None) -> str | None:
    if not log:
        return None
    dates = DATE_RE.findall(log)
    return max(dates) if dates else None


def inbox_pending(repo: Repo, cid: str) -> list[dict]:
    out = []
    d = repo.research / "inbox"
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.md")):
        if p.name.lower() == "readme.md":
            continue
        t = read_text(p) or ""
        m = re.match(r"^---\s*\n(.*?)\n---", t, re.S)
        if not m:
            continue
        fm = {}
        for ln in m.group(1).splitlines():
            if ":" in ln:
                k, v = ln.split(":", 1)
                fm[k.strip()] = v.split("#")[0].strip() if k.strip() == "status" else v.strip()
        if fm.get("campaign") == cid and fm.get("status", "pending") == "pending":
            out.append({"file": p.name, **{k: fm.get(k) for k in ("idea", "action", "args", "reason", "created")}})
    return out


def charter_info(text: str | None) -> dict:
    if not text:
        return {"goal": None, "line": None, "stop_rules": None}
    goal = section(text, "Goal")
    if not goal:
        m0 = re.search(r"\bGoal:\s*(.+?)(?:\n\s*\n|\Z)", text, re.S)
        goal = " ".join(m0.group(1).split()) if m0 else ""
    if not goal:
        body = text.split("\n", 1)[1] if text.startswith("#") else text
        paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip() and not p.strip().startswith("#")]
        goal = paras[0] if paras else ""
    m = re.search(r"^\s*-\s*Line:\s*`?([\w.-]*\w)", section(text, "Scope") or text, re.M)
    return {"goal": goal[:900] or None, "line": m.group(1) if m else None,
            "stop_rules": section(text, "Stop rules") or None}


def campaign_ideas(repo: Repo, cid: str, state: dict, line: str | None) -> list[dict]:
    ideas: list[dict] = []
    seen: set[str] = set()

    def add(ref: str, base: dict):
        key = ref.replace("\\", "/").rstrip("/").split("/")[-1]
        if not key or key in seen:
            return
        seen.add(key)
        base = dict(base)
        base["_line"] = line
        ideas.append(idea_detail(repo, ref, base))

    for h in state.get("history") or []:
        if isinstance(h, dict) and h.get("idea"):
            add(h["idea"], {k: h.get(k) for k in ("status", "reason", "branch", "now", "lb", "decision", "t")})
    port = state.get("portfolio")
    if isinstance(port, list):
        for e in port:
            if isinstance(e, dict) and e.get("slug"):
                base = {k: e.get(k) for k in ("status", "reason", "branch", "wave", "ceiling", "verdict",
                                              "measured_delta_c2", "measured_delta_c2_hard", "displacements")
                        if k in e}
                add(e["slug"], base)
    cur = state.get("current") or state.get("active_idea")
    if cur:
        add(cur, {"status": "current"})
    for d in repo.idea_dirs():
        _, meta = read_status(d)
        if meta.get("campaign") == cid:
            add(repo.rel(d), {})
    return ideas


def idea_best(i: dict) -> list[tuple[float, str]]:
    out = []
    for k in ("measured_delta_c2", "measured_delta_c2_hard"):
        v = to_float(i.get(k))
        if v is not None:
            out.append((v, k.replace("measured_delta_", "")))
    h = i.get("harness") or {}
    for name, c in (h.get("cohorts") or {}).items():
        if c.get("advisory"):
            continue
        v = to_float(c.get("delta_other"))
        v = v if v is not None else to_float(c.get("delta"))
        if v is not None:
            out.append((v, name))
    return out


def summarise(repo: Repo, cid: str) -> dict:
    cdir = repo.research / "campaigns" / cid
    state = read_json(cdir / "state.json")
    state = state if isinstance(state, dict) else {}
    charter = read_text(cdir / "charter.md")
    log = read_text(cdir / "log.md")
    ch = charter_info(charter)
    line = state.get("line") or ch["line"]
    ideas = campaign_ideas(repo, cid, state, line)
    if not line:
        for i in ideas:
            if i.get("folder"):
                line = i["folder"].split("/")[1] if i["folder"].count("/") >= 2 else None
                if line:
                    break

    stop = state.get("stop")
    st = state.get("status")
    if stop:
        run = f"stopped: {stop}"
    elif st and str(st).lower() not in {"running", "active"}:
        run = f"stopped: {st}"
    else:
        run = "running"
    cur = state.get("current") or state.get("active_idea")
    cur_ref = None
    cur_status = None
    if cur:
        cur_ref = str(cur).rstrip("/").split("/")[-1]
        for i in ideas:
            if i["idea"].rstrip("/").split("/")[-1] == cur_ref:
                cur_status = i.get("status_live") or i.get("status")
                break

    lbs = [(to_float(i.get("lb")), i["idea"]) for i in ideas if to_float(i.get("lb")) is not None]
    best_lb = max(lbs, default=None)
    deltas = [(v, c, i["idea"]) for i in ideas for v, c in idea_best(i)]
    best_delta = max(deltas, default=None)

    inbox = inbox_pending(repo, cid)
    needs = []
    for i in ideas:
        s = i.get("status_live") or i.get("status")
        qs = i.get("open_questions") or []
        if s in NEEDS_STATUS:
            needs.append({"idea": i["idea"], "status": s, "reason": i.get("reason"), "questions": qs})
        elif qs and s not in TERMINAL_STATUS:
            needs.append({"idea": i["idea"], "status": s, "reason": "open questions", "questions": qs})
    needs_count = len(needs) + len(inbox)

    return {
        "id": cid,
        "line": line,
        "running": run == "running",
        "run": run,
        "cycle": state.get("cycle"),
        "lb_probes": state.get("lb_probes"),
        "stop": stop,
        "current": cur_ref,
        "current_status": cur_status,
        "calls": state.get("calls"),
        "charter": ch,
        "ideas": ideas,
        "best_lb": {"lb": best_lb[0], "idea": best_lb[1]} if best_lb else None,
        "best_delta": {"delta": best_delta[0], "cohort": best_delta[1], "idea": best_delta[2]} if best_delta else None,
        "needs_human": needs,
        "inbox_pending": inbox,
        "needs_count": needs_count,
        "last_activity": last_activity(log),
        "log_tail": (log.splitlines()[-15:] if log else None),
    }


def gpu_info(repo: Repo):
    """Same arithmetic as gpu.week()/gpu.used(), read from this repo's ledger (works in any checkout)."""
    import datetime as dt
    rows = read_json(repo.research / "gpu_ledger.json")
    if not isinstance(rows, list):
        return None
    y, w, _ = dt.date.today().isocalendar()
    wk = f"{y}-W{w:02d}"
    used = 0.0
    for r in rows:
        if isinstance(r, dict) and r.get("week") == wk:
            used += to_float(r.get("hours")) or 0.0
    return {"week": wk, "used_h": round(used, 3)}


# ---------------------------------------------------------------- rendering
def fmt_delta(v: float) -> str:
    return f"{v:+.4f}"


def two_lines(n: int, c: dict) -> str:
    cur = f"{c['current']} ({c['current_status'] or 'STATUS unknown'})" if c["current"] else "none"
    run = c["run"] if len(c["run"]) <= 90 else c["run"][:87] + "..."
    l1 = f"{n}. {c['id']} [{c['line'] or '?'}] — {run} · cycle {c['cycle']} · current: {cur}"
    best = []
    if c["best_lb"]:
        best.append(f"LB {c['best_lb']['lb']:.3f} ({c['best_lb']['idea'].split('/')[-1]})")
    if c["best_delta"]:
        d = c["best_delta"]
        best.append(f"Δ {fmt_delta(d['delta'])} {d['cohort']} ({d['idea'].split('/')[-1]})")
    l2 = (f"   best: {' · '.join(best) if best else 'none measured'} · needs you: {c['needs_count']}"
          f" · last activity {c['last_activity'] or 'unknown'}")
    return l1 + "\n" + l2


def render_deep(c: dict, gpu) -> str:
    o = [f"# {c['id']} [{c['line'] or '?'}] — {c['run']}", ""]
    ch = c["charter"]
    o += ["## Charter", f"- goal: {ch['goal']}", f"- line: {ch['line'] or c['line']}",
          "- stop rules:", *(["  " + ln for ln in (ch["stop_rules"] or "null").splitlines()]), ""]
    o += ["## State", f"- cycle: {c['cycle']} · lb_probes: {c['lb_probes']} · stop: {c['stop']}",
          f"- current: {c['current']} ({c['current_status']})", f"- calls: {json.dumps(c['calls'])}", ""]
    o.append("## Ideas")
    for i in c["ideas"]:
        o.append(f"### {i['idea']} — {i.get('status')}")
        o.append(f"- reason: {i.get('reason')}")
        o.append(f"- gate: {i.get('gate_line')}")
        o.append(f"- LB: {i.get('lb')}")
        h = i.get("harness")
        if h:
            for name, cc in h["cohorts"].items():
                o.append(f"- harness {name}: Δ {cc.get('delta')} · leak-hit {cc.get('delta_leak_hit')}"
                         f" · other {cc.get('delta_other')} · n {cc.get('n')}")
        if i.get("necropsy_cause"):
            o.append(f"- necropsy: {i['necropsy_cause']}")
        for q in i.get("open_questions") or []:
            o.append(f"- open question: {q}")
    o += ["", "## Needs human"]
    for n in c["needs_human"]:
        o.append(f"- {n['idea']} [{n['status']}]: {n['reason']}")
    for p in c["inbox_pending"]:
        o.append(f"- inbox {p['file']}: {p['action']} {p['args']}")
    if not c["needs_human"] and not c["inbox_pending"]:
        o.append("- nothing")
    o += ["", "## GPU", f"- {json.dumps(gpu)}", "", "## Log tail"]
    o += [f"    {ln}" for ln in (c["log_tail"] or ["null"])]
    return "\n".join(o)


def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--campaign")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--repo", default=os.environ.get("CASMI_REPO"))
    a = ap.parse_args(argv)
    repo = Repo(Path(a.repo).resolve() if a.repo else Path(__file__).resolve().parents[2])

    if a.campaign:
        if a.campaign not in repo.campaign_ids():
            print(f"unknown campaign {a.campaign!r}; known: {', '.join(repo.campaign_ids())}", file=sys.stderr)
            return 2
        c = summarise(repo, a.campaign)
        g = gpu_info(repo)
        if a.json:
            print(json.dumps({**c, "gpu": g}, indent=2, ensure_ascii=False))
        else:
            print(render_deep(c, g))
        return 0

    cs = [summarise(repo, cid) for cid in repo.campaign_ids()]
    if a.json:
        print(json.dumps(cs, indent=2, ensure_ascii=False))
    else:
        print("\n".join(two_lines(n, c) for n, c in enumerate(cs, 1)) if cs else "no campaigns found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
