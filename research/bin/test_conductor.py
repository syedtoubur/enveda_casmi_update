"""Unit tests for conductor.py (fake subprocess, fake clock, tmp repo; no real CLIs, no network).

    python research/bin/test_conductor.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
import conductor as C  # noqa: E402

CFG = {"conductor": {"tick_s": 1, "max_workers": 2, "commit_state": False, "state_branch": "main",
                     "digest_hour": 7, "campaigns_file": "research/conductor/conductor.yaml"}}


class FakeProc:
    n = 100

    def __init__(self, rc=None):
        FakeProc.n += 1
        self.pid, self.rc = FakeProc.n, rc

    def poll(self):
        return self.rc


class FakePopen:
    def __init__(self):
        self.calls, self.procs = [], []

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        p = FakeProc()
        self.procs.append(p)
        return p


class FakeRunner:
    def __init__(self, outputs=None):
        self.calls, self.outputs = [], outputs or {}

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        for k, v in self.outputs.items():
            if any(k in str(x) for x in cmd):
                return SimpleNamespace(returncode=v[0], stdout=v[1], stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")


def make_repo(tmp: Path, campaigns=("camp-a", "camp-b"), paused=False):
    (tmp / "research" / "inbox").mkdir(parents=True)
    (tmp / "research" / "conductor").mkdir(parents=True)
    for c in campaigns:
        d = tmp / "research" / "campaigns" / c
        d.mkdir(parents=True)
        (d / "charter.md").write_text("# charter\n", encoding="utf-8")
        (d / "log.md").write_text("", encoding="utf-8")
        (d / "state.json").write_text(json.dumps({"stop": None, "cycle": 0}), encoding="utf-8")
    C.dump_yaml(tmp / "research" / "conductor" / "conductor.yaml",
                {"campaigns": [f"research/campaigns/{c}" for c in campaigns], "desks": ["chem"], "paused": paused})
    C.REPO, C.RESEARCH = tmp, tmp / "research"


def mk(popen=None, runner=None, now=[1_000_000.0], cfg=CFG):
    return C.Conductor(cfg=cfg, popen=popen or FakePopen(), runner=runner or FakeRunner(),
                       clock=lambda: now[0], toast=lambda t, b: None, describe=lambda c: f"next {c.name}")


def inbox_file(tmp, name, **fm):
    fm.setdefault("status", "pending")
    p = tmp / "research" / "inbox" / name
    p.write_text(C.join_front(fm, "body text\n"), encoding="utf-8")
    return p


def fm_of(p):
    return C.split_front(p.read_text(encoding="utf-8"))[0]


def with_repo(fn):
    def w():
        saved = (C.REPO, C.RESEARCH, C.BIN)
        tmp = Path(tempfile.mkdtemp())
        try:
            make_repo(tmp)
            fn(tmp)
        finally:
            C.REPO, C.RESEARCH, C.BIN = saved
            shutil.rmtree(tmp, ignore_errors=True)
    w.__name__ = fn.__name__
    return w


# ------------------------------------------------------------------ inbox
@with_repo
def test_inbox_pause_resume_kill_steer_note_approve(tmp):
    idea = tmp / "research" / "exploit" / "20260101-x"
    idea.mkdir(parents=True)
    (idea / "STATUS").write_text("attacked\nline: exploit\n", encoding="utf-8")
    p1 = inbox_file(tmp, "1.md", campaign="camp-a", action="pause")
    p2 = inbox_file(tmp, "2.md", campaign="camp-a", idea="research/exploit/20260101-x", action="kill", reason="bad")
    p3 = inbox_file(tmp, "3.md", campaign="camp-a", action="steer", args="try X")
    p4 = inbox_file(tmp, "4.md", campaign="camp-a", action="note", reason="hello")
    p5 = inbox_file(tmp, "5.md", campaign="camp-a", idea="research/exploit/20260101-x", action="approve", reason="ok go")
    mk().inbox()
    assert C.load_yaml(tmp / "research/conductor/conductor.yaml")["paused_campaigns"] == ["camp-a"]
    st = (idea / "STATUS").read_text(encoding="utf-8")
    assert st.startswith("dead\n") and "human (inbox): bad" in st
    assert "## Human steer" in (tmp / "research/campaigns/camp-a/charter.md").read_text(encoding="utf-8")
    assert "human note: hello" in (tmp / "research/campaigns/camp-a/log.md").read_text(encoding="utf-8")
    assert "## Approved by human" in (idea / "questions.md").read_text(encoding="utf-8")
    for p in (p1, p2, p3, p4, p5):
        fm = fm_of(p)
        assert fm["status"] == "applied" and fm["applied_at"] and fm["result"], (p, fm)
        assert "body text" in p.read_text(encoding="utf-8")
    inbox_file(tmp, "6.md", campaign="camp-a", action="resume")
    mk().inbox()
    assert C.load_yaml(tmp / "research/conductor/conductor.yaml")["paused_campaigns"] == []
    # already-applied files are not re-applied
    before = p3.read_text(encoding="utf-8")
    mk().inbox()
    assert p3.read_text(encoding="utf-8") == before


@with_repo
def test_inbox_rejections_and_raise_cap(tmp):
    bad = inbox_file(tmp, "a.md", campaign="camp-a", action="explode")
    nokill = inbox_file(tmp, "b.md", campaign="camp-a", idea="research/exploit/nope", action="kill")
    norev = inbox_file(tmp, "c.md", campaign="camp-a", idea="research/exploit/nope", action="revive", args="")
    cap = inbox_file(tmp, "d.md", campaign="camp-a", action="raise-cap", args="claude:opus 20")
    (tmp / "research/inbox/README.md").write_text("---\nstatus: pending\n---\n", encoding="utf-8")
    toasts = []
    c = mk()
    c.toast = lambda t, b: toasts.append(t)
    c.inbox()
    for p in (bad, nokill, norev):
        assert fm_of(p)["status"] == "rejected", p
    assert fm_of(cap)["status"] == "applied"
    assert C.load_yaml(tmp / "research/conductor/caps_override.yaml") == {"claude:opus": 20}
    assert "status: pending" in (tmp / "research/inbox/README.md").read_text(encoding="utf-8")
    assert len(toasts) == 3


@with_repo
def test_inbox_revive(tmp):
    src = tmp / "research" / "exploit" / "20260101-x"
    src.mkdir(parents=True)
    (src / "STATUS").write_text("dead\nline: exploit\n", encoding="utf-8")
    (src / "necropsy.md").write_text("# n\ncause: pool-reach\n", encoding="utf-8")
    new = tmp / "research" / "exploit" / "20261018-x-rev"

    def runner(cmd, **kw):
        new.mkdir(parents=True)
        return SimpleNamespace(returncode=0, stdout="research/exploit/20261018-x-rev\n", stderr="")
    p = inbox_file(tmp, "r.md", campaign="camp-a", idea="research/exploit/20260101-x", action="revive", args="widen pool")
    mk(runner=runner).inbox()
    h = (new / "hypothesis.md").read_text(encoding="utf-8")
    assert "revives: research/exploit/20260101-x" in h and "cause: cause: pool-reach" in h and "fix: widen pool" in h
    assert fm_of(p)["status"] == "applied"


# ------------------------------------------------------------------ workers
@with_repo
def test_workers_max_and_skips(tmp):
    pop = FakePopen()
    c = mk(popen=pop)
    c.workers_step()
    assert len(pop.calls) == 2 and all("--once" in x and "--router" in x for x in pop.calls)
    c.workers_step()
    assert len(pop.calls) == 2                       # slots full, workers alive
    # max_workers 1
    pop1 = FakePopen()
    cfg1 = {"conductor": dict(CFG["conductor"], max_workers=1)}
    mk(popen=pop1, cfg=cfg1).workers_step()
    assert len(pop1.calls) == 1 and "camp-a" in pop1.calls[0][2]


@with_repo
def test_workers_paused_and_stopped_skipped(tmp):
    pop = FakePopen()
    s = C.load_yaml(tmp / "research/conductor/conductor.yaml")
    s["paused_campaigns"] = ["camp-a"]
    C.dump_yaml(tmp / "research/conductor/conductor.yaml", s)
    (tmp / "research/campaigns/camp-b/state.json").write_text(json.dumps({"stop": "max_cycles"}), encoding="utf-8")
    mk(popen=pop).workers_step()
    assert pop.calls == []
    s["paused_campaigns"], s["paused"] = [], True
    C.dump_yaml(tmp / "research/conductor/conductor.yaml", s)
    mk(popen=pop).workers_step()
    assert pop.calls == []                            # global pause starts nothing


@with_repo
def test_workers_backoff_after_three_noops(tmp):
    pop, now = FakePopen(), [1_000_000.0]
    cfg1 = {"conductor": dict(CFG["conductor"], max_workers=1)}
    c = mk(popen=pop, now=now, cfg=cfg1)
    (tmp / "research/conductor/conductor.yaml").write_text(
        "campaigns: [research/campaigns/camp-a]\npaused: false\n", encoding="utf-8")
    for i in range(3):
        c.workers_step()
        assert len(pop.calls) == i + 1
        pop.procs[-1].rc = 0                          # finished, nothing changed
        c.reap()
    assert c.state["backoff"]["camp-a"] > now[0]
    c.workers_step()
    assert len(pop.calls) == 3                        # backed off
    now[0] += C.BACKOFF_S + 1
    c.workers_step()
    assert len(pop.calls) == 4
    # progress resets the counter
    pop.procs[-1].rc = 0
    (tmp / "research/campaigns/camp-a/state.json").write_text(json.dumps({"stop": None, "cycle": 9}), encoding="utf-8")
    c.reap()
    assert c.state["noop"]["camp-a"] == 0


# ------------------------------------------------------------------ desk / autosubmit hooks
@with_repo
def test_desk_and_autosubmit_hooks(tmp):
    bin_ = tmp / "bin"
    bin_.mkdir()
    C.BIN = bin_
    pop, run = FakePopen(), FakeRunner({"desk.py": (0, '["chem-triage"]'), "autosubmit.py": (0, "autosubmit: off (x)\n")})
    c = mk(popen=pop, runner=run)
    c.desks()
    c.autosubmit()
    assert pop.calls == [] and run.calls == []        # scripts missing -> skipped
    (bin_ / "desk.py").write_text("", encoding="utf-8")
    (bin_ / "autosubmit.py").write_text("", encoding="utf-8")
    c.desks()
    c.autosubmit()
    assert pop.calls and pop.calls[0][-2:] == ["run", "chem-triage"]
    assert any("autosubmit.py" in str(x) for call in run.calls for x in call)
    assert any("autosubmit: off" in ln for ln in (tmp / "research/conductor/conductor.log").read_text(encoding="utf-8").splitlines())


# ------------------------------------------------------------------ commit
@with_repo
def test_commit_step_gates(tmp):
    run = FakeRunner({"--show-current": (0, "main\n")})
    mk(runner=run).commit_state()
    assert run.calls == []                            # commit_state false -> no git at all
    cfg = {"conductor": dict(CFG["conductor"], commit_state=True, state_branch="main")}
    run = FakeRunner({"--show-current": (0, "feature\n")})
    mk(runner=run, cfg=cfg).commit_state()
    assert not any("add" in c or "commit" in c for c in run.calls)   # branch mismatch
    run = FakeRunner({"--show-current": (0, "main\n"), "--quiet": (1, "")})
    mk(runner=run, cfg=cfg).commit_state()
    assert any("add" in c for c in run.calls) and any("commit" in c for c in run.calls)
    assert not any("-f" in c or "--force" in c or "push" in c for c in run.calls)


@with_repo
def test_tick_survives_step_failure(tmp):
    c = mk()
    c.queue_digest = lambda: 1 / 0
    c.needs_human = lambda: []
    c.tick()
    assert "ERROR in queue" in (tmp / "research/conductor/conductor.log").read_text(encoding="utf-8")
    assert c.state["last_tick"]


def test_status_real_repo():
    assert C.status() == 0


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{'OK' if not fails else 'FAILED'}: {fails} failure(s)")
    sys.exit(1 if fails else 0)
