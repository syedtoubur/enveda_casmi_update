"""Tests for research/bin (no network, no Claude, no Colab, no Kaggle).

    python research/bin/test_bin.py      (plain runner below; pytest also works)
"""
from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))
import _common  # noqa: E402
import call_claude  # noqa: E402
import colab_run  # noqa: E402
import kaggle_push  # noqa: E402
import nb_to_colab  # noqa: E402
import poll_lb  # noqa: E402


def repo(tmp_path, monkeypatch):
    """A throwaway repo root with research/config.yaml, a template and one idea."""
    research = tmp_path / "research"
    (research / "_template").mkdir(parents=True)
    (research / "_template" / "STATUS").write_text("drafting\nmode: autonomous\nbounces: 0\ncu_used: 0.0\n")
    (research / "_template" / "gates.md").write_text("| gate |\n")
    cfg = _common.load_config()
    cfg["claude"]["settings"] = ".claude/settings.headless.json"
    cfg["lines"]["exploit"]["worktree"] = str(tmp_path / "no_worktree")
    cfg["catalog"] = "catalog.json"
    (tmp_path / "catalog.json").write_text(json.dumps({"presets": {
        "p": {"enable_gpu": "true", "competition_sources": ["wrong-slug"],
              "dataset_sources": ["a/b", "c/d"]}}}))
    import yaml
    (research / "config.yaml").write_text(yaml.safe_dump(cfg))
    for mod in (_common, call_claude, colab_run, kaggle_push, poll_lb, nb_to_colab):
        for name, val in (("REPO", tmp_path), ("RESEARCH", research)):
            if hasattr(mod, name):
                monkeypatch.setattr(mod, name, val)
    monkeypatch.setattr(_common, "load_config", lambda path=None: yaml.safe_load((research / "config.yaml").read_text()))
    for mod in (call_claude, colab_run, kaggle_push, poll_lb, nb_to_colab):
        monkeypatch.setattr(mod, "load_config", _common.load_config)
    idea = research / "exploit" / "20261010-foo"
    idea.mkdir(parents=True)
    (idea / "STATUS").write_text("ready-for-judge\nline: exploit\ncu_used: 0.0\n")
    (idea / "gates.md").write_text("| gate |\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path, idea


# ---------------- _common
def test_status_roundtrip(repo):
    _, idea = repo
    _common.write_status(idea, "g1", bounces=2)
    state, meta = _common.read_status(idea)
    assert state == "g1" and meta["bounces"] == "2" and meta["line"] == "exploit"


def test_win_to_wsl():
    assert _common.win_to_wsl(Path("C:/Users/h/x.py")) == "/mnt/c/Users/h/x.py"


# ---------------- call_claude
def test_limit_detection_only_on_failed_result():
    assert call_claude.is_limit_line(json.dumps({"type": "result", "is_error": True,
                                                 "result": "Claude usage limit reached"}))
    assert not call_claude.is_limit_line(json.dumps({"type": "user", "content": "rate limit in code"}))
    assert not call_claude.is_limit_line(json.dumps({"type": "result", "subtype": "success",
                                                     "result": "fixed the rate limit bug"}))


def test_cmd_is_headless_and_scoped(repo):
    root, _ = repo
    cmd = call_claude.build_cmd(_common.load_config(), "research/exploit/x", root / "wt")
    assert cmd[:3] == ["claude", "-p", "/run-idea research/exploit/x"]
    assert "dontAsk" in cmd and "--strict-mcp-config" in cmd
    assert "--dangerously-skip-permissions" not in cmd
    assert cmd[cmd.index("--add-dir") + 1] == str(root / "wt")


def test_second_caller_queues_instead_of_running(repo, monkeypatch):
    root, idea = repo
    cfg = _common.load_config()
    lock = call_claude.Lock(root / cfg["claude"]["lock_file"])
    assert lock.acquire("other")  # held by this (live) process
    ran = []
    monkeypatch.setattr(call_claude, "run_one", lambda *a, **k: ran.append(a) or "done")
    assert call_claude.drain(cfg, "research/exploit/20261010-foo") == 0
    assert ran == []
    assert call_claude.queue_read(cfg) == ["research/exploit/20261010-foo"]
    lock.release()


def test_stale_lock_is_taken_over(repo):
    root, _ = repo
    p = root / "research/queue/claude.lock"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"pid": 999999, "idea": "x", "t": 0}))
    assert call_claude.Lock(p).acquire("me")


def test_drain_runs_queue_and_stops_on_pause(repo, monkeypatch):
    _, idea = repo
    cfg = _common.load_config()
    call_claude.queue_write(cfg, ["a", "b", "c"])
    outcomes = iter(["done", "paused"])
    monkeypatch.setattr(call_claude, "run_one", lambda cfg, rel, dry=False: next(outcomes))
    assert call_claude.drain(cfg, None) == 3
    assert call_claude.queue_read(cfg) == ["b", "c"]  # paused idea stays at the head


def test_run_one_marks_paused_on_limit(repo, monkeypatch):
    _, idea = repo
    lines = [json.dumps({"type": "result", "is_error": True, "result": "usage limit reached"}) + "\n"]

    class P:
        stdout = iter(lines)
        def wait(self): return 1
    monkeypatch.setattr(call_claude.subprocess, "Popen", lambda *a, **k: P())
    assert call_claude.run_one(_common.load_config(), "research/exploit/20261010-foo") == "paused"
    state, meta = _common.read_status(idea)
    assert state == "paused-quota" and meta["resume_from"] == "judging"
    assert (idea / "claude_run.jsonl").exists()
    # resuming restores the pre-pause state before Claude runs
    _common.write_status(idea, None, resume_from="g3a")
    seen = []

    class Q:
        stdout = iter([])
        def wait(self): return 0
    monkeypatch.setattr(call_claude.subprocess, "Popen",
                        lambda *a, **k: seen.append(_common.read_status(idea)[0]) or Q())
    assert call_claude.run_one(_common.load_config(), "research/exploit/20261010-foo") == "done"
    assert seen == ["g3a"]


def test_run_one_skips_wrong_state(repo):
    _, idea = repo
    _common.write_status(idea, "drafting")
    assert call_claude.run_one(_common.load_config(), "research/exploit/20261010-foo") == "skipped"


# ---------------- colab_run
def _fake_runner(calls, fail_on=None):
    def run(cli, args, timeout=None):
        calls.append(args[0])
        rc = 1 if args[0] == fail_on else 0
        return types.SimpleNamespace(returncode=rc, stdout=f"{args[0]} out", stderr="")
    return run


def test_colab_always_stops_even_on_failure(repo):
    _, idea = repo
    (idea / "s.py").write_text("print(1)")
    calls = []
    rc = colab_run.main([str(idea), "s.py", "--kind", "smoke"], runner=_fake_runner(calls, fail_on="exec"))
    assert rc == 1 and calls[-1] == "stop" and "exec" in calls
    assert "G3a | " in (idea / "gates.md").read_text()
    calls.clear()
    colab_run.main([str(idea), "s.py", "--kind", "smoke"], runner=_fake_runner(calls, fail_on="new"))
    assert calls == ["new", "stop"]


def test_colab_cu_cap_blocks(repo):
    _, idea = repo
    _common.write_status(idea, None, cu_smoke=0.5)
    calls = []
    assert colab_run.main([str(idea), "s.py", "--kind", "smoke"], runner=_fake_runner(calls)) == 4
    assert calls == []


def test_budget_seconds_caps_timeout():
    cfg = {"colab": {"cu_cap_smoke": 0.5, "cu_cap_val": 4.0, "cu_per_hour": 2.0, "exec_timeout_s": 3600}}
    assert colab_run.budget_seconds(cfg, "smoke", 0.0) == 900
    assert colab_run.budget_seconds(cfg, "val", 0.0) == 3600


# ---------------- kaggle_push
def test_metadata_from_catalog_overrides_slug(repo):
    md = kaggle_push.build_metadata(_common.load_config(), "p", "casmi-foo", "nb.ipynb")
    assert md["id"] == "syedtoubur/casmi-foo" and md["is_private"] and not md["enable_internet"]
    assert md["competition_sources"] == ["enveda-CASMI26-molecule-id-mass-spectra"]
    assert md["dataset_sources"] == ["a/b", "c/d"] and md["machine_shape"] == "NvidiaTeslaT4"


def test_parse_status():
    assert kaggle_push.parse_status('x has status "KernelWorkerStatus.COMPLETE"') == "complete"
    assert kaggle_push.parse_status('x has status "running"') == "running"


def test_push_never_submits_and_logs_g4(repo):
    root, idea = repo
    nb = root / "probe.ipynb"
    nb.write_text("{}")
    calls = []

    def run(args, timeout=600):
        calls.append(args)
        out = 'has status "KernelWorkerStatus.COMPLETE"' if args[1] == "status" else "ok"
        if args[1] == "output":
            d = Path(args[args.index("-p") + 1]); d.mkdir(parents=True, exist_ok=True)
            (d / "k.log").write_text("TRACK4 fired 37\nother\n")
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")
    rc = kaggle_push.main([str(idea), str(nb), "--preset", "p", "--wait", "100",
                           "--counter", "TRACK4"], run=run, sleep=lambda s: None)
    assert rc == 0
    assert not any("submit" in a for c in calls for a in c)
    assert "TRACK4 fired 37" in (idea / "gates.md").read_text()
    assert _common.read_status(idea)[0] == "ready-to-submit"


# ---------------- nb_to_colab
def test_colab_script_has_shim_then_cells(repo):
    root, idea = repo
    (root / "catalog.json").write_text(json.dumps({"presets": {}, "datasets": {
        "g": {"x": {"slug": "ahmedberatozer/casmi26-fpnet-full1"}}}}))
    nb = root / "p.ipynb"
    nb.write_text(json.dumps({"cells": [
        {"cell_type": "markdown", "source": ["# t"]},
        {"cell_type": "code", "source": ["import os\n", "X = 1"]},
        {"cell_type": "code", "source": "print(X)"}]}))
    nb_to_colab.main([str(idea), str(nb), "--kind", "val"])
    text = (idea / "colab" / "val.py").read_text()
    compile(text, "val.py", "exec")
    assert text.index("end shim") < text.index("X = 1") < text.index("print(X)")
    assert "'casmi26-fpnet-full1': 'ahmedberatozer'" in text and "_KIND = 'val'" in text
    assert "/kaggle/input/competitions/" in text and "# t" not in text


# ---------------- poll_lb
def test_poll_matches_by_slug_in_description(repo, capsys):
    _, idea = repo
    _common.write_status(idea, "submitted", kernel="syedtoubur/casmi-foo")
    subs = [{"date": "d1", "description": "casmi-foo control", "status": "COMPLETE", "publicScore": "0.430"},
            {"date": "d0", "description": "", "status": "COMPLETE", "publicScore": "0.332"}]
    poll_lb.main(["--write"], fetcher=lambda cfg: subs)
    out = capsys.readouterr().out
    assert "UNMATCHED" in out and "LB=0.430" in out
    state, meta = _common.read_status(idea)
    assert state == "scored" and meta["lb"] == "0.430"


# ---------------- plain runner (the repo runs tests as scripts; see build/run_all_tests.sh)
class _MonkeyPatch:
    def __init__(self):
        self._undo = []

    def setattr(self, obj, name, val):
        self._undo.append((obj, name, getattr(obj, name)))
        setattr(obj, name, val)

    def chdir(self, path):
        import os
        old = os.getcwd()
        self._undo.append((None, old, None))
        os.chdir(path)

    def undo(self):
        import os
        for obj, name, val in reversed(self._undo):
            if obj is None:
                os.chdir(name)
            else:
                setattr(obj, name, val)


class _Capsys:
    def __init__(self, buf):
        self.buf = buf

    def readouterr(self):
        out = self.buf.getvalue()
        self.buf.truncate(0); self.buf.seek(0)
        return types.SimpleNamespace(out=out, err="")


def _run_all() -> int:
    import contextlib, inspect, io, os, tempfile, traceback
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed, home = 0, os.getcwd()
    for name, fn in tests:
        mp, buf = _MonkeyPatch(), io.StringIO()
        td = tempfile.mkdtemp()
        try:
            with contextlib.redirect_stdout(buf):
                kw = {}
                for prm in inspect.signature(fn).parameters:
                    kw[prm] = {"repo": lambda: repo(Path(td), mp), "monkeypatch": lambda: mp,
                               "capsys": lambda: _Capsys(buf)}[prm]()
                fn(**kw)
        except Exception:
            failed += 1
            print(f"FAIL {name}\n{traceback.format_exc()}")
        finally:
            mp.undo()
            os.chdir(home)
            import shutil
            shutil.rmtree(td, ignore_errors=True)
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
