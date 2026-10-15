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
import run_tests  # noqa: E402


HOOKS: dict[str, list] = {"spawn": [], "toast": [], "queue": [], "fix": []}
PUSH_OK = "Kernel version 1 successfully pushed.  Please check progress at https://www.kaggle.com/code/x"


def repo(tmp_path, monkeypatch):
    """A throwaway repo root with research/config.yaml, a template and one idea."""
    research = tmp_path / "research"
    (research / "_template").mkdir(parents=True)
    (research / "_template" / "STATUS").write_text("drafting\nmode: autonomous\nbounces: 0\ncu_used: 0.0\n")
    (research / "_template" / "gates.md").write_text("| gate |\n")
    cfg = _common.load_config()
    cfg["claude"]["settings"] = ".claude/settings.headless.json"
    cfg["lines"]["exploit"]["worktree"] = str(tmp_path / "no_worktree")
    cfg["lines"]["exploit"]["control"] = None
    cfg["lines"]["exploit"]["g4_required"] = []
    cfg["gates"]["g4_draft_diff"] = False   # the draft-diff tests switch it on with a control draft
    for lc in cfg["lines"].values():
        lc.pop("control_draft", None)       # default: <control>/kaggle_output/submission.csv inside the tmp repo
    cfg["catalog"] = "catalog.json"
    (tmp_path / "catalog.json").write_text(json.dumps({"presets": {
        "p": {"enable_gpu": "true", "competition_sources": ["wrong-slug"],
              "dataset_sources": ["a/b", "c/d"]}}}))
    import yaml
    (research / "config.yaml").write_text(yaml.safe_dump(cfg))
    for mod in (_common, call_claude, colab_run, kaggle_push, poll_lb, nb_to_colab, run_tests):
        for name, val in (("REPO", tmp_path), ("RESEARCH", research)):
            if hasattr(mod, name):
                monkeypatch.setattr(mod, name, val)
    monkeypatch.setattr(_common, "load_config", lambda path=None: yaml.safe_load((research / "config.yaml").read_text()))
    for mod in (call_claude, colab_run, kaggle_push, poll_lb, nb_to_colab, run_tests):
        monkeypatch.setattr(mod, "load_config", _common.load_config)
    monkeypatch.setattr(colab_run, "login_problem", lambda shell=None: "")  # never read the real token
    # never start a real watcher, toast or Claude job from a test; record the calls instead
    for k in HOOKS:
        HOOKS[k] = []
    monkeypatch.setattr(kaggle_push, "spawn_watcher", lambda idea: HOOKS["spawn"].append(idea) or 0)
    monkeypatch.setattr(kaggle_push, "toast", lambda title, body: HOOKS["toast"].append(f"{title}: {body}"))
    monkeypatch.setattr(kaggle_push, "queue_ideas", lambda rels: HOOKS["queue"].append(list(rels)))
    monkeypatch.setattr(kaggle_push, "queue_fix", lambda rel: HOOKS["fix"].append(rel))
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
def _fake_runner(calls, fail_on=None, mount_ok=True, exec_out="exec out"):
    def run(cli, args, timeout=None):
        calls.append(args[0])
        rc = 1 if args[0] == fail_on else 0
        if args[0] == "exec" and any("_mount_probe" in x for x in args):
            return types.SimpleNamespace(returncode=0, stdout="MOUNT_OK" if mount_ok else "MOUNT_MISSING",
                                         stderr="")
        out = exec_out if args[0] == "exec" else f"{args[0]} out"
        return types.SimpleNamespace(returncode=rc, stdout=out, stderr="")
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


def test_colab_error_in_output_is_fail_even_with_rc0(repo):
    _, idea = repo
    (idea / "s.py").write_text("print(1)")
    calls = []
    err = "\x1b[0;31mAssertionError\x1b[0m: Drive not mounted"
    rc = colab_run.main([str(idea), "s.py", "--kind", "smoke"], runner=_fake_runner(calls, exec_out=err))
    assert rc == 3 and "| G3a |" in (idea / "gates.md").read_text()
    assert "| FAIL |" in (idea / "gates.md").read_text()


def test_colab_missing_script_starts_no_session(repo):
    _, idea = repo
    calls = []
    assert colab_run.main([str(idea), "nope.py", "--kind", "smoke"], runner=_fake_runner(calls)) == 1
    assert calls == []


def test_colab_remounts_once_then_fails(repo):
    _, idea = repo
    (idea / "s.py").write_text("print(1)")
    calls = []
    rc = colab_run.main([str(idea), "s.py", "--kind", "smoke"], runner=_fake_runner(calls, mount_ok=False))
    assert rc == 1 and calls.count("drivemount") == 2 and calls[-1] == "stop"
    assert calls.count("exec") == 2  # two probes, the real script never ran


def test_colab_drivemount_error_fails_fast(repo):
    _, idea = repo
    (idea / "s.py").write_text("print(1)")
    calls = []
    base = _fake_runner(calls)

    def run(cli, args, timeout=None):
        if args[0] == "drivemount":
            calls.append("drivemount")
            return types.SimpleNamespace(returncode=0, stdout="<p><b>400.</b>",
                                         stderr="\x1b[0;31mValueError\x1b[0m: mount failed\n")
        return base(cli, args, timeout)
    assert colab_run.main([str(idea), "s.py", "--kind", "smoke"], runner=run) == 1
    assert calls == ["new", "drivemount", "stop"]
    assert "Drive scope" in (idea / "colab_smoke.log").read_text(encoding="utf-8")


def test_login_problem_reads_scopes():
    def shell(out):
        return lambda *a, **k: types.SimpleNamespace(stdout=out, returncode=0)
    full, narrow = colab_run.DRIVE_SCOPE, colab_run.DRIVE_SCOPE + ".file"
    assert colab_run.login_problem(shell(json.dumps({"scopes": [full, "openid"]}))) == ""
    assert "drive.file" in colab_run.login_problem(shell(json.dumps({"scopes": [narrow]})))
    assert "not logged in" in colab_run.login_problem(shell("cat: no such file"))


def test_colab_login_needed_asks_and_starts_no_session(repo, monkeypatch):
    _, idea = repo
    (idea / "s.py").write_text("print(1)")
    monkeypatch.setattr(colab_run, "login_problem", lambda shell=None: "token lacks the full Drive scope")
    calls = []
    assert colab_run.main([str(idea), "s.py", "--kind", "smoke"], runner=_fake_runner(calls)) == 7
    assert calls == []
    q = (idea / "questions.md").read_text(encoding="utf-8")
    assert "Colab login needed" in q and "token.json.bak" in q
    assert "| G3a |" in (idea / "gates.md").read_text() and "BLOCKED" in (idea / "gates.md").read_text()


def test_colab_retries_used_up_falls_back(repo):
    _, idea = repo
    (idea / "s.py").write_text("print(1)")
    (idea / "gates.md").write_text("| gate |\n| G3a | t | FAIL | x | 0.07 |\n| G3a | t | FAIL | y | 0.07 |\n")
    calls = []
    assert colab_run.main([str(idea), "s.py", "--kind", "smoke"], runner=_fake_runner(calls)) == 4
    assert calls == []


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
        out = 'has status "KernelWorkerStatus.COMPLETE"' if args[1] == "status" else PUSH_OK
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


def _kaggle_run(calls, log):
    def run(args, timeout=600):
        calls.append(args)
        out = 'has status "KernelWorkerStatus.COMPLETE"' if args[1] == "status" else PUSH_OK
        if args[1] == "output":
            assert "--file-pattern" in args
            d = Path(args[args.index("-p") + 1]); d.mkdir(parents=True, exist_ok=True)
            (d / "k.log").write_text(log)
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")
    return run


def test_g4_fails_on_engine_failed_even_if_complete(repo):
    root, idea = repo
    nb = root / "probe.ipynb"
    nb.write_text("{}")
    log = json.dumps([{"stream_name": "stdout", "time": 1, "data": "ENGINE FAILED -> v4g lists kept\n"},
                      {"stream_name": "stdout", "time": 2, "data": "TRACK4 fired 3\n"}])
    rc = kaggle_push.main([str(idea), str(nb), "--preset", "p", "--wait", "100", "--counter", "TRACK4"],
                          run=_kaggle_run([], log), sleep=lambda s: None)
    assert rc == 2
    assert "FAIL (complete; marker 'ENGINE FAILED'" in (idea / "gates.md").read_text()
    assert _common.read_status(idea)[0] != "ready-to-submit"


def test_g4_requires_line_health_lines(repo):
    root, idea = repo
    import yaml
    cfgp = root / "research" / "config.yaml"
    cfg = yaml.safe_load(cfgp.read_text())
    cfg["lines"]["exploit"]["g4_required"] = [r"CLAW STAGE SCORED \| ice=[1-9]"]
    cfgp.write_text(yaml.safe_dump(cfg))
    nb = root / "probe.ipynb"
    nb.write_text("{}")
    args = [str(idea), str(nb), "--preset", "p", "--wait", "100"]
    assert kaggle_push.main(args, run=_kaggle_run([], "CLAW STAGE SCORED | ice=0 gl=0\n"),
                            sleep=lambda s: None) == 2
    assert kaggle_push.main(args, run=_kaggle_run([], "CLAW STAGE HEALTH | ICE NOT RUN\n"
                                                      "CLAW STAGE SCORED | ice=11 gl=11\n"),
                            sleep=lambda s: None) == 0


def test_g4_fails_when_counter_never_prints(repo):
    root, idea = repo
    nb = root / "probe.ipynb"
    nb.write_text("{}")
    rc = kaggle_push.main([str(idea), str(nb), "--preset", "p", "--wait", "100", "--counter", "TRACK4"],
                          run=_kaggle_run([], "all fine\n"), sleep=lambda s: None)
    assert rc == 2 and "counter 'TRACK4' not printed" in (idea / "gates.md").read_text()


def test_check_finalizes_only_finished_kernels(repo):
    root, idea = repo
    _common.write_status(idea, "g4-pushed", kernel="syedtoubur/casmi-foo")
    (idea / "kaggle_push").mkdir()
    (idea / "kaggle_push" / "counters.txt").write_text("TRACK4\n")
    status = {"s": "RUNNING"}

    def run(args, timeout=600):
        if args[1] == "status":
            return types.SimpleNamespace(returncode=0, stdout=f'has status "KernelWorkerStatus.{status["s"]}"',
                                         stderr="")
        return _kaggle_run([], "TRACK4 fired 3\n")(args, timeout)
    assert kaggle_push.main(["--check"], run=run) == 0
    assert _common.read_status(idea)[0] == "g4-pushed"
    status["s"] = "COMPLETE"
    assert kaggle_push.main(["--check", str(idea)], run=run) == 0
    assert _common.read_status(idea)[0] == "ready-to-submit"
    assert "| G4 | " in (idea / "gates.md").read_text() and "TRACK4 fired 3" in (idea / "gates.md").read_text()


def test_probe_refused_until_control_passes(repo):
    root, idea = repo
    import yaml
    cfgp = root / "research" / "config.yaml"
    cfg = yaml.safe_load(cfgp.read_text())
    cfg["lines"]["exploit"]["control"] = "research/exploit/ctl"
    cfgp.write_text(yaml.safe_dump(cfg))
    ctl = root / "research" / "exploit" / "ctl"
    ctl.mkdir()
    (ctl / "STATUS").write_text("scored\nline: exploit\nkind: control\ndocker_image: gcr.io/x@sha256:1\n")
    (ctl / "gates.md").write_text("| gate |\n| G4 | t | FAIL (error) | x | |\n")
    nb = root / "probe.ipynb"
    nb.write_text("{}")
    calls = []
    assert kaggle_push.main([str(idea), str(nb), "--preset", "p"], run=_kaggle_run(calls, "")) == 6
    assert calls == []
    (ctl / "gates.md").write_text("| gate |\n| G4 | t | PASS | x | |\n")
    assert kaggle_push.main([str(idea), str(nb), "--preset", "p"], run=_kaggle_run(calls, "")) == 0
    _common.write_status(idea, None, kind="control")  # a control push is never blocked
    (ctl / "gates.md").write_text("| gate |\n")
    assert kaggle_push.main([str(idea), str(nb), "--preset", "p"], run=_kaggle_run(calls, "")) == 0


def test_push_with_dropped_source_fails(repo):
    root, idea = repo
    nb = root / "probe.ipynb"
    nb.write_text("{}")

    def run(args, timeout=600):
        out = ("The following are not valid dataset sources and could not be added to the kernel: "
               "['a/b']\nKernel version 2 successfully pushed.")
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")
    assert kaggle_push.main([str(idea), str(nb), "--preset", "p"], run=run) == 1
    # only a catalog edit fixes an invalid source, so no fix agent: straight to the human
    assert _common.read_status(idea)[0] == "needs-human"
    assert "G4 needs a human" in (idea / "questions.md").read_text()
    assert HOOKS["fix"] == [] and HOOKS["spawn"] == []


def test_source_audit(repo):
    cat = {"datasets": {"g": {
        "rows": {"slug": "ahmed/casmi26-simulated-ranker-rows", "key_files": ["sim_rank_rows_nofp.npz"]},
        "fp2": {"slug": "prv/casmi26-fp-models-v2", "key_files": ["fp_0.pt"]}}}}
    code = ("subs = E.main(E.find('sim_rank_rows_nofp.npz'), E.find('test.parquet'))\n"
            "fp = [p for p in glob.glob('/kaggle/input/**/fp_*.pt', recursive=True) "
            "if 'casmi26-fp-models-v2' in p]\nx = find('mystery.npy')\n")
    missing, unresolved = kaggle_push.audit_sources(code, cat, ["a/b"])
    assert any("sim_rank_rows_nofp.npz -> needs ahmed/casmi26-simulated-ranker-rows" in m for m in missing)
    assert any("prv/casmi26-fp-models-v2" in m for m in missing)
    assert unresolved == ["mystery.npy"]
    missing, _ = kaggle_push.audit_sources(code, cat, ["ahmed/casmi26-simulated-ranker-rows",
                                                       "prv/casmi26-fp-models-v2"])
    assert missing == []


def test_probe_inherits_control_image(repo):
    """A probe runs on its control's pinned image (E006 asserts Python 3.12; Kaggle's latest is 3.13)."""
    import yaml
    root, idea = repo
    ctrl = root / "research" / "exploit" / "20261012-control-x"
    ctrl.mkdir()
    (ctrl / "STATUS").write_text("scored\nkind: control\ndocker_image: gcr.io/k/python@sha256:abc\n")
    (ctrl / "gates.md").write_text("| G4 | t | PASS | x |  |\n")
    cfgp = root / "research" / "config.yaml"
    cfg = yaml.safe_load(cfgp.read_text())
    cfg["lines"]["exploit"]["control"] = "research/exploit/20261012-control-x"
    cfgp.write_text(yaml.safe_dump(cfg))
    nb = root / "probe.ipynb"
    nb.write_text("{}")
    assert kaggle_push.main([str(idea), str(nb), "--preset", "p"], run=_kaggle_run([], PUSH_OK)) == 0
    md = json.loads((idea / "kaggle_push" / "kernel-metadata.json").read_text())
    assert md["docker_image"] == "gcr.io/k/python@sha256:abc"


def test_audit_blocks_push(repo):
    root, idea = repo
    (root / "catalog.json").write_text(json.dumps({"presets": {"p": {"dataset_sources": ["a/b"]}},
        "datasets": {"g": {"r": {"slug": "x/rows", "key_files": ["rows.npz"]}}}}))
    nb = root / "probe.ipynb"
    nb.write_text(json.dumps({"cells": [{"cell_type": "code", "source": "z = find('rows.npz')"}]}))
    calls = []
    assert kaggle_push.main([str(idea), str(nb), "--preset", "p"], run=_kaggle_run(calls, "")) == 5
    assert calls == [] and "needs x/rows" in (idea / "gates.md").read_text()


# ---------------- run_tests
def test_run_tests_uses_worktree_cwd(repo):
    root, idea = repo
    wt = root / "no_worktree"
    (wt / "build").mkdir(parents=True)
    (wt / "build" / "run_all_tests.sh").write_text(
        "set -e\npython build/test_a.py | tail -1\nrun build/test_b.py\nsoft build/test_c.py\nrun build/test_gone.py\n")
    for n in ("a", "b", "c"):
        (wt / "build" / f"test_{n}.py").write_text("")
    seen = []
    outcome = {"build/test_a.py": (0, "ok"), "build/test_b.py": (1, "FileNotFoundError: DreamS/dreams/utils/spectra.py"),
               "build/test_c.py": (1, "AssertionError: 22 != 21")}

    def run(cmd, **kw):
        seen.append((kw["cwd"], cmd[1]))
        rc, out = outcome[cmd[1]]
        return types.SimpleNamespace(returncode=rc, stdout=out + "\n", stderr="")
    # keep-going: every test runs in the worktree; a missing input is SKIP-INPUT, a real failure is FAIL
    assert run_tests.main(["exploit", "--idea", str(idea)], run=run) == 1
    assert [s[1] for s in seen] == ["build/test_a.py", "build/test_b.py", "build/test_c.py"]
    assert all(Path(c) == wt for c, _ in seen)
    g = (idea / "gates.md").read_text()
    assert "| G2 suite | " in g and "FAIL: build/test_c.py" in g and "SKIP-INPUT: build/test_b.py" in g
    outcome["build/test_c.py"] = (0, "ok")
    assert run_tests.main(["exploit"], run=run) == 0                         # skips alone don't fail G2


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


# ---------------- kaggle_push watcher and G4 hand-off (L34)
def _pushed(idea, push_id="1", **meta):
    _common.write_status(idea, "g4-pushed", kernel="syedtoubur/casmi-foo", push_id=push_id, **meta)
    (idea / "kaggle_push").mkdir(exist_ok=True)
    (idea / "kaggle_push" / "counters.txt").write_text("TRACK4\n")


def _seq_run(statuses, log="TRACK4 fired 3\n", calls=None):
    it = iter(statuses)

    def run(args, timeout=600):
        if calls is not None:
            calls.append(args)
        if args[1] == "status":
            s = next(it)
            if s == "ERR":
                return types.SimpleNamespace(returncode=1, stdout="", stderr="503 Service Unavailable")
            return types.SimpleNamespace(returncode=0, stdout=f'has status "KernelWorkerStatus.{s}"', stderr="")
        return _kaggle_run([], log)(args, timeout)
    return run


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_watch_finalizes_on_done_with_heartbeat(repo, capsys):
    _, idea = repo
    _pushed(idea)
    clk = _Clock()
    rc = kaggle_push.watch(_common.load_config(), idea, _seq_run(["QUEUED", "RUNNING", "COMPLETE"]),
                           clk.sleep, clk)
    assert rc == 0 and _common.read_status(idea)[0] == "ready-to-submit"
    out = capsys.readouterr().out
    assert out.count("[watch] syedtoubur/casmi-foo") == 3 and "t+60s" in out
    assert not (idea / "kaggle_watch" / "watch.lock").exists()
    assert HOOKS["toast"] and "G4 PASS" in HOOKS["toast"][0] and HOOKS["queue"] == []  # not a control


def test_watch_keeps_polling_through_cli_errors(repo):
    _, idea = repo
    _pushed(idea)
    clk = _Clock()
    rc = kaggle_push.watch(_common.load_config(), idea, _seq_run(["ERR", "RUNNING", "COMPLETE"]), clk.sleep, clk)
    assert rc == 0 and _common.read_status(idea)[0] == "ready-to-submit"


def test_watch_times_out_and_check_finishes_later(repo):
    import itertools
    _, idea = repo
    _pushed(idea)
    clk = _Clock()
    rc = kaggle_push.watch(_common.load_config(), idea, _seq_run(itertools.repeat("RUNNING")), clk.sleep, clk)
    assert rc == 4 and _common.read_status(idea)[0] == "g4-timeout"
    assert "| TIMEOUT |" in (idea / "gates.md").read_text()
    assert any("TIMEOUT" in t for t in HOOKS["toast"])
    assert kaggle_push.main(["--check"], run=_seq_run(["COMPLETE"])) == 0  # backup path picks it up
    assert _common.read_status(idea)[0] == "ready-to-submit"


def test_watch_exits_quietly_when_superseded(repo):
    import itertools
    _, idea = repo
    _pushed(idea, push_id="1")
    clk = _Clock()

    def sleep(s):  # a new push lands while this watcher sleeps
        _common.write_status(idea, None, push_id="2")
        clk.sleep(s)
    rc = kaggle_push.watch(_common.load_config(), idea, _seq_run(itertools.repeat("RUNNING")), sleep, clk)
    assert rc == 0 and _common.read_status(idea)[0] == "g4-pushed"
    assert "| G4 |" not in (idea / "gates.md").read_text() and HOOKS["toast"] == []


def test_watch_lock_fresh_refuses_stale_is_taken(repo):
    import os
    import time as _t
    _, idea = repo
    _pushed(idea, push_id="1")
    lock = idea / "kaggle_watch" / "watch.lock"
    lock.parent.mkdir()
    lock.write_text(json.dumps({"pid": os.getppid(), "t": _t.time(), "push_id": "1"}))  # a live owner
    calls = []
    assert kaggle_push.watch(_common.load_config(), idea, _seq_run(["COMPLETE"], calls=calls)) == 0
    assert calls == [] and _common.read_status(idea)[0] == "g4-pushed"
    assert kaggle_push.main(["--check"], run=_seq_run(["COMPLETE"], calls=calls)) == 0  # check skips it too
    assert calls == [] and _common.read_status(idea)[0] == "g4-pushed"
    lock.write_text(json.dumps({"pid": os.getppid(), "t": 0, "push_id": "1"}))  # stale by age
    assert kaggle_push.watch(_common.load_config(), idea, _seq_run(["COMPLETE"]), lambda s: None) == 0
    assert _common.read_status(idea)[0] == "ready-to-submit" and not lock.exists()


def test_watch_lock_of_older_push_is_taken(repo):
    import os
    import time as _t
    _, idea = repo
    _pushed(idea, push_id="2")
    lock = idea / "kaggle_watch" / "watch.lock"
    lock.parent.mkdir()
    lock.write_text(json.dumps({"pid": os.getppid(), "t": _t.time(), "push_id": "1"}))
    assert kaggle_push.watch(_common.load_config(), idea, _seq_run(["COMPLETE"]), lambda s: None) == 0
    assert _common.read_status(idea)[0] == "ready-to-submit"


def _fail_log():
    body = [f"line {i}" for i in range(100)]
    body[50] = "Traceback (most recent call last):"
    body[51] = "FileNotFoundError: x.npz"
    return "\n".join(body) + "\n"


def test_g4_fail_writes_brief_and_starts_fix_round(repo):
    root, idea = repo
    _pushed(idea)
    (idea / "kaggle_push" / "push_args.json").write_text(json.dumps(
        {"notebook": str(root / "probe.ipynb"), "preset": "p", "slug": "casmi-foo",
         "counters": ["TRACK4"], "extra_sources": []}))
    (idea / "spec.md").write_text("# Spec\n## Problem Statement\nOne change: rerank.\n## Solution\nsol\n"
                                  "## User Stories\nUS-1\n")
    rc = kaggle_push.watch(_common.load_config(), idea, _seq_run(["COMPLETE"], log=_fail_log()), lambda s: None)
    assert rc == 2
    state, meta = _common.read_status(idea)
    assert state == "g4-fix" and meta["g4_fixes"] == "1"
    assert HOOKS["fix"] == ["research/exploit/20261010-foo"]
    brief = (idea / "g4_fix_brief.md").read_text()
    ctx = brief.split("## Log tail")[0]
    assert "line 30" in ctx and "line 71" in ctx and "line 29" not in ctx  # ±20 lines of context
    assert "line 99" in brief.split("## Log tail")[1]
    assert 'kaggle_push.py research/exploit/20261010-foo' in brief and '--counter "TRACK4"' in brief
    assert "One change: rerank." in brief and "US-1" not in brief
    assert "counter 'TRACK4' not printed" in brief and "The catalog itself is never edited" in brief
    assert any("fix agent round 1" in t for t in HOOKS["toast"])


def test_g4_fail_after_fix_rounds_needs_human(repo):
    _, idea = repo
    _pushed(idea, g4_fixes=2)
    assert kaggle_push.watch(_common.load_config(), idea, _seq_run(["COMPLETE"], log=_fail_log()),
                             lambda s: None) == 2
    assert _common.read_status(idea)[0] == "needs-human"
    q = (idea / "questions.md").read_text()
    assert "G4 needs a human" in q and "g4_fix_brief.md" in q
    assert HOOKS["fix"] == [] and HOOKS["queue"] == []


def test_control_pass_queues_waiting_probes_on_its_line(repo):
    root, idea = repo
    _pushed(idea, kind="control")
    for line, name, state in (("exploit", "20261010-p1", "g2"), ("exploit", "20261010-p2", "ready-to-submit"),
                              ("explore", "20261010-p3", "g2")):
        d = root / "research" / line / name
        d.mkdir(parents=True)
        (d / "STATUS").write_text(f"{state}\nline: {line}\n")
    assert kaggle_push.watch(_common.load_config(), idea, _seq_run(["COMPLETE"]), lambda s: None) == 0
    assert HOOKS["queue"] == [["research/exploit/20261010-p1"]]  # one write, same line, waiting only
    assert "| G4 queue |" in (idea / "gates.md").read_text()
    assert "queued: 20261010-p1" in HOOKS["toast"][0]
    assert _common.read_status(idea)[1]["g4_fixes"] == "0"


def test_push_spawns_watcher_only_without_wait(repo):
    root, idea = repo
    nb = root / "probe.ipynb"
    nb.write_text("{}")
    assert kaggle_push.main([str(idea), str(nb), "--preset", "p", "--counter", "TRACK4"],
                            run=_kaggle_run([], "")) == 0
    assert len(HOOKS["spawn"]) == 1
    state, meta = _common.read_status(idea)
    assert state == "g4-pushed" and meta["push_id"]
    args = json.loads((idea / "kaggle_push" / "push_args.json").read_text())
    assert args["counters"] == ["TRACK4"] and args["notebook"].endswith("probe.ipynb")
    assert kaggle_push.main([str(idea), str(nb), "--preset", "p", "--wait", "100", "--counter", "TRACK4"],
                            run=_kaggle_run([], "TRACK4 x 3\n"), sleep=lambda s: None) == 0
    assert len(HOOKS["spawn"]) == 1


def test_repush_with_watch_log_open(repo):
    root, idea = repo
    nb = root / "probe.ipynb"
    nb.write_text("{}")
    (idea / "kaggle_watch").mkdir()
    with open(idea / "kaggle_watch" / "watch.log", "a") as f:  # an old watcher still holds it
        f.write("[watch] old\n")
        assert kaggle_push.main([str(idea), str(nb), "--preset", "p"], run=_kaggle_run([], "")) == 0
        assert kaggle_push.main([str(idea), str(nb), "--preset", "p"], run=_kaggle_run([], "")) == 0


def test_check_fail_also_starts_fix_round(repo):
    _, idea = repo
    _pushed(idea)
    assert kaggle_push.main(["--check"], run=_seq_run(["ERROR"], log=_fail_log())) == 2
    assert _common.read_status(idea)[0] == "g4-fix" and HOOKS["fix"] == ["research/exploit/20261010-foo"]
    assert "kernel status error" in (idea / "g4_fix_brief.md").read_text()


def test_extra_source_must_be_in_catalog_and_persists(repo):
    root, idea = repo
    cat = json.loads((root / "catalog.json").read_text())
    cat["datasets"] = {"g": {"dreams": {"slug": "x/dreams-np", "key_files": ["dreams_np_mass.npy"]}}}
    (root / "catalog.json").write_text(json.dumps(cat))
    nb = root / "probe.ipynb"
    nb.write_text("{}")
    calls = []
    assert kaggle_push.main([str(idea), str(nb), "--preset", "p", "--source", "x/not-there"],
                            run=_kaggle_run(calls, "")) == 5
    assert calls == []
    assert kaggle_push.main([str(idea), str(nb), "--preset", "p", "--source", "x/dreams-np"],
                            run=_kaggle_run(calls, "")) == 0
    assert _common.read_status(idea)[1]["extra_sources"] == "x/dreams-np"
    assert kaggle_push.main([str(idea), str(nb), "--preset", "p"], run=_kaggle_run(calls, "")) == 0
    md = json.loads((idea / "kaggle_push" / "kernel-metadata.json").read_text())
    assert md["dataset_sources"] == ["a/b", "c/d", "x/dreams-np"]


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
    assert state == "scored-unpaired" and meta["lb"] == "0.430"  # no control on the line


def test_poll_exact_token_and_claimed_refs(repo, capsys):
    root, idea = repo
    _common.write_status(idea, "ready-to-submit", kernel="syedtoubur/casmi-foo")
    other = root / "research" / "exploit" / "20261010-foo-v2"
    other.mkdir()
    (other / "STATUS").write_text("ready-to-submit\nline: exploit\nkernel: syedtoubur/casmi-foo-v2\n")
    subs = [{"ref": "9", "date": "d2", "description": "casmi-foo-v2", "publicScore": "0.441"},
            {"ref": "8", "date": "d1", "description": "", "publicScore": "0.437"}]
    got = poll_lb.match(subs, {"exploit/20261010-foo": "syedtoubur/casmi-foo",
                               "exploit/20261010-foo-v2": "syedtoubur/casmi-foo-v2"})
    assert [(i, s["ref"]) for i, s in got] == [("exploit/20261010-foo-v2", "9")]      # no substring match
    # an empty description is claimed by ref right after the submit
    assert poll_lb.main(["--claim", "research/exploit/20261010-foo"], fetcher=lambda cfg: subs) == 0
    state, meta = _common.read_status(idea)
    assert state == "submitted" and meta["lb_refs"] == "8"
    poll_lb.main(["--write"], fetcher=lambda cfg: subs)
    assert _common.read_status(idea)[1]["lb"] == "0.437"


def test_decide_rules():
    assert poll_lb.decide([0.432], None, True)[0] == "scored-unpaired"
    assert poll_lb.decide([0.440], 0.433, False)[0] == "revert"          # counters never fired
    assert poll_lb.decide([0.436], 0.433, True)[0] == "revert"           # inside the noise floor
    assert poll_lb.decide([0.437], 0.433, True)[0] == "keep-pending"
    assert poll_lb.decide([0.437, 0.437], 0.433, True)[0] == "keep"
    assert poll_lb.decide([0.437, 0.433], 0.433, True)[0] == "revert"    # replicate regressed


def _probe_with_g4_log(idea, log, counters="X fired\n"):
    (idea / "kaggle_push").mkdir(exist_ok=True)
    (idea / "kaggle_push" / "counters.txt").write_text(counters)
    (idea / "kaggle_output").mkdir(exist_ok=True)
    (idea / "kaggle_output" / "k.log").write_text(log)


def test_poll_pairs_probe_with_line_control(repo, capsys):
    root, idea = repo
    import yaml
    cfgp = root / "research" / "config.yaml"
    cfg = yaml.safe_load(cfgp.read_text())
    ctl = root / "research" / "exploit" / "20261010-ctl"
    ctl.mkdir()
    (ctl / "STATUS").write_text("ready-to-submit\nline: exploit\nkernel: syedtoubur/casmi-ctl\n")
    cfg["lines"]["exploit"]["control"] = "research/exploit/20261010-ctl"
    cfgp.write_text(yaml.safe_dump(cfg))
    _common.write_status(idea, "submitted", kernel="syedtoubur/casmi-foo", lb_skip=1)
    _probe_with_g4_log(idea, "X fired 5\n")
    subs = [{"date": "2026-10-11", "description": "casmi-foo", "publicScore": "0.440"},
            {"date": "2026-10-10", "description": "casmi-foo broken v1", "publicScore": "0.371"}]
    poll_lb.main(["--write"], fetcher=lambda cfg: subs)
    assert _common.read_status(idea)[0] == "scored-unpaired"          # control not scored yet
    subs.append({"date": "2026-10-11", "description": "casmi-ctl", "publicScore": "0.433"})
    poll_lb.main(["--write"], fetcher=lambda cfg: subs)
    state, meta = _common.read_status(idea)
    assert state == "keep-pending" and meta["lb"] == "0.440"           # lb_skip dropped 0.371
    assert _common.read_status(ctl)[0] == "scored"
    subs.append({"date": "2026-10-12", "description": "casmi-foo rep", "publicScore": "0.438"})
    poll_lb.main(["--write"], fetcher=lambda cfg: subs)
    assert _common.read_status(idea)[0] == "keep"


def test_poll_two_lines_sharing_one_control(repo, capsys):
    # 10-14: exploit and explore both point at control-e006; a dict keyed by path kept only the last line,
    # so exploit probes stayed scored-unpaired against a scored control
    root, idea = repo
    import yaml
    cfgp = root / "research" / "config.yaml"
    cfg = yaml.safe_load(cfgp.read_text())
    ctl = root / "research" / "exploit" / "20261010-ctl"
    ctl.mkdir()
    (ctl / "STATUS").write_text("scored\nline: exploit\nkind: control\nlb: 0.440\n")
    for ln in ("exploit", "explore"):
        cfg["lines"][ln]["control"] = "research/exploit/20261010-ctl"
    cfgp.write_text(yaml.safe_dump(cfg))
    _common.write_status(idea, "ready-to-submit", kernel="syedtoubur/casmi-foo", lb="0.431")
    _probe_with_g4_log(idea, "X fired 5\n")
    poll_lb.main(["--write"], fetcher=lambda cfg: [])
    state, meta = _common.read_status(idea)
    assert state == "revert" and "vs bar 0.444" in meta["decision"]


def test_instrument_pass_is_not_submittable_and_fail_skips_fix_agent(repo):
    root, idea = repo
    nb = root / "probe.ipynb"
    nb.write_text("{}")
    _common.write_status(idea, None, kind="instrument")
    args = [str(idea), str(nb), "--preset", "p", "--wait", "100", "--counter", "POST EQUIV ok"]
    assert kaggle_push.main(args, run=_kaggle_run([], "POST EQUIV ok 12 of 12\n"), sleep=lambda s: None) == 0
    assert _common.read_status(idea)[0] == "instrument-ready"
    assert kaggle_push.main(args, run=_kaggle_run([], "Traceback (most recent call last):\n"),
                            sleep=lambda s: None) == 2
    assert _common.read_status(idea)[0] == "needs-human" and not HOOKS["fix"]


def test_push_error_with_rc0_starts_no_watcher(repo):
    # 10-10: "Kernel push error: Maximum batch GPU session count of 2 reached." with exit code 0
    root, idea = repo
    nb = root / "probe.ipynb"
    nb.write_text("{}")
    run = lambda args, timeout=600: types.SimpleNamespace(
        returncode=0, stdout="Kernel push error: Maximum batch GPU session count of 2 reached.", stderr="")
    assert kaggle_push.main([str(idea), str(nb), "--preset", "p"], run=run) == 1
    assert _common.read_status(idea)[0] == "g4-failed" and not HOOKS["spawn"]


def test_counter_value_zero_fails_g4():
    # the real adduct-iceberg-rerank G4 log line that passed on presence alone (10-10)
    log = ("[ADDUCT_ICE] eligible molecules 1\n[ADDUCT_ICE] eligible 1 | base_reranked 0 | "
           "fused_reranked 0 | molecules reranked 0 | changed_top25 0 | candidates added 16\n")
    bad = kaggle_push.check_log(log, [r"\[ADDUCT_ICE\] eligible", r"\[ADDUCT_ICE\].*molecules reranked"], [])
    assert len(bad) == 1 and "molecules reranked" in bad[0] and "value 0" in bad[0]
    assert kaggle_push.counter_value(log, "candidates added") == 16
    assert not kaggle_push.check_log(log, [r"?\[ADDUCT_ICE\].*molecules reranked"], [])
    assert "value missing" in kaggle_push.check_log("TRACK4 ok\n", ["TRACK4"], [])[0]


def test_counter_value_reads_the_matched_count_not_the_runtime():
    # 10-12: `engine lists \d+` swallowed the count and read the runtime 619 (`engine lists 0 5s` passed)
    assert kaggle_push.counter_value("engine lists 12 619s\n", r"engine lists \d+") == 12
    assert kaggle_push.check_log("engine lists 0 5s\n", [r"engine lists \d+"], [])
    assert kaggle_push.counter_value("lists 7 of 9\n", r"lists (\d+) of") == 7
    assert kaggle_push.counter_value("TRACK4 fired 37\n", "TRACK4") == 37      # literal digit, not \d


def _drafts(root, idea, probe_rows, ctrl_rows, gate=True):
    import yaml
    cfgp = root / "research" / "config.yaml"
    cfg = yaml.safe_load(cfgp.read_text())
    ctl = root / "research" / "exploit" / "20261010-ctl"
    (ctl / "kaggle_output").mkdir(parents=True)
    (ctl / "STATUS").write_text("scored\nline: exploit\nkind: control\ndocker_image: gcr.io/x@sha256:1\n")
    (ctl / "gates.md").write_text("| G4 | x | PASS | |\n")
    cfg["lines"]["exploit"]["control"] = "research/exploit/20261010-ctl"
    cfg["gates"]["g4_draft_diff"] = gate
    cfgp.write_text(yaml.safe_dump(cfg))
    csv = lambda rows: "molecule_id,smiles\n" + "".join(f"{k},{v}\n" for k, v in rows.items())
    if ctrl_rows is not None:
        (ctl / "kaggle_output" / "submission.csv").write_text(csv(ctrl_rows))
    return lambda args, timeout=600: _write_probe_out(args, csv(probe_rows))


def _write_probe_out(args, sub):
    out = 'has status "KernelWorkerStatus.COMPLETE"' if args[1] == "status" else PUSH_OK
    if args[1] == "output":
        d = Path(args[args.index("-p") + 1]); d.mkdir(parents=True, exist_ok=True)
        (d / "k.log").write_text("X fired 3\n")
        (d / "submission.csv").write_text(sub)
    return types.SimpleNamespace(returncode=0, stdout=out, stderr="")


def test_g4_fails_a_probe_whose_draft_equals_the_control(repo):
    # 10-12 promo-soften: counters print-only, `promoted: 0`, and G4 still passed (N1)
    root, idea = repo
    nb = root / "probe.ipynb"; nb.write_text("{}")
    args = [str(idea), str(nb), "--preset", "p", "--wait", "100", "--counter", "X fired"]
    same = {"m1": "A;B", "m2": "C;D"}
    assert kaggle_push.main(args, run=_drafts(root, idea, same, dict(same)), sleep=lambda s: None) == 2
    assert "the change never fired" in (idea / "gates.md").read_text()
    assert _common.read_status(idea)[0] == "g4-fix" and HOOKS["fix"]


def test_g4_passes_a_probe_whose_draft_differs(repo):
    root, idea = repo
    nb = root / "probe.ipynb"; nb.write_text("{}")
    args = [str(idea), str(nb), "--preset", "p", "--wait", "100", "--counter", "X fired"]
    run = _drafts(root, idea, {"m1": "B;A", "m2": "C;D"}, {"m1": "A;B", "m2": "C;D"})
    assert kaggle_push.main(args, run=run, sleep=lambda s: None) == 0
    assert _common.read_status(idea)[0] == "ready-to-submit"


def test_push_refused_without_control_image_or_after_g2_fail(repo):
    root, idea = repo
    import yaml
    cfgp = root / "research" / "config.yaml"
    cfg = yaml.safe_load(cfgp.read_text())
    ctl = root / "research" / "exploit" / "20261010-ctl"
    ctl.mkdir()
    (ctl / "STATUS").write_text("scored\nline: exploit\nkind: control\n")
    (ctl / "gates.md").write_text("| G4 | x | PASS | |\n")
    cfg["lines"]["exploit"]["control"] = "research/exploit/20261010-ctl"
    cfgp.write_text(yaml.safe_dump(cfg))
    nb = root / "probe.ipynb"; nb.write_text("{}")
    calls = []
    assert kaggle_push.main([str(idea), str(nb), "--preset", "p"], run=_kaggle_run(calls, "")) == 10
    assert "docker_image" in (idea / "gates.md").read_text() and not calls
    (ctl / "STATUS").write_text("scored\nline: exploit\nkind: control\ndocker_image: gcr.io/x@sha256:1\n")
    with (idea / "gates.md").open("a") as f:
        f.write("| G2 suite | 2026 | FAIL | x |\n")
    assert kaggle_push.main([str(idea), str(nb), "--preset", "p"], run=_kaggle_run(calls, "")) == 10 and not calls
    with (idea / "gates.md").open("a") as f:
        f.write("| G2 suite | 2026 | PASS | x |\n")
    assert kaggle_push.main([str(idea), str(nb), "--preset", "p"], run=_kaggle_run(calls, "")) == 0
    md = json.loads((idea / "kaggle_push" / "kernel-metadata.json").read_text())
    assert md["docker_image"] == "gcr.io/x@sha256:1"


def test_gpu_pool_refuses_over_budget_and_keeps_g4_reserve(repo):
    import gpu
    root, idea = repo
    cfg = _common.load_config()
    assert gpu.refuse_reason(cfg, "training", 5.0).startswith("training run planned")     # > 4 h per run
    gpu.record("x", "k", 21.0, "instrument")
    assert gpu.refuse_reason(cfg, "probe", 0.5) == ""                                    # G4 may use the reserve
    assert "weekly GPU pool" in gpu.refuse_reason(cfg, "instrument", 3.0)                # 21 + 3 > 30 - 8
    nb = root / "probe.ipynb"; nb.write_text("{}")
    _common.write_status(idea, None, kind="training", gpu_h_est=3)
    assert kaggle_push.main([str(idea), str(nb), "--preset", "p"], run=_kaggle_run([], "")) == 8
    assert "GPU pool" in (idea / "questions.md").read_text()
    assert gpu.log_hours(json.dumps([{"time": 7.5}, {"time": 1080.0}])) == 0.3


def test_colab_train_needs_approval_and_fetches_on_failure(repo):
    root, idea = repo
    (idea / "t.py").write_text("print(1)")
    calls = []

    def runner(cli, args, timeout=None):
        calls.append(args[0])
        out = "MOUNT_OK" if args[0] == "exec" and "_mount_probe" in " ".join(args) else ""
        rc = 1 if args[0] == "exec" and "t.py" in " ".join(args) else 0
        return types.SimpleNamespace(returncode=rc, stdout=out, stderr="")
    assert colab_run.main([str(idea), "t.py", "--kind", "train"], runner=runner) == 9 and not calls
    _common.write_status(idea, None, colab_train="approved")
    import yaml
    cfgp = root / "research" / "config.yaml"
    cfg = yaml.safe_load(cfgp.read_text()); cfg["colab"]["cu_cap_train"] = 10.0; cfgp.write_text(yaml.safe_dump(cfg))
    rc = colab_run.main([str(idea), "t.py", "--kind", "train", "--fetch", "/content/ckpt.pt:ckpt.pt"], runner=runner)
    assert rc == 1 and "download" in calls and calls[-1] == "stop"          # outputs pulled although the run failed
    assert colab_run.check_mount(_common.load_config(), runner, shell=None) == 0


PROPOSAL = """# Proposals: cycle 1
## P1: pin the control image
- action: add
- kind: lesson
- decision: approve
### Text
A probe must run on its control's Kaggle image, or the pair is not paired.
### Reasoning
engine2-fppair v1 crashed on cp313.

## P2: retire an old lesson
- action: retire
- kind: lesson
- target: L01
- decision: reject
### Reasoning
superseded

## P3: counters need a number
- action: add
- kind: rule
- decision: approve
### Text
```python
ID = "R90"
TAGS = ["g4"]
LESSON = "test rule"
SRC = "test"
FLAGS = ["rules/fixtures/R90_bad"]


def check(idea):
    return ["bad"] if (idea / "BAD").exists() else []
```
### Fixture
```file:fixtures/R90_bad/BAD
x
```
"""


def test_promote_applies_only_approved_items_and_rolls_back_red_rules(repo):
    import promote
    root, _ = repo
    research = root / "research"
    (research / "LESSONS.md").write_text("- **L35** last.\n")
    (research / "rules" / "fixtures").mkdir(parents=True)
    (research / "proposals").mkdir()
    p = research / "proposals" / "20261013-c1.md"
    p.write_text(PROPOSAL)
    rep = promote.promote(p, test=lambda: (True, "ok"))
    lessons = (research / "LESSONS.md").read_text()
    assert "- **L36** A probe must run on its control's Kaggle image" in lessons and "L01" not in lessons
    assert (research / "rules" / "R90_counters_need_a_number.py").exists()
    assert (research / "rules" / "fixtures" / "R90_bad" / "BAD").exists()
    md = p.read_text()
    assert md.count("decision: promoted") == 2 and "decision: reject" in md
    assert promote.promote(p, test=lambda: (True, "ok")) and lessons == (research / "LESSONS.md").read_text()  # idempotent
    # a rule that turns the selftest red is rolled back and marked failed
    (research / "rules" / "R90_counters_need_a_number.py").unlink()
    __import__("shutil").rmtree(research / "rules" / "fixtures" / "R90_bad")
    p.write_text(PROPOSAL.replace("- kind: lesson\n- decision: approve", "- kind: lesson\n- decision: reject"))
    runs = iter([(False, "FAIL R02: a live idea"), (False, "FAIL R02: a live idea\nFAIL R90: flags x")])
    rep = promote.promote(p, test=lambda: next(runs))                 # before: R02 already red; after: + R90
    assert not (research / "rules" / "R90_counters_need_a_number.py").exists()
    assert "promotion: failed" in p.read_text() and any("failed" in r for r in rep)
    # an existing failure alone (a live idea an old rule flags) does not block a clean new rule
    p.write_text(PROPOSAL.replace("- kind: lesson\n- decision: approve", "- kind: lesson\n- decision: reject"))
    promote.promote(p, test=lambda: (False, "FAIL R02: a live idea"))
    assert (research / "rules" / "R90_counters_need_a_number.py").exists()


def test_dossier_ranks_ready_probes_and_writes_cards(repo):
    import dossier
    root, idea = repo
    mk = lambda name, comp, other: _ready(root, name, comp, other)
    mk("20261013-a", 0.004, 0.003); mk("20261013-b", 0.012, -0.001)
    _common.write_status(idea, "revert", lb="0.437", decision="0.437 vs bar 0.444")
    q = dossier.write_queue().read_text(encoding="utf-8")
    assert q.index("20261013-b") < q.index("20261013-a")                     # ranked by composite
    assert "leak-hit molecules only" in q and "| submit |" in q              # b's gain is leak-hit only
    assert "`syedtoubur/casmi-a` v3" in q and "LB 0.437" in q
    assert (root / "research" / "exploit" / "20261013-a" / "card.md").exists()


def _ready(root, name, comp, other):
    d = root / "research" / "exploit" / name
    d.mkdir(parents=True)
    (d / "STATUS").write_text(f"ready-to-submit\nline: exploit\nkernel: syedtoubur/casmi-{name[-1]}\nkernel_version: 3\n")
    (d / "harness_result.json").write_text(json.dumps(dict(
        schema="scorecard-1", target="C2", composite=comp, flags=[], knobs={"K": 1},
        cohorts={"C2": dict(n=400, delta=comp, ci90=[comp / 2, comp * 2], up=1, down=0, delta_leak_hit=comp,
                            delta_other=other, displaced=0)})))
    return d


def test_g4_without_control_draft_needs_human_not_a_fix_agent(repo):
    root, idea = repo
    nb = root / "probe.ipynb"; nb.write_text("{}")
    args = [str(idea), str(nb), "--preset", "p", "--wait", "100", "--counter", "X fired"]
    assert kaggle_push.main(args, run=_drafts(root, idea, {"m1": "A"}, None), sleep=lambda s: None) == 2
    assert _common.read_status(idea)[0] == "needs-human" and not HOOKS["fix"]
    assert "no control draft" in (idea / "questions.md").read_text()


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
