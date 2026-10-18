"""Unit tests for providers.py (fake subprocess, no network, no real CLIs).

    python research/bin/test_providers.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import providers as P  # noqa: E402
from _common import REPO, RESEARCH, load_config, read_status  # noqa: E402

OPENCODE_ERR = ("Error: [Google AI Studio] google/gemma-4-31b-it:free is temporarily rate-limited upstream. "
                "Please retry shortly")
QODER_LOGIN = "Not logged in. Run `qodercli login` to authenticate."
AGY_OK = json.dumps({"status": "SUCCESS", "result": "wrote spec.md (quota table, rate limit notes)"})
AGY_QUOTA = json.dumps({"status": "FAILED", "error": "429 RESOURCE_EXHAUSTED: quota exceeded"})


class FakeProc:
    def __init__(self, lines, rc):
        self.stdout = iter(l + "\n" for l in lines)
        self._rc = rc

    def wait(self):
        return self._rc

    def kill(self):
        pass


class FakePopen:
    def __init__(self, lines=(), rc=0):
        self.lines, self.rc, self.calls = list(lines), rc, []

    def __call__(self, cmd, **kw):
        self.calls.append((cmd, kw))
        return FakeProc(self.lines, self.rc)


def cfg() -> dict:
    return load_config()


def test_cmds(tmp: Path) -> None:
    c = cfg()
    wt = tmp / "wt"
    agy = P.AgyCLI(c).build_cmd("PROMPT", "claude-sonnet-4-6", wt)
    assert agy == [c["routing"]["bin"]["agy"], "-p", "PROMPT", "--model", "claude-sonnet-4-6", "--output-format", "json",
                   "--print-timeout", "2400s", "--add-dir", str(wt)], agy
    assert "--add-dir" not in P.AgyCLI(c).build_cmd("p", "m", None)
    q = P.QoderCLI(c).build_cmd("PROMPT", "qwen3.8-flash", wt)
    assert q == [c["routing"]["bin"]["qoder"], "-p", "PROMPT", "-m", "qwen3.8-flash", "-o", "json", "--permission-mode",
                 "dont_ask", "--no-session-persistence", "-w", str(wt)], q
    assert "-w" not in P.QoderCLI(c).build_cmd("p", "m", None)
    o = P.OpencodeCLI(c).build_cmd("PROMPT", "openrouter/openrouter/free", wt)
    assert o == ["opencode", "run", "--pure", "--format", "json", "-m", "openrouter/openrouter/free", "--dir", str(wt),
                 "PROMPT"], o
    assert "--auto" not in o and P.OpencodeCLI(c).build_cmd("p", "m", None)[-2] == str(REPO)
    print("commands ok")


def test_regex_and_paused(tmp: Path) -> None:
    c = cfg()
    for s in (OPENCODE_ERR, "usage limit reached", "HTTP 429", "You hit your limit", "rate_limit", "quota exceeded"):
        assert P.QUOTA_RE.search(s), s
    assert not P.QUOTA_RE.search(AGY_OK.replace("quota table, rate limit notes", "done"))
    log = tmp / "log.jsonl"
    # agy
    assert P.AgyCLI(c, FakePopen([AGY_OK])).run("/spec x --brief b.md", "m", log) == "done"
    assert P.AgyCLI(c, FakePopen([AGY_OK], 0)).classify(0, [AGY_OK]) == "done"    # prose with 'quota' is not a hit
    assert P.AgyCLI(c, FakePopen([AGY_QUOTA], 1)).run("/spec x --brief b.md", "m", log) == "paused"
    assert P.AgyCLI(c, FakePopen(["Error: usage limit reached for this model"], 1)).run("p", "m", log) == "paused"
    assert P.AgyCLI(c, FakePopen(["boom"], 1)).run("p", "m", log) == "error"
    # opencode: rc 0 is still 'done' (progressed() decides), the rate-limit error line pauses
    oc = lambda lines, rc=0: P.OpencodeCLI(c, FakePopen(lines, rc))
    wt = tmp / "wt"
    wt.mkdir()
    assert oc([OPENCODE_ERR], 1).run("/spec x --brief b.md", "openrouter/openrouter/free", log, wt) == "paused"
    assert oc(['{"type":"text","text":"all good"}']).run("p", "openrouter/openrouter/free", log, wt) == "done"
    assert oc(['{"type":"text","text":"hit the rate limit docs"}']).run("p", "openrouter/openrouter/free", log, wt) == "done"
    assert oc(["crashed"], 2).run("p", "openrouter/openrouter/free", log, wt) == "error"
    # qoder
    q = P.QoderCLI(c, FakePopen([QODER_LOGIN], 1))
    assert q.run("p", "qwen3.8-flash", log) == "paused" and q.note == "login"
    assert P.QoderCLI(c, FakePopen(["Error: quota exhausted"], 1)).run("p", "m", log) == "paused"
    assert P.QoderCLI(c, FakePopen(['{"result":"ok","is_error":false}'], 0)).run("p", "m", log) == "done"
    # real 2026-10-18: a wrong model id silently falls back to the credit-costly "auto" model
    q2 = P.QoderCLI(c, FakePopen(['Model "qwen3.8-flash" is not available right now; using "auto" instead.',
                                  '{"type":"result","subtype":"success","is_error":false,"result":"PONG"}'], 0))
    assert q2.run("p", "qwen3.8-flash", log) == "error" and q2.note == "model-unavailable"
    # real agy quota error: the reset time becomes the pause length
    a = P.AgyCLI(c, FakePopen(['error: Individual quota reached. Resets in 166h40m34s.',
                               '{"status":"ERROR","response":"","error":"Individual quota reached. Resets in 166h40m34s."}'], 3))
    assert a.run("p", "gemini-3.8-flash-medium", log) == "paused" and a.pause_for == 166 * 3600 + 40 * 60 + 34
    assert P.reset_seconds("Resets in 25m") == 1500 and P.reset_seconds("no hint") is None
    # log: start + raw lines + end with rc, no ANTHROPIC_API_KEY, stdin devnull
    ev = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.startswith('{"type": "provider"')]
    assert any("start" in e and "cmd" in e for e in ev) and any(e.get("rc") == 1 for e in ev)
    import os
    os.environ["ANTHROPIC_API_KEY"] = "sk-test"
    try:
        fp = FakePopen([AGY_OK])
        P.AgyCLI(c, fp).run("p", "m", log)
        env = fp.calls[0][1]["env"]
        assert "ANTHROPIC_API_KEY" not in env and fp.calls[0][1]["stdin"] is not None
    finally:
        del os.environ["ANTHROPIC_API_KEY"]
    print("regex / paused mapping ok")


def test_opencode_json(tmp: Path) -> None:
    c = cfg()
    log = tmp / "oc.jsonl"
    d = tmp / "oc_wt"
    d.mkdir()
    fp = FakePopen(['{"type":"text"}'])
    a = P.OpencodeCLI(c, fp)
    assert a.run("p", "openrouter/openrouter/free", log, d) == "done"
    want = json.loads((RESEARCH / "conductor" / "opencode.json").read_text(encoding="utf-8"))
    assert json.loads((d / "opencode.json").read_text(encoding="utf-8")) == want and want["_conductor"] is True
    assert want["permission"]["bash"]["*"] == "deny"
    (d / "opencode.json").write_text(json.dumps({"_conductor": True, "permission": {"bash": {"*": "allow"}}}))
    assert a.run("p", "openrouter/openrouter/free", log, d) == "done"          # stale managed file is rewritten
    assert json.loads((d / "opencode.json").read_text(encoding="utf-8")) == want
    foreign = tmp / "foreign"
    foreign.mkdir()
    (foreign / "opencode.json").write_text('{"model": "x"}')
    fp2 = FakePopen(['{"type":"text"}'])
    assert P.OpencodeCLI(c, fp2).run("p", "openrouter/openrouter/free", log, foreign) == "error"
    assert not fp2.calls and (foreign / "opencode.json").read_text() == '{"model": "x"}'
    assert "foreign" in log.read_text(encoding="utf-8")
    print("opencode.json ok")


def test_paid_refusal(tmp: Path) -> None:
    c = cfg()
    log = tmp / "paid.jsonl"
    d = tmp / "paid_wt"
    d.mkdir()
    assert P.is_paid_model("openrouter/deepseek/deepseek-v4")
    assert not P.is_paid_model("openrouter/nvidia/nemotron-3-super-120b-a12b:free")
    assert not P.is_paid_model("openrouter/openrouter/free")
    fp = FakePopen(['{"type":"text"}'])
    assert P.OpencodeCLI(c, fp).run("p", "openrouter/deepseek/deepseek-v4", log, d) == "error" and not fp.calls
    assert P.OpencodeCLI(c, fp).run("p", "openrouter/deepseek/deepseek-v4", log, d, paid=True) == "done" and fp.calls
    print("paid refusal ok")


def test_translate() -> None:
    p = "/spec research/c3/x --brief research/campaigns/y/briefs/spec.md"
    t = P.translate(p)
    assert "research/campaigns/y/briefs/spec.md" in t and ".claude/skills/spec/SKILL.md" in t, t
    assert "research/c3/x --brief research/campaigns/y/briefs/spec.md" in t and "STEP-RESULT" in t
    assert ".claude/skills/build/SKILL.md" in P.translate("/build research/c3/x harness --brief b.md")
    assert P.translate("plain text") == "plain text"
    print("translate ok")


class FakeAdapter:
    def __init__(self, results, note=""):
        self.results, self.calls, self.note = list(results), [], note

    def run(self, prompt, model, log, worktree=None, web=False, paid=False):
        self.calls.append(dict(model=model, prompt=prompt, web=web, paid=paid))
        return self.results.pop(0) if self.results else "done"


class Clock:
    t = 1_800_000_000.0

    def __call__(self):
        return self.t


def make_router(tmp: Path, adapters=None, clock=None, **cfg_over):
    c = cfg()
    c["routing"].update(cfg_over)
    ad = adapters if adapters is not None else {p: FakeAdapter([]) for p in ("claude", "agy", "qoder", "opencode")}
    d = Path(tempfile.mkdtemp(dir=tmp))
    return P.Router(c, d / "ledger.json", clock=clock or Clock(), adapters=ad, lock_dir=d / "locks",
                    claude_lock=d / "locks" / "claude.lock"), ad


def test_router(tmp: Path) -> None:
    log = tmp / "r.jsonl"
    # routes + family + step mapping
    r, ad = make_router(tmp)
    assert r.routes("build-harness") == r.routes("build") == r.routes("build-ship")
    assert r.routes("spec")[0] == "claude:opus" and r.family("claude:opus") == "anthropic"
    assert r.family("agy:gemini-3.8-flash-high") == "google" and r.family("qoder:qwen3.8-flash") == "qwen"
    assert r.family("opencode:other") == "opencode"
    # fallback order on paused: claude pauses -> agy done; ledger records the pause
    ad["claude"] = FakeAdapter(["paused"])
    res, route = r.run("spec", "/spec x --brief b", log)
    assert (res, route) == ("done", "agy:claude-opus-4-6-thinking"), (res, route)
    assert r.ledger()["paused_until"]["claude:opus"] > Clock.t
    assert r.pick("spec") == "agy:claude-opus-4-6-thinking"             # still paused: claude skipped without a call
    n = len(ad["claude"].calls)
    r.run("spec", "/spec x --brief b", log)
    assert len(ad["claude"].calls) == n
    Clock.t += 1801
    assert r.pick("spec") == "claude:opus"                              # pause expired
    # error and done return immediately (no fallthrough)
    ad["claude"] = FakeAdapter(["error"])
    assert r.run("spec", "/spec x", log) == ("error", "claude:opus") and len(ad["agy"].calls) == 2
    # web flag only for web steps
    r2, ad2 = make_router(tmp)
    r2.run("scout", "/scout x", log)
    r2.run("spec", "/spec x", log)
    assert ad2["qoder"].calls[0]["web"] is True and ad2["claude"].calls[0]["web"] is False
    # cap exhaustion skips claude:opus
    r3, ad3 = make_router(tmp, caps_day={"claude:opus": 2})
    got = [r3.run("spec", "/spec x", log)[1] for _ in range(3)]
    assert got == ["claude:opus", "claude:opus", "agy:claude-opus-4-6-thinking"], got
    # cross-family review: exclude anthropic -> first non-Anthropic H route (DeepSeek via Qoder)
    r4, _ = make_router(tmp)
    assert r4.pick("attack", exclude_family="anthropic") == "qoder:DeepSeek-V4-Pro"
    assert r4.pick("attack", exclude_family="deepseek") == "claude:opus"
    assert r4.run("attack", "/attack x", log, exclude_family="anthropic")[1] == "qoder:DeepSeek-V4-Pro"
    # Qoder daily credit budget: paid Qoder models stop, the 0-credit Qwen3.8-Flash keeps running
    rq, adq = make_router(tmp, credits_day={"qoder": 1.0}, credit_free=["Qwen3.8-Flash"])
    rq.yield_claude(3600)
    rq.mark_unhealthy("agy:claude-sonnet-4-6")
    adq["qoder"].credits = 1.5
    assert rq.run("build", "/build x", log)[1] == "qoder:Qwen3.7-Plus"
    assert rq.ledger()[rq._today()]["qoder:Qwen3.7-Plus"]["credits"] == 1.5
    assert "qoder:Qwen3.7-Plus" not in rq.candidates("build") and "qoder:Qwen3.8-Flash" in rq.candidates("build")
    # yield_claude removes every claude route
    r5, _ = make_router(tmp)
    r5.yield_claude(3600)
    assert all(not x.startswith("claude:") for x in r5.candidates("build"))
    assert r5.pick("build") == "agy:claude-sonnet-4-6"
    # unhealthy today
    r5.mark_unhealthy("agy:claude-sonnet-4-6")
    assert r5.pick("build") == "qoder:Qwen3.7-Plus"
    # slot busy -> next provider, no blocking
    r6, ad6 = make_router(tmp)
    held = P.call_claude.Lock(r6.claude_lock)
    assert held.acquire("someone else")
    assert r6.run("spec", "/spec x", log)[1] == "agy:claude-opus-4-6-thinking" and not ad6["claude"].calls
    held.release()
    r6.r["slots"]["agy"] = 1
    a0 = P.call_claude.Lock(r6.lock_dir / "agy.0.lock")
    assert a0.acquire("x")
    assert r6.run("spec", "/spec x", log)[1].startswith("claude:")
    a0.release()
    # all paused -> ("paused", None); paid last resort only for H and under budget
    for step in ("spec", "build"):
        ads = {p: FakeAdapter(["paused"] * 5) for p in ("claude", "agy", "qoder", "opencode")}
        r7, _ = make_router(tmp, adapters=ads)
        assert r7.run(step, f"/{step} x", log) == ("paused", None)
        paid_tried = any(c["paid"] for c in ads["opencode"].calls)
        assert paid_tried == (step == "spec"), (step, ads["opencode"].calls)    # paid only for tier H
    r8, ad8 = make_router(tmp, adapters={"claude": FakeAdapter(["paused"]), "agy": FakeAdapter(["paused"]),
                                         "qoder": FakeAdapter(["paused"] * 5), "opencode": FakeAdapter(["paused", "done"])})
    res, route = r8.run("spec", "/spec x", log)
    assert (res, route) == ("done", "opencode:openrouter/~deepseek/deepseek-pro-latest"), (res, route)
    paid_call = ad8["opencode"].calls[-1]
    assert paid_call["paid"] is True and ad8["opencode"].calls[0]["paid"] is False
    led = r8.ledger()
    assert led[r8._today()]["opencode:openrouter/~deepseek/deepseek-pro-latest"]["usd"] == 0.02
    # budget exhausted -> no paid route
    r9, _ = make_router(tmp, adapters={p: FakeAdapter([]) for p in ("claude", "agy", "qoder", "opencode")})
    r9.r["tiers"]["H"] = []
    assert r9.pick("spec") == "opencode:openrouter/~deepseek/deepseek-pro-latest"
    led = r9.ledger()
    led.setdefault(r9._today(), {})["opencode:openrouter/~deepseek/deepseek-pro-latest"] = {"usd": 0.25}
    r9._save(led)
    assert r9.pick("spec") is None
    r9.r["tiers"]["M"] = []
    assert r9.pick("build") is None                                      # not an H step
    # slot lock released after a run; ledger is valid JSON (atomic write left no tmp)
    assert not list(r.lock_dir.glob("*.lock")) and not list(r.ledger_path.parent.glob("*.tmp"))
    json.loads(r.ledger_path.read_text(encoding="utf-8"))
    print("router ok")


def test_campaign_router(tmp: Path) -> None:
    import campaign
    TAG = "zz-test-providers"
    cdir = RESEARCH / "campaigns" / TAG
    idea = RESEARCH / "exploit" / f"{TAG}-1"
    try:
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "charter.md").write_text("# Charter\n## Stop rules\n- max_cycles: 3\n")
        idea.mkdir(parents=True, exist_ok=True)
        (idea / "STATUS").write_text("proposed\nline: exploit\nbranch: b\n")

        class FakeRouter:
            def __init__(self):
                self.calls = []

            def routes(self, step):
                return ["claude:opus", "agy:x"]

            def family(self, route):
                return "anthropic" if route.startswith("claude") else "google"

            def run(self, step, prompt, log, worktree=None, exclude_family=None, skip=()):
                self.calls.append(dict(step=step, exclude=exclude_family, prompt=prompt))
                verb = prompt.split()[0]
                if verb == "/spec":
                    (idea / "spec.md").write_text("spec")
                    P_write(idea, "specced")
                elif verb == "/attack":
                    P_write(idea, "attacked")
                return "done", "claude:opus"

        def P_write(d, st):
            from _common import write_status
            write_status(d, st)

        fr = FakeRouter()
        c = campaign.Campaign(cdir, sleep=lambda s: None, router=fr)
        c.state.update(current=idea.relative_to(REPO).as_posix())
        c.save()
        assert c.step()                                                # /spec
        st, meta = read_status(idea)
        assert st == "specced" and meta["spec_by"] == "anthropic", (st, meta)
        c.step()                                                       # preflight (code)
        st = read_status(idea)[0]
        if st == "preflight-ok":
            assert c.step()                                            # /attack
            assert fr.calls[-1]["step"] == "attack" and fr.calls[-1]["exclude"] == "anthropic", fr.calls
        else:                                                          # preflight may flag the stub spec: drive attack directly
            from _common import write_status
            write_status(idea, "preflight-ok")
            assert c.step()
            assert fr.calls[-1]["step"] == "attack" and fr.calls[-1]["exclude"] == "anthropic", fr.calls
        assert c.state["calls"]["claude:opus"] >= 2
        assert fr.calls[0]["exclude"] is None
        # a router web step does not need claude.web_settings
        c2 = campaign.Campaign(cdir, sleep=lambda s: None, router=fr)
        assert c2.web_allowed()
        # all routes at their campaign cap -> stop
        c2.caps = {"claude:opus": 1, "agy:x": 1}
        c2.state["calls"] = {"claude:opus": 1, "agy:x": 1}
        assert c2.call("spec", "/spec x", idea) == "stopped" and "call cap" in c2.state["stop"]
        print("campaign router ok")
    finally:
        shutil.rmtree(cdir, ignore_errors=True)
        shutil.rmtree(idea, ignore_errors=True)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="test_providers_"))
    try:
        test_cmds(tmp)
        test_regex_and_paused(tmp)
        test_opencode_json(tmp)
        test_paid_refusal(tmp)
        test_translate()
        test_router(tmp)
        test_campaign_router(tmp)
        print("ALL OK")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
