"""autosubmit tests: fakes only, never calls `kaggle competitions submit`.

    python research/bin/test_autosubmit.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _common  # noqa: E402
import autosubmit  # noqa: E402
from _common import read_status  # noqa: E402

TODAY = autosubmit.utc_today()
CFG = {"competition": "comp", "gates": {},
       "autosubmit": {"enabled": True, "daily_submissions": 5, "reserve_for_human": 1, "max_auto_per_day": None},
       "lines": {"exploit": {"control": "research/exploit/ctrl"}}}
GOOD_RESULT = {"schema": "scorecard-1", "target": "C2", "composite": 0.01, "displaced": 0, "n_target": 100,
               "cohorts": {"C2": {"delta": 0.01, "ci90": [0.002, 0.02], "delta_other": 0.005}}}


def mk(root: Path, name: str, state: str, line="exploit", sub="exploit", result=None, g4="PASS", **meta) -> Path:
    d = root / "research" / sub / name
    d.mkdir(parents=True, exist_ok=True)
    body = [state, f"line: {line}"] + [f"{k}: {v}" for k, v in meta.items()]
    (d / "STATUS").write_text("\n".join(body) + "\n", encoding="utf-8")
    if result is not None:
        (d / "harness_result.json").write_text(json.dumps(result), encoding="utf-8")
    if g4:
        (d / "gates.md").write_text(f"| G4 | 2026-10-18 01:00 | {g4} | k; counters: x |  |\n", encoding="utf-8")
    return d


def setup():
    tmp = Path(tempfile.mkdtemp())
    _common.REPO, _common.RESEARCH = tmp, tmp / "research"
    (tmp / "research" / "conductor").mkdir(parents=True)
    autosubmit.conductor_dir = lambda: tmp / "research" / "conductor"
    return tmp


def probe(tmp, name="p1", **kw):
    meta = dict(kernel="u/k-" + name, kernel_version="2")
    meta.update(kw)
    return mk(tmp, name, "ready-to-submit", result=kw.pop("result", GOOD_RESULT), **meta)


def test_off_states():
    tmp = setup()
    calls = []
    sub = lambda cfg, c: calls.append(c) or (True, "")      # noqa: E731
    fetch = lambda cfg: []                                   # noqa: E731
    lines = []
    for cfg in ({**CFG, "autosubmit": {**CFG["autosubmit"], "enabled": False}},
                {**CFG, "autosubmit": {**CFG["autosubmit"], "daily_submissions": None}}):
        r = autosubmit.run_once(cfg, fetcher=fetch, submit=sub, out=lines.append)
        assert r["off"] and lines[-1].startswith("autosubmit: off ("), lines[-1]
    (tmp / "research" / "conductor" / "conductor.yaml").write_text("paused: true\n", encoding="utf-8")
    r = autosubmit.run_once(CFG, fetcher=fetch, submit=sub, out=lines.append)
    assert "paused" in r["off"] and not calls


def test_budget_math():
    tmp = setup()
    subs = [{"date": f"{TODAY} 01:00:00"}] * 3
    assert autosubmit.count_today(subs + [{"date": "2000-01-01 00:00:00"}]) == 3
    autosubmit.append_log({"time": f"{TODAY} 00:30:00", "idea": "x", "ok": True})
    autosubmit.append_log({"time": f"{TODAY} 00:31:00", "idea": "y", "ok": False})
    assert autosubmit.auto_today() == 1
    assert autosubmit.budget(CFG, 3, 1) == 1          # 2 human + 1 auto
    assert autosubmit.budget(CFG, 4, 0) == 0          # 4 used of 5
    assert autosubmit.budget(CFG, 0, 0) == 4
    capped = {**CFG, "autosubmit": {**CFG["autosubmit"], "max_auto_per_day": 2}}
    assert autosubmit.budget(capped, 0, 1) == 1


def ctrl(tmp, lb=None, state="scored"):
    extra = {"lb": lb} if lb else {}
    return mk(tmp, "ctrl", state, kind="control", kernel="u/ctrl", kernel_version="1", **extra)


FIRE_OK = lambda cfg, idea, meta: ""                         # noqa: E731


def test_eligibility():
    tmp = setup()
    ctrl(tmp, lb="0.440")
    good = probe(tmp)
    assert autosubmit.eligibility(CFG, good, read_status(good)[1], FIRE_OK) == []
    cases = {
        "weak": dict(result={"schema": "scorecard-1", "target": "C2", "composite": 0.0,
                             "cohorts": {"C2": {"delta": 0.0, "ci90": [-0.01, 0.01]}}}),
        "no kernel": dict(kernel=""),
        "no version": dict(kernel_version=""),
        "override": dict(override="human"),
        "no G4": dict(g4=None),
        "G4 fail": dict(g4="FAIL (x)"),
    }
    for n, kw in cases.items():
        res = kw.get("result", GOOD_RESULT)
        meta = dict(kernel="u/k", kernel_version="2") | {k: v for k, v in kw.items() if k not in ("result", "g4")}
        d = mk(tmp, "e-" + n.replace(" ", "-"), "ready-to-submit", result=res, g4=kw.get("g4", "PASS"), **meta)
        why = autosubmit.eligibility(CFG, d, read_status(d)[1], FIRE_OK)
        assert why, f"{n} should be ineligible"
    # draft not differing from control
    why = autosubmit.eligibility(CFG, good, read_status(good)[1], lambda c, i, m: "draft identical to control")
    assert any("draft" in w for w in why)
    # control without lb
    tmp2 = setup()
    ctrl(tmp2, lb=None, state="ready-to-submit")
    g2 = probe(tmp2)
    why = autosubmit.eligibility(CFG, g2, read_status(g2)[1], FIRE_OK)
    assert any("no lb" in w for w in why), why
    # kind control/instrument
    inst = mk(tmp2, "inst", "ready-to-submit", kind="instrument", kernel="u/i", kernel_version="1")
    assert autosubmit.eligibility(CFG, inst, read_status(inst)[1], FIRE_OK)


def test_priorities():
    tmp = setup()
    c = ctrl(tmp, lb=None, state="ready-to-submit")
    unp = mk(tmp, "waiting", "scored-unpaired", kernel="u/w", kernel_version="1", lb="0.45")
    rep = mk(tmp, "rep", "keep-pending", kernel="u/r", kernel_version="3", lb="0.46")
    rep_done = mk(tmp, "rep2", "keep-pending", kernel="u/r2", kernel_version="3", lb="0.46", lb_refs="1,2")
    # ctrl has no lb so a queue probe is ineligible; give a second line with a scored control for the queue item
    cfg = {**CFG, "lines": {**CFG["lines"], "l2": {"control": "research/exploit/ctrl2"}}}
    mk(tmp, "ctrl2", "scored", kind="control", kernel="u/c2", kernel_version="1", lb="0.44")
    q = mk(tmp, "queued", "ready-to-submit", line="l2", result=GOOD_RESULT, kernel="u/q", kernel_version="1")
    cands, skipped = autosubmit.candidates(cfg, [c, unp, rep, rep_done, q], FIRE_OK)
    assert [x["kind"] for x in cands] == ["a", "b", "c"], [(x["kind"], x["idea"]) for x in cands]
    assert cands[0]["idea"].endswith("rep") and cands[1]["idea"].endswith("ctrl") and cands[2]["idea"].endswith("queued")
    assert any("rep2" in s["idea"] for s in skipped)


def test_success_and_failure():
    tmp = setup()
    ctrl(tmp, lb="0.440")
    good = probe(tmp, "p-ok")
    bad = probe(tmp, "p-bad")
    submitted, claimed = [], []

    def sub(cfg, c):
        submitted.append(c["idea"])
        return (False, "boom") if c["name"] == "p-bad" else (True, "ok")
    fetch = lambda cfg: [{"date": f"{TODAY} 00:00:00"}]      # noqa: E731
    lines = []
    r = autosubmit.run_once(CFG, fetcher=fetch, submit=sub, claimer=lambda rel: claimed.append(rel) or 0,
                            idea_dirs=[good, bad], firing=FIRE_OK, out=lines.append)
    assert sorted(submitted) == sorted(r["candidates"] and submitted) and len(submitted) == 2
    assert claimed == ["research/exploit/p-ok"], claimed
    log = autosubmit.read_log()
    assert [x["ok"] for x in log].count(True) == 1 and any(not x["ok"] for x in log)
    ok_row = next(x for x in log if x["ok"])
    assert ok_row["idea"] == "research/exploit/p-ok" and ok_row["kernel"] == "u/k-p-ok" and ok_row["kind"] == "c"
    assert read_status(bad)[1]["autosubmit_error"] == "boom"
    # budget: 1 existing today, 1 auto logged now -> budget respected on next pass
    submitted.clear()
    r2 = autosubmit.run_once(CFG, fetcher=fetch, submit=sub, claimer=lambda rel: 0, idea_dirs=[good], firing=FIRE_OK,
                             out=lines.append)
    assert r2["budget"] >= 0


def test_budget_blocks_and_dry_run():
    tmp = setup()
    ctrl(tmp, lb="0.440")
    good = probe(tmp)
    calls = []
    fetch = lambda cfg: [{"date": f"{TODAY} 00:00:00"}] * 4      # noqa: E731
    r = autosubmit.run_once(CFG, fetcher=fetch, submit=lambda c, x: calls.append(x) or (True, ""),
                            idea_dirs=[good], firing=FIRE_OK, out=lambda s: None)
    assert not calls and r["submitted"] == []
    fetch1 = lambda cfg: []                                      # noqa: E731
    r = autosubmit.run_once(CFG, dry_run=True, fetcher=fetch1, submit=lambda c, x: calls.append(x) or (True, ""),
                            idea_dirs=[good], firing=FIRE_OK, out=lambda s: None)
    assert not calls and len(r["candidates"]) == 1


def test_final_pick_note():
    tmp = setup()
    a = mk(tmp, "a", "keep", kernel="u/a", kernel_version="1", lb="0.46", lb_rep="0.45")
    b = mk(tmp, "b", "keep", sub="c3", line="c3", kernel="u/b", kernel_version="1", lb="0.44")
    c = mk(tmp, "c", "revert", kernel="u/c", kernel_version="1", lb="0.50")
    note = autosubmit.final_pick_note([a, b, c])
    assert "`a`" in note and "`b`" in note and "0.4550" in note and "`c`" not in note
    assert "no idea is in `keep`" in autosubmit.final_pick_note([c])


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
