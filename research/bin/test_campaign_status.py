"""Tests for campaign_status.py (read-only; no network, no model calls).

    python research/bin/test_campaign_status.py      (plain runner below; pytest also works)

The "real repo" tests use this checkout's research/campaigns if present, else $CASMI_REPO, else E:/ENVEDA_casmi.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import campaign_status as cs  # noqa: E402


def real_repo() -> cs.Repo:
    here = Path(__file__).resolve().parents[2]
    for cand in (here, os.environ.get("CASMI_REPO"), "E:/ENVEDA_casmi"):
        if cand and len(cs.Repo(Path(cand)).campaign_ids()) >= 5:
            return cs.Repo(Path(cand))
    raise RuntimeError("no checkout with >= 5 campaigns found")


def test_status_list_real_repo():
    repo = real_repo()
    ids = repo.campaign_ids()
    assert len(ids) >= 5
    out = [cs.two_lines(n, cs.summarise(repo, cid)) for n, cid in enumerate(ids, 1)]
    assert all(o.count("\n") == 1 for o in out)
    assert any("antigravity" in o for o in out)
    ag = cs.summarise(repo, "20261017-antigravity-sota-exploit")
    assert ag["ideas"], "portfolio entries must map to ideas"
    assert any(i["status"] for i in ag["ideas"])


def test_campaign_deep_has_status_and_reasons():
    repo = real_repo()
    c = cs.summarise(repo, "20261017-c3-iso")
    assert c["ideas"]
    assert all(i["status"] for i in c["ideas"])
    assert any(i["reason"] for i in c["ideas"])
    json.dumps(c)  # serialisable


def test_cli_json_and_utf8(capsys=None):
    repo = real_repo()
    rc = cs.main(["--repo", str(repo.root), "--json"])
    assert rc == 0
    assert cs.main(["--repo", str(repo.root), "--campaign", "nope"]) == 2


def test_missing_files_give_nulls():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "research" / "campaigns" / "20260101-x").mkdir(parents=True)
        (root / "research" / "campaigns" / "20260101-x" / "state.json").write_text("{}", encoding="utf-8")
        (root / "research" / "campaigns" / "20260101-y").mkdir(parents=True)
        (root / "research" / "campaigns" / "20260101-y" / "state.json").write_text("not json", encoding="utf-8")
        repo = cs.Repo(root)
        assert repo.campaign_ids() == ["20260101-y", "20260101-x"]
        for cid in repo.campaign_ids():
            c = cs.summarise(repo, cid)
            assert c["charter"]["goal"] is None and c["log_tail"] is None
            assert c["ideas"] == [] and c["last_activity"] is None and c["best_lb"] is None
            assert "none measured" in cs.two_lines(1, c)
        assert cs.gpu_info(repo) is None
        assert cs.main(["--repo", str(root)]) == 0
        assert cs.main(["--repo", str(root), "--campaign", "20260101-x"]) == 0


def test_fixture_idea_inbox_and_harness():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        r = root / "research"
        camp = r / "campaigns" / "20260102-demo"
        camp.mkdir(parents=True)
        (camp / "state.json").write_text(json.dumps(
            {"cycle": 1, "current": "research/l1/20260102-a", "stop": None,
             "history": [{"idea": "research/l1/20260102-a", "status": "needs-human", "reason": "r", "lb": "0.44"}]}),
            encoding="utf-8")
        (camp / "charter.md").write_text("# Charter\n\nGoal: do x.\n\n## Scope\n- Line: l1\n\n## Stop rules\nmax_cycles: 3\n",
                                         encoding="utf-8")
        (camp / "log.md").write_text("- 2026-01-03 a\n- 2026-01-05 b\n", encoding="utf-8")
        idea = r / "l1" / "20260102-a"
        idea.mkdir(parents=True)
        (idea / "STATUS").write_text("needs-human\ncampaign: 20260102-demo\nlb: 0.45\n", encoding="utf-8")
        (idea / "harness_result.json").write_text(json.dumps(
            {"target": "C2", "cohorts": {"C2": {"n": 4, "delta": 0.01, "delta_other": 0.002, "delta_leak_hit": 0.03}}}),
            encoding="utf-8")
        (idea / "questions.md").write_text("# Q\n\n## Pick a knob\n", encoding="utf-8")
        (r / "inbox").mkdir()
        (r / "inbox" / "x.md").write_text(
            "---\ncampaign: 20260102-demo\nidea:\naction: note\nargs:\nreason: hi\nstatus: pending   # c\ncreated: t\n---\n",
            encoding="utf-8")
        c = cs.summarise(cs.Repo(root), "20260102-demo")
        assert c["charter"]["goal"] == "do x." and c["charter"]["line"] == "l1"
        assert c["charter"]["stop_rules"] == "max_cycles: 3"
        assert c["last_activity"] == "2026-01-05"
        assert len(c["ideas"]) == 1 and c["ideas"][0]["status"] == "needs-human"
        assert c["ideas"][0]["harness"]["cohorts"]["C2"]["delta_other"] == 0.002
        assert c["best_lb"]["lb"] == 0.45
        assert c["needs_count"] == 2 and len(c["inbox_pending"]) == 1


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
