"""Tests for desk.py (no network, no real model).

    python research/bin/test_desk.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import desk as D  # noqa: E402

DAY = "2026-10-18"
CLOCK = lambda: 1792300000.0  # noqa: E731  (fixed instant; today() is derived from it)
TODAY = D.today(CLOCK)
CFG = {"desks": {"library": "x", "triage_per_day": 3, "cards_per_day": 2, "synth_every_days": 7}}

ATOM = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
<link href="http://arxiv.org/api/q"/>
<entry><id>http://arxiv.org/abs/2410.01234v2</id><published>2026-10-10T00:00:00Z</published>
<title>  Spec to
 structure </title><summary> We predict molecules. </summary><arxiv:doi>10.1/ABC</arxiv:doi></entry>
<entry><id>http://arxiv.org/abs/2410.09999v1</id><published>2026-10-11T00:00:00Z</published>
<title>Other</title><summary>Abs</summary></entry>
</feed>"""
OA = {"results": [
    {"id": "https://openalex.org/W1", "doi": "https://doi.org/10.1/abc", "title": "Spec to structure",
     "publication_date": "2026-10-10", "abstract_inverted_index": {"We": [0], "predict": [1], "molecules.": [2]}},
    {"id": "https://openalex.org/W2", "doi": None, "display_name": "No doi", "publication_date": "2026-10-12",
     "abstract_inverted_index": None}]}


class Lib:
    def __enter__(self):
        self.td = Path(tempfile.mkdtemp())
        self.cfg = {"desks": dict(CFG["desks"], library=str(self.td))}
        D.REPO = Path("/")                      # absolute library path wins over REPO join
        self.lib = self.td
        return self

    def __exit__(self, *a):
        shutil.rmtree(self.td, ignore_errors=True)


def put(lib, desk, name, rows):
    D.append_jsonl(lib / desk / name, rows)


def paper(i, **kw):
    return dict({"id": f"arxiv:{i}", "title": f"t{i}", "abstract": "a", "url": f"https://arxiv.org/abs/{i}",
                 "date": "2026-10-10", "query": "q", "doi": None}, **kw)


class FakeRouter:
    def __init__(self, writer=None):
        self.calls = []
        self.writer = writer or (lambda step, prompt, kw: ("done", "claude:x"))

    def family(self, route):
        return route.split(":")[0]

    def run(self, step, prompt, log, worktree=None, exclude_family=None, skip=()):
        self.calls.append((step, prompt, exclude_family))
        return self.writer(step, prompt, {"exclude_family": exclude_family})


def test_due_logic():
    with Lib() as L:
        assert D.due(L.cfg, CLOCK) == ["search:chem", "search:ml"]
        D.write_state(L.lib / "chem" / "state.json", {"last_search": TODAY})
        put(L.lib, "chem", "inbox.jsonl", [paper(1), paper(2)])
        assert D.due(L.cfg, CLOCK) == ["triage:chem", "search:ml"]
        # triage limit reached today
        D.write_state(L.lib / "chem" / "state.json", {"last_search": TODAY, "day": TODAY, "triage_count": 3})
        assert D.due(L.cfg, CLOCK) == ["search:ml"]
        # a stale day resets the counters
        D.write_state(L.lib / "chem" / "state.json", {"last_search": TODAY, "day": "2000-01-01", "triage_count": 99})
        assert "triage:chem" in D.due(L.cfg, CLOCK)
        # kept but uncarded -> card, until the card limit
        put(L.lib, "chem", "triage.jsonl", [{"id": "arxiv:1", "keep": True, "why": "w"},
                                            {"id": "arxiv:2", "keep": False, "why": "w"}])
        D.write_state(L.lib / "chem" / "state.json", {"last_search": TODAY})
        assert D.due(L.cfg, CLOCK) == ["card:chem", "search:ml"]
        D.write_state(L.lib / "chem" / "state.json", {"last_search": TODAY, "day": TODAY, "card_count": 2})
        assert D.due(L.cfg, CLOCK) == ["search:ml"]


def test_search_parsing_and_dedupe():
    rows = D.parse_arxiv(ATOM, "q")
    assert [r["id"] for r in rows] == ["arxiv:2410.01234", "arxiv:2410.09999"]
    assert rows[0]["title"] == "Spec to structure" and rows[0]["doi"] == "10.1/abc" and rows[0]["date"] == "2026-10-10"
    oa = D.parse_openalex(OA, "q")
    assert oa[0]["id"] == "doi:10.1/abc" and oa[0]["abstract"] == "We predict molecules."
    assert oa[1]["id"] == "openalex:W2" and oa[1]["abstract"] == ""
    # the same paper via arXiv and OpenAlex collapses on DOI; cross-desk keys are respected
    new = D.dedupe(rows + oa, set())
    assert [r["id"] for r in new] == ["arxiv:2410.01234", "arxiv:2410.09999", "openalex:W2"]
    assert D.dedupe(rows, {"doi:10.1/abc"}) == [rows[1]]


def test_run_search_offline_and_error():
    with Lib() as L:
        (L.lib).mkdir(exist_ok=True)
        (L.lib / "queries.yaml").write_text("chem:\n  arxiv_categories: [physics.chem-ph]\n  queries: [one]\n", "utf-8")
        calls = []

        def fetch(url):
            calls.append(url)
            return ATOM.encode() if "arxiv" in url else json.dumps(OA).encode()
        n = D.run_search(L.cfg, "chem", fetch=fetch, sleep=lambda s: None, clock=CLOCK, log=lambda m: None)
        assert n == 3 and len(calls) == 2
        assert D.read_state(L.lib / "chem" / "state.json")["last_search"] == TODAY
        # a second desk run finds only duplicates
        (L.lib / "queries.yaml").write_text("chem:\n  queries: [one]\nml:\n  queries: [one]\n", "utf-8")
        assert D.run_search(L.cfg, "ml", fetch=fetch, sleep=lambda s: None, clock=CLOCK, log=lambda m: None) == 0

    with Lib() as L:
        (L.lib / "queries.yaml").write_text("chem:\n  queries: [one]\n", "utf-8")

        def boom(url):
            raise OSError("down")
        assert D.run_search(L.cfg, "chem", fetch=boom, sleep=lambda s: None, clock=CLOCK, log=lambda m: None) == 0
        assert "last_search" not in D.read_state(L.lib / "chem" / "state.json")   # retried next time


def test_parse_triage():
    ids = {"a", "b"}
    good = '{"id":"a","keep":true,"why":"x"}\n{"id":"b","keep":false,"why":"y"}\n'
    assert len(D.parse_triage(good, ids)) == 2
    assert D.parse_triage('{"id":"a","keep":true,"why":"x"}', ids) is None          # missing b
    assert D.parse_triage('{"id":"a","keep":"yes","why":"x"}\n{"id":"b","keep":false,"why":"y"}', ids) is None
    assert D.parse_triage("not json", ids) is None


def _triage_writer(good_after):
    state = {"n": 0}

    def w(step, prompt, kw):
        state["n"] += 1
        out = Path(prompt.split("\n", 1)[0].split()[-1])
        ids = [json.loads(l)["id"] for l in prompt.split("<papers>\n")[1].split("\n</papers>")[0].splitlines()]
        if state["n"] >= good_after:
            out.write_text("\n".join(json.dumps({"id": i, "keep": i.endswith("1"), "why": "w"}) for i in ids), "utf-8")
        else:
            out.write_text("garbage", "utf-8")
        return "done", "qwen:flash"
    return w


def test_triage_retry_then_ok_and_error():
    with Lib() as L:
        put(L.lib, "ml", "inbox.jsonl", [paper(1), paper(2)])
        r = FakeRouter(_triage_writer(2))
        assert D.run_triage(L.cfg, "ml", r, CLOCK, lambda m: None) == "done"
        assert len(r.calls) == 2 and all(c[0] == "scout" for c in r.calls)
        tri = D.read_jsonl(L.lib / "ml" / "triage.jsonl")
        assert {t["id"]: t["keep"] for t in tri} == {"arxiv:1": True, "arxiv:2": False}
        assert tri[0]["route"] == "qwen:flash"
        assert D.read_state(L.lib / "ml" / "state.json")["triage_count"] == 2
    with Lib() as L:
        put(L.lib, "ml", "inbox.jsonl", [paper(1)])
        r = FakeRouter(_triage_writer(99))
        assert D.run_triage(L.cfg, "ml", r, CLOCK, lambda m: None) == "triage_error"
        assert len(r.calls) == 2
        assert D.untriaged(L.lib, "ml") == []                                      # not retried forever
        assert D.read_jsonl(L.lib / "ml" / "triage_errors.jsonl")[0]["error"] == "triage_error"


def test_card_spotcheck_excludes_writer_family():
    with Lib() as L:
        put(L.lib, "chem", "inbox.jsonl", [paper(1)])
        put(L.lib, "chem", "triage.jsonl", [{"id": "arxiv:1", "keep": True, "why": "w"}])

        def writer(step, prompt, kw):
            if step == "scout":
                p = Path(prompt.split("\n")[0].split()[3])
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("---\nid: arxiv:1\nstatus: draft\n---\n## Claim\nx\n", "utf-8")
                return "done", "agy:gemini"
            p = Path(prompt.split("\n")[0].split()[2])
            p.write_text(p.read_text("utf-8") + "\n## Spot check (claude)\nverdict: wrong\nbad\n", "utf-8")
            return "done", "claude:haiku"
        r = FakeRouter(writer)
        assert D.run_card(L.cfg, "chem", r, CLOCK, lambda m: None) == "done"
        assert [c[0] for c in r.calls] == ["scout", "fetch"]
        assert r.calls[0][2] is None and r.calls[1][2] == "agy"
        card = D.card_path(L.lib, "chem", "arxiv:1")
        assert D.card_status(card) == "rejected"
        assert D.read_state(L.lib / "chem" / "state.json")["card_count"] == 1
        assert D.uncarded(L.lib, "chem") == []


def test_card_ok_accepts_and_failed_card_not_retried():
    with Lib() as L:
        put(L.lib, "ml", "inbox.jsonl", [paper(1), paper(2)])
        put(L.lib, "ml", "triage.jsonl", [{"id": "arxiv:1", "keep": True, "why": "w"},
                                          {"id": "arxiv:2", "keep": True, "why": "w"}])
        r = FakeRouter(lambda s, p, k: ("error", "agy:g"))
        assert D.run_card(L.cfg, "ml", r, CLOCK, lambda m: None) == "card_error"
        assert [x["id"] for x in D.uncarded(L.lib, "ml")] == ["arxiv:2"]


def test_synth_due_after_five_new_accepted_cards():
    with Lib() as L:
        for i in range(4):
            c = D.card_path(L.lib, "ml", f"arxiv:{i}")
            c.parent.mkdir(parents=True, exist_ok=True)
            c.write_text("---\nstatus: accepted\n---\n", "utf-8")
        D.write_state(L.lib / "chem" / "state.json", {"last_search": TODAY})
        D.write_state(L.lib / "ml" / "state.json", {"last_search": TODAY})
        assert "synth" not in D.due(L.cfg, CLOCK)
        draft = D.card_path(L.lib, "ml", "arxiv:d")
        draft.write_text("---\nstatus: draft\n---\n", "utf-8")
        assert "synth" not in D.due(L.cfg, CLOCK)                                  # drafts do not count
        c = D.card_path(L.lib, "chem", "arxiv:9")
        c.parent.mkdir(parents=True, exist_ok=True)
        c.write_text("---\nstatus: accepted\n---\n", "utf-8")
        assert D.due(L.cfg, CLOCK) == ["synth"]
        # synced recently: not due even with 5 cards
        D.write_state(L.lib / "state.json", {"last_synth": TODAY, "synth_cards": []})
        assert D.due(L.cfg, CLOCK) == []
        # old synth that already covered those cards: not due either
        D.write_state(L.lib / "state.json", {"last_synth": "2000-01-01", "synth_cards": [
            f"{p.parent.parent.name}/{p.name}" for p in D.accepted_cards(L.lib)]})
        assert D.due(L.cfg, CLOCK) == []


def test_run_synth_marks_cards_seen():
    with Lib() as L:
        for i in range(5):
            c = D.card_path(L.lib, "ml", f"arxiv:{i}")
            c.parent.mkdir(parents=True, exist_ok=True)
            c.write_text("---\nstatus: accepted\n---\n", "utf-8")
        outs = []

        def writer(step, prompt, kw):
            assert step == "ideate"
            out = D.REPO / prompt.split("\n")[0].split()[1]
            outs.append(out)
            return "done", "claude:opus"
        D.REPO = L.td                                                              # ideas/ goes under the temp repo
        (L.td / "research" / "ideas").mkdir(parents=True)
        r = FakeRouter(lambda s, p, k: (((L.td / p.split("\n")[0].split()[1]).write_text("x", "utf-8")), writer(s, p, k))[1])
        assert D.run_synth(L.cfg, r, CLOCK, lambda m: None) == "done"
        assert len(D.read_state(L.lib / "state.json")["synth_cards"]) == 5
        assert D.new_accepted(L.lib) == []


def _run_all() -> int:
    import traceback
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    saved = D.REPO
    for n, f in tests:
        try:
            f()
            print(f"ok   {n}")
        except Exception:
            failed += 1
            print(f"FAIL {n}\n{traceback.format_exc()}")
        finally:
            D.REPO = saved
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
