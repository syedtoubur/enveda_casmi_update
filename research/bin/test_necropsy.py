"""Tests for research/bin/necropsy.py and rule R12 (no network, no model calls).

    python research/bin/test_necropsy.py      (plain runner below; pytest also works)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import necropsy  # noqa: E402
import preflight  # noqa: E402

# six real `reason` strings from research/campaigns/*/state.json history (cut at a clause) -> expected cause
REAL_REASONS = [
    ("K2 v0 DEAD: C1 fire 0.171 (bar 0.01), C2 r2-5 fire 0.603 (bar 0.02); AUC 0.656", "non-discriminative"),
    ("I1 reach_20 0.073 < 0.10 (reach_all 0.167; A5 pre-screen loses it)", "pool-reach"),
    ("gate: attack A4 died (C2 other +0.0003 < 25% of +0.0038; gain on leak-hit +0.0042)", "leak-only"),
    ("target C2 delta -0.0017 <= 0; harm on C1_npx", "c1c2-harm"),
    ("LB 0.429 vs bar 0.444 (bundled rows + w_pv 0.50)", "unknown"),
    ("DEAD: 0.446 < 0.451 keep bar", "unknown"),
    # synthetic, for the branches no real string exercises
    ("delta +0.001 within noise", "noise-level"),
    ("Traceback in kernel log", "infra"),
]


def test_cause_guess():
    for reason, want in REAL_REASONS:
        assert necropsy.guess_cause(reason) == want, (reason, necropsy.guess_cause(reason), want)


def test_skeleton_not_overwritten():
    with tempfile.TemporaryDirectory() as d:
        idea = Path(d) / "research" / "c3" / "x-idea"
        idea.mkdir(parents=True)
        (idea / "STATUS").write_text("dead\nreason: C1 fire 0.2\n", encoding="utf-8")
        research = Path(d) / "research"
        assert necropsy.skeleton(idea, research) is True
        txt = (idea / "necropsy.md").read_text(encoding="utf-8")
        assert "cause: non-discriminative  # guess, confirm" in txt and "# Necropsy: x-idea" in txt
        (idea / "necropsy.md").write_text("FILLED", encoding="utf-8")
        assert necropsy.skeleton(idea, research) is False
        assert (idea / "necropsy.md").read_text(encoding="utf-8") == "FILLED"
        assert [r["has_necropsy"] for r in necropsy.inventory(research)] == [True]


def test_reason_fallbacks():
    with tempfile.TemporaryDirectory() as d:
        research = Path(d) / "research"
        a = research / "c3" / "a"
        b = research / "c3" / "b"
        for p in (a, b):
            p.mkdir(parents=True)
            (p / "STATUS").write_text("killed-falsifier\n", encoding="utf-8")
        (a / "gates.md").write_text("| g | t | PASS | ok | 0 |\n| g2 | t | FAIL | reach 3% | 0 |\n", encoding="utf-8")
        (research / "campaigns" / "c").mkdir(parents=True)
        (research / "campaigns" / "c" / "state.json").write_text(
            '{"history": [{"idea": "research/c3/b", "reason": "leak only"}]}', encoding="utf-8")
        rows = {Path(r["path"]).name: r for r in necropsy.inventory(research)}
        assert rows["a"]["reason"] == "reach 3%" and rows["a"]["cause_guess"] == "pool-reach"
        assert rows["b"]["reason"] == "leak only"


def test_inventory_finds_dead_ideas():
    rows = necropsy.inventory()
    assert len(rows) >= 10, len(rows)
    assert all({"path", "line", "status", "reason", "has_necropsy"} <= set(r) for r in rows)


def test_r12_fixtures():
    r12 = next(m for m in preflight.load_rules() if m.ID == "R12")
    fx = preflight.RULES / "fixtures"
    for name in ("R12_knob_retune", "R12_dead_signal", "R12_shared_stratum"):
        assert r12.check(fx / name), name
        assert f"rules/fixtures/{name}" in r12.FLAGS
    assert r12.check(fx / "R12_clean_revival") == []
    assert "rules/fixtures/R12_clean_revival" not in r12.FLAGS


def test_r12_ignores_non_revival_and_missing_fix():
    r12 = next(m for m in preflight.load_rules() if m.ID == "R12")
    with tempfile.TemporaryDirectory() as d:
        idea = Path(d)
        (idea / "spec.md").write_text("# spec\nfix: raise the margin threshold\n", encoding="utf-8")
        assert r12.check(idea) == []          # no revives: line, rule does not apply
        (idea / "spec.md").write_text("revives: x\n", encoding="utf-8")
        assert len(r12.check(idea)) == 2      # cause and fix both missing


def test_dead_signal_sources_exist():
    import yaml
    for d in yaml.safe_load((RESEARCH_DIR / "necropsy" / "dead_signals.yaml").read_text(encoding="utf-8")):
        assert (RESEARCH_DIR.parent / d["src"]).exists(), d


RESEARCH_DIR = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print("FAIL", name, repr(e))
    raise SystemExit(1 if failed else 0)
