"""Write the per-step context brief a headless agent reads INSTEAD of README / LESSONS / EXPERIMENTS.

    python research/bin/brief.py <campaign dir> <step> [<idea dir>]     -> prints the brief path

A brief (<idea>/brief_<step>.md, or <campaign>/brief_replan.md) holds: the step's goal, the campaign scope it
inherits (charter.md `## Scope`, `## Bars`, `## Out of scope`), the rules that apply (by tag), the judgment
lessons that apply (LESSONS lines whose IDs the charter or spec cites), the files the step may open, and the exact
output contract. Target size: <= ~1.5k tokens (6k characters); brief() reports the size so campaign.py can log it.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import preflight  # noqa: E402

STEPS = {
    "replan": dict(
        goal="Pick the next idea of this campaign (or stop) from the portfolio, the loss map and every finished cycle's result.",
        tags=["spike", "measurement", "decision"],
        reads=["{campaign}/charter.md", "{campaign}/state.json", "{campaign}/log.md", "research/harness/loss_map.md",
               "research/harness/capability_map.md (verdict table: rank ideas in BLIND / BASE-COUPLED families by prior, not harness)",
               "research/harness/gate_report.md (Results section only)", "result files named in log.md"],
        output="Either `research/bin/new_idea.py exploit <slug>` + hypothesis.md (claim, branch, slice, mechanism with "
               "code cites, prior with source) and state.json `current` = that folder; or state.json `stop` = <reason>. "
               "Append one line to log.md."),
    "spec": dict(
        goal="Write spec.md: one change as a casmi_post.py knob (default = base), the harness experiment, numeric bars.",
        tags=["spec", "premise", "measurement"],
        reads=["{idea}/hypothesis.md", "{idea}/preflight.md (if present)", "C:/casmi/lines/exploit/build/casmi_post.py",
               "research/exploit/BASE.md (only the section you cite)", "scratch/casmi-notebook-0-433.py (only cited lines)",
               "research/harness/capability_map.md (family table) + research/harness/backtests.yaml (knob families)"],
        output="spec.md from research/_template/spec.md with the CASMI section filled: knob name and default, target "
               "slice (C2 / C1_npx), harness command, LB-slot bar, counters (value-checked), heavy-path flag, the knob's "
               "family + its capability-map verdict (a new knob is added to backtests.yaml families; BLIND or BASE-COUPLED -> "
               "write prior.md, LB decides). STATUS specced."),
    "attack": dict(
        goal="Try to break the idea before any code is written: 3-6 concrete attacks, each with a pre-registered death condition.",
        tags=["spec", "premise", "measurement", "spike", "g4", "decision"],
        reads=["{idea}/spec.md", "{idea}/hypothesis.md", "C:/casmi/lines/exploit/build/casmi_post.py",
               "research/harness/gate_report.md (Scope section)"],
        output="attack.md: one row per attack = what could make the claim false (empty lists, GLACIER down, adduct mix, "
               "formula-group collisions, leak proxy flip, downside cohort, runtime) | how stress_test.py checks it | "
               "'idea dies if ...' (numeric). STATUS attacked."),
    "build-harness": dict(
        goal="Implement the knob in casmi_post.py (default off), write and run stress_test.py from attack.md, run the harness compare.",
        tags=["spike", "measurement"],
        reads=["{idea}/spec.md", "{idea}/attack.md", "C:/casmi/lines/exploit/build/casmi_post.py",
               "C:/casmi/lines/exploit/build/test_patch_harness_dump.py"],
        output="Committed knob + test in the worktree; {idea}/stress_test.py and stress_result.md (one verdict per attack); "
               "{idea}/harness_result.json from `harness.py compare --json`. STATUS harness-done, or dead + reason."),
    "build-ship": dict(
        goal="Ship the measured change: probe notebook that writes casmi_post output, G1/G2, push (no --wait).",
        tags=["g4", "decision", "submit"],
        reads=["{idea}/spec.md", "{idea}/harness_result.json", "C:/casmi/lines/exploit/build/patch_harness_dump.py"],
        output="Probe notebook in the worktree, gates.md rows G1 and G2, submit_card.md naming the paired control "
               "kernel, kaggle_push.py with value-checked counters. STATUS g4-pushed."),
}


def section(md: str, title: str) -> str:
    m = re.search(rf"^## {re.escape(title)}\s*\n(.*?)(?=^## |\Z)", md, re.S | re.M)
    return m.group(1).strip() if m else ""


def lessons(ids: set[str]) -> list[str]:
    text = (RESEARCH / "LESSONS.md").read_text(encoding="utf-8")
    return [ln for ln in text.splitlines() if (m := re.match(r"- \*\*(L\d+)\*\*", ln)) and m.group(1) in ids]


def brief(campaign: Path, step: str, idea: Path | None = None) -> tuple[Path, int]:
    s = STEPS[step]
    charter = (campaign / "charter.md").read_text(encoding="utf-8") if (campaign / "charter.md").exists() else ""
    cited = set(re.findall(r"\bL\d{2}\b", charter))
    for name in ("hypothesis.md", "spec.md"):
        if idea and (idea / name).exists():
            cited |= set(re.findall(r"\bL\d{2}\b", (idea / name).read_text(encoding="utf-8")))
    fmt = dict(campaign=campaign.as_posix(), idea=idea.as_posix() if idea else "<idea>")
    rules = preflight.load_rules(set(s["tags"]))
    parts = [f"# Brief: {step}" + (f" for {idea.name}" if idea else ""), "", f"**Goal**: {s['goal']}", "",
             "## Campaign scope (from charter.md; decisions already made, do not reopen)",
             section(charter, "Scope") or "(no charter scope)", "", "### Bars", section(charter, "Bars") or "defaults in research/harness/gate_report.md",
             "", "### Out of scope", section(charter, "Out of scope") or "-", "",
             "## Rules preflight.py enforces on your output (do not restate them; satisfy them)",
             *[f"- {r.ID}: {r.LESSON}" for r in rules], "",
             "## Judgment lessons cited by this campaign or idea", *(lessons(cited) or ["- none cited"]), "",
             "## You may open only", *[f"- {r.format(**fmt)}" for r in s["reads"]], "",
             "## Output contract", s["output"].format(**fmt), "",
             "Never read README, LESSONS or EXPERIMENTS in full, never read a raw .ipynb, never ask a question "
             "(write it to questions.md), never submit."]
    text = "\n".join(parts) + "\n"
    out = (idea or campaign) / f"brief_{step}.md"
    out.write_text(text, encoding="utf-8")
    return out, len(text)


if __name__ == "__main__":
    c, st = Path(sys.argv[1]), sys.argv[2]
    path, n = brief(c, st, Path(sys.argv[3]) if len(sys.argv) > 3 else None)
    print(path, f"{n} chars (~{n // 4} tokens)")
