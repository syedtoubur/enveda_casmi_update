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
        goal="Pick the next idea of this campaign (or stop) from the portfolio and the idea pool, the loss map and every "
             "finished cycle's result (state.json history carries each idea's current STATUS and LB).",
        tags=["spike", "measurement", "decision"],
        reads=["{campaign}/charter.md", "{campaign}/state.json", "{campaign}/log.md", "research/harness/loss_map.md",
               "research/ideas/ (newest candidates file, if any)",
               "research/harness/gate_report.md (Results section only)", "result files named in log.md"],
        output="Either `research/bin/new_idea.py {line} <slug>` + hypothesis.md (claim, branch, target cohort, mechanism "
               "with code cites, prior with source: MEASURED or ASSUMED, `novelty: new|variant`, source) and "
               "state.json `current` = that folder; or, when neither the portfolio nor the newest candidates file has a "
               "live idea left, state.json `ideate` = <why> (the driver runs scout/diagnose/ideate and calls you again); "
               "or state.json `stop` = <reason> only for a scope problem a human must settle. Append one line to log.md."),
    "spec": dict(
        goal="Write spec.md: one change behind a default-off switch ({code_form}), the scorecard experiment, counters.",
        tags=["spec", "premise", "measurement"],
        reads=["{idea}/hypothesis.md", "{idea}/preflight.md (if present)", "{post}",
               "research/{line}/BASE.md (only the section you cite)", "{base_export} (the base export; only cited lines)",
               "research/harness/backtests.yaml (knob families)"],
        output="spec.md from research/_template/spec.md with the CASMI section filled: switch name and default, target "
               "cohort (one of {cohorts}), the scorecard command (`python research/harness/harness.py scorecard --a "
               "\"{base_cfg}\" --b \"{base_cfg},KNOB=value\" --target <cohort> --json {idea}/harness_result.json`), "
               "counters (value-checked), the expected number of changed draft molecules (`g4_min_changed`), heavy-path "
               "flag (+ how its ceiling is measured), the knob's family. The gate is rank-only (research/bin/gate.py): "
               "no LB-slot bar to invent. STATUS specced."),
    "attack": dict(
        goal="Try to break the idea before any code is written: 3-6 concrete attacks, each with a pre-registered death condition.",
        tags=["spec", "premise", "measurement", "spike", "g4", "decision"],
        reads=["{idea}/spec.md", "{idea}/hypothesis.md", "{post}", "research/harness/gate_report.md (Scope section)"],
        output="attack.md: one row per attack = what could make the claim false (empty lists, GLACIER down, adduct mix, "
               "formula-group collisions, leak proxy flip, downside cohort, runtime) | how stress_test.py checks it | "
               "'idea dies if ...' (numeric) | blocking yes/no (a blocking attack left UNTESTED kills the idea at the "
               "gate). STATUS attacked."),
    "build-harness": dict(
        goal="Implement the change ({code_form}, default off), write and run stress_test.py from attack.md, run the scorecard.",
        tags=["spike", "measurement"],
        reads=["{idea}/spec.md", "{idea}/attack.md", "{post}", "{worktree}/build/test_patch_harness_dump.py"],
        output="Committed change + test on a per-idea branch in the worktree; {idea}/stress_test.py, stress_result.md and "
               "stress_result.json (one row per attack: id, died true|false|null, status, blocking, and `deviation` "
               "{{original, proxy, accepted_by}} whenever a check other than the pre-registered one ran); "
               "{idea}/harness_result.json from `harness.py scorecard --json` (`--a` = \"{base_cfg}\"). heavy-path: yes -> "
               "stress_test.py checks the inputs offline and measures a ceiling; harness_result.json = "
               "{{\"heavy_path\": true, \"ceiling\": <measured>, \"family\": <family>}}. STATUS harness-done, or dead + reason."),
    "build-ship": dict(
        goal="Ship the measured change: probe notebook, G1/G2, push (no --wait).",
        tags=["g4", "decision", "submit"],
        reads=["{idea}/spec.md", "{idea}/harness_result.json", "{worktree}/build/patch_harness_dump.py"],
        output="Probe notebook in the worktree, gates.md rows G1 and G2, card.md naming the paired control kernel, "
               "STATUS `g4_min_changed:` from the spec, kaggle_push.py with value-checked counters (G4 also diffs the "
               "draft against the control's draft). STATUS g4-pushed."),
    # ---- ideation engine (grill 2026-10-13): scout (L, web) -> diagnose (M) -> ideate (H); fetch answers requests (L, web)
    "scout": dict(
        goal="Literature and code scout: find methods from MS/MS structure-identification work that could transfer to "
             "this pipeline's weakest buckets.",
        tags=[],
        reads=["research/harness/loss_map.md", "research/scout/ (the previous scout file only, to avoid repeats)",
               "the web (WebSearch / WebFetch): papers, preprints, repos, competition write-ups"],
        output="research/scout/{date}.md: >= 10 sources not in the previous scout file, each with: title, link, year, "
               "the method in 2 lines, 'plugs in at:' (which stage of our pipeline / which loss-map bucket), "
               "licence + public-weights status if it ships a model, and a confidence tag (read / skimmed / abstract "
               "only). Facts only; no ranking."),
    "diagnose": dict(
        goal="Loss diagnosis: turn where the truth is lost (loss map, scorecard misses, calibration) into testable "
             "hypotheses.",
        tags=["measurement"],
        reads=["research/harness/loss_map.md", "research/harness/calibration.md", "research/harness/capability_map.md",
               "{campaign}/state.json (history) and the harness_result.json files it names",
               "research/harness/dumps/ through `python research/harness/harness.py lossmap` (re-run it if stale)"],
        output="research/ideas/{date}-diagnosis.md: per loss bucket (pool miss / in a channel list but not top 25 / "
               "rank 2-5 / rank 6-25): molecules, headroom, which channel holds the truth, and 1-3 hypotheses each with "
               "a falsifying measurement you can name. Every number cites the command that produced it."),
    "ideate": dict(
        goal="Invent the next ideas: combine the scout's methods, the diagnosis, past near-misses and mutation "
             "operators into ranked, falsifiable candidates; at least two must be genuinely new mechanisms.",
        tags=["spec", "premise", "decision"],
        reads=["research/ideas/ (newest diagnosis + candidates files)", "research/scout/ (newest file)",
               "research/harness/calibration.md", "{campaign}/charter.md", "{campaign}/state.json",
               "research/LESSONS.md (lesson IDs + one-liners only)", "research/requests/ (answered requests)",
               "EXPERIMENTS.md (dead-ends section only)"],
        output="research/ideas/{date}-candidates.md: >= 5 candidates, each: slug | line (exploit/explore) | target "
               "cohort + reach | mechanism with a code cite | source (scout link, diagnosis row, or mutation of "
               "<idea>: combine / transfer-to-slice / invert / unbundle) | novelty: new|variant | prior "
               "MEASURED/ASSUMED with source | falsifier (the first measurement that would kill it) | cost (knob / "
               "pipeline / GPU training <= 4 h Kaggle / Colab request) | EV. For a training idea, first file a "
               "model-search request in research/requests/. Rank by EV; drop anything a LESSON or dead end already "
               "kills (cite it). Then set {campaign}/state.json `ideate` to null."),
    "report": dict(
        goal="Learn from the cycle that just closed: draft lesson and rule changes (add, modify, retire) for the human.",
        tags=["decision"],
        reads=["{idea}/ (STATUS, gates.md, stress_result.md, harness_result.json, questions.md, card.md)",
               "{campaign}/log.md (this cycle's lines)", "research/LESSONS.md (IDs + one-liners only)",
               "research/rules/ (the rule files; `python research/bin/preflight.py --selftest`)"],
        output="research/proposals/{date}-{idea_name}.md in the promote.py item format (`## Pn: title`, `- action: "
               "add|modify|retire`, `- kind: lesson|rule`, `- target:`, `- decision: pending`, `### Text`, "
               "`### Reasoning`, `### Evidence` with file:line, and for a rule a ```python module + ```file: fixture "
               "blocks that reproduce the mistake). Only surprises: a gate that passed or blocked wrongly, a stall, a "
               "wrong premise, a harness-vs-LB mismatch. Zero items is a valid result: then write no file. Never edit "
               "LESSONS.md or research/rules/ yourself."),
    "fetch": dict(
        goal="Answer the open requests in research/requests/: one specific fact or one model search each, from the web.",
        tags=[],
        reads=["research/requests/*.md with `status: open`", "the web (WebSearch / WebFetch)"],
        output="In each open request file: an `## Answer` section (the fact, or for a model search: candidate models "
               "with link, licence, whether weights are public, training data and its overlap risk with MassSpecGym/"
               "NIST/our val cohorts), every claim with its source link, then `status: answered` (or `status: "
               "not-found` with what you searched). Never edit anything else."),
    "necropsy": dict(
        goal="Explain why one dead idea died (cause code + evidence) and which change or combination could revive it.",
        tags=["measurement"],
        reads=["{idea}/STATUS", "{idea}/necropsy.md (skeleton: `python research/bin/necropsy.py --skeleton {idea}`)",
               "{idea}/spec.md", "{idea}/attack.md", "{idea}/gates.md", "{idea}/harness_result.json",
               "{idea}/result.md or {idea}/card.md", ".claude/skills/necropsy/SKILL.md"],
        output="{idea}/necropsy.md filled per the necropsy skill: confirmed `cause:` (drop the `# guess` mark), evidence "
               "as file:line, what_was_learned, revival_paths, combine_with. Never edit STATUS."),
    "necropsy-synth": dict(
        goal="Turn every necropsy into ranked revival and combination candidates that R12 accepts.",
        tags=["spec", "premise", "decision"],
        reads=["every research/*/*/necropsy.md (`python research/bin/necropsy.py --json`)",
               "research/necropsy/dead_signals.yaml", "research/harness/gate_report.md (headers)",
               "{campaign}/charter.md", "{campaign}/state.json", ".claude/skills/necropsy-synth/SKILL.md"],
        output="research/ideas/{date}-revivals.md: >= 3 ranked candidates, each with `revives:`, `cause:`, `fix:` (a "
               "mechanism, data or stratum change, never a re-tune), `combines:` with the cohorts/strata when it combines, "
               "falsifier, cheapest harness measurement, expected delta vs the 0.004 noise floor, line {line}. Then set "
               "{campaign}/state.json `ideate` to null."),
}


def line_of(campaign: Path, idea: Path | None, cfg: dict) -> str:
    if idea and (idea / "STATUS").exists():
        m = re.search(r"^line:\s*(\w+)", (idea / "STATUS").read_text(encoding="utf-8"), re.M)
        if m and m.group(1) in cfg["lines"]:
            return m.group(1)
    charter = (campaign / "charter.md").read_text(encoding="utf-8") if (campaign / "charter.md").exists() else ""
    m = re.search(r"^\s*-?\s*Line\s*:\s*(\w+)", charter, re.M | re.I)
    return m.group(1).lower() if m and m.group(1).lower() in cfg["lines"] else "exploit"


def line_fmt(cfg: dict, line: str) -> dict:
    """The {placeholders} a brief fills from config `lines.<line>` (loop audit: 22 exploit hard-codes)."""
    lc = cfg["lines"][line]
    wt = lc.get("worktree", "")
    rel = lambda p: p if not p or re.match(r"^[A-Za-z]:|^/", p) else f"{wt}/{p}"
    return dict(line=line, worktree=wt, post=rel(lc.get("post_module", "build/casmi_post.py")),
                base_export=rel(lc.get("base_export", "")), base_cfg=lc.get("harness_base_cfg", ""),
                code_form=lc.get("code_form", "a casmi_post.py knob"),
                cohorts=", ".join((cfg.get("harness") or {}).get("cohorts", {})))


def section(md: str, title: str) -> str:
    m = re.search(rf"^## {re.escape(title)}\s*\n(.*?)(?=^## |\Z)", md, re.S | re.M | re.I)
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
    from _common import load_config
    cfg = load_config()
    import datetime as dt
    fmt = dict(campaign=campaign.as_posix(), idea=idea.as_posix() if idea else "<idea>",
               idea_name=idea.name if idea else "campaign",
               date=dt.date.today().strftime("%Y%m%d"),
               **line_fmt(cfg, line_of(campaign, idea, cfg)))
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
             "(write it to questions.md), never submit. Need one specific fact that is not in the repo (a paper "
             "detail, a model's licence, whether pretrained weights exist)? Write `research/requests/<yyyymmdd>-<slug>.md` "
             "(`status: open`, the question, why it matters) and continue; a web-enabled fetch step answers it before "
             "your next step."]
    text = "\n".join(parts) + "\n"
    out = (idea or campaign) / f"brief_{step}.md"
    out.write_text(text, encoding="utf-8")
    return out, len(text)


if __name__ == "__main__":
    c, st = Path(sys.argv[1]), sys.argv[2]
    path, n = brief(c, st, Path(sys.argv[3]) if len(sys.argv) > 3 else None)
    print(path, f"{n} chars (~{n // 4} tokens)")
