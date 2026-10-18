# Phase 5 spec: research desks (chemistry + ML paper study)

Plan: `C:/Users/h/.claude/plans/pasted-content-id-c01c-here-what-distributed-clarke.md` (decisions 5, 6, 7, 10).
Builder: Sonnet subagent working DIRECTLY in E:/ENVEDA_casmi (no worktree; other agents write `research/bin/conductor.py`
and `research/bin/autosubmit.py` concurrently; touch only the files listed).

## Goal
Two desks study papers continuously and cheaply, and turn them into pipeline-specific evidence the campaigns can use.
Chemistry desk: MS/MS fragmentation, bond dissociation energies, adduct and in-source chemistry, isomers/tautomers,
natural-product biosynthesis and scaffolds. ML desk: MS→structure models (DreaMS, MIST, ICEBERG, MADGEN, JESTR, CSI:FingerID,
MassSpecGym baselines), contrastive retrieval, fingerprint prediction, generative/de novo decoders (SELFIES), rerankers.

## Already in place
- `research/bin/providers.py` `Router(cfg, ledger)`: `run(step, prompt, log, worktree)`; step tiers come from
  `routing.steps` in config (add nothing to config; use step names `scout` (L, web) for triage and cards, `ideate` (H)
  for synthesis; the Router maps them). Non-Claude providers get the prompt wrapped by `providers.translate()`, which
  points them at `.claude/skills/<step>/SKILL.md` — so for desk jobs pass a PLAIN prompt (not a slash command); check
  `translate()` and if it requires a slash command, give desk prompts as `/desk-<job> ...` and write matching skills below.
- Vendored skills (both `.claude/skills/` and `.agents/skills/`): paper-lookup (arXiv/bioRxiv/OpenAlex/Crossref APIs with
  scripts), literature-review, citation-management, nature-paper-card, nature-reader, deepchem, torch-geometric, torchdrug,
  pytorch-lightning. Read `.claude/skills/VENDOR.md`, `.claude/skills/paper-lookup/SKILL.md`, `.claude/skills/nature-paper-card/SKILL.md`.
- Config block `desks:` (library, triage_per_day, cards_per_day, synth_every_days).
- Existing `.claude/skills/scout/SKILL.md` (current literature scout format: "plugs in at").

## Files
- NEW `research/bin/desk.py`, NEW `research/bin/test_desk.py`
- NEW `research/library/queries.yaml` (per desk: 6–10 search queries + arXiv categories, e.g. chem: `physics.chem-ph`,
  `q-bio.BM`; ml: `cs.LG`, `q-bio.QM`), NEW `research/library/README.md` (layout + card schema)
- NEW `.claude/skills/paper-card/SKILL.md` and identical `.agents/skills/paper-card/SKILL.md`
- NEW `.claude/skills/desk-triage/SKILL.md`, `.claude/skills/desk-synth/SKILL.md` (+ identical `.agents/skills/` copies)
  if translate() needs slash-command skills (see above).
- EDIT `research/bin/run_tests.py`: add test_desk.py.

## Layout
```
research/library/<desk>/inbox.jsonl        # candidate papers found by search (deterministic; id, title, abstract, url, date, query)
research/library/<desk>/triage.jsonl       # one line per screened paper: id, keep (bool), why (one line), model route
research/library/<desk>/cards/<id>.md      # paper cards
research/library/<desk>/state.json         # last search/triage/card/synth dates, daily counters
research/ideas/<date>-desk-candidates.md   # synth output
```

## desk.py
- `python research/bin/desk.py due` → JSON list of due jobs, e.g. `["search:chem", "triage:ml", "card:chem", "synth"]`, based
  on state.json dates and `desks:` limits (search once a day per desk; triage while untriaged inbox items exist and the
  day's triage count < triage_per_day; card while kept-but-uncarded papers exist and count < cards_per_day; synth when
  last synth older than synth_every_days and ≥ 5 new cards since).
- `run search:<desk>`: deterministic, no model. Query arXiv Atom API (`export.arxiv.org/api/query`, sortBy=submittedDate,
  last 30 days on first run then since last search) and OpenAlex (`api.openalex.org/works?search=...&filter=from_publication_date:`)
  using the paper-lookup scripts if they are importable as plain functions (check `scripts/arxiv_atom.py`,
  `openalex_abstract.py`); otherwise minimal `urllib` code. Dedupe by arXiv id / DOI across both desks. Polite: 1 request
  per 3 s, `User-Agent` with a contact placeholder `casmi-desk (mailto:TODO)`. Network errors → log and exit 0.
- `run triage:<desk>`: batch up to 20 untriaged abstracts into ONE router call (step `scout`) whose prompt lists them and asks
  for strict JSON lines `{"id":..,"keep":true|false,"why":".."}` written to a temp file path given in the prompt; desk.py
  validates and appends to triage.jsonl (invalid output → retry once, then mark the batch `triage_error`). Keep rule given
  to the model: keep only if it could change MRR@25 for LC-MS/MS 2D structure ID (CASMI 2026: InChIKey14, Class 1/2/3,
  timsTOF), or gives chemistry that explains fragmentation/adducts/isomers we must rank.
- `run card:<desk>`: one kept paper → router call (step `scout`, web) with `/paper-card <id> <url> <out path>`-style prompt;
  the paper-card skill writes `cards/<id>.md`. Then a cross-family spot check: a second router call (step `attack`-like is
  too expensive — use step `fetch` with `exclude_family=<family of the card writer>`) that reads only the card + abstract
  and appends `## Spot check (<family>)` with `verdict: ok|overclaims|wrong` and one line. Cards with `wrong` get
  `status: rejected` in front matter.
- `run synth`: one H-tier router call (step `ideate`) with `/desk-synth` reading the new accepted cards (list paths in the
  prompt), the newest `research/harness/loss_map.md` if it exists, and `research/necropsy/dead_signals.yaml`; it writes
  `research/ideas/<date>-desk-candidates.md` (≥ 3 ranked candidates: hypothesis, plugs-in-at file+knob, falsifier,
  cheapest harness measurement, expected Δ vs ±0.004 noise, novelty class, leak risk vs val cohorts, licence).
- Every model call logs to `research/library/<desk>/desk_run.jsonl` (router log path).

## paper-card skill (adapt nature-paper-card; keep ≤ 120 lines)
Front matter: `id, title, url, desk, date, status: draft|accepted|rejected, novelty_class: C1|C2|C3|tooling, route`.
Sections: Claim (1–3 lines) · Method (what is computed, inputs/outputs) · Numbers (exact metrics + dataset + split; "not
reported" if absent) · Data & licence (training data, whether it overlaps MassSpecGym/NIST/GNPS val structures → leak
risk high/med/low; code + weights licence; Kaggle-offline feasible?) · Plugs in at (our pipeline: file/stage + knob;
"nowhere" is allowed) · Falsifier (the cheapest harness measurement that would kill it) · Chemistry notes (desk chem
only: mechanism in plain words). Rules: quote numbers only from the paper; mark inference as inference; no invented URLs.

## Tests (no network, no real model)
- `due` logic with fake state/dates and limits.
- search parsing from a saved small arXiv Atom XML string and an OpenAlex JSON string (inline fixtures), dedupe.
- triage JSON validation (good, malformed → retry, then triage_error) with a fake router.
- card spot-check passes `exclude_family` of the writer route's family (fake router records calls).
- synth due only after ≥ 5 new accepted cards.

## Done checklist
- `python research/bin/test_desk.py` green; `python research/bin/test_bin.py` no new failures (67/68 baseline).
- `python research/bin/desk.py due` on the real repo prints a JSON list (paste it). You MAY run
  `python research/bin/desk.py run search:ml` once for real (network to arXiv/OpenAlex only; small) and paste the count;
  do NOT run triage/card/synth for real (they spend model quota).
- Do not commit. Report: files, test tails, deviations.
