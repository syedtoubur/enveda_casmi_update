# Handoff — 2026-10-13, session "qoder-structural-grill"
**For:** a fresh session, any model. **State:** campaign `20261013-e006-structural` is chartered and **NOT yet started**
(no `state.json`, no driver running, no idea folders created).

## One line
The portfolio grill is done and the charter is written; the next action is stage 4 (spec cycle 1 = `fusion-tail`) or
`campaign.py research/campaigns/20261013-e006-structural --detach` — **only after the human says go**.

## Verified this session (facts, not hypotheses)
- **LB top is 0.491** (CHU1PC 10-12, MarvinTMB 0.484, pikachu 0.481) with ~14 teams above our 0.440. Verified with
  `kaggle competitions leaderboard -c enveda-CASMI26-molecule-id-mass-spectra --show --page-size 15`. No public notebook
  discloses a mechanism above 0.439. This reframes the loop: ±0.004 knob work cannot close a +0.051 gap.
- `research/runtime.md` **did not exist**; written. Multi-tier mode confirmed: Claude Code 2.1.296 headless works
  (`claude -p --model haiku` returned `RUNTIME_OK`), `agy` 1.1.15 present but its headless ideation is still UNPROBED.
- Loop tests all green: `test_bin.py` 52/52, `test_campaign.py`, `harness/test_suite.py`, `preflight.py --selftest`.
- **`engine2-pv88` is submitted but unscored** (human confirmed). Kaggle shows it as the 2026-10-12 17:55:58 UTC
  submission with an empty score; `poll_lb` cannot match it (W13 still live). Its pre-registered reading bands are in the
  **old** charter's amendment (`research/campaigns/20261012-e006-engine2-pubchem/charter.md:60-64`).
- **W20 is still live**: `poll_lb.py` again tried to flip the retired `20261010-control-0433` `scored -> revert`.
- Competition slug matters: `enveda-casmi26` returns **403** on ListSubmissions; use
  `enveda-CASMI26-molecule-id-mass-spectra`.
- Local `test.parquet` shape (provisional, L04): 400 molecules, 1213 spectra, median 3/mol, max 9, 24.2% dual-polarity.

## Files written this session
- `research/runtime.md` — model/tool/service probe table.
- `research/scout/2026-10-13-kaggle.md` (8 items) and `research/scout/2026-10-13-papers.md` (15 items) — stage 1, tier L.
- `research/ideas/2026-10-13-candidates.md` — stage 2, with **two premises corrected against the code** and the D audit.
- `research/campaigns/20261013-e006-structural/charter.md` — the 5 required sections.
- `research/harness/gate_report.md` — appended the `### 2026-10-13: LB-slot effect-size bar, PRE-REGISTERED` block
  (written **before** any measurement; the file is append-only and nothing older was edited).

## Human decisions (grill 2026-10-13, do not re-litigate)
1. Posture: **structural mechanisms only**; no probe may be a variant of a campaign-1 idea or a recorded dead end.
2. **The loop is a constant.** Do not edit `research/bin/`, `research/harness/`, `research/rules/` or the skills this
   campaign, even to fix a Tier-A bug you can see. §7 fixes are deferred; the human will change the loop later.
3. Portfolio (4 probes): `fusion-tail` (B) → `refspectra-lib` (C, exactly 1 probe, no tuning pass) →
   `formula-window` (E) → `xattn-rerank` (F). `+1.45 ppm centring` (A) = free diagnostic, no slot.
   Per-molecule aggregation (D) = **dead**, evidence recorded in the candidates file.
4. **4 probes, no control replicate** — W19 accepted knowingly; the 0.444 KEEP bar rests on a single unreplicated run.
5. Autonomy: stages 1-8 unattended; the human approves only each LB-slot spend and each submit.
6. G4 firing proof: diff the probe's draft `submission.csv` vs the control's draft on the same smoke molecules,
   **≥1 changed molecule** required, else G4 FAIL (closes W6). Implement it in `gates.md` + submit card, not in loop code.
7. Bar: harness Δ ≥ **+0.008** with grouped-bootstrap (by molecule_id) 90% CI lower bound > 0; heavy ideas need a
   measured ceiling ≥ **+0.016**. `config.yaml:68` stays 0.005 (loop code) — the charter and judge enforce the new bar.

## Next step
1. Ask the human for the LB score of `syedtoubur/casmi-engine2-pv88` when it lands; apply the old charter's reading bands,
   record it in `EXPERIMENTS.md` via `/update-experiments`, and add it to the harness-vs-LB ledger in
   `.scratch/handoffs/loop_audit_evidence.md` (it is calibration pair #6/7 — the bar's magnitude is a guess until these stack up).
2. Start cycle 1: `python research/bin/new_idea.py exploit fusion-tail`, then spec → preflight → attack → build.
   The spec must resolve the **open question in the charter's B row**: does E006's existing overlay top-up
   (`scratch/casmi26-e006.py:1006 safe_extend(_candidates[:25], _candidates[25:] + _original)`) already deliver the
   primary tail? If it does, B is partly a no-op and must be re-scoped before it spends a slot.
3. For C, the data path is the gating risk: an external measured-spectra library must be staged as a Kaggle dataset and
   added to the preset **before** the push (L30/L35 — a silent fallback there cost −0.061 once).

## Open questions
- Is the local `test.parquet` the real visible test set or a train slice (L04)? B's and E's reach numbers depend on it.
- Does anyone hold evidence that the +1.45 ppm offset was measured per-molecule-window rather than globally? A's
  downgrade rests on same-formula isomers having identical mass; a per-window measurement could change that.
- `agy` headless ideation is unprobed; if it works, C/E's literature pass should be handed to it rather than re-done here.
