# Campaign charter: 20261013-e006-structural

Grill 2026-10-13 (interactive, this session). Portfolio and bars were settled in 4 grill rounds; the evidence and
the two corrected premises are in `research/ideas/2026-10-13-candidates.md` and `research/scout/2026-10-13-{kaggle,papers}.md`.

**Posture: structural mechanisms only.** The public LB top is **0.491** with ~14 teams above our 0.440 (verified via
`kaggle competitions leaderboard -c enveda-CASMI26-molecule-id-mass-spectra --show`), and no public notebook discloses a
mechanism above 0.439. A ±0.004 knob cannot close that. Campaign 1 spent 3/3 LB probes on E006 knob variants and all three
reverted, so **no probe in this campaign is a variant of a campaign-1 idea or of a recorded dead end.**

**The loop is a constant.** The human's explicit decision 2026-10-13: run a new campaign on the loop **exactly as it is**;
`research/bin/*`, `research/harness/*`, `research/rules/*` and the skills are **not edited this campaign**, and §7's Tier-A/B
fixes are deferred. Two manual guardrails cover the loop's known live bugs (they are workarounds, not fixes):
- **W13 guardrail**: every submission this campaign is made **with the idea slug in the Kaggle submission description**,
  so `poll_lb.match` resolves it. The human submits; the slug is written in each `submit_card.md`.
- **W20 guardrail**: `poll_lb.py --write` is run **only with the human present**, and the retired
  `research/exploit/20261010-control-0433` is re-checked after every `--write` (it keeps getting flipped `scored -> revert`).

## Scope
- Line: exploit. Base and paired control: **E006 v1** = `syedtoubur/casmi26-e006-window-40-ice-fallback-true` version 1,
  LB **0.440** (`research/exploit/20261012-control-e006`). Version 2 of that kernel is the smart gate (0.429, REVERT) and is
  never the base. **Not replicated, by the human's decision (2026-10-13): 4 probes, no control replicate** — so every
  KEEP/REVERT is measured against a single-run base carrying ±0.004 of its own noise (W19, accepted and recorded).
- Probe base notebook: `casmi26-e006-window-40-ice-fallback-true.ipynb` in the worktree `E:/casmi/lines/exploit`
  (branch `line/exploit`). Cite base lines from `scratch/casmi26-e006.py`.
- Harness compares use the control's config as `--a` in `KEY=value` form (never JSON):
  `METRIC_FUSION=true,ICE_FALLBACK=true,FUSION_WINDOW=40`; `--b` adds the knob. Report C2 and C1_npx, the 90% CI,
  coupling, and the leak-hit / other split.
- **Autonomy** (§6): stages 1–8 run unattended. The human approves only the portfolio (done) and each LB-slot spend
  (stage 9). Only the human runs `kaggle competitions submit`.
- Cycle order: **B (fusion-tail) → C (measured reference library) → E (formula-window posterior) → F (cross-attention reranker)**.
  A is a free diagnostic, not a cycle.

## Portfolio
| slug | branch | slice | claim | knob | heavy-path | prior | EV |
|---|---|---|---|---|---|---|---|
| fusion-tail | evidence-depth | C2 rank 2–5 (51%, +0.298 headroom, `loss_map.md`) | The **primary** engine's own ranks 26–60 are discarded long before fusion, while Engine 2 contributes deep, so corroboration the primary already computed is thrown away: carry 40 candidates (not 25) into `BASE` and let `fuse2` score them | `BASE[m] = (smis[:25], keys[:25], lib_max)` at `scratch/casmi26-e006.py:578` (with `fuse2` at `:717-727`, `n=40` cut *after* fusion) | no | MEASURED: the asymmetry is in the code (primary capped at 25 via `:578`/`PREFUSION_LISTS :729`, Engine 2 read whole at `:738`). NOT the TOPN 60→100 dead end: that deepened the *scored* window; this restores a *vote* for candidates already scored. L10. Risk: E006's overlay already top-ups tails (`:1006 safe_extend`) | med |
| refspectra-lib | evidence-depth | C1/C2 library channel | The tier-1 library channel knows only the competition's own train spectra (`Library('work')`, `casmi/engine.py:313`). A **measured** external reference library (MassBank ~139k / MoNA, filtered to structures inside our pool) is orthogonal evidence of a type the pipeline has never read | new retrieval channel fused into `fuse2`, or a `lib_max` override — spec must pick one, and stage a Kaggle dataset + preset entry (L30/L35) | **yes** | MEASURED orthogonal wins in this lineage: BIO +0.009, GLACIER +0.007, second engine +0.016 (L10). L19 caution: must be *measured* spectra, not MassSpecGym-trained simulation. Scout: `2026-10-13-kaggle.md` item 1 (dual-band MassBank search, 0.439 public kernel) | high, build cost highest |
| formula-window | candidate-recall | C2 "in no channel list" (3%, +0.030) + any truth whose formula group is wrong | Candidates in the **wrong formula group** can never be rescued by any reranker; a formula posterior that re-weights or re-groups `P.window(target, ppm_win)` raises recall upstream of every lever we have tried | the formula window / `sf_*`,`n_formula`,`formula_share` feature block (`casmi/engine.py:66-80`) | partly | MEASURED: FIDDLE reports formula top-1 88.3% (`2026-10-13-papers.md`). Our features already include `formula_share`/`formula_is_top`, so the ranker sees formula evidence but the *grouping* is hard. Not a dead end (no window-widening: same window, better assignment) | med |
| xattn-rerank | post-scoring | C2 rank 2–5 | A candidate-conditioned **cross-attention** scorer over (spectrum, candidate structure) pairs is a different evidence type than FPNet f·z (predict-then-match), so it is not the L11 "second model of the same task" trap | new rerank stage over the fused 40-window; weights must be an offline wheel + ≤9 h on one T4 (L24) | **yes** | MEASURED in the literature: GLMR Recall@1 67.8% vs JESTR 29.1%; MSAlign = frozen DreaMS + MolDeBERTa, candidate-contrastive, <2 min/epoch (`2026-10-13-papers.md`). Explicitly *not* GLMR's generate-then-rerank path (generated candidates at ranks 1–5 = −0.064 dead end) | unknown — highest build risk |

Diagnostic (no probe, no slot): **+1.45 ppm timsTOF centring** — count the molecules whose formula-window membership
changes at `+1.45 ppm`. Revive as a probe only if > 2% change. `CFG.PPM_CENTER`/`1.45` is absent from the base
(`PPM_WIN=10.0`, `PPM_FALLBACK=30.0` only); same-formula isomers have identical mass, so the shift cannot reorder the
rank 2–5 bucket where 51% of the loss and +0.298 headroom sit.

**Killed before spec, with evidence (recorded 2026-10-13):** *per-molecule cross-spectrum aggregation* — the engine already
does it properly: `analog_query='merged'` merges **within polarity** (`casmi/engine.py:131-135`), library search is
same-polarity per spectrum, `FEATURES` carry `agree_lib`, `agree_ap`, `lib_mean`, `lib_top3`, `n_pos`, `n_neg`, and
`spectra[:cfg.max_query_spectra=16]` never truncates (measured on our local `test.parquet`: 400 molecules, 1213 spectra,
median 3/mol, max 9 — **provisional per L04**, local test.parquet may be a train slice). Narrower remainders worth a later
look: collision-energy weighting (representatives are picked by "most cleaned peaks" per structure/polarity/library at
`:108-118`, not by CE).

## Bars
- KEEP (`poll_lb.decide`): LB ≥ E006 0.440 + 0.004 = **0.444**, the change's counters fired with values > 0 (L05, R05),
  **and** the G4 firing proof below passes. A replicate is required before a final KEEP.
- **G4 firing proof (decided 2026-10-13)**: beyond counters and `g4_required`, diff the probe's draft `submission.csv`
  against the control's draft output over the same smoke molecules and require **≥ 1 changed molecule**. A probe whose
  draft output equals the control's is a no-op and FAILs G4. (Closes W6: promo-soften's counters were print-only.)
  Implemented as a ship-step check in `gates.md` + the submit card, without editing the loop's gate code.
- Harness (pre-registered supersede, written before any measurement): the LB-slot effect-size bar becomes
  **harness Δ ≥ +0.008 on the target slice with a grouped-bootstrap 90% CI lower bound > 0**; heavy ideas additionally
  need a measured ceiling ≥ 0.016. This supersedes `config.yaml:68 lb_slot_delta: 0.005`. It is appended to
  `research/harness/gate_report.md` as a dated PRE-REGISTERED block before cycle 1's first measurement; `config.yaml` is
  loop code and is **not** edited, so the bar is enforced by the charter and the judge, not by the config value.
- **BASE-COUPLED families are uncitable** (unchanged, `capability_map.md` 2026-10-12 11:06): fusion 0.563 and reranker
  0.323 cover `fusion-tail` and `xattn-rerank`; those go to the LB on a written `prior.md` and a measured ceiling, never on
  a harness number. Do not override an /attack death call with a BASE-COUPLED harness number (row 57, promo-soften).

## Stop rules
max_cycles: 4
max_lb_probes: 4
dead_streak: 2
quota_pause_h: 12

## Out of scope
- **Any edit to `research/bin/`, `research/harness/`, `research/rules/` or the skills** — the loop is a constant this
  campaign; Tier-A/B fixes are deferred by the human's decision and must not be smuggled in mid-campaign.
- Knob variants of campaign-1 ideas and of every recorded dead end: FPNet swap or a second FPNet of the same task (L11,
  row 56), fusion/λ retuning (L08, row 17), TOPN widening, popularity as a window selector (L12), promotion
  `REL_PROMOTE`/`SLOTS_*` tweaks (rows 57–58), the Class-1 smart library gate (row 55), absolute-score gates hijacking
  ranks 1–5 (row 25).
- New FPNet training or ranker retraining; any FPNet swap without regenerated `rank_train.npz` (L02).
- Class-3 / de novo generation (L18) and the explore line.
- A control replicate of E006 (declined 2026-10-13).
- `kaggle competitions submit` — the human only.
