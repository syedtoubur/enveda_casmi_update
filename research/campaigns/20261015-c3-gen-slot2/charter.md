# Campaign charter: 20261015-c3-gen-slot2

Planned interactively 2026-10-15 (Opus, plan `~/.claude/plans/here-i-am-working-crispy-matsumoto.md`). Human decisions:
gated insertion in the top 4 (slot 2 by default), a union of BOTH generators (Track 2 SELFIES decoder + c3v2 mutator),
run as a campaign on the loop, and GPU approved for the pseudo_c3 dump and Track 2 P4 (G4.0/G4.1).

**Supervised run.** The human steps `campaign.py --once` and decides at three checkpoints: after K1 (generator),
after K2 (detector), and after K3 (replay) before any ship or LB spend.

Why this campaign (what the planning established):
- **Class 3 has never been measured.** There is no pseudo_c3 dump, so every C3 idea flew blind to the LB (rows 22–26, 34
  at or below noise). The instrument comes first.
- **Slot routing was the second failure.** Ranks 21–25 cap the gain at about +0.001; rank 1 cost −0.064 (bio2). A single
  insert at **slot 2 never moves a rank-1 truth**, and a misfire on a truth at rank r ≥ 2 costs only 1/r − 1/(r+1).
- **The Track 2 decoder's hits are almost all top-1** (pseudo_c3 oracle-formula exact@1/10/25 = 7.67/8.00/8.17%), which
  is exactly what a one-slot insert needs. The c3v2 mutator covers ring-bearing analog edits (depth-1 recall 85.7% given
  the anchor); the decoder covers acyclic and unique-scaffold truths, where the mutator misses (70.9% of misses).
- **The old detector used weak signals.** New signal: `gen_ice_gap` (the generated structure's ICEBERG spectrum matches the query better than
  any retrieved candidate's; f·z is leaky on pseudo_c3, see the gate amendment), plus decoder log-prob, on top of lib_max / rel / margin.

## Scope
- Line: exploit
- Base and paired control: **E006 v1** = `syedtoubur/casmi26-e006-window-40-ice-fallback-true` v1, LB 0.440
  (`research/exploit/20261012-control-e006`); control draft `research/exploit/20261013-control-draft`.
  - Worktree: `E:/casmi/lines/exploit` (branch `line/exploit`). Post-scoring knobs go in `build/casmi_post.py`.
- Harness: `--a "METRIC_FUSION=true,ICE_FALLBACK=true,FUSION_WINDOW=40"` over every dumped cohort plus **pseudo_c3**
  (target). C1/C2 judged by leak-free "other" Δ (L40).
- Supersedes `20261014-e180-npool-mutator` (stopped 2026-10-15). Its pseudo-c3-dump idea folder carries over.
- Order (each gated on the previous):
  1. **pseudo-c3-dump** (instrument; K0). `build/patch_harness_dump.py --shard pseudo_c3`: queries = the cohort's
     train rows; library ban plain IK14 ∪ score_key (645 keys); pool ban in Engine 1, Engine 2, the PubChem window
     and the popularity union; truths into the harness truth source. Kaggle GPU run approved.
  2. **c3-gen-union** (K1). Track 2 P4 on Colab (resume from the Drive checkpoint), predicted formulas; c3v2 mutator on
     non-oracle anchors; union + one leak-free re-scorer (decoder log-p + ICEBERG cosine; NO FPNet, gate amendment 2026-10-15); writes `novel[mid]` and `gen_fz_gap` into a sidecar the replay reads.
  3. **c3-detector** (K2). Break-even operating point, scaffold-grouped CV.
  4. **c3-slot2-route** (K3 → K4 → LB). Knob `NOVEL_SLOT` (default off), stage after `_metric_overlay`/e180/
     `_lib_isomer`; rule + fixture: never edits rank 1, ≤ `NOVEL_N` inserts above rank 20.
- Any new GPU/Colab job beyond the two approved, and any new Kaggle dataset (decoder weights, DreaMS embedder, selfies
  wheel), is approved by the human at a checkpoint.
- The loop code (`research/bin`, `research/harness`, `research/rules`, skills) is a constant, EXCEPT the instrument
  changes this charter names (harness truth source for pseudo_c3; the new rule + fixture). Other changes go to
  `questions.md`.

## Portfolio
| slug | branch | target cohort | claim | switch | heavy-path | prior | novelty | EV |
|---|---|---|---|---|---|---|---|---|
| pseudo-c3-dump | instrument | — | A pool-banned pseudo_c3 dump gives the scorecard its C3 column, K0 (BASE C3 MRR) and the rank distribution that sets L | `harness.cohorts.pseudo_c3` | — | spike SIGNAL (`research/exploit/20261014-pseudo-c3-dump/spike_result.md`) | infra | priority 1 |
| c3-gen-union | class3-gen | pseudo_c3 | Decoder ∪ mutator, re-scored on one scale, puts the truth at top-1 for h ≥ 3% of pseudo_c3 with predicted formulas | sidecar `novel.pkl` per dump | partly (generation offline; replay reads the sidecar) | MEASURED oracle: decoder exact@1 7.67%, mutator depth-1 85.7% given anchor; predicted-formula h UNMEASURED | new (union + common re-scorer) | priority 2 |
| c3-detector | class3-route | pseudo_c3 vs C1/C2 other | Relative signals + `gen_ice_gap` reach break-even precision p·h/2 > (1−p)·L | `NOVEL_TAU` | no | c3cal CC.3/CC.4 FAIL with weaker signals | variant + new signal | priority 3 |
| c3-slot2-route | class3-route | pseudo_c3 | One insert at slot 2 on fired molecules lifts C3 without touching rank 1 | `NOVEL_SLOT`, `NOVEL_N` | no (replay) | ASSUMED +0.004 to +0.012 at a 25% C3 share | new routing | priority 4 |

When no live idea is left, `/replan` stops (no ideation in this campaign: it is a fixed chain).

## Bars
- Gate: the rank-only gate (`research/harness/gate_report.md`, block `### 2026-10-13: rank-only gate`), plus the block
  `### 2026-10-15: c3-gen-slot2 campaign, PRE-REGISTERED` (K0–K4 and the LB rule), written before any measurement.
- KEEP at submit (`poll_lb.decide`): LB ≥ **0.444** and ≥ paired E006 replicate control + 0.004, counters `novel_fired`
  printed, draft submission.csv differs from the control draft only where the knob fired (L38); replicate before
  it becomes the base.

## Stop rules
max_cycles: 5
max_lb_probes: 3
dead_streak: 1
quota_pause_h: 12

## Out of scope
- Generated candidates at rank 1; absolute-score gates (L14); λ retuning (L08).
- Tail-only (21–25) as the main lever (ceiling ≈ +0.001); it may ride along with the slot-2 insert.
- Retraining FPNet or the rankers; any FPNet swap without its rows (L01, L02).
- Re-proposing the closed dossier ideas (B5, B1, B6, A3, A6) or any 2026-10-14 campaign idea.
- Edits to `CLAUDE.md`, README, skills, config or `.claude/settings*`.
- Any `kaggle competitions submit`: the human submits.
