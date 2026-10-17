# Spec: gen-rank1-keep (R1, knob `GEN1_KEEP`)

Campaign `20261017-c3-iso`. Gate block: gate_report `### 2026-10-17: c3-iso campaign, PRE-REGISTERED`, bullet R1, plus
the rank-only gate (2026-10-13). Target cohort pseudo_c3.

## Hypothesis
Engine 1 (`generate=True`) sometimes puts a generated structure at BASE #1 (formula tagged `'|gen'`, L17). Under E006
this was 138/600 pseudo_c3 (55 correct) but only 2/245, 1/300, 6/400, 4/200 on C1_npx/C1_xlib/C2/C2_hard (K0
observation). Post-scoring then moved 44 of the 55 correct ones to ranks 2–7. Keeping BASE #1 at rank 1 when ICEBERG
also prefers it (leak-free guard; f·z is leaky on pseudo_c3) should recover most of them, with almost no C1/C2 fires.

## Knob (`E:/casmi/lines/<idea worktree>/build/casmi_post.py`, default off = byte-identical)
`GEN1_KEEP=False` in CFG. When True, per active molecule m:
- g = BASE[m] #1 as stored in the dump **before** the C8 rerank (`state['base'][m]`: smiles[0], formula[0]).
- Fire iff formula[0] ends with `'|gen'` AND g's ICE cosine (`state['ice'][m]`, same key convention as `ice_rerank`)
  is not None and is ≥ the ICE cosine of every other candidate among BASE top 5 that has one (ties fire) AND at least
  one other top-5 candidate has an ICE score (no comparison → no fire).
- Effect: the final row becomes [g] + [x for x in row if identity(x) != identity(g)], cut at 25 (metric_identity dedupe).
  Applied as the very last step of `run()` (after E2R2 and lib-isomer). Counter `stats['gen1_kept']` (fires) and
  `stats['gen1_moved']` (fires where g was not already rank 1).
- No FPNet value is read by the trigger.

## Measurement
`python research/harness/harness.py scorecard --line c3 --post <idea worktree>/build/casmi_post.py --a "<V22>" --b "<V22>,GEN1_KEEP=true" --target pseudo_c3 --json research/c3/20261017-gen-rank1-keep/harness_result.json`.
Plus `result.md`: fires per cohort, gen1_moved, how many of the correct generated BASE #1 on pseudo_c3 are kept,
rank-1 displacements per cohort, and the leak-hit / other split.
Also write `stress_result.json` rows for the attacks (A1–A4) with died/status/blocking.

## Tests (`build/test_casmi_post.py`, new tests only)
- knob off → output identical to V22 on a toy state; knob on with no `'|gen'` → identical.
- fires only with ICE agreement; dedupe; length ≤ 25; rank 1 = g after firing.
- Non-vacuous: on the real pseudo_c3 dump, gen1 candidates exist (> 50 molecules with `'|gen'` at BASE #1).

STATUS → `harness-done` when harness_result.json and stress_result.json exist (gate.py decides).
