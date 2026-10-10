# Track 4 v2: submit-now probes on the 0.411 notebook (no new datasets)

## Context

The best public score is **0.411** (`casmi26-v29-glacier-m-h-full-fusion-0-411.ipynb`). I want to submit improvements now: only the datasets already attached to that notebook, no new training, and no Class-3 generation, which is deferred to a later stage.

This plan:
1. maps the earlier `t4*` plan onto what the leaderboard notebooks actually did;
2. pulls out which kinds of change have raised the score consistently, using `EXPERIMENTS.md` and the 0.409 notebook's own ablation table;
3. turns the strongest untried levers into a probe queue on top of 0.411.

Sources:
- `compare.py` diffs of 0.405 / 0.409 / 0.411.
- `scratch/casmi26-v29-glacier-m-h-full-fusion-0-411.py`. Cells: C7 base list and pool prior at lines 226–274; C9 merge at 403–440; C10 `fuse2` and the final rerank at 450–496.
- `specs/track4_priors_probes/comparision_summary.md`.

## 1. The t4 plan vs the leaderboard notebooks

| Lever | In the t4 plan? | Done by an LB notebook? | Measured LB effect |
|---|---|---|---|
| A3: pool prior μ 0.15 → 0.25 | yes | **yes, 0.409** | **+0.005** (0.404 → 0.409), the cleanest single result |
| B1: popularity inside the PubChem channel | yes ("best single bet") | **yes, 0.411** (V17_PATCH, λ=0.25) | **+0.006** |
| A4: Class-1 freeze / library gate (lib_max ≥ 0.9) | yes | yes, 0.409 (`ICE_LAM_LIB=GL_LAM_LIB=0`) | **0.000** (0.404 → 0.404); safety only |
| A1: prior applied *before* the TOPN cut | yes | **no** | untested |
| A2: prior on the engine-A / fused list | yes | **no.** In 0.411 the engine list (40 from BIO+AFIX) and the `fuse2` output get no prior. | untested |
| B3: richer prior (LOTUS/NPAtlas flags, patents) | yes | **no.** No notebook reads `pool_np`, `pc_np` or `lpatent`. | untested |
| B2: more PubChem slots | yes | **no** | untested |
| Wave C: MS-GPT for Class 3 | yes | no | deferred (user decision) |
| PubChem pass-1 `POP_UNION=200` (union with the most popular) | no (LB discovery) | 0.411 | part of the +0.006 |
| `fz_top` "bouncer vs tie-breaker" gate | no (LB discovery) | 0.411 | part of the +0.006; this was why 0.409's PubChem prior failed (0.393–0.405) |
| MetFrag-lite in the forward-model gap | no (LB discovery) | 0.409 | ≈0 (inside the 0.399 → 0.404 step); applied everywhere it scored **0.390** |
| DreaMS k-NN third list | no | 0.396 notebook | **−0.009** vs its 0.405 base |
| Adduct-aware MetFrag in engine 2 | no | 0.397 notebook | **−0.014** vs 0.411-identical code |

## 2. Consistent signals in the ledger

**Pays every time it has been tried:**

1. **A new orthogonal source of evidence, fused at rank level:**

   | Change | Gain |
   |---|---|
   | ICEBERG | +0.035 |
   | BIO | +0.009 |
   | GLACIER | +0.007 |
   | second engine + RRF | +0.016 |
   | popularity prior (three notebooks) | +0.006 / +0.010 / +0.006 |

2. **Making every scorer see every candidate:**

   | Change | Gain |
   |---|---|
   | ICE_UNION | +0.004 |
   | PubChem pass-1 depth 1000 → 5000 | in +0.004 |
   | `POP_UNION=200` | in +0.006 |
   | prior added to the PubChem list | +0.006 |

3. **Giving an independent signal a fuller voice, up to a point:**

   | Change | Gain |
   |---|---|
   | ICE/GL λ 0.5 → 1.0 | +0.007 |
   | prior μ 0.15 → 0.25 | +0.005 |
   | λ = 2.0 | **−0.034** |

   The ceiling is roughly 1 z-unit per signal.

4. **Purifying a noisy input:** GLACIER limited to `[M+H]+` (part of +0.006), and the `fz_top` gate.

**Flat or negative every time:**
- Swapping a model within the same family: FPNet swap −0.002.
- Deepening without a new ordering signal: TOPN 60 → 100 gave 0.000.
- Tuning on local validation: −0.015.
- Absolute-score gates: −0.064.
- Weak heuristics applied globally: MetFrag everywhere 0.390; DreaMS k-NN −0.009; adduct MetFrag −0.014.
- Protective gates: library gate 0.000.

**What this means for the plan:** the popularity prior is the strongest live signal. It is currently applied to 2 of the 4 lists (pool base and PubChem), at μ = 0.15 in the pool, after the cut, with no natural-product flag. Every lever below extends that one proven signal to a place it doesn't reach yet. Rules 1–3 say this is the right direction; rule 4 and the "flat" list say what to avoid.

## 3. Probe queue (2–3 per day; each probe is the current best plus one change)

| # | Change on the 0.411 base | Signal it follows | Confidence | Expected ΔLB |
|---|---|---|---|---|
| **P1** | `POOLPOP_MU` 0.15 → **0.25** (C7, line 226) | #3; the identical knob gave +0.005 on the sibling notebook | high | +0.003…+0.005 |
| **P2** | **Prior on the engine-A list** (`--pop-eng 0.25`). Before `fuse2`, re-sort `ENG[mid]` by `z(−rank) + μ·pop(key)`, using a pool `key → pop` dict built once from `P.key` + `POOL_POP`; missing keys get the median. No double counting, because the base and PubChem lists already carry the prior. | #2 + #1: the last list that never sees the prior. Engine A is mostly COCONUT, so the pool map covers most of it. | medium-high | +0.002…+0.005 |
| **P3** | **Natural-product bonus** (`--np-bonus 1.0`): `pop += 1.0·[(np & 6) > 0]` (LOTUS or NPAtlas) in the pool prior, and in the PubChem prior via `pc_np` in V17_PATCH | #1, with a plant-specific refinement of the proven prior (the test set comes from biological extracts) | medium. **Precondition:** the attached popularity dataset must be Version 2 (has `pool_np.npy`/`pc_np.npy`). The notebook prints `np flags: found/absent`; if they are absent it is a no-op and the probe is skipped. | +0.002…+0.006 |
| ~~**P4**~~ **DROPPED 10-03**: A1 on 0.405 scored 0.401 (−0.004) | **Prior before the TOPN cut** (`--pop-precut`): keep the top 200 by ranker score, re-sort by `z + μ·pop`, then cut to 60 | #2 (it reaches truths at ranks 61–200). Not the same as the flat TOPN 100 test, because a new ordering signal picks the 60. | medium | +0.001…+0.004 |
| P5 | Second point on the best knob: μ 0.35 if P1 wins, or bonus 2.0 if P3 wins | #3 (stop before the λ=2 cliff) | — | ± |
| P6 | Stack all winners into one notebook (the combination itself is one probe). Optionally include the library gate (`--lib-gate`: ICE/GL λ = 0 when `lib_max ≥ 0.9`, in both the C8 base rerank and the C10 fused rerank) as a free safety rider. It does not get its own probe, because it measured 0.000. | — | — | sum of the winners minus overlap |

**Order:**
- Day 1: P1, P2, P3, all on the plain 0.411 base. They are independent, so each one's attribution stays clean.
- Day 2: stack the winners. (P4 is dropped: the identical A1 change lost 0.004 on 0.405.)
- Day 3: P5 / P6.

**Pre-registered rule** (goes into `gate_report.md` before any submission): keep a change if LB ≥ best + 0.001, revert it if not, and on a tie keep the simpler variant. If a probe moves the LB but its counters show no change, treat the move as noise.

**Deliberately not probed:** the library gate (0.000), MetFrag in the gap (≈0), patent counts (they favour synthetic chemistry), DreaMS k-NN (−0.009), ALPHA/KRR retuning (tuning, not a signal), and Class-3 generation (deferred).

## 4. Implementation

- **`build/patch_v29.py`**: modelled on `build/patch_supernext.py`.
  - Reuse its helpers `src_of`, `set_src`, `must_replace` and `find` by import.
  - Its source is `casmi26-v29-glacier-m-h-full-fusion-0-411.ipynb`; it writes `casmi26-t4v2-<tags>.ipynb`.
  - Flags: `--mu X`, `--pop-eng X`, `--np-bonus G`, `--pop-precut`, `--lib-gate`, `--out`.
  - Each flag touches only C7, C4 (V17_PATCH, for P3) or C10, inside the existing `try/except`, so any failure keeps 0.411 behaviour (rule R5).
  - Clear outputs and papermill metadata. Keep `IS_RERUN = True` exactly as in 0.411.
- **Counters printed in every variant:**
  - `pop stats` (min / median / p99 of μ·pop)
  - `engine lists re-sorted N, top-1 moved M`
  - `np flags found/absent`, plus the share of NP-flagged top-1s
  - `promoted into top60 N`
  - the existing `top-1 changed in`
- **`build/test_patch_v29.py`**, using the pattern of `build/test_patch_supernext.py`:
  - every variant compiles, keeps the same cell count, touches ≤ 3 cells, and has outputs cleared;
  - the base variant is behaviour-neutral;
  - each flag's code is present;
  - the engine re-sort helper and the NP bonus are executed against stubs.
  - Register it in `build/run_all_tests.sh`.
- **Docs:**
  - `specs/track4_priors_probes/plan_v2.md` (sections 1–3 of this plan); P1–P6 rows in `gate_report.md`.
  - `specs/README.md`: base is 0.411.
  - Fix `comparision_summary.md`: the library gate and MetFrag gap measured 0.000; only μ = 0.25 is proven.
  - Retire the old 0.405-based `casmi26-t4-*.ipynb` probes.
- After each LB result, log it with `/update-experiments` (re-read `EXPERIMENTS.md` first, because another agent edits it).

## 5. Verification

- **Local:**
  - `python build/test_patch_v29.py`, then `sh build/run_all_tests.sh`.
  - `python .agents/skills/analyze-notebook/scripts/compare.py casmi26-v29-glacier-m-h-full-fusion-0-411.ipynb casmi26-t4v2-<tag>.ipynb` must show only the intended cell changes.
- **Kaggle:**
  - Attach the same datasets as 0.411.
  - The run log must show the new counters with non-zero movement, plus `np flags found` for P3.
  - `submission.csv` is (400, 2), with ≤ 25 unique IK14 per row.
  - Runtime about 4.4 h, the same as 0.411. The added cost is under 1 minute.
