# Track 4: popularity-prior probes. Handover summary (2026-10-03)

This file lets a new agent start Track 4 with the full context. Read it first. Then read `plan_v2.md` (the approved plan) and the "Track 4 v2 probe register" at the bottom of `gate_report.md` (pre-registered thresholds and the results so far).

Also follow the project rules in `CLAUDE.md`:
- Never read raw `.ipynb` JSON. Use `python .agents/skills/analyze-notebook/scripts/compare.py a.ipynb b.ipynb`.
- Log every LB result with `/update-experiments`.
- Another agent edits `EXPERIMENTS.md` concurrently, so re-read it before you write.
- Commit only when the user asks.

---

## 0. UPDATE 2026-10-03 (read this first: it supersedes sections 1, 6 and 8)

- **New base: 0.415.** `casmi26-v29-fusion-pop-glmh-0-415.ipynb` is code-identical to 0.411. Only the run-mode line differs, and one attached dataset version (20161857 → 20235258); its author used private popularity arrays for the scored run. **The same code scores ±0.004 from run to run**, which is now the noise floor. Always submit with the T4 on: the 0.411 save log ran on CPU, and ICEBERG scored only 90 of 400 molecules there.
- **The popularity lever is closed.** P1 (μ 0.25) scored 0.399 (−0.012, GPU confirmed). 0.411/0.415 already carry popularity in two lists (pool μ 0.15 + PubChem λ 0.25), so adding more weight anywhere goes past the peak. P2 and P3 are dropped and P5/P6 are void (EXPERIMENTS rows 30–31, graveyard).
- **The track now targets Class 3: mutator v2 (c3v2).** See `c3v2_plan.md`, the c3v2 register in `gate_report.md`, and EXPERIMENTS §2 B3.
  - **Generator.** Anchored whole-molecule mutation from engine-A spectral analogs whose exact mass Δ matches ≤ 2 signed moves (adding or removing groups: sugars, acyl groups, methyl, hydroxyl). The f·z-ranked beam prunes intermediates more than τ below their anchor.
  - **Gate.** Stage 1 compares f·z within the molecule: the best edit must beat its anchor and the pool, and also the PubChem-only top for `sure`. Stage 2 adds the real `lib_max`, ICE/GL competition with the retrieved top-1, and engine agreement. A run-wide cap applies (sure ≤ 3%, fire ≤ 12%, the channel switches off above 25%).
  - **Splice.** sure → [1,2,3], likely → [2,3,4], after the fused rerank.
  - **Ready to run:**
    - `casmi26-c3v2-tau8.ipynb`: LB probe C1. Read the `C3V2 *` counters in the log.
    - `casmi26-c3cal.ipynb`: calibration, commit run only. Attach the 0.415 datasets plus a dataset containing `p1_novel/pseudo_c3.parquet` and `c2_displace.parquet`.
  - **Regenerate** with `python build/patch_c3v2.py [--tau 8] [--no-rank1] ...` and `python build/patch_c3cal.py`.

---

## 1. Where we are, in one paragraph

The best public LB is **0.411**, from `casmi26-v29-glacier-m-h-full-fusion-0-411.ipynb` (repo root). Track 4 tries to get above it **without attaching any new dataset and without new training**. The approach is to extend the one signal that keeps paying, the **PubChem/PubMed popularity prior**, to the places in 0.411 that it does not reach yet.

Three probe notebooks on 0.411 are built, tested and ready to submit:
- `casmi26-t4v2-mu0.25.ipynb` (P1)
- `casmi26-t4v2-popeng0.25.ipynb` (P2)
- `casmi26-t4v2-np1.ipynb` (P3)

None of them has an LB score yet. Class-3 de novo generation is **deferred** by the user decision; do not plan it in this track.

---

## 2. The 0.411 pipeline (what we patch)

Reference copy in py:percent form: `scratch/casmi26-v29-glacier-m-h-full-fusion-0-411.py`. Line numbers below refer to it.

| Cell | Lines | What it does |
|---|---|---|
| C2 | — | Run mode. The original hard-codes `IS_RERUN = True`, which makes every save take about 4.4 h. Our probes restore the commit/rerun switch (see section 5). |
| C4 | — | `V17_PATCH`, the PubChem-only channel run in a subprocess. Settings: `POP_LAM=0.25`, `POP_UNION=200` (pass-1 union with the most popular candidates). `fz_top` gate: if `rel>600`, use `SLOTS_AGG [2,4,6,8,10]`; otherwise use `SLOTS_GENTLE [4,8,12,16,20]`. |
| C7 | 226–274 | Base = v4n pool + ranker, cut to `TOPN=60`. Then the pool popularity prior `POOLPOP_MU=0.15` on `pool_lsid + pool_lpmid` is applied **after the cut** (it only re-orders the top 60). |
| C8 | 371–385 | ICEBERG/GLACIER rerank of the base, λ=1. GLACIER is used on `[M+H]+` only. |
| C9 | 403–440 | Merge of base, engine A and the PubChem channel. |
| C10 | 450–496 | `fuse2` RRF (`ALPHA=0.6`, `KRR=3`) of the base list and the engine-A list (BIO+AFIX, 40 candidates, keys from `E.canon_key`). Final ICE/GL rerank. Writes the submission. |

**Runtime:** about 4.4 h on a T4.

| Stage | Time |
|---|---|
| Engine 2 | 2642 s |
| PubChem | 2930 s |
| Base | 1188 s |
| ICE + GL | 8690 s |

**Attached popularity dataset:** `dmitriigluzdov/casmi26-pubchem-popularity-prior`. Version 2 adds:
- `pc_np` / `pool_np` (bit flags: 1 = COCONUT, 2 = LOTUS, 4 = NPAtlas)
- `pc_lpatent`, `pc_cid`, `pc_ik14`

See `specs/Dataset_intro/popularity_dataset_intro.txt`. **We have not confirmed** that the version attached to 0.411 is v2. P3 prints `np flags found/absent` to check this.

---

## 3. History: how we got here

| Step | Notebook | LB | What changed |
|---|---|---|---|
| Earlier base | the 0.399 notebook (see `EXPERIMENTS.md`) | 0.399 | v4n + ICE/GL + engine-2 RRF |
| Ours | `evenda-casmi-inferencesupernext` / `enveda-casmi-2026-0-405.ipynb` | 0.405 | pool popularity prior μ 0.15, after the cut |
| Public | `casmi26-apex-v4b-0-409-sota.ipynb` | 0.409 | μ 0.25 (+0.005; its own ablation: 0.404 → 0.409). Library gate (lib_max ≥ 0.9 → ICE/GL λ = 0) gave **0.000**. MetFrag-in-gap gave about 0. |
| Public | `casmi26-v29-…-0-411.ipynb` | **0.411** | Popularity inside the PubChem channel (V17_PATCH, λ 0.25, POP_UNION 200, fz_top gate) gave +0.006. GLACIER limited to [M+H]+. |
| Public | `casmi-dreamS-0-396.ipynb` | 0.396 | DreaMS k-NN third list: −0.009. Dead end. |
| Public | `casmi26-sota-v37b-adduct-aware-metfrag-0-397.ipynb` | 0.397 | Adduct-aware MetFrag in engine 2: −0.014. Dead end. |

The full LB-notebook comparison is in `comparision_summary.md`. Its claims about the library gate and "0.415+" are corrected inline: **only μ 0.25 is proven.**

### Track 4 v1 (on the 0.405 base, now superseded)

`build/patch_supernext.py` generated these probes:
- `casmi26-t4-base`
- `-mu0.08` and `-mu0.3`
- `-a1precut`
- `-a2popfused0.15`
- `-a4c1freeze`
- `-a5genonly`

After that, 0.409 and 0.411 appeared and already contained A3 (μ 0.25), B1 (PubChem prior) and A4 (library gate). **Do not submit the remaining `casmi26-t4-*` notebooks.** Their base is 0.405 and they are obsolete.

**A1 result (10-03): `casmi26-t4-a1precut` scored 0.401.**

`compare.py` shows it was built on **0.405** (supernext), not on 0.399. Only C7 and C10 differ from supernext, and the C10 counters are inactive. So the real delta is **−0.004 → FAIL, REVERT**. The user initially read it as +0.002 over 0.399; that reading is wrong.

What A1 did: it applied the prior over the ranker's top 200 and then cut to 60. That lets famous but spectrally poor compounds from ranks 61–200 push good candidates out of the ICE/GL window.

**Lesson (now in the `EXPERIMENTS.md` graveyard):** popularity must **re-order** a spectral shortlist, never **choose** it. This is consistent with the earlier result that TOPN 60 → 100 gave 0.000.

Logged as `EXPERIMENTS.md` row 29 (−0.004, REVERT), plus a graveyard entry. `gate_report.md` and `plan_v2.md` are updated.

---

## 4. Consistent signals: the rules that pick probes

From `EXPERIMENTS.md` and the 0.409 ablation.

**These pay every time:**

1. **A new orthogonal evidence source, fused at rank level (RRF):**
   - ICEBERG +0.035
   - BIO +0.009
   - GLACIER +0.007
   - second engine +0.016
   - popularity prior +0.006 / +0.010 / +0.006
2. **Every scorer sees every candidate:**
   - ICE_UNION +0.004
   - PubChem depth 1000 → 5000
   - POP_UNION 200
   - prior on the PubChem list
3. **A fuller voice for an independent signal, up to about 1 z-unit:**
   - ICE/GL λ 0.5 → 1.0: +0.007
   - μ 0.15 → 0.25: +0.005
   - λ = 2: **−0.034** (that is the cliff)
4. **Purifying a noisy input:** GLACIER on [M+H]+ only; the fz_top gate.

**These are flat or negative every time:**
- Swapping a model within the same family (FPNet swap −0.002).
- Deepening without a new ordering signal (TOPN 100: 0.000; A1 precut: −0.004).
- Local tuning (−0.015).
- Absolute-score gates (−0.064).
- Weak heuristics applied globally (MetFrag everywhere 0.390, DreaMS k-NN −0.009, adduct MetFrag −0.014).
- Protective gates (library gate 0.000).

---

## 5. Implementation: `build/patch_v29.py`

- **Source and output:** the generator reads the 0.411 notebook and writes `casmi26-t4v2-<tags>.ipynb` to the repo root.
- **Shared helpers:** it imports `find`, `must_replace`, `set_src` and `src_of` from `build/patch_supernext.py`.
- **Outputs:** it clears outputs and papermill metadata.
- **Safety:** every change sits inside the notebook's existing `try/except` fallbacks, so a failure falls back to 0.411 behaviour.

| Flag | Cell | Effect |
|---|---|---|
| (always) | C2 | Replaces `IS_RERUN = True` with `IS_RERUN = bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN')) or _sig != '654e030e4173780e945252b4a5adf70b'`. **A saved version runs the 12-molecule smoke test (`SMOKE_N=12`, `ICE_BUDGET=300`) in about 15–20 min. The full 400 molecules run only on submission.** |
| (always) | C7, C10 | Counters: `TRACK4v2 pop stats` (min/median/p99 of μ·pop) and `TRACK4v2 MU_ENG ...` (engine lists re-sorted, top-1 moved, pop key hits). |
| `--mu X` | C7 | `POOLPOP_MU = X` |
| `--pop-eng X` | C10 | `_pop_eng` re-sorts the engine-A list by `z(-rank) + X·pop(key)` before `fuse2`. It uses a `_KPOP` dict (`P.key` → `POOL_POP`, bytes decoded). Missing keys get the median of the hits. Counters: `N_ENGPOP*`. |
| `--np-bonus G` | C7 + C4 | `pop += G·[(np & 6) > 0]` (LOTUS or NPAtlas). In the pool it reads `pool_np.npy`. In PubChem, `V17_PATCH` is rewritten at run time to load `pc_np.npy` in `init_worker`, and the setting is passed via `CASMI_NP_BONUS`. Prints `np flags found/absent`. |
| `--pop-precut` | C7 | `TOPN_PRE=200`, then the prior, then the cut to 60. **Dropped (same as A1, −0.004). Do not use.** |
| `--lib-gate` | C8 + C10 | ICE/GL λ = 0 when `lib_max ≥ 0.9`. Measured 0.000 on 0.409; it may only ride along on the final stack. |
| `--src`, `--out` | — | Input/output overrides. |

**Tests:** `python build/test_patch_v29.py` runs 77 checks, which cover:
- compile checks;
- cell count;
- at most 5 cells touched;
- the base variant touches exactly 3 cells;
- the run-mode switch is restored;
- the V17 anchors appear exactly once, and the C4 block is exec'd;
- `_pop_eng` is exec'd against stubs.

It is registered in `build/run_all_tests.sh`. The whole suite passes except `test_stage_d_colab.py`, which was already broken before this track.

**Example:**
```
python build/patch_v29.py --mu 0.25                  # -> casmi26-t4v2-mu0.25.ipynb
python build/patch_v29.py --mu 0.25 --pop-eng 0.25   # stacked
```

---

## 6. Probe queue and status

Keep rule (pre-registered): **LB ≥ best + 0.001** keeps the change. Below that, revert it. On a tie, keep the simpler version. One change per probe; submit 2–3 per day.

| id | Notebook | Change | Counter that must move | Expected | Status |
|---|---|---|---|---|---|
| **P1** | `casmi26-t4v2-mu0.25.ipynb` | pool μ 0.15 → 0.25 | `pop stats` p99 rises ×1.67 | +0.003…+0.005 (the same knob gave +0.005 on 0.409) | **ready, submit first** |
| **P2** | `casmi26-t4v2-popeng0.25.ipynb` | engine-A list gets the prior | `engine lists re-sorted` > 0, `top-1 moved` > 0, `pop key hits` ≥ 50% | +0.002…+0.005 | **ready.** If key hits ≈ 0 (`E.canon_key` ≠ `P.key` format), record VOID, not FAIL. |
| **P3** | `casmi26-t4v2-np1.ipynb` | NP bonus 1.0 (pool + PubChem) | both `np flags ... found` | +0.002…+0.006 | **ready, if the attached popularity dataset is Version 2.** Check the dataset version on Kaggle, or a fast-save log line. If `absent`, VOID: do not submit. |
| P4 | — | prior before the cut | — | — | **DROPPED** (A1 = −0.004) |
| P5 | to generate | second point on the winning knob (μ 0.35 if P1 wins; G 2 if P3 wins) | — | ± (stay below the λ = 2 cliff) | after P1–P3 |
| P6 | to generate | stack of all winners (+ optional `--lib-gate`) | all | sum minus overlap | after P1–P3 |

**Not probed, on purpose:**
- library gate on its own (0.000)
- MetFrag in the gap (≈ 0)
- patent counts (they favour synthetic chemistry)
- DreaMS k-NN
- ALPHA/KRR retuning
- deeper TOPN
- Class-3 generation (deferred)

---

## 7. How to submit a probe (user workflow)

1. Upload or import the notebook to Kaggle. Attach **exactly the same datasets as 0.411**.
2. **Save Version.** This runs the smoke test (about 15–20 min). Check that the log shows the `TRACK4v2` lines and, for P3, `np flags found`.
   - Saved outputs are placeholder runs (a slice of train), so `lib_max`, `rel` and the merge stats there mean nothing. Only check that the code paths ran.
3. **Submit to the competition.** The hidden rerun runs the full 400 molecules in about 4.4 h.
4. When the score arrives:
   - fill the LB/verdict cells in the probe register in `gate_report.md`;
   - log it with `/update-experiments` (the delta is **against 0.411**, or against the current best if one is newer);
   - always check with `compare.py` which base a submitted notebook really came from before you compute a delta. The A1 confusion came from skipping this.

---

## 8. Next steps for the incoming agent

1. Wait for, or ask for, the P1/P2/P3 LB results. Log each one as described in section 7.
2. Apply the VOID rules for P2 and P3 before judging them.
3. Generate P6, the stack of winners, with `patch_v29.py` (combined flags). Optionally add `--lib-gate`. Run `python build/test_patch_v29.py`, then `compare.py` against 0.411, and check that only the intended cells changed.
4. Then generate P5, the second point on the strongest knob.
5. If every probe fails, the popularity lever is saturated on 0.411. Go back to signal #1 (a new orthogonal source). Class 3 is still the user's later stage, not this track's.
6. Update `specs/README.md` and the `EXPERIMENTS.md` banner if a probe sets a new best.

## 9. File map

| File | Purpose |
|---|---|
| `specs/track4_priors_probes/summary.md` | this handover |
| `specs/track4_priors_probes/plan_v2.md` | approved v2 plan (lever mapping, signals, P1–P6) |
| `specs/track4_priors_probes/gate_report.md` | pre-registered gates; v1 rows A1–A5/B1–B3; v2 probe register |
| `specs/track4_priors_probes/comparision_summary.md` | the user's LB-notebook comparison (with corrections) |
| `specs/track4_priors_probes/roadmap.md`, `track1_summary.md` | earlier roadmap; Track 1 summary (novel generation and Class-3 routing) |
| `specs/Dataset_intro/` | dataset notes: popularity prior, ICEBERG, GLACIER, simulated ranker rows |
| `build/patch_v29.py`, `build/test_patch_v29.py` | v2 generator and its tests (on 0.411) |
| `build/patch_supernext.py`, `build/test_patch_supernext.py` | v1 generator (on 0.405), superseded |
| `scratch/casmi26-v29-glacier-m-h-full-fusion-0-411.py` | py:percent of 0.411, for reading |
| `EXPERIMENTS.md` | master ledger (row 29 = A1; graveyard has the precut rule) |
| `updated_knowledge_base.txt` | the knowledge base written earlier in this track |

**Uncommitted work.** Nothing from this track is committed yet:
- `build/patch_v29.py`, `build/test_patch_v29.py`
- the `specs/track4_priors_probes/` files
- the edits to `EXPERIMENTS.md` and `specs/README.md`
- the probe notebooks

Commit only when the user asks. Never stage `.npz`, `.pt`, `.parquet` or `.zip` files.
