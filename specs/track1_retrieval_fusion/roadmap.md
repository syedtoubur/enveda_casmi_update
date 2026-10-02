# specs_v4n_update — roadmap

*Version 1.1, 2026-10-01. Target: `casmi26-v4n-engine-fusion-union-lb-0-399.ipynb` (forked, public
LB 0.399). Strategy and rules: `plan_v2.md` §3 (R1–R5). Chemistry: `plan.md` §3, **superseded by §4
below**. Decisions taken with the user on 2026-10-01: no baseline re-submission, S2 dropped,
scaffolds from PubChem top-25 only, fast first submission then re-decide. ~8–9 weeks remain.*

**Protocol** (as `specs_novel/roadmap.md`): each gate's thresholds go into
`specs_v4n_update/gate_report.md` **before** the measurement runs. A failed gate stops that stage.
Every recorded number cites the cell or script that produced it.

---

## 1. The notebook as forked

16 cells. The v4n core (cells 7, 8, 9, 10, 13) is **byte-identical** to
`casmi26-v4n-inference (1).ipynb` — all five diffed. The 0.399 lift is cells 3, 5, 12, 15.

| cell | role | we touch it |
|---|---|---|
| 1 | config, dataset discovery, MANIFEST SHA-256 check (64 files) | yes — budgets, `K_NOVEL` |
| 3 | commit switch (`IS_RERUN`, `SMOKE_N = 12`) | yes — cap the mutator on commits |
| 5 | **engine A**: embedded `pv.py` + `pv_fp.py` + `casmi_engine.py` → `eng_runner.py` → `ENG` | no |
| 7 | PubChem channel subprocess → `PC` | yes — export `z` (M1) |
| 8–10 | pool symlinks, spectrum cache, v4n engine + ranker → `BASE` | no |
| 12 | ICEBERG + GLACIER, incl. the `ADDED` union block | yes — union entry (M3) |
| 13 | gated PubChem merge → `submission.csv` | no |
| 15 | weighted RRF (`ALPHA=0.6`, `KRR=3`) + forward re-rank → overwrite `submission.csv` | yes — third list (M4) |

### 1.1 Hard constraints

- **MANIFEST lock.** Cell 1 asserts `sha256` of 64 files in `casmi26-v4b-models`. Nothing may edit
  `casmi/engine.py` or any v4n model file. Every change below is additive and outside that package.
- **Subprocess isolation.** Engine A, the PubChem channel and the v4n engine each run their own
  `casmi` snapshot. The mutator gets its own subprocess for the same reason (R5).
- **Rank-level only.** `fuse2` and the engine's `blend_scores` both operate on rank-normalised
  values, never raw scores. The mutator joins the same way (R3).
- **Forward weight ceiling.** `ICE_LAM = GL_LAM = 1.0`. λ = 2.0 measured −0.034. Never raise (R4).

---

## 2. Dataset inventory — current attachments are sufficient

The fork carries every file the mutator needs. **No new dataset.**

| the mutator needs | already attached |
|---|---|
| RDKit 2026.03.3 (`MurckoScaffold`, `rdChemReactions`, `MolToInchiKey`) | rdkit wheel ds, installed cell 1 |
| scaffold source | `PC` (cell 7), in memory |
| 10,226-bit fingerprints for `f·z` | `chem.raw_fingerprint` + `fp_bits.npy` in `casmi26-v2-pool` |
| the `z` logit vectors | computed in `pc_runner.py`; **must be exported** — M1 |
| dedup key set | `pool_meta.parquet` `key` + `pc_keys` + `ENG` keys |
| forward scoring | `casmi26-iceberg`, `casmi26-glacier` |
| formula enumeration, palette | pure Python |

**Tooling:** RDKit only, plus the notebook's own `casmi.chem` (`raw_fingerprint`, `score_key`) and
numpy. No new package, no internet, no GPU. Runs as a CPU subprocess importing `V3CODE`, structurally
a clone of `pc_runner.py`. Measured locally: 3,430 decorations/s, 10,443 Morgan-4096 fp/s per core.

Full inventory of what the notebook consumes (all present): `casmi26-v4b-models`,
`casmi26-v3-models`, `casmi26-v2-pool`, `casmi26-pubchem-tier`, `casmi26-fpnet-full1`,
`casmi26-iceberg`, `casmi26-glacier`, rdkit wheel, prvsiyan pool (`coco_*`, 6,930-bit `fp_bits.npy`),
`prvsiyan/chebi-lipidmaps-casmi26` (`bio_*`), `sim_rank_rows_nofp.npz`, `rank_train.npz`,
`casmi26-fp-models-v2`, competition data.

### 2.1 Why S2 (rank-row rebuild) was dropped

Recorded for the file. Both of engine A's rankers are **trained at runtime**
(`casmi_engine.py:396`, `fit_rankers`) from pre-computed feature rows built with
`MAX_PEAKS = 256`, `SIM_POWER = 3.0`, `PPM_WIN = 10.0`. Our tier-1 measurements favour 64, 4.0, 8.5,
but those constants feed `_clean` and the similarity kernels that *produced those stored rows* —
setting them at inference while fitting on old rows is train/inference skew
(`casmi-ranker-fpnet-coupling`, `casmi-feature-schema-append-only`; same shape as v4m's −0.002).
Transplanting them needs an offline row rebuild and our own dataset. Dropped by decision; revisit
only if the mutator stalls.

Also noted: **`PPM_CENTER = 1.45` is already in engine A** in a better form —
`casmi_engine.py:308` emits `[ppm, |ppm|, |ppm + 1.5|]` as ranker features, so the offset is learned
rather than hard-centred. `plan_v2.md` §4 overstated our orthogonality advantage here.

---

## 3. GM.0a — oracle-scaffold reachability (measured, local, no Kaggle data)

Run 2026-10-01 against `val_scripts/val_key.pkl` (1,145 truths with `cls`/`cohort`; 1,037 have a
Murcko scaffold, 108 are acyclic). For each truth: take **its own** Murcko scaffold (the oracle —
a strict upper bound, since the real mutator must *harvest* the right scaffold) and ask whether
Δ = formula(truth) − formula(scaffold) is expressible in the palette at depth ≤ N.

| palette | depth ≤4 | depth ≤6 | depth ≤8 | distinct Δ at ≤6 |
|---|---|---|---|---|
| `plan.md` (9 transforms) | 42.5% | **52.4%** | 57.5% | 1,556 |
| **extended (20 transforms)** | 66.3% | **85.5%** | 92.8% | 114,854 |

Why `plan.md`'s palette fails on 47.6% at depth 6:

| failure | n | share |
|---|---|---|
| decoration contains an element the palette cannot add — **N 264**, F 87, Cl 77, S 59, P 14, Br 10, I 7 | 391 | 34.1% |
| CHO-only Δ not composable even at depth 6 | 103 | 9.0% |
| acyclic — no Murcko scaffold at all | 108 | 9.4% |

**Two conclusions that change the spec.**

1. **The palette, not the search, is the binding constraint.** Eleven extra transforms lift the
   ceiling 52.4% → 85.5%. Nitrogen alone accounts for 264 molecules; `plan.md` has no
   nitrogen-adding transform at all, which is a serious omission for alkaloids and anything
   amide-linked.
2. **Depth 4 is too shallow, as suspected, and by more than estimated.** Ops needed when reachable
   peak at 3–4, but 56 cases need 5 and 46 need 6 — `plan.md`'s `Σ|c| ≤ 4` discards 19% of the
   reachable set. Depth 8 buys a further +7.3pp for 8.5× more Δ vectors; **depth 6 is the knee.**

Per-cohort at extended/≤6: `C1_npx` 86%, `C1_xlib` 88%, `C2` 82%, `C2_hard` 88%. The local cohort has
**no class-3 molecules** (545 class-1, 600 class-2 — cf. `casmi-val-blind-to-pool-misses`), so
`C2_hard` (200) is the closest proxy, and it reaches 88%.

**Gate GM.0a: PASS**, conditional on adopting the extended palette at depth 6.

---

## 4. The palette (supersedes `plan.md` §3 step 3)

Formula-space vectors over `(C,H,N,O,S,P,F,Cl,Br,I)`. Each entry replaces an H at an open socket
unless noted. Keep `plan.md`'s nine:

| transform | Δ | mechanism |
|---|---|---|
| `+O` | O+1 | P450 hydroxylation |
| `+CH2` | C+1 H+2 | SAM methylation |
| `+CH2O` | C+1 H+2 O+1 | methoxylation |
| `+CO` | C+1 O+1 | oxidation to ketone |
| `+C2H2O` | C+2 H+2 O+1 | acetylation |
| `+C6H10O5` | C+6 H+10 O+5 | hexosylation |
| `+C6H10O4` | C+6 H+10 O+4 | deoxyhexosylation |
| `+C5H8` | C+5 H+8 | prenylation |
| `-2H` | H−2 | desaturation |

Add these eleven:

| transform | Δ | mechanism |
|---|---|---|
| `+NH` | N+1 H+1 | amination / transaminase |
| `+CONH` | C+1 H+1 N+1 O+1 | amide synthetase |
| `+NHCH2` | C+1 H+3 N+1 | N-methyl amine |
| `+SH` | S+1 | thiolation |
| `+SO3` | S+1 O+3 | sulfotransferase |
| `+PO3` | P+1 O+3 | phosphotransferase |
| `+Cl` / `+F` / `+Br` / `+I` | X+1 H−1 | halogenase |
| `+H2O` | H+2 O+1 | hydration / ring opening |

Depth `Σ|c_i| ≤ 6`. Each formula-space transform needs a matching reaction SMARTS for the
enumeration step; `plan.md` §3 step 4 already has five, so eleven more must be written —
that is the bulk of M2's work.

**Acyclic fallback:** 9.4% of truths have no Murcko scaffold. For a PubChem hit whose scaffold is
empty, decorate the **whole hit** instead of its scaffold. Cheap guard, recovers a tenth of the space.

---

## 5. Execution order

```
M1 export z  ->  M2 mutator + local gates  ->  M3 union entry  ->  M4 third RRF list  ->  SUBMIT
```

No baseline re-submission: the fork's 0.399 is the author's own published score. S2 dropped.
S1 (safe engine-A re-tune) deferred until after the first mutator read — one change per submission.

---

## M1 — export `z` from the PubChem channel (cell 7)

`pc_runner.py` computes `z = pc.molecule_logits(bank, spectra)` per molecule and discards it. Add one
save beside the existing `json.dump`:

```python
np.savez_compressed('/kaggle/working/pc_z.npz',
                    mids=np.array([t[0] for t in tasks], dtype=object),
                    z=np.stack([t[2] for t in tasks]).astype(np.float32))
```

400 × 10,226 float32 ≈ 16 MB, zero extra compute — the vectors are already in RAM. This is the only
thing the mutator cannot otherwise obtain.

---

## M2 — `build/delta_m_mutator.py` + `build/mutator_runner.py`

Local and test-driven, before any notebook edit.

```
scaffolds: Murcko of PC[mid]['pc'][:25]        (measured ~17 unique/molecule, median 19)
                                               (empty scaffold -> use the whole hit, §4)
  -> enumerate target formulas within +-10 ppm (CHNOPS + halogens)
  -> Delta = formula(target) - formula(scaffold); require >= 0 and expressible, |c| <= 6
  -> reaction-SMARTS enumeration over open sockets, generate WIDE (10^3-10^4)
  -> pass 1: Morgan-4096 partial f.z on the ECFP block, keep top 2000
  -> pass 2: chem.raw_fingerprint[bits] . z, sort descending
  -> chem.score_key -> ik14; drop if in pool_keys U pc_keys U eng_keys, or already listed
  -> top K_NOVEL
```

Decompose in **formula space, not mass space** — exact, and it removes the ±10 ppm knapsack
tolerance entirely. Guard ΔM < 0 (0.8% measured on PubChem hits).

Output, byte-identical in shape to `eng_lists.json`:
`{mid: {"smiles": [...], "keys": [...], "formula": [...]}}`. Carrying `formula` saves cell 15 a
`CalcMolFormula` call per candidate — we constructed these, so the formula is known exactly.

`build/test_delta_m_mutator.py` alongside the existing `build/test_*.py`; `build/run_all_tests.sh`
green before M3.

**Gates, both measured locally on `val_scripts/val_key.pkl`:**
- **GM.0a — PASS** (§3), re-confirm ≥ 80% at the final palette and depth.
- **GM.0b (new, kill):** with the oracle scaffold and oracle formula, the **enumeration must contain
  the exact truth ik14** for ≥ 25% of the GM.0a-reachable set. GM.0a says the *formula* is
  reachable; GM.0b asks whether the reaction SMARTS actually place the groups correctly. If
  enumeration cannot hit the truth even given the right scaffold and formula, no amount of `f·z`
  screening will — stop.

Scaffold-harvest rate (is the truth's scaffold among the ~17 from PubChem?) **cannot** be measured
locally: it needs the 6 GB `casmi26-pubchem-tier`. It is folded into the LB read at M4.

---

## M3 — union entry (cell 12)

Clone the `ADDED (ICE_UNION)` block for `MUT` so every generated candidate gets both forward scores.
**Mandatory, not an optimisation.** Generated structures have no honest retrieval score — they were
never in any pool — so RRF rank alone orders them near-arbitrarily among the mutator's own outputs.
The forward models are the only signal that can order them, and because the mutator emits
*same-formula regioisomers* they land exactly in the same-formula groups ICE/GL re-rank inside. This
is the source for which R2 matters most.

```python
if MUT:                              # ADDED (ICE_UNION_MUT)
    n_add = 0
    for m in mids:
        u = MUT.get(str(m))
        if not u or not u.get('smiles'):
            continue
        known = set(BASE[m][1][:TOPN]) | set((ENG.get(str(m)) or {}).get('keys', []))
        ms = [s for s, k in zip(u['smiles'], u['keys']) if k not in known]
        mf = [f for f, k in zip(u['formula'], u['keys']) if k not in known]
        before = len(cands.get(m, []))
        cands[m] = list(dict.fromkeys(list(cands.get(m, [])) +
                                      ice_fuse.ice_candidates(ms, mf, len(ms))))
        n_add += len(cands[m]) - before
    print('ICE_UNION_MUT: mutator candidates added:', n_add, flush=True)
```

Raise budgets in step, as the 0.399 notebook did for its own union: `ICE_BUDGET` 5400 → 7200,
`GL_BUDGET` 4000 → 5200. Headroom: the author reports a 3.5–4.5 h rerun against the 9 h ceiling,
with ICE at ~41 pred/s and GL at ~97 pred/s.

---

## M4 — third RRF list (cell 15) and the first submission

```python
ALPHA, KRR, ALPHA_MUT = 0.6, 3.0, 0.25
def fuse3(v_smis, e_smis, e_keys, m_smis, m_keys, n=40):
    # 1.0/(K+r) v4n, ALPHA/(K+r) engine A, ALPHA_MUT/(K+r) mutator
```

`ALPHA_MUT ≈ 0.25`, below engine A's 0.6 — generated candidates are the least trustworthy source,
and nothing in this lineage was given full weight on first appearance (R4).

**Gate the mutator on weak retrieval**: emit only where `rel = pc_fz[0] - best_pool_fz` is low *and*
`best_pool_fz` is weak in absolute terms. Never unconditionally.

**Sizing the gate so the probe is informative.** Set the threshold so the mutator touches
**80–150 of the 400** molecules. Below ~80 a ±0.002 LB move is indistinguishable from noise and the
submission teaches us nothing; above ~200 the displacement risk stops being bounded. Log the count.

**Accepted risk.** Shipping without a scaffold-harvest measurement means the first read could be a
regression — the lineage's one catastrophe (λ = 2.0, −0.034) was a displacement failure. With
`ALPHA_MUT = 0.25`, a weak-retrieval gate and `K_NOVEL ≤ 25`, the plausible downside is roughly
−0.005…−0.010, against an upside of +0.005…+0.015. With 8–9 weeks and 5 submissions/day that is a
cheap, well-bounded experiment — but it is a *probe*, not an expected gain, and GM.0b must pass first.

**Instrumentation to log (this is the real deliverable of the probe):**
- molecules the mutator fired on, and `K_NOVEL` actually emitted per molecule;
- `ICE_UNION_MUT: mutator candidates added: N`;
- how many mutator candidates survive into the final top-25, and at which ranks;
- `top-1 changed in N` versus the no-mutator run.

If the LB moves and no mutator candidate reached the top 25, the move is noise — read the counters,
not the score.

---

## 6. Commit-switch maintenance (cell 3)

Every new subprocess must honour `IS_RERUN` or commits will time out:

```python
if not IS_RERUN:
    COMP = SM; ICE_BUDGET = 300
    GL_BUDGET_SMOKE = 200          # cell 12 hardcodes GL_BUDGET = 4000 unconditionally
    MUT_MAXM = 3                   # cap the mutator on commits
```

Existing asymmetry worth fixing while we are in there: cell 3 shortens `ICE_BUDGET` but **not**
`GL_BUDGET`, which cell 12 re-assigns to 4000 regardless.

> Only a **rerun** log is meaningful. A commit runs 12 molecules against a dummy `test.parquet` that
> is a slice of `train.parquet` (`casmi-saved-outputs-are-placeholder`), so `lib_max`, `rel` and
> `merge stats` from a commit mean nothing. Use commits to verify plumbing and per-molecule cost only.

---

## 7. After the first read

| if | then |
|---|---|
| mutator candidates reach the top 25 and LB ≥ 0.401 | sweep `ALPHA_MUT` and `K_NOVEL`, widen scaffolds to `ENG`/`BASE`, consider depth 8 |
| candidates reach the top 25 but LB drops | displacement dominates — tighten the gate, drop `ALPHA_MUT` to 0.15, keep the union entry |
| no candidate reaches the top 25 | the `f·z` screen or the scaffold harvest is failing, not the chemistry — instrument before changing anything |
| GM.0b fails | the mutator ends. Fall back to S1 (engine-A re-tune) and `specs_novel` as the orthogonal source (`plan_v2.md` §5 P3) |

---

## 8. Status

*Updated 2026-10-01 after implementation.*

| stage | needs dataset | status |
|---|---|---|
| GM.0a oracle-scaffold reachability | no | **PASS** — 85.5% with the 20-transform palette at depth 6 (52.4% with `plan.md`'s nine) |
| GM.0b exact-truth recovery | no | **21.6%** of reachable (16.7% of all) against a pre-registered bar of 25% — **marginal miss**, see below |
| M1 export `z` from `pc_runner` | no | **DONE** — `np.savez_compressed` before `json.dump`, verified to round-trip |
| M2 `build/delta_m_mutator.py` + `build/mut_runner.py` + tests | no | **DONE** — 65 checks green, end-to-end subprocess test against mocked `casmi` (`build/test_mut_runner.py`) |
| M3 ICE/GL union entry | no | **DONE** — `ICE_UNION_MUT` block, budgets 7200/5200 |
| M4 third RRF term + submit | no | **DONE** — `ALPHA_MUT = 0.25`; notebook is `casmi26-v4n-mutator-submit.ipynb` |
| S1 engine-A safe re-tune | no | DEFERRED until after the first mutator read |
| S2 rank-row rebuild | yes | **DROPPED** (§2.1) |
| B1 BIO+AFIX as a second scaffold source | no | **DONE** — `casmi26-v4n-mutator-bio-submit.ipynb`, §9 |
| S4 `specs_novel` decoder | yes | own track |

### GM.0b and what it means for the first read

21.6% of formula-reachable molecules have their exact truth recovered **given the oracle scaffold
and the oracle formula**. That is the ceiling before three further lossy stages: harvesting the
right scaffold from the PubChem top-25, the right formula being among those hits, and the truth
surviving the `f.z` screen into the top 25. `C2_hard` — the closest available proxy for class 3 —
recovered 1 of 20.

Multiplying through at plausible rates gives roughly **+0.001 MRR**, an order of magnitude below the
+0.005 GM.1 target. So the first submission is a **probe, not an expected gain**. It was shipped
anyway by decision: submissions are cheap, ~8–9 weeks remain, and the counters it prints are the
real deliverable.

The downside is structurally bounded. With `ALPHA_MUT = 0.25` and `KRR = 3`, a mutator rank-1
candidate scores `0.25/4 = 0.0625`, which equals a v4n rank of 13 — so mutator candidates can only
enter at about ranks 13, 17, 20, 22. v4n's top 12 are untouchable by construction, which confines
displacement to the tail.

### Read the counters, not the score

`MUTATOR: N molecules, M candidates` · `ICE_UNION_MUT: mutator candidates added to ICE input: N` ·
`MUTATOR IN FINAL TOP-25: N candidates across M molecules`.

If the last line is 0, any LB movement is noise and the chemistry was never tested — fix the
harvest or the screen before touching the palette. If it is non-zero and the LB drops, displacement
dominates: lower `ALPHA_MUT` or tighten `MUT_FRAC`. Decision table in §7.

### Known limits, in priority order

1. **Scaffold harvest — now measured for the spectral route (§10).** On `pseudo_c3`, **53.2% of
   truths are the only structure in the 194,668-molecule universe carrying their Murcko scaffold**,
   so no retrieval list can supply it and 65% of the oracle-conditions hits are unobtainable. The
   PubChem route is still unmeasured (it needs `casmi26-pubchem-tier`), so 4.2% is a ceiling for
   analog/engine seeding and a lower bound for PubChem seeding.
2. **Acyclic truths: 21.5%, not 9.4%.** Measured on `pseudo_c3` (129 of 600), where they account for
   70.9% of all misses. Murcko returns nothing; the whole hit becomes the scaffold, which cannot
   add mass-bearing decoration, so these are effectively unreachable.
3. **CHO-only Δ unreachable at depth 6: 9.0%.** Depth 8 would recover ~7pp for 8.5× more Δ vectors.
4. **One ordering per decomposition.** `MAX_DEC = 4` routes per Δ, each applied socket-creators-first.
   More orderings would widen coverage at linear cost.
5. **No regioisomer *shuffling*.** Every transform adds mass, so a truth that is a positional isomer of
   a hit at the same formula is out of reach by construction.

---

## 9. BIO+AFIX as a second scaffold source — `casmi26-v4n-mutator-bio-submit.ipynb`

The mutator above harvests scaffolds from one source: each molecule's PubChem top-25. Engine A
(cell 5) independently produces a top-40 per molecule from a different structure universe —
COCONUT + **ChEBI + LIPID MAPS (BIO)** + training structures — ranked by *spectral* evidence
(**AFIX** analog propagation at `N_ANALOG = 200`, MetFrag-lite fragments, FPNet logits, two
LightGBM rankers) rather than by fingerprint `f·z`. Natural-product-biased where PubChem is
synthetic-heavy, spectrally ranked where PubChem is fingerprint-ranked: an orthogonal scaffold
source, which is the only lever this lineage has ever been paid for.

Shipped as a **separate** notebook. `casmi26-v4n-mutator-submit.ipynb` is byte-unchanged, so both
go to the LB and the two scaffold sources stay separable.

### 9.1 What the audit established

- **ENG candidates are mass-window constrained**: `casmi_engine.compute_channels` sets
  `cand = pool_window(target, CFG.PPM_WIN = 10)` (30 ppm fallback when the window is empty), so
  engine structures sit at the same target mass as the PubChem hits and their formulas are valid
  target hypotheses.
- **Key spaces are identical**: `casmi_engine.canon_key` is
  `MolToInchiKey(TautomerEnumerator().Canonicalize(m))[:14]`, the same as `chem.score_key`.
- **`ENG` is already in scope** at the mutator cell, and `/kaggle/working/eng_lists.json` already
  exists. **No new dataset attachment, no cell reordering** — §2 still holds.
- **The trap**: `harvest_scaffolds(smiles, limit)` truncates its input at `limit`, and `mutate`
  passed `max_scaffolds = 25` to it. Concatenating 25 PubChem and 40 engine hits and keeping that
  one limit throws the engine list away silently. Hence per-source harvest, then union.
- **`mutate` returns early at `cap`** and walked scaffolds in dict order, so a concatenated order
  would let PubChem claim the whole budget. Hence round-robin.
- Latent in the pc-only notebook: a mutator candidate whose key is already in `ENG` collected
  `ALPHA + ALPHA_MUT` in `fuse2` — a double vote. Adding engine keys to the novelty set removes it.

### 9.2 Measured locally (RDKit 2026.03.3, `pc_lists.json`, all 400 molecules)

| quantity | measured |
|---|---|
| distinct formulas per PubChem top-25 | **1 for 253/400**, 2 for 78, ≥3 for 69 |
| scaffolds/molecule, PubChem top-25 | mean 17.3, median 19, max 25 |
| scaffolds/molecule, 65 hits (zero-overlap worst case) | 37–63, i.e. **+24…+39** |
| `mutate` generation, 1 source | **3.7 s**/molecule |
| same scaffold growth, formulas held to PubChem's | **5.3 s**/molecule (**1.46×**) |
| full worst case, 12 molecules, 45 s deadline | mean **22.7 s**, median 18.1 s, max 46.4 s; deadline hit 2/12 |
| engine-only scaffolds, worst case | **28–38**/molecule against 4–24 from PubChem |
| share of generated products from an engine scaffold | **13–95%**, median ≈42% |

**ENG adds scaffolds, not formulas.** A PubChem top-25 almost always carries exactly one formula,
and ENG sits in the same ±10 ppm window, so `max_formulas = 6` stays untouched and the formula
union is cheap insurance rather than a change of behaviour.

**On reading the worst-case row.** The stand-in second source is built from *other* molecules'
PubChem hits at nearby masses, so it has near-zero scaffold overlap with the molecule's own list
**and** brings 5–11 extra formulas. The real engine list sits at the same target mass, so its
overlap is higher and its formula contribution is ~0. These numbers are an upper bound on cost, not
a prediction. The 1.46 s row isolates the scaffold effect alone; the 22.7 s row is what happens if
everything goes wrong at once.

**Runtime still clears GB.3 at the upper bound**: 120 gated molecules ÷ 4 workers × (22.7 s + ~3.3 s
for the two `f·z` passes) ≈ **780 s**, against a 1,200 s gate and a 32,400 s ceiling.

### 9.3 Design

1. `mutate_sources(sources, reach, per_source=(25, 40), …)` harvests **each source with its own
   limit**, unions on canonical scaffold SMILES, and walks them **round-robin**. A scaffold found by
   several sources is owned by the first in `sources` order, so `owner == "eng"` means *absent from
   PubChem* independently of the interleaving.
2. `mutate()` is now a one-source wrapper. Its output is **byte-identical** to the previous
   implementation — verified against a reconstruction of it on 10 real molecules, matching output
   and all six stats counters, including at `cap` = 1, 7 and 50. This is what makes the bio notebook
   a single-variable change.
3. **Per-molecule wall-clock deadline** (`MUT_DEADLINE = 45 s`, generation only), returning partial
   results rather than raising. There was no time guard at all before — only `cap` — and the pair
   count roughly triples. It is not decorative: in the worst-case run it fired on 2 of 12 molecules,
   capping two that would otherwise have run past a minute each.
4. `known = pc_keys ∪ eng_keys ∪ pool_keys`. `pool_meta.parquet` does not carry ChEBI/LIPID MAPS, so
   the engine keys are genuinely not already covered.
5. **`MUT_K` 25 → 40** under `--bio`, pure `f·z` order, so engine-derived products add on top rather
   than displacing PubChem-derived ones. Fusion cannot see past ~rank 25 anyway (`ALPHA_MUT/(KRR+26)`
   = 0.0086 ≡ v4n rank ~114), and **only the top 25 enter the ICEBERG/GLACIER union**, so the
   forward-simulation load is identical to the pc-only notebook.
6. Per-candidate `src` provenance in `mut_lists.json`, aggregated into the counters below.

### 9.4 Gates

| | gate | status |
|---|---|---|
| **GB.1** | `build/run_all_tests.sh` green | **PASS** on the mutator track (4 test files); one pre-existing unrelated failure, see the note in §9.7 |
| **GB.2** | no `MUT_ENG` ⇒ output identical to pc-only | **PASS** (module + subprocess) |
| **GB.3** | mutator subprocess ≤ 1,200 s in the rerun | **PASS** projected at 780 s (worst case); confirm from the log |
| **GB.4** | eng-only scaffolds > 0 **and** candidates from them > 0 | read on the commit run |

### 9.5 Read the counters, not the score

```
mutator scaffold sources: PubChem top-25 + engine-A top-40
MUT SOURCES: scaffolds pc=N eng=N eng-only=N | candidates from eng-only scaffolds=N | deadline hit on N molecules
MUT SOURCES (listed): N of M candidates grew from an engine-only scaffold, across K molecules
MUTATOR IN FINAL TOP-25: N candidates across M molecules
```

**GB.4 first.** If `candidates from eng-only scaffolds` is 0, the plumbing broke or `f·z` always
prefers PubChem scaffolds — either way the LB number says nothing about BIO+AFIX chemistry.

Then, against the pc-only notebook's score:

- **bio > pc-only** → keep; next lever.
- **bio ≈ pc-only with engine candidates landing** → the source reaches the list but does not help.
  The `f·z` screen is the bottleneck (GM.0b: 21.6%), not the palette or the source.
- **bio < pc-only** → displacement. Revert to pc-only, or re-ship with a per-source floor in
  `mutate_sources`' output.

### 9.6 The known weakness of this design

Two measurements say displacement is a live risk, not a theoretical one:

- **The engine source dominates the generated pool.** It contributes 28–38 scaffolds per molecule
  against PubChem's 4–24, and 13–95% (median ≈42%) of the generated products — before the screen
  sees anything. A longer input list (40 vs 25) mechanically yields more scaffolds.
- **Under an uninformative screen it takes everything.** In the mocked end-to-end run, where `z` is
  random noise and `f·z` carries no signal, *every* listed candidate came from an engine-only
  scaffold.

`MUT_K = 40` makes displacement *less likely*; it does not prevent it. Pure `f·z` order means the
engine can still occupy ranks 1–25 whenever the screen cannot separate the sources — and GM.0b put
that screen's oracle-conditions recovery at 21.6%, so it is not strong. `MUT SOURCES (listed)` is
the number that settles it. A share near 100% means the screen is not discriminating, and the
per-source floor in `mutate_sources`' output is the right next move.

### 9.7 Files

| file | change |
|---|---|
| `build/delta_m_mutator.py` | `mutate_sources`, `_union_scaffolds`, `_union_formulas`, deadline, provenance; `mutate` → wrapper |
| `build/mut_runner.py` | `MUT_ENG` / `MUT_DEADLINE` env, two-source task, engine keys in `known`, `src` + diag fields |
| `build/patch_v4n_mutator.py` | `--bio` flag; no-flag output still reproduces the pc-only notebook exactly |
| `build/test_delta_m_mutator.py` | 65 checks (42 + 23), incl. single-source equivalence and the limit trap |
| `build/test_mut_runner.py` | **new** — subprocess test against a mocked `casmi`, 9 scenarios |
| `build/test_mut_cell.py` | **new** — executes the notebook's mutator cell with stubbed globals; runs against both notebooks |
| `build/run_all_tests.sh` | `run()` helper: `\| tail -1` made `set -e` read *tail's* status, so a failing test passed silently |

Unchanged: `casmi26-v4n-mutator-submit.ipynb`, `casmi/engine.py` and the 63 other MANIFEST-locked
files, `ALPHA`, `KRR`, `ALPHA_MUT`, `MUT_FRAC`, `TOPN`, forward λ, and the PubChem channel's depth.

**Note on the suite**: `build/test_stage_d_colab.py` fails (`casmi_validate_ab_colab.ipynb` has 22
cells, it expects 21). Pre-existing, on the unrelated Stage-D track, and only visible now because
the `run()` helper exposed how `| tail -1` hides exit statuses — the old-style lines still mask it,
so `run_all_tests.sh` exits 0 despite that failure.

---

## 10. Class-3 routing — `casmi26-v4n-mutator-c3-submit.ipynb`

Both mutator notebooks are scored: PubChem-only **0.399** (no change) and +BIO/AFIX **0.397**
(−0.002). Neither tested the chemistry. §10.1 says why, and the fix is the one the LB reads point
at: fire only where the molecule is judged Class 3, and rank the mutator at the top *there*.

### 10.1 Why neither read moved

**The mutator placed exactly one candidate in the top 25, at rank 20.** With the real list lengths
— v4n ≤ 25 (parsed from `submission.csv`, capped in cell 15) and engine-A 40 — weighted RRF at
`ALPHA_MUT = 0.25, KRR = 3` puts the mutator's candidates at fused ranks **20, 27, 35**, and this
is identical for `MUT_K = 25` and `MUT_K = 40`:

- **`MUT_K` 25 → 40 was a provable no-op.** The extra 15 score below v4n's rank-40 equivalent and
  are cut by `fuse2`'s `n = 40`; the `ICE_UNION_MUT [:25]` cap shipped with it is likewise inert.
  The bio notebook therefore never tested the thing it was built to test at the fusion stage.
- **The upside was capped at ≈ +0.0007.** One candidate at rank 20 pays `1/20`; at a 5% recovery
  rate over 120 gated molecules that is `120 × 0.05 × 0.05 / 400`. Kaggle shows three decimals.
  §8 predicted ≈ +0.001 before shipping — 0.399 → 0.399 is that prediction landing, not a bug.

**The −0.002 is one molecule, lost inside the truth's own formula group.** `plan_v2.md` §1.2: *"Top
40 of the fused list then gets re-ranked inside same-formula groups by `z(RRF) + 1.0·z(ICEBERG) +
1.0·z(GLACIER)`"*. The mutator emits same-formula regioisomers of the target, and the truth, when
found, is at the target formula — so mutator candidates always land in the correct answer's group
and a λ=1+1 forward blend arbitrates. Within-group reranking permutes the positions that group
holds, so `{rank 1 truth, rank 20 mutator}` can return swapped: `(1 − 1/20)/400 = 0.0024`. At three
decimals the two runs are indistinguishable, and this is **not** evidence against BIO+AFIX
chemistry.

It also breaks rule R4 (`plan_v2.md`): *"the forward models resolve isomers within a formula but
must never override retrieval confidence."* A generated structure has no retrieval confidence at
all, so its RRF score is an artefact of the mutator's own `f·z` order. That is the mechanism that
cost −0.034 at λ = 2.0.

**The gate asked the wrong question about the wrong database.** `MUT_FRAC = 0.30` on the weakest
absolute `best_pool_fz` is a *rank* gate, so it fires on exactly 120 molecules whether or not any
is novel; `best_pool_fz` scores train ∪ COCONUT, while Class 3 means absent from *PubChem* —
missing from the pool but present in PubChem is the definition of Class **2**; and it is negative
evidence only, so it cannot separate a novel molecule from an uninformative spectrum.

### 10.2 Measured on `p1_novel` (local, `build/eval_mutator_cohort.py`)

`val_key.pkl` carries `cls ∈ {1,2}` only — **zero Class-3 molecules** — and
`build/mk_stage_d.py:226` says why it never can. `pseudo_c3` (600 molecules, `pool_ban_keys =
[inchikey14]`, every MassSpecGym/ICEBERG-seen structure excluded) is the first honest Class-3 set
available offline. Full detail and the corrections to the original report are in
`validate_mutator.md` §8.

| cut | n | hits | rate | % of cohort |
|---|---|---|---|---|
| all reachable cyclic (**oracle** scaffold *and* formula) | 418 | 72 | 17.2% | 12.0% |
| + scaffold harvestable (`group_size > 1`) | 240 | 25 | 10.4% | **4.2%** |
| + delta within 2 steps | 71 | 20 | 28.2% | 3.3% |

**53.2% of the cohort is alone on its Murcko scaffold**, and 65% of the 72 hits sit in that group,
so they are unreachable however good the screen gets. The harvestable cut is a valid ceiling, not a
loose bound: Murcko scaffolds are identical across a scaffold group, so where a relative exists the
oracle scaffold is exactly what it would have supplied.

The number that drives the design is `q₃` — the chance a *fired* molecule ends with the truth in
the mutator's **top 3**, which is what promotion needs, under a random screen (a floor):

| band | fires on | truth generated | **q₃ \| fired** | candidates (median) |
|---|---|---|---|---|
| depth ≤ 1 | 2.5% | **80.0%** | **31.4%** | 16 |
| depth ≤ 2 | 11.8% | 28.2% | 7.6% | 24 |
| depth ≤ 6 | 40.0% | 10.4% | 2.3% | 75 |

Depth ≤ 1 carries the same expected gain as depth ≤ 2 from **a quarter of the exposure**, so a
wrong gate costs a quarter as much. Hence `c3_gate.MAX_STEPS = 1`, with 2 left as a probe.

### 10.3 Design

1. **`build/c3_gate.py`** — one pure function, shared by the cohort harness and the notebook.
   Fires on: `lib_max < 0.9` (the pipeline's own Class-1 test) **and** `ana_max ≥ 0.50` **and**
   `ana_max − exact_max ≥ 0.15` **and** the analog→target delta realisable in ≤ `MAX_STEPS`
   transforms, with an optional ICE/GL veto. Every term is a similarity in [0,1] or a count —
   nothing reads `f·z`, because a commit run scores a slice of `train.parquet` and no absolute
   `f·z` cut can be calibrated against the hidden test. **Firing on zero molecules is legal**,
   which the rank gate could not express.
2. **A strong mass-shifted analog is the load-bearing term.** It is positive evidence of novelty,
   it is the spectrum-quality control (a bad spectrum cannot produce a high analog similarity), and
   it is the inference-time proxy for "the scaffold is harvestable" — the condition §10.2 shows
   governs everything. `casmi_engine.molecule_channels` already computes it (`ana`, top-200 over
   ±200 Da at `N_ANALOG = 200`); `ana[0][1]` is already a ranker feature.
3. **Analog scaffolds become a third source.** `mutate_sources` takes named sources, so this is
   additive, with two new controls: `contributes_formulas=False` (the analog is at a different mass,
   so **its formula must not become a target** or the budget goes to formulas the precursor rules
   out) and `max_steps={"ana": C3_MAX_STEPS}`.
4. **Two-stage by necessity.** `BASE`, and so `lib_max`, is not built until cell 12 while the
   mutator is cell 9 — so stage 1 uses the engine's own exact-mass library similarity for the
   Class-1 veto and cell 17 re-checks with the real `lib_max` and the forward scores.
5. **Promotion replaces fusion.** The mutator is **absent** from `fuse2`, which removes the §10.1
   displacement mechanism outright, and is spliced into ranks reserved by tier with the existing
   `merge()` helper from cell 15 — the same idiom as `SLOTS_AGG`/`SLOTS_GENTLE`. `sure → [1,2,3]`,
   `likely → [2,3,4]`; molecules the gate declines get **no mutator at all**, so they reproduce the
   0.399 fork exactly. Within the promoted list the forward models order the mutator's own
   candidates against *each other*, which honours R4 — they resolve isomers without overriding
   retrieval confidence.
6. **The diagnostics export touches no locked file.** `eng_runner.py` is a string written by cell 5
   and already loops over `recs`; `casmi/engine.py` and the other 63 MANIFEST-locked files are
   untouched.

### 10.4 Gates

| | gate | threshold | status |
|---|---|---|---|
| **GC.0** | harvestable-scaffold ceiling on `pseudo_c3` | report; proceed if ≥ 3% | **PASS — 4.2%** (25/600) |
| **GC.1** | `q₃ \| fired` in the chosen band | ≥ 0.15 | **PASS — 31.4%** at depth ≤ 1 |
| **GC.2** | pipeline top-1 accuracy on the fired set (`a`) | ≤ 0.05 for slots `[1,2,3]`, else `[2,3,4]` | **open** — needs the engine channels (Colab) |
| **GC.3** | detector precision: fire rate on `pseudo_c3` vs `c2_displace` ∪ C1 | ≥ 3× | **open** — same run as GC.2 |
| **GC.4** | break-even `q₃ > 1.25 · a` | derived | from GC.1/GC.2 |
| **GC.5** | mutator subprocess ≤ 1,200 s in the rerun | — | read from the Kaggle log |
| **GC.6** | the channel can decline | fire count printed; 0 is legal | read from the Kaggle log |

GC.2 is the one that still matters. A reserved rank is free on a true Class-3 molecule — its
reciprocal rank is already 0 — so the entire cost of this change is misfires, and `a` is what
prices them. `build/compare_routers.py` tabulates the exposure:

| router | fires | slots | a = 0.02 | a = 0.10 | a = 0.30 |
|---|---|---|---|---|---|
| absolute `best_pool_fz < 300` (bio2) | 49 | 5 | −0.0020 | −0.0102 | −0.0306 |
| `c3_gate`, depth ≤ 1 | 10 | 3 | −0.0004 | −0.0019 | −0.0056 |
| `c3_gate`, depth ≤ 2 | 47 | 3 | −0.0018 | −0.0088 | −0.0264 |

against gains of **+0.0048** (depth ≤ 1) and **+0.0054** (depth ≤ 2). So depth ≤ 1 is net positive
across the whole plausible range of `a`, while depth ≤ 2 turns negative somewhere above `a ≈ 0.06`.

Counters to read on the commit, in this order, before the score:

```
mutator scaffold sources: ... | Class-3 gate at depth <= N
C3 SIGNALS (percentiles 1/10/25/50/75/90/99): ana_max / lib_exact / margin
  ... would pass ana_max alone: N of M | and the margin too: N
C3 GATE: fired on N of 400 | sure=N likely=N declined=N
C3 GATE REASONS: <the term that vetoed, as a funnel>
MUT SOURCES: scaffolds pc=N eng=N ana=N | ... | pairs refused as too deep=N
C3 ROUTER: promoted N molecules {tiers} | forward-sim reordered N | suppressed on N
MUTATOR IN FINAL TOP-25: N candidates across M molecules
```

If `C3 GATE` fired on 0, the LB number says nothing about Class-3 chemistry and must not be read as
if it did — but unlike the rank gate, 0 is now a legitimate answer rather than a failure.

**`C3 SIGNALS` exists because the thresholds are the weakest part of this change.** `ANA_MIN` and
`MARGIN_MIN` are guesses until GC.3: nothing locally reveals the scale of a mass-shifted analog
similarity, and a threshold set too high fires on nothing and burns a submission without saying so.
The percentile block makes one run sufficient to set them — read it *before* the score, and if
`would pass ana_max alone` is 0 or near 400, move `C3_ANA_MIN` / `C3_MARGIN_MIN` (both environment
variables, no code change) rather than concluding anything about the chemistry.

### 10.5 A second notebook exists for the same idea

`casmi-v4n-mutator-bio2-submit.ipynb` (built in parallel) routes on `best_pool_fz < 300.0` into
ranks 1–5. It independently reaches two of the same conclusions — the mutator must leave the
retrieval fusion, and promotion belongs in reserved ranks — but its threshold is an **absolute
`f·z` cut annotated as calibrated against `pc_lists_bio.json`, which holds exactly 12 molecules**:
the `SMOKE_N = 12` commit artefact, scoring a slice of `train.parquet`, so every molecule in it is
a training structure. `build/compare_routers.py` quantifies the consequence: the median
`best_pool_fz` moves **+172.3** between that file and the real 400-molecule list, and shifting the
distribution by ±300 swings the fire count between **16 and 147 molecules**. At 147 × 5 reserved
ranks a 30% retrieval top-1 rate would cost ≈ −0.09.

Either fix removes the scale dependence: make the cut a **quantile of the current run's own
distribution** (one line), or gate on scale-free quantities as `c3_gate` does.

### 10.6 Files

| file | change |
|---|---|
| `build/c3_gate.py` | **new** — the detector as one pure function; thresholds env-overridable |
| `build/eval_mutator_cohort.py` | **new** — reproduces the oracle number, then the production cuts and `q₃`; `c2_displace`/`ts_dec` as controls |
| `build/compare_routers.py` | **new** — scale-sensitivity and exposure comparison of the two routers |
| `build/delta_m_mutator.py` | `contributes_formulas` and per-source `max_steps`; `pairs_too_deep` counter |
| `build/mut_runner.py` | `MUT_C3`/`MUT_ANA` env, the gate in the worker, the analog source, per-molecule `tier`, the gate funnel. **No `MUT_C3` ⇒ the old rank gate, unchanged** |
| `build/patch_v4n_mutator.py` | `--c3`; diagnostics export from the embedded `eng_runner.py`; the router in cell 17 |
| `build/test_c3_gate.py` | **new** — 42 checks: every veto, both tier boundaries, the bad-spectrum case |
| `build/test_delta_m_mutator.py` | +4 tests (93 checks) for the formula opt-out and the depth caps |
| `build/test_mut_runner.py` | +5 subprocess scenarios (45 checks over 14); stale `specs_v4n_update` path fixed |
| `build/test_mut_cell.py` | the c3 variant; env capture widened to `C3_*` |
| `build/test_novel_p3.py` | stale `specs_novel` path fixed — it was failing silently under `\| tail -1` |
| `build/run_all_tests.sh` | `test_c3_gate.py` and the c3 notebook added |

Unchanged: `casmi26-v4n-mutator-submit.ipynb`, `casmi26-v4n-mutator-bio-submit.ipynb`,
`casmi/engine.py` and the 63 other MANIFEST-locked files, `ALPHA`, `KRR`, `TOPN`, `LIB_TAU`,
`REL_TH`, forward λ.
