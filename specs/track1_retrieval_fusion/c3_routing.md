# Class-3 routing for the ΔM mutator — plan, measurements and code map

> **Read this first** if you are picking up the mutator channel. It is self-contained: the problem,
> what was measured, why the design is what it is, where every piece of code lives, how to
> reproduce every number, and what is still open.
>
> **Status 2026-10-02.** Built and locally verified. `casmi26-v4n-mutator-c3-submit.ipynb` is ready
> to upload. Two gates (GC.2, GC.3) need a Colab/Kaggle run and are open.
>
> Companion documents: `roadmap.md` §10 (same content, in roadmap form), `validate_mutator.md` §8
> (corrections to the first mutator validation), `plan_v2.md` (the R1–R5 fusion rules this obeys),
> `EXPERIMENTS.md` §1 rows 22–23 and §3 (the ledger).

---

## 1. Where this started

| notebook | scaffold source | LB | vs 0.399 |
|---|---|---|---|
| `casmi26-v4n-engine-fusion-union-lb-0-399.ipynb` | — (SOTA base) | **0.399** | — |
| `casmi26-v4n-mutator-submit.ipynb` | PubChem top-25 | 0.399 | **0.000** |
| `casmi26-v4n-mutator-bio-submit.ipynb` | + engine-A top-40 (BIO + AFIX) | 0.397 | **−0.002** |

Neither mutator submission tested the chemistry it was built to test. Both outcomes are explained
by the fusion arithmetic, not by the palette, the scaffolds or the screen. §2 is that explanation;
everything after it follows from it.

**Both shipped notebooks must stay byte-unchanged** — they are on the leaderboard, so the two
scaffold sources stay separable. Regenerating them would re-embed the modified modules; only
generate the `--c3` variant.

---

## 2. Why neither read moved

### 2.1 The mutator could place exactly one candidate in the top 25, at rank 20

`fuse2` is weighted RRF: v4n `1.0/(KRR+r)`, engine-A `ALPHA/(KRR+r)`, mutator `ALPHA_MUT/(KRR+r)`
with `ALPHA=0.6, ALPHA_MUT=0.25, KRR=3.0`. The v4n list is **≤25 long** — it is parsed out of
`submission.csv`, which cell 15 caps at 25 — and engine-A's is 40. Sorting all three and cutting at
`fuse2`'s `n=40`:

| | fused ranks the mutator occupies | inside the pre-rerank top 25 |
|---|---|---|
| `MUT_K = 25` | 20, 27, 35 | **1 candidate, rank 20** |
| `MUT_K = 40` | 20, 27, 35 | **1 candidate, rank 20** |

Two consequences, both load-bearing:

- **`MUT_K` 25 → 40 was a provable no-op.** The extra 15 candidates score below v4n's rank-40
  equivalent (the top-40 cut score is 0.0370) and are discarded by `n=40`. The `ICE_UNION_MUT [:25]`
  cap that shipped alongside it is inert for the same reason. *The bio notebook's headline change
  could not do anything at the fusion stage.*
- **The upside was capped at ≈ +0.0007.** One candidate at rank 20 pays `1/20 = 0.05` when correct.
  Over 120 gated molecules at the measured 5% recovery rate: `120 × 0.05 × 0.05 / 400 = +0.00075`.
  Kaggle displays three decimals. `roadmap.md` §8 predicted ≈+0.001 *before* shipping — so
  0.399 → 0.399 is a prediction landing, not a bug.

Reproduce: `python build/compare_routers.py` prints the surrounding arithmetic, or inline —

```python
A, K, AM = 0.6, 3.0, 0.25
sc  = [(1.0/(K+r), 'v4n', r) for r in range(1, 26)]
sc += [(A  /(K+r), 'eng', r) for r in range(1, 41)]
sc += [(AM /(K+r), 'mut', r) for r in range(1, 41)]
sc.sort(key=lambda x: -x[0])
print([(i+1, t[2]) for i, t in enumerate(sc[:25]) if t[1] == 'mut'])   # -> [(20, 1)]
```

### 2.2 The −0.002 is one molecule, lost inside the truth's own formula group

`plan_v2.md` §1.2 states the pipeline's own semantics: *"Top 40 of the fused list then gets
re-ranked inside same-formula groups by `z(RRF) + 1.0·z(ICEBERG) + 1.0·z(GLACIER)` → top 25."*

The mutator emits **same-formula regioisomers of the target**, and the truth, when found, is also at
the target formula. So mutator candidates are *always* placed in the correct answer's formula group,
and a λ=1+1 forward-simulation blend arbitrates between them. Within-group reranking permutes the
positions that group already occupies, so a group holding `{rank 1 = truth, rank 20 = mutator}` can
come back as `{rank 1 = mutator, rank 20 = truth}`.

Cost of exactly one such event: `(1 − 1/20)/400 = 0.0024` → 0.399 → 0.397. **The observed drop is
one molecule.** At three decimals the two runs are statistically indistinguishable; this is *not*
evidence that BIO+AFIX scaffolds are worse chemistry.

It also violates the lineage's own rule **R4** (`plan_v2.md`): *"The forward models resolve isomers
within a formula but must never override retrieval confidence."* A generated structure has **no**
retrieval confidence — it was never in any pool — so its RRF score is an artefact of the mutator's
internal `f·z` ordering. Letting ICE/GL arbitrate between it and a retrieved candidate is the same
failure mode that cost −0.034 at forward λ=2.0.

### 2.3 The gate asked the wrong question, about the wrong database

```python
# build/mut_runner.py, before this change
scored.sort()                                      # ascending best_pool_fz
fire = [m for _, m in scored[:int(0.30 * len(scored))]]
```

1. **A rank gate always fires on exactly 120 of 400**, whether or not any molecule is novel. It
   cannot express "nothing qualifies today".
2. **`best_pool_fz` scores train ∪ COCONUT.** Class 3 means absent from **PubChem**. Missing from
   the pool but present in PubChem is the textbook definition of Class **2**, which retrieval
   already handles at MRR ~0.50–0.65 (`baseline_intro.txt` §6). The gate preferentially fired on
   Class 2.
3. **It is negative evidence only**, so it cannot separate a novel molecule from an uninformative
   spectrum: a sparse spectrum depresses every `f·z` in the pipeline. Those molecules are also
   exactly where the ICE/GL blend is least reliable — so the channel injected candidates whose only
   evidence was a blend it had just made untrustworthy.

**The constraint that forced the original design**, and which any replacement must respect:
`mut_runner`'s docstring chose a rank gate because *"the f·z scale is not comparable between the
visible dummy test and the hidden test"*. A commit run scores a slice of `train.parquet`
(`casmi-saved-outputs-are-placeholder`), so **no absolute `f·z` cut can be calibrated in advance.**

---

## 3. The insight the design rests on

> **If a molecule is genuinely Class 3, its truth is absent from every retrieval list, so its
> reciprocal rank is already 0. Reserving top ranks for generated candidates on *that* molecule
> costs nothing.**

A confident Class-3 call is what converts the mutator from a two-sided bet into a one-sided one. It
removes the need for rank-protection machinery: the entire cost of the change becomes *misfires*.

That reframes the problem usefully. Writing `a` for the pipeline's top-1 accuracy on the fired set
and `q₃` for the chance a fired molecule ends with the truth in the mutator's top 3:

- loss ≈ `a × (1 − 1/(slots+1))` per fired molecule
- gain ≈ `q₃ × mean(1/rank over the reserved slots)`
- **break-even: `q₃ > 1.25 · a`** for three slots

Almost all displacement risk sits in **slot 1**: a retrieved truth at rank 1 pushed to rank 4 loses
0.75 RR, while every deeper shift costs an order of magnitude less. Hence two tiers (§5).

---

## 4. What was measured, and on what

### 4.1 The cohort — and why `val_scripts` could not do this

`val_scripts/val_key.pkl` holds 1,145 molecules with `cohort ∈ {C1_npx, C1_xlib, C2, C2_hard}` and
**`cls ∈ {1, 2}` only — zero Class-3 molecules.** `build/mk_stage_d.py:226` says why it never can:

> *"every val truth comes from `train.parquet`, which is inside the pool, so pool presence is ~1.0
> by construction. Real Class-3 test molecules are not in the pool at all and score 0 regardless of
> model quality. Val MRR will always read above the leaderboard; the gap is the pool miss rate, and
> nothing here can measure it."*

`p1_novel/pseudo_c3.parquet` (built by Track 2, Phase 1) is the first honest offline Class-3 set:
600 natural products, `pool_ban_keys = [inchikey14]` so the truth is banned from candidate
retrieval, and every MassSpecGym / ICEBERG-seen structure excluded. `c2_displace.parquet` (400) and
`ts_dec.parquet` (155) serve as controls.

**Trap**: if you measure mutator recall on a val cohort, the novelty filter
(`known = pc_keys ∪ eng_keys ∪ pool_keys`) excludes the truth because it *is* in the pool, so recall
reads 0 by construction. Substitute `ban_keys` for `pool_keys` for any such measurement.

### 4.2 The headline 17.2% is an oracle number

`scratch/eval_mutator_pseudo_c3.py` (the original validation, by another agent) seeds from
`row["scaffold"]` and targets `elem_vec(truth_mol)` — **the truth's own Murcko scaffold and its own
formula.** Those are GM.0a/GM.0b conditions on a better cohort (GM.0b read 21.6%, so 17.2% is a
useful independent corroboration). It is not an end-to-end rate: in production the scaffold must be
*harvested from a retrieved hit*.

Joining `group_size` against `dec_split.parquet` (194,668 structures; the two agree on all 600 rows):

- **319 of 600 (53.2%) are the only structure in that universe carrying their Murcko scaffold.** No
  retrieval list can supply a scaffold that exists nowhere else.
- **47 of the 72 exact hits (65%) fall in that group** and are unobtainable in production.

| cut | n | hits | rate | % of cohort |
|---|---|---|---|---|
| all formula-reachable cyclic *(oracle scaffold **and** formula)* | 418 | 72 | 17.2% | 12.0% |
| **+ scaffold harvestable** (`group_size > 1`) | 240 | 25 | 10.4% | **4.2%** |
| **+ delta within 2 transforms** | 71 | 20 | 28.2% | 3.3% |

The harvestable cut is a **valid ceiling, not a loose bound**: Murcko scaffolds are identical across
a scaffold group by construction, so wherever a relative exists the oracle scaffold is exactly what
that relative would have supplied. Nothing is given away for `group_size > 1`.

Recovery is *higher* for unique-scaffold molecules (26.4%, 47/178) than for harvestable ones (10.4%,
25/240). That inversion is the tell: a molecule alone on its scaffold tends to have a scaffold that
is nearly the whole molecule — 11 of the 14 depth-0 hits are unique-scaffold. The oracle gives away
most of the answer precisely where the result cannot be used.

### 4.3 The number that actually drove the design: candidate dilution

Promotion into reserved ranks needs the truth in the mutator's **top 3**, not merely somewhere in
its list — and the `f·z` screen that has to find it recovered only 21.6% under oracle conditions
(GM.0b). `f·z` needs FPNet logits and cannot be computed locally, so `q₃` below is the probability a
*random* screen lands a recovered truth in the top 3 (`3/n_candidates`) — a **floor**, since `f·z`
should beat random.

| band (harvestable +) | fires on | truth generated | **q₃ \| fired** | candidates (median) |
|---|---|---|---|---|
| **depth ≤ 1** | **2.5%** | **80.0%** | **31.4%** | **16** |
| depth ≤ 2 | 11.8% | 28.2% | 7.6% | 24 |
| depth ≤ 6 | 40.0% | 10.4% | 2.3% | 75 |

Per-depth detail on `pseudo_c3` (oracle conditions, all 418 reachable):

| depth | n | hits | rate | candidates (median) |
|---|---|---|---|---|
| 0 | 14 | 14 | 100.0% | 1 |
| 1 | 39 | 27 | 69.2% | 9 |
| 2 | 98 | 21 | 21.4% | 100 |
| 3 | 93 | 7 | 7.5% | 100 |
| 4 | 95 | 3 | 3.2% | 119 |
| 5 | 51 | 0 | 0.0% | 100 |
| 6 | 28 | 0 | 0.0% | 0 |

**Depth ≤ 1 carries the same expected gain as depth ≤ 2 from a quarter of the exposure**, so a wrong
gate costs a quarter as much. That is why `c3_gate.MAX_STEPS = 1`, with 2 retained as a probe.

### 4.4 Controls

| cohort | n | reachable | oracle hits | production-viable (harvestable ∧ depth ≤ 2) |
|---|---|---|---|---|
| `pseudo_c3` | 600 | 418 (88.7% of cyclic) | 72 (17.2%) | 20 → **3.3%** |
| `c2_displace` | 400 | 272 (79.8%) | 41 (15.1%) | 11 → 2.8% |
| `ts_dec` | 155 | 123 (82.6%) | 27 (22.0%) | 17 → 11.0% |

Read these carefully. They measure **mutator recovery**, not gate precision — the gate needs the
engine's channels, so GC.3 is still open. `pseudo_c3` and `c2_displace` land within half a point of
each other, i.e. the *generator* is not novelty-specific; specificity has to come from the gate.
`ts_dec` is higher largely because it is an easier, more homogeneous cohort (96.1% cyclic vs 78.5%,
100% CE known, 2.67 spectra/molecule) and is ICEBERG-seen — `phase3_plan.md` forbids using it for
any ICEBERG/merged-list metric.

### 4.5 Other facts worth carrying forward

- **Acyclic wall: 21.5%** of `pseudo_c3` (129/600) has no Murcko scaffold, accounting for **70.9% of
  all misses**. `roadmap.md` §8 previously assumed 9.4%; that has been corrected.
- **Runtime**: 434 ms/molecule at `beam=100, cap=300`, median 100 candidates. Production runs
  `BEAM=200, MAX_PRODUCTS=8000` — oversized for the payoff.

---

## 5. The design

### 5.1 The gate — `build/c3_gate.py`

One pure function, no RDKit dependency, shared by the cohort harness and the notebook. It fires when
**all** of:

| # | term | source | why |
|---|---|---|---|
| 1 | `lib_max < LIB_TAU` (0.90) | `BASE[m][2]` / engine `lib` | rules out Class 1; the pipeline's own Class-1 test |
| 2 | `ana_max ≥ ANA_MIN` (0.50) | `rec['ana'][0][1]` | the spectral family is recognised at all |
| 3 | `ana_max − exact_max ≥ MARGIN_MIN` (0.15) | `max(rec['lib'])` | the analog beats everything at this exact mass |
| 4 | analog→target delta realisable in ≤ `MAX_STEPS` (**1**) | `build_reach()` | §4.3: the band where the channel can be ranked |
| 5 | *(optional)* `ice_max < ICE_MAX` (0.60) | `ICE_SCORES` | no retrieved candidate already reproduces the spectrum |

**Term 2 is the load-bearing one** and does three jobs at once:

- *positive* novelty evidence — "I recognise this family at a mass offset, but nothing at this exact
  mass explains it" — instead of §2.3's negative evidence;
- the **spectrum-quality control**, because a bad spectrum cannot produce a *high* analog
  similarity. That is what separates "novel" from "uninformative", which the old gate could not do;
- the inference-time proxy for **"the scaffold is harvestable"** — a strong analog *is* the scaffold
  relative, and finding it is the evidence that one exists. §4.2 shows that is the condition
  governing everything.

Every term is a similarity in [0,1] or a count. **Nothing reads `f·z`**, which is what makes
absolute thresholds legal here where §2.3 forbids them. Thresholds are environment-overridable
(`C3_LIB_TAU`, `C3_ANA_MIN`, `C3_ANA_SURE`, `C3_MARGIN_MIN`, `C3_MARGIN_SURE`, `C3_MAX_STEPS`,
`C3_ICE_MAX`) so they can move without a code edit.

Terms are evaluated cheapest-first and the **first failure wins**, so the `reasons` counter reads as
a funnel rather than overlapping counts. **Firing on zero molecules is a legal outcome.**

### 5.2 Two stages, because of cell order

`BASE` — and therefore `lib_max` — is not built until **cell 12**, while the mutator is **cell 9**.
So:

- **Cell 9 (generate)**: terms 1–4, with the engine's own exact-mass library similarity standing in
  for `lib_max`. The expensive half stays where it is.
- **Cell 17 (promote)**: the decision that carries the risk, with the real `lib_max` and the ICE/GL
  agreement available.

No cell reordering, so `roadmap.md` §2 still holds.

### 5.3 The analog as a third scaffold source

```python
srcs = [("pc", pc_hits), ("eng", eng_hits), ("ana", analog_smiles[:MUT_ANA])]
```

`mutate_sources` already takes named sources and walks them round-robin, so this is additive. Two
new controls were required:

- **`contributes_formulas=[True, True, False]`** — the analog sits in a ±200 Da window, so its
  molecules are at the **wrong mass** and their formulas are not target hypotheses. Letting them in
  would spend the whole budget growing scaffolds toward formulas the precursor mass rules out. This
  is the subtlest correctness point in the change.
- **`max_steps={"ana": C3_MAX_STEPS}`** — per-source depth cap, counted as `pairs_too_deep`.

This is the project's own standing Class-3 recommendation, previously unimplemented
(`baseline_intro.txt` §6: *"In-silico scaffold propagation: Propagating Murcko scaffolds from
mass-shifted spectral analogues"*).

### 5.4 Promotion replaces fusion

The mutator is **absent from `fuse2`**, which removes §2.2's displacement mechanism outright, and is
spliced into reserved ranks by tier using the **existing `merge()` helper from cell 15** — the same
idiom the PubChem channel already uses with `SLOTS_AGG = [2,4,6,8,10]` / `SLOTS_GENTLE =
[4,8,12,16,20]`.

| tier | condition | slots |
|---|---|---|
| `sure` | all terms, `ana_max ≥ 0.70` **and** margin ≥ 0.30 | `[1, 2, 3]` |
| `likely` | all terms | `[2, 3, 4]` — rank 1 untouched |
| declined | — | **none**: reproduces the 0.399 fork exactly |

Inside the promoted list the forward models order the mutator's own candidates **against each
other**, which honours R4 — they resolve isomers without overriding retrieval confidence. The
reorder is self-disabling: if neither key space matches `ICE_SCORES`, all scores read 0 and the
`f·z` order is kept.

**`sure` takes slot 1 by default.** That is the one judgement call not fully backed by measurement:
§7 shows it is net-positive across the plausible range of `a`, but `a` itself is unmeasured. Point
`C3_SLOTS['sure']` at the `likely` pattern in cell 17 to hedge — it is one line.

### 5.5 Nothing locked is touched

`eng_runner.py` is a *string written by cell 5* and already loops over `recs`, so the diagnostics
export is notebook-local. `casmi/engine.py` and the other 63 MANIFEST-locked files in
`casmi26-v4b-models` are untouched. No new dataset attachment is needed.

---

## 6. Code map

| file | role | what to edit, and when |
|---|---|---|
| `build/c3_gate.py` | **the gate.** `classify()`, `min_steps()`, `delta_of()`, `heavy_negative()`, `SLOTS` | thresholds; prefer the `C3_*` env vars over editing |
| `build/eval_mutator_cohort.py` | reproduces §4: the oracle number, then the production cuts and `q₃`; controls | add a cohort; change `--beam` / `--cap` |
| `build/compare_routers.py` | §7: scale-sensitivity and exposure of the two routers. Read-only report | — |
| `build/delta_m_mutator.py` | generator. `mutate_sources(..., contributes_formulas, max_steps)`; `build_reach`; palette | palette, depth, beam |
| `build/mut_runner.py` | subprocess driver. `MUT_C3` / `MUT_ANA` env, the gate in `mutate_one`, the `C3 SIGNALS` calibration print, the funnel | gate wiring, diagnostics |
| `build/patch_v4n_mutator.py` | **the notebook generator.** `--bio`, `--c3`; `mut_cell(variant)`; the cell-17 router; `esc()` for patching embedded literals | any notebook change goes here, never in the `.ipynb` |
| `build/test_c3_gate.py` | 42 checks: every veto, both tier boundaries, the bad-spectrum case, env overrides | — |
| `build/test_delta_m_mutator.py` | 93 checks incl. single-source equivalence, the limit trap, formula opt-out, depth caps | — |
| `build/test_mut_runner.py` | real-subprocess test against a mocked `casmi`; 45 checks over 14 scenarios, 5 of them C3 | — |
| `build/test_mut_cell.py` | executes the notebook's actual mutator cell with `subprocess.run` stubbed; runs against all three notebooks | — |
| `build/run_all_tests.sh` | the suite. `run()` reports and stops; `soft()` reports and continues | — |

**The notebook is generated, never hand-edited:**

```sh
python build/patch_v4n_mutator.py           # -> casmi26-v4n-mutator-submit.ipynb      (do not run: shipped)
python build/patch_v4n_mutator.py --bio     # -> casmi26-v4n-mutator-bio-submit.ipynb  (do not run: shipped)
python build/patch_v4n_mutator.py --c3      # -> casmi26-v4n-mutator-c3-submit.ipynb
```

`--c3` differs from `--bio` in exactly four cells: **5** (diagnostics export), **8** (markdown),
**9** (gate, analog source, embedded `c3_gate.py`), **17** (router). Everything else is identical.

### Data flow

```
cell 5   engine A  ->  eng_lists.json           (top-40 structures)
                   ->  eng_lists.json.c3.json   (NEW: target, lib_exact, ana[(smiles, mass, sim)])
cell 7   pc_runner ->  pc_lists.json, pc_z.npz  (FPNet logits for the f.z screen)
cell 9   mut_runner subprocess:
           c3_gate.classify per molecule -> tier or decline
           mutate_sources([pc, eng, ana]) -> two-pass f.z screen -> MUT_K kept
         -> mut_lists.json {mid: {smiles, keys, formula, fz, src, tier}}
cell 14  ICE/GL score the union, mutator list capped at [:25]
cell 17  fuse2(v4n, engine)  [mutator deliberately absent]
         -> ICE/GL rerank inside formula groups
         -> if tier: merge(vs, ..., mutator list, C3_SLOTS[tier])
```

---

## 7. The parallel router, and how they compare

`casmi-v4n-mutator-bio2-submit.ipynb` was built in parallel by another agent for the same idea. It
independently reaches two of the same conclusions — the mutator must leave the retrieval fusion, and
promotion belongs in reserved ranks — and uses `merge()` the same way. **The structure is agreed;
only the gate differs.** It fires on `best_pool_fz < 300.0` into ranks 1–5, annotated "calibrated:
known mols >= 456.0, novel mols <= 265.8".

That calibration cannot hold. `pc_lists_bio.json` has exactly **12 molecules** — the `SMOKE_N = 12`
commit artefact — and a commit run scores a slice of `train.parquet`, so every molecule in it is a
*training* structure and the "novel" half of the split is not measuring novel molecules.

`python build/compare_routers.py`:

| source | n | median `best_pool_fz` | below 300 |
|---|---|---|---|
| `pc_lists_bio.json` (commit) | 12 | 548.3 | 3 (25.0%) |
| `pc_lists.json` (real 400) | 400 | **720.6** | 49 (12.2%) |

The median moves **+172.3** between them, against a threshold of 300. Shifting the distribution by
±300 — less than the spread already observed — swings the fire count between **16 and 147
molecules**:

| Δ to every `best_pool_fz` | −300 | −200 | −100 | 0 | +100 | +200 | +300 |
|---|---|---|---|---|---|---|---|
| absolute `< 300` | 147 | 96 | 66 | 49 | 36 | 21 | 16 |
| same quantile (q=0.122) | 49 | 49 | 49 | 49 | 49 | 49 | 49 |
| `c3_gate` | *invariant — it never reads `f·z`* | | | | | | |

Exposure, as worst-case MRR loss (`a` = pipeline top-1 accuracy on the fired set):

| router | fires | slots | a=0.02 | a=0.10 | a=0.30 |
|---|---|---|---|---|---|
| absolute `< 300` (bio2) | 49 | 5 | −0.0020 | −0.0102 | −0.0306 |
| `c3_gate`, depth ≤ 1 | 10 | 3 | −0.0004 | −0.0019 | −0.0056 |
| `c3_gate`, depth ≤ 2 | 47 | 3 | −0.0018 | −0.0088 | −0.0264 |

Against gains of **+0.0048** (depth ≤ 1) and **+0.0054** (depth ≤ 2) from §4.3's `q₃`. So depth ≤ 1
is net positive across the whole plausible range of `a`; depth ≤ 2 turns negative above `a ≈ 0.06`;
147 misfires at 5 slots and `a = 0.30` would cost ≈ −0.09, against the lineage's worst-ever −0.034.

**This comparison cannot say which scores better** — that needs the engine channels. It says the
absolute threshold is the riskier of the two. If it is kept, making the cut a **quantile of the
current run's own distribution** is one line and removes the scale dependence entirely.

---

## 8. Gates

| | gate | threshold | status |
|---|---|---|---|
| **GC.0** | harvestable-scaffold ceiling on `pseudo_c3` | ≥ 3% | **PASS — 4.2%** (25/600) |
| **GC.1** | `q₃ \| fired` in the chosen band | ≥ 0.15 | **PASS — 31.4%** at depth ≤ 1 |
| **GC.2** | pipeline top-1 accuracy on the fired set (`a`) | ≤ 0.05 for slots `[1,2,3]` | **OPEN** — needs the engine channels |
| **GC.3** | gate precision: fire rate on `pseudo_c3` vs `c2_displace` ∪ C1 | ≥ 3× | **OPEN** — same run |
| **GC.4** | break-even `q₃ > 1.25 · a` | derived | from GC.1/GC.2 |
| **GC.5** | mutator subprocess ≤ 1,200 s in the rerun | — | read from the Kaggle log |
| **GC.6** | the channel can decline | fire count printed; 0 legal | read from the Kaggle log |

**GC.2 cannot be measured on `pseudo_c3`** — the truth is banned there, so `a = 0` by construction.
It needs retrieval to run on labelled molecules where the truth *is* findable, i.e. extending
`build/mk_stage_d.py` (which already loads the pool, ban-aware channels, `TRUTH`, `COHORT`, `BAN`
and reports per-cohort MRR) with a fired-set table. **That is the single highest-value piece of
remaining work** and the next thing to build.

### Local verification (all green)

```sh
python build/test_c3_gate.py               # 42 checks
python build/test_delta_m_mutator.py       # 93 checks
python build/test_mut_runner.py            # 45 checks over 14 subprocess scenarios
python build/test_mut_cell.py <notebook>   # 29 checks, run for each of the three notebooks
python build/eval_mutator_cohort.py        # reproduces §4 (~8 min for all three cohorts)
python build/compare_routers.py            # reproduces §7
sh build/run_all_tests.sh                  # the lot
```

`run_all_tests.sh` exits 0 and ends with `SUITE: passed, with known-broken tests still failing:
build/test_stage_d_colab.py` — a pre-existing failure on the Stage-D track
(`casmi_validate_ab_colab.ipynb` has 22 cells, the test expects 21). It is reported through `soft()`
rather than silenced; **do not simply update the expected count**, since that could mask a real
regression in a notebook nobody has examined.

---

## 9. Reading the next LB result

Read these from the Kaggle log **before** the score:

```
mutator scaffold sources: ... | Class-3 gate at depth <= N
C3 SIGNALS (percentiles 1/10/25/50/75/90/99): ana_max / lib_exact / margin
  would pass ana_max alone: N of M | and the margin too: N
C3 GATE: fired on N of 400 | sure=N likely=N declined=N
C3 GATE REASONS: <the term that vetoed, as a funnel>
MUT SOURCES: scaffolds pc=N eng=N ana=N | ... | pairs refused as too deep=N
C3 ROUTER: promoted N molecules {tiers} | forward-sim reordered N | suppressed on N
MUTATOR IN FINAL TOP-25: N candidates across M molecules
```

**`C3 SIGNALS` first.** `ANA_MIN` and `MARGIN_MIN` are the weakest part of this change — nothing
locally reveals the scale of a mass-shifted analog similarity, and a threshold set too high fires on
nothing and burns a submission silently. If `would pass ana_max alone` reads 0 or near 400, **move
`C3_ANA_MIN` / `C3_MARGIN_MIN`** (environment variables, no code change) and conclude nothing about
the chemistry.

Then:

| if | then |
|---|---|
| `C3 GATE` fired on 0 | the thresholds are wrong, not the chemistry. Recalibrate from `C3 SIGNALS`. Unlike the old rank gate, 0 is a legitimate answer |
| fired, `MUTATOR IN FINAL TOP-25` = 0 | the router did not place anything — a plumbing bug. Check `tier` survives into `mut_lists.json` and that `merge()` received a non-empty list |
| fired and placed, LB > 0.399 | keep. Then probe `C3_MAX_STEPS = 2` (one character, cell 9 line 7) and consider `sure → [1,2,3]` if it was hedged |
| fired and placed, LB ≈ 0.399 | the gate is firing on molecules that were already lost. GC.3 is the measurement that explains it |
| fired and placed, LB < 0.399 | misfires dominate: `a` is higher than hoped. Hedge `sure` to `[2,3,4]`, or tighten `C3_ANA_MIN` |

To ship the second band: change `C3_MAX_STEPS = 1` to `2` on line 7 of cell 9. Nothing else.

---

## 10. Traps

1. **Do not regenerate the two shipped notebooks.** The embedded modules changed; regenerating would
   alter files that are already on the leaderboard.
2. **The analog source must not contribute formulas.** It is at a different mass. `_union_formulas`
   has the opt-out; `test_three_sources_is_additive` pins it.
3. **`harvest_scaffolds(smiles, limit)` truncates its input.** Concatenating a 25-hit and a 40-hit
   list and passing one limit silently discards the second source. Harvest per source, then union —
   `test_per_source_limit_trap` pins this.
4. **`mutate_sources` returns early at `cap`/deadline**, so sources are walked round-robin; a
   concatenated order would let the first source claim the whole budget.
5. **Never calibrate a threshold on commit output.** `pc_lists_bio.json`, `mut_lists_bio.json`,
   `eng_lists_bio.json` are all 12-molecule artefacts over training structures.
6. **Build the reach table at full depth (6) even when capping at 1.** `min_steps` needs the full
   table to report the true minimum depth; a shallow table makes a 2-step delta look *unreachable*
   rather than *too deep*, destroying the funnel's meaning.
7. **Another agent works in this repo concurrently.** Result files in `scratch/` have been
   overwritten mid-session. Write intermediates elsewhere and re-check row counts before quoting a
   number.
8. **A failing test can hide behind `| tail -1`** — `set -e` reads *tail's* exit status. Two real
   failures were hidden this way (`test_novel_p3.py`, `test_stage_d_colab.py`). Use `run()`/`soft()`.

---

## 11. If this does not pay

The mutator's absolute ceiling on `pseudo_c3` is **4.2%** and it carries a hard structural limit:
53.2% of Class-3 molecules are alone on their scaffold, so no amount of screen or palette work
reaches them.

The same router then belongs behind the `specs_novel` decoder, which scores **8.17% exact@25 on this
very cohort** against a 0.0% null (`EXPERIMENTS.md` §2C, G3.1 PASS) — nearly 2× the mutator's
ceiling, generated from the DreaMS spectrum embedding with **no scaffold-harvest dependency**, so it
has no exposure to the 53% wall at all.

The two are complementary rather than competing: the mutator is narrow and precise (28.2% on ~12% of
the field), the decoder broad and shallow (8.17% everywhere). The gate, the tiers, the `merge()`
splice and the ICE/GL union entry are all source-agnostic — swapping the source is a change of
inputs, not of architecture.
