# specs_v4n_update — plan v2: the fusion philosophy, and where scaffold mutation fits

*Written 2026-10-01, after auditing `casmi26-v4n-engine-fusion-union-lb-0-399.ipynb` against
`casmi26-v4n-inference (1).ipynb` (0.384), `casmi26-v4h-inference.ipynb` (0.373) and
`enveda-casmi-2026-tier1-patched.ipynb`. Supersedes `plan.md`, which is kept for its chemistry
palette. Read §3 first — it is the only part that generalises.*

---

## 1. What the 0.399 notebook actually changed

The entire v4n pipeline is **byte-identical**. I diffed every cell: old cell 1→new 7 (PubChem
channel), 2→8 (pool/cache), 3→9 (engine+ranker), 4→10 (base lists), 6→13 (gated merge) — all
IDENTICAL. `LIB_TAU, REL_TH = 0.9, 600.0` unchanged. `TOPN = 60` unchanged.

The 0.384 → 0.399 delta is exactly four changes, three of them score-bearing:

| # | change | where | score-bearing |
|---|---|---|---|
| 1 | Commit switch: dummy test → 12 molecules; hidden test detected via `KAGGLE_IS_COMPETITION_RERUN` or a test-signature hash | new cell 3 | no (ops only) |
| 2 | **A second, independent engine** in its own subprocess, writing `{mid: {smiles[40], keys[40]}}` | new cell 5 | **yes** |
| 3 | **`ADDED` (ICE_UNION) block**: the engine's top-40 join the ICEBERG/GLACIER input, so same-formula groups form over *v4n top-60 ∪ engine top-40* | new cell 12 | **yes** |
| 4 | **Weighted RRF fusion + forward re-rank**, reading back the v4n `submission.csv` and overwriting it | new cell 15 | **yes** |
| — | `ICE_BUDGET` 2700→5400, `GL_BUDGET` 2400→4000 | cells 1, 12 | enabling (#3 needs it) |

### 1.1 The second engine is orthogonal on four axes

From the decoded `eng_runner.py`:

```python
E.RANK.W_A = (0.35, 0.55); E.RANK.SEEDS = (0, 1); E.CFG.N_ANALOG = 200
# and a find() override:
#   "v4g ships its own fp_bits.npy (10,226 bits): take prvsiyan's 6,930"
bl = E.blend_scores(S['pv'], S['ours'], 0.88)
```

| axis | v4n | second engine |
|---|---|---|
| candidate pool | train + COCONUT (710,701) | + **ChEBI + LIPID MAPS** (BIO) |
| fingerprint basis | `fp_bits.npy`, **10,226 bits** | prvsiyan's, **6,930 bits** |
| neural FP models | `fpnet_full1`, `fpnet_0/1` | `casmi26-fp-models-v2/fp_*.pt` |
| ranker | 4 LightGBM boosters on 160 `fe_v4` features | two-ranker blend, `W_A=(0.35,0.55)`, 2 seeds, 31 features |
| analog index | engine-internal | **AFIX**: formula/mass index over the library with timsTOF analog reps, `N_ANALOG=200` |

It is an ensemble inside an ensemble: `blend_scores(pv, ours, 0.88)` fuses two score systems
*within* the engine, then RRF at α=0.6 fuses the engine with v4n.

### 1.2 The fusion itself is deliberately dumb

```python
ALPHA, KRR = 0.6, 3.0
sc[k] += 1.0   / (KRR + r)   # v4n list, rank r
sc[k] += ALPHA / (KRR + r)   # engine list, rank r
```

Weighted reciprocal-rank fusion on tautomer-canonical InChIKey-14. **No retraining, no shared
feature schema, no score calibration.** The two systems' scores are not comparable, so it fuses
*ranks*. Top 40 of the fused list then gets re-ranked inside same-formula groups by
`z(RRF) + 1.0·z(ICEBERG) + 1.0·z(GLACIER)` → top 25.

### 1.3 Why the union step is separate from the fusion

The notebook's own explanation, and it is the key insight:

> The fusion however pulls engine candidates into the final list that the base never sent to the
> forward models, so their isomer order would come from retrieval scores alone; scoring the union
> fixes that.

Measured on the v4g base: **0.366 → 0.382 (fusion) → 0.386 (union)**. Fusion is worth +0.016,
scoring the union a further +0.004. On the v4n base the two together give +0.015.

---

## 2. The lineage, with deltas

| version | LB | Δ | what changed | kind |
|---|---|---|---|---|
| two-ranker engine | 0.341 | — | analog propagation + two rankers | base |
| + BIO | 0.350 | **+0.009** | ChEBI + LIPID MAPS into the pool | **add a source** |
| v13 | 0.341 | — | two-ranker blend + PubChemLite | add a source |
| v14 | 0.326 | **−0.015** | tuned on local GNPS/Enveda val | **tune one view** |
| v4g (v17) | 0.366 | — | FPNet A + 160 feat + 4 LGBM + PubChem + ICEBERG | add views |
| v4h | 0.373 | **+0.007** | GLACIER as a third scorer, λ=0.5 | **add a view** |
| v4i | — | ~0 | PubChem pass-1 depth 1000→5000 | **deepen one view** |
| v4l | 0.380 | **+0.007** | ICE/GL λ 0.5 → 1.0 | re-weight views |
| v4m | 0.378 | **−0.002** | engine FP bank `fpnet_0` → `fpnet_full1` | **replace a view** |
| v4n | 0.384 | +0.004 | v4i + v4l + v4m combined | — |
| v4g + fusion | 0.382 | **+0.016** | RRF with the second engine | **add a source** |
| v4g + fusion + union | 0.386 | **+0.004** | engine candidates get ICE/GL too | **extend scorers** |
| **v4n + fusion + union** | **0.399** | **+0.015** | same recipe on the stronger base | **add a source** |
| — forward λ 2.0 | 0.352 | **−0.034** | "too much forward weight destroys confident library matches" | **over-weight one view** |
| — `TOPN` 100 vs 60 | 0.386 | **0.000** | same-formula window depth | **deepen one view** |

---

## 3. The recurring philosophy

The table splits cleanly in two, with no exceptions:

**Everything that gained added an orthogonal source or scorer, or combined existing ones better.**
BIO +0.009 (new databases). GLACIER +0.007 (a transformer beside a GNN). λ 0.5→1.0 +0.007 (let the
independent scorers speak). Fusion +0.016 (a whole second retrieval system). Union +0.004 (let every
scorer see every candidate).

**Everything that tuned, deepened, or replaced a single view failed.** PubChem depth 1000→5000 ≈ 0.
`TOPN` 60→100 = exactly 0. FPNet backbone swap −0.002. Forward weight 2.0 −0.034. Local-val tuning
−0.015.

Five rules follow, and they are the whole plan:

**R1 — Orthogonality, not quality, is what pays.** The second engine is not *better* than v4n; it is
*different* (different pool, bit basis, FP models, ranker, analog index). Different systems make
different mistakes, so rank fusion beats either. Corollary: before building anything, ask "which
axis of the existing system does this not share?" An improvement that shares every axis is the
`TOPN 100` experiment: exactly zero.

**R2 — Every candidate must be seen by every scorer.** This is the union step. Adding a candidate
source without extending the downstream scorers to cover it leaves most of the gain on the table
(+0.016 → +0.020 only once the forward models saw the new candidates). Generalised: a new source
needs *both* an entry into the merge *and* an entry into ICE/GL.

**R3 — Combine at the rank level.** RRF on InChIKey-14 needs no retraining, no shared feature
schema, no comparable scores. This is why the lineage can fuse systems with 6,930-bit and
10,226-bit fingerprint bases. It also sidesteps the coupling trap recorded in
`casmi-ranker-fpnet-coupling` — v4m's −0.002 is that trap, and RRF never touches it.

**R4 — Weight new and physical signals conservatively; the risk is asymmetric.** Forward λ: 0.5 →
1.0 gains +0.007, 1.0 → 2.0 loses −0.034. The forward models resolve isomers *within* a formula but
must never override retrieval confidence. Engine RRF weight is α=0.6, not 1.0. Nothing in the
lineage was ever given full weight on its first appearance.

**R5 — Integrate additively, with a fallback to the previous version.** `if ENG:`,
`except → 'ENGINE FAILED -> v4g lists kept'`, `GL FAILED -> v4g behaviour`, subprocess isolation so
each system keeps its own `casmi` snapshot. A new source can only add score, never risk the base.
This is why a 0.384 base could be extended in four cells.

**Compute corollary:** spend compute on *breadth of scoring*, not depth of one list. The budgets went
2700/4000 → 5400/4000 only because the union added candidates. Depth alone (`TOPN` 100) bought
nothing.

---

## 4. Where we stand — we already own the orthogonal engine

`enveda-casmi-2026-tier1-patched.ipynb` and `enveda-casmi-2026-phase1-enhanced-baseline.ipynb` are
**the same family as the second engine in the 0.399 recipe** — the analog-propagation line, with
`N_ANALOG`, `SIM_POWER`, `W1_PRIORS`, `SEEDS`, `N_FEAT = 31`, two rankers. The 0.399 notebook runs
prvsiyan's version of this family with two settings we have turned the other way:

| our setting | 0.399's setting | measured |
|---|---|---|
| `USE_BIO_DB = False` — *"disabled by default to prevent decoy dilution"* | **BIO on** (ChEBI + LIPID MAPS) | **+0.009** |
| `N_ANALOG = 100` | `N_ANALOG = 200` | part of 0.341 → 0.350 |

Our line also carries three levers the 0.399 engine does not obviously use: `PPM_CENTER = 1.45`,
`STRUCT_MASS = True` (+0.006 measured, cf. `casmi-query-self-retrieval`) and `MAX_PEAKS = 64`
(+0.003). Under R1 those are *additional orthogonality on the analog axis*, which makes our engine a
better fusion partner than the public one — not a competitor to v4n.

And `specs_novel` (DreaMS embeddings + SELFIES decoder, P3) is orthogonal on an axis **no source in
the lineage touches**: learned spectral embeddings and generative decoding rather than
fingerprint retrieval. Under R1 that is the most valuable remaining axis, which reframes it from
"risky de novo project" to "the next fusion partner".

---

## 5. Plan

### Phase 0 — adopt the 0.399 recipe as the base (do first)

Fork `casmi26-v4n-engine-fusion-union-lb-0-399.ipynb` unchanged, confirm it reproduces, and keep the
commit switch. Everything below is an additive source on top of it (R5). Do **not** re-litigate
`LIB_TAU`/`REL_TH` yet: they are untouched at 0.9/600 in a 0.399 notebook, so they are not what is
blocking progress. (The gate analysis in `plan.md`'s predecessor relied on version-save outputs —
see `casmi-saved-outputs-are-placeholder`.)

**Retracted from the earlier draft:** `TOPN` 60 → 120. The 0.399 notebook reports `TOPN` 100 vs 60 =
**0.386 vs 0.386, no change**. Depth on one list is a dead lever (R1).

### Phase 1 — swap in our engine as the fusion partner

Our tier1/phase1 engine already implements the analog family, plus `PPM_CENTER`, `STRUCT_MASS` and
`MAX_PEAKS = 64`. Replace `eng_runner.py`'s call with ours, exposing the same interface
(`{mid: {smiles[40], keys[40]}}`), and set `USE_BIO_DB = True`, `N_ANALOG = 200`.

Probe order, one LB submission each:
1. 0.399 as-is (baseline reproduction).
2. Our engine in place of the public one, same α=0.6, K=3.
3. Whichever wins, α ∈ {0.4, 0.8} — a two-point sweep, not a search.

Keep both engines available: if ours and the public one are themselves orthogonal (different
`N_ANALOG`, different mass indexing), **fusing all three lists** is the R1-consistent move, with
weights 1.0 / 0.6 / 0.4. That is the cheapest untested gain on the board.

### Phase 2 — scaffold mutation as the third source

The mutator is not an engine change. Under R1–R5 it is a **third candidate source**, and the recipe
for adding one is now fully specified by the 0.399 diff. Keep `plan.md`'s biogenetic palette; replace
its integration story entirely.

**2a. Why it fits the architecture unusually well.** The mutator emits same-formula regioisomers of a
harvested scaffold. ICEBERG and GLACIER re-rank *inside same-formula groups*. So the mutator
manufactures exactly the kind of ambiguity the forward models exist to resolve — the two components
are matched by construction. No other source has that property.

**2b. Why R2 is non-negotiable here.** Generated structures have **no honest retrieval score** — they
were never in any pool, so RRF rank is the only thing ordering them, and among a generator's own
outputs that ordering is near-arbitrary. The forward models are the *only* signal that can order
them. A mutator fused without the union step is a mutator with no ranking signal at all. The
`ADDED` block pattern in cell 12 is therefore mandatory, not an optimisation.

**2c. Interface (R5).** `build/mutator_runner.py`, own subprocess, writes
`{mid: {smiles: [...], keys: [...]}}` — byte-identical in shape to `eng_lists.json` and
`pc_lists.json`. Wrapped in `if MUT:` / `except → fused list kept`. No `casmi/engine.py` edit, so the
`MANIFEST.json` SHA-256 check in cell 1 stays valid. (`plan.md`'s "wire into `BASE[mid]` and
featurize with `fe_v4`" is impossible — `fe_v4` runs inside `V.run()`, `BASE` is post-ranking, and
the engine code is hash-locked.)

**2d. Selection, which is the real problem.** Measured from `pc_lists.json`: the top-25 PubChem hits
give **17.3 unique Murcko scaffolds per molecule** (median 19), not the 3–8 `plan.md` assumes; ΔM
median 57 Da, p90 134 Da; median 6 aromatic CH sockets. That is ~1,100 regioisomers per molecule
typically and tens of thousands at p90, against a slot budget of ~25. `plan.md`'s answer — "cap at
≤20 per scaffold by natural product heuristics" — is an unspecified 1000:1 cut.

Use the two-pass `f·z` screen the PubChem channel already runs (`probe_core2.probe_one`):

```
scaffolds (PubChem top-25 + library analogues)
  -> formula-space decomposition, |c| <= 6        (exact; no ppm knapsack)
  -> reaction-SMARTS enumeration, generate WIDE   (10^3-10^4/molecule)
  -> pass 1: Morgan-4096 partial f.z, keep 2000   (10,443 fp/s measured)
  -> pass 2: full 10,226-bit f.z, sort            (~700 fp/s/worker)
  -> dedupe ik14 against pool_keys U pc_keys U eng_keys
  -> top K_NOVEL
```

Decompose in **formula space**, not mass space: require
`formula(target) − formula(scaffold)` to be a non-negative atom-count vector expressible in the
palette. Exact, and it removes the ±10 ppm knapsack tolerance entirely. Depth `Σ|c_i| ≤ 6` —
`plan.md`'s ≤4 cannot reach the measured p90 ΔM of 134 Da except through the hexose term.

**2e. Weight and budget (R4 + compute corollary).** RRF weight α_mut ≈ 0.2–0.3 — below the engine's
0.6, because generated candidates are the least trustworthy source. **Never raise ICE/GL λ above
1.0** (λ=2.0 cost −0.034). Budget: the 0.399 run is 3.5–4.5 h against the 9 h ceiling, with ICE at
41 pred/s and GL at 97 pred/s over a v4n-60 ∪ engine-40 union. That leaves room for roughly
**`K_NOVEL` ≈ 25 per molecule**; generation (3,430 products/s) and pass 1 are negligible, pass 2
≈ 285 s. Raise `ICE_BUDGET`/`GL_BUDGET` in step with the added candidates, as the 0.399 notebook did.

**2f. Gates — pre-register in `specs_v4n_update/gate_report.md` before measuring.** Mirrors
`specs_novel`'s G4:
- **GM.0 (kill):** oracle-formula exact@25 of the generated list ≥ 0.01 on a class-3 cohort. Below
  that it cannot beat retrieval — stop.
- **GM.1:** fused-list MRR@25 gain ≥ 0.005 at the chosen `K_NOVEL`.
- **GM.2 (displacement):** MRR loss on a pool-truth cohort ≤ 0.005 at the same `K_NOVEL`. The 0.399
  lineage's one catastrophe (−0.034) was a displacement failure; this is the gate that catches it.
- **GM.3:** projected runtime ≤ 7.5 h.

Fire the mutator only where both existing sources are weak (low `rel` *and* weak absolute
`best_pool_fz`) — never unconditionally. `specs_novel`'s `pseudo_c3` (600) and `c2_displace` (400)
are the cohorts for GM.0 and GM.2; reuse them without merging the tracks.

**2g. Build order.**
1. `build/delta_m_mutator.py` — scaffold harvest, formula-space decomposer, reaction engine,
   validation, ik14 dedupe. Guard clauses for acyclic (0.25% measured) and ΔM < 0 (0.8% measured).
2. `build/test_delta_m_mutator.py`, alongside the existing `build/test_*.py`.
3. Offline cohort run → **GM.0. Stop here if it fails.**
4. `build/mutator_runner.py` (subprocess, `ENG`-shaped output).
5. Notebook: `ADDED`-block entry into ICE/GL + a third RRF term → GM.1/GM.2/GM.3 → LB probe.

### Phase 3 — the orthogonal axis nobody has used

If GM.0 passes for the mutator, the same five rules say `specs_novel`'s decoder is the stronger
version of the same idea (learned, formula-constrained, conditioned on DreaMS embeddings rather than
on a harvested scaffold) and is orthogonal on an axis no current source touches. Its P5 integration
should emit the same `{mid: {smiles, keys}}` interface and enter through the same two doors (union +
RRF). Whether the tracks merge stays a later decision; the interface makes it cheap either way.

---

## 6. Expected value

| | effort | expected ΔMRR | confidence |
|---|---|---|---|
| **P0** adopt 0.399 base | ~1 h | +0.015 vs our 0.384 | high — published |
| **P1** our engine as partner (BIO on, `N_ANALOG` 200, `STRUCT_MASS`, `PPM_CENTER`) | ~1 day + 2–3 probes | +0.003 … +0.012 | medium |
| **P1′** three-way fusion (v4n + public engine + ours) | ~2 h + 1 probe | +0.003 … +0.010 | medium — pure R1 |
| **P2** scaffold mutator as third source, gated | ~1–2 weeks | +0.003 … +0.015 | low, gated |
| **P3** specs_novel decoder as fourth source | weeks | +0.005 … +0.020 | low, best axis |
| *`TOPN` depth, PubChem depth, FPNet swaps* | — | *~0* | *disconfirmed* |

P1′ before P2: it is hours of work on an axis the lineage has already paid twice for, whereas the
mutator is weeks on an unproven one.

---

## 7. Verification

- Every probe: `submission.csv` is (400, 2), no NaNs, ≤25 semicolon-separated candidates per row.
- Phase 1/1′: `fused (+ICE) submission written; top-1 changed in N | ICE reordered M` must show
  non-zero `N` — if the fusion changes nothing, the partner is not orthogonal.
- Phase 2: `ICE_UNION: engine candidates added to ICE input: N` must grow by ≈ `K_NOVEL × 400`, and
  `ICE meta`/`GL meta` `seconds` must stay inside the raised budgets.
- Gates GM.0–GM.3 written to `specs_v4n_update/gate_report.md` **before** the run that tests each.
- Cohort MRR is relative only (`casmi-val-blind-to-pool-misses`); the LB is the only level reading.
