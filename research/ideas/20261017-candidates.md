# Candidates — 2026-10-17 (Qoder IDE, diagnose + ideate)

Produced from: `research/scout/2026-10-17-qoder.md` (12 sources), CORPUS.md PART A
(§A5 instrument, §A6 loss map, §A7 graveyard, §A11 untried seeds), `research/LESSONS.md`
(L01–L49), and the two prior idea backlogs (`research/ideas/innovative_ideas.md` A1–A6/
B1–B8, `research/ideas/20261014-candidates.md`). Nothing here is a re-proposal of a §A7
dead end or an already-run idea; where it resembles one, the "why-not-dead" line says what
changes the outcome. Ties to the executed set: cross-formula-cluster/nl-sim-rda/
transductive-net-stoich/tautomer-canon-unify are DEAD, e180-drop PASS, v4b-union-xenc
deferred — all excluded below.

---

## Diagnosis (loss-map buckets → falsifiable hypotheses)

Levels read high on val (BASE leak, L07); **shares** are the informative part (§A6).

- **B1 · C2 rank 2–5 ordering — headroom +0.298, the single biggest bucket.** 51% of C2
  truths sit at rank 2–5, 0 of them outside the union pool → a *pure ordering* problem.
  Hypothesis: it is unmeasurable offline only because BASE's leaked rank-1 dominates the
  existing scalars (xattn −0.0218 when denied BASE order). If a **new evidence type that
  is not a function of BASE's scalars** is added, the truth-vs-decoy separation should
  appear even on the leak-free "other" subset. Falsifying measurement: Spearman(new score,
  truth rank) on C2_hard (whole-Murcko-banned) and on the leak-free "other" C2 subset; kill
  if the new score correlates > 0.85 with `f·z` (correlated, L11) or its Δ_other ≤ +0.003
  (L45).
- **B2 · The instrument is blind (no honest cohort).** No dumped cohort measures BASE on
  genuinely-unseen NPs (BASE top-1 0.897–0.947 everywhere, §A5 leak 2). Hypothesis: every
  B1-family idea is currently un-gatable because there is nowhere to read it. Measurement:
  retrain one FPNet+ranker pair on a fold that excludes an NP structure cluster (MCES≥10 /
  Murcko-scaffold held out, per scout #5) and confirm BASE top-1 on the held-out cohort
  falls to a plausible 0.4–0.6 (not ~0.95). This gates B1 and any rerank family.
- **B3 · Unused physical signal — base_peak_intensity (BPI).** BPI (raw counts, min 1000)
  is populated 100% on timsTOF test but 45.8% null in heterogeneous train libs (data
  contract); the pipeline's static 0.1% peak floor and unweighted cross-spectrum score
  averaging ignore detector SNR. Hypothesis: weak spectra (BPI≈1200) inject shot-noise
  spikes that corrupt entropy similarity and steal FPNet peak tokens. Measurement: on
  C1_npx/C2_np (100% BPI), compare BPI-floor + BPI-weighted aggregation ΔMRR; kill if
  Δ_other < +0.003.
- **B4 · Unused physical signal — collision-energy ladder.** CE known 100% (20/40/60 +
  merged 24.6%); `ice_ex_mean`/`gl_ex_mean` average over all CE, discarding that high-CE
  spectra carry skeletal/ring cleavages (isomer-discriminative) while low-CE carry
  peripheral losses (formula-conserving). Hypothesis: CE-band-weighted simulator rerank
  separates positional isomers better than the flat average. Measurement: within-formula
  top-10 isomer accuracy on C2_np with vs without CE weighting; kill if gain < +0.003.
- **B5 · Class-3 is now UNPARKED.** L18 parked Class-3 until the line reached ≈0.400; V22
  is at 0.447–0.451, so the gate is met. Track-2 decoder is at 8.17% exact@25 (oracle
  formula, G3.1 PASS) and MADGEN is vendored locally. Hypothesis: cross-modal *generative
  retrieval* (GLMR/MADGEN/MS-BART) reaches structures the mutator's 53% scaffold wall
  cannot, and slot-fill behind a calibrated detector adds non-negative MRR. Measurement:
  oracle-formula exact@25 ≥ 1% on pseudo_c3 (K1 gate already 7.7%/12% mutator), then a
  predicted-formula reach spike.

---

## Ranked candidates (by EV = reach × gain × P(survive) ÷ cost)

### 1. `honest-oof-cohort` — instrument fix (bucket B2) — **novelty: new (to execution)**
- **Mechanism**: retrain ONE FPNet(family P or A) + its ranker on a split that holds out a
  natural-product structure cluster (Murcko-scaffold / MCES≥10 grouped), dump a new val
  cohort on the held-out NPs so BASE top-1 is honest. Ships as `dumps/<cohort>/post_dump/
  state.pkl` + a `val_key` entry. It is not a knob — it is what makes buckets B1/B4/B5
  *measurable* and citable by the rank-only gate.
- **Target**: measurement of C2 rank-2-5 ordering family + any contrastive channel.
- **Arithmetic ceiling**: indirect but ≥ the whole +0.298 C2 ordering headroom becomes
  addressable; without it, ideas 2–6 stay ASSUMED and cannot book a slot honestly.
- **Falsifier (pre-slot)**: if the held-out NP cohort still shows BASE top-1 > 0.85, the
  split failed to remove memorization → cohort is not honest, discard.
- **Cost**: Kaggle GPU ≤ 4 h (single fold) from the weekly pool, or a Colab training run
  (human approval). CPU replay afterwards is free.
- **Why-not-dead**: §A5 open question + dossier B6 "Leak-Free Tri-Split" were pre-registered
  but **never executed**; L07/L19 prove train-derived shards leak, so the fix is a
  structure-distance split that includes FPNet-unseen queries (satisfies L01).

### 2. `jestr-contrast-channel` — contrastive spectrum↔candidate RRF source (bucket B1) — **novelty: new**
- **Mechanism**: a JESTR/MIST-contrast/`Aligning-MS` style joint embedding: encode the
  query spectrum and each union candidate into one space, rank by **cosine**, add as a
  **new RRF vote** in `_metric_overlay` (build/casmi_post.py) over BASE∪ENG∪PC, gated off
  when `lib_max ≥ 0.90` (protect Class-1). This is *direct similarity*, not predict-then-
  match `f·z` and not a forward-sim entropy — a genuinely orthogonal evidence source, the
  only class of change that has ever paid (L10: second engine +0.016, GLACIER +0.007,
  BIO pool +0.009).
- **Target**: C2 rank 2–5 ordering; secondary C1 cross-instrument order.
- **Arithmetic ceiling**: promote 15–20 of ~180 C2 rank-2-4 truths to rank 1
  (ΔRR ≈ +0.55) → (18×0.55)/400 ≈ **+0.0247** optimistic; realistic +0.006–+0.012.
- **Falsifier (pre-slot, heavy-path ceiling script on cohort from idea 1)**: kill if
  Spearman(contrast-rank, `f·z`-rank) > 0.85 on the held-out NP cohort (correlated, L11)
  or Δ_other ≤ +0.003 (L45). Retraining the encoder is a proven fallback (scout #10).
- **Cost**: heavy path (needs the honest cohort = idea 1). Offline staging of open weights
  or a ≤ 4 h Kaggle contrastive-head retrain; licence must be verified (scout blocker).
  **RESOLVED 2026-10-17** → `research/requests/20261017-jestr-feasibility.md`: JESTR is MIT +
  released MassSpecGym weights, cosine-retrieval, DGL (shim precedent exists). Prototype needs
  **no training**; falsifier spike (Spearman vs f·z on C2_hard) is the only gate.
- **Why-not-dead**: `20261013-xattn-rerank` collapsed (−0.0218) because it trained a
  ranker on the **4 existing dump scalars** (no separation without BASE order); this adds a
  **new representation from the raw spectrum**, not a reweight of existing scalars. Distinct
  from R4/`second-FPNet` (L11) because it is contrastive retrieval, not another
  predict-then-match fingerprint head on the same task.

### 3. `bpi-snr-weighted-aggregation` — detector-SNR channel (bucket B3) — **novelty: known-untried (backlog B5), cheap**
- **Mechanism**: per-spectrum peak floor `θ(BPI)=clip(25/BPI, 5e-4, 0.025)` applied before
  entropy similarity, and cross-spectrum aggregation of `ice/gl/fpnet` scores weighted by
  `w_s ∝ log10(max(BPI_s,1000)) − 2.5`. Post-scoring knob (default-off `BPI_SNR=true`),
  ≤ 5 NumPy lines, **measurable on the harness today**, zero GPU.
- **Target**: C1_npx + C2_np (100% BPI, timsTOF-like).
- **Arithmetic ceiling**: ~160 high-BPI-variance molecules × small ΔRR → optimistic
  (8×0.55)/400 ≈ **+0.0110**; realistic +0.004–+0.006.
- **Falsifier**: on C2_np/C1_npx, BPI-weighted MRR − baseline < +0.003 → kill.
- **Cost**: CPU knob, harness-replayable. Cheapest real candidate.
- **Why-not-dead**: BPI is a raw column never consumed (null 45.8% in train is why it was
  skipped, but 100% on the test-like cohorts). Reweights *evidence provenance*, not a tuned
  model hyperparameter (so not L08 val-tuning).

### 4. `ce-band-weighted-sim-rerank` — collision-energy channel (bucket B4) — **novelty: new**
- **Mechanism**: instead of averaging `ice_ex_mean`/`gl_ex_mean` over all spectra of a
  molecule, weight by CE band: high-CE (60/merged) contributions dominate *within-formula
  isomer* ordering, low-CE (20) dominate confident/precurson-like matches. A CE-conditioned
  weight on the existing per-spectrum simulator scores (already computed by the runners) —
  **no extra GPU forward pass**, post-scoring knob `CE_WEIGHT=true`.
- **Target**: C2/C2_hard regioisomer order; ~300 multi-CE test molecules.
- **Arithmetic ceiling**: resolve ~10 isomer pairs rank 2–3→1 → (10×0.55)/400 ≈ **+0.0138**;
  realistic +0.004–+0.007.
- **Falsifier**: within-formula top-10 same-formula isomer accuracy with vs without CE
  weighting on C2_np (100% CE-ladder) fails to beat flat average by ≥ +2.0% → kill.
- **Cost**: CPU knob on already-dumped per-CE scores; harness-replayable.
- **Why-not-dead**: distinct from backlog B4 (CE *difference-spectrum* matching, needs new
  per-CE sims) and from `ice-margin-route` (dead: used the *fused margin*, a within-molecule
  class signal); this uses the *collision energy* physical axis to weight existing sims.

### 5. `formula-coherence-prior` — MIST-CF-style formula posterior (buckets pool-miss/B1) — **novelty: combine (A3-dead ⊕ scout #2)**
- **Mechanism**: an inter-formula prior computed from the +1.45 ppm-centred precursor
  mass-defect + candidate-formula exact-mass residual + fragment mass-defect coherence
  (MIST-CF / FIDDLE idea), used to promote the **correct formula group** *before* the
  within-group simulator rerank, as an RRF feature. New because the pipeline currently only
  *groups* by formula for the forward rerank; it never scores *which formula* is right.
- **Target**: pool-miss + cross-formula ordering (C2).
- **Arithmetic ceiling**: reach-gated — must first measure how many C2 truths have the
  correct formula *not* first (else ceiling ≈ 0).
- **Falsifier (before any build)**: on the dumps, share of C2 rank-2-25 truths whose correct
  formula is already group-ranked #1; if the formula posterior cannot move a group that
  contains the truth, kill (avoids the A3 "n_wrong_f1==0 on non-leak lists" degeneracy).
- **Cost**: CPU knob (post-scoring) if a per-formula score can be replayed; else ceiling
  script only.
- **Why-not-dead**: `cross-formula-cluster` (A3) died because K3 *within* non-leak lists had
  zero cross-formula wins; this is an upstream *formula identification* prior (different
  evidence: exact-mass defect coherence), and carries A3's own degeneracy as the falsifier.

### 6. `c3-cross-modal-slotfill` — generated probe behind detector (bucket B5) — **novelty: new**
- **Mechanism**: graft MADGEN (vendored `MADGEN/`) scaffold-ranked / DiffMS (MIT) generated
  structures into **empty tail slots only** (ranks 21–25) behind the K2 Class-3 detector,
  tagged `'|gen'` (L17) so the forward rerank cannot swap them with a truth. Never touches
  rank-1, so downside is ~0 by construction; upside is C3 truths retrieval scores can't
  reach at all.
- **Target**: Class 3 (unparked at > 0.400); ~25% of hidden test (ASSUMED).
- **Arithmetic ceiling**: tail-fill is small (per §A11 "tiny ceiling"); the real upside is
  *detector-gated* ranks 2–5, which needs a precise C3 detector (K2, in flight on the c3
  line).
- **Falsifier**: oracle-formula exact@25 < 1% on the honest pseudo_c3 (G4.0 kill gate), or
  the detector's false-positive rate on C2 queries > the Class-3 share it rescues.
- **Cost**: heavy path (generation + detector); GPU for the decoder eval. Highest risk.
- **Why-not-dead**: prior mutator/c3_gate hits the 53% scaffold wall with a *rule-based*
  generator; cross-modal generative retrieval is a different mechanism, and slot-fill avoids
  the `bio2` rank-hijack disaster (−0.064, L14).

---

## Recommended order for /replan

> **REPLAYABILITY CORRECTION (measured 2026-10-17).** Inspected
> `research/harness/dumps/C2_np/post_dump/state.pkl`: `ice` and `gl` are keyed
> `dict[mol_id][candidate_smiles] -> float` — i.e. the simulator/FPNet similarities are
> **already averaged over the molecule's spectra (and over CE) during scoring, before the
> post_dump is written**. `consts` holds only scalars (TOPN, ICE_LAM, GL_LAM, LIB_TAU, ...).
> Consequence: ideas **3 (BPI-SNR)** and **4 (CE-band)** are **NOT post-scoring-replayable
> CPU knobs** — the per-spectrum signal they reweight is absent from the dump. They are
> **heavy-path**: each needs ONE val-mode re-run that dumps *per-spectrum* simulator
> similarities + per-spectrum BPI/CE (raw BPI/CE exist in `val.parquet`, test schema), after
> which both become cheap replayable scorecard knobs. This re-run is also the input the
> contrastive channel (idea 2) needs. So the corrected cheap-first move is the per-spectrum
> dump re-run (≈1–2 GPU-h within the authorized pool), then measure 3/4 as knobs.

1. **`honest-oof-cohort`** (idea 1) — do first / in parallel: it is the *precondition* for
   honestly measuring 2, 4, 5 and for the c3 detector's gate. Low idea-risk, one training run.
2. **`bpi-snr-weighted-aggregation`** + **`ce-band-weighted-sim-rerank`** (3, 4) — cheapest
   measurable post-scoring knobs; pre-register and spike on C1_npx/C2_np/C2_hard now. Likely
   sub-noise individually (L10 warns reweighting ≈ 0) but cost is minutes of harness replay.
3. **`jestr-contrast-channel`** (2) — the doctrinally-correct big orthogonal bet; start with
   the licence/weights feasibility check (scout blocker) and a retrain fallback, measure only
   once idea 1's cohort exists.
4. **`formula-coherence-prior`** (5) and **`c3-cross-modal-slotfill`** (6) — hold until a
   reach/false-positive gate on idea 1's cohort clears them.

Two of the six (`jestr-contrast-channel`, `ce-band-weighted-sim-rerank`, `c3-cross-modal-
slotfill`) are tagged `novelty: new` (≥2 required — satisfied).

---

## MEASURED FALSIFIER 2026-10-17 (corrects the "cheap knobs 3/4" note above)

The earlier note said BPI/CE "raw values exist in val.parquet" — measurement says otherwise. Read
`ice_runner.py::score_molecule` + ran `spike/spike_per_spec.py` on the real parquets (source:
`research/antigravity-exploit/20261017-per-spectrum-dump/spike/spike_per_spec.log`). Among
*covered* spectra: `base_peak_intensity` present = **val 0.000 (all-NaN)** vs **test 1.000**;
finite per-spectrum CE = **val 0.358** vs **test 1.000**; molecules with >1 covered spectrum =
**val 0.476** vs **test 0.809**.

- **Architecture OK**: ICEBERG/GLACIER already average a per-spectrum `sims` list with a plain
  `np.mean`, so per-spectrum scores are recoverable at ~zero GPU cost. The seam is the vendored GPU
  runner (`iceberg.zip::ice_runner.py`, re-staged as the `casmi26-iceberg` dataset), not the dump
  glue — a heavier, human-gated dependency re-pin (its questions.md Q1).
- **CPU-measurement purpose is crippled**: the *val* cohorts the harness scores on carry **no BPI**
  and only **36% finite CE** — the exact features ideas 3/4 reweight — while the hidden test has
  both at 100%. A knob tuned there fits val-noise and ships blind to test (regime mismatch, cousin
  of L40/L45). **`per-spectrum-dump` is DEAD as specced**; ideas 3/4 stay heavy-path AND blocked on
  a **val-side BPI/CE proxy** (num-peaks / TIC / SNR from `ms2_*`) — a separate, pre-registerable
  idea, not native BPI.
- **Net for the wave**: `jestr-contrast-channel` (idea 2) is unaffected and is the better single
  build to dispatch next.
