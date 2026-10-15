tier: H
# Candidate ideas, 2026-10-14 (campaign `20261014-qoder-structural`, ideation round 1)

Inputs read: diagnosis `research/ideas/2026-10-14-libgate-diagnosis.md` (human seed; H1, H2, H3 closed), candidates
`research/ideas/2026-10-13-candidates.md` (pre-slot measurements), scouts `research/scout/2026-10-13-papers.md` and
`research/scout/2026-10-13-kaggle.md`, `research/harness/calibration.md`, charter + state history (refspectra-lib: no
leak-free signal; formula-window: dropped, reach-starved), LESSONS IDs, EXPERIMENTS graveyard.
Code cites are in the exploit worktree `E:/casmi/lines/exploit`.

## Where the truth is lost (the targets these ideas aim at)
| bucket | size (MEASURED) | source |
|---|---|---|
| C1: BASE ranks a library-backed isomer above the truth (truth at BASE rank 2: 17 of 26 on C1_npx) | C1_npx −0.052, C1_xlib −0.065 | diagnosis §2 |
| C1: BASE right, demoted by C8 rerank / PubChem merge / promotion / ICE fallback | C1_npx −0.013, C1_xlib −0.015 | diagnosis §2 |
| C2: truth at fused rank 2–5, 0 of them outside the pool | 204/400, oracle +0.298, but no reranker is citable from the harness (BASE leak) | 2026-10-13 pre-slot table (F) |
| C2: truth formula carried by no candidate | 2.5% of C2, 0% of C1_npx | 2026-10-13 pre-slot table (E) |
| Class 3 (novel structure) | 0% ceiling for retrieval (L18); no scorecard cohort | L18 |

Calibration reading (calibration.md): both scored pairs read positive on the harness and negative on the LB; the
candidate predictor is the leak-free "other" Δ (L40). Caveat from the diagnosis loop note: on C1 cohorts "leak-hit"
mostly means *the library matched*, so a C1 library idea's gain lands in the leak-hit column and reads ~0 on "other".
Each C1 idea below therefore also states its `d_lib_leak` reading from `.scratch/libgate_diag/diag.py`.

## Raw ideas (14) and kills
| # | raw idea | operator | verdict |
|---|---|---|---|
| 1 | Comparative per-candidate library score among library-backed top-3 isomers | transfer (diagnosis H1) | **kept → A** |
| 2 | Transductive test-set molecular network: candidates that agree with spectrally related *other test molecules* | fill a gap (new evidence: cross-molecule) | **kept → B** |
| 3 | Route the ICE fallback by a list-uncertainty margin instead of `lib_max` | mutate H2 (move the lever from an absolute score to a relative margin) | **kept → C** |
| 4 | Skip ICE fallback when `lib_max ≥ 0.9` | diagnosis H2 | **kept → D** (stack partner only) |
| 5 | Cross-formula explained-intensity score (MIST-CF / FIDDLE formula posterior) as a reorder signal across formula groups | transfer (scout papers) | **kept → E** |
| 6 | MSAlign frozen-encoder (DreaMS ↔ MolDeBERTa) pair scorer as a 3rd RRF vote | transfer (scout papers #2) | **kept → F** |
| 7 | Generated structures used only as a similarity probe (GLMR) | combine (Track-2 generator × E006 window) | **kept → G** |
| 8 | Track-2 de novo structures into ranks 21–25 for low-confidence molecules | fill a gap (Class 3) | **kept → H** (low EV, by L13 arithmetic) |
| 9 | Lower `LIB_TAU` / an absolute library lock | — | killed: diagnosis H3 (C2_np false-lock 24.5% at 0.7) and graveyard row "Smart library gate on E006" (−0.011) |
| 10 | Soft ICE λ scaled by `(1 − lib_max)` | mutate H2 | killed: out of scope ("Fusion/λ retuning"); graveyard "Forward λ = 2.0"; C keeps the gating idea without touching λ |
| 11 | ICEBERG/GLACIER NIST'23 checkpoint swap | transfer (scout papers) | killed: not freely licensed weights (L19: only MassSpecGym-trained checkpoints are legal) |
| 12 | Library-neighbour scaffold vote (MS2Query analogue) | transfer | killed: analog propagation is already a BASE evidence channel (`scratch/casmi26-e006.py:225`); B is its cross-*test*-molecule counterpart, which BASE does not have |
| 13 | Learned LambdaRank tail reranker (kaggle scout) | transfer | killed: `xattn-rerank` spike, a learned model denied BASE's order −0.0218 (2026-10-13 table F); L11 |
| 14 | RRF duplicate-vote collapse / IK14 pool re-keying (kaggle scout) | remove | killed: already present, `metric_fusion` collapses votes per metric identity (`build/casmi_post.py:311`) |
Also not re-proposed: cross-polarity / per-molecule spectrum weighting (D audit, charter out of scope), MassBank band
(refspectra-lib: dother = 0 on every cohort), formula-window (reach-starved), TOPN / window widening (graveyard), NP-
likeness priors (graveyard row "NP-likeness / class-consistency"), popularity voice changes (L12).

---

## Ranked candidates (by EV)

### 1. `lib-isomer-margin` (A)
- **line**: exploit
- **target cohort + reach**: C1_npx and C1_xlib, the "BASE wrong, truth at BASE rank 2–4" bucket: 24 of the 26 C1_npx
  BASE misses (17 at rank 2, 7 at rank 3–4); C1_xlib 28 molecules. Fires only when ≥ 2 of BASE's top-3 have their own
  library references, so C2 (truth has no references) is reached only when two decoys both have references.
- **mechanism**: after the C8 rerank (`build/casmi_post.py:195-206`) and before the C9 merge
  (`build/casmi_post.py:225`), for molecules whose top-3 hold ≥ 2 library-backed candidates, reorder those candidates by
  a *comparative* score: each candidate's own best reference spectrum vs the query spectra (masked, high-m/z-filtered
  entropy), and swap only when the margin exceeds a gap calibrated on C2_np, where every such swap is a false one.
  Needs a new dump field `base[m][5]` = per-candidate library similarity (today `base[m][3]` holds fused ranker scores).
- **source**: diagnosis H1 (`2026-10-14-libgate-diagnosis.md` §2 and "Hypotheses" 1)
- **novelty**: new (comparative between-candidate score; every past library idea was an absolute threshold or lock)
- **prior**: bucket size MEASURED (diagnosis §2); separability ASSUMED. Median `lib_max` in the bucket is 0.78, so the
  references are cross-instrument. The question is whether the truth's references still beat the isomer's.
- **falsifier**: Kaggle CPU instrument over the train library on C1_npx: the truth's own reference similarity exceeds
  the BASE-rank-1 isomer's in < 50% of the 17 rank-2 cases → dead. Second kill: on C2_np, the swap fires on > 2% of
  molecules at the calibrated margin (false-swap rate).
- **cost**: pipeline (a new dump field, one Kaggle CPU instrument kernel, then a replayable `casmi_post` knob). No GPU.
- **EV**: ceiling ≈ +0.04 on C1_npx (17 × (1 − 1/2) / 245 + the rank-3–4 share). If 50% of it is realised, that is
  +0.02 on C1 ≈ +0.004–0.006 LB at an ASSUMED 20–30% Class-1 share. Stacked with D: ≈ +0.005–0.007. **Highest
  measured-bucket EV.** Read on `d_lib_leak`, not "other" (loop note).

### 2. `test-network-vote` (B)
- **line**: exploit
- **target cohort + reach**: C2 rank 2–5 (51% of C2) and C2_np; reach = molecules with ≥ 1 spectrally related *other*
  query molecule. Hidden-test reach is ASSUMED: 400 Enveda NP molecules from plant extracts, where analog series are
  common.
- **mechanism**: a transductive evidence channel that BASE does not have. Molecular networking among the test queries:
  pairwise modified cosine between molecules (precursor-shift aware), and edges at ≥ τ. For molecule *m*, a
  candidate's network score = max over neighbours *n* of edge_w × max Tanimoto(candidate, top-3 of *n*'s fused list).
  It enters as a 3rd low-weight RRF list in the overlay's `metric_fusion` (`build/casmi_post.py:311`;
  notebook `scratch/casmi26-e006.py:967`), restricted to reordering ranks 2–10, so rank 1 is never touched (rank-1
  displacement bar).
  - Orthogonal by construction (L10): it reads *other molecules' spectra*. That is not a re-slice of FPNet f·z (L11)
    and not a library lookup.
- **source**: fill a gap (no channel uses cross-molecule evidence); scout papers row MS2Query/molecular networking
  query #15; combine with E006's fused lists
- **novelty**: new
- **prior**: ASSUMED (no measurement yet). Known risk: the neighbours' lists share the same decoy families, so the vote
  can reinforce a shared wrong scaffold (an L12-like failure).
- **falsifier** (ceiling script on the val parquet + post_dump lists, 0 GPU):
  - (i) < 10% of C2_np / C2 molecules have a neighbour at modified cosine ≥ 0.7 → reach-starved, dead.
  - (ii) Among rank-2–5 misses with a neighbour, the truth's network score > the BASE-rank-1 decoy's in ≤ 50% → no
    signal, dead.
  - (iii) Replay as a knob: target "other" Δ ≤ 0 → dead.
- **cost**: pipeline (CPU: ~400² spectrum pairs, Tanimoto on ≤ 40 × 3 per edge; minutes). No training.
- **EV**: highest upside, lowest certainty. If (ii) holds at 60% on half of a 20% reach: C2 rank-2→1 on ~10 molecules
  per 400 → +0.012 C2, ≈ +0.006 LB. The ceiling script is cheap, so run it first.

### 3. `ice-margin-route` (C)
- **line**: exploit
- **target cohort + reach**: all C1 cohorts (ICE removal everywhere would give C1_npx +0.011, C1_xlib +0.013), kept
  neutral on C2 (where ICE removal costs −0.069).
- **mechanism**: the ICE sign flips by class (diagnosis §3). The ICE fallback (`build/casmi_post.py:322-326`) should
  fire only when the fused list is uncertain. Gate: the relative margin between the fused top-1 and top-2
  `metric_fusion` scores (`scores` at `build/casmi_post.py:311`). This is a within-molecule normalised quantity, not
  an absolute score (graveyard row "Absolute f·z gate"). A confident, library-driven top-1 skips the ICE rerank; an
  uncertain list keeps it. Default-off knob `ICE_ROUTE_MARGIN`.
- **source**: mutation of H2 `ice-libprotect` (invert the gate signal: list confidence instead of `lib_max`); scout
  papers "Selective prediction" (retrieval-level uncertainty is the strongest risk signal)
- **novelty**: variant
- **prior**: ASSUMED. Bounds are MEASURED: upper bound +0.011…+0.013 on C1 (ICE fully off) and C2 at risk −0.069 if
  the gate leaks.
- **falsifier**: harness replay over the battery with the margin set *a priori* (the median C2_np top-1/top-2 RRF
  gap, pre-registered; no tuning, L08): the C1_npx Δ ≤ +0.0054 (does no better than D), or the C2 / C2_hard CI upper
  bound < −0.004 → dead.
- **cost**: knob (`casmi_post` only; fully replayable, 0 GPU).
- **EV**: ≈ +0.002–0.003 LB alone, below noise. Worth measuring because it is free, and it **replaces D** in the A
  stack if it beats D's +0.0054.

### 4. `formula-explain-rerank` (E)
- **line**: exploit
- **target cohort + reach**: C2 / C2_hard rank-2–5 misses where BASE's rank 1 has a **different molecular formula**
  from the truth. Its size is unmeasured: the ICE/GL rerank acts inside formula groups (`gl_rerank`,
  `scratch/casmi26-e006.py:553`), so cross-formula errors are untouched by the simulator.
- **mechanism**: a per-formula posterior of the (formula, adduct) pair from MS2 sub-peak explanation (MIST-CF style,
  or explained-intensity fraction under a subformula decomposition), added as a cross-formula reorder term on the
  fused window (overlay, `build/casmi_post.py:311`).
- **source**: transfer of scout papers MIST-CF / FIDDLE to the cross-formula part of the C2 rank-2–5 bucket
- **novelty**: new (formula-window targeted *recall*; this targets the *ordering* between formulas)
- **prior**: ASSUMED. Overlap risk: BASE already uses MetFrag-lite fragment matching (`scratch/casmi26-e006.py:225`),
  which partly encodes formula explanation.
- **falsifier**:
  - (i) On the C2 + C2_hard dumps, the share of rank-2–5 misses whose BASE rank-1 formula ≠ the truth formula
    (`base[m][4]` forms vs the truth formula) is < 15% → reach-starved, dead. This takes minutes.
  - (ii) The new score's Spearman with the MetFrag-lite feature on candidates is > 0.8 → no new evidence (L11).
- **cost**: pipeline (CPU subformula enumeration). A pretrained MIST-CF/FIDDLE checkpoint would be GPU inference; check
  its licence via a request only if (i) passes.
- **EV**: unknown until (i). Cheap to kill, so rank 4.

### 5. `msalign-rerank` (F)
- **line**: exploit
- **target cohort + reach**: C2_hard (FPNet-unseen; L20 says to audit FP changes there) rank 2–5.
- **mechanism**: frozen DreaMS spectrum embedding ↔ frozen MolDeBERTa candidate embedding, with two MLP heads trained
  candidate-contrastively. Its pair score enters the overlay as a 3rd RRF vote over the 40-window
  (`build/casmi_post.py:311`); it never widens the window.
- **source**: scout papers #2 (MSAlign); a combine with E006 fusion
- **novelty**: variant (of `xattn-rerank`; a different evidence model, a frozen spectral foundation encoder, rather
  than a second spectrum→FP, so L11 does not apply directly)
- **prior**: cheap-form reranker MEASURED as no signal (`xattn-rerank` spike); MSAlign form ASSUMED. Requires
  `research/requests/20261014-model-msalign.md` (filed): licence, released heads, training data.
- **falsifier**: on C2_hard, the MSAlign score ranks the truth above BASE's rank-1 decoy in ≤ 50% of rank-2–5 misses,
  or its Spearman with FPNet f·z on candidates is > 0.8 → dead.
- **cost**: GPU training ≤ 4 h on Kaggle (heads only) if no released heads; embedding precompute per pool molecule.
- **EV**: moderate upside, but the reranker family is BASE-COUPLED on the harness, so it is LB-decided. Rank 5.

### 6. `ice-libprotect` (D)
- **line**: exploit
- **target cohort + reach**: C1 molecules demoted by ICE (C1_npx fired 42, 4 up / 0 down).
- **mechanism**: skip the overlay's ICE fallback (`build/casmi_post.py:322`) when `lib_max ≥ 0.9`. A default-off knob.
- **source**: diagnosis H2
- **novelty**: variant
- **prior**: MEASURED (diagnosis §3): C1_npx +0.0054 [+0.001, +0.011], C2 −0.0005, C2_hard +0.0010, 0 displacements on
  C1_npx.
- **falsifier**: the knob as built: C1_npx Δ ≤ 0, or a C2 / C2_hard CI upper bound < −0.004.
- **cost**: knob.
- **EV**: ≈ +0.001–0.002 LB, below noise. **Do not spend an LB slot alone**; ship only stacked with A (or replaced by
  C).

### 7. `gen-probe-rerank` (G)
- **line**: exploit
- **target cohort + reach**: C2_hard / C2_np rank 2–5 and, uniquely, Class-3-like molecules where the truth is absent
  but a near analogue is present (MCES gain, not exact).
- **mechanism**: GLMR's safe half. Sample k structures from a spectrum(+formula)-conditioned generator (our Track-2
  DreaMS + SELFIES DFS, or a released DiffMS/FlowMS). The generated structures **never enter the list**; each fused
  candidate gets max Tanimoto to the generated set, added as a 3rd low-weight RRF vote in the overlay
  (`build/casmi_post.py:311`).
  - This respects the graveyard: generated candidates inside the RRF and rank-1–5 injection both failed. It also
    grafts Class-3 work onto E006 (L18).
- **source**: scout papers #3 (GLMR); combine Track 2 × E006
- **novelty**: new
- **prior**: ASSUMED; GLMR reports +40% top-1 on MassSpecGym (not our metric). Requires
  `research/requests/20261014-model-generated-probe.md` (filed): weights, licence, training-set overlap with the val
  cohorts (L19).
- **falsifier**:
  - On C2_hard rank-2–5 misses, Tanimoto(truth, generated set) > Tanimoto(BASE rank-1, generated set) in ≤ 55% → dead.
  - Spearman of the probe score with FPNet f·z > 0.8 → correlated, dead (L11).
- **cost**: GPU inference (sampling time per spectrum unknown; must fit in the 9 h T4 budget alongside ICE). Track-2
  training is a Colab request, not this campaign.
- **EV**: upside comparable to F at a higher cost. Rank 7 until the request answers the weights question.

### 8. `c3-tailfill` (H)
- **line**: exploit
- **target cohort + reach**: Class 3 (no scorecard cohort exists, so this needs a ceiling script).
- **mechanism**: for molecules whose fused list is low-confidence (C's margin), replace ranks 21–25 with formula-
  constrained Track-2 generations, under `assert_monotone` for ranks 1–20 (`scratch/casmi26-e006.py:860`). Today
  generations only fill *vacant* slots (`build/casmi_post.py:349-356`).
- **source**: fill a gap (Class 3, L18)
- **novelty**: new (the current fill uses a mass-keyed DreaMS pool and only empty slots)
- **prior**: ASSUMED; field de novo exact top-10 ≈ 4% leakage-free (scout papers, de novo trio).
- **falsifier** (L13 arithmetic): ceiling = C3 share × exact-hit@5 × 1/21, minus the truths now at ranks 21–25 on the
  cohorts (measurable on dumps). At an ASSUMED 30% share and 10% hit rate that is ≈ +0.0014. Dead unless the specs_novel
  G4.0 oracle exact@25 shows hit@5 ≥ 25%.
- **cost**: GPU inference + Track-2 weights.
- **EV**: lowest (≤ +0.002 even if it works). Listed because Class 3 is the largest open gap and nothing else targets
  it; do not slot it before G4.0 numbers exist.

---

## Recommended order for /replan
1. **E(i)** and **B(i–ii)**: free ceiling scripts, minutes each. Run both before any spec; either may die on reach.
2. **C then D**: free replays. Keep whichever is better as the C1 stack partner.
3. **A**: one Kaggle CPU instrument for its falsifier, then a spec. It is the stack's main lever (A + C/D = one LB
   probe).
4. **F / G**: wait on the two filed requests; heavy, LB-decided.
5. **H**: parked until the specs_novel G4.0 oracle numbers exist.
