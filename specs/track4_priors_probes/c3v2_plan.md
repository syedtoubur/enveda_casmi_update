# Class-3 mutator v2 on the 0.415 base — gated, anchored, two-stage

## Context

Track 4's popularity lever is used up: μ 0.25 lost 0.012 on a GPU-confirmed run, and the pre-cut prior lost 0.004. The new base is `casmi26-v29-fusion-pop-glmh-0-415.ipynb`, which runs the same code as 0.411. Its runs vary by about ±0.004 between submissions.

Class 3 (novel structures) is the one class where retrieval scores 0 by construction. The user wants to improve the Track-1 ΔM scaffold mutator, with no new model training, and fire it **only when we are confident the molecule is Class 3**, so the score never drops.

Track 1 ended flat at 0.399 (`casmi26-v4n-mutator-c3-submit`). Exploration found these root causes:
- **Unique scaffolds.** 53% of novel truths have a Murcko scaffold that occurs nowhere else, so harvesting scaffolds can't seed them. Production recovery was 4.2%, against 17.2% in the oracle test (which seeded from the truth's own scaffold).
- **Additive palette only.** The mutator can't produce "analog minus a group" (deglycosylation, demethylation, deacetylation).
- **Target formula taken from PubChem.** The step count was measured from the top analog's scaffold to the PubChem top formula. It never checked whether the precursor-mass difference matches a known biochemical transform.
- **The stage-2 gate was never implemented.** The fusion cell only read `u['tier']`. The real v4n `lib_max` and the ICE veto were never applied (`mut_runner.py:124` passes the exact-mass score in place of `lib_max`).
- **The `'|gen'` tag is lost in the last cell.** `_form2` recomputes formulas at S:481/484 (S = `scratch/casmi26-v29-glacier-m-h-full-fusion-0-411.py`). Generated candidates there also had raw ICE+GL summed, not z-scored.
- **Thresholds never calibrated.** `ANA_MIN` and `MARGIN_MIN` were set without measurement; there is no Kaggle log of gate signals.

**User decisions:**
1. Run two tracks in parallel: a submittable LB notebook now, and a separate Kaggle calibration notebook whose numbers feed later revisions.
2. Rank 1 is allowed for the `sure` tier only, so `sure` must be strict and capped.

## Approach: the mutator v2 innovations (no training)

### I0. Fingerprint-improvement validation (core idea, from the user)

Suppose the query's analog A sits at Δm (e.g. 30 Da) and the signed-transform search turns A into candidate B. The PubChem channel already computes a predicted-fingerprint vector `z` for the query (FPNet logits; the runner saves it as `pc_z.npz`). If B is closer to the hidden truth than A, then `fp_B·z > fp_A·z`.

- **Score:** `Δfz(B) = (fp_B − fp_A)·z`. It is computed per molecule with one shared `z`, so the result doesn't depend on that molecule's z scale.
- **The vector is the same one** that produces `best_pool_fz` and `fz_top` (`casmi26-v3-models` FPNet A+B), so these three comparisons are all on one scale:

| Comparison | What it shows |
|---|---|
| `fz(B) > fz(A)` | the edit moves toward the query's predicted structure |
| `fz(B) ≥ best_pool_fz` | B beats every pool candidate in the mass window |
| `fz(B) ≥ fz_top` | B beats every PubChem-only candidate |

- **Use the raw values only.** The comparisons read `fz_top` and `best_pool_fz`, never `pc_fz`. Since the V17 patch, `pc_fz` is `z(fz) + 0.25·pop`, so it has popularity mixed in (Track-1 Rule 1).
- **Null guard against size bias.** Adding a group adds bits. For each anchor, compare `Δfz` of the best B with the `Δfz` distribution over **all** of A's generated edits, and require the best B in the top 5% (a per-anchor percentile). One extra bit can't then pass the test by luck.

**Broad generation with a matrix pre-filter, like the PubChem route.** Enumerate many signed add/subtract combinations per anchor (≤ 3 steps, all matching sites, cap about 20k products per molecule). Then filter in two passes, re-using `mut_runner.mutate_one`:
- **Pass 1:** ECFP4 partial fingerprint · `z` as one matrix product; keep the top `MUT_N1` (2000).
- **Pass 2:** full selected-bit fingerprint · `z`.

Only candidates with `Δfz > 0` against their own anchor survive. This replaces the "beam keeps the alphabetically first 200" truncation in `delta_m_mutator.py` L187. The beam can be widened because ranking now happens after generation and costs only matrix products.

### I0b. Prune paths early with intermediate Δfz (from the user)

For a two-step path A→C→B, score the intermediate: `Δfz(C) = (fp_C − fp_A)·z`. If `Δfz(C) < −τ`, prune C and every path through it.

**Why this is nearly lossless:**
- f·z is a sum over bits, so `Δ(B) ≈ Δ(e1) + Δ(e2)` plus an interaction term from ECFP environments that span both edit sites.
- Independent edits can be applied in either order, so B is lost only if **both** intermediates fail. Then `Δ(B) ≤ 0` anyway, and B would fail the final I0 test.

**Two conditions:**
1. **Macro-transforms.** Merge each socket-creating step with the step that uses it (`+O` then `+hexose` becomes `+O-hexose`, C6H10O6; likewise `+O-acetyl`, `+O-methyl`, `+NH-…`). Otherwise the free-OH intermediate scores below the O-glycoside truth and the glycoside path gets pruned. This also retires the forced `+O`/`+NH`-first `_order`.
2. **Tolerance plus a ranked beam.** Prune only below −τ; adjacent edits are non-additive and `z` is noisy. Then keep the top-`BEAM` intermediates by Δfz, replacing the alphabetical truncation in `grow` (L187).

**Implementation.**
- `grow` and `mutate_sources` take an optional `score_fn(smiles_list) -> np.ndarray` callback, so `delta_m_mutator.py` stays RDKit-only and testable. With no callback the old behaviour is unchanged.
- `mut_runner` passes a callback built from `chem.raw_fingerprint` and the query's `z`.

**Effect.** The tree shrinks exponentially, so the same time budget reaches depth 3 over all sites.

**Validation, local and before Kaggle.**
- Use an oracle `z` (the truth's own fingerprint mapped to ±1 logits) on `pseudo_c3` in `build/eval_mutator_cohort.py`.
- Report the share of reachable truths lost for τ ∈ {0, 0.5, 1, 2} (in z units), and the runtime saved.
- Pick the smallest τ that loses ≤ 2% of truths.
- The calibration notebook then repeats this with the real FPNet `z`.

### I0c. Neighbour-consensus fingerprint prior (from the user)

**Build.** For each molecule, take the direct hits (`lib`) and the shifted analogs (`ana`) from `.c3.json`, weighted by `w_k = sim_k^p`. Compute the per-bit probability `p_i = Σ w_k·fp_k[i] / Σ w_k` in FPNet's selected bit space (`chem.raw_fingerprint(...)[bits]`, the same space as `z`).

**One-sided core prior.**
- `C = {i : p_i ≥ 0.8}`, the conserved-present bits.
- `core(B) = fraction of C kept in fp_B`.
- Bits that no neighbour carries are scored 0, so novel bits that edits add are not penalised.
- A symmetric `logit(p)` prior must **not** be used: the anchors are inside the consensus, so it would always prefer A over every B.

**Where it is used:**

| Use | How |
|---|---|
| Ranking and I0b pruning | `score(B) = fp_B·z + β·Σ_{i∈C} fp_B[i]·logit(p_i)`, with β ≤ 1 (the ~1 z-unit ceiling rule) |
| Filter (revised after the `summary_maccs.md` review) | **exact substructure check, not hashed bits.** The consensus core is the generic Murcko ring framework shared by ≥ 60% of the analog weight; require `B.HasSubstructMatch(core)`, except for rings that a transform on B's path removes (`-hexose`/`-deoxyhexose`). It applies only to scaffold-harvest candidates; anchored edits keep A's core by construction. Hashed-bit cores (FPNet bits and MACCS alike) are rejected: an RDKit test on quercetin showed +hexose flips 291 of 519 path bits and 10 of 29 MACCS keys (one of them a ring key), and the two methoxy regioisomers differ by only 1 MACCS key. |
| Stage-1 gate | tight consensus (≥ 3 neighbours with sim ≥ 0.5 and `|C|` large) is positive evidence of "a known family with an unseen member" |
| A-vs-B test (I0) | **not used.** That test keeps FPNet `z` only, so the anchor bias cannot leak in. |

**Off switch.** With fewer than 3 neighbours, β = 0 and there is no core filter.

**Not adopted from `summary_maccs.md`:**
- MACCS core keys: they bleed too (4 to 10 keys flip per edit) and cannot separate regioisomers.
- The 100% core-key veto: it rejects the glycoside and aglycone truths.
- `disallowed_keys` as a veto: it is the anchor-bias trap.

The radius-bleed point is accepted. It is what motivates the substructure filter above, and the soft β term rewards only conserved-present bits, so bleed costs at most β times a few bits per edit.

**Code.** A new `consensus_fp(neighbours, bits, power)` helper in `mut_runner.py`. The I0b `score_fn` callback receives the combined score.

**Calibration (Kaggle cohorts).**
- How often `core(truth) ≥ 0.9` on `pseudo_c3` positives.
- The truth's rank among the generated candidates with β ∈ {0, 0.5, 1}.
- Keep β > 0 only if it improves the truth's rank without lowering recovery.

### I1. Anchored whole-molecule mutation (BAM-style)

**Seeds.** Use the engine-A spectral analogs (`r['ana']`: pool index and shifted similarity, ±200 Da window). Keep only analogs whose **exact precursor-mass difference** to the query's neutral mass, centred with the +1.45 ppm offset, matches a palette transform combination of ≤ 2 steps within ±3 mDa.

**Mutation.** Mutate the **full analog structure**, not its Murcko scaffold. Its side chains are kept, and ring-adding transforms (+hexose) create a new scaffold. This gets past the unique-scaffold wall for one-step relatives.

**Target formula.** It becomes `formula(analog) + Δ` and no longer depends on PubChem's top formula. The existing scaffold-harvest path stays as a fallback source at lower priority.

**Code changes:**
- New `match_mass_delta(dm, tol)` over a mass-indexed `build_reach`.
- `mutate_sources` gains an `anchored` source type that skips `MurckoScaffold`.

### I2. Inverse transforms (bidirectional palette)

Add subtractive SMARTS:

| Transform | Reaction |
|---|---|
| `-hexose` / `-deoxyhexose` | O-glycoside → OH |
| `-CH2` | OMe → OH, NMe → NH |
| `-C2H2O` | OAc → OH |
| `-O` | aliphatic OH → H |
| `+2H` | C=C → C–C |

They go into `PALETTE` with negative delta vectors. `build_reach` must search signed combinations, keeping `MAXOPS` for the scaffold path and ≤ 2 steps for the anchored path.

### I3. Exact-mass delta evidence

The gate's `delta_steps` is now measured on the **anchored analog's mass delta** (I1), not formula-to-formula. A shifted analog match (sim ≥ τ) whose Δm hits a transform within 3 mDa is positive evidence of a one-edit relative, the STEP/BAM signal in the knowledge base. This fixes the "steps measured from the wrong thing" caveat.

### I4. Two-stage gate with competition vetoes (the "confident only" requirement)

**Stage 1** runs in `mut_runner`, before the base. It uses engine-A `lib_exact`, `ana_max`, margin, and the I3 steps, and decides whether to generate at all. After generation it applies the **I0 fingerprint tests**:

| Tier | Conditions |
|---|---|
| decline | best `Δfz ≤ 0`, or the per-anchor percentile is below the top 5% |
| `likely` (at most) | `fz(B_best) > fz(A)` and `fz(B_best) ≥ best_pool_fz` |
| `sure` candidate | additionally `fz(B_best) ≥ fz_top` (B beats every database candidate on the predicted fingerprint) |

All three are within-molecule comparisons, so nothing needs an absolute calibration.

**Stage 2** is new and runs in the fusion cell, after the ICE/GL scores exist. It re-runs `c3_gate.classify_v2` with:

| Check | Type | Rule |
|---|---|---|
| Real v4n `lib_max` | veto | `lib_max ≥ 0.90` → decline |
| Retrieval consensus | veto | v4n top-1 IK14 == engine-A top-1 IK14 → decline `sure`, demote to `likely` |
| Forward-sim competition | — | compare raw ICE/GL similarity (in [0,1]) of the best generated candidate against the retrieved top-1. `sure` requires the stage-1 `sure` candidate (I0 beats `fz_top`), gen > retrieved on ICE **and** on GL where GL exists, plus `ana_max ≥ 0.70` and margin ≥ 0.30. `likely` requires gen ≥ retrieved − 0.05 on ICE. With this, `sure` (rank 1) needs FPNet, ICEBERG, GLACIER and the analog evidence to agree independently. |
| ICE veto | veto | `ice_max(retrieved top-1) ≥ ICE_MAX` → decline (the existing `ICE_MAX` veto, now actually wired in) |
| **Fire-rate cap** | scale-free guard | at most `SURE_MAX_FRAC = 3%` of molecules can be `sure` and `FIRE_MAX_FRAC = 12%` can fire at all. Ranked by a confidence score (`ana_max − exact_max` + ICE advantage); the excess is demoted or declined. A bio2-style misfire storm (−0.064) cannot happen. If more than 25% of molecules would fire, the whole channel is disabled and the log says so. |

**Slots:** `sure` → [1,2,3], `likely` → [2,3,4]. At most 3 generated candidates per molecule, inserted with the existing `merge()` (S:403), which deduplicates by IK14.

### I5. Plumbing fixes, ported from M = `scratch/casmi26-v4n-mutator-c3-submit.py` onto 0.415

- **eng_runner export.** Write `eng_lists.json.c3.json` = `{mid: {target, lib_exact, ana: [[smiles, mass, sim] × 20]}}`, as in M:137.
- **RUNNER export.** Save `pc_z.npz` (FPNet z) for the f·z screen (M:158), keeping the V17 patch and `fz_top` intact.
- **Mutator cell.** CPU subprocess after the PubChem cell, as in M:184-262: global deadline 30 min, 45 s per molecule. Writes `mut_lists.json` with per-molecule signals and the stage-1 tier.
- **ICE input** (next to S:303-315). Add ≤ 25 generated candidates per **stage-1-fired** molecule with the formula `F|gen`. Raise `ICE_BUDGET` to 7200 and `GL_BUDGET` to 5200; the ~16,700 s of slack covers both.
- **Fusion cell.**
  - Order each generated list by `z(fz) + z(ICE) + z(GL)`, computed within the generated list.
  - Run stage 2 and the cap, then splice after S:486 with `merge`.
  - Change the cell guard at S:470 to `if ENG or GEN`.
- **Base-neutral constraint.** Leave `_form2` untouched for v4n's own generated candidates. Fixing that changes base behaviour and would need its own probe.
- **Safe fallback.** Every step sits inside a `try/except`; any failure leaves the 0.415 output unchanged.

### I6. Counters printed in every run (judge LB results by these, given ±0.004 noise)

- **Stage 1:** stage-1 fires, with the decline reasons funnel.
- **Stage 2:** stage-2 tiers (sure / likely / declined, by reason); cap demotions; channel-disabled flag.
- **Generation:** anchored versus scaffold seeds; inverse transforms used.
- **Rank changes:** number of molecules whose top-1 changed through the splice.
- **Bound on LB effect:** `max |ΔMRR| = (n_sure·1 + n_likely·0.5)/400`. An LB move larger than this bound is noise.

## Files

| File | Change |
|---|---|
| `build/delta_m_mutator.py` | `INVERSE` palette entries, signed `build_reach`, `match_mass_delta`, `anchored` source in `mutate_sources`. Existing behaviour stays the default, so the old tests still pass. |
| `build/c3_gate.py` | `classify_v2(...)` (stage 2) and `cap_fires(rows, sure_frac, fire_frac, disable_frac)`. `classify` is unchanged. |
| `build/mut_runner.py` | Anchored mode reads analog masses from `.c3.json`, plus the signals dump. Adds I0: `fz(anchor)` from the same two-pass screen, `Δfz` and the per-anchor percentile for each product, and the in-molecule comparisons against `best_pool_fz`/`fz_top` (read from `pc_lists.json`). Fixes the `classify(exact, exact, …)` call to pass `exact_max` correctly. |
| **New** `build/patch_c3v2.py` | Generator from the 0.415 notebook to `casmi26-c3v2-<tags>.ipynb`. Reuses `find`, `must_replace`, `set_src` and `src_of` from `build/patch_supernext.py`, and the embedding pattern (`repr()` into `MUT_FILES`) and cell code from `build/patch_v4n_mutator.py --c3`. Flags: `--anchored`, `--inverse`, `--sure-frac`, `--fire-frac`, `--no-rank1`, `--ice-budget`. |
| **New** `build/patch_c3cal.py` | Calibration notebook generator (see below). |
| Tests | Extend `build/test_delta_m_mutator.py`, `test_c3_gate.py`, `test_mut_runner.py` and `test_mut_cell.py`. Add a new `build/test_patch_c3v2.py`, modelled on `test_patch_v29.py`: compile, cell count, only intended cells touched, `'|gen'` present in the ICE input, stage-2 and cap exec'd against stubs, base-neutral when `GEN` is empty. Register it in `build/run_all_tests.sh`. |
| Specs | New `specs/track4_priors_probes/c3v2_plan.md`; pre-registered rows in `gate_report.md` written **before** any run; `summary.md` updated (base 0.415, noise floor ±0.004, P1/P2/P3 closed). |

## Parallel track A: LB notebooks

| Probe | Content | Keep rule |
|---|---|---|
| **C1** | Full v2 channel: anchored + inverse + two-stage gate + caps, with `sure` → [1,2,3]. Thresholds as above (Track-1 values, tightened for `sure`). | The ±0.004 noise floor is used only on the counters' bound: keep if LB ≥ 0.415 **and** the counters show fires; flag as noise if the LB move exceeds the bound. |
| **C2** | Thresholds updated from calibration, once track B reports. | Same rule. |
| **C3** | Optional `--no-rank1` attribution: does the `sure` → rank-1 policy pay? | Same rule. |

**Kill rule:** if C1 shows `top-1 changed` > 0.05·400 or the channel-disabled flag, investigate before submitting anything else.

## Parallel track B: calibration notebook (`casmi26-c3cal.ipynb`, Kaggle, commit run only)

**Pseudo-test**, built from `train.parquet` rows listed in the local cohorts. Only the cohort parquets are shipped as a small dataset; there is no `.parquet` in git (`.gitignore`).

| Cohort | Role | Handling |
|---|---|---|
| `p1_novel/pseudo_c3.parquet` (600) | positives | ban their library rows (`lib_ban_keys`) and pool keys (`pool_ban_keys`) in engine A's library and pool |
| `p1_novel/c2_displace.parquet` (400) | Class-2 negatives | library ban only |
| ~200 random train molecules | Class-1 control | no bans |

**Pipeline:** runs engine A (with bans), then mutator v2 and stage 1, then ICE/GL on the engine top-40 ∪ generated candidates, then stage 2. It dumps one row per molecule: all signals, the tier, the truth's rank in the engine list and in the generated list, and whether it lands in the top 3 after the splice.

**Metrics, pre-registered in `gate_report.md`:**
- Fire rate on Class-1 controls ≤ 1%. `sure` on Class-1 = 0.
- Fire rate on `c2_displace` ≤ 3%, and displaced top-1 truths ≤ 0.5% of the cohort.
- On `pseudo_c3`: fire rate, precision, and recovery@3 when fired.
- **I0 check:** across the positives, how often `fz(truth) > fz(anchor)` holds, and the truth's percentile among the anchor's edits. This tests the user's hypothesis directly. If it holds in under ~70% of truths, loosen the I0 tests to a ranking signal rather than a veto.
- Anchored vs scaffold recovery, and the inverse-transform share.

**Decision rule:** threshold settings must meet the payoff condition `q₃ > 1.25·a` from `c3_routing.md` §7 before they are moved into C2.

**Caveat (memory `casmi-val-blind-to-pool-misses`):** these cohorts aren't timsTOF-matched, so treat the numbers as relative. Use them to set thresholds, not to predict the LB.

## Verification

1. **Local tests.** `python build/test_delta_m_mutator.py`, `test_c3_gate.py`, `test_mut_runner.py` and `test_patch_c3v2.py`, then `sh build/run_all_tests.sh`. Everything must pass except the known `test_stage_d_colab.py` failure.
2. **Local oracle.** Run `python build/eval_mutator_cohort.py pseudo_c3 c2_displace` with the new palette (signed reach, inverse transforms) to confirm the reach and recovery deltas from I2 before any Kaggle run. Add an `--oracle-z --tau X` mode for the I0b pruning-loss sweep, and record the chosen τ in `gate_report.md` before the Kaggle runs.
3. **Cell diff.** `python .agents/skills/analyze-notebook/scripts/compare.py casmi26-v29-fusion-pop-glmh-0-415.ipynb casmi26-c3v2-<tag>.ipynb` must show only the intended cells: C1 budgets, the C3 export, the C4 `pc_z` export, the new mutator cell, the ICE input and the fusion cell.
4. **Kaggle save (GPU on).** Check that the log shows `cuda` in the ICE meta, the stage-1/2 counters, and that the channel ran on the 12 smoke molecules. Smoke data is a placeholder, so don't read any thresholds from it. Check `submission.csv` is (400, 2) with ≤ 25 unique IK14 per row.
5. **Record results.** Log every LB result and the calibration numbers with `/update-experiments`, and fill the `gate_report.md` rows.

---

## Implementation record (2026-10-03)

### What was built

| File | Contents |
|---|---|
| `build/delta_m_mutator.py` | Additive v2 section; the v1 defaults are unchanged (the reach table still has 114,854 deltas).<br>• `EXTRA_PALETTE`: pentose, glucuronide, malonyl, coumaroyl, caffeoyl, feruloyl, galloyl.<br>• `MACRO_PALETTE`: `+O-hexose`, `+O-dhexose`, `+O-acetyl`, `+O-prenyl`.<br>• `REVERSE_SMARTS`: `+2H`, `-H2O`.<br>• `_excise`: every other inverse, by cutting one acyclic single bond and requiring the exact formula.<br>• `V2_SITES`: sp3 / sp2 / N–H attachment sites.<br>• `build_reach_signed`, `mass_index`, `match_mass_delta`, `grow_signed` (all move orders), `mutate_anchored`. |
| `build/c3_gate.py` | `stage1_v2` (I0 within-molecule f·z tests), `classify_v2` (stage 2, with an `ice_unit` guard), `cap_fires`. The v1 `classify` is unchanged. |
| `build/mut_runner.py` | `MUT_V2` path `mutate_one_v2`: anchors from `.c3.json` via the exact-mass Δ (+1.45 ppm centred), a scale-free τ, the optional consensus core (`MUT_BETA`), `sig` exported for stage 2.<br>Calibration-only switches: `MUT_KEEP_DECLINED`, `CASMI_BAN_KEYS`. |
| `build/patch_c3v2.py` | Generates `casmi26-c3v2-<tag>.ipynb` from 0.415. It changes 5 cells (config, engine export, PubChem `pc_z` export, ICE union with `\|gen`, fusion splice) and adds the mutator cell. |
| `build/patch_c3cal.py` | Generates `casmi26-c3cal.ipynb` (track B). |
| `build/eval_anchored_oracle.py` | Local round-trip oracle (I0/I0b/I1/I2). |
| Tests | `test_c3v2.py` (12 groups; every one of the 62 signed moves checked against its vector), `test_mut_runner_v2.py` (the real runner subprocess with a mocked `casmi`), `test_patch_c3v2.py` (structure, then the splice executed against stubs), `test_patch_c3cal.py` (the data builder run on a synthetic `train.parquet`). All are registered in `run_all_tests.sh`; the suite is green apart from the already-broken `test_stage_d_colab.py`. |

### Deviations from the plan above

1. **v2 is anchored-only.** Scaffold harvesting (production recovery 4.2%) is not used as a fallback in v2. Anchored edits keep A's core by construction, so the I0c substructure core filter (scaffold-harvest only) is not implemented. The consensus core prior is implemented, but `MUT_BETA` defaults to 0 until calibration shows it helps.
2. **Pruning tolerance is scale-free.** The floor is `score(anchor) − MUT_TAU·mean|z|`. The oracle zero-loss point is about 6.3 in these units at both σ=1 and σ=3, so the default is `MUT_TAU=8`.
3. **More attachment sites (`V2_SITES`).** Added after the first oracle run. Depth-1 "up" recall went from 66.0% to 85.7%: excision can remove a group from any single bond, but the forward SMARTS only re-attached at aromatic CH, OH and NH.
4. **The oracle is a separate script**, `eval_anchored_oracle.py`, rather than an `--oracle-z` flag on `eval_mutator_cohort.py`. Its relatives are synthetic (truth ± 1–2 moves), so it measures generator completeness and pruning loss, not a recovery rate.
5. **The calibration notebook covers stage 1 only.** It skips the PubChem probe (so `fz_top` is None and `sure` is an upper bound) and the v4n/ICE/GL/fusion cells. Stage 2 is read from the C1 LB counters.
6. **Stage 2 applies the absolute `ICE_MAX` veto only when the run's ICE scores are verified to lie in [0,1].** ICEBERG's `fuse.py` is not in the repo, so its scale is unverified. The gen-vs-retrieved comparisons don't depend on scale.

### Offline results (`build/eval_anchored_oracle.py`, `pseudo_c3`, direction "up")

| run | recall (no prune) | recall at chosen τ | truths lost to pruning | median candidates |
|---|---|---|---|---|
| depth 1, σ=1, n=147 | 85.7% | 85.7% | 0 | 11 → 3–4 |
| depth 2, σ=1, n=88 (before the sp2/N sites) | 60.2% | 73.9% (raw τ 8) | 0 | 130 → 34 |
| depth 2, σ=3, n=88 (re-run with the final palette: identical) | 60.2% | 71.6% (raw τ 16) | 0 | 130 → 66 |

Pruning *raises* recall: the old alphabetical beam was cutting the truth's own intermediates.
