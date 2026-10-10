# Track 4b: our own natural-product signals for the PubChem channel (summary)

Status as of 2026-10-05: **CLOSED, nothing kept.** The offline benchmark passed two signals, but all three LB probes scored **0.399** against a base mean of 0.400 (keep bar 0.404).

Detailed records:
- `specs/track4_priors_probes/np_signals_plan.md`: the plan and status log.
- `specs/track4_priors_probes/gate_report.md`: the pre-registered gates and measured numbers ("Track 4b register").
- `EXPERIMENTS.md`: section B4 and the Graveyard.

---

## 1. The problem

- Test molecules are natural products (NPs) from biological extracts.
- A mass search over the **105.9M-row PubChem tier** returns hundreds to thousands of same-mass isomers. Most are synthetic screening compounds, so the true NP gets diluted.
- The only spectrum-blind lever that had ever paid was the **borrowed popularity prior**, `pop = log1p(SID) + log1p(PMID)` from the public Gluzdov arrays.
- **Goal:** build NP signals from our own datasets and code, as extra additive terms on top of pop (not replacing it), and test them on the LB cheaply (2–4 submissions).

## 2. The plan (settled in the grilling session)

| Decision | Choice |
|---|---|
| Scope | New NP signals only. The popularity arrays are kept unchanged. |
| Candidate signals | (1) **vendor-only flag**: every PubChem depositor is a "Chemical Vendor" and PMID = 0 (penalty). (2) **Ertl NP-likeness** (RDKit `Contrib/NP_Score`). (3) **Class consistency**: NPClassifier superclass agreement with the molecule's analogue hits. |
| Order | vendor → NP-likeness → class consistency. A signal is killed as soon as it fails its gate. |
| Where it applies | PubChem channel first. The fused list only after a PubChem-channel win. |
| Form | additive `z(spectral) + μ·signal`. Rule 5: μ ≤ 0.5 on a signal's first appearance. |
| Labels | NP knowledge table (`np_superclass`), plus kNN to the nearest labelled NP (Tanimoto ≥ 0.7, Morgan r2, 2048 bits). Unlabelled = 0, never a penalty. NPClassifier on all candidates is the deferred fallback. |
| Build venue | Colab + Drive, using the datasets already on Drive. |
| Offline gate | Same-formula PubChem-tier window, ranked by `pop + μ·s` vs `pop`. KEEP if ΔMRR@25 ≥ +0.010 on cohort L (NPs without training spectra) **and** ≥ −0.005 on cohort T (training structures). |
| LB base | **supernext** (`evenda-casmi-inferencesupernext.ipynb`): our own runs scored **0.399 and 0.401**. The v29 "0.415" notebook scores 0.393 from our account. The 0.405 in the ledger is the original author's score. |
| LB keep bar | ≥ mean of our unchanged supernext runs (including a paired control) **+ 0.004** (noise floor), **and** the signal's log counters are non-zero. |

## 3. What has been built

| File | What it is |
|---|---|
| `build/np_signals_core.py` | Shared core: tie-aware expected-rank metric, vendor flags from PubChem SID-Map, NP_Score, kNN labels, class consistency, tier/formula window helpers, `eval_query` (one benchmark query) |
| `build/test_np_signals_core.py` | 37 local checks (in `run_all_tests.sh`) |
| `build/np_signals_colab.py` → `casmi_np_signals_colab.ipynb` | Colab build + benchmark: vendor arrays, LOTUS fingerprint bank, train labels, cohorts T/L, gates NB.* (ran: 9,811 s) |
| `build/np_signals_realism_colab.py` → `casmi_np_signals_realism_colab.ipynb` | Realism check: channel-faithful cohort P plus the `member@μ` and `stack@μ` variants (gates R.1–R.4) |
| `build/mk_np_signals.py` | Generates both Colab notebooks (core embedded verbatim; never hand-edit the .ipynb) |
| `build/patch_supernext.py` (`--nplike`, `--classcons`) | Patches supernext. After the ICE/GL rerank, the NP block re-orders the PubChem top-25 list. `pc_fz[0]` is kept, so the spectral merge gate (`rel`) is unchanged. Wrapped in try/except: on failure it prints `TRACK4b NP signals FAILED -> no-op`. |
| `build/test_patch_supernext.py` | 92 checks, including an executable NP-block test: a flavone climbs from 3rd to 2nd above sulfonamides and the top-1 is kept |
| LB notebooks | `casmi26-t4-base.ipynb` (paired control), `casmi26-t4b-nplike0.5.ipynb`, `casmi26-t4b-classcons0.5.ipynb`, `casmi26-t4b-nplike0.5-classcons0.5.ipynb` (stack, on hold) |
| Drive outputs (`MyDrive/casmi_2026/_np_signals/`) | `pc_vendor_only.npy`, `pool_vendor_only.npy`, `lotus_fp.npy`, `lotus_superclass.npy`, `lotus_ik14.npy`, `train_labels.parquet`, `np_bench_rows.parquet`, `np_bench.json`, `MANIFEST.json` |

Ledger housekeeping:
- **EXPERIMENTS row 24:** corrected; our own supernext runs are 0.399/0.401.
- **Row 29:** A1 precut re-read as neutral.
- **Rows 32–33:** np1/npx +0.004 marked as noise.
- **Section B4:** added.
- **Graveyard:** vendor flag added.

Fixes along the way:
- Colab RDKit wheel was cp312 on Python 3.13. Fixed by matching the Python tag and falling back to PyPI 2026.03.3.
- The realism notebook expected a `pool_ik14.npy` that doesn't exist. Pool keys now come from `casmi26-v2-pool/pool_meta.parquet` if it is on Drive; otherwise from the NP table's `in_casmi_pool` rows plus the training keys.

## 4. Results so far

### Offline benchmark (cohorts of 2,000; truth found in its window: T 93.2%, L 97.7%)

| Gate | Signal | Result | Verdict |
|---|---|---|---|
| NB.V0 | vendor categorisation sanity check | 0.12% of NP-table compounds flagged (bar < 1%) | PASS |
| NB.V | vendor-only penalty | μ 0.5: L **+0.0005**, T −0.0006 | **KILLED**: redundant with pop (vendor decoys already rank below documented NPs). 61.7% of T truths are themselves vendor-only. |
| NB.N | Ertl NP-likeness | μ 1.0: L +0.0316, T −0.0105 (fails the T bar). **μ 0.5: L +0.0245, T −0.0042** | **PASS at μ 0.5 (narrow)**, under amendment A-NB.1 (rule 5 caps μ at 0.5) |
| NB.C0 | class-consistency coverage | 55.0% of queries have a labelled analogue (bar 50%); 50.7% of windows have ≥ 2 superclasses (bar 20%) | PASS |
| NB.C | class consistency | μ 1.0: L +0.0345, T +0.0030. **μ 0.5: L +0.0194, T +0.0016** | **PASS at μ 0.5** |

Caveats (why the LB is still the arbiter):
- The L gains are optimistic. The Ertl model and the NP table share NP-database provenance.
- The class-consistency analogues were structural oracles (nearest training structure), not real spectral hits.
- 90% of L truths carry a table label, versus 8% of T truths.
- The benchmark window includes pool members, which supernext's PubChem channel never lists. The realism check (cohort P) fixes that.

On keeping NP-likeness despite the T loss:
- Test molecules are NPs, so cohort L is closer to the target than T.
- The decision still rests on the pre-registered rule, μ 0.5 under rule 5, not on discounting T after the fact.

### LB probes: done (10-05)

| id | Notebook | LB | Verdict |
|---|---|---|---|
| T4B-0 | `casmi26-t4-base.ipynb` (control) | not reported | base mean 0.400 (our runs 0.399 / 0.401) |
| T4B-N | `casmi26-t4b-nplike0.5.ipynb` | **0.399** | NOT KEPT |
| T4B-C | `casmi26-t4b-classcons0.5.ipynb` | **0.399** | NOT KEPT |
| T4B-S | `casmi26-t4b-nplike0.5-classcons0.5.ipynb` | **0.399** | NOT KEPT |

**Why the offline gain did not show up.** Three different re-orderings scoring the same to three decimals means one of two things:
1. The block silently did nothing on the hidden run. The commit log would show `TRACK4b NP signals FAILED -> no-op`.
2. More likely, the lever's **reach** is tiny:
   - The terms only re-order supernext's PubChem-only list.
   - That list only fills a few merge slots, and only when spectral evidence lets it (`rel` / `lib_max` gates).
   - So only test molecules whose truth is PubChem-only and missing from the pool can move.
   - The offline +0.02 was an MRR over exactly such truths. Scaled by their small share of the test set, the expected LB move is about 0.001, under the ±0.004 noise floor.

**Lesson:** estimate a lever's reach (the share of test molecules it can change) before trusting an offline cohort gain.

### Realism check: done (10-05). It explains the LB null

Cohort P = NP-table compounds missing from the pool: the population the PubChem channel exists for. Pool members are removed from each window.

| id | Question | Measured | Verdict |
|---|---|---|---|
| R.1 | Does class consistency carry more than "is a known NP"? (classcons@0.5 − member@0.5, on L; bar +0.005) | +0.0028. On P, membership alone does better (+0.0197 vs +0.0181) | **FAIL**: class consistency is effectively the NP-membership bonus |
| R.2 | Class consistency on P | +0.0181 | PASS |
| R.3 | NP-likeness on P | +0.0219 | PASS |
| R.4 | Stack vs the better single, on P | +0.0178 (stack +0.0397) | PASS (moot) |

**What it means:**
- **Only 21.3% of the P truths exist in the PubChem tier at all.** The other ~79% of pool-missing NPs can never be listed by the PubChem channel, whatever the ranking.
- For the 21% that are there, the windows are small (median 51) and `pop` already ranks them well (MRR 0.431).
- The NP terms do help those (+0.02), so the signals have the right sign. But they act on a small slice of a small slice of the test set: expected LB change ≈ 0.001, matching the observed 0.399.
- The class signal is just "is a known NP" (R.1), so it duplicates the old np1 bonus.
- **The bottleneck for pool-missing NPs is candidate coverage, not ranking.** If this direction is ever reopened, it should *add* the missing NP-table structures as candidates (as P3e / `npx` did on the v29 base: 0.397 vs a 0.393 control, within noise), not re-rank PubChem.

## 5. Outcome and what is left

- **Track 4b is closed.** No NP signal is kept. The supernext base (about 0.400) stays the reference.
- **Optional, cheap:** read one probe's commit-run log line (`TRACK4b NP signals | ... reordered / top1_changed`) to tell a no-op from a real but tiny effect. It doesn't change the verdict.
- **Not pursued:**
  - `--fused` (applying the term to the whole fused list, which has wider reach);
  - NP terms in the pool list (already mostly NPs);
  - NPClassifier labels.
  - Re-open only if a later track needs a final-list prior.
- **Next:** Track 2 (de novo generation) and Track 3 (MS simulator).
