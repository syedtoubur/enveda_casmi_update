# Track 4 — gate report (pre-registered)

*Registered 2026-10-02, BEFORE any Track 4 probe was run. Base = `evenda-casmi-inferencesupernext.ipynb`, 0.405 public LB.*

## Decision rules (apply to every LB probe)

1. One change per probe, on top of the current best notebook.
2. **KEEP** if LB >= best + 0.001. Ties -> keep the simpler variant. **REVERT** anything below best.
3. Two consecutive flat/negative probes inside a wave -> move to the next wave.
4. Rerun log must print `top-1 changed in N`, list overlap vs the base, and `pop stats`
   (min / median / p99 of the prior term). A score move with no counter change is noise.
5. No new signal at weight 1.0 on first appearance (R4). Forward λ stays <= 1.0.
6. Commit runs (12 molecules, placeholder test) verify plumbing and cost only — never
   calibrate an absolute threshold on them (`casmi-saved-outputs-are-placeholder`).

## Probe register

| id | change | expected ΔLB | pre-registered keep bar | LB | verdict |
|---|---|---|---|---|---|
| A1 | prior applied before the `TOPN` cut (`--pop-precut`) | +0.002…+0.006 | >= 0.406 | **0.401** (10-03) | **FAIL → REVERT** (−0.004 vs 0.405) |
| A2 | prior applied to the fused list before the ICE/GL rerank (`--pop-fused`, μ_f 0.15) | +0.001…+0.004 | >= best+0.001 | | |
| A3 | μ sweep {0.08, 0.30} on the A1/A2 winner | ±0.003 | >= best+0.001 | | |
| A4 | class-1 top-1 freeze, `lib_max >= 0.9` (`--c1-freeze`) | 0…+0.003 | >= best+0.001 | | |
| A5 | (attribution) `POOLPOP_MU = 0`, `'|gen'` kept | — | not a keep candidate | | |
| B1 | popularity-ordered PubChem-only channel (`--pc-pop`) | +0.003…+0.010 | >= best+0.001 | | |
| B2 | PubChem slots 6–8 for low-`rel` molecules (only if B1 kept) | +0…+0.003 | >= best+0.001 | | |
| B3 | richer prior: pmid + patent + lotus_plant + in_pubchemlite | +0.001…+0.004 | >= best+0.001 | | |

## Wave C gates (MS-GPT class-3 channel) — measured offline before any LB probe

| gate | cohort | metric | threshold | measured | verdict |
|---|---|---|---|---|---|
| GC4.0 (kill) | `p1_novel/pseudo_c3.parquet` | exact IK14 @25 | >= 12% (DreaMS arm: 8.17%) | | |
| GC4.1 | same | seconds / molecule on T4 | <= 20 s | | |
| GC4.2 | — | encoder weights trained on MassSpecGym only | must hold | | |
| GC4.3 (displacement) | `p1_novel/c2_displace.parquet` | MRR@25 loss with routing on | <= 0.005 | | |

## Track 4 v2 probe register (base: 0.411 `casmi26-v29-glacier-m-h-full-fusion-0-411.ipynb`)

Pre-registered 2026-10-02, before any submission. Rules 1–6 above still apply. The keep bar is **≥ best + 0.001**.
The A1–A5 / B1–B3 rows above were built on the 0.405 base and are superseded: A3 was done by the 0.409 notebook, B1 by 0.411, and A4 by 0.409 (it measured 0.000).
Notebooks come from `build/patch_v29.py` (`casmi26-t4v2-<tag>.ipynb`).

| id | flag / notebook | change | counter that must move | expected ΔLB | keep bar | LB | verdict |
|---|---|---|---|---|---|---|---|
| P1 | `--mu 0.25` / `casmi26-t4v2-mu0.25` | pool prior μ 0.15 → 0.25 | `pop stats` p99 rises ×1.67 | +0.003…+0.005 | ≥ 0.412 | **0.399** (10-03) | **FAIL → REVERT** (−0.012 vs 0.411). Popularity already saturated on 0.411; see EXPERIMENTS row 30. Submitted with GPU (user-confirmed), so not a hardware artefact |
| P2 | `--pop-eng 0.25` / `casmi26-t4v2-popeng0.25` | engine-A list re-sorted by z(−rank) + 0.25·pop before `fuse2` | `engine lists re-sorted` > 0, `top-1 moved` > 0, `pop key hits` ≥ 50% | +0.002…+0.005 | ≥ 0.412 | | |
| P3 | `--np-bonus 1` / `casmi26-t4v2-np1` | pop += 1·[LOTUS or NPAtlas] in the pool and PubChem priors | both `np flags ... found` lines | +0.002…+0.006 | ≥ 0.412 | | |
| ~~P3+~~ | ~~`--np-bonus 1 --expand-pool 1`~~ | **SPLIT 10-03** into P3e and P3/0.415 below: bundling broke rule 1, and its BASE-cell expansion code could not reach the top 25 (pid −1 = "generated" slot, no ranker score) | — | — | — | — | not submitted |
| **P3e** | `--src <0.415> --np-expand` / `casmi26-t4v3-npx` | the LOTUS/NPAtlas structures with `in_casmi_pool == False` (150–1250 Da) join **engine 2's pool** next to BIO (same `build_pool` → `fp_and_mass` path; ChEBI/LIPID MAPS paid +0.009 there) | engine log `TRACK4 NP expansion +N` with N > 0; fusion line `TRACK4 NPX ... in final top-25 B` with B > 0; `ENG duplicate keys` ≈ 0 | 0…+0.006 | ≥ 0.416 **and** B > 0. An LB move larger than the printed `dMRR bound` is noise. VOID if N = 0 or the keys file is absent | **0.397** (10-03) | **VOID, pending a base control run**: same score as the unrelated P3 below, which points to a shared base offset (public vs private popularity arrays). See EXPERIMENTS rows 32–33 |
| **P3 / 0.415** | `--src <0.415> --np-bonus 1` / `casmi26-t4v3-np1` | P3 rebased onto 0.415 (identical code to the 0.411 P3) | both `np flags ... found` lines | 0…+0.004; risk: row 30 says popularity is saturated | ≥ 0.416 | **0.397** (10-03) | **VOID, pending a base control run** (as for P3e) |
| P4 | `--pop-precut` / `casmi26-t4v2-precut` | pool prior over the top 200, then cut to 60 | `promoted into top 60` > 0 | +0.001…+0.004 | ≥ best + 0.001 | — | **DROPPED**: A1 is the same C7 change and scored −0.004 on 0.405 (10-03). Do not submit. |
| P5 | second point on the winning knob (μ 0.35 or G 2) | — | — | ± | ≥ best + 0.001 | | |
| P6 | stack of the winners (+ optional `--lib-gate` rider) | — | all of the above | sum of the winners | ≥ best + 0.001 | | |

**Notes:**
- **P2 no-op case.** If `pop key hits` is about 0, the engine keys and the pool keys don't match, so P2 was a no-op. Record it as VOID, not FAIL.
- **P3 no-op case.** If `np flags ... absent` is printed, the attached popularity dataset is Version 1. P3 is then VOID; do not submit it.

## Track 4 c3v2 register: gated Class-3 mutator v2 (base: 0.415 `casmi26-v29-fusion-pop-glmh-0-415.ipynb`)

Pre-registered 2026-10-03, **before** any Kaggle run of the c3v2 or c3cal notebooks. Plan: `c3v2_plan.md`.
Same-code LB noise is about ±0.004 (0.411 and 0.415 come from identical code), so a single LB number can't confirm a +0.001…+0.004 effect. Every LB verdict is read together with the run's own counters.

### Offline (local, before Kaggle): generator and pruning on `pseudo_c3` round trips (`build/eval_anchored_oracle.py`)

| gate | metric | threshold | measured | verdict |
|---|---|---|---|---|
| CO.1 | depth-1 "up" round-trip recall (the anchor is the truth minus a group) | ≥ 70% | 85.7% (n=147) | PASS |
| CO.2 | pruning loss at the chosen τ, depth 2, σ=1 and σ=3 | ≤ 2% of recoverable truths | 0 at τ = 8·mean\|z\| (σ=1: raw τ 8, 0 lost; σ=3: raw τ 16, 0 lost) | PASS |
| CO.3 | the I0 hypothesis under the oracle: fz(truth) > fz(anchor) among recovered truths | ≥ 70% | 96.9% (σ=1), 90.5% (σ=3) | PASS (oracle only; real FPNet in CC.6) |

### Calibration (`casmi26-c3cal.ipynb`, Kaggle commit run, never submitted)

Cohort: 300 `pseudo_c3` (library and pool banned), 200 `c2_displace` (library banned, structure kept in the pool), 100 Class-1 controls (nothing banned). The PubChem probe is skipped, so `sure` is an upper bound.

| gate | metric | threshold | measured | verdict |
|---|---|---|---|---|
| CC.1 | Class-1 stage-1 fire rate | ≤ 1% | 0.0% (0 / 100) | PASS |
| CC.2 | Class-1 `sure` count | 0 | 0 | PASS |
| CC.3 | `c2_displace` fire rate | ≤ 3% | 3.5% (7 / 200: 1 sure, 6 likely) | **FAIL** (re-read 10-03: the bar is ≤ 3%; it was first logged as "marginal") |
| CC.4 | `c2_displace` fired with engine top-1 correct (displacement) | ≤ 0.5% of the cohort | 2.5% (5 / 200 fired with `eng_rank == 1`: 1 sure, 4 likely) | **FAIL** (re-read 10-03: the metric counts every fire. The first log counted only `sure` and called it PASS, which changes the metric after the fact. A `likely` fire still pushes ranks 2–4 down 3 places) |
| CC.5 | `pseudo_c3` q3 given fired vs exposure: q3\|fired > 1.25 × CC.3 | must hold | 52.5% (21 / 40) vs 4.4% threshold | PASS (12× margin) |
| CC.6 | I0 with real FPNet z: fz(truth) > fz(top analog) on `pseudo_c3` | ≥ 70%, else I0 becomes a ranking signal instead of a veto | **71.4%** (20 / 28 recovered truths that share the best candidate's anchor beat that anchor, which is the quantity `stage1_v2` compares); 71.9% (23 / 32) against the top analog `fz_anchor0` | **PASS, narrowly** (re-derived 10-03). The first log's 65.6% = 21/32 was the number of truths that *passed the gate*, which is circular. The loosening rule does not trigger. 9 truths were still vetoed by `no_fp_improvement`, and all 9 beat `best_pool_fz` |

**Status (10-03).** CC.3 and CC.4 fail, so the rule below says *tighten*, not loosen. No C2 until C1 has been attributed (see C0/C1-null).

**Decision rule.** C2 (the recalibrated LB probe) ships only with thresholds that pass CC.1–CC.5. If CC.1 or CC.3 fails at the default thresholds, tighten `C3_ANA_MIN` / `C3_MARGIN_MIN` until both pass, and re-check CC.5.

### LB probes (from `build/patch_c3v2.py`)

| id | notebook | change vs 0.415 | counters that must be read | expected ΔLB | keep rule | LB | verdict |
|---|---|---|---|---|---|---|---|
| **C0** | `casmi26-v29-fusion-pop-glmh-0-415.ipynb` **unchanged**, submitted from our account with our attached datasets | none: the base control (memory `casmi-imported-base-unmeasured`). Our runs of this code have scored 0.397–0.399, while 0.415 is the authors' private-array run | — | — | becomes the base for C1 and C1-null. Pre-registered 10-03, before submission | | |
| C1 | `casmi26-c3v2-tau8.ipynb` | full v2 channel: anchored + signed moves + exact-mass Δ + τ-pruned f·z beam, stage 1 + stage 2 + cap; sure → [1,2,3], likely → [2,3,4] | `C3V2 STAGE 1` fired / tiers; `C3V2 STAGE 2` funnel; `C3V2 CAP`; `C3V2 SPLICE` n, top-1 changed, `max \|dMRR\| bound` | +0.000…+0.006 | keep if LB ≥ 0.415 **and** splice n > 0. An LB move larger than the printed bound is noise. Kill: `top-1 changed by splice` > 20 or `CAP disabled` → investigate before anything else | **0.393** (10-03) | **FAIL vs the 0.415 bar; ΔLB UNATTRIBUTED.** Against our own runs of the base (0.397–0.399) it is −0.004…−0.006, at the noise floor. The hidden-run counters can't be read. The probe also changed two things at once (splice, plus budgets and the `\|gen` ICE/GL union), against rule 1. Resolved by C0 and C1-null |
| **C1-null** | `casmi26-c3v2-tau8_nosplice.ipynb` (`build/patch_c3v2.py --no-splice`) | identical to C1 (the embedded mutator, gate and runner sources are byte-identical), except the splice `merge` is skipped. Stage 2 and the cap still run and print `C3V2 SPLICE DISABLED ... would splice N` | `ICE_UNION_MUT` > 0, `SPLICE DISABLED` line, `cuda` in the ICE meta | 0 vs C0 | reading against C0 (noise ±0.004): **C1-null ≈ C0 and C1 < C0 − 0.004** → the splice costs; run C3 `--no-rank1` with fire_frac 0.04 once, then close the LB channel if it is not ≥ C0 + 0.001 (the calibration ceiling is about +0.010). **C1-null < C0 − 0.004** → the plumbing hurts the base: score generated candidates in a separate ICE/GL call, restore the 5400/4000 budgets, and resubmit C1. **C1 ≈ C1-null ≈ C0** → the channel is neutral within noise, so re-probe with a tightened gate (CC.3/CC.4) only if the counters show fires | | |
| C2 | C1 with thresholds from CC.* | recalibrated gate | same | ± | same | | |
| C3 | `--no-rank1` | attribution of the rank-1 policy | same | ± | same | | |

## Track 4b register: our own NP signals in the PubChem channel (base: supernext, our runs 0.399 / 0.401)

Pre-registered 2026-10-04, **before** any dataset build, benchmark or submission. Plan: `np_signals_plan.md`.
Base = `evenda-casmi-inferencesupernext.ipynb`. Our own unchanged runs scored 0.399 and 0.401; the 0.405 in EXPERIMENTS row 24 is the author's score.
The popularity arrays are not changed. Every signal is one additive term, `+ μ_s·s`, in the sort key of the PubChem channel (`probe_one`). The f·z values that feed the `rel` merge gate stay purely spectral.

### Offline gates (Colab benchmark `build/np_signals_bench.ipynb`, before any LB probe)

The window is the truth's same-formula PubChem-tier isomers, deduplicated by scorer key, with the truth guaranteed present. The ranking compares `pop` alone with `pop + μ·s`, for μ ∈ {0.25, 0.5, 1.0}.
- Cohort T: training structures.
- Cohort L: `natural_products.parquet` with `in_casmi_train == False`.
- About 2k molecules each.

| gate | signal | metric | threshold | measured | verdict |
|---|---|---|---|---|---|
| NB.0 | all | truth present in its own window | 100% of the sampled cohort | T 93.2% (median window 2,621), L 97.7% (median 451). Metrics are computed on the found truths only | PASS (on found truths) |
| NB.V0 | vendor_only | share of NP-table compounds flagged `vendor_only = 1` (checks the categorisation) | < 1% | 0.12% | PASS |
| NB.V | vendor_only (penalty) | ΔMRR@25 at the best μ: cohort L / cohort T | L ≥ +0.010 **and** T ≥ −0.005 | μ 0.5: L +0.0005, T −0.0006 | **FAIL → KILL.** Redundant with pop: vendor decoys already sit below L truths on pop. 61.7% of T truths are themselves vendor-only (screening-library spectra) |
| NB.N | Ertl NP-likeness (z within window) | same | L ≥ +0.010 **and** T ≥ −0.005 | μ 1.0: L +0.0316, T −0.0105 (μ 0.25: L +0.0138, T −0.0007) | **FAIL** at the pre-registered best-L μ. μ 0.25 would pass both bars, but choosing μ after seeing T is post-hoc. Not shipped unless re-registered |
| NB.C0 | class consistency (coverage) | (a) queries with ≥ 1 labelled analogue (leave-one-out nearest training structure); (b) windows with ≥ 2 distinct labelled superclasses | (a) ≥ 50% **and** (b) ≥ 20%. Otherwise DROP, or move to the NPClassifier fallback | (a) 55.0%, (b) 50.7% | PASS |
| NB.C | class consistency (unlabelled = 0) | same as NB.V | L ≥ +0.010 **and** T ≥ −0.005 | μ 1.0: L +0.0345, T +0.0030 (monotone in μ) | **PASS.** Optimistic for two reasons: the analogues are structural oracles (nearest training structure, not spectral hits), and 90% of L truths carry a table label versus 8% of T truths |

A signal that fails its gate stops at that gate. The μ chosen offline is the one shipped. It is not re-tuned on the LB.

### Amendment A-NB.1 (2026-10-04, after the benchmark, before any LB probe)

The selection rule "the μ with the best ΔMRR on L" chose μ 1.0 for NB.N and NB.C. That conflicts with rule 5 of this report, registered before any measurement: "No new signal at weight 1.0 on first appearance". With μ ≤ 0.5 enforced, the same pre-registered bars give:

| gate | μ | L | T (bar −0.005) | verdict |
|---|---|---|---|---|
| NB.N NP-likeness | 0.5 | +0.0245 | −0.0042 | **PASS (narrow)**, ships at μ 0.5 |
| NB.C class consistency | 0.5 | +0.0194 | +0.0016 | **PASS**, ships at μ 0.5 |

The guard margin for NP-likeness is narrow (0.0008), and both L gains are optimistic: the Ertl model and the NP table share NP-database provenance, and the class analogues were structural oracles. The LB, with a paired control, is the arbiter. In supernext, μ applies in z-units: `z(score) + μ·z(signal)`.
The realism-check run (pool-missing cohort, membership-only ablation) is a diagnostic. It does not change these verdicts after the fact.

### Realism check R.* (`casmi_np_signals_realism_colab.ipynb`), pre-registered 2026-10-04, before it runs

The check runs **while** T4B-N / T4B-C are on the LB. It is diagnostic: it explains the LB result and decides the stack (T4B-S). It does **not** override an LB verdict.
- **Cohort P:** every NP-table compound with `in_casmi_pool == False` (150–1250 Da), about 2.6k. Pool members are removed from each window, the same set supernext's PubChem channel can list.
- **Cohorts L and T** re-use the first run's seed and size, for the ablation.
- **New variant `member@μ`:** `pop + μ·[candidate has an NP label]`.

| id | question | metric | threshold | measured | verdict |
|---|---|---|---|---|---|
| R.1 | does class consistency carry class information beyond NP membership? | ΔMRR L: classcons@0.5 − member@0.5 | ≥ +0.005, else classcons ≈ the np1 membership bonus | +0.0028 (classcons +0.0194 vs member +0.0166 on L; on P member@0.5 +0.0197 > classcons +0.0181) | **FAIL**: class consistency ≈ the NP-membership bonus |
| R.2 | does class consistency help the channel's own population? | ΔMRR P, classcons@0.5 vs pop | ≥ 0 | +0.0181 | PASS |
| R.3 | does NP-likeness help the channel's own population? | ΔMRR P, nplike@0.5 vs pop | ≥ 0 | +0.0219 | PASS |
| R.4 | is the stack worth a probe? | ΔMRR P: stack@0.5 (pop + 0.5·z(NP) + 0.5·cc) minus the better single at 0.5 | > 0 → submit T4B-S if both singles keep; else skip | +0.0178 (stack +0.0397 vs NP-likeness +0.0219) | PASS (moot: no single kept on the LB) |

### LB probes (`build/patch_supernext.py`, NP-signal flags → `casmi26-t4b-<tag>.ipynb`)

- **Paired control:** submit unchanged supernext in the same batch as the first probe. `base` = the mean of all our unchanged supernext runs (0.399, 0.401, plus the paired run).
- **KEEP** only if LB ≥ base + 0.004 (the noise floor) **and** the rerun counters are non-zero: signal hit-rate, `top-1 changed in N`, and the term's min/median/p99.
- If a probe moves the LB while its counters show no change, the move is noise.

| id | flag | change | expected ΔLB | LB | verdict |
|---|---|---|---|---|---|
| T4B-0 | — | unchanged supernext (paired control) | 0 | not reported | base mean = 0.400 (0.399, 0.401); keep bar **0.404** |
| ~~T4B-V~~ | — | vendor_only: killed offline (NB.V) | — | — | not submitted |
| T4B-N | `--nplike 0.5` | +0.5·z(NP_Score) when the PubChem list is re-ordered at merge time | +0.000…+0.005 | **0.399** | **NOT KEPT** (−0.001 vs base mean, noise) |
| T4B-C | `--classcons 0.5` | +0.5·z(superclass agreement with BASE's top-5 pool candidates) when the PubChem list is re-ordered at merge time | +0.000…+0.004 | **0.399** | **NOT KEPT** (−0.001, noise) |
| T4B-S | `--nplike 0.5 --classcons 0.5` | both terms | must beat the best single signal | **0.399** | **NOT KEPT** (= singles) |

**LB outcome (10-05).** All three probes scored 0.399, below the keep bar of 0.404, so **Track 4b closes with no NP signal kept**. Three different re-orderings landing on the same score to three decimals means one of two things:
1. The NP block was a no-op on the hidden run (`TRACK4b NP signals FAILED -> no-op`).
2. Its reach is too small to show on the LB. The terms only re-order the PubChem-only list, and that list only feeds a few merge slots, gated by `rel` and `lib_max`. So only molecules whose truth is PubChem-only *and* missing from the pool can move. The offline +0.02 is a cohort MRR over exactly those truths; scaled by their small share of the test set, the expected LB move is about 0.001, under the noise floor.

The commit-run log lines (`reordered`, `top1_changed`) tell the two apart. They don't change the verdict either way. The realism check R.* is now optional: it can't rescue an LB verdict.

**Realism outcome (10-05, `np_realism.json`).**
- Cohort P: 951 sampled. **Only 21.3% of the truths are in the PubChem tier at all**; the median window is 51 and the `pop` baseline MRR is already 0.431.
- On those found truths, both signals help (R.2, R.3) and stack (R.4). So the LB null is **not** a sign error. It is reach: about 79% of pool-missing NPs can never be listed by the PubChem channel, and the remaining ones are a small share of the test set.
- R.1 FAIL: class consistency carries no information beyond "the candidate is a labelled NP".
- None of this changes the LB verdicts above.

---

## Track 4c register — class vote on the zenith fused list (pre-registered 2026-10-06, before any run)
Plan: `np_class_sig_high_influence.md`. Base: `enveda-casmi-2026-v18-sovereign-zenith.ipynb` (unchanged), control **Z0** = our one zenith LB run.
Probes from `build/patch_zenith.py --classvote μ` → `casmi26-t4c-classvote<μ>.ipynb`.

| Rule | Threshold |
|---|---|
| Fire gate | `lib_max < 0.9` **and** ≥ 2 labelled voters (top 10 delta-list analog hits, shift-aware entropy ≥ 0.5) **and** winner superclass share ≥ 0.5 |
| Term | signed agreement on the fused list before the GLACIER rerank: `z(RRF) + μ·t`, t = +share(candidate label) if voted, −share(winner) if labelled but unvoted, 0 if unlabelled |
| Commit check | no `TRACK4c ... FAILED` line; `CLAW STAGE SCORED` unchanged; dry-reach (lib_max gate ignored, not applied): voted > 0 **and** top-1 changed > 0; extra runtime fits 9 h |
| KEEP | LB ≥ Z0 + 0.004 **and** the commit check passed. Single control → a pass is a candidate base, confirm with a rerun |

| id | flag | LB | verdict |
|---|---|---|---|
| T4C-0 | unchanged zenith (Z0) | _pending_ | |
| T4C-05 | `--classvote 0.5` | | |
| T4C-10 | `--classvote 1.0` | | |
