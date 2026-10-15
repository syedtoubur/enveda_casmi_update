# Candidate ideas — 2026-10-13 (post-scout, pre-charter)
tier: H (my own analysis, this session) · posture: **structural mechanisms only** (human, grill 2026-10-13)
Scout inputs: `research/scout/2026-10-13-kaggle.md` (8 items), `research/scout/2026-10-13-papers.md` (15 items).
Reality check: public LB top **0.491**, ~14 teams above our 0.440 (verified: `kaggle competitions leaderboard -c
enveda-CASMI26-molecule-id-mass-spectra --show`). No public notebook discloses a mechanism above **0.439**.

The human picked all four candidates (A–D) with 4 LB probes and no control replicate. Below is each one's claim,
the mechanism **as it actually applies to our base**, and the weakest assumption. Two premises were wrong and are
corrected here — read the code cites before any of them is speced.

---

## A. +1.45 ppm timsTOF precursor centring — **premise fails a ceiling check (L13)**
- Claim (AGENTS.md, L21): test masses carry a +1.45 ppm offset; centring the query target reduces decoy dilution.
- Verified against the base: `PPM_CENTER` and `1.45` appear **nowhere** in `scratch/casmi26-e006.py`. The only ppm
  constants are `PPM_WIN = 10.0`, `PPM_FALLBACK = 30.0` (CFG inside the embedded `pv.py`, line 233).
- **Why the mechanism does not do what the doc claims.** The loss bucket is rank 2–5 (51% of C2, +0.298 headroom,
  `research/harness/loss_map.md`), and those are **same-molecular-formula isomers** — the rerank works inside formula
  groups (`fuse2` → same-formula ICE/GL groups). A same-formula isomer has *identical* theoretical mass, so a 1.45 ppm
  shift **cannot reorder them at all**. It can only change which *formulas* enter the ±10 ppm window, and at 1.45 ppm
  that band is only formulas sitting 8.55–10 ppm out. Reach ≈ a thin slice of the 3% "in no channel list" bucket.
- Verdict: **not a probe.** Downgrade to a read-only diagnostic (count molecules whose formula-window membership
  changes at +1.45 ppm) before anyone spends a slot. If the diagnostic says >2% of molecules change, revive it.

## B. Consensus tail backfill at the fusion cut — **premise inverted, still alive**
- Scout's claim (from the 0.439 public kernel): candidates at rank 26+ lose their second-engine vote when fusion cuts at 25.
- What our base actually does (`casmi26-e006.py:717-727, 738`): `fuse2(v_smis, e_smis, e_keys, n=40)` scores the **whole
  of both lists** and only then cuts to 40 — so Engine 2's tail *does* vote. The truncation is on the **other side**:
  `v_smis` comes from `submission.csv` / `PREFUSION_LISTS` (`:729`) and `BASE[m] = (smis[:25], keys[:25], lib_max)`
  (`:578`), i.e. the **primary, weight-1.0 list is capped at 25** while Engine 2 contributes far deeper.
- Corrected mechanism: the missing evidence is the *primary engine's own ranks 26–60*, discarded at `:578` long before
  fusion. A probe would carry `smis[:40]` into `BASE` and let `fuse2` see the primary tail.
- Not the TOPN 60→100 dead end: that deepened the *scored* window; this restores a *vote* for candidates already scored.
- Risk: E006's overlay already top-ups tails (`:1006 safe_extend(_candidates[:25], _candidates[25:] + _original)`), so
  part of this may already fire. Family = fusion → **BASE-COUPLED 0.563**, harness cannot judge it → LB slot or ceiling only.

## C. Real experimental reference spectra as a 3rd retrieval engine — **highest upside, heaviest build**
- Mechanism: today's tier-1 library channel uses only the competition's own train spectra. Adding measured reference
  spectra (MassBank ~139k, MoNA/GNPS filtered to structures inside our pool) is a genuinely **orthogonal evidence
  source** — L10 says orthogonal sources are the only thing that has ever paid (BIO +0.009, GLACIER +0.007, 2nd engine +0.016).
- What must be answered before a spec: (1) where the data comes from **offline in ≤9 h** (a Kaggle dataset must be
  staged and added to the preset, L30/L35); (2) dedup against train (a test molecule whose spectrum is in MassBank is a
  Class-1 match — that is the win, but it must not double-count); (3) whether it is *measured* spectra or in-silico
  (L19: MassSpecGym-trained simulators already leak into val gains).
- Heavy path → harness BLIND, G4 + LB decide. Cost: one full build + likely a data-staging cycle.

## D. Per-molecule cross-spectrum evidence weighting — **needs a read-only audit first**
- Verified: aggregation already exists. The base groups by `molecule_id` (`:350`) and feeds **all** of a molecule's
  spectra into one `V.run(spectra, target)` with `target = median(neutral mass)` (`:362`). So "we fuse per spectrum and
  score per molecule" (the scouts' framing) is **wrong**.
- Open question the audit must answer: how are the spectra ordered/weighted inside `V.run` (the cell is labelled "v3.6
  spectra construction and ordering")? If it is a plain concatenation or a fixed CE ordering, then cross-adduct /
  cross-polarity agreement (MS2DeepScore cross-mode, MVP multi-view consensus) is an unexploited signal, and a
  per-molecule confidence weight is a post-scoring knob.
- Do not spec before the audit; the whole idea hinges on what that function already does.

---

## Ranking for the charter (my recommendation, cost-adjusted)
1. **B** — one upstream line (`:578`) + a counter; the corrected mechanism is real and cheap.
2. **C** — the only candidate with a plausible **+0.02…** path toward the 0.46–0.49 pack; expensive.
3. **D** — free audit first, then it either becomes a real idea or dies.
4. **A** — diagnostic only, unless the diagnostic contradicts the physics argument above.

## Grill decisions (2026-10-13, rounds 2-4) — final
- Posture: structural mechanisms only. Loop code is a **constant**; §7 Tier-A/B fixes deferred by the human.
- **A** → downgraded to a free diagnostic, no probe (physics argument above accepted).
- **B** → probe 1 (`fusion-tail`).
- **C** → probe 2 (`refspectra-lib`), **exactly 1 probe, no tuning follow-up**.
- **D** → **DEAD**, evidence accepted and recorded above; audit ran before signing.
- Probes 3-4 filled by the human's picks: **E** formula-window posterior (`formula-window`), **F** cross-attention
  reranker (`xattn-rerank`).
- Budget: 4 probes, **no control replicate** (W19 accepted knowingly: every KEEP bar rests on a single 0.440 run).
- Autonomy: stages 1-8 unattended; the human approves only LB-slot spends and each submit.
- G4 firing proof: diff the probe's draft `submission.csv` vs the control's draft, **>=1 changed molecule** required.
- Bar: harness **delta >= +0.008** with grouped-bootstrap 90% CI lower bound > 0; heavy needs a measured ceiling >= 0.016.
  Pre-registered in `research/harness/gate_report.md` (2026-10-13 block).
- Cycle order in the charter: B -> C -> E -> F.

Charter: `research/campaigns/20261013-e006-structural/charter.md`.

## D audit result (ran 2026-10-13, read-only; `casmi/engine.py`)
- `analog_query='merged'` merges spectra **per polarity** (`:131-135`); library search is same-polarity per query spectrum.
- `FEATURES` already carry cross-spectrum agreement: `agree_lib`, `agree_ap`, `lib_mean`, `lib_top3`, `lib_nspec`,
  `n_pos`, `n_neg`, plus the `xs_*` cross-scorer block (`:66-80`).
- `spectra[:cfg.max_query_spectra]` with the default **16** (`:38,:243`) never truncates on the measured test shape:
  400 molecules / 1213 spectra, median 3 per molecule, max 9, p90 5; 24.2% of molecules carry both polarities;
  108 molecules have >1 adduct (max 3). **Provisional per L04** - the local `test.parquet` may be a train slice.
- Verdict: per-molecule multi-spectrum aggregation is already done properly, so D has no mechanism left and dies.
- Narrower residue worth a later look (not this campaign's portfolio): collision-energy weighting - library
  representatives are picked by "most cleaned peaks" per (structure, polarity, library) (`:108-118`), not by CE.

## Pre-slot measurements (ran 2026-10-13, read-only, 0 LB slots spent)

| idea | instrument | result | verdict |
|---|---|---|---|
| B `fusion-tail` | `.scratch/fusion_tail/ceiling.py` (RRF replay on the C2 + C1_npx dumps) | cap 25->40: dMRR **-0.0001** [-0.0002, 0.0000], 0 up / 2 down; ceiling **+0.0000** | **DEAD** - the discarded primary tail holds no truths |
| E `formula-window` | `.scratch/fusion_tail/formula_reach.py` | truth already in the primary list: C2 **97.0%**, C1_npx **99.6%**; no candidate carries the truth formula: C2 **2.5%**, C1_npx **0.0%** | **REACH-STARVED** - 10 molecules on C2, 0 on C1_npx |
| C `refspectra-lib` | `.scratch/refspectra/reach.py` (MassBank harmonized, 18,316 InChIKey14) | covers **16.8%** of C2 truths (62 of 67 already rank 1 -> **5** targets, ceiling ~+0.012) and 81.2% of C1_npx (17 targets) | **SMALL but the only measured non-zero effect left**; it is a new *channel*, not a reordering |
| F `xattn-rerank` | `research/exploit/20261013-xattn-rerank/spike/rerank_headroom.py` | C2 rank 2-5 = 204/400 (51.0%), **0 of them outside the pool**, oracle **+0.2984**; but `p_base` alone scores 0.9251 vs E006 0.6312, and a learned model denied the pipeline's own ordering gets **-0.0218** [-0.057, +0.014] | **NO SIGNAL** (cheap form) / **UNMEASURED** (cross-attention form): the offline gain is BASE's leaked order, so no reranker on C2 is citable from the harness - see `spike_result.md` |

Loop note for the human (no code changed): `campaign.py:116-120` counts any STATUS `dead` toward
`dead_streak`, which closes a branch after 2. A pre-slot measurement kill is not a harness death, so these
verdicts are recorded in `spike_result.md` / this table instead of in STATUS - otherwise B, E and F would close
the exploit branch for succeeding.
