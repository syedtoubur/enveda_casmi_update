# G4 fix brief: 20261018-iso-rank

- kernel: `syedtoubur/casmi26-iso-rank` (round 1 of 2; log in `kaggle_output/`)
- kind: probe; line: c3; worktree: `E:/casmi/lines/c3`

## Re-push command (no --wait; add `--source <owner/slug>` for a catalog dataset)
```
python research/bin/kaggle_push.py research/c3/20261018-iso-rank "E:\casmi\lines\c3-iso-rank\casmi26-v22-iso-rank.ipynb" --preset sota_0433_inference_preset --slug casmi26-iso-rank --counter "ISO_COUNTERS iso_children=" --counter "?iso_fired=" --counter "?ISO_CHANGED rows" --source syedtoubur/casmi26-e180-keys
```

## The one change (spec.md, up to the 3rd section)
# Spec: iso-rank (c3-iso campaign, human grill 2026-10-18)

Plan: `~/.claude/plans/scratch-handoffs-handoff-20261018-c3-ov-floating-zebra.md`. Gate block: gate_report
`### 2026-10-18: c3-iso iso-rank, PRE-REGISTERED`. Predecessor `../20261017-isomer-mut-ice` (reach_150 0.140; reach_20 by
ICE 0.1017; I3 trigger "best child > best retrieved" fires on 74% of molecules, so it doesn't discriminate).
Worktree `E:/casmi/lines/c3-iso-rank`, branch `idea/20261018-iso-rank` (off 9c1ef2b). Knob `ISO_MUT` (default off).

### Claim
The isomer children already reach Class-3 truths. Ranking them by a truth-blind RRF of orthogonal signals, and inserting
the best one **directly below its own parent** only when it beats that parent on ICE AND frag, gains on pseudo_c3. It also
costs little on C1/C2: there the parent is usually the truth at rank 1, and a child placed below it displaces only the
candidates below.

### Components (all truth-blind; truth only scores)
- **Children:** `build/isomer_mut.py` unchanged (`mutate_many`, cap 600, 120 s deadline) on the V22 fused top 10 (the
  parents). Plain-IK14 dedupe; drop children whose IK14 is already in the V22 list. CPU frag (`build/frag_rescore.py`,
  measured spectrum) cuts to the top K (offline K = 150).
- **Parent map:** child IK14 → set of (parent fused rank, operator). parent(child) = the parent with the best fused rank;
  multiplicity = number of distinct (parent, operator) pairs.
- **Lists** (each ranks every child of one molecule; a list missing for a child, e.g. no ICE coverage or no GLACIER adduct,
  is simply absent from that child's RRF sum):
  1. ICE cosine vs the measured spectra, aggregated as the base does (`build/iceberg_worker.py`);
  2. CPU frag score;
  3. multiplicity (higher is better);
  4. parent fused rank (lower is better);
  5. NL: `score_molecule(spectra, cand_smiles)` from `research/antigravity-exploit/nl-sim-rda/nl_stoich_rda.py`, ported as
     `build/iso_nl.py` (exact-mass diagnostic neutral losses + 6-ring RDA; measured spectrum only, R10);
  6. GLACIER cosine on `[M+H]+`/`[M+Na]+` spectra (`gl_runner.py`, dataset `ahmedberatozer/casmi26-glacier`).
- **RRF:** score(child) = Σ_lists 1/(k + rank_list(child)), k = 10, ranks 1-based, ties take the average rank.
- **Fire rule** (per molecule, on the RRF top child c with parent p): fire iff ICE(c) and ICE(p) both exist AND
  ICE(c) − ICE(p) > m_ice AND frag(c) > frag(p). If p is not in the final top 25 row, there is no fire.
- **Placement:** when fired, c goes at final position pos(p) + 1, the rest shift down, and the row is cut at 25.
  Always (fired or not), ranks 21–25 are replaced by the next RRF children not already in the row (tail ride-along).
  Dedupe by IK14; never rank 1 (a parent at rank 1 puts c at rank 2 at best).
- **Counters:** `iso_children` (molecules with ≥ 1 child), `iso_fired`, `iso_tail`, `iso_skipped_time`. Wall-clock guard:
  stop processing new molecules once elapsed time > 8.3 h; their rows stay unchanged.


## Fail reasons
- required 'engine lists \\d+' not printed
- required 'ICE meta \\{"status": "ok".*"device": "cuda"' not printed
- required 'CLAW STAGE SCORED \\| ice=[1-9]\\d* gl=[1-9]' not printed
- counter 'ISO_COUNTERS iso_children=' not printed
- counter 'iso_fired=' not printed
- counter 'ISO_CHANGED rows' not printed

## Log tail (last 80 lines)
```
(no log downloaded)
```

## gates.md
| G4 | 2026-10-18 08:40 | PUSHED | syedtoubur/casmi26-iso-rank |  |
| G4 | 2026-10-18 09:04 | FAIL (complete; required 'engine lists \\d+' not printed / required 'ICE meta \\{"status": "ok".*"device": "cuda"' not printed / required 'CLAW STAGE SCORED \\| ice=[1-9]\\d* gl=[1-9]' not printed / counter 'ISO_COUNTERS iso_children=' not printed / counter 'iso_fired=' not printed / counter 'ISO_CHANGED rows' not printed) | syedtoubur/casmi26-iso-rank; counters: none matched |  |

## Fix scope
- The fix agent may change the patch code in the line worktree (`E:/casmi/lines/<line>`), then rerun G1 and G2.
- It may add push sources **that already exist in `dataset_catalog.json`** (`kaggle_push.py --source <owner/slug>`).
- A dataset missing from the catalog, or a fix that would change the idea itself: write to `questions.md`, set STATUS `needs-human`, stop.
- For a **control** idea only sources and metadata may change, never the notebook code.
- The catalog itself is never edited.

