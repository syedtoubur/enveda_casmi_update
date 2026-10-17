# G4 fix 1: gen-rank1-keep

## Symptom
G4 FAIL: "draft identical to control on 400/400 molecules (changed 0 < 1)". Log: `GEN1_KEEP gen1_checked 12`,
`GEN1_KEEP gen1_kept 0`, `GEN1_KEEP gen1_moved 0`; no `GEN1_FAILED`; ICE scored 11/12 molecules (`ICE meta ... n_mols_scored 11`).

## Root cause
Not a code defect. The stage ran on all 12 draft molecules (gen1_checked 12), ICE scores exist and use the same
`ICE_SCORES[str(m)]` -> {smiles: score} convention as the base C8 rerank (`ice = ICE_SCORES.get(str(m), {})`), so the
trigger had its inputs. It simply had no case: it fires only when BASE #1 is an Engine-1 generated structure
(`|gen`), which K0 measured at 2/245, 1/300, 6/400, 4/200 on C1/C2 cohorts (~1-2%). The 12 visible molecules overlap
training (library hits), so P(0 fires in 12) is about 0.8. card.md flag (3) and ship_note.md already predicted
`gen1_kept`/`gen1_moved` = 0 on the draft.

`g4_min_changed: 1` was expected to be met by V22-vs-E006 differences (card.md flag 3: the control draft was the E006
placeholder). The control is now the V22 replicate, so a knob that does not fire leaves the draft byte-identical to it.

## Fix
None within scope. The only ways to pass are (a) set `g4_min_changed: 0` for this idea (a gate/metadata decision,
weakens L38), or (b) change the draft so it covers a `|gen` #1 molecule (changes the base's fast-commit protocol =
changes the idea). Both are left to the human (questions.md). Re-pushing unchanged would burn a GPU run and fail the
same way.
