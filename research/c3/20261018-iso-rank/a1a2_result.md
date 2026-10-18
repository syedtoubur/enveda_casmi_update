# iso-rank A1 + A2 result

Worktree E:/casmi/lines/c3-iso-rank, branch idea/20261018-iso-rank.

Built (all under build/): iso_nl.py (A1, `nl_scores`, copy of nl_stoich_rda scorer, maths unchanged; NaN for unparseable),
iso_rank.py (A2: `parent_map`, `rrf`, `apply`), ISO_MUT knob + `_iso_rank` last step in casmi_post.py (default off,
counters iso_children / iso_fired / iso_tail, set only when the knob is on), test_iso_rank.py (20), test_iso_nl.py (3).

Tests: `pytest test_iso_rank.py test_iso_nl.py` -> 22 passed. test_casmi_post.py: 5 passed, 1 failed
(test_e2r2_row_pure_invariants), which fails identically on the parent commit 9c1ef2b (pre-existing, unrelated).

Decisions / deviations (small):
- `apply` takes the first RRF child not already in the row as c (so a child retrieved elsewhere is skipped, not blocking).
- Row entries are matched by plain or tautomer-canonical IK14 (children carry plain IK14 from isomer_mut).
- Parent at rank 25: no fire (nothing to insert below it). Fired child and its parent keep their slots in the 21-25 tail
  replacement (so c stays directly below p); the spec did not say.
- iso_tail counts molecules whose tail took >= 1 child (info['tail'] is the number of children used).
- Rows shorter than 21 get no tail fill. NL NaN is treated as "list missing" in the RRF.
- Truth-blind audit greps for truth / ik14_true / val_key / quoted 'inchikey' and file-read calls.

Open: none blocking. Commit hash in the report.
