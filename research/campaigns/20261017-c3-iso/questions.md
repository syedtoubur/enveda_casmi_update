# Questions for the human (replan 2026-10-18 20:45, cycle 3 closed)

## Q1: c3-iso stopped on scope. Re-grill or close?
Every in-charter route is measured dead:
- fixed slot: S1 s*=7, −0.0009;
- generated #1 keep: LB 0.446 (ref 57325719) < 0.451;
- isomer insert above the parent: iso-rank Stage A half B +0.0015 vs bar 0.012, fire-only −0.0003;
- any CPU child-vs-parent trigger: signal audit ALL DEAD, C1/C2 FPR 0.21–0.71 vs TPR ≤ 0.017, rule R11;
- tail 21–25: composite +0.0004, under the noise floor.

Sources: gate_report.md blocks 2026-10-18 "iso-rank Stage A MEASURED" and "c3 signal audit MEASURED";
`research/c3/20261018-signal-audit/result.md`.

The audit's pre-registered next step is a **K2 v1 Class-3 detector spec**. This charter rules out a separate detector
(K2 v0 dead), so the loop cannot go on without a new scope. Options:
- (a) run `/campaign-grill` for a K2 v1 detector campaign: features, L40 negatives, GPU costing before any go;
- (b) close the c3 line and move the effort to C2 rank 2–5 ordering. That bucket has +0.298 headroom on C2 (loss map);
  the lead candidate is `jestr-contrast-channel` in `research/ideas/20261017-candidates.md`.

Remaining caps if the campaign resumes: 2 cycles and 3 LB probes.

Also open: `research/c3/20261017-isomer-mut-ice` is still `needs-human`, superseded by iso-rank (state.json `i3`).
Recommendation: mark it dead.
