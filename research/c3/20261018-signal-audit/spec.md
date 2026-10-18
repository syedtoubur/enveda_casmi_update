# c3 signal audit

Pre-registration: `research/harness/gate_report.md`, block "2026-10-18: c3 signal audit, PRE-REGISTERED" (commit ba344aa).
Plan: `~/.claude/plans/breezy-gathering-sloth.md` (Step 1).

- Question: does any CPU child-vs-parent signal (dnl, mult, dfrag, Rule A, Rule B) keep the pooled C1/C2 false-alarm rate
  at or below 4.5 % with enough half-A pseudo_c3 true positives to clear the LB noise floor (0.25·TPR − 0.275·FPR ≥ 0.004)?
- Input: the existing Stage A A3 tables (no new children). Parent nl is recomputed (A3 stored nl for children only).
- Code: worktree `E:/casmi/lines/c3-signal-audit` (`stage_a/audit.py`, `audit_kernel.py`, `test_audit.py`), commit 1a09587.
- Kernel: `syedtoubur/c3iso-signal-audit` (Kaggle CPU); outputs `audit_result.json`, `audit_rows.csv`, `audit.log`.
- Void unless the frag-only half-B fire rates reproduce Stage A (±0.005).
