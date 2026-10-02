# Phase 1 plan: condition grid, merge rule, T4 throughput and budget decision

## Context

Phase 0 passed (`gate_report.md`, revision 2): `msg_all` / CE in eV / QTOF token. Six adducts are covered by
simulation: `[M+H]+`, `[M-H]-`, `[M+Na]+`, `[M+NH4]+`, `[M+CH2O2-H]-`, `[M+Cl]-`. The other four go
to MetFrag-lite. The pool is **711,690 unique ik14**, which already answers roadmap Phase 1 action 3.
Throughput is **67.6 mol/s on an A100**, so one (adduct, CE) condition costs ≈ 2.9 A100-h.

Both ICEBERG stages are CE-conditioned (`gen_hparams.embed_collision = True`). A fragment DAG therefore
cannot be reused across CEs, and every grid condition is a full pass over the pool.

Phase 1 turns these facts into a **condition grid**, a **merged-spectrum rule**, a **nearest-grid
tolerance** and a **precompute-vs-on-the-fly decision**, all backed by measured numbers.

Inputs I already have:
- **Test CE structure.** The statistical-tour notebook (cell 41) says the hidden test acquires each
  molecule at **20, 40 and 60 eV plus a merged [20,40,60]**, in both polarities. It counts 298 of
  1,213 public spectra as merges.
  - The public test is **`enveda-180` rows verbatim** (cell 53).
  - Locally, the timsTOF rows in `val.parquet` also carry 35, 50, 80, [20,50] and [20,40,60,80].
  - So the hidden-test CE claim must be confirmed on `enveda-np-examples`, the same pipeline as the
    hidden test, and not taken from the public file.
- **Windows.** NP-like masses sit in sparser mass space (tour cell 24). The gate measured median
  ±8.5 ppm pool windows of ~100 around train truths.

User decisions (2026-09-27):
- **Precompute scope is decided by Phase 1 itself.** The rule is below.
- **Merge rule is chosen empirically.**
- **A Kaggle T4 ×2 throughput run is added.**

## Deliverables

| file | role |
|---|---|
| `build/simsim_common.py` (new) | Shared helpers, inlined via a `# @@SIMSIM_COMMON@@` placeholder: `prep`/`esim`, `run_iceberg`, `request_from`, `to_model_ce`, `formula_mass`, the POOL build, `pb_sf`. Lifted from `build/simple_sim_gate.py`. The gate notebook stays untouched. |
| `build/simple_sim_grid.py` → `casmi_simple_sim_grid_colab.ipynb` | Colab (A100/L4 + High-RAM): steps 1–6 below. |
| `build/simple_sim_t4_kaggle.py` → `casmi_simple_sim_t4_kaggle.ipynb` | Kaggle, T4 ×2, internet off: step 5. |
| `build/mk_simple_sim_grid.py` | Generator. It reuses `build()` and `phase1_kernels()` from `build/mk_simple_sim_gate.py` and adds the `@@SIMSIM_COMMON@@` substitution. It also embeds the current `iceberg_worker.py` (with the precursor + 2 Da cut) via `WORKER_TAG`. |
| `build/test_simple_sim_grid.py` | Offline tests, added to `build/run_all_tests.sh`. |
| Drive `_simlib/grid/` | `cond_grid.json`, `grid_report.json`, `req_t4.tsv`, `t4.json` |
| `specs_simple_ms_simulator/gate_report.md` | New "Phase 1" section with all tables and the decision |
| `roadmap.md` / `tech_stack.md` | Phase 1 implementation note, the resolved merge rule, and the filled "to measure" cells |

## Steps (Colab notebook unless stated)

**1. CE × adduct census.** Tabulate `(adduct, tuple(collision_energy_ev))` per spectrum and the
distinct conditions per molecule for three sources:
- `enveda-np-examples`, the hidden-test proxy,
- `enveda-180`,
- the public `test.parquet`.

**Check:** confirm or refute the 20/40/60 + merged [20,40,60] ladder on np-examples. If it holds,
the base CE set is {20, 40, 60}. If it does not, the base CE set is the distinct single CEs covering
≥ 95% of np-examples spectra, plus every constituent of the merged lists.

**2. Nearest-grid tolerance.** Simulate 300 `[M+H]+` and 300 `[M-H]-` pool structures at 5 eV steps
from 10 to 80 eV. For each Δ ∈ {2.5, 5, 10, 15}, compute the median `esim(pred@c, pred@c±Δ)`.

**Rule:** the tolerance is the largest Δ whose median is ≥ 0.90. Anything beyond it is off-grid and
simulated on the fly on Kaggle. Also report which public-test CEs (35, 50, 80, …) fall off-grid.
They affect only the fast-commit run.

**3. Merge rule (empirical).** Take np-examples molecules that have a merged-CE spectrum, one per
molecule and adduct, restricted to covered adducts (the `[M+H]+` and `[M-H]-` groups must each have
n ≥ 15). Rank the truth against ≤ 20 pool decoys from its ±8.5 ppm window, all simulated at each
constituent CE. Score two rules:
- (a) **sum-merge:** the renormalised sum of the constituent-CE predictions, then `prep`;
- (b) **max:** the maximum over constituent CEs of `esim`.

**Rule:** pick the rule with the higher MRR. Ties within 0.01 go to (b), because it needs no extra
code path. Report the exact Poisson-binomial p against chance with the gate's `pb_sf`. Either way,
merged spectra add **no new grid conditions**.

**4. Pool trimming and on-the-fly load projection.**
- Report the number of pool structures with neutral mass in [150, 1170] Da. The hidden test is
  157–1,159 Da, and the 120 ppm backfill adds a small margin. Anything outside that range can never
  be a candidate.
- Split the count into no-real-spectra (the Class-2 targets, first priority) and has-real.
- For np-examples molecules, reproduce phase1's window logic: ±8.5 ppm, then `PPM_EXPAND` tiers until
  ≥ 25 ik14. Count the predictions a molecule needs: Σ over distinct covered (adduct, grid-CE) of the
  window size.
- Report p50, mean and p95, scaled to 400 molecules. **Load(option)** counts only the conditions that
  option leaves to be simulated on the fly.

**5. T4 throughput (Kaggle notebook).** *As built: the Kaggle notebook builds the request itself, with
the same sampling, from the competition data plus the COCONUT dataset, and it embeds the current
worker. So no `req_t4.tsv` upload and no dataset re-version are needed in Phase 1. The steps below
are the original design.*
- Colab writes `req_t4.tsv`: 2,000 pool SMILES sampled to match the np-examples window mass
  distribution, at `[M+H]+`/`[M-H]-` × 20/40/60.
- Re-version the `casmi26-iceberg-offline` dataset with `req_t4.tsv` and the current worker. The
  roadmap already requires that refresh because the old worker predates the precursor cut.
- On Kaggle, measure mol/s at batch sizes 64 and 128 on 1 T4, and on 2 T4s with two worker
  subprocesses pinned via `CUDA_VISIBLE_DEVICES`, each taking half the rows. Also record the
  worker's startup cost and the failure rate by error reason.
- Output `t4.json`, then copy it to Drive `_simlib/grid/t4.json`.

**6. Decision cell.** This cell re-runs cheaply once `t4.json` is present. It projects three options.

| option | precompute | projected Colab cost |
|---|---|---|
| A | nothing; Kaggle simulates every covered condition on the fly | 0 A100-h |
| B | `[M+H]+` and `[M-H]-` × base CEs | ≈ 6 × 2.9 × (trimmed pool ÷ 711,690) A100-h |
| C | all 6 covered adducts × base CEs | ≈ 18 × 2.9 × (trimmed pool ÷ 711,690) A100-h |

Kaggle time = install 177 s + pool 264 s + library 30 s + simlib load (from the storage projection)
+ 400 × t_base + p95 Load ÷ T4 ×2 throughput. Here t_base is the per-molecule library + MetFrag cost,
taken from phase1's timing minus analog and FPNet.

**Rule:** choose the cheapest option (in Colab A100-h) whose projected Kaggle time is ≤ 6.3 h, which
is 9 h with a 30% margin. If none fits, stop and re-plan, the same stop-on-failure policy as Phase 0.

**Storage:**
`n_structures × conditions × mean cleaned peaks × 6 B + offsets`. The mean cleaned peak count is
measured from step 2's predictions after `_clean` (`INT_FLOOR 0.002`, `MAX_PEAKS 64`).

`cond_grid.json` contains:
- condition id → (adduct, CE eV, `ce_mode="ev"`, `instrument="QTOF"`);
- the tolerance;
- the merge rule;
- the chosen option;
- the covered and uncovered adduct lists;
- the checkpoint SHA1s, carried from G1.

## Reuse (do not rewrite)

- `build/mk_simple_sim_gate.py`: `build()`, `phase1_kernels()`, the `KERNEL_TAG`/`WORKER_TAG` pattern
- `build/simple_sim_gate.py`: `prep` (211), `esim` (216), `run_iceberg` (293), `to_model_ce` (307),
  `request_from` (310), `formula_mass` + POOL build (461–492), `pb_sf` (525), `sh`, and the venv setup
  (113–130) → moved into `simsim_common.py`
- `build/simple_sim_gate_kaggle.py`: the offline wheelhouse install and SHA1 check, reused for the T4
  notebook
- `build/iceberg_worker.py`: unchanged

## Verification

1. `python build/mk_simple_sim_grid.py` builds both notebooks with no surviving `@@`. `bash
   build/run_all_tests.sh` passes. The new tests cover:
   - the merge-rule functions on synthetic spectra;
   - the tolerance chooser;
   - the Load arithmetic on a toy window table;
   - the decision rule on its edge cases (none fits → STOP).
2. The Colab run prints the census, tolerance, merge-rule and projection tables, and writes
   `cond_grid.json`.
3. The Kaggle T4 run: SHA1 OK, ≥ 99% of the 2,000 SMILES predicted, `t4.json` written.
4. The decision cell prints the chosen option with its A100-h, `_simlib/` GB and Kaggle-hour
   projections. These numbers go into the Phase 1 section of `gate_report.md`, and the roadmap Phase
   1 **Check** is satisfied: a grid table, GPU-hours, `_simlib/` size, and an on-the-fly load that
   fits in 9 h.

Run order: Colab steps 1–5 → upload the dataset version → Kaggle T4 notebook → put `t4.json` on
Drive → re-run the Colab decision cell → update `gate_report.md`.
