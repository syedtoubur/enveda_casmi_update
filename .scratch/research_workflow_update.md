# Autoresearch Workflow Analysis & Improvement Ledger

**Created:** 2026-10-12  
**Purpose:** Critical operational review of the autonomous research loop (`research/` campaign system) during live runs. Documents friction points, architectural mismatches, and concrete fixes to improve agent collaboration and execution efficiency.

---

## 1. Finding 1: Local Resource Blindness in `/attack` & Stress Tests

### What Happened
During the `/attack` phase for `20261012-engine2-fppair`, Claude Opus designed six pre-registered tests (A1 to A6). Two of them (A2 and A5) required running full spectral inference and candidate ranking across 80 validation molecules on the local machine's CPU.
* **The Conflict**: Running full Engine 2 requires loading the candidate pool (COCONUT, ChEBI, LIPID MAPS), building the library index, and evaluating neural models, which peaks at **3.5–4.5 GB RAM**.
* **The Machine Reality**: The local development machine only had **~860 MB of free RAM**. Running A2 and A5 locally would have caused an immediate system Out-Of-Memory (OOM) crash or hard process kill.
* **The Ad-Hoc Recovery**: Claude was forced to mark A2 and A5 as `UNTESTED` locally and defer their verification to Kaggle GPU execution via G4 log counters.

### Root Cause
The attack generator currently assumes infinite or standard cloud server memory when proposing local stress tests, failing to inspect local machine telemetry (RAM availability, CPU thread limits, and GPU absence) before pre-registering stress test requirements.

### Required Workflow Update
Stress tests and attack gates must be explicitly split into two formal tiers in `research/` rules:
1. **Tier 1: Lightweight Local Checks (CPU / Memory < 500 MB)**:
   * Matrix shape / width compatibility (e.g. A1).
   * Feature schema and block name validation.
   * Dataset metadata / leakage overlap checks (e.g. A3).
   * Fast replay harness evaluations on existing frozen state dumps (e.g. A4).
2. **Tier 2: Heavy-Path Pipeline Checks (Delegated to Kaggle G4 Drafts)**:
   * Any test requiring the full candidate pool or live neural spectrum inference must NOT be scheduled as a local stress test.
   * Instead, it must be pre-registered as a **Fast-Draft G4 Gate** (running 10–20 molecules on Kaggle T4 GPU with pre-registered log counter regexes).

---

## 2. Finding 2: Worktree Directory Shift & Headless Permissions Bug

### What Happened
When the headless `/spec` session began, the agent executed:
```bash
cd /e/casmi/lines/exploit
```
Once inside the git worktree, relative permission paths in `.claude/settings.headless.json` (such as `Edit(research/**)`) resolved relative to `E:/casmi/lines/exploit/research/**` instead of the actual project repository at `E:/ENVEDA_casmi/research/**`.
* As a result, writes to `spec.md` were denied.
* The agent attempted to fix `.claude/settings.headless.json`, but was blocked by its own self-modification safety rule (`Edit(.claude/**)` in `deny`).
* Cycle 1 stalled until manual intervention.

### Resolution & Prevention
* **Immediate Fix**: Absolute path variants for the workspace (`E:/ENVEDA_casmi`) were added to both `allow` and `deny` in `.claude/settings.headless.json`.
* **Workflow Rule**: Any future permission configuration must always include both POSIX-style (`//E:/...`) and standard Windows-style (`E:/...`) absolute paths for all active workspace roots and worktrees.

---

## 3. Finding 3: Upfront Ensemble Reach & Dilution Screening

### What Happened & Generalized Pattern
When an idea targets an internal sub-component of a multi-stage pipeline or ensemble, agents frequently assess the improvement based solely on the component's isolated benchmarks, without calculating how much voice that component actually carries in the final prediction list.
* If a sub-component is heavily down-weighted or gated by upstream filters, even a significant component-level gain (e.g. +10% MRR) can become diluted to near zero when fused into the final output.
* In the worst case, this leads to generating specs, running attack checks, and pushing GPU kernels that are mathematically incapable of exceeding the competition noise floor ($\pm 0.004$).

### Required Workflow Update
The `/hypothesis` and `/spec` phase must enforce an **Upfront Net Reach Calculation**:
$$\text{Expected Net Impact} \approx (\text{Component Target Headroom}) \times (\text{Component Blend Weight / Voice}) \times (\text{Component Relative Gain})$$
* **Rule**: If the calculated net impact sits below the statistical noise floor, the agent must not proceed with code generation until the architecture/weights are adjusted to give the improved component sufficient voice, or the idea must be rejected immediately at the hypothesis stage to save research bandwidth.

---

## 4. Summary of Planned Rule & Template Updates
1. **Update `spec.md` and `attack.md` templates**: Enforce strict RAM budgeting ($< 500\text{ MB}$ local threshold) before declaring a test runnable locally.
2. **Update `hypothesis.md` template**: Mandate an explicit end-to-end reach/dilution calculation before approving a sub-component modification.
3. **Update `research/LESSONS.md`**:
   * Add lesson **L36**: *Heavy-path pipeline tests must not run on local CPU; pre-register them as Kaggle G4 draft counters to prevent local OOM.*
   * Add lesson **L37**: *Ensemble sub-components must be screened for end-to-end voice dilution; do not spec ideas whose net reach falls below the noise floor.*
3. **Keep Human Gating Active**: Always retain human authority over leak-coupled gates, ensuring automated tools provide diagnostics rather than unilateral vetoes.
