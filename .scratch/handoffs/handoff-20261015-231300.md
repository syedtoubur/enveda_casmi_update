# Session Handoff Dossier: ISDB CFM-ID Fallback Campaign (G1 & G2 Test Fix Ready)
**Generated:** 2026-10-15 23:13:00
**Source Conversation ID:** 6ed2e47d-50b6-40c2-9685-4ccf56e2448d
**Context Size:** ~151k tokens (Smart Zone Exit)

---

## 1. High-Level Mission & Active Objective
- **Competition**: Enveda CASMI 2026 (Molecular Structure Identification from LC-MS/MS).
- **Core Objective**: Implement an orthogonal **CFM-ID 4.0 in-silico spectral database (ISDB)** fallback to fix negative-mode (`[M-H]-`) and evidence gap blindness in baseline notebook ([`casmi26-e006-window-40-ice-fallback-true.ipynb`](file:///E:/casmi/lines/exploit/casmi26-e006-window-40-ice-fallback-true.ipynb), LB 0.440).
- **Campaign Line**: `exploit` line (`E:/casmi/lines/exploit`), tracked under [`research/exploit/20261015-isdb-cfm-fallback/`](file:///e:/ENVEDA_casmi/research/exploit/20261015-isdb-cfm-fallback/).
- **Current Milestone**: Gate G2 Unit Testing & Verification (`test_patch_isdb_cfm_fallback.py`).

---

## 2. Settled Decisions (Do NOT Re-Litigate)
- **Decision 1: Dynamic Cell Lookup in G1**:
  - *Rationale:* Hardcoding cell indices broke when markdown cells were present in the base notebook (32 cells total, 30 code cells).
  - *Impact:* `test_g1_notebook_delta_and_compilation` now dynamically locates the loader cell and `_factorial_entry` cell, passing cleanly.
- **Decision 2: Chemically Valid SMILES in G2 Replay Tests**:
  - *Rationale:* `casmi_post.metric_identity` calls RDKit canonicalization `canonicalize_smiles_to_ik14`. Dummy strings like `"A"` are rejected as invalid SMILES.
  - *Impact:* Replaced with valid aliphatic chains (`"CC"`, `"CCC"`, `"CCCC"`, `"CCCCC"`), ensuring all replay engine tests (`test_g2_metric_overlay_*`) run against real RDKit normalization.
- **Decision 3: Scope Invariants for Factorial Rerank**:
  - *Rationale:* `rerank_row` checks `mid in scope["active"]`. If `"active"` is omitted from the mock scope, `rerank_row` returns `branch: "placeholder"`.
  - *Impact:* Adding `'active': {'m1', 'm2'}` to mock scope in `test_g2_factorial_entry_rerank_row` ensures proper routing to `"glacier"`, `"isdb_fallback"`, and `"unrescored"`.

---

## 3. Modified & Created Files
- [`E:/casmi/lines/exploit/build/test_patch_isdb_cfm_fallback.py`](file:///E:/casmi/lines/exploit/build/test_patch_isdb_cfm_fallback.py) - G1 & G2 test suite (AST compilation, spectral cosine math, loader mock, replay engine K3 invariance & K2 activation, and Cell 32 execution).
- [`e:/ENVEDA_casmi/.scratch/test_patch_isdb_cfm_fallback.py`](file:///e:/ENVEDA_casmi/.scratch/test_patch_isdb_cfm_fallback.py) - Scratch version for staging and editing.
- [`E:/casmi/lines/exploit/build/patch_isdb_cfm_fallback.py`](file:///E:/casmi/lines/exploit/build/patch_isdb_cfm_fallback.py) - Notebook patcher script.
- [`E:/casmi/lines/exploit/casmi26-e006-isdb-cfm-fallback.ipynb`](file:///E:/casmi/lines/exploit/casmi26-e006-isdb-cfm-fallback.ipynb) - Patched probe notebook.
- [`E:/casmi/lines/exploit/build/casmi_post.py`](file:///E:/casmi/lines/exploit/build/casmi_post.py) - Updated with `ISDB_FALLBACK=False`, `ISDB_LAM=0.20`, and `_metric_overlay` branch.
- [`specs/Dataset_intro/dataset_catalog.json`](file:///e:/ENVEDA_casmi/specs/Dataset_intro/dataset_catalog.json) - Registered ISDB predicted index and spectra datasets.

---

## 4. Current Work in Flight
- `test_g1_notebook_delta_and_compilation` is **PASS**.
- `test_g2_spectral_cosine` is **PASS**.
- `test_g2_load_isdb_scores_mocked` is **PASS**.
- `test_g2_metric_overlay_invariance_k3` is **PASS**.
- `test_g2_metric_overlay_activation_k2` is **PASS**.
- `test_g2_metric_overlay_graceful_degradation` is **PASS**.
- In `test_g2_factorial_entry_rerank_row`: need to add `'active': {'m1', 'm2'}` into the `scope` dictionary so `rerank_row` processes candidates rather than returning the `"placeholder"` branch.

---

## 5. The Immediate Frontier (Next Steps)
1. **Fix mock scope in `test_patch_isdb_cfm_fallback.py`**:
   Add `'active': {'m1', 'm2'}` to `scope = MockScope({...})` in `test_g2_factorial_entry_rerank_row`.
2. **Execute G2 Verification**:
   Run:
   ```powershell
   Copy-Item -Path "e:\ENVEDA_casmi\.scratch\test_patch_isdb_cfm_fallback.py" -Destination "E:\casmi\lines\exploit\build\test_patch_isdb_cfm_fallback.py" -Force
   python E:/casmi/lines/exploit/build/test_patch_isdb_cfm_fallback.py
   ```
   Confirm all G1 & G2 tests pass.
3. **Run Gate G3a Smoke Test & G4 Kaggle Push**:
   - Check local execution on 12-20 sample molecules.
   - Push probe to Kaggle:
     ```powershell
     python research/bin/kaggle_push.py research/exploit/20261015-isdb-cfm-fallback E:/casmi/lines/exploit/casmi26-e006-isdb-cfm-fallback.ipynb --counter "ISDB_FALLBACK" --source n10705013/casmi26-isdb-predicted-index
     ```

---

## 6. Recommended Skills & Boot Command
- **Primary Skill**: [smart-zone-handoff](file:///e:/ENVEDA_casmi/.agents/skills/smart-zone-handoff/SKILL.md) / [tdd](file:///C:/Users/h/.gemini/config/plugins/mattpocock-skills/skills/tdd/SKILL.md)

### Copy-Paste Boot Prompt for New Session:
> `@.scratch/handoffs/latest.md continue`
