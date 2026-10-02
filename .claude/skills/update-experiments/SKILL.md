---
name: update-experiments
description: Use when recording a new submission, leaderboard score, Colab gate result, or imported notebook to the central experiment ledger
---

# Update Experiments Skill

## Overview
Maintains [EXPERIMENTS.md](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/EXPERIMENTS.md) as the single source of truth for all Kaggle leaderboard submissions, offline component gate evaluations, and negative ablation results across the CASMI 2026 competition.

---

## When to Use

Invoke this skill whenever:
1. **A new Kaggle submission is completed**: You or the user submitted a notebook and received a Public Leaderboard (LB) score.
2. **A spec gate or offline benchmark finishes**: A phase in `specs/`, a Colab validation run, or a gate test (`gate_report.md`, `printouts.txt`, `.json` results) has new numbers.
3. **An external notebook is imported**: An `.ipynb` file from another participant or public source is copied into the project and needs to be analyzed and recorded.

---

## The Three Input Ingestion Paths

```
                             [New Event]
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
  [User Submission]      [External Notebook]      [Spec Gate / Colab]
         │                        │                        │
  • Read active spec       • Analyze .ipynb         • Read gate report/json
  • Extract hypothesis     • Extract architecture   • Extract metric & lift
  • Compute LB delta       • Extract LB from text   • Record Pass/Fail
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  ▼
                   Update [EXPERIMENTS.md]
                   Section 1, 2, or 3 + Header SOTA
```

---

### Path A: User's Own Submission (Spec-Driven)

When the user announces a submission or provides an LB score:

1. **Locate the Driving Spec**:
   - Check the currently active spec folder under `specs/` (`specs/track1_retrieval_fusion/`, `specs/track2_novel_generation/`, `specs/track3_ms_simulator/`).
   - Inspect `plan.md`, `plan_v2.md`, or `roadmap.md` to identify the hypothesis behind this run.
2. **Extract Key Information**:
   - **Notebook**: Exact filename (e.g., `casmi26-v4n-engine-fusion-union-lb-0-399.ipynb`).
   - **Core Approach in 2–3 Bullets**: What specifically changed? (e.g., *"Added ICE_UNION block: engine top-40 scored by ICE/GL"*, *"Weighted RRF fusion with α=0.6"*).
   - **Scores**: Public LB score and calculate the delta (`+` or `−`) compared to the previous best or base model.
3. **Update `EXPERIMENTS.md`**:
   - Append a new row to **Section 1: Master Public Leaderboard Lineage** with the next sequential `#`.
   - If the score beats the current record, update the header banner:
     `**Current Workspace State of the Art**: **X.XXX Public LB** (<notebook_name>)`.
   - If the submission was a negative ablation with an important lesson, also append a row to **Section 3: The Graveyard**.

---

### Path B: Imported / Copied External Notebook

When a user copies someone else's notebook into the repository:

1. **Analyze the `.ipynb` Internally**:
   - Read the markdown cells at the top for description, citations, and reported scores.
   - Inspect the pipeline components across code cells:
     - **Candidate pool**: train only, COCONUT, PubChem, ChEBI, LIPID MAPS?
     - **Feature representation**: Fingerprint bits (ECFP4, 4096-bit, 10226-bit), DreaMS embeddings, or peak cosine?
     - **Scoring & Ranking**: LightGBM GBDT ranker, neural bi-encoder, ICEBERG simulator, GLACIER, or formula re-ranker?
     - **Fusion**: Single model, rank blend, RRF, or score averaging?
2. **Synthesize in Plain English**:
   - Distill the core technique into 2–3 simple bullet points without copying raw code.
3. **Record Score**:
   - Extract LB score from notebook title/markdown (e.g., `lb-0-399` -> 0.399). If not present, ask the user or mark as `Pending LB`.
4. **Append to Section 1**:
   - Add the entry to **Section 1: Master Public Leaderboard Lineage**.

---

### Path C: Spec Gate / Colab Validation Completed

When an offline experiment, Colab notebook, or gate verification finishes:

1. **Inspect the Result Files**:
   - Check for updated files in the relevant spec directory:
     - `printouts.txt` or `logs.txt`
     - `gate_report.md`
     - Result JSONs (`p3_eval.json`, `lx_gpu.json`, `new_gate_results.txt`, `phase0_colab_gate.json`)
2. **Extract Benchmark Metrics**:
   - Component name (e.g., *Novel Decoder P3*, *Simulator Adduct Gate G2b*, *FPNet Unseen Checkpoint*).
   - Metric & Pass threshold (e.g., *median maxTan on pseudo_c3 vs null $\ge +0.10$*).
   - Measured result and lift (e.g., *0.3092 vs 0.1206, lift +0.1885*).
   - Verdict: **PASS**, **HOLD**, or **FAIL**.
3. **Append to Section 2**:
   - Add a clean row or update the corresponding component table in **Section 2: Component & Offline Validations**.
4. **Handle Failures**:
   - If the gate failed and revealed a technical dead end or design flaw (e.g. data leak, memory explosion, isomer failure), add an entry to **Section 3: The Graveyard** explaining why it failed and the rule to remember.

---

## Table Schemas & Markdown Formatting Rules

### Section 1: Public Leaderboard Row Format
```markdown
| **XX** | `<notebook_filename>` | MM-DD | • <Bullet 1: Main technique><br>• <Bullet 2: Specific change/data source> | 0.XXX | **+0.0XX** | <Brief verdict or status> |
```

### Section 2: Gate / Component Row Format
```markdown
| **<Gate Name>** | `<notebook_or_script>` | <Condition / Threshold> | <Exact Measured Numbers> | **PASS / FAIL** |
```

### Section 3: The Graveyard Row Format
```markdown
| **<Attempted Change>** | LB dropped **−0.0XX** (or failed gate) | <Clear explanation of root cause> | <Rule / Principle to remember> |
```

---

## Quality Checklist Before Saving `EXPERIMENTS.md`

- [ ] Sequential numbering in Section 1 is preserved (no duplicate or skipped IDs).
- [ ] Deltas (`+` or `−`) are explicitly calculated and highlighted in bold.
- [ ] If a new score is the highest so far, the header banner (`Current Workspace State of the Art`) is updated.
- [ ] Bullets use plain, readable English (focused on *what it did*, not code line numbers).
- [ ] The file strictly maintains the 3 core sections without re-adding deprecated sections.
- [ ] Markdown table vertical pipes `|` are properly closed and aligned.
