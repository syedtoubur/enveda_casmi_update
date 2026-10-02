---
name: analyze-notebook
description: Token-efficient inspection, semantic diffing, and multi-notebook lineage comparison using Jupytext py:percent
---

# Analyze Notebook Skill

## Overview
This skill guides agents to inspect, analyze, and compare Jupyter notebooks (`.ipynb`) using the **`py:percent`** format. Raw `.ipynb` files waste 80%–98% of the agent's context budget on JSON metadata, cell outputs, and base64 plots. Converting to `py:percent` isolates clean, executable code while preserving exact cell delineations (`# %%`).

---

## When to Use
Invoke this skill whenever:
1. **Analyzing an `.ipynb` notebook**: You need to understand a notebook's pipeline, dependencies, models, or logic.
2. **Comparing 2 notebooks**: You need to see the exact code differences between a baseline and a new submission.
3. **Comparing 3+ notebooks (Lineage / Ablation Analysis)**: You need to trace architecture evolution across multiple notebook checkpoints without blowing the context window.

---

## The Automated Comparison Tool

The workspace includes a dedicated helper script:
```bash
python .agents/skills/analyze-notebook/scripts/compare.py [notebooks...]
```

### 1. Single Notebook Inspection
Prints cell counts, lines per cell, cell hashes, and initial line summaries:
```bash
python .agents/skills/analyze-notebook/scripts/compare.py path/to/notebook.ipynb
```

### 2. Pairwise Notebook Comparison (2 Notebooks)
Aligns code cells, strips metadata/papermill noise, and prints a targeted semantic code diff:
```bash
python .agents/skills/analyze-notebook/scripts/compare.py baseline.ipynb candidate.ipynb
```

### 3. Multi-Notebook Comparison (3+ Notebooks)
Produces a bird's-eye **Cell Alignment Matrix** across all notebooks followed by **Sequential Lineage Deltas** (Notebook 1 → Notebook 2 → Notebook 3):
```bash
python .agents/skills/analyze-notebook/scripts/compare.py v4h.ipynb v4n.ipynb supernext.ipynb
```

---

## Manual Fallback Commands

If running custom shell commands or inspecting directly:

1. **Convert to `scratch/`**:
   ```bash
   jupytext --to py:percent path/to/notebook.ipynb -o scratch/notebook.py
   ```
2. **Inspect Specific Cells**:
   Open `scratch/notebook.py` with `view_file` using line numbers. Cells begin with `# %%`.
3. **Batch Convert All Notebooks**:
   ```powershell
   Get-ChildItem -Filter *.ipynb | ForEach-Object {
       jupytext --to py:percent $_.FullName -o "scratch/$($_.BaseName).py"
   }
   ```

---

## Core Rules for Agents
- **NEVER view or grep raw `.ipynb` JSON files directly.**
- Always check if a companion `.py` file already exists (enforced by `jupytext.toml`).
- Keep diffs focused on semantic code changes (ignore papermill timestamps, cell run durations, and pure comment shifts).
