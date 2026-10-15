---
name: notebook-miner
description: Fast batch extraction, semantic comparison, markdown/comment guidance analysis, and orthogonal idea mining across multiple Kaggle notebooks (.ipynb) against a baseline notebook.
---

# Notebook Miner Skill

## Overview
This skill enables rapid, token-efficient mining of ideas across dozens of competition `.ipynb` notebooks. It solves three critical problems:
1. **Token efficiency**: Avoids dumping massive raw `.ipynb` JSON outputs and base64 plots by stripping outputs on-the-fly and isolating code + markdown guidance.
2. **Duplicate/Clone Filtering**: Automatically hashes code cells to identify forks and near-identical copies (e.g. 95%+ identical to baseline), highlighting only parameter or single-line tweaks.
3. **Orthogonal Innovation Extraction**: Pinpoints completely new data sources (`/kaggle/input/...`), new Python packages/libraries, novel algorithms, and author guidance from markdown/comments that do not exist in the baseline notebook.

---

## Tooling & Commands

The skill provides a dedicated Python script:
```bash
python .agents/skills/notebook-miner/scripts/batch_miner.py [options]
```

### Key Arguments:
- `--baseline <path>`: Path to the reference notebook (e.g. `casmi26-e006-window-40-ice-fallback-true.ipynb`).
- `--candidates <paths...>` or `--dir <directory>`: One or more candidate `.ipynb` or `.py` files, or a folder containing them.
- `--output <path>`: Markdown report output path (e.g. `.scratch/ideas_kaggle.md`).
- `--include-markdown`: Extract and summarize markdown guidance cells.
- `--min-novelty <float>`: Threshold for flagging non-trivial divergence (default: 0.05).

---

## Workflow for Competition Research

1. **Pull Public Kernels**:
   ```bash
   kaggle kernels list --competition enveda-CASMI26-molecule-id-mass-spectra --page-size 100
   kaggle kernels pull <author/slug> -p .scratch/kaggle_notebooks/<slug>
   ```

2. **Run Batch Mining**:
   ```bash
   python .agents/skills/notebook-miner/scripts/batch_miner.py \
       --baseline casmi26-e006-window-40-ice-fallback-true.ipynb \
       --dir .scratch/kaggle_notebooks \
       --output .scratch/ideas_kaggle.md
   ```

3. **Synthesize Findings**:
   Inspect the generated report, verify high-leverage orthogonal ideas (novel datasets, pretrained models, new post-processing rules), and integrate them into the repo roadmap (`research/`).
