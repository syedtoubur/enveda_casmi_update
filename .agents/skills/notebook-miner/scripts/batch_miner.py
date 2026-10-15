#!/usr/bin/env python3
"""Token-efficient batch notebook analyzer and orthogonal idea miner.

Extracts text guidance, external datasets, dependencies, and code deltas
from a collection of Jupyter notebooks (.ipynb) and compares them against
a baseline notebook to discover novel, orthogonal ideas.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def fast_load_notebook(path: str) -> Dict[str, Any]:
    """Fast parse of .ipynb or .py without loading outputs/plots."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Notebook file not found: {path}")

    if path_obj.suffix == '.py':
        return parse_py_percent(path)

    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            data = json.load(f)
    except Exception as e:
        # Fallback if corrupt or non-standard
        return {"error": str(e), "cells": [], "metadata": {}}

    cells = []
    for c in data.get('cells', []):
        cell_type = c.get('cell_type', '')
        source = c.get('source', [])
        if isinstance(source, list):
            text = "".join(source)
        else:
            text = str(source)
        cells.append({'cell_type': cell_type, 'source': text})

    return {'cells': cells, 'metadata': data.get('metadata', {})}


def parse_py_percent(path: str) -> Dict[str, Any]:
    """Parse a .py file formatted in py:percent format."""
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    raw_cells = re.split(r'\n(?=# %%|\Z)', content)
    cells = []
    for c in raw_cells:
        c = c.strip()
        if not c:
            continue
        lines = c.splitlines()
        header = lines[0] if lines else ''
        cell_type = 'markdown' if '# %% [markdown]' in header else 'code'
        body = '\n'.join(lines[1:] if lines else [])
        cells.append({'cell_type': cell_type, 'source': body})
    return {'cells': cells, 'metadata': {}}


def clean_code(source: str) -> str:
    """Strip pure comments and blank lines for robust semantic cell hashing."""
    lines = []
    for line in source.splitlines():
        s = line.strip()
        if s and not s.startswith('#') and not s.startswith('//'):
            lines.append(s)
    return '\n'.join(lines)


def hash_code(source: str) -> str:
    cleaned = clean_code(source)
    if not cleaned:
        return 'empty'
    return hashlib.sha256(cleaned.encode('utf-8')).hexdigest()[:12]


def extract_features(nb_data: Dict[str, Any], path: str) -> Dict[str, Any]:
    """Extract semantic features: markdown guidance, datasets, imports, functions, code hashes."""
    cells = nb_data.get('cells', [])
    filename = Path(path).name

    code_cells = []
    markdown_cells = []
    all_code_text = []

    datasets: Set[str] = set()
    imports: Set[str] = set()
    functions: Set[str] = set()
    classes: Set[str] = set()
    score_mentions: List[str] = []

    # Regex patterns
    ds_pattern = re.compile(r'(?:/kaggle/input/|input/|/content/)([\w\-]+)', re.IGNORECASE)
    import_pattern1 = re.compile(r'^\s*import\s+([a-zA-Z0-9_\.]+)', re.MULTILINE)
    import_pattern2 = re.compile(r'^\s*from\s+([a-zA-Z0-9_\.]+)\s+import', re.MULTILINE)
    func_pattern = re.compile(r'^\s*def\s+([a-zA-Z0-9_]+)\s*\(', re.MULTILINE)
    class_pattern = re.compile(r'^\s*class\s+([a-zA-Z0-9_]+)', re.MULTILINE)
    score_pattern = re.compile(r'(?:LB|CV|score|MRR|val|val_mrr|public)[\s:=_-]*([01]\.\d{3,5})', re.IGNORECASE)

    for idx, c in enumerate(cells):
        ctype = c.get('cell_type', '')
        source = c.get('source', '').strip()
        if not source:
            continue

        if ctype == 'markdown':
            markdown_cells.append({'idx': idx, 'source': source})
            for m in score_pattern.finditer(source):
                score_mentions.append(m.group(0).strip())
        elif ctype == 'code':
            h = hash_code(source)
            code_cells.append({'idx': idx, 'source': source, 'hash': h})
            all_code_text.append(source)

            # Datasets
            for ds in ds_pattern.findall(source):
                datasets.add(ds.strip())

            # Imports
            for imp in import_pattern1.findall(source):
                imports.add(imp.split('.')[0])
            for imp in import_pattern2.findall(source):
                imports.add(imp.split('.')[0])

            # Functions & Classes
            for fn in func_pattern.findall(source):
                functions.add(fn)
            for cls in class_pattern.findall(source):
                classes.add(cls)

            # Code comments score mentions
            for m in score_pattern.finditer(source):
                score_mentions.append(m.group(0).strip())

    # Look for title or header in first few markdown cells
    title = filename
    for mc in markdown_cells[:3]:
        lines = mc['source'].splitlines()
        for line in lines:
            line_s = line.strip()
            if line_s.startswith('# '):
                title = line_s.lstrip('# ').strip()
                break
        if title != filename:
            break

    # Look for markdown summary / guidance
    guidance_paragraphs = []
    for mc in markdown_cells:
        src = mc['source']
        # Look for headers like Overview, Changelog, Strategy, Summary, What's New
        if any(keyword in src.lower() for keyword in ['overview', 'changelog', 'strategy', 'approach', 'v', 'update', 'difference', 'novel', 'pipeline']):
            guidance_paragraphs.append(src[:500])

    return {
        'path': path,
        'filename': filename,
        'title': title,
        'total_cells': len(cells),
        'code_cells': code_cells,
        'code_count': len(code_cells),
        'markdown_cells': markdown_cells,
        'markdown_count': len(markdown_cells),
        'datasets': sorted(list(datasets)),
        'imports': sorted(list(imports)),
        'functions': sorted(list(functions)),
        'classes': sorted(list(classes)),
        'score_mentions': list(dict.fromkeys(score_mentions)),
        'guidance_paragraphs': guidance_paragraphs,
    }


def compare_against_baseline(candidate: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    """Compare candidate notebook against baseline to extract orthogonal ideas."""
    base_hashes = {c['hash']: c for c in baseline['code_cells'] if c['hash'] != 'empty'}
    cand_hashes = [c['hash'] for c in candidate['code_cells'] if c['hash'] != 'empty']

    # Exact cell matches
    matching_cells = [h for h in cand_hashes if h in base_hashes]
    total_cand = len(cand_hashes) if cand_hashes else 1
    match_ratio = len(matching_cells) / total_cand

    # Orthogonal Datasets
    base_ds = set(baseline['datasets'])
    new_datasets = [d for d in candidate['datasets'] if d not in base_ds]

    # Orthogonal Imports
    base_imps = set(baseline['imports'])
    new_imports = [i for i in candidate['imports'] if i not in base_imps]

    # Orthogonal Functions
    base_fns = set(baseline['functions'])
    new_functions = [f for f in candidate['functions'] if f not in base_fns]

    # Novel Code Cells (cells that do not exist in baseline)
    novel_cells = []
    modified_cells_diff = []

    for c in candidate['code_cells']:
        h = c['hash']
        if h == 'empty':
            continue
        if h not in base_hashes:
            # Extract first 4 non-comment lines as description
            lines = [l for l in c['source'].splitlines() if l.strip() and not l.strip().startswith('#')]
            summary = lines[0].strip()[:80] if lines else 'empty'
            novel_cells.append({
                'idx': c['idx'],
                'summary': summary,
                'source': c['source'],
            })

    # Lineage / Classification
    if match_ratio >= 0.95 and not new_datasets and not new_imports:
        classification = "Direct Clone / Param Tweak"
    elif match_ratio >= 0.70:
        classification = "Derivative / Branch"
    elif match_ratio >= 0.30:
        classification = "Hybrid / Substantial Mod"
    else:
        classification = "Distinct / Novel Pipeline"

    # Identify specific config diffs if near clone
    param_tweaks = []
    if match_ratio >= 0.85:
        # Check differences across novel cells
        for nc in novel_cells[:5]:
            src = nc['source']
            for line in src.splitlines():
                if any(param in line for param in ['WINDOW', 'FALLBACK', 'FUSION', 'PPM', 'THRESHOLD', 'ALPHA', 'BETA', 'WEIGHT', 'TOP_K']):
                    param_tweaks.append(line.strip())

    return {
        'filename': candidate['filename'],
        'title': candidate['title'],
        'classification': classification,
        'match_ratio': match_ratio,
        'novel_cell_count': len(novel_cells),
        'new_datasets': new_datasets,
        'new_imports': new_imports,
        'new_functions': new_functions[:15],
        'score_mentions': candidate['score_mentions'],
        'param_tweaks': list(dict.fromkeys(param_tweaks)),
        'novel_cells': novel_cells,
        'guidance_paragraphs': candidate['guidance_paragraphs'],
    }


def find_notebook_files(target_dir: str) -> List[str]:
    """Find all .ipynb files recursively in target directory."""
    files = []
    for root, _, filenames in os.walk(target_dir):
        for fn in filenames:
            if fn.endswith('.ipynb') and not fn.startswith('.'):
                files.append(os.path.join(root, fn))
    return sorted(files)


def generate_markdown_report(results: List[Dict[str, Any]], baseline_meta: Dict[str, Any]) -> str:
    """Format mining results into a comprehensive ideas report."""
    md = []
    md.append("# Kaggle Public Notebooks: Orthogonal Ideas Analysis\n")
    md.append(f"**Baseline Notebook**: `{baseline_meta['filename']}` ({baseline_meta['code_count']} code cells, {len(baseline_meta['datasets'])} datasets)\n")
    md.append(f"**Total Notebooks Analyzed**: {len(results)}\n")

    # Table of Contents / High Level Summary
    md.append("## 1. Executive Summary & Top Orthogonal Finds\n")
    
    # Filter for truly novel or distinct ideas
    novel_candidates = [r for r in results if r['classification'] != "Direct Clone / Param Tweak" or r['new_datasets'] or r['new_imports']]
    
    md.append(f"- **Distinct / Derivative Architectures**: {len(novel_candidates)}")
    md.append(f"- **Direct Clones / Hyperparameter Variations**: {len(results) - len(novel_candidates)}\n")

    md.append("| Notebook / Title | Match % | Classification | New Datasets / Inputs | New Packages | Score Claims |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in results:
        ds_str = ", ".join(f"`{d}`" for d in r['new_datasets'][:2]) or "—"
        imp_str = ", ".join(f"`{i}`" for i in r['new_imports'][:2]) or "—"
        scores = ", ".join(r['score_mentions'][:2]) or "—"
        md.append(f"| **{r['filename'][:35]}**<br>*{r['title'][:40]}* | {r['match_ratio']*100:.1f}% | {r['classification']} | {ds_str} | {imp_str} | {scores} |")
    md.append("\n---\n")

    # Section 2: Novel Datasets & External Resources
    all_new_datasets = {}
    for r in results:
        for ds in r['new_datasets']:
            all_new_datasets.setdefault(ds, []).append(r['filename'])

    if all_new_datasets:
        md.append("## 2. New External Datasets & Models Discovered\n")
        md.append("These datasets and pretrained assets are used by public kernels but **do NOT exist** in the baseline:\n")
        for ds, nbs in all_new_datasets.items():
            nb_list = ", ".join(f"`{n}`" for n in nbs[:3])
            md.append(f"- **`{ds}`** (used in {nb_list})")
        md.append("\n---\n")

    # Section 3: Detailed Deep Dives on Orthogonal Notebooks
    md.append("## 3. Detailed Orthogonal Idea Breakdowns\n")
    for r in results:
        if r['classification'] == "Direct Clone / Param Tweak" and not r['new_datasets'] and not r['new_imports']:
            continue

        md.append(f"### 💡 {r['title']} (`{r['filename']}`)")
        md.append(f"- **Relationship to Baseline**: {r['classification']} ({r['match_ratio']*100:.1f}% shared code cells)")
        if r['score_mentions']:
            md.append(f"- **Score / Metric Notes**: {', '.join(r['score_mentions'])}")
        if r['new_datasets']:
            md.append(f"- **New Datasets**: {', '.join(f'`{d}`' for d in r['new_datasets'])}")
        if r['new_imports']:
            md.append(f"- **New Packages**: {', '.join(f'`{i}`' for i in r['new_imports'])}")
        if r['new_functions']:
            md.append(f"- **New Functions**: {', '.join(f'`{f}`' for f in r['new_functions'][:8])}")

        # Author Guidance
        if r['guidance_paragraphs']:
            md.append("\n**Author Markdown Guidance / Hypotheses**:")
            for gp in r['guidance_paragraphs'][:2]:
                quoted = "\n".join(f"> {l}" for l in gp.splitlines()[:10])
                md.append(quoted)

        # Snippets of novel cells
        if r['novel_cells']:
            md.append("\n**Key Novel Code Highlights**:")
            for nc in r['novel_cells'][:3]:
                md.append(f"```python\n# [Cell {nc['idx']}] {nc['summary']}\n")
                lines = nc['source'].splitlines()[:25]
                md.append("\n".join(lines))
                if len(nc['source'].splitlines()) > 25:
                    md.append("\n# ... [truncated]")
                md.append("\n```")
        md.append("\n")

    # Section 4: Param Tweaks in Clones
    param_clones = [r for r in results if r['param_tweaks']]
    if param_clones:
        md.append("## 4. Hyperparameter Shifts Observed in Fork Trees\n")
        for pc in param_clones[:10]:
            md.append(f"- **`{pc['filename']}`**:")
            for pt in pc['param_tweaks']:
                md.append(f"  - `{pt}`")
        md.append("\n")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="Mine orthogonal ideas across Kaggle notebooks.")
    parser.add_argument("--baseline", required=True, help="Baseline notebook (.ipynb or .py)")
    parser.add_argument("--dir", help="Directory of candidate notebooks")
    parser.add_argument("--candidates", nargs="*", help="List of candidate notebooks")
    parser.add_argument("--output", default=".scratch/ideas_kaggle.md", help="Output markdown path")
    parser.add_argument("--json", help="Optional JSON output path")
    args = parser.parse_args()

    # Find candidates
    candidate_paths = []
    if args.candidates:
        candidate_paths.extend(args.candidates)
    if args.dir:
        candidate_paths.extend(find_notebook_files(args.dir))

    if not candidate_paths:
        print("No candidate notebooks found.")
        sys.exit(1)

    print(f"Loading baseline: {args.baseline} ...")
    base_data = fast_load_notebook(args.baseline)
    base_features = extract_features(base_data, args.baseline)
    print(f"Baseline parsed: {base_features['code_count']} code cells, {len(base_features['datasets'])} datasets.")

    results = []
    print(f"\nMining {len(candidate_paths)} candidate notebooks...")
    for idx, path in enumerate(candidate_paths, 1):
        try:
            cand_data = fast_load_notebook(path)
            cand_features = extract_features(cand_data, path)
            comparison = compare_against_baseline(cand_features, base_features)
            results.append(comparison)
            print(f"[{idx}/{len(candidate_paths)}] {comparison['filename'][:30]}: {comparison['classification']} ({comparison['match_ratio']*100:.1f}%)")
        except Exception as e:
            print(f"[{idx}/{len(candidate_paths)}] Failed to parse {path}: {e}")

    # Generate Markdown Report
    report_md = generate_markdown_report(results, base_features)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report_md)
    print(f"\nReport written to: {out_path} ({len(report_md)} bytes)")

    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"JSON results saved to: {args.json}")


if __name__ == '__main__':
    main()
