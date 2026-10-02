#!/usr/bin/env python3
"""Token-efficient Jupyter Notebook analyzer and multi-notebook comparator.

Usage:
    # 1 notebook: inspect structure & cell summary
    python compare.py notebook.ipynb

    # 2 notebooks: clean semantic diff
    python compare.py nb1.ipynb nb2.ipynb

    # 3+ notebooks: cell alignment matrix & chronological lineage delta
    python compare.py nb1.ipynb nb2.ipynb nb3.ipynb ...
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import os
import re
import subprocess
import sys


def ensure_py_percent(path: str, scratch_dir: str = 'scratch') -> str:
    """Ensure path is a py:percent script. If .ipynb, convert to scratch/ via jupytext."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    if path.endswith('.py'):
        return path

    if not path.endswith('.ipynb'):
        return path

    base = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(scratch_dir, exist_ok=True)
    out_py = os.path.join(scratch_dir, f"{base}.py")

    # Convert if missing or older than .ipynb
    if not os.path.exists(out_py) or os.path.getmtime(path) > os.path.getmtime(out_py):
        cmd = [sys.executable, '-m', 'jupytext', '--to', 'py:percent', path, '-o', out_py]
        r = subprocess.run(cmd, capture_output=True, text=True, errors='replace')
        if r.returncode != 0:
            subprocess.run(['jupytext', '--to', 'py:percent', path, '-o', out_py], check=True)

    return out_py


def parse_cells(py_path: str, code_only: bool = True) -> list[dict]:
    """Parse a py:percent file into structured cells."""
    with open(py_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    raw_cells = re.split(r'\n(?=# %%|\Z)', content)
    cells = []
    code_idx = 0

    for idx, c in enumerate(raw_cells):
        c = c.strip()
        if not c:
            continue

        lines = c.splitlines()
        header = lines[0] if lines else ''
        cell_type = 'markdown' if '# %% [markdown]' in header else 'code'

        if code_only and cell_type == 'markdown':
            continue

        body_lines = []
        for line in lines[1:]:
            if line.startswith('# %% papermill=') or line.startswith('# %% [markdown] papermill='):
                continue
            body_lines.append(line)

        body = '\n'.join(body_lines).strip()
        if not body and cell_type == 'code':
            continue

        first_line = ''
        for bl in body_lines:
            sbl = bl.strip().lstrip('#').strip()
            if sbl and not sbl.startswith('jupyter:') and not sbl.startswith('papermill:'):
                first_line = sbl
                break

        code_text = '\n'.join(
            l.strip() for l in body_lines if l.strip() and not l.strip().startswith('#')
        )
        h = hashlib.sha256(code_text.encode('utf-8')).hexdigest()[:10] if code_text else 'empty'

        cells.append({
            'global_idx': idx,
            'code_idx': code_idx,
            'type': cell_type,
            'first_line': first_line[:75],
            'body': body,
            'code_text': code_text,
            'hash': h,
            'line_count': len(body_lines),
        })
        if cell_type == 'code':
            code_idx += 1

    return cells


def analyze_single(path: str):
    """Analyze a single notebook token-efficiently."""
    py_path = ensure_py_percent(path)
    cells = parse_cells(py_path, code_only=False)

    raw_size = os.path.getsize(path) if path.endswith('.ipynb') else os.path.getsize(py_path)
    py_size = os.path.getsize(py_path)

    print(f"\n=== Notebook Analysis: {os.path.basename(path)} ===")
    print(f"Format: py:percent ({py_size:,} bytes, ~{py_size // 4:,} tokens)")
    if path.endswith('.ipynb'):
        saving = (1.0 - (py_size / max(1, raw_size))) * 100
        print(f"Original .ipynb: {raw_size:,} bytes | Token Reduction: {saving:.1f}%")

    code_cells = [c for c in cells if c['type'] == 'code']
    print(f"Total Cells: {len(cells)} (Code: {len(code_cells)}, Markdown: {len(cells) - len(code_cells)})\n")
    print(f"{'Cell':<6} | {'Type':<8} | {'Lines':<6} | {'Hash':<10} | {'Cell Summary / First Line'}")
    print("-" * 80)
    for c in cells:
        idx_str = f"C{c['code_idx']}" if c['type'] == 'code' else f"M{c['global_idx']}"
        summary = c['first_line'] or '(empty / directives)'
        print(f"{idx_str:<6} | {c['type']:<8} | {c['line_count']:<6} | {c['hash']:<10} | {summary}")


def compare_two(p1: str, p2: str):
    """Clean semantic diff between two notebooks."""
    py1 = ensure_py_percent(p1)
    py2 = ensure_py_percent(p2)

    name1 = os.path.basename(p1)
    name2 = os.path.basename(p2)

    cells1 = parse_cells(py1, code_only=True)
    cells2 = parse_cells(py2, code_only=True)

    print(f"\n=== Comparing Code Cells: {name1} vs {name2} ===")
    print(f"Code Cells: {len(cells1)} vs {len(cells2)}\n")

    print(f"{'Cell':<6} | {'Status':<12} | {'Summary'}")
    print("-" * 75)
    max_cells = max(len(cells1), len(cells2))
    diff_cells = []

    for i in range(max_cells):
        c1 = cells1[i] if i < len(cells1) else None
        c2 = cells2[i] if i < len(cells2) else None

        if c1 and c2:
            if c1['hash'] == c2['hash']:
                status = "Identical"
            else:
                status = "MODIFIED"
                diff_cells.append((i, c1, c2))
            summary = c2['first_line'] or c1['first_line']
        elif c1 and not c2:
            status = "REMOVED"
            diff_cells.append((i, c1, None))
            summary = c1['first_line']
        else:
            status = "ADDED"
            diff_cells.append((i, None, c2))
            summary = c2['first_line']

        print(f"C{i:<5} | {status:<12} | {summary[:55]}")

    if not diff_cells:
        print("\nAll code cells are functionally identical.")
        return

    print(f"\n--- Detailed Diffs for Modified Cells ({len(diff_cells)} cells) ---")
    for idx, c1, c2 in diff_cells:
        print(f"\n[ Code Cell {idx} Difference ]")
        l1 = [l.strip() for l in c1['body'].splitlines() if l.strip()] if c1 else []
        l2 = [l.strip() for l in c2['body'].splitlines() if l.strip()] if c2 else []
        diff = list(difflib.unified_diff(l1, l2, fromfile=f"{name1}:C{idx}", tofile=f"{name2}:C{idx}", lineterm=''))

        changed_lines = [l for l in diff if l.startswith(('+', '-')) and not l.startswith(('+++', '---'))]
        # Only show the relevant changes
        for line in changed_lines:
            if 'papermill' not in line:
                if len(line) > 130:
                    print(line[:130] + " ... [truncated]")
                else:
                    print(line)


def compare_multiple(paths: list[str]):
    """Compare 3 or more notebooks using an alignment matrix and sequential lineage."""
    names = [os.path.basename(p) for p in paths]
    py_paths = [ensure_py_percent(p) for p in paths]
    all_cells = [parse_cells(p, code_only=True) for p in py_paths]

    print(f"\n=== Multi-Notebook Code Alignment Matrix ({len(paths)} Notebooks) ===")
    for i, name in enumerate(names, 1):
        print(f"  [{i}] {name} ({len(all_cells[i-1])} code cells)")

    max_len = max(len(c) for c in all_cells)

    col_w = max(14, 60 // len(names))
    header_cols = " | ".join(f"[{i}] {names[i-1][:col_w-4]:<{col_w-4}}" for i in range(1, len(names) + 1))
    print(f"\n{'Cell':<6} | {'Summary':<30} | {header_cols}")
    print("-" * (42 + len(names) * (col_w + 3)))

    for ci in range(max_len):
        hashes = []
        summary = ""
        for n_idx, cells in enumerate(all_cells):
            if ci < len(cells):
                hashes.append(cells[ci]['hash'])
                if not summary and cells[ci]['first_line']:
                    summary = cells[ci]['first_line'][:28]
            else:
                hashes.append("---")

        base_h = hashes[0]
        status_strs = []
        for h in hashes:
            if h == "---":
                status_strs.append(f"{'Absent':<{col_w}}")
            elif h == base_h:
                status_strs.append(f"{'Identical':<{col_w}}")
            else:
                status_strs.append(f"{'MODIFIED':<{col_w}}")

        row = " | ".join(status_strs)
        print(f"C{ci:<5} | {summary:<30} | {row}")

    print(f"\n=== Sequential Lineage Deltas ===")
    for i in range(len(paths) - 1):
        p_from = paths[i]
        p_to = paths[i + 1]
        print(f"\n>> Step {i+1}: {names[i]} -> {names[i+1]}")
        c_from = all_cells[i]
        c_to = all_cells[i + 1]

        deltas = []
        for ci in range(max(len(c_from), len(c_to))):
            h_from = c_from[ci]['hash'] if ci < len(c_from) else "ABSENT"
            h_to = c_to[ci]['hash'] if ci < len(c_to) else "ABSENT"
            if h_from != h_to:
                title = (c_to[ci]['first_line'] if ci < len(c_to) else c_from[ci]['first_line'])[:45]
                status_desc = "ADDED" if h_from == "ABSENT" else ("REMOVED" if h_to == "ABSENT" else "MODIFIED")
                deltas.append(f"  • Cell C{ci} [{status_desc}]: {title}")

        if deltas:
            for d in deltas:
                print(d)
        else:
            print("  • No functional code cell changes detected.")


def main():
    parser = argparse.ArgumentParser(description="Token-efficient notebook comparator.")
    parser.add_argument("notebooks", nargs="+", help="Path to 1 or more .ipynb or .py files")
    args = parser.parse_args()

    if sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    if len(args.notebooks) == 1:
        analyze_single(args.notebooks[0])
    elif len(args.notebooks) == 2:
        compare_two(args.notebooks[0], args.notebooks[1])
    else:
        compare_multiple(args.notebooks)


if __name__ == '__main__':
    main()
