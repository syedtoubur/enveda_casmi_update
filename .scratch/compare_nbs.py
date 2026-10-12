import re
from pathlib import Path

def get_cells(py_path):
    text = Path(py_path).read_text(encoding='utf-8')
    raw_cells = text.split('# %%')
    cells = []
    for c in raw_cells:
        c_strip = c.strip()
        if not c_strip:
            continue
        is_md = c_strip.startswith('[markdown]')
        lines = c.splitlines()
        body_lines = []
        for l in lines:
            l_strip = l.strip()
            if l_strip.startswith('# %%') or l_strip.startswith('# [markdown]') or 'papermill=' in l_strip:
                continue
            body_lines.append(l)
        body = '\n'.join(body_lines).strip()
        cells.append({
            'is_md': is_md,
            'body': body
        })
    return cells

cells1 = get_cells('.scratch/nb1_433.py')
cells2 = get_cells('.scratch/nb2_e006.py')

code1 = [c for c in cells1 if not c['is_md'] and c['body']]
code2 = [c for c in cells2 if not c['is_md'] and c['body']]

print(f"Code cells: NB1={len(code1)}, NB2={len(code2)}")

print("\n--- NB1 Code Cells ---")
for i, c in enumerate(code1):
    fl = [l.strip() for l in c['body'].splitlines() if l.strip() and not l.strip().startswith('#')]
    header = fl[0] if fl else "(empty/comments)"
    print(f"NB1[{i}] ({len(c['body'].splitlines())} lines): {header[:90]}")

print("\n--- NB2 Code Cells ---")
for i, c in enumerate(code2):
    fl = [l.strip() for l in c['body'].splitlines() if l.strip() and not l.strip().startswith('#')]
    header = fl[0] if fl else "(empty/comments)"
    print(f"NB2[{i}] ({len(c['body'].splitlines())} lines): {header[:90]}")
