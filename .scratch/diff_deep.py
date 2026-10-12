import difflib
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

# Let's map NB1 cells to NB2 cells by matching significant code lines
print("Comparing NB1 cells to NB2:")
matched_nb2_indices = set()

for i1, c1 in enumerate(code1):
    b1 = c1['body']
    # find best match in code2
    best_ratio = 0
    best_i2 = -1
    for i2, c2 in enumerate(code2):
        ratio = difflib.SequenceMatcher(None, b1, c2['body']).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_i2 = i2
    print(f"NB1[{i1}] best match NB2[{best_i2}] with ratio {best_ratio:.3f}")
    if best_ratio < 0.99:
        diff = difflib.unified_diff(
            b1.splitlines(), code2[best_i2]['body'].splitlines(),
            fromfile=f"NB1[{i1}]", tofile=f"NB2[{best_i2}]", lineterm=""
        )
        print("Diff:")
        for line in list(diff)[:30]:
            print("  ", line)

print("\nNB2 cells not closely matched (>0.95) to any NB1 cell:")
for i2, c2 in enumerate(code2):
    best_ratio = max(difflib.SequenceMatcher(None, c1['body'], c2['body']).ratio() for c1 in code1)
    if best_ratio < 0.95:
        fl = [l.strip() for l in c2['body'].splitlines() if l.strip() and not l.strip().startswith('#')]
        header = fl[0] if fl else "(empty)"
        print(f"NB2[{i2}] (max ratio {best_ratio:.3f}): {header[:80]}")
