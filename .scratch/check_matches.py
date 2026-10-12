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

print(f"Code cells count: NB1={len(code1)}, NB2={len(code2)}")

# Let's map each cell
for i, c in enumerate(code2):
    # see if it exactly matches any cell in code1
    matches = [j for j, c1 in enumerate(code1) if c1['body'] == c['body']]
    if matches:
        print(f"NB2[{i}] EXACT match NB1[{matches[0]}]")
    else:
        # find closest
        best_j = max(range(len(code1)), key=lambda j: difflib.SequenceMatcher(None, c['body'], code1[j]['body']).ratio())
        ratio = difflib.SequenceMatcher(None, c['body'], code1[best_j]['body']).ratio()
        print(f"NB2[{i}] CLOSE match NB1[{best_j}] (similarity {ratio:.3f})")
