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

print("=== NB1[11] ===")
print(code1[11]['body'][:300])

print("\n=== NB1[12] ===")
print(code1[12]['body'][:300])

print("\n=== NB2[26] ===")
print(code2[26]['body'][:300])

print("\n=== NB2[27] ===")
print(code2[27]['body'][:300])

print("\n=== NB2[28] ===")
print(code2[28]['body'])

print("\n=== NB2[29] ===")
print(code2[29]['body'])

print("\n=== NB2[30] ===")
print(code2[30]['body'])
