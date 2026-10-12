from pathlib import Path
exec(Path('.scratch/inspect_diffs.py').read_text())
comment_cells = [1, 2, 3, 4, 5, 8, 10, 12, 14, 17, 19, 21, 23, 25]
for idx in comment_cells:
    b = code2[idx]['body']
    # check if lines contain statements
    lines = [l.strip() for l in b.splitlines() if l.strip()]
    is_doc = (b.strip().startswith("'''") and b.strip().endswith("'''")) or (b.strip().startswith('"""') and b.strip().endswith('"""'))
    print(f"NB2[{idx}]: lines={len(lines)}, is_pure_docstring={is_doc}, text={lines[0][:60]}")
