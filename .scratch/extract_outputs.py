import nbformat

def print_outputs(nb_path):
    nb = nbformat.read(nb_path, as_version=4)
    print(f"=== Outputs for {nb_path} ===")
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            outs = cell.get('outputs', [])
            for out in outs:
                if out.output_type == 'stream' and 'text' in out:
                    text = out['text'].strip()
                    for line in text.splitlines():
                        if any(k in line for k in ['VERIFICATION', 'audit', 'summary', 'LB', 'changed', 'CLAW', 'SCORE', 'E004', 'FACTORIAL']):
                            print(f"Cell {idx}: {line[:120]}")
                elif out.output_type == 'execute_result' and 'data' in out and 'text/plain' in out['data']:
                    text = out['data']['text/plain'].strip()
                    for line in text.splitlines():
                        if any(k in line for k in ['VERIFICATION', 'audit', 'summary', 'LB', 'changed', 'CLAW', 'SCORE', 'E004', 'FACTORIAL']):
                            print(f"Cell {idx} (res): {line[:120]}")

print_outputs('casmi-notebook-0-433.ipynb')
print()
print_outputs('casmi26-e006-window-40-ice-fallback-true.ipynb')
