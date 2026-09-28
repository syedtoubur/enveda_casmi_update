import json

path = 'casmi_rank_matrix_colab.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

old_text = """# Bootstrap offline RDKit wheel if available in attached datasets
rdkit_whls = glob.glob('/content/casmi_stage/**/rdkit-*.whl', recursive=True)
if rdkit_whls:
    print(f"[INFO] Installing offline RDKit wheel: {rdkit_whls[0]}")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', rdkit_whls[0]], check=False)

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator, MACCSkeys
    from rdkit.Chem.Descriptors import ExactMolWt
    RDLogger.DisableLog('rdApp.*')
    HAVE_RDKIT = True
    print("[SUCCESS] RDKit 2026.3.3 is active and verified.")
except Exception as e:
    HAVE_RDKIT = False
    print(f"[WARNING] RDKit unavailable ({e}). Running without in-silico fragmentation.")"""

new_text = """# Bootstrap RDKit: try offline wheel first, fall back to PyPI for Colab Python version
try:
    from rdkit import Chem
except ImportError:
    rdkit_whls = glob.glob('/content/casmi_stage/**/rdkit-*.whl', recursive=True)
    installed = False
    if rdkit_whls:
        res = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', rdkit_whls[0]])
        installed = (res.returncode == 0)
    if not installed:
        print("[INFO] Offline wheel incompatible or missing. Installing RDKit from PyPI...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'rdkit'], check=True)

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator, MACCSkeys
    from rdkit.Chem.Descriptors import ExactMolWt
    RDLogger.DisableLog('rdApp.*')
    HAVE_RDKIT = True
    print("[SUCCESS] RDKit is active and verified.")
except Exception as e:
    HAVE_RDKIT = False
    print(f"[WARNING] RDKit unavailable ({e}). Running without in-silico fragmentation.")"""

modified = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        if old_text in src:
            src = src.replace(old_text, new_text)
            cell['source'] = [line + '\n' for line in src.splitlines()]
            modified = True
            break

if modified:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("SUCCESS: Patched casmi_rank_matrix_colab.ipynb")
else:
    print("Pattern not found in notebook.")
