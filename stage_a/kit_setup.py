"""Kaggle kernel bootstrap: rdkit pin, locate the harness kit + this idea's code/data dataset. Import first."""
import glob, os, shutil, subprocess, sys, tarfile

def one(pat):
    h = sorted(glob.glob(pat, recursive=True), key=len)
    assert h, pat
    return h[0]

def pip_rdkit():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit==2026.3.3", "pyyaml"], check=False)
    import rdkit
    print("rdkit", rdkit.__version__, flush=True)

def code_dir():
    d = os.path.dirname(one("/kaggle/input/**/code/a3.py"))
    sys.path.insert(0, d)
    return d

def data_dir():
    return os.path.dirname(one("/kaggle/input/**/code/a3.py")[:-len("a3.py")].rstrip("/"))

def kit_root(dst="/kaggle/working/kit"):
    tars = [t for t in glob.glob("/kaggle/input/**/kit.tar", recursive=True) if os.path.isfile(t)]
    if tars:
        tarfile.open(tars[0]).extractall(dst)
    else:
        hp = one("/kaggle/input/**/ENVEDA_casmi/research/harness/harness.py")
        shutil.copytree(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(hp)))), dst)
    r = f"{dst}/ENVEDA_casmi"
    sys.path.insert(0, f"{r}/research/harness")
    return r
