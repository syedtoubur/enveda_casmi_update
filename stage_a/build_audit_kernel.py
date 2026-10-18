"""Stage the c3 signal-audit kernel: audit.py into the Stage A code dataset, kernel dir next to the Stage A kernels.
Nothing here touches Kaggle (push: kaggle datasets version, then kaggle kernels push)."""
import json, os, shutil
SA = os.path.dirname(os.path.abspath(__file__))
W = "E:/casmi/stage_a_kaggle"; CODE = f"{W}/data/code"; KD = f"{W}/kernels/audit"
shutil.copy(f"{SA}/audit.py", CODE)
os.makedirs(KD, exist_ok=True)
shutil.copy(f"{SA}/audit_kernel.py", f"{KD}/run_audit.py")
json.dump({"id": "syedtoubur/c3iso-signal-audit", "title": "c3iso-signal-audit", "code_file": "run_audit.py",
           "language": "python", "kernel_type": "script", "is_private": "true", "enable_gpu": "false", "enable_tpu": "false",
           "enable_internet": "true",
           "dataset_sources": ["syedtoubur/casmi26-harness-kit", "syedtoubur/casmi26-c3iso-stagea-in",
                               "syedtoubur/validation-data-casmi-2026"],
           "competition_sources": [],
           "kernel_sources": ["syedtoubur/c3iso-stagea-a3-s0", "syedtoubur/c3iso-stagea-a3-s1",
                              "syedtoubur/c3iso-stagea-a3-pseudo"]},
          open(f"{KD}/kernel-metadata.json", "w"), indent=1)
print("staged", KD)
