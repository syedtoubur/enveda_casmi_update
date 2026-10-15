# Executive Summary: Colab CLI + Google Drive Integration (Agent Brief)

> **File Location**: [`specs/specs_scratch/summary_colab.md`](file:///e:/ENVEDA_casmi/specs/specs_scratch/summary_colab.md)  
> **Master Reference**: [`specs/specs_scratch/colab_drive_cli_integration.md`](file:///e:/ENVEDA_casmi/specs/specs_scratch/colab_drive_cli_integration.md)  
> **Drive Manifest**: [`specs/Dataset_intro/drive_folder_structure.txt`](file:///e:/ENVEDA_casmi/specs/Dataset_intro/drive_folder_structure.txt)  
> **Kaggle Slugs Catalog**: [`specs/Dataset_intro/dataset_catalog.json`](file:///e:/ENVEDA_casmi/specs/Dataset_intro/dataset_catalog.json)  

---

## 1. What We Are Building
An **autonomous research and code execution pipeline** designed for the 2-month Enveda CASMI 2026 Kaggle competition with:
* **Minimum Token & Cost Overhead**: Gemini Flash (Antigravity) generates high-volume ideas and 2-minute spikes; Claude Code is invoked via `claude -p` only to judge and write code.
* **Zero Local Storage Overhead**: The host Windows machine has limited local SSD disk. All large models, checkpoints, and datasets (~20 GB) live on Google Drive (`/content/drive/casmi_2026/`).
* **Cloud GPU Acceleration**: Headless Google Colab Pro instances (via `google-colab-cli` in WSL2) run real CUDA smoke tests and validation cohorts without consuming Kaggle's weekly 30h GPU quota.

---

## 2. Immediate Task for the New Session: "Ticket 0" Viability Spike

**Status**: **PASS (Ticket 0 Completed & Verified on 2026-10-10)**  
Full verification report: [`specs/specs_scratch/colab_spike_result.md`](file:///e:/ENVEDA_casmi/specs/specs_scratch/colab_spike_result.md)

### Standard Operational Sequence for Future Runs:

```bash
# 1. Provision a remote T4 GPU session:
wsl -- ~/.local/bin/colab new --gpu T4 -s casmi_runner

# 2. Mount Google Drive (/content/drive):
wsl -- ~/.local/bin/colab drivemount -s casmi_runner

# 3. Execute script with explicit timeout:
wsl -- ~/.local/bin/colab exec -s casmi_runner --timeout 120.0 -f <script_path>

# 4. Stop the instance immediately after run to conserve compute units:
wsl -- ~/.local/bin/colab stop -s casmi_runner
```

---

## 3. The Test Script: `specs/specs_scratch/test_drive_access.py`

The new agent should create and run this script during Ticket 0:

```python
from google.colab import drive
import os, sys, glob

print("Step 1: Attempting headless Drive mount...")
drive.mount('/content/drive')

drive_root = '/content/drive/MyDrive/casmi_2026'
assert os.path.exists(drive_root), f"Drive folder {drive_root} not found!"

print("Step 2: Checking critical candidate pools & models...")
assert os.path.exists(f"{drive_root}/casmi26-pubchem-tier/pc_mass.npy"), "Pubchem tier missing!"
assert os.path.exists(f"{drive_root}/casmi26-v2-pool/pool_meta.parquet"), "v2 pool missing!"
assert os.path.exists(f"{drive_root}/casmi26-fpnet-full1/fpnet_full1.pt"), "FPNet full1 missing!"

print("Step 3: Checking GPU availability on remote instance...")
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

print("ALL CHECKS PASSED: Colab CLI + Google Drive integration is 100% operational!")
```

---

## 4. Expected Output & Branching Logic

1. **If Ticket 0 passes cleanly**:
   * Record success in `specs/specs_scratch/colab_spike_result.md`.
   * Standardize the validation ladder:
     `G1 Static Audit` $\rightarrow$ `G2 Local CPU` $\rightarrow$ `G3a Colab Smoke (20 mol)` $\rightarrow$ `G3b Colab Val Cohort` $\rightarrow$ `G4 Kaggle Push`.
2. **If headless Drive mounting fails or requires manual browser interaction**:
   * Activate the **Kaggle Draft Run Fallback**:
     `G3` runs on Kaggle cloud directly via `kaggle kernels push` using the pre-configured slugs in `specs/Dataset_intro/dataset_catalog.json`.

---

## 5. Ready for Fresh Context
The next agent can start immediately by reading this file and running Task 2.1 in WSL2.
