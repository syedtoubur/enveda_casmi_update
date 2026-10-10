import os
import sys

print("==================================================================")
print("CASMI 2026: TICKET 0 VIABILITY SPIKE - GOOGLE COLAB + DRIVE CHECK")
print("==================================================================")

print("[1/4] Checking Google Drive mount at /content/drive...")
if not os.path.exists('/content/drive/MyDrive') and not os.path.exists('/content/drive/casmi_2026'):
    try:
        from google.colab import drive
        print("  Attempting python drive.mount...")
        drive.mount('/content/drive')
    except Exception as e:
        print(f"FAILED to mount Drive: {e}")
        sys.exit(1)
else:
    print("  [OK] /content/drive is mounted and ready!")

drive_root = '/content/drive/MyDrive/casmi_2026'
if not os.path.exists(drive_root):
    # Check alternate casing or direct path
    print(f"Warning: {drive_root} not found. Searching under /content/drive/MyDrive/...")
    candidates = [p for p in os.listdir('/content/drive/MyDrive') if 'casmi' in p.lower()]
    print(f"Found related folders: {candidates}")
    if candidates:
        drive_root = f"/content/drive/MyDrive/{candidates[0]}"
        print(f"Using alternate root: {drive_root}")
    else:
        print("ERROR: No casmi_2026 directory found in Google Drive.")
        sys.exit(1)

print(f"\n[2/4] Verifying folder contents at {drive_root}...")
expected_targets = [
    'casmi26-pubchem-tier/pc_mass.npy',
    'casmi26-v2-pool/pool_meta.parquet',
    'casmi26-fpnet-full1/fpnet_full1.pt',
    'casmi26-pubchem-popularity-prior/pc_lsid.npy',
    'casmi26-scratch-pool/pool_mass.npy'
]

found_count = 0
for rel_path in expected_targets:
    full_path = os.path.join(drive_root, rel_path)
    if os.path.exists(full_path):
        size_mb = os.path.getsize(full_path) / (1024 * 1024)
        print(f"  [OK] Found: {rel_path} ({size_mb:.2f} MB)")
        found_count += 1
    else:
        print(f"  [MISSING] Could not find: {rel_path}")

print(f"\n[3/4] Checkpoint Verification Summary: {found_count}/{len(expected_targets)} critical files confirmed.")

print("\n[4/4] Hardware and Accelerator Check:")
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    print(f"  CUDA Available: {cuda_ok}")
    if cuda_ok:
        print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
except Exception as e:
    print(f"  PyTorch CUDA check error: {e}")

print("\n==================================================================")
if found_count >= 3:
    print("SUCCESS: Remote Colab instance has fast, direct access to Drive assets!")
else:
    print("WARNING: Some files were not found. Verify drive_folder_structure path.")
print("==================================================================")
