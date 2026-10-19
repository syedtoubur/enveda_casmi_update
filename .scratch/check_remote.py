import os
import subprocess

print("--- CHECKING DRIVE MOUNT STATUS ---")
print("Exists /content/drive:", os.path.exists('/content/drive'))
print("Exists /content/drive/MyDrive:", os.path.exists('/content/drive/MyDrive'))

if os.path.exists('/content/drive/MyDrive'):
    print("MyDrive contents:", os.listdir('/content/drive/MyDrive')[:10])
    casmi_path = '/content/drive/MyDrive/casmi_2026'
    if os.path.exists(casmi_path):
        print("FOUND casmi_2026! Contents:", os.listdir(casmi_path)[:10])
    else:
        print("casmi_2026 not found directly under MyDrive, searching...")
else:
    print("Checking mount output:")
    try:
        out = subprocess.check_output(['mount']).decode()
        for line in out.splitlines():
            if 'drive' in line.lower() or 'fuse' in line.lower():
                print("Mount line:", line)
    except Exception as e:
        print("Error checking mount:", e)
