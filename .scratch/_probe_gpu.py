import subprocess, sys, os, platform
print("python", sys.version.split()[0], platform.platform())
print(subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv,noheader"],capture_output=True,text=True).stdout.strip())
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
x = torch.randn(2048,2048,device="cuda"); print("matmul ok", float((x@x).sum()) != 0)
print("drive dir exists:", os.path.isdir("/content/drive/MyDrive"))
print("PROBE_OK")
