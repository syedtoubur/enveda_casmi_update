import os, shutil, hashlib
src = "/content/p4_inputs.pkl"
dst_dir = "/content/drive/MyDrive/casmi_2026/_novel/p4"
assert os.path.isdir("/content/drive/MyDrive"), "drive not mounted"
os.makedirs(dst_dir, exist_ok=True)
dst = os.path.join(dst_dir, "p4_inputs.pkl")
shutil.copyfile(src, dst)
os.sync()
h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
print("SRC", os.path.getsize(src), h(src))
print("DST", os.path.getsize(dst), h(dst))
print("LIST", os.listdir(dst_dir))
