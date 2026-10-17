"""c3-iso I1b.1 (GPU): ICEBERG (casmi26-iceberg ice_runner, msg_all MassSpecGym weights) scores the frozen top-150 children
+ V22 fused top 10 of each pseudo_c3 molecule against the molecule's MEASURED query spectra (train.parquet rows, R10).
NO TRUTH is available in this kernel (the inputs dataset holds none). Output: ice_out_pseudo_c3.json + meta + counters.
Recipe = the working harness-iceonly kernel: pinned E006 image (py3.12), rdkit 2025.3.6 wheel from the ICE package preseeded
into --site, fuse.run_ice. LIMIT (env CASMI_LIMIT or limit.txt next to this file) = first N molecules (smoke)."""
import glob, hashlib, json, os, shutil, subprocess, sys, time
T0 = time.time()
LIMIT = int(os.environ.get("CASMI_LIMIT", "0") or 0)
BUDGET_S = float(os.environ.get("CASMI_BUDGET", "32000"))
SITE, WORK, OUT = "/kaggle/working/ice_site", "/kaggle/working/ice_work", "/kaggle/working"


def find(pattern):
    hits = sorted(glob.glob(os.path.join("/kaggle/input", pattern), recursive=True))
    assert hits, f"not mounted: {pattern}"
    return hits[0]


ICE_PKG = os.path.dirname(find("**/casmi26-iceberg/**/ice_runner.py"))
sys.path.append(ICE_PKG)
import fuse as ice_fuse
print("python", sys.version.split()[0], "| ICE_PKG", ICE_PKG, "| LIMIT", LIMIT, flush=True)


def simulator_rdkit_wheel(pkg):
    from pip._vendor.packaging.tags import sys_tags
    from pip._vendor.packaging.utils import parse_wheel_filename
    compatible = set(sys_tags())
    bundled = sorted(glob.glob(os.path.join(pkg, "wheels", "rdkit-2025.3.6-*.whl*")))
    external = sorted(glob.glob("/kaggle/input/**/rdkit-2025.3.6-*.whl", recursive=True))
    for path in bundled + external:
        _, version, _, tags = parse_wheel_filename(os.path.basename(path).removesuffix(".ice"))
        if str(version) == "2025.3.6" and compatible.intersection(tags):
            return path
    raise FileNotFoundError(f"no RDKit 2025.3.6 wheel for Python {sys.version_info[:2]}: {bundled + external}")


def preseed_site(pkg, site):
    srcs = sorted(glob.glob(os.path.join(pkg, "wheels", "*.whl.ice")) + glob.glob(os.path.join(pkg, "wheels", "*.whl")))
    keep = [w for w in srcs if not os.path.basename(w).lower().startswith("rdkit")] + [simulator_rdkit_wheel(pkg)]
    wdir = os.path.join(site, ".wheels")
    os.makedirs(wdir, exist_ok=True)
    whl = []
    for w in keep:
        dst = os.path.join(wdir, os.path.basename(w).removesuffix(".ice"))
        shutil.copyfile(w, dst)
        whl.append(dst)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--no-compile",
                        "--disable-pip-version-check", "--upgrade", "--target", site] + whl,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print("preseed rc", r.returncode, (r.stdout or "")[-400:], flush=True)
    if r.returncode != 0:
        raise RuntimeError("preseed pip failed: " + (r.stdout or "")[-600:])
    shutil.rmtree(wdir, ignore_errors=True)
    with open(os.path.join(site, ".ice_site_ok"), "w") as f:
        f.write("\n".join(os.path.basename(w) for w in srcs))
    print("site ready; rdkit wheel", os.path.basename(keep[-1]), flush=True)


preseed_site(ICE_PKG, SITE)
import torch
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, flush=True)
assert torch.cuda.is_available(), "ICE FAILED: no GPU"

fin = find("**/ice_in_pseudo_c3.json")
raw = open(fin, "rb").read()
items = json.loads(raw)
if LIMIT:
    cov = [i for i in items if i["spectra"]]
    items = cov[:LIMIT]                      # smoke: covered molecules only
n_c = sum(len(i["cands"]) for i in items)
print(f"input md5 {hashlib.md5(raw).hexdigest()} molecules {len(items)} candidates {n_c}", flush=True)
t1 = time.time()
out = ice_fuse.run_ice(ICE_PKG, items, workdir=WORK, device="cuda", budget_s=BUDGET_S, site=SITE, extra_args=["--any-rdkit"])
gpu_s = time.time() - t1
mf = os.path.join(WORK, "ice_out.json.meta.json")
meta = json.load(open(mf)) if os.path.exists(mf) else {"status": "missing"}
n_scored = sum(1 for m in out.values() for v in m.values() if v is not None)
n_none = sum(1 for m in out.values() for v in m.values() if v is None)
mols_scored = sum(1 for m in out.values() if any(v is not None for v in m.values()))
n_cov = sum(1 for i in items if i["spectra"])
res = dict(meta=meta, limit=LIMIT, n_mols_in=len(items), n_mols_covered=n_cov, n_cands_in=n_c, n_scored=n_scored,
           n_failed_or_null=n_none, mols_scored=mols_scored, gpu_seconds=gpu_s, gpu_minutes=gpu_s / 60,
           cands_per_s=(n_scored / gpu_s if gpu_s else None), total_s=time.time() - T0)
json.dump(out, open(f"{OUT}/ice_out_pseudo_c3.json", "w"))
json.dump(res, open(f"{OUT}/ice_run_summary.json", "w"), indent=1, default=str)
shutil.rmtree(SITE, ignore_errors=True)
shutil.rmtree(WORK, ignore_errors=True)
print("ICE COUNTERS", json.dumps({k: v for k, v in res.items() if k != "meta"}), flush=True)
print("ICE META", {k: meta.get(k) for k in ("status", "n_mols", "n_mols_covered", "n_mols_scored", "n_cands", "n_scored", "seconds", "rdkit", "device")}, flush=True)
# hard errors: ICE failing silently is not allowed
assert meta.get("status") == "ok", f"ICE FAILED status={meta.get('status')}"
assert n_scored > 0 and mols_scored > 0, "ICE FAILED: zero candidates scored"
assert mols_scored >= 0.9 * n_cov, f"ICE FAILED: only {mols_scored}/{n_cov} covered molecules scored (budget?)"
print("ICE I1b.1 ALL OK", flush=True)
