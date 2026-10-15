# %% [markdown]
# # Harness ICE-only completion: C2_hard + C1_xlib
# The 2026-10-13 dump kernels `casmi-0433-harness-{c2hard,c1xlib}` ran on the pinned E006 image (Python 3.12); the
# 0.433 notebook's ICEBERG preseed only knows the cp313 rdkit wheel, so ICEBERG never ran and `state["ice"]` is empty.
# Everything else in those dumps is complete. This kernel scores only the missing part: the ICE runner input, rebuilt
# offline from each dump's state.pkl + val_comp/test.parquet exactly as the notebook builds it (verified: identical to
# the in-run GLACIER input, gl_in.json, on every [M+H]+ item), through the same `ice_runner.py`, with the same
# 5400 s val-mode budget per cohort and `--any-rdkit`. The rdkit wheel is chosen by interpreter tag (the E006 way).
# Output: /kaggle/working/ice_out_<shard>.json (+ .meta.json), merged into state["ice"] offline.

# %%
import glob, hashlib, json, os, shutil, subprocess, sys, time
T0 = time.time()
SHARDS = ['C2_hard', 'C1_xlib']
BUDGET_S = 5400.0
SITE, WORK = '/kaggle/working/ice_site', '/kaggle/working/ice_work'


def find(pattern):
    hits = sorted(glob.glob(os.path.join('/kaggle/input', pattern), recursive=True))
    assert hits, f'not mounted: {pattern}'
    return hits[0]


ICE_PKG = os.path.dirname(find('**/casmi26-iceberg/**/ice_runner.py'))
sys.path.append(ICE_PKG)
import fuse as ice_fuse
print('python', sys.version.split()[0], '| ICE_PKG', ICE_PKG, flush=True)

# %%
def simulator_rdkit_wheel(pkg):
    """E006: the RDKit 2025.3.6 wheel (bundled .whl.ice or a mounted one) whose tags this interpreter supports."""
    from pip._vendor.packaging.tags import sys_tags
    from pip._vendor.packaging.utils import parse_wheel_filename
    compatible = set(sys_tags())
    bundled = sorted(glob.glob(os.path.join(pkg, 'wheels', 'rdkit-2025.3.6-*.whl*')))
    external = sorted(glob.glob('/kaggle/input/**/rdkit-2025.3.6-*.whl', recursive=True))
    for path in bundled + external:
        _, version, _, tags = parse_wheel_filename(os.path.basename(path).removesuffix('.ice'))
        if str(version) == '2025.3.6' and compatible.intersection(tags):
            return path
    raise FileNotFoundError(f'no RDKit 2025.3.6 wheel for Python {sys.version_info[:2]}: {bundled + external}')


def preseed_site(pkg, site):
    """E006 _claw_preseed_site: install the package wheels with the interpreter-matched rdkit, then write the
    .ice_site_ok marker exactly as install_site() derives it, so the runner never invokes pip itself."""
    srcs = sorted(glob.glob(os.path.join(pkg, 'wheels', '*.whl.ice')) + glob.glob(os.path.join(pkg, 'wheels', '*.whl')))
    keep = [w for w in srcs if not os.path.basename(w).lower().startswith('rdkit')] + [simulator_rdkit_wheel(pkg)]
    wdir = os.path.join(site, '.wheels')
    os.makedirs(wdir, exist_ok=True)
    whl = []
    for w in keep:
        dst = os.path.join(wdir, os.path.basename(w).removesuffix('.ice'))
        shutil.copyfile(w, dst)
        whl.append(dst)
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', '--no-compile',
                        '--disable-pip-version-check', '--upgrade', '--target', site] + whl,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print('preseed rc', r.returncode, (r.stdout or '')[-400:], flush=True)
    if r.returncode != 0:
        raise RuntimeError('preseed pip failed: ' + (r.stdout or '')[-600:])
    shutil.rmtree(wdir, ignore_errors=True)
    with open(os.path.join(site, '.ice_site_ok'), 'w') as f:
        f.write('\n'.join(os.path.basename(w) for w in srcs))
    print('site ready; rdkit wheel', os.path.basename(keep[-1]), flush=True)


preseed_site(ICE_PKG, SITE)

# %%
import torch
results = {}
for shard in SHARDS:
    fin = find(f'**/ice_in_{shard}.json')
    raw = open(fin, 'rb').read()
    items = json.loads(raw)
    n_c = sum(len(it['cands']) for it in items)
    print(f'ICE ONLY {shard} input md5 {hashlib.md5(raw).hexdigest()} molecules {len(items)} candidates {n_c}', flush=True)
    torch.cuda.empty_cache()
    wd = os.path.join(WORK, shard)
    out = ice_fuse.run_ice(ICE_PKG, items, workdir=wd, device='cuda', budget_s=BUDGET_S, site=SITE,
                           extra_args=['--any-rdkit'])
    meta_f = os.path.join(wd, 'ice_out.json.meta.json')
    meta = json.load(open(meta_f)) if os.path.exists(meta_f) else {'status': 'missing'}
    json.dump(out, open(f'/kaggle/working/ice_out_{shard}.json', 'w'))
    json.dump(meta, open(f'/kaggle/working/ice_out_{shard}.json.meta.json', 'w'))
    results[shard] = meta
    print(f"ICE ONLY {shard} status {meta.get('status')} mols_scored {meta.get('n_mols_scored')} of {meta.get('n_mols')} "
          f"cands_scored {meta.get('n_scored')} of {meta.get('n_cands')} out_mols {len(out)} {time.time() - T0:.0f}s",
          flush=True)

# %%
shutil.rmtree(SITE, ignore_errors=True)   # keep the output small: only ice_out_*.json and their meta
shutil.rmtree(WORK, ignore_errors=True)
bad = {s: m.get('status') for s, m in results.items() if m.get('status') != 'ok' or not m.get('n_mols_scored')}
assert not bad, f'ICE ONLY FAILED {bad}'
print('ICE ONLY ALL OK', {s: m.get('n_mols_scored') for s, m in results.items()}, f'{time.time() - T0:.0f}s')
