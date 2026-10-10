casmi26-iceberg -- ICEBERG isomer re-scorer (offline, subprocess)
ICEBERG 2.1 (ms-pred, MassSpecGym msg_all checkpoint) predicts each candidate's MS/MS at every covered query
spectrum's adduct / instrument / 5 eV CE bucket; the score ice_ex_mean = mean entropy similarity (query vs
prediction) over the molecule's covered spectra ([M+H]+ / [M+Na]+, positive mode). Identical to the iceberg-track
panel definition (work/analysis/fwd/iceberg). Used as a post-ranker re-scorer of same-formula groups.

notebook use (no internet; the notebook keeps its own RDKit 2026.03)
import sys, glob, os, torch
PKG = os.path.dirname(sorted(glob.glob('/kaggle/input/**/casmi26-iceberg/**/ice_runner.py', recursive=True) +
                             glob.glob('/kaggle/input/**/ice_runner.py', recursive=True), key=len)[0])
sys.path.insert(0, PKG)
from fuse import build_ice_input, ice_candidates, run_ice, rerank

torch.cuda.empty_cache()                       # free cached GPU memory of the main process first (needs ~5 GB at bs 96)
try:                                           # the whole ICE block is optional: any exception -> keep the ranker order
    cands = {mid: ice_candidates(R[mid].smiles, R[mid].formulas, top_n=N) for mid in mids}   # ranked order, top-N only
    items = build_ice_input(test_df, cands)    # test.parquet rows -> runner input (covered spectra only)
    ice = run_ice(PKG, items, workdir='/tmp/ice_work', device='cuda', budget_s=BUDGET, site='/tmp/ice_site')
except Exception as e:
    print('ICE skipped:', repr(e)); ice = {}
for mid in mids:                               # ice: {mid: {smiles: score | None}}
    # scores = the ranker's score, HIGHER = BETTER (negate a distance / loss); lists already in ranker order
    order = rerank(R[mid].smiles, R[mid].keys, R[mid].scores, R[mid].formulas, ice.get(mid, {}), lam=0.5, top_n=N)
    ...                                        # apply the permutation `order` to the ranked lists
Uploading: the Kaggle CLI skips sub-directories by default -- push with
kaggle datasets create -p pkg --dir-mode zip (or tar), then check in a kernel that ckpt/, src/, shim/ and
wheels/*.whl.ice exist under the mounted dataset (the runner reports no wheels / import errors in the meta
otherwise; scores are then all null).

run_ice runs python ice_runner.py <PKG> in.json out.json --device cuda --budget S --site /tmp/ice_site in a
subprocess. The runner installs wheels/*.whl.ice offline (pip --no-index --no-deps --target /tmp/ice_site,
~150 MB, a few seconds, done once), puts shim/, src/ and the site dir first on ITS sys.path (RDKit 2025.03.6 --
required: 2026.03 kekulizes differently and changes predictions; the runner refuses any other version), loads the
checkpoints and predicts in batches (deterministic torch algorithms by default). out.json is written all-null at
start and after every chunk; the budget stops cleanly; any failure -> null scores, exit code 0. A per-run
out.json.meta.json holds status / counts / timings / errors.

Wheels are shipped as *.whl.ice on purpose: the inference notebook installs /kaggle/input/**/rdkit-*.whl by glob
and must never pick up this RDKit 2025.03.

Recommended: N = 60 (keeps ~97% of the full-group fusion gain on the syn / np panels; N = 40 keeps 87-95%),
lam = 0.5, BUDGET = 2700 s. Expected T4 time for 400 test molecules at N = 60: ~25-30 min (786 unique
adduct / CE-bucket conditions x <= 60 candidates; ~25-32 predictions/s estimated, 79/s measured on an RTX 5090
laptop). Molecules not reached within the budget keep null scores = ranker order. Pass items in priority order.

Runner options: --bs 96 (auto-halved for long SMILES), --chunk-jobs 3000, --memfrac 0 (torch per-process GPU
fraction), --threads 0, --nondeterministic (~20% faster, run-to-run float noise), --any-rdkit (NOT exact).


whole directory structure

/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/MANIFEST.json
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/ice_runner.py
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/README.md
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/LICENSE-NOTICE.txt
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/fuse.py
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/ckpt/gen.pt
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/ckpt/inten.pt
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/wheels/einops-0.8.2-py3-none-any.whl.ice
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/wheels/torchmetrics-1.9.0-py3-none-any.whl.ice
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/wheels/lightning_utilities-0.15.3-py3-none-any.whl.ice
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/wheels/rdkit-2025.3.6-cp312-cp312-manylinux_2_28_x86_64.whl.ice
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/wheels/pytorch_lightning-2.6.6-py3-none-any.whl.ice
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/src/ms_pred/LICENSE
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/src/ms_pred/__init__.py
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/src/ms_pred/magma/fragmentation.py
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/src/ms_pred/magma/run_magma.py
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/src/ms_pred/iceberg/dag_data.py
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/src/ms_pred/iceberg/gen_model.py
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/src/ms_pred/iceberg/joint_model.py
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/src/ms_pred/iceberg/inten_model.py
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/src/ms_pred/nn_utils/mol_graph.py
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/src/ms_pred/nn_utils/nn_utils.py
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/src/ms_pred/nn_utils/form_embedder.py
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/src/ms_pred/nn_utils/dgl_graph_ops.py
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/src/ms_pred/nn_utils/transformer_layer.py
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/src/ms_pred/nn_utils/dgl_modules.py
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/src/ms_pred/nn_utils/__init__.py
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/src/ms_pred/common/parallel_utils.py
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/src/ms_pred/common/splitter.py
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/src/ms_pred/common/chem_utils.py
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/src/ms_pred/common/fingerprint.py
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/src/ms_pred/common/__init__.py
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/src/ms_pred/common/denoising_utils.py
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/src/ms_pred/common/misc_utils.py
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/shim/dgl/function.py
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/shim/dgl/__init__.py
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/shim/dgl/nn/__init__.py
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/shim/dgl/backend/pytorch.py
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/shim/dgl/backend/__init__.py
/kaggle/input/datasets/ahmedberatozer/casmi26-iceberg/shim/torch_scatter/__init__.py